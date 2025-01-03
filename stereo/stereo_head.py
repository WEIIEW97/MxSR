from common import get_torch_device

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from .gmstereo import transforms
from .gmstereo import UniMatch
from .crestereo import CREStereo
from torchvision.transforms.functional import hflip
from collections import OrderedDict

class InferGMStereo:
    def __init__(self):
        self.device = get_torch_device()
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        self.set_params()

    def initialize(self, model_path, focal=None, baseline=None, input_size=None):
        self.input_size = input_size
        self.focal = focal
        self.baseline = baseline

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
            ]
        )

        self.model = UniMatch(
            num_scales=2,
            feature_channels=128,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=3,
            task="stereo",
        )

        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt["model"])
        self.model = self.model.to(self.device)

    def set_params(
        self,
        attn_type="self_swin2d_cross_swin1d",
        attn_splits_list=[2, 8],
        corr_radius_list=[-1, 4],
        prop_radius_list=[-1, 1],
        padding_factor=16,
        num_reg_refine=1,
        pred_bidir_disp=False,
        pred_right_disp=False,
    ):
        self.attn_type = attn_type
        self.attn_splits_list = attn_splits_list
        self.corr_radius_list = corr_radius_list
        self.prop_radius_list = prop_radius_list
        self.padding_factor = padding_factor
        self.num_reg_refine = num_reg_refine
        self.pred_bidir_disp = pred_bidir_disp
        self.pred_right_disp = pred_right_disp

    def predict(self, left, right):
        pair = {"left": left, "right": right}
        pair = self.transform(pair)

        left = pair["left"].to(self.device).unsqueeze(0)  # [1, 3, H, W]
        right = pair["right"].to(self.device).unsqueeze(0)  # [1, 3, H, W]

        if self.input_size is None:
            nearest_size = [
                int(np.ceil(left.size(-2) / self.padding_factor)) * self.padding_factor,
                int(np.ceil(left.size(-1) / self.padding_factor)) * self.padding_factor,
            ]
            self.input_size = nearest_size

        ori_size = left.shape[-2:]
        self.resize = (
            self.input_size[0] != ori_size[0] or self.input_size[1] != ori_size[1]
        )
        if self.resize:
            left = F.interpolate(
                left, size=self.input_size, mode="bilinear", align_corners=True
            )
            right = F.interpolate(
                right, size=self.input_size, mode="bilinear", align_corners=True
            )

        with torch.no_grad():
            if self.pred_bidir_disp:
                new_left, new_right = hflip(right), hflip(left)
                left = torch.cat((left, new_left), dim=0)
                right = torch.cat((right, new_right), dim=0)

            if self.pred_right_disp:
                left, right = hflip(right), hflip(left)

            disp = self.model(
                left,
                right,
                self.attn_type,
                self.attn_splits_list,
                self.corr_radius_list,
                self.prop_radius_list,
                self.num_reg_refine,
                task="stereo",
            )["flow_preds"][
                -1
            ]  # [1, H, W]

        if self.resize:
            disp = F.interpolate(
                disp.unsqueeze(1), size=ori_size, mode="bilinear", align_corners=True
            ).squeeze(1)
            disp = disp * ori_size[-1] / float(self.input_size[-1])

        if self.pred_right_disp:
            disp = hflip(disp)

        if self.pred_bidir_disp:
            assert disp.size(0) == 2  # [2, H, W]
            disp = hflip(disp[1])

        return disp[0, ...].cpu().numpy()


class InferCREStereo:
    def __init__(self):
        self.device = get_torch_device()

    def initialize(self, model_path, max_disp=256, mixed_precision=False):
        self.model = CREStereo(
            max_disp=max_disp, mixed_precision=mixed_precision, test_mode=True
        )
        ckpt = torch.load(model_path)

        if "optim_state_dict" in ckpt.keys():
            model_state_dict = OrderedDict()
            for k, v in ckpt["state_dict"].items():
                name = k[7:]  # remove `module.`
                model_state_dict[name] = v
        else:
            model_state_dict = ckpt

        self.model.load_state_dict(model_state_dict, strict=True)
        self.model.to(self.device).eval()

    def make_divisible(self, m, divisible=8):
        h, w = m.shape[:2]
        pad_h = (divisible - h % divisible) % divisible
        pad_w = (divisible - w % divisible) % divisible
        if pad_h > 0:
            m = np.pad(
                m, ((pad_h // 2, pad_h - pad_h // 2), (0, 0), (0, 0)), mode="reflect"
            )
        if pad_w > 0:
            m = np.pad(
                m, ((0, 0), (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode="reflect"
            )

        return m

    def predict(self, left, right, n_iters=20, flow_init=True):
        ori_h, ori_w = left.shape[:2]
        left = self.make_divisible(left)
        right = self.make_divisible(right)

        in_h, in_w = left.shape[:2]
        assert in_h % 8 == 0, "input height should be divisible by 8"
        assert in_w % 8 == 0, "input width should be divisible by 8"

        if in_h != ori_h or in_w != ori_w:
            left = cv2.resize(left, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
            right = cv2.resize(right, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

        left = np.transpose(left, (2, 0, 1)).astype(np.float32)
        right = np.transpose(right, (2, 0, 1)).astype(np.float32)

        left = torch.from_numpy(left).unsqueeze(0).to(self.device)
        right = torch.from_numpy(right).unsqueeze(0).to(self.device)

        if flow_init:
            l_dw2 = F.interpolate(
                left,
                size=(left.shape[2] // 2, left.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            r_dw2 = F.interpolate(
                right,
                size=(left.shape[2] // 2, left.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )

        with torch.no_grad():
            if flow_init:
                pred_flow_dw2 = self.model(l_dw2, r_dw2, iters=n_iters)
                pred_flow = self.model(
                    left, right, iters=n_iters // 2, flow_init=pred_flow_dw2
                )
            else:
                pred_flow = self.model(left, right, iters=n_iters, flow_init=None)

        return torch.squeeze(pred_flow[:, 0, ...]).cpu().detach().numpy()
