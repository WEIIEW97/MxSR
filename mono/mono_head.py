from common import get_torch_device

from .depth_anything.dpt import DPT_DINOv2
from .depth_anything_v2.dpt import DepthAnythingV2
from .depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

import torch
import torch.nn.functional as F
import numpy as np
import cv2

DPTV2_model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


def preprocess(model_path, device):
    model = DPT_DINOv2("vitl", features=256, out_channels=[256, 512, 1024, 1024])
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    model = model.to(device).eval()

    total_params = sum(param.numel() for param in model.parameters())
    print("Total parameters: {:.2f}M".format(total_params / 1e6))

    transform = Compose(
        [
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )
    return model, transform


class InferDAM:
    def __init__(self):
        self.device = get_torch_device()

    def initialize(
        self, model_path, is_scale=False, mode="v2", encoder="vitl", input_size=518
    ):

        self.is_scale = is_scale
        self.mode = mode

        assert mode in (
            "v1",
            "v2",
        ), f"{mode} is not supported! only support v1 and v2 now!"

        if mode == "v2":
            assert (
                encoder is not None and input_size is not None
            ), "you should specify encoder and input_size arguments."

        if mode == "v1":
            self.model, self.transform = preprocess(
                model_path=model_path, device=self.device
            )
        else:
            self.model = DepthAnythingV2(**DPTV2_model_configs[encoder])
            ckpt = torch.load(model_path)
            self.model.load_state_dict(ckpt)
            self.input_size = input_size

    def _infer_v1(self, image):
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            image = self.transform({"image": image})["image"]
            image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            h, w = image.shape[:-2]
        else:
            raise ValueError(f"Unsupported input type!")

        with torch.no_grad():
            depth = self.model(image)

        depth = F.interpolate(
            depth[None], (h, w), mode="bilinear", align_corners=False
        )[0, 0]
        depth_raw = depth.cpu().numpy().astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)
        if self.is_scale:
            return depth
        else:
            return depth_raw

    def _infer_v2(self, image):
        self.model = self.model.to(self.device).eval()
        depth = self.model.infer_image(image, self.input_size)
        depth_raw = depth.astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.astype(np.uint8)
        if self.is_scale:
            return depth
        else:
            return depth_raw

    def get_params_count(self):
        total_params = sum(param.numel() for param in self.model.parameters())
        print("Total parameters: {:.2f}M".format(total_params / 1e6))

    def predict(self, image):
        if self.mode == "v1":
            return self._infer_v1(image)
        else:
            return self._infer_v2(image)
