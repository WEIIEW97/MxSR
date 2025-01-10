from torch.utils.data import Dataset, DataLoader

# from torchvision import transforms
import ast
import re

import numpy as np
import cv2
import os
import torch

from common.utils import read_pfm

from . import transforms


class Middlebury2014(Dataset):
    def __init__(self, data_list_path, img_h, img_w):
        with open(data_list_path, "r") as f:
            self.data_list = [line.rstrip() for line in f.readlines()]
        self.l_name = "im0.png"
        self.r_name = "img1.png"
        self.l_disp_name = "disp0.pfm"
        self.r_disp_name = "disp1.pfm"
        self.calib_name = "calib.txt"

        self.img_h = img_h
        self.img_w = img_w

        self.int_re = re.compile(r"^-?\d+$")
        self.float_re = re.compile(r"^-?\d+\.\d+$")
        train_transform_list = [
            transforms.RandomScale(min_scale=0, max_scale=1.0, crop_width=self.img_w),
            transforms.RandomCrop(img_h, img_w),
            transforms.RandomRotateShiftRight(),
            transforms.RandomColor(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=transforms.IMAGENET_MEAN, std=transforms.IMAGENET_STD
            ),
        ]

        self.transform = transforms.Compose(train_transform_list)

    def parse_cam_matrix(self, matrix_str):
        # Remove enclosing brackets
        matrix_str = matrix_str.strip("[]")
        # Split rows on ';'
        rows = matrix_str.split(";")
        # Split each row into individual numbers and convert to float
        matrix = []
        for row in rows:
            row_vals = [float(x) for x in row.strip().split() if x]
            matrix.append(row_vals)
        return np.array(matrix, dtype=np.float32)

    def parse_calib(self, calib_path):
        with open(calib_path, "r") as f:
            data_str = [line.rstrip() for line in f.readlines()]
        data_list = ast.literal_eval(data_str)
        calib_dict = {}
        for item in data_list:
            key, val = item.split("=", maxsplit=1)
            if key.startswith("cam"):
                calib_dict[key] = self.parse_cam_matrix(val)
            else:
                if self.int_re.match(val):
                    parsed_val = int(val)
                elif self.float_re.match(val):
                    parsed_val = float(val)
                else:
                    parsed_val = val  # Keep as string if it doesn't match int or float

                calib_dict[key] = parsed_val
        return calib_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        scene_dir = self.data_list[index]
        left_img_path = os.path.join(scene_dir, self.l_name)
        right_img_path = os.path.join(scene_dir, self.r_name)
        l_disp_path = os.path.join(scene_dir, self.l_disp_name)
        r_disp_path = os.path.join(scene_dir, self.r_disp_name)
        calib_path = os.path.join(scene_dir, self.calib_name)

        l_img = cv2.imread(left_img_path, cv2.IMREAD_ANYCOLOR)[:, :, ::-1]
        r_img = cv2.imread(right_img_path, cv2.IMREAD_ANYCOLOR)[:, :, ::-1]
        l_disp = read_pfm(l_disp_path)
        r_disp = read_pfm(r_disp_path)
        calib_data = self.parse_calib(calib_path)

        if self.transform:
            # Convert images to PIL format for compatibility with transforms
            l_img_pil = transforms.ToPILImage()(l_img)
            r_img_pil = transforms.ToPILImage()(r_img)

            # Apply the same transformation to both images
            l_img_transformed = self.transform(l_img_pil)
            r_img_transformed = self.transform(r_img_pil)
        else:
            # If no transform is provided, convert to tensors manually
            l_img_transformed = torch.from_numpy(l_img).permute(2, 0, 1).float() / 255.0
            r_img_transformed = torch.from_numpy(r_img).permute(2, 0, 1).float() / 255.0

        l_disp_tensor = (
            torch.from_numpy(l_disp).unsqueeze(0).float()
        )  # Shape: [1, H, W]
        r_disp_tensor = (
            torch.from_numpy(r_disp).unsqueeze(0).float()
        )  # Shape: [1, H, W]

        sample = {
            "left_image": l_img_transformed,  # Tensor shape: [3, H, W]
            "right_image": r_img_transformed,  # Tensor shape: [3, H, W]
            "left_disp": l_disp_tensor,  # Tensor shape: [1, H, W]
            "right_disp": r_disp_tensor,  # Tensor shape: [1, H, W]
            "calib_data": calib_data,  # dict
        }

        return sample
