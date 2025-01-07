import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .ssim import ssim
from .pose_net import PoseNet, euler_angles_to_rotation_matrix


class SimpleRefineNet(nn.Module):
    def __init__(self):
        super(SimpleRefineNet, self).__init__()
        self.shape_conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )
        self.shape_bn1 = nn.BatchNorm2d(32)
        self.shape_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.shape_bn2 = nn.BatchNorm2d(64)

        self.value_conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )
        self.value_bn1 = nn.BatchNorm2d(32)
        self.value_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.value_bn2 = nn.BatchNorm2d(64)

        # Feature Fusion
        # After concatenation, the number of channels is 64 (shape) + 64 (value) = 128
        self.fusion_conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.fusion_bn1 = nn.BatchNorm2d(128)
        self.fusion_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.fusion_bn2 = nn.BatchNorm2d(64)

        self.refine_conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.refine_bn1 = nn.BatchNorm2d(32)
        self.refine_conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.refine_bn2 = nn.BatchNorm2d(16)

        self.output_conv = nn.Conv2d(
            16, 1, 1
        )  # 1x1 convolution for single-channel output

        self.relu = nn.ReLU(inplace=True)

    def forward(self, shape_input, value_input):
        x_shape = self.relu(self.shape_bn1(self.shape_conv1(shape_input)))
        x_shape = self.relu(self.shape_bn2(self.shape_conv2(x_shape)))

        x_value = self.relu(self.value_bn1(self.value_conv1(value_input)))
        x_value = self.relu(self.value_bn2(self.value_conv2(x_value)))

        x = torch.cat((x_shape, x_value), dim=1)  # Concatenate along channel dimension
        x = self.relu(self.fusion_bn1(self.fusion_conv1(x)))
        x = self.relu(self.fusion_bn2(self.fusion_conv2(x)))

        x = self.relu(self.refine_bn1(self.refine_conv1(x)))
        x = self.relu(self.refine_bn2(self.refine_conv2(x)))

        out = self.output_conv(x)

        return out


class AttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    

def warp_image(img, depth, pose, intrinsics):
    # Assuming intrinsics: [B, 3, 3]
    # pose: [B, 4, 4]
    # depth: [B, 1, H, W]
    B, _, H, W = depth.size()
    device = depth.device

    y, x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    y = y.to(device).float()
    x = x.to(device).float()
    ones = torch.ones_like(x)
    pix_coords = torch.stack([x, y, ones], dim=0).view(3, -1)  # [3, H*W]
    pix_coords = pix_coords.unsqueeze(0).repeat(B, 1, 1)  # [B, 3, H*W]

    inv_intrinsics = torch.inverse(intrinsics)
    cam_coords = inv_intrinsics.bmm(pix_coords) * depth.view(B, 1, -1)  # [B, 3, H*W]

    cam_coords = torch.cat([cam_coords, torch.ones((B, 1, H*W), device=device)], dim=1)  # [B, 4, H*W]
    world_coords = pose.bmm(cam_coords)  # [B, 4, H*W]

    proj_coords = intrinsics.bmm(world_coords[:, :3, :])  # [B, 3, H*W]
    proj_coords = proj_coords[:, :2, :] / (proj_coords[:, 2:3, :] + 1e-6)

    proj_x = 2.0 * (proj_coords[:, 0, :] / (W - 1)) - 1.0
    proj_y = 2.0 * (proj_coords[:, 1, :] / (H - 1)) - 1.0
    grid = torch.stack((proj_x, proj_y), dim=2).view(B, H, W, 2)

    warped_img = F.grid_sample(img, grid, padding_mode='zeros', align_corners=True)
    return warped_img


def compute_ssim_loss(img1, img2):
    return ssim(img1, img2)


def compute_smoothness_loss(d_refined, I_L):
    grad_x = torch.abs(d_refined[:, :, :, :-1] - d_refined[:, :, :, 1:])
    grad_y = torch.abs(d_refined[:, :, :-1, :] - d_refined[:, :, 1:, :])

    img_grad_x = torch.mean(torch.abs(I_L[:, :, :, :-1] - I_L[:, :, :, 1:]), 1, keepdim=True)
    img_grad_y = torch.mean(torch.abs(I_L[:, :, :-1, :] - I_L[:, :, 1:, :]), 1, keepdim=True)

    grad_x_weighted = grad_x * torch.exp(-img_grad_x)
    grad_y_weighted = grad_y * torch.exp(-img_grad_y)
    return torch.mean(grad_x_weighted) + torch.mean(grad_y_weighted)



def train_unsupervised_depth_refinement(model, monocular_net, stereo_net, pose_net, dataloader, optimizer, intrinsics, device):
    model.train()
    monocular_net.eval() 
    stereo_net.eval()     
    pose_net.train()

    for batch in dataloader:
        I_L, I_R = batch['left_image'].to(device), batch['right_image'].to(device)

        with torch.no_grad():
            d_mono = monocular_net(I_L)     # Shape: (B, 1, H, W)
            d_stereo = stereo_net(I_L, I_R)  # Shape: (B, 1, H, W)

        d_refined = model(d_mono, d_stereo)  # Shape: (B, 1, H, W)

        translation, rotation = pose_net(I_L, I_R)
        R = euler_angles_to_rotation_matrix(rotation)
        t = translation.unsqueeze(2)

        pose = torch.cat((R, t), dim=2)

        I_R_reprojected = warp_image(I_L, d_refined, pose, intrinsics)

        loss_photo_ssim = compute_ssim_loss(I_R, I_R_reprojected)
        loss_photo_l1 = F.l1_loss(I_R, I_R_reprojected)
        loss_photo = loss_photo_ssim + loss_photo_l1

        loss_smooth = compute_smoothness_loss(d_refined, I_L)

        loss_consistency = F.l1_loss(d_mono, d_stereo)

        lambda_photo = 1.0
        lambda_smooth = 0.1
        lambda_consistency = 0.1
        loss_total = lambda_photo * loss_photo + lambda_smooth * loss_smooth + lambda_consistency * loss_consistency

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        print(f"Photo Loss: {loss_photo.item():.4f}, Smooth Loss: {loss_smooth.item():.4f}, Consistency Loss: {loss_consistency.item():.4f}, Total Loss: {loss_total.item():.4f}")