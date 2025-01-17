import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssim import ssim

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        channel_attention = self.fc2(self.relu(self.fc1(avg_pool)))
        return x * self.sigmoid(channel_attention)
    

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention_map = self.conv1(x)
        attention_map = self.sigmoid(attention_map)
        return x * attention_map


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        # Encoder
        self.encoder_dual = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder_mono = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Fusion Module
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.spatial_attention = SpatialAttention(128)
        self.channel_attention = ChannelAttention(128)
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )

        self.encoder_dual.to("cuda")
        self.encoder_mono.to("cuda")
        self.fusion.to("cuda")
        self.decoder.to("cuda")
        self.spatial_attention.to("cuda")
        self.channel_attention.to("cuda")

    def forward(self, disparity_dual, disparity_mono):
        # Extract features
        feat_dual = self.encoder_dual(disparity_dual)
        feat_mono = self.encoder_mono(disparity_mono)
        # Fusion
        feat_fused = torch.cat([feat_dual, feat_mono], dim=1)
        feat_fused = self.fusion(feat_fused)
        feat_fused = self.spatial_attention(feat_fused)
        feat_fused = self.channel_attention(feat_fused)
        # Decode
        output = self.decoder(feat_fused)
        return output


class AttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def compute_ssim_loss(img1, img2):
    return ssim(img1, img2)


def compute_smoothness_loss(d_refined, I_L):
    grad_x = torch.abs(d_refined[:, :, :, :-1] - d_refined[:, :, :, 1:])
    grad_y = torch.abs(d_refined[:, :, :-1, :] - d_refined[:, :, 1:, :])

    img_grad_x = torch.mean(
        torch.abs(I_L[:, :, :, :-1] - I_L[:, :, :, 1:]), 1, keepdim=True
    )
    img_grad_y = torch.mean(
        torch.abs(I_L[:, :, :-1, :] - I_L[:, :, 1:, :]), 1, keepdim=True
    )

    grad_x_weighted = grad_x * torch.exp(-img_grad_x)
    grad_y_weighted = grad_y * torch.exp(-img_grad_y)
    return torch.mean(grad_x_weighted) + torch.mean(grad_y_weighted)


def edge_loss(output, target):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to('cuda')
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to('cuda')
    edge_output = torch.sqrt(F.conv2d(output, sobel_x)**2 + F.conv2d(output, sobel_y)**2)
    edge_target = torch.sqrt(F.conv2d(target, sobel_x)**2 + F.conv2d(target, sobel_y)**2)
    return F.l1_loss(edge_output, edge_target)
