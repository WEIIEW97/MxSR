import torch
import torch.nn as nn

from .ssim import ssim


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
