import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DinoV2FeatureExtractor(nn.Module):
    def __init__(self, dino_model):
        super().__init__()
        self.dino_model = dino_model
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, x):
        f = self.dino_model.get_intermediate_layers(x, n=1)[0]
        b, n, d = f.shape
        p = f[:, 1:, :]
        s = int((n - 1) ** 0.5)
        p = p.reshape(b, s, s, d).permute(0, 3, 1, 2)
        return p

class DisparityRefinementNet(nn.Module):
    def __init__(self, in_channels=2, feat_channels=384, out_channels=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + feat_channels, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, out_channels, 3, 1, 1)
        )
    def forward(self, dm, ds, dino_feat):
        b, _, h, w = dm.shape
        c = torch.cat([dm, ds], dim=1)
        u = F.interpolate(dino_feat, size=(h, w), mode='bilinear', align_corners=False)
        x = torch.cat([c, u], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x