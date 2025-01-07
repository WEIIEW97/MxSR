import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class PoseNet(nn.Module):
    """
    PoseNet architecture for estimating relative pose between two images.
    
    Attributes:
        encoder (nn.Module): Pretrained CNN encoder (e.g., ResNet18).
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        translation (nn.Linear): Fully connected layer for translation.
        rotation (nn.Linear): Fully connected layer for rotation.
    """
    def __init__(self, encoder='resnet18', pretrained=True):
        super(PoseNet, self).__init__()
        
        # Load a pretrained ResNet18 model and remove the fully connected layers
        if encoder == 'resnet18':
            self.encoder = models.resnet18(pretrained=pretrained)
            modules = list(self.encoder.children())[:-2]  # Remove avgpool and fc
            self.encoder = nn.Sequential(*modules)
            feature_dim=1024
        elif encoder == 'mobilenetv3-small':
            self.encoder = models.mobilenet_v3_small(pretrained=pretrained).features
            feature_dim = 576
        elif encoder == 'mobilenetv3-large':
            self.encoder = models.mobilenet_v3_large(pretrained=pretrained).features
            feature_dim = 960
        else:
            raise NotImplementedError(f"{encoder} has not been supported!, please use one of 'resnet18', 'mobilenetv3-small', 'mobilenetv3-small' instead. ")
        # Freeze encoder weights if needed
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        # Define the pose regression head
        # Assuming encoder outputs features of size 512 x H/32 x W/32 for ResNet18
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(feature_dim * 2, 256)  # Concatenate features from two images
        self.fc2 = nn.Linear(256, 128)
        
        # Output layers for translation and rotation
        self.translation = nn.Linear(128, 3)  # tx, ty, tz
        self.rotation = nn.Linear(128, 3)     # rx, ry, rz (Euler angles)
        
        # Initialize weights for regression layers
        self._init_weights()
        
    def _init_weights(self):
        """
        Initializes the weights of the fully connected layers.
        """
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)
        
        nn.init.xavier_normal_(self.translation.weight)
        nn.init.constant_(self.translation.bias, 0)
        
        nn.init.xavier_normal_(self.rotation.weight)
        nn.init.constant_(self.rotation.bias, 0)
    
    def forward(self, img1, img2):
        """
        Forward pass to estimate the relative pose between two images.
        
        Args:
            img1 (torch.Tensor): Batch of first images [B, 3, H, W].
            img2 (torch.Tensor): Batch of second images [B, 3, H, W].
        
        Returns:
            translation (torch.Tensor): Estimated translations [B, 3].
            rotation (torch.Tensor): Estimated rotations [B, 3].
        """
        feat1 = self.encoder(img1)  # [B, C, H', W']
        feat1 = self.avgpool(feat1)  # [B, C, 1, 1]
        feat1 = feat1.view(feat1.size(0), -1)  # [B, C]
        
        feat2 = self.encoder(img2)  # [B, C, H', W']
        feat2 = self.avgpool(feat2)  # [B, C, 1, 1]
        feat2 = feat2.view(feat2.size(0), -1)  # [B, C]
        
        combined = torch.cat((feat1, feat2), dim=1)  # [B, C*2]
        print(combined.size())
        x = F.relu(self.fc1(combined))  # [B, C*2]
        x = F.relu(self.fc2(x))         # [B, C*2]
        
        translation = self.translation(x)  # [B, 3]
        rotation = self.rotation(x)        # [B, 3]
        
        return translation, rotation
    


def euler_angles_to_rotation_matrix(euler_angles):
    """
    Converts Euler angles to rotation matrices.

    Args:
        euler_angles (torch.Tensor): Euler angles [B, 3], where each row is [rx, ry, rz].

    Returns:
        rotation_matrices (torch.Tensor): Rotation matrices [B, 3, 3].
    """
    rx, ry, rz = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]

    cos_rx = torch.cos(rx)
    sin_rx = torch.sin(rx)
    cos_ry = torch.cos(ry)
    sin_ry = torch.sin(ry)
    cos_rz = torch.cos(rz)
    sin_rz = torch.sin(rz)

    # Rotation matrix around x-axis
    Rx = torch.stack([
        torch.ones_like(rx), torch.zeros_like(rx), torch.zeros_like(rx),
        torch.zeros_like(rx), cos_rx, -sin_rx,
        torch.zeros_like(rx), sin_rx, cos_rx
    ], dim=1).view(-1, 3, 3)

    # Rotation matrix around y-axis
    Ry = torch.stack([
        cos_ry, torch.zeros_like(ry), sin_ry,
        torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry),
        -sin_ry, torch.zeros_like(ry), cos_ry
    ], dim=1).view(-1, 3, 3)

    # Rotation matrix around z-axis
    Rz = torch.stack([
        cos_rz, -sin_rz, torch.zeros_like(rz),
        sin_rz, cos_rz, torch.zeros_like(rz),
        torch.zeros_like(rz), torch.zeros_like(rz), torch.ones_like(rz)
    ], dim=1).view(-1, 3, 3)

    # R = Rz * Ry * Rx
    rotation_matrices = torch.bmm(Rz, torch.bmm(Ry, Rx))

    return rotation_matrices