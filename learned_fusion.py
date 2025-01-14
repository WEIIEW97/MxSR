# import torch
# import torch.nn as nn
# import torch.optim as optim

# # --------------------------------------------------
# # 1) Define a tiny FusionNetwork
# # --------------------------------------------------
# class FusionNetwork(nn.Module):
#     """
#     A small encoder-decoder that fuses stereo + mono depth (and optionally RGB).
#     This is a simplistic example. Real models may be more complex (UNet, etc.).
#     """
#     def __init__(self, in_channels=3, out_channels=1):
#         super(FusionNetwork, self).__init__()
        
#         # Example: simple convolutional encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
        
#         # Example: simple decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
#         )
    
#     def forward(self, x):
#         # x shape: [B, in_channels, H, W]
#         latent = self.encoder(x)
#         out = self.decoder(latent)
#         return out

# # --------------------------------------------------
# # 2) Training Loop (Simple Supervised Example)
# # --------------------------------------------------
# def train_fusion_network(
#     fusion_net, 
#     dataloader, 
#     optimizer, 
#     criterion,
#     device="cuda"
# ):
#     fusion_net.train()
    
#     total_loss = 0
#     for batch in dataloader:
#         # Suppose each batch is a dict with:
#         # {
#         #   "stereo": [B, 1, H, W],
#         #   "mono":   [B, 1, H, W],
#         #   "rgb":    [B, 3, H, W], (optional)
#         #   "gt_depth": [B, 1, H, W]
#         # }
#         D_stereo = batch["stereo"].to(device)
#         D_mono   = batch["mono"].to(device)
#         I_rgb    = batch["rgb"].to(device)
#         D_gt     = batch["gt_depth"].to(device)
        
#         # Concatenate along the channel dimension: [B, (1+1+3)=5, H, W]
#         # If you don't use RGB, just cat stereo & mono.
#         x_in = torch.cat([D_stereo, D_mono, I_rgb], dim=1)
        
#         optimizer.zero_grad()
#         D_fused = fusion_net(x_in)  # [B, 1, H, W]
        
#         loss = criterion(D_fused, D_gt)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
    
#     return total_loss / len(dataloader)

# # --------------------------------------------------
# # 3) Example usage
# # --------------------------------------------------
# if __name__ == "__main__":
#     # Create a random dataset (toy example)
#     class RandomDepthDataset(torch.utils.data.Dataset):
#         def __init__(self, length=10, H=64, W=64):
#             super().__init__()
#             self.length = length
#             self.H = H
#             self.W = W
        
#         def __len__(self):
#             return self.length
        
#         def __getitem__(self, idx):
#             # Return random data
#             stereo = torch.rand(1, self.H, self.W)
#             mono   = torch.rand(1, self.H, self.W)
#             rgb    = torch.rand(3, self.H, self.W)
#             gt     = torch.rand(1, self.H, self.W)
#             return {
#                 "stereo": stereo,
#                 "mono": mono,
#                 "rgb": rgb,
#                 "gt_depth": gt
#             }
    
#     # Hyperparameters / Setup
#     device = "cpu"  # or "cuda"
#     dataset = RandomDepthDataset(length=50)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
#     model = FusionNetwork(in_channels=5, out_channels=1).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.L1Loss()  # simple L1
    
#     # Training
#     for epoch in range(5):
#         avg_loss = train_fusion_network(model, dataloader, optimizer, criterion, device)
#         print(f"Epoch [{epoch+1}/5], Loss: {avg_loss:.4f}")

#     # After training, model(x_in) will produce a learned fused depth map.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def generate_disparity_grid(disparity):
    """
    Create a flow-field for torch.nn.functional.grid_sample to warp the right image
    horizontally by the predicted disparity.
    
    Args:
        disparity: [B, 1, H, W], values in pixels (horizontal shift).
    Returns:
        flow: [B, H, W, 2], suitable for grid_sample.
    """
    B, _, H, W = disparity.shape

    # Create base grid of normalized coordinates: x in [-1, 1], y in [-1, 1]
    # shape: [1, H, W], then broadcast to [B, H, W]
    base_y, base_x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing="ij"
    )
    base_x = base_x.to(disparity.device)
    base_y = base_y.to(disparity.device)
    
    base_x = base_x.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    base_y = base_y.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    
    # Convert disparity from pixel shift to normalized shift in [-1,1]
    # If W is the width in pixels, a shift of 'disp' means shift in x by disp*(2/W).
    disp_norm = 2.0 * disparity / (W - 1)  # [B, 1, H, W]
    disp_norm = disp_norm.squeeze(1)       # [B, H, W]
    
    # Warped x = base_x + disp_norm in normalized coords
    warped_x = base_x - disp_norm  # subtract means we are warping the right image into left
    warped_y = base_y              # no vertical shift assumed

    # grid_sample expects shape [B, H, W, 2]
    flow = torch.stack([warped_x, warped_y], dim=-1)  # [B, H, W, 2]
    return flow

class SimpleStereoDepthNet(nn.Module):
    """
    A very simple CNN that predicts disparity for the left image.
    In practice, you'd have a bigger UNet or encoder-decoder.
    """
    def __init__(self):
        super(SimpleStereoDepthNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.ReLU()  # disparity >= 0
        )

    def forward(self, x):
        # x: [B, 3, H, W] = left image
        return self.main(x)  # [B, 1, H, W]

def photometric_loss(left_img, warped_right_img, mask=None):
    """
    Basic L1 photometric loss between left image and warped right image.

    Args:
        left_img:        [B, 3, H, W]
        warped_right_img:[B, 3, H, W]
        mask:            optional [B, 1, H, W] to ignore invalid regions
    Returns:
        scalar loss
    """
    diff = (left_img - warped_right_img).abs()
    if mask is not None:
        diff = diff * mask
        return diff.mean()
    else:
        return diff.mean()

def smoothness_loss(disparity, image):
    """
    Edge-aware smoothness loss: encourages smooth disparity,
    but allows edges where the image has edges.
    
    For brevity, we do a simple L1 on disparities' gradients.
    Real methods often scale by exp(-|∂I|).
    """
    grad_disp_x = torch.abs(disparity[:, :, :, :-1] - disparity[:, :, :, 1:])
    grad_disp_y = torch.abs(disparity[:, :, :-1, :] - disparity[:, :, 1:, :, :])

    grad_img_x = torch.mean(
        torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True
    )
    grad_img_y = torch.mean(
        torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :, :]), dim=1, keepdim=True
    )
    
    # weights to reduce smoothness penalty if image gradient is large
    weight_x = torch.exp(-grad_img_x)
    weight_y = torch.exp(-grad_img_y)
    
    smooth_x = grad_disp_x * weight_x[:, :, :, :]
    smooth_y = grad_disp_y * weight_y[:, :, :, :]

    return (smooth_x.mean() + smooth_y.mean())

def train_self_supervised_stereo(
    depth_net, 
    left_img, 
    right_img, 
    optimizer, 
    lambda_smooth=0.1
):
    """
    One training step for a self-supervised stereo approach:
    1) Forward pass: predict disparity
    2) Warp right image to left
    3) Photometric loss + smoothness loss
    4) Backprop
    """
    depth_net.train()
    optimizer.zero_grad()

    # 1) Predict disparity for left image
    disp_left = depth_net(left_img)  # [B, 1, H, W]

    # 2) Warp the right image using the predicted disparity
    flow = generate_disparity_grid(disp_left)  # [B, H, W, 2]
    warped_right = F.grid_sample(right_img, flow, mode='bilinear', padding_mode='border')
    
    # 3) Photometric loss
    photo_loss = photometric_loss(left_img, warped_right)

    # 4) Smoothness loss
    smooth_loss = smoothness_loss(disp_left, left_img)

    # Weighted sum
    loss = photo_loss + lambda_smooth * smooth_loss
    loss.backward()
    optimizer.step()

    return loss.item()

# --------------------------------------------
# Example usage
# --------------------------------------------
if __name__ == "__main__":
    # Suppose we have a mini-batch of left/right images [B,3,H,W], e.g., from a DataLoader
    B, C, H, W = 2, 3, 64, 64
    left_batch = torch.rand(B, C, H, W)
    right_batch = torch.rand(B, C, H, W)

    # Initialize depth net
    depth_net = SimpleStereoDepthNet()
    optimizer = optim.Adam(depth_net.parameters(), lr=1e-3)

    # Run a few training steps
    for step in range(10):
        loss_val = train_self_supervised_stereo(depth_net, left_batch, right_batch, optimizer)
        print(f"Step {step+1}, Loss = {loss_val:.4f}")
    
    # After training, depth_net(left_image) produces a predicted disparity.
    # That disparity can be turned into depth if you know the camera baseline + focal length.
