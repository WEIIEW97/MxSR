import torch
import torch.nn.functional as F


def warp_right_to_left(right_img, disp):
    b, c, h, w = right_img.shape
    y = (
        torch.linspace(0, h - 1, h, device=right_img.device)
        .view(1, h, 1)
        .expand(b, h, w)
    )
    x = (
        torch.linspace(0, w - 1, w, device=right_img.device)
        .view(1, 1, w)
        .expand(b, h, w)
    )
    x_warp = x - disp[:, 0, :, :]
    grid_x = 2.0 * x_warp / (w - 1) - 1.0
    grid_y = 2.0 * y / (h - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=3)
    return F.grid_sample(
        right_img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def warp_image_by_pose(img, depth, pose, intrinsics):
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

    cam_coords = torch.cat(
        [cam_coords, torch.ones((B, 1, H * W), device=device)], dim=1
    )  # [B, 4, H*W]
    world_coords = pose.bmm(cam_coords)  # [B, 4, H*W]

    proj_coords = intrinsics.bmm(world_coords[:, :3, :])  # [B, 3, H*W]
    proj_coords = proj_coords[:, :2, :] / (proj_coords[:, 2:3, :] + 1e-6)

    proj_x = 2.0 * (proj_coords[:, 0, :] / (W - 1)) - 1.0
    proj_y = 2.0 * (proj_coords[:, 1, :] / (H - 1)) - 1.0
    grid = torch.stack((proj_x, proj_y), dim=2).view(B, H, W, 2)

    warped_img = F.grid_sample(img, grid, padding_mode="zeros", align_corners=True)
    return warped_img


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