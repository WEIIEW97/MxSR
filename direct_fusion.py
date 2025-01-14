import torch

def median_scale_mono_to_stereo(D_stereo, D_mono, confidence=None):
    """
    Scale the monocular depth map to match the median of the stereo depth
    in high-confidence regions.

    Args:
        D_stereo:   [B, 1, H, W] stereo depth (metric scale).
        D_mono:     [B, 1, H, W] monocular depth (relative scale).
        confidence: [B, 1, H, W] stereo confidence in [0, 1], optional.
                    If None, we assume we trust all valid pixels equally.

    Returns:
        D_mono_scaled: [B, 1, H, W] scaled monocular depth.
    """
    # Flatten to shape [B, H*W] for easier median calculation
    D_stereo_flat = D_stereo.view(D_stereo.size(0), -1)
    D_mono_flat   = D_mono.view(D_mono.size(0), -1)

    if confidence is not None:
        conf_flat = confidence.view(confidence.size(0), -1)
    else:
        # If no confidence is provided, treat all as "conf=1"
        conf_flat = torch.ones_like(D_stereo_flat)

    D_mono_scaled = D_mono.clone()

    for b in range(D_stereo.size(0)):
        # Mask out low-confidence or invalid regions if needed
        valid_mask = (conf_flat[b] > 0.5)  # e.g., threshold 0.5
        if valid_mask.sum() < 10:
            # If almost no valid pixels, skip or do nothing
            continue

        stereo_vals = D_stereo_flat[b, valid_mask]
        mono_vals   = D_mono_flat[b, valid_mask]

        # Avoid zeros or negative (just in case)
        stereo_vals = stereo_vals[stereo_vals > 0]
        mono_vals   = mono_vals[mono_vals > 0]

        if len(stereo_vals) == 0 or len(mono_vals) == 0:
            continue

        # Compute median
        median_stereo = stereo_vals.median()
        median_mono   = mono_vals.median()

        if median_mono > 1e-6:
            scale_factor = median_stereo / median_mono
            # Scale the entire monocular map for this batch item
            D_mono_scaled[b] *= scale_factor

    return D_mono_scaled

def confidence_fusion(D_stereo, D_mono_scaled, confidence, alpha_threshold=0.5):
    """
    Simple confidence-based fusion:
    D_fused = alpha * D_stereo + (1-alpha) * D_mono_scaled

    Args:
        D_stereo:      [B, 1, H, W]
        D_mono_scaled: [B, 1, H, W]
        confidence:    [B, 1, H, W] in [0,1]
        alpha_threshold: optional threshold to convert confidence to binary,
                         or do a soft blend.

    Returns:
        D_fused: [B, 1, H, W]
    """
    # Option A: Soft blend
    # D_fused = confidence*D_stereo + (1 - confidence)*D_mono_scaled
    
    # Option B: Hard threshold:
    alpha_mask = (confidence > alpha_threshold).float()
    D_fused = alpha_mask * D_stereo + (1 - alpha_mask) * D_mono_scaled
    
    return D_fused

if __name__ == "__main__":
    # Toy example
    B, C, H, W = 2, 1, 64, 64
    D_stereo = torch.rand(B, C, H, W) * 10.0  # random values up to 10 meters
    D_mono   = torch.rand(B, C, H, W) * 5.0   # random relative scale
    conf     = torch.rand(B, C, H, W)

    # 1) Scale monocular to stereo median
    D_mono_scaled = median_scale_mono_to_stereo(D_stereo, D_mono, conf)

    # 2) Fuse
    D_fused = confidence_fusion(D_stereo, D_mono_scaled, conf)

    print("D_fused shape:", D_fused.shape)
