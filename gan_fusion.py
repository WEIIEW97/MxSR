import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------
# 1) Generator (G)
# --------------------------------------------------
class DepthFusionGenerator(nn.Module):
    """Simple UNet-like generator for depth fusion."""
    def __init__(self, in_channels=5, out_channels=1):
        super(DepthFusionGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)

# --------------------------------------------------
# 2) Discriminator (D)
# --------------------------------------------------
class PatchDiscriminator(nn.Module):
    """
    A small patch-based discriminator: tries to classify
    local patches as real/fake.
    """
    def __init__(self, in_channels=1):
        super(PatchDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 1)  # output shape: [B, 1, H/4, W/4]
        )

    def forward(self, x):
        return self.main(x)

# --------------------------------------------------
# 3) Adversarial training loop (simplified)
# --------------------------------------------------
def gan_training_step(
    generator, 
    discriminator, 
    batch, 
    g_optimizer, 
    d_optimizer, 
    recon_criterion, 
    adv_criterion, 
    lambda_recon=10.0,
    device="cuda"
):
    """
    One training step for both generator and discriminator.
    """
    generator.train()
    discriminator.train()
    
    # Unpack batch
    D_stereo = batch["stereo"].to(device)
    D_mono   = batch["mono"].to(device)
    I_rgb    = batch["rgb"].to(device)
    # If you have real GT depth, you can use it for reconstruction or 
    # you might just rely on D_stereo or other constraints.
    D_gt = batch["gt_depth"].to(device) if "gt_depth" in batch else D_stereo
    
    # -------------------------
    # Train Generator (G)
    # -------------------------
    g_optimizer.zero_grad()
    
    # 1) Forward pass G
    x_in = torch.cat([D_stereo, D_mono, I_rgb], dim=1)  # [B, 1+1+3=5, H, W]
    D_fused = generator(x_in)                          # [B, 1, H, W]
    
    # 2) Reconstruction loss (e.g. L1 to GT or to stereo)
    recon_loss = recon_criterion(D_fused, D_gt)
    
    # 3) Adversarial loss -> discriminator should classify D_fused as "real"
    pred_fake = discriminator(D_fused)
    # Create target labels of 1 (real) for generator’s output
    valid_labels = torch.ones_like(pred_fake, device=device)
    adv_loss = adv_criterion(pred_fake, valid_labels)
    
    g_loss = lambda_recon * recon_loss + adv_loss
    g_loss.backward()
    g_optimizer.step()
    
    # -------------------------
    # Train Discriminator (D)
    # -------------------------
    d_optimizer.zero_grad()
    
    # a) Real depth pass (using GT or high-quality data)
    pred_real = discriminator(D_gt)
    real_labels = torch.ones_like(pred_real, device=device)
    loss_real = adv_criterion(pred_real, real_labels)
    
    # b) Fake depth pass (from G)
    pred_fake = discriminator(D_fused.detach())  # detach so G isn’t trained here
    fake_labels = torch.zeros_like(pred_fake, device=device)
    loss_fake = adv_criterion(pred_fake, fake_labels)
    
    d_loss = 0.5 * (loss_real + loss_fake)
    d_loss.backward()
    d_optimizer.step()
    
    return g_loss.item(), d_loss.item()

# --------------------------------------------------
# 4) Example Usage
# --------------------------------------------------
if __name__ == "__main__":
    # Toy dataset again
    class RandomDepthDataset(torch.utils.data.Dataset):
        def __init__(self, length=10, H=64, W=64):
            super().__init__()
            self.length = length
            self.H = H
            self.W = W
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            # Return random data
            stereo = torch.rand(1, self.H, self.W)
            mono   = torch.rand(1, self.H, self.W)
            rgb    = torch.rand(3, self.H, self.W)
            gt     = torch.rand(1, self.H, self.W)
            return {
                "stereo": stereo,
                "mono": mono,
                "rgb": rgb,
                "gt_depth": gt
            }
    
    device = "cpu"  # or "cuda"
    dataset = RandomDepthDataset(length=50)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize generator + discriminator
    G = DepthFusionGenerator(in_channels=5, out_channels=1).to(device)
    D = PatchDiscriminator(in_channels=1).to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    # Loss functions
    recon_criterion = nn.L1Loss()   # Reconstruction
    adv_criterion   = nn.BCEWithLogitsLoss()  # Adversarial
    
    # Training loop
    EPOCHS = 5
    for epoch in range(EPOCHS):
        epoch_g_loss, epoch_d_loss = 0.0, 0.0
        for batch_i, batch in enumerate(dataloader):
            g_loss, d_loss = gan_training_step(
                G, D, batch, 
                g_optimizer, d_optimizer, 
                recon_criterion, adv_criterion, 
                lambda_recon=10.0,
                device=device
            )
            epoch_g_loss += g_loss
            epoch_d_loss += d_loss
        
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | G_loss: {avg_g_loss:.4f} | D_loss: {avg_d_loss:.4f}")
    
    # After training, use G(x_in) to get your fused depth.
