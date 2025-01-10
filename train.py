import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dataloader.middlebury import Middlebury2014

from tqdm import tqdm  # For progress bars

from mono.mono_head import InferDAM
from stereo.stereo_head import InferCREStereo, InferGMStereo
from refine.pose_net import PoseNet
from refine.simple_refine_net import SimpleRefineNet


def calculate_epe(pred, target):
    return torch.mean(torch.abs(pred - target))


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        left_images = batch["left_image"].to(device)  # [B, 3, H, W]
        right_images = batch["right_image"].to(device)  # [B, 3, H, W]
        left_disps = batch["left_disp"].to(device)  # [B, 1, H, W]

        # Concatenate left and right images along the channel dimension
        inputs = torch.cat([left_images, right_images], dim=1)  # [B, 6, H, W]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)  # [B, 1, H, W]

        # Compute loss
        loss = criterion(outputs, left_disps)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_epe = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            left_images = batch['left_image'].to(device)
            right_images = batch['right_image'].to(device)
            left_disps = batch['left_disp'].to(device)
            
            # Concatenate left and right images
            inputs = torch.cat([left_images, right_images], dim=1)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, left_disps)
            
            # Compute EPE
            epe = calculate_epe(outputs, left_disps)
            
            running_loss += loss.item() * inputs.size(0)
            running_epe += epe.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_epe = running_epe / len(dataloader.dataset)
    return epoch_loss, epoch_epe


def save_checkpoint(state, checkpoint_dir, filename="model_checkpoint.pth.tar"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)


def get_dataloaders(
    train_list_path,
    val_list_path,
    batch_size,
    num_workers,
    transform=None,
    augment=False,
    use_calib=False,
):
    train_dataset = Middlebury2014(
        data_list_path=train_list_path,
        transform=transform,
        use_calib=use_calib,
        augment=augment,
    )

    val_dataset = Middlebury2014(
        data_list_path=val_list_path,
        transform=transform,
        use_calib=use_calib,
        augment=False,  # Typically, no augmentation during validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# 4. Define Training Parameters
def get_training_parameters():
    # Hyperparameters
    num_epochs = 50
    batch_size = 4
    learning_rate = 1e-4
    num_workers = 4  # Adjust based on your CPU cores
    checkpoint_dir = "checkpoints"
    log_dir = "runs/stereo_matching_experiment_1"

    # Data transformations (if any additional)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # IMAGENET_MEAN
            ),  # IMAGENET_STD
        ]
    )

    return (
        num_epochs,
        batch_size,
        learning_rate,
        num_workers,
        checkpoint_dir,
        log_dir,
        transform,
    )


# 5. Main Training Function
def main():
    # Get training parameters
    (
        num_epochs,
        batch_size,
        learning_rate,
        num_workers,
        checkpoint_dir,
        log_dir,
        transform,
    ) = get_training_parameters()

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, loss function, optimizer
    model = StereoMatchingCNN().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare DataLoaders
    train_list_path = (
        "/path/to/train_list.txt"  # Replace with your training list file path
    )
    val_list_path = (
        "/path/to/val_list.txt"  # Replace with your validation list file path
    )

    train_loader, val_loader = get_dataloaders(
        train_list_path=train_list_path,
        val_list_path=val_list_path,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform,
        augment=True,  # Apply augmentations to training data
        use_calib=False,  # Set to True if you want to use calibration data
    )

    # Initialize variables for tracking best performance
    best_val_loss = float("inf")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch [{epoch}/{num_epochs}]")

        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss:.4f}")
        writer.add_scalar("Loss/Train", train_loss, epoch)

        # Validate
        val_loss, val_epe = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation EPE: {val_epe:.4f}")
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('EPE/Validation', val_epe, epoch)

        scheduler.step()

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                checkpoint_dir, f"best_model_epoch_{epoch}.pth.tar"
            )
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                checkpoint_dir,
                filename=f"best_model_epoch_{epoch}.pth.tar",
            )
            print(
                f"Saved Best Model at Epoch {epoch} with Validation Loss {val_loss:.4f}"
            )

        # Optionally, save every few epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"model_epoch_{epoch}.pth.tar"
            )
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                checkpoint_dir,
                filename=f"model_epoch_{epoch}.pth.tar",
            )
            print(f"Saved Checkpoint at Epoch {epoch}")

    print("Training Completed.")
    writer.close()


if __name__ == "__main__":
    main()
