import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import argparse
import os
import torch.nn.functional as F
from model import UNet
from utils import (
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions,
)

class DiceLoss(nn.Module):
    """
    Computes the Dice Loss for multi-class segmentation.
    Dice Loss measures the overlap between predicted and target masks.
    """
    def __init__(self, num_classes, ignore_background=True):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.softmax(inputs, dim=1)
        # Convert targets to one-hot for per-channel comparison
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        start_class = 1 if self.ignore_background else 0
        
        intersection = (inputs[:, start_class:] * targets_one_hot[:, start_class:]).sum(dim=(2,3))
        union = inputs[:, start_class:].sum(dim=(2,3)) + targets_one_hot[:, start_class:].sum(dim=(2,3))
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    """
    Focal Loss helps address heavy class imbalance by down-weighting well-classified 
    examples and focusing on hard ones (like small lesions).
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weights for different classes
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # Probability of the correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_fn(loader, model, optimizer, focal_loss_fn, dice_loss_fn, scaler, device):
    """
    Performs one epoch of training.
    Includes mixed precision handling and gradient clipping.
    """
    model.train()
    loop = tqdm(loader)
    total_focal_loss = 0
    total_dice_loss = 0
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward pass with mixed precision (fp16) to speed up training and save memory
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            predictions = model(data)
            loss_focal = focal_loss_fn(predictions, targets)
            loss_dice = dice_loss_fn(predictions, targets)
            loss = 0.5 * loss_focal + 0.5 * loss_dice

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Unscale gradients before clipping to ensure they are in the correct range
        scaler.unscale_(optimizer)
        # Gradient clipping prevents the 'exploding gradients' problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        # Monitor gradient norm for debugging stability
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item()
        
        scaler.step(optimizer)
        scaler.update()

        # Track losses
        total_focal_loss += loss_focal.item()
        total_dice_loss += loss_dice.item()
        total_loss += loss.item()

        # Update tqdm progress bar
        loop.set_postfix(
            loss=loss.item(),
            focal=loss_focal.item(),
            dice=loss_dice.item(),
            grad=f"{grad_norm:.2f}"
        )
    
    return total_loss / len(loader), total_focal_loss / len(loader), total_dice_loss / len(loader)

def main(args):
    """
    The main control flow for initializing training, data, and the model.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data Augmentations: crucial for medical imaging with small datasets
    train_transform = A.Compose(
        [
            A.Resize(height=args.image_height, width=args.image_width),
            A.Rotate(limit=25, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=args.image_height, width=args.image_width),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )

    CLASS_NAMES = ['Haemorrhages', 'Hard Exudates', 'Microaneurysms', 'Optic Disc', 'Soft Exudates']

    train_loader, val_loader = get_loaders(
        args.train_img_dir, args.train_mask_dir, args.val_img_dir, args.val_mask_dir, 
        args.batch_size, train_transform, val_transform, args.num_workers, args.pin_memory,
        CLASS_NAMES
    )
    
    # 6 Classes: Background (0) + 5 Lesions (1-5)
    # Background weight is low (0.1) while lesions are high (2.0) to force model to learn rare features
    class_weights = torch.tensor([0.1, 2.0, 2.0, 2.0, 2.0, 2.0]).to(DEVICE)
    focal_loss_fn = FocalLoss(alpha=class_weights, gamma=2)
    dice_loss_fn = DiceLoss(num_classes=6, ignore_background=True)
    
    model = UNet(in_channels=3, out_channels=6).to(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # LR Scheduler: reduces learning rate when validation performance plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    if args.load_model and os.path.exists(args.checkpoint):
        load_checkpoint(torch.load(args.checkpoint), model, optimizer)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_dice = 0.0
    
    # Main Epoch Loop
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        avg_loss, avg_focal, avg_dice = train_fn(train_loader, model, optimizer, focal_loss_fn, dice_loss_fn, scaler, DEVICE)
        print(f"Average - Total Loss: {avg_loss:.4f}, Focal Loss: {avg_focal:.4f}, Dice Loss: {avg_dice:.4f}")

        # Validate performance
        dice_score = check_accuracy(val_loader, model, device=DEVICE)
        
        # Step the scheduler based on the Dice score
        scheduler.step(dice_score)
        
        # Save 'best' model if improvement is noted
        if dice_score > best_dice:
            best_dice = dice_score
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
            }
            save_checkpoint(checkpoint, filename="best_checkpoint.pth")
            print(f"New best model saved with Dice: {best_dice:.4f}")
        
        # Periodically save regular checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        save_checkpoint(checkpoint)
        
        # Save visualization of predictions
        if (epoch + 1) % 5 == 0:
            save_predictions(val_loader, model, folder="saved_images/", device=DEVICE)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Attention UNet for Retinal Lesion Segmentation")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (small due to high resolution)")
    parser.add_argument("--image_height", type=int, default=768, help="Resize height")
    parser.add_argument("--image_width", type=int, default=768, help="Resize width")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--pin_memory", type=bool, default=True, help="Fast CPU to GPU transfer")
    parser.add_argument("--load_model", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Checkpoint path")
    parser.add_argument("--num_epochs", type=int, default=7, help="Number of training epochs")
    parser.add_argument("--train_img_dir", type=str, default="Segmentation/Original_Images/Training Set")
    parser.add_argument("--train_mask_dir", type=str, default="Segmentation/Segmentation_Groundtruths/Training Set")
    parser.add_argument("--val_img_dir", type=str, default="Segmentation/Original_Images/Testing Set")
    parser.add_argument("--val_mask_dir", type=str, default="Segmentation/Segmentation_Groundtruths/Testing Set")
    
    args = parser.parse_args()
    main(args)

