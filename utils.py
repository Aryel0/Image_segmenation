import torch 
import torchvision
import torch.nn.functional as F
import os
from dataloader import CustomDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves the model state, optimizer state, and other training metadata to a file.
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    """
    Loads model and optimizer states from a saved checkpoint dictionary.
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def get_loaders(train_dir, train_mask_dir, val_dir, val_mask_dir, batch_size, train_transform, val_transform, num_workers, pin_memory, class_names):
    """
    Creates and returns training and validation PyTorch DataLoaders.
    """
    train_dataset = CustomDataset(image_dir=train_dir, mask_dir=train_mask_dir, class_names=class_names, transform=train_transform)
    val_dataset = CustomDataset(image_dir=val_dir, mask_dir=val_mask_dir, class_names=class_names, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader

def multiclass_dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, epsilon: float = 1e-6) -> float:
    """
    Compute mean Dice score for multi-class segmentation.
    Calculates Dice for each lesion class individually and then averages them.

    Args:
        preds (torch.Tensor): Model raw outputs (logits) of shape (N, C, H, W)
        targets (torch.Tensor): Ground truth labels of shape (N, H, W) with class indices
        num_classes (int): Number of classes (including background)
        epsilon (float): Small constant to avoid division by zero

    Returns:
        float: Mean Dice score across all lesion classes (excludes background).
    """
    # Convert logits to predicted class indices
    preds = torch.argmax(preds, dim=1)  # Shape: (N, H, W)

    # One-hot encode predictions and targets for per-channel dice calculation
    preds_one_hot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dice_scores = []

    # Iterate through each lesion class (skipping background class 0)
    for c in range(1, num_classes):
        pred_c = preds_one_hot[:, c]
        target_c = targets_one_hot[:, c]

        intersection = torch.sum(pred_c * target_c)
        union = torch.sum(pred_c) + torch.sum(target_c)

        if union == 0:
            # If the class is not present in ground truth or predictions, skip it
            continue

        dice = (2.0 * intersection) / union
        dice_scores.append(dice)

    if len(dice_scores) == 0:
        return 0.0

    return torch.mean(torch.stack(dice_scores)).item()

def check_accuracy(loader, model, device):
    """
    Evaluates the model on a given dataset loader.
    Calculates pixel-wise accuracy and multi-class Dice score.
    """
    model.eval()
    correct = 0
    total = 0
    dice_score = 0
    num_classes = 6 # Background + 5 Lesions

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device) 
            
            logits = model(x)
            preds = torch.argmax(logits, dim=1) 
            
            # Pixel-wise accuracy
            correct += (preds == y).sum()
            total += torch.numel(preds)
            # Dice score (per batch)
            dice_score += multiclass_dice_score(logits, y, num_classes=num_classes)
    
    accuracy = correct/total * 100
    avg_dice = dice_score/len(loader)
    
    print(f"Got {correct}/{total} right with {accuracy:.2f}% accuracy")
    print(f"Dice score: {avg_dice:.4f}")
    
    model.train()
    return avg_dice

def save_predictions(loader, model, folder="saved_images/", device="cuda"):
    """
    Runs inference on a few validation samples and saves the predicted masks 
    alongside ground truth for visual inspection.
    """
    model.eval()
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    num_classes = 6
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.argmax(model(x), dim=1)
        
        # Scale indices (0-5) to (0.0-1.0) range for grayscale visualization
        # Note: 0 is background (black), 5 is the last class (white)
        preds_vis = preds.float() / (num_classes - 1)
        y_vis = y.float() / (num_classes - 1)
        
        torchvision.utils.save_image(preds_vis.unsqueeze(1), f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y_vis.unsqueeze(1), f"{folder}/y_{idx}.png")
        
        if idx > 10: # Only save a subset to save time/space
            break
            
    model.train()
