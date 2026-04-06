import os 
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    """
    Custom Dataset for Retinal Lesion Segmentation.
    Loads original images and merges multiple binary masks (one per lesion type) 
    into a single multi-class ground truth label map.
    """
    def __init__(self, image_dir, mask_dir, class_names, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing original retinal images (.jpg).
            mask_dir (str): Path to the directory containing lesion-specific subfolders.
            class_names (list): List of class folder names (e.g., ['Haemorrhages', ...]).
            transform (callable, optional): Albumentations transform pipeline to be applied.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.class_names = class_names  # List of folder names e.g., ['Haemorrhages', 'Hard Exudates', ...]
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Retrieves an image and its corresponding multi-class mask.
        The mask is constructed by iterating through each lesion class folder and 
        placing the binary mask values into a single integer-encoded label map.
        """
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load Image
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Initialize mask with background (class 0)
        H, W = image.shape[:2]
        final_mask = np.zeros((H, W), dtype=np.int64)

        base_name = os.path.splitext(img_name)[0]  # e.g., 'IDRiD_01'

        # Mapping between folder names and the suffix used in their ground truth filenames
        folder_to_suffix = {
            'Haemorrhages': 'HE',
            'Hard Exudates': 'EX',
            'Microaneurysms': 'MA',
            'Optic Disc': 'OD',
            'Soft Exudates': 'SE'
        }

        # Merge masks from different folders into one multi-class mask
        for class_idx, class_name in enumerate(self.class_names, start=1):
            suffix = folder_to_suffix.get(class_name)
            if not suffix:
                raise ValueError(f"Unknown class name: {class_name}")

            # Construct path to the specific lesion mask (usually .tif)
            mask_filename = f"{base_name}_{suffix}.tif"
            mask_path = os.path.join(self.mask_dir, class_name, mask_filename)

            if os.path.exists(mask_path):
                # Load mask, convert to grayscale
                mask_np = np.array(Image.open(mask_path).convert("L"))
                # Pixels where the lesion is present are assigned the class index
                # Note: If lesions overlap, the last one processed takes priority
                final_mask[mask_np > 0] = class_idx

        # Apply augmentations (if any)
        if self.transform:
            augmentations = self.transform(image=image, mask=final_mask)
            image = augmentations["image"]
            final_mask = augmentations["mask"]
            
        return image, final_mask.long()
