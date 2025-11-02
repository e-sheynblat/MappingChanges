import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from fetch_current import fetch_google_tile
from fetch_historical import fetch_gee_tile

class ChangeDetectionDataset(Dataset):
    def __init__(self, locations, transform=None):
        self.locations = locations
        self.transform = transform or A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ], additional_targets={'image0': 'image'}) 

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        lat, lon, label = self.locations[idx]

        # Fetch tiles
        hist = fetch_gee_tile(lat, lon)
        curr = fetch_google_tile(lat, lon)

        # Fallback if tiles are missing
        if hist is None:
            hist = np.zeros((256, 256, 3), dtype=np.uint8)
        if curr is None:
            curr = np.zeros((256, 256, 3), dtype=np.uint8)

        # Ensure HWC and uint8
        def to_hwc_uint8(img):
            if img.ndim == 2:  # grayscale to RGB
                img = np.stack([img]*3, axis=-1)
            elif img.shape[2] > 3:  # take first 3 channels if more
                img = img[:, :, :3]
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)  # handle float [0,1]
            return img

        hist = to_hwc_uint8(hist)
        curr = to_hwc_uint8(curr)

        # Apply augmentations
        augmented = self.transform(image=hist, image0=curr)
        hist_t = augmented['image']
        curr_t = augmented['image0']

        # Pseudo-label: 0=same location, 1=random/distant location
        return hist_t, curr_t, torch.tensor(label, dtype=torch.float)
