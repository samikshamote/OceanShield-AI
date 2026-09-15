from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class OilSpillDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        self.images = sorted(
            [
                p
                for p in self.image_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            ]
        )

        if not self.images:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        mask_path = self.mask_dir / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found for {image_path.name}: {mask_path}"
            )

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        image = image.astype(np.float32) / 255.0

        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask