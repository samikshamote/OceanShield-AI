from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset


class OilSpillDataset(Dataset):
    """
    Oil-spill segmentation dataset built directly from Sentinel-1
    image/mask TIFF pairs.

    Expected structure:

    dataset_root/
    └── oil_spill_23scenes/
        ├── train/
        │   ├── images/
        │   └── masks/
        └── test/
            ├── images/
            └── masks/
    """

    def __init__(
        self,
        dataset_root,
        scene_names=None,
        image_size=256,
        stride=128,
        max_patches_per_class=None,
        augment=False,
    ):
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size
        self.stride = stride
        self.augment = augment

        self.images_dir = self.dataset_root / "train" / "images"
        self.masks_dir = self.dataset_root / "train" / "masks"

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found:\n{self.images_dir}"
            )

        if not self.masks_dir.exists():
            raise FileNotFoundError(
                f"Masks directory not found:\n{self.masks_dir}"
            )

        # If scenes aren't specified, use every available training scene.
        if scene_names is None:
            scene_names = sorted(
                p.name
                for p in self.images_dir.glob("*.tif")
            )

        self.scene_names = scene_names

        # Store patch locations as:
        # (image_path, mask_path, x, y)
        self.positive_patches = []
        self.negative_patches = []

        self._build_patch_index()

        # Balance positive and negative patches.
        if max_patches_per_class is not None:
            self.positive_patches = self.positive_patches[
                :max_patches_per_class
            ]
            self.negative_patches = self.negative_patches[
                :max_patches_per_class
            ]

        n = min(
            len(self.positive_patches),
            len(self.negative_patches),
        )

        if n == 0:
            raise RuntimeError(
                "Could not find both positive and negative patches."
            )

        # Equal number of oil and background patches.
        self.samples = (
            self.positive_patches[:n]
            + self.negative_patches[:n]
        )

        # Deterministic ordering.
        rng = np.random.default_rng(42)
        rng.shuffle(self.samples)

        print("\nDataset created")
        print("----------------")
        print(f"Scenes:              {len(self.scene_names)}")
        print(f"Positive candidates: {len(self.positive_patches)}")
        print(f"Negative candidates: {len(self.negative_patches)}")
        print(f"Final samples:       {len(self.samples)}")
        print(f"Patch size:          {self.image_size}x{self.image_size}")
        print(f"Stride:              {self.stride}")

    @staticmethod
    def _positions(length, patch_size, stride):
        """
        Generate patch positions and make sure the final edge
        of the image is also covered.
        """
        if length < patch_size:
            return []

        positions = list(
            range(0, length - patch_size + 1, stride)
        )

        last_position = length - patch_size

        if positions[-1] != last_position:
            positions.append(last_position)

        return positions

    def _build_patch_index(self):
        """
        Scan masks and create a list of oil-containing and
        completely-background patches.
        """

        print("\nBuilding patch index...")

        for scene_number, scene_name in enumerate(
            self.scene_names, start=1
        ):
            image_path = self.images_dir / scene_name
            mask_path = self.masks_dir / scene_name

            if not image_path.exists():
                print(f"WARNING: image missing: {scene_name}")
                continue

            if not mask_path.exists():
                print(f"WARNING: mask missing: {scene_name}")
                continue

            with rasterio.open(mask_path) as mask_src:

                width = mask_src.width
                height = mask_src.height

                x_positions = self._positions(
                    width,
                    self.image_size,
                    self.stride,
                )

                y_positions = self._positions(
                    height,
                    self.image_size,
                    self.stride,
                )

                scene_positive = 0
                scene_negative = 0

                for y in y_positions:
                    for x in x_positions:

                        window = Window(
                            x,
                            y,
                            self.image_size,
                            self.image_size,
                        )

                        mask_patch = mask_src.read(
                            1,
                            window=window,
                        )

                        oil_pixels = np.count_nonzero(
                            mask_patch > 0
                        )

                        sample = (
                            str(image_path),
                            str(mask_path),
                            x,
                            y,
                        )

                        if oil_pixels > 0:
                            self.positive_patches.append(sample)
                            scene_positive += 1

                        else:
                            self.negative_patches.append(sample)
                            scene_negative += 1

            print(
                f"[{scene_number}/{len(self.scene_names)}] "
                f"{scene_name}: "
                f"+{scene_positive} / -{scene_negative}"
            )

    @staticmethod
    def _normalize_sar(image):
        """
        Robust normalization for Sentinel-1 SAR intensity.

        Uses the 1st and 99th percentiles to reduce the influence
        of extreme values.
        """

        image = image.astype(np.float32)

        # Handle invalid values.
        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        low = np.percentile(image, 1)
        high = np.percentile(image, 99)

        if high > low:
            image = np.clip(image, low, high)
            image = (image - low) / (high - low)
        else:
            image = np.zeros_like(image)

        return image.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path, x, y = self.samples[index]

        window = Window(
            x,
            y,
            self.image_size,
            self.image_size,
        )

        # Read Sentinel-1 VV image.
        with rasterio.open(image_path) as src:
            image = src.read(
                1,
                window=window,
            )

        # Read corresponding segmentation mask.
        with rasterio.open(mask_path) as src:
            mask = src.read(
                1,
                window=window,
            )

        if image.shape != (
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                f"Unexpected image shape: {image.shape}"
            )

        if mask.shape != (
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                f"Unexpected mask shape: {mask.shape}"
            )

        # Normalize SAR.
        image = self._normalize_sar(image)

        # Convert mask to binary.
        mask = (mask > 0).astype(np.float32)

        # Convert NumPy -> PyTorch.
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        # Simple spatial augmentation.
        if self.augment:

            if torch.rand(1).item() > 0.5:
                image_tensor = torch.flip(
                    image_tensor,
                    dims=[2],
                )
                mask_tensor = torch.flip(
                    mask_tensor,
                    dims=[2],
                )

            if torch.rand(1).item() > 0.5:
                image_tensor = torch.flip(
                    image_tensor,
                    dims=[1],
                )
                mask_tensor = torch.flip(
                    mask_tensor,
                    dims=[1],
                )

            # Random 90-degree rotation.
            k = torch.randint(
                0,
                4,
                (1,),
            ).item()

            if k > 0:
                image_tensor = torch.rot90(
                    image_tensor,
                    k=k,
                    dims=[1, 2],
                )
                mask_tensor = torch.rot90(
                    mask_tensor,
                    k=k,
                    dims=[1, 2],
                )

        return image_tensor.float(), mask_tensor.float()