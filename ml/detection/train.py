from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import OilSpillDataset
from model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"C:\Users\Ninad\Desktop\OceanShield-data\oil_spill_23scenes\oil_spill_23scenes"

IMAGE_SIZE = 256
BATCH_SIZE = 2
NUM_WORKERS = 0

EPOCHS = 15
LEARNING_RATE = 1e-4

MODEL_DIR = Path("ml/detection/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "unet_best.pth"


# ============================================================
# TRAIN / VALIDATION SCENE SPLIT
# ============================================================

ALL_SCENES = [
    "2018_08_21_.tif",
    "2018_09_14_.tif",
    "2018_12_07.tif",
    "2018_12_07_b.tif",
    "2018_12_19.tif",
    "2018_12_19_b.tif",
    "2018_12_31_b.tif",
    "20190816.tif",
    "20190908.tif",
    "20200224.tif",
    "20200307.tif",
    "20200319.tif",
    "20200331.tif",
    "20200822.tif",
]

# Three completely unseen scenes for validation.
VAL_SCENES = [
    "2018_09_14_.tif",
    "2018_12_07_b.tif",
    "20200307.tif",
]

TRAIN_SCENES = [
    scene for scene in ALL_SCENES
    if scene not in VAL_SCENES
]


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (probabilities * targets).sum()

        dice = (
            (2.0 * intersection + self.smooth)
            /
            (
                probabilities.sum()
                + targets.sum()
                + self.smooth
            )
        )

        return 1.0 - dice


# ============================================================
# COMBINED LOSS
# ============================================================

class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return bce_loss + dice_loss


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(logits, targets):

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities > 0.5).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    tp = ((predictions == 1) & (targets == 1)).sum().float()
    tn = ((predictions == 0) & (targets == 0)).sum().float()
    fp = ((predictions == 1) & (targets == 0)).sum().float()
    fn = ((predictions == 0) & (targets == 1)).sum().float()

    epsilon = 1e-7

    iou = (
        tp /
        (tp + fp + fn + epsilon)
    )

    dice = (
        2 * tp /
        (2 * tp + fp + fn + epsilon)
    )

    precision = (
        tp /
        (tp + fp + epsilon)
    )

    recall = (
        tp /
        (tp + fn + epsilon)
    )

    return (
        iou.item(),
        dice.item(),
        precision.item(),
        recall.item(),
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    running_loss = 0.0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0

    batches = 0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        running_loss += loss.item()

        iou, dice, precision, recall = calculate_metrics(
            outputs,
            masks,
        )

        total_iou += iou
        total_dice += dice
        total_precision += precision
        total_recall += recall

        batches += 1

    return (
        running_loss / batches,
        total_iou / batches,
        total_dice / batches,
        total_precision / batches,
        total_recall / batches,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("OceanShield AI - Oil Spill Segmentation Training")
    print("=" * 60)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    print("\nCreating training dataset...")

    train_dataset = OilSpillDataset(
        dataset_root=DATASET_ROOT,
        scene_names=TRAIN_SCENES,
        image_size=IMAGE_SIZE,
        stride=128,
        augment=True,
    )

    print("\nCreating validation dataset...")

    val_dataset = OilSpillDataset(
        dataset_root=DATASET_ROOT,
        scene_names=VAL_SCENES,
        image_size=IMAGE_SIZE,
        stride=128,
        augment=False,
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nDataLoader ready")
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Epochs:             {EPOCHS}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nCreating U-Net...")

    model = UNet(
        in_channels=1,
        out_channels=1,
    ).to(device)

    # --------------------------------------------------------
    # LOSS + OPTIMIZER
    # --------------------------------------------------------

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    best_val_dice = 0.0

    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        (
            val_loss,
            val_iou,
            val_dice,
            val_precision,
            val_recall,
        ) = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f}"
        )

        print(
            f"IoU:        {val_iou:.4f}"
        )

        print(
            f"Dice:       {val_dice:.4f}"
        )

        print(
            f"Precision:  {val_precision:.4f}"
        )

        print(
            f"Recall:     {val_recall:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_dice > best_val_dice:

            best_val_dice = val_dice

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                },
                BEST_MODEL_PATH,
            )

            print(
                f"✓ Best model saved "
                f"(Dice={val_dice:.4f})"
            )

    print("\n" + "=" * 60)
    print("Training complete")
    print("=" * 60)

    print(
        f"Best validation Dice: {best_val_dice:.4f}"
    )

    print(
        f"Model saved to:\n{BEST_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
    