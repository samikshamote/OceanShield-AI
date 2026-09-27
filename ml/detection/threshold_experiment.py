from pathlib import Path
import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    Path(r"C:\Users\Ninad\Desktop\OceanShield-data")
    / "oil_spill_23scenes"
    / "oil_spill_23scenes"
)

GROUND_TRUTH_PATH = (
    DATASET_ROOT
    / "test"
    / "masks"
    / "2018_09_26.tif"
)

PROBABILITY_PATH = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "prediction_probability.npy"
)


THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def calculate_metrics(prediction, ground_truth):
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = np.logical_and(prediction, ground_truth).sum()
    tn = np.logical_and(~prediction, ~ground_truth).sum()
    fp = np.logical_and(prediction, ~ground_truth).sum()
    fn = np.logical_and(~prediction, ground_truth).sum()

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    dice = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 0
    )
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    return iou, dice, precision, recall


print("=" * 70)
print("OCEANSHIELD AI - THRESHOLD EXPERIMENT")
print("=" * 70)

print(f"Probability map : {PROBABILITY_PATH}")
print(f"Ground truth    : {GROUND_TRUTH_PATH}")

probability_map = np.load(PROBABILITY_PATH)

with rasterio.open(GROUND_TRUTH_PATH) as src:
    ground_truth = src.read(1)

ground_truth = (ground_truth > 0).astype(np.uint8)

if probability_map.shape != ground_truth.shape:
    raise ValueError(
        f"Shape mismatch: probability={probability_map.shape}, "
        f"ground_truth={ground_truth.shape}"
    )

print(f"\nProbability shape : {probability_map.shape}")
print(f"Ground truth shape : {ground_truth.shape}")

print("\n" + "-" * 70)
print(
    f"{'Threshold':<12}"
    f"{'IoU':<12}"
    f"{'Dice':<12}"
    f"{'Precision':<14}"
    f"{'Recall':<12}"
)
print("-" * 70)

results = []

for threshold in THRESHOLDS:

    prediction = probability_map >= threshold

    iou, dice, precision, recall = calculate_metrics(
        prediction,
        ground_truth
    )

    results.append(
        (threshold, iou, dice, precision, recall)
    )

    print(
        f"{threshold:<12.2f}"
        f"{iou * 100:<12.2f}"
        f"{dice * 100:<12.2f}"
        f"{precision * 100:<14.2f}"
        f"{recall * 100:<12.2f}"
    )

print("-" * 70)

best_iou = max(results, key=lambda x: x[1])
best_dice = max(results, key=lambda x: x[2])

print("\nBEST BY IoU:")
print(
    f"Threshold {best_iou[0]:.2f} -> "
    f"IoU {best_iou[1] * 100:.2f}%"
)

print("\nBEST BY DICE/F1:")
print(
    f"Threshold {best_dice[0]:.2f} -> "
    f"Dice/F1 {best_dice[2] * 100:.2f}%"
)

print("=" * 70)