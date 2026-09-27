from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROUND_TRUTH = (
    Path(r"C:\Users\Ninad\Desktop\OceanShield-data")
    / "oil_spill_23scenes"
    / "oil_spill_23scenes"
    / "test"
    / "masks"
    / "2018_09_26.tif"
)

PREDICTION = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "spill_mask.tif"
)


# ============================================================
# LOAD MASKS
# ============================================================

print("=" * 60)
print("OCEANSHIELD AI - MODEL EVALUATION")
print("=" * 60)

print("\nLoading ground-truth mask...")

with rasterio.open(GROUND_TRUTH) as src:
    ground_truth = src.read(1)

print("Ground truth loaded.")

print("\nLoading AI prediction mask...")

with rasterio.open(PREDICTION) as src:
    prediction = src.read(1)

print("Prediction loaded.")


# ============================================================
# VALIDATE
# ============================================================

if ground_truth.shape != prediction.shape:
    raise ValueError(
        f"Shape mismatch: "
        f"Ground truth {ground_truth.shape}, "
        f"Prediction {prediction.shape}"
    )

ground_truth = (ground_truth > 0).astype(np.uint8)
prediction = (prediction > 0).astype(np.uint8)


print()
print("Ground truth shape :", ground_truth.shape)
print("Prediction shape   :", prediction.shape)
print("Ground truth values:", np.unique(ground_truth))
print("Prediction values  :", np.unique(prediction))


# ============================================================
# CONFUSION COUNTS
# ============================================================

true_positive = np.logical_and(
    prediction == 1,
    ground_truth == 1
).sum()

true_negative = np.logical_and(
    prediction == 0,
    ground_truth == 0
).sum()

false_positive = np.logical_and(
    prediction == 1,
    ground_truth == 0
).sum()

false_negative = np.logical_and(
    prediction == 0,
    ground_truth == 1
).sum()


# ============================================================
# METRICS
# ============================================================

epsilon = 1e-8

iou = (
    true_positive
    / (
        true_positive
        + false_positive
        + false_negative
        + epsilon
    )
)

dice = (
    2 * true_positive
    / (
        2 * true_positive
        + false_positive
        + false_negative
        + epsilon
    )
)

precision = (
    true_positive
    / (
        true_positive
        + false_positive
        + epsilon
    )
)

recall = (
    true_positive
    / (
        true_positive
        + false_negative
        + epsilon
    )
)

f1 = (
    2 * precision * recall
    / (
        precision
        + recall
        + epsilon
    )
)

accuracy = (
    (true_positive + true_negative)
    / (
        true_positive
        + true_negative
        + false_positive
        + false_negative
        + epsilon
    )
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("CONFUSION COUNTS")
print("=" * 60)

print(f"True Positive  : {true_positive:,}")
print(f"True Negative  : {true_negative:,}")
print(f"False Positive : {false_positive:,}")
print(f"False Negative : {false_negative:,}")


print()
print("=" * 60)
print("SEGMENTATION METRICS")
print("=" * 60)

print(f"IoU       : {iou * 100:.2f}%")
print(f"Dice      : {dice * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print(f"Accuracy  : {accuracy * 100:.2f}%")

print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)