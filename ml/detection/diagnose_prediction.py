from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_PATH = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "spill_mask.tif"
)

GROUND_TRUTH_PATH = (
    Path(r"C:\Users\Ninad\Desktop\OceanShield-data")
    / "oil_spill_23scenes"
    / "oil_spill_23scenes"
    / "test"
    / "masks"
    / "2018_09_26.tif"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "evaluation_diagnostic.png"
)


with rasterio.open(PREDICTION_PATH) as src:
    prediction = src.read(1) > 0

with rasterio.open(GROUND_TRUTH_PATH) as src:
    ground_truth = src.read(1) > 0


if prediction.shape != ground_truth.shape:
    raise ValueError(
        f"Shape mismatch: prediction={prediction.shape}, "
        f"ground_truth={ground_truth.shape}"
    )


tp = prediction & ground_truth
fp = prediction & ~ground_truth
fn = ~prediction & ground_truth


# White background
image = np.ones(
    (*prediction.shape, 3),
    dtype=np.uint8
) * 255


# TP = green
image[tp] = [0, 180, 0]

# FP = red
image[fp] = [220, 0, 0]

# FN = blue
image[fn] = [0, 80, 220]


Image.fromarray(image).save(OUTPUT_PATH)


print("=" * 60)
print("OCEANSHIELD AI - DIAGNOSTIC VISUALIZATION")
print("=" * 60)

print(f"Prediction shape : {prediction.shape}")
print(f"TP pixels        : {tp.sum():,}")
print(f"FP pixels        : {fp.sum():,}")
print(f"FN pixels        : {fn.sum():,}")

print(f"\nDiagnostic map saved to:")
print(OUTPUT_PATH)

print("\nLegend:")
print("GREEN = Correct detection (TP)")
print("RED   = False positive (FP)")
print("BLUE  = Missed spill (FN)")
print("WHITE = Correct background")