from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import label


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


with rasterio.open(PREDICTION_PATH) as src:
    prediction = src.read(1) > 0

with rasterio.open(GROUND_TRUTH_PATH) as src:
    ground_truth = src.read(1) > 0


if prediction.shape != ground_truth.shape:
    raise ValueError(
        f"Shape mismatch: prediction={prediction.shape}, "
        f"ground_truth={ground_truth.shape}"
    )


# Label predicted connected components
labeled, number_of_regions = label(
    prediction,
    structure=np.ones((3, 3))
)


regions = []

for region_id in range(1, number_of_regions + 1):

    region_mask = labeled == region_id

    area = int(region_mask.sum())

    # Ground-truth pixels contained inside this predicted region
    overlap_pixels = int(
        np.logical_and(region_mask, ground_truth).sum()
    )

    overlap_percentage = (
        overlap_pixels / area * 100
        if area > 0
        else 0
    )

    regions.append(
        (
            area,
            overlap_pixels,
            overlap_percentage
        )
    )


# Largest predicted regions first
regions.sort(
    key=lambda x: x[0],
    reverse=True
)


print("=" * 75)
print("OCEANSHIELD AI - PREDICTED REGION OVERLAP ANALYSIS")
print("=" * 75)

print(f"Total predicted regions : {number_of_regions}")
print()

print(
    f"{'Rank':<7}"
    f"{'Area(px)':<15}"
    f"{'GT overlap(px)':<18}"
    f"{'Overlap %':<12}"
)
print("-" * 75)


for rank, (area, overlap, percentage) in enumerate(
    regions[:20],
    start=1
):

    print(
        f"{rank:<7}"
        f"{area:<15,}"
        f"{overlap:<18,}"
        f"{percentage:<12.2f}"
    )


print("\n" + "=" * 75)

# Summary buckets
areas = np.array([r[0] for r in regions])
overlaps = np.array([r[2] for r in regions])

print("\nPREDICTED REGION QUALITY")

for minimum_overlap in [0, 10, 25, 50, 75, 90]:

    count = int(
        np.sum(overlaps >= minimum_overlap)
    )

    print(
        f"Regions with >= {minimum_overlap:>2}% "
        f"ground-truth overlap : {count}"
    )


print("\n" + "=" * 75)

# Largest region
largest = regions[0]

print("\nLARGEST PREDICTED REGION")
print(f"Area              : {largest[0]:,} pixels")
print(f"Ground-truth      : {largest[1]:,} pixels")
print(f"Overlap           : {largest[2]:.2f}%")

print("\n" + "=" * 75)