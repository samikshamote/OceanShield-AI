from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import label, find_objects


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASK_PATH = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "spill_mask.tif"
)

PROBABILITY_PATH = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "prediction_probability.npy"
)


with rasterio.open(MASK_PATH) as src:
    prediction = src.read(1) > 0

probability = np.load(PROBABILITY_PATH)

if prediction.shape != probability.shape:
    raise ValueError(
        f"Shape mismatch: mask={prediction.shape}, "
        f"probability={probability.shape}"
    )


labeled, region_count = label(
    prediction,
    structure=np.ones((3, 3))
)

objects = find_objects(labeled)

regions = []

height, width = prediction.shape


for region_id, slc in enumerate(objects, start=1):

    if slc is None:
        continue

    region_mask = labeled[slc] == region_id
    region_probability = probability[slc][region_mask]

    area = int(region_mask.sum())

    mean_confidence = float(region_probability.mean())
    max_confidence = float(region_probability.max())

    high_confidence_pixels = np.sum(
        region_probability >= 0.80
    )

    high_confidence_percentage = (
        high_confidence_pixels / area * 100
    )

    top, bottom = slc[0].start, slc[0].stop
    left, right = slc[1].start, slc[1].stop

    touches_border = (
        top == 0
        or left == 0
        or bottom == height
        or right == width
    )

    regions.append(
        {
            "id": region_id,
            "area": area,
            "mean": mean_confidence,
            "max": max_confidence,
            "high_pct": high_confidence_percentage,
            "border": touches_border,
        }
    )


regions.sort(
    key=lambda r: r["area"],
    reverse=True
)


print("=" * 90)
print("OCEANSHIELD AI - REGION QUALITY ANALYSIS")
print("=" * 90)

print(f"Total regions: {len(regions)}")

print()

print(
    f"{'Rank':<6}"
    f"{'Area':<14}"
    f"{'Mean Conf.':<14}"
    f"{'Max Conf.':<14}"
    f"{'>=0.80 %':<14}"
    f"{'Border':<10}"
)

print("-" * 90)


for rank, region in enumerate(regions[:20], start=1):

    print(
        f"{rank:<6}"
        f"{region['area']:<14,}"
        f"{region['mean']:<14.3f}"
        f"{region['max']:<14.3f}"
        f"{region['high_pct']:<14.2f}"
        f"{str(region['border']):<10}"
    )


print("\n" + "=" * 90)

print("\nREGION STATISTICS")

areas = np.array([r["area"] for r in regions])
means = np.array([r["mean"] for r in regions])
high_pcts = np.array([r["high_pct"] for r in regions])

print(f"Mean region confidence   : {means.mean():.3f}")
print(f"Median region confidence : {np.median(means):.3f}")

print(
    f"Regions with mean >= 0.80 : "
    f"{np.sum(means >= 0.80)}"
)

print(
    f"Regions with mean >= 0.70 : "
    f"{np.sum(means >= 0.70)}"
)

print(
    f"Regions with >=50% pixels "
    f"above 0.80 confidence : "
    f"{np.sum(high_pcts >= 50)}"
)

print(
    f"Regions touching image border : "
    f"{sum(r['border'] for r in regions)}"
)

print("=" * 90)