from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import label, find_objects


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


def analyze_regions(mask):
    labeled, count = label(
        mask,
        structure=np.ones((3, 3))
    )

    objects = find_objects(labeled)

    regions = []

    for region_id, slc in enumerate(objects, start=1):

        if slc is None:
            continue

        region = labeled[slc] == region_id

        area = int(region.sum())

        height = slc[0].stop - slc[0].start
        width = slc[1].stop - slc[1].start

        aspect_ratio = (
            max(width, height) / max(min(width, height), 1)
        )

        regions.append(
            {
                "area": area,
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
            }
        )

    return regions


with rasterio.open(PREDICTION_PATH) as src:
    prediction = src.read(1) > 0

with rasterio.open(GROUND_TRUTH_PATH) as src:
    ground_truth = src.read(1) > 0


tp_mask = prediction & ground_truth
fp_mask = prediction & ~ground_truth
fn_mask = ~prediction & ground_truth


tp_regions = analyze_regions(tp_mask)
fp_regions = analyze_regions(fp_mask)
fn_regions = analyze_regions(fn_mask)


def print_summary(name, regions):

    if not regions:
        print(f"\n{name}: No regions")
        return

    areas = np.array([r["area"] for r in regions])
    ratios = np.array([r["aspect_ratio"] for r in regions])

    print(f"\n{name}")
    print("-" * 60)

    print(f"Number of regions     : {len(regions)}")
    print(f"Total pixels          : {areas.sum():,}")
    print(f"Largest region        : {areas.max():,}")
    print(f"Median region         : {np.median(areas):,.0f}")
    print(f"Mean region           : {areas.mean():,.0f}")
    print(f"Median aspect ratio   : {np.median(ratios):.2f}")

    print("\nArea distribution:")
    print(f"  >= 100 px   : {(areas >= 100).sum()}")
    print(f"  >= 500 px   : {(areas >= 500).sum()}")
    print(f"  >= 1,000 px : {(areas >= 1000).sum()}")
    print(f"  >= 5,000 px : {(areas >= 5000).sum()}")
    print(f"  >= 10,000 px: {(areas >= 10000).sum()}")


print("=" * 70)
print("OCEANSHIELD AI - REGION MORPHOLOGY ANALYSIS")
print("=" * 70)

print_summary("TRUE POSITIVE REGIONS", tp_regions)
print_summary("FALSE POSITIVE REGIONS", fp_regions)
print_summary("FALSE NEGATIVE REGIONS", fn_regions)

print("\n" + "=" * 70)