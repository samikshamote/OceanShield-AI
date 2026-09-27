from pathlib import Path
import json

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

JSON_PATH = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "detection_result.json"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "candidate_assessment.json"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

with rasterio.open(MASK_PATH) as src:
    prediction = src.read(1) > 0

probability = np.load(PROBABILITY_PATH)

if prediction.shape != probability.shape:
    raise ValueError(
        f"Shape mismatch: mask={prediction.shape}, "
        f"probability={probability.shape}"
    )


height, width = prediction.shape


# ---------------------------------------------------------
# Connected components
# ---------------------------------------------------------

labeled, region_count = label(
    prediction,
    structure=np.ones((3, 3))
)

objects = find_objects(labeled)

regions = []


# ---------------------------------------------------------
# Analyze each candidate
# ---------------------------------------------------------

for region_id, slc in enumerate(objects, start=1):

    if slc is None:
        continue

    region_mask = labeled[slc] == region_id

    area = int(region_mask.sum())

    region_probability = probability[slc][region_mask]

    mean_confidence = float(
        region_probability.mean()
    )

    max_confidence = float(
        region_probability.max()
    )

    high_confidence_percentage = float(
        np.mean(region_probability >= 0.80) * 100
    )

    top = slc[0].start
    bottom = slc[0].stop
    left = slc[1].start
    right = slc[1].stop

    region_height = bottom - top
    region_width = right - left

    touches_border = (
        top == 0
        or left == 0
        or bottom == height
        or right == width
    )

    # -----------------------------------------------------
    # Candidate score
    #
    # This is NOT probability of oil spill.
    # It is an explainable prioritization score.
    # -----------------------------------------------------

    score = 0.0

    # Confidence contribution
    score += mean_confidence * 40

    # High-confidence pixel contribution
    score += high_confidence_percentage * 0.30

    # Size contribution (capped)
    size_score = min(area / 10000, 1.0) * 20
    score += size_score

    # Border penalty
    if touches_border:
        score -= 10

    score = max(0.0, min(score, 100.0))


    # -----------------------------------------------------
    # Candidate class
    # -----------------------------------------------------

    if score >= 70:
        classification = "HIGH_PRIORITY"

    elif score >= 50:
        classification = "MEDIUM_PRIORITY"

    else:
        classification = "REVIEW"


    regions.append(
        {
            "region_id": region_id,
            "area_pixels": area,
            "mean_confidence": round(
                mean_confidence, 4
            ),
            "maximum_confidence": round(
                max_confidence, 4
            ),
            "high_confidence_pixels_percent": round(
                high_confidence_percentage, 2
            ),
            "width_pixels": region_width,
            "height_pixels": region_height,
            "touches_image_border": touches_border,
            "priority_score": round(score, 2),
            "classification": classification,
        }
    )


# ---------------------------------------------------------
# Rank candidates
# ---------------------------------------------------------

regions.sort(
    key=lambda x: x["priority_score"],
    reverse=True
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

high_priority = sum(
    r["classification"] == "HIGH_PRIORITY"
    for r in regions
)

medium_priority = sum(
    r["classification"] == "MEDIUM_PRIORITY"
    for r in regions
)

review = sum(
    r["classification"] == "REVIEW"
    for r in regions
)


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

result = {
    "module": "OceanShield-AI Candidate Assessment",

    "description": (
        "Explainable region-level prioritization of "
        "U-Net spill candidates. Priority score is "
        "not a probability of oil spill."
    ),

    "total_candidates": len(regions),

    "summary": {
        "high_priority": high_priority,
        "medium_priority": medium_priority,
        "review": review,
    },

    "top_candidates": regions[:10],

    "all_candidates": regions,
}


with open(OUTPUT_JSON, "w", encoding="utf-8") as f:

    json.dump(
        result,
        f,
        indent=4
    )


print("=" * 70)
print("OCEANSHIELD AI - CANDIDATE ASSESSMENT")
print("=" * 70)

print(f"Total candidates : {len(regions)}")
print(f"High priority    : {high_priority}")
print(f"Medium priority  : {medium_priority}")
print(f"Review            : {review}")

print("\nTOP 10 CANDIDATES")
print("-" * 70)

print(
    f"{'Rank':<6}"
    f"{'Area':<14}"
    f"{'Mean':<10}"
    f"{'High%':<10}"
    f"{'Score':<10}"
    f"{'Class':<18}"
)

print("-" * 70)

for rank, region in enumerate(
    regions[:10],
    start=1
):

    print(
        f"{rank:<6}"
        f"{region['area_pixels']:<14,}"
        f"{region['mean_confidence']:<10.3f}"
        f"{region['high_confidence_pixels_percent']:<10.2f}"
        f"{region['priority_score']:<10.2f}"
        f"{region['classification']:<18}"
    )


print("\nAssessment saved to:")
print(OUTPUT_JSON)

print("=" * 70)