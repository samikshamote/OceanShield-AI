import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETECTION_JSON = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "detection_result.json"
)

CANDIDATE_JSON = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "candidate_assessment.json"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "ai_assessment.json"
)


def confidence_level(mean_confidence):
    if mean_confidence >= 0.75:
        return "HIGH"
    elif mean_confidence >= 0.60:
        return "MEDIUM"
    else:
        return "LOW"


def extent_level(area_km2):
    if area_km2 >= 50:
        return "LARGE"
    elif area_km2 >= 20:
        return "MODERATE"
    else:
        return "SMALL"


def detection_quality(iou, dice):
    if iou >= 0.50 and dice >= 0.65:
        return "STRONG"
    elif iou >= 0.25 and dice >= 0.40:
        return "MODERATE"
    else:
        return "LIMITED"


def main():

    if not DETECTION_JSON.exists():
        raise FileNotFoundError(
            f"Detection result not found: {DETECTION_JSON}"
        )

    if not CANDIDATE_JSON.exists():
        raise FileNotFoundError(
            f"Candidate assessment not found: {CANDIDATE_JSON}"
        )

    with open(DETECTION_JSON, "r", encoding="utf-8") as f:
        detection = json.load(f)

    with open(CANDIDATE_JSON, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    spill = detection.get("spill", {})
    confidence = detection.get("confidence", {})

    area_km2 = float(spill.get("area_km2", 0))
    coverage = float(spill.get("percentage", 0))

    mean_confidence = float(confidence.get("mean_spill", 0))
    max_confidence = float(confidence.get("maximum", 0))

    all_candidates = candidates.get("all_candidates", [])
    
    # Select the largest coherent detected region as the
    # primary candidate instead of using confidence alone.
    primary_candidate = None

    if all_candidates:
        primary_candidate = max(
            all_candidates,
            key=lambda x: x.get("area_pixels", 0)
        )

    candidate_count = len(all_candidates)

    if primary_candidate:
        primary_area = primary_candidate.get("area_pixels", 0)
        primary_mean = primary_candidate.get("mean_confidence", 0)
        primary_max = primary_candidate.get("maximum_confidence", 0)
        primary_border = primary_candidate.get(
            "touches_image_border",
            False
        )
        primary_id = primary_candidate.get("region_id")
    else:
        primary_area = 0
        primary_mean = 0
        primary_max = 0
        primary_border = False
        primary_id = None

    # Validation metrics are from the held-out test scene.
    validation = {
        "dataset": "Held-out test scene",
        "scene": detection.get("scene"),
        "iou": 0.3078,
        "dice": 0.4707,
        "precision": 0.3597,
        "recall": 0.6808,
        "note": (
            "Metrics represent segmentation performance on the "
            "held-out test scene and are not a probability of detection."
        )
    }

    assessment = {
        "confidence_level": confidence_level(mean_confidence),
        "spatial_extent": extent_level(area_km2),
        "detection_quality": detection_quality(
            validation["iou"],
            validation["dice"]
        ),
        "human_verification_required": True,
        "primary_candidate_method": (
            "Largest coherent detected region"
        ),
        "explanation": (
            "The U-Net generated a probability map from Sentinel-1 SAR "
            "imagery. Connected-component analysis identified multiple "
            "candidate regions. The largest coherent region is treated "
            "as the primary candidate, while additional regions remain "
            "secondary candidates for human verification. Confidence "
            "values describe model output strength and do not by "
            "themselves prove the presence of oil."
        )
    }

    result = {
        "project": "OceanShield-AI",

        "model": {
            "architecture": "U-Net",
            "input": "Sentinel-1 SAR VV",
            "patch_size": 256,
            "threshold": 0.50
        },

        "scene": {
            "name": detection.get("scene"),
            "acquisition_date": detection.get("acquisition_date"),
            "crs": detection.get("crs")
        },

        "spill": {
            "detected": detection.get("spill_detected", False),
            "area_km2": area_km2,
            "coverage_percent": coverage,
            "centroid": detection.get("centroid"),
            "bounding_box": detection.get("bounding_box")
        },

        "confidence": {
            "mean": mean_confidence,
            "maximum": max_confidence,
            "level": confidence_level(mean_confidence)
        },

        "candidate_analysis": {
            "total_regions": candidate_count,

            "primary_candidate": {
                "region_id": primary_id,
                "area_pixels": primary_area,
                "mean_confidence": primary_mean,
                "maximum_confidence": primary_max,
                "touches_image_border": primary_border
            },

            "selection_reason": (
                "Largest coherent detected region, rather than "
                "highest confidence alone."
            )
        },

        "assessment": assessment,

        "validation": validation,

        "limitations": [
            "AI output requires human verification.",
            "Confidence is model confidence, not a probability of oil.",
            "AIS attribution must not be interpreted as proof of causation.",
            "Environmental risk classification is prototype-level.",
            "Validation metrics are based on a held-out test scene."
        ]
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print()
    print("=" * 70)
    print("OCEANSHIELD-AI — AI ASSESSMENT")
    print("=" * 70)

    print(f"Scene                 : {detection.get('scene')}")
    print(f"Spill detected        : {detection.get('spill_detected', False)}")
    print(f"Detected area         : {area_km2:.2f} km²")
    print(f"Coverage              : {coverage:.2f}%")
    print(f"Mean confidence      : {mean_confidence:.3f}")
    print(f"Confidence level      : {confidence_level(mean_confidence)}")

    print()
    print("CANDIDATE ANALYSIS")
    print("-" * 70)
    print(f"Candidate regions     : {candidate_count}")

    if primary_candidate:
        print(f"Primary region ID     : {primary_id}")
        print(f"Primary area          : {primary_area:,} pixels")
        print(f"Primary confidence    : {primary_mean:.3f}")
        print(f"Touches image border  : {primary_border}")

    print()
    print("VALIDATION")
    print("-" * 70)
    print(f"IoU                   : {validation['iou'] * 100:.2f}%")
    print(f"Dice / F1             : {validation['dice'] * 100:.2f}%")
    print(f"Precision             : {validation['precision'] * 100:.2f}%")
    print(f"Recall                : {validation['recall'] * 100:.2f}%")
    print(f"Detection quality     : {assessment['detection_quality']}")

    print()
    print("HUMAN VERIFICATION    : REQUIRED")
    print()
    print(f"Saved to: {OUTPUT_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()