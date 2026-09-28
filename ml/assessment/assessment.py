from pathlib import Path
import json


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DETECTION_FILE = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "detection_result.json"
)

GIS_FILE = (
    PROJECT_ROOT
    / "Person3_GIS"
    / "output"
    / "gis_analysis.json"
)

AIS_FILE = (
    PROJECT_ROOT
    / "ml"
    / "attribution"
    / "outputs"
    / "vessel_investigation.json"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "models"
    / "unet_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "assessment"
    / "outputs"
)

OUTPUT_FILE = OUTPUT_DIR / "ai_assessment.json"


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def classify_confidence(mean_confidence):
    if mean_confidence >= 0.80:
        return "HIGH"
    elif mean_confidence >= 0.60:
        return "MODERATE"
    else:
        return "LOW"


def classify_spatial_extent(area_km2):
    if area_km2 >= 50:
        return "LARGE"
    elif area_km2 >= 20:
        return "MODERATE"
    else:
        return "SMALL"


def classify_detection_quality(mean_confidence, coverage):
    if mean_confidence >= 0.70 and coverage >= 2:
        return "STRONG"
    elif mean_confidence >= 0.50:
        return "MODERATE"
    else:
        return "LOW"


# ============================================================
# LOAD EXISTING PIPELINE RESULTS
# ============================================================

print("Loading detection result...")
detection = load_json(DETECTION_FILE)

print("Loading GIS analysis...")
gis = load_json(GIS_FILE)

print("Loading AIS investigation...")
ais = load_json(AIS_FILE)


# ============================================================
# DETECTION INFORMATION
# ============================================================

spill = detection.get("spill", {})
confidence = detection.get("confidence", {})
centroid = detection.get("centroid", {})

scene = detection.get("scene")
acquisition_date = detection.get("acquisition_date")

area_km2 = float(spill.get("area_km2", 0))
coverage = float(spill.get("percentage", 0))

mean_confidence = float(
    confidence.get("mean_spill", 0)
)

maximum_confidence = float(
    confidence.get("maximum", 0)
)


# ============================================================
# MODEL INFORMATION
# ============================================================

model_checkpoint = None
validation_dice = None
validation_iou = None
best_epoch = None

if MODEL_FILE.exists():

    try:
        import torch

        checkpoint = torch.load(
            MODEL_FILE,
            map_location="cpu",
            weights_only=False
        )

        if isinstance(checkpoint, dict):

            best_epoch = checkpoint.get("epoch")

            validation_dice = checkpoint.get(
                "val_dice"
            )

            validation_iou = checkpoint.get(
                "val_iou"
            )

    except Exception as error:

        print(
            f"Warning: could not read model metadata: {error}"
        )


# ============================================================
# AIS CANDIDATE ANALYSIS
# ============================================================

candidates = ais.get("candidates", [])

primary_candidate = None

if candidates:
    primary_candidate = sorted(
        candidates,
        key=lambda candidate: candidate.get(
            "evidence_score",
            0
        ),
        reverse=True
    )[0]


candidate_summary = {
    "total_regions": 1,
    "primary_candidate": {
        "region_id": "SPILL-001",
        "area_pixels": spill.get("pixels", 0),
        "area_km2": area_km2,
        "mean_confidence": mean_confidence,
        "coverage_percentage": coverage
    }
}


# ============================================================
# CLASSIFICATIONS
# ============================================================

confidence_level = classify_confidence(
    mean_confidence
)

spatial_extent = classify_spatial_extent(
    area_km2
)

detection_quality = classify_detection_quality(
    mean_confidence,
    coverage
)


# ============================================================
# EXPLAINABLE ASSESSMENT
# ============================================================

explanation_parts = []

if detection.get("spill_detected"):
    explanation_parts.append(
        f"The U-Net model detected a candidate spill region "
        f"in scene {scene}."
    )

explanation_parts.append(
    f"The predicted footprint covers approximately "
    f"{area_km2:.2f} km² ({coverage:.2f}% of the scene)."
)

explanation_parts.append(
    f"Mean confidence within the predicted spill region "
    f"is {mean_confidence * 100:.1f}%, with a maximum "
    f"pixel confidence of {maximum_confidence * 100:.1f}%."
)

gis_risk = gis.get("risk_level")

if gis_risk:
    explanation_parts.append(
        f"GIS analysis assigns a prototype risk level of "
        f"{gis_risk} based on the configured spatial-risk rules."
    )

if primary_candidate:

    explanation_parts.append(
        f"AIS investigation identified MMSI "
        f"{primary_candidate.get('mmsi')} as the highest "
        f"evidence-ranked candidate, with an evidence score "
        f"of {primary_candidate.get('evidence_score', 0):.2f}."
    )

explanation_parts.append(
    "The assessment supports investigation and prioritization; "
    "it does not independently prove that the detected feature "
    "is oil or that any vessel caused the spill."
)

assessment_explanation = " ".join(
    explanation_parts
)


# ============================================================
# BUILD ASSESSMENT RESULT
# ============================================================

assessment_result = {

    "project": "OceanShield-AI",

    "assessment_type": (
        "AI-assisted spill interpretation"
    ),

    "status": "READY",

    "assessment": {

        "scene": scene,

        "acquisition_date": acquisition_date,

        "spill_detected": bool(
            detection.get("spill_detected", False)
        ),

        "confidence_level": confidence_level,

        "spatial_extent": spatial_extent,

        "detection_quality": detection_quality,

        "explanation": assessment_explanation,

        "centroid": centroid
    },

    "candidate_analysis": candidate_summary,

    "model": {

        "architecture": "U-Net",

        "checkpoint": str(
            MODEL_FILE
        ),

        "patch_size": 256,

        "stride": 128,

        "threshold": 0.50,

        "best_epoch": best_epoch
    },

    "validation": {

        "iou": validation_iou,

        "dice": validation_dice,

        "precision": None,

        "recall": None,

        "note": (
            "Validation metrics stored in the best model "
            "checkpoint. Precision and recall were not stored "
            "in the checkpoint."
        )
    },

    "gis_context": {

        "risk_level": gis.get(
            "risk_level"
        ),

        "risk_classification": gis.get(
            "risk_classification"
        ),

        "movement": gis.get(
            "movement"
        )
    },

    "ais_context": {

        "data_type": ais.get(
            "data_type"
        ),

        "candidate_count": len(
            candidates
        ),

        "primary_candidate": (
            primary_candidate
            if primary_candidate
            else None
        )
    },

    "limitations": [

        "AI confidence does not prove the detected feature is oil.",

        "GIS risk is a prototype rule-based classification.",

        "AIS data in this demonstration is synthetic.",

        "AIS evidence ranking does not prove vessel causation.",

        "Human verification is required before operational action."
    ]
}


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        assessment_result,
        file,
        indent=4
    )


print()
print("=" * 60)
print("AI ASSESSMENT GENERATED")
print("=" * 60)

print(f"Scene: {scene}")
print(f"Spill area: {area_km2:.2f} km²")
print(f"Coverage: {coverage:.2f}%")
print(
    f"Mean confidence: "
    f"{mean_confidence * 100:.2f}%"
)
print(
    f"Confidence level: "
    f"{confidence_level}"
)
print(
    f"Detection quality: "
    f"{detection_quality}"
)

if primary_candidate:
    print(
        f"Top AIS candidate: "
        f"{primary_candidate.get('mmsi')}"
    )
    print(
        f"Evidence score: "
        f"{primary_candidate.get('evidence_score')}"
    )

print(
    f"Output: {OUTPUT_FILE}"
)