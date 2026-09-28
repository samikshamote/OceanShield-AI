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

ASSESSMENT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "assessment"
    / "outputs"
    / "ai_assessment.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "decision"
    / "outputs"
)

OUTPUT_FILE = OUTPUT_DIR / "decision_result.json"


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# LOAD PIPELINE RESULTS
# ============================================================

print("Loading detection result...")
detection = load_json(DETECTION_FILE)

print("Loading GIS analysis...")
gis = load_json(GIS_FILE)

print("Loading AIS investigation...")
ais = load_json(AIS_FILE)

print("Loading AI assessment...")
assessment = load_json(ASSESSMENT_FILE)


# ============================================================
# DETECTION SIGNAL
# ============================================================

spill_detected = bool(
    detection.get("spill_detected", False)
)

spill = detection.get("spill", {})
confidence = detection.get("confidence", {})

area_km2 = float(
    spill.get("area_km2", 0)
)

coverage = float(
    spill.get("percentage", 0)
)

mean_confidence = float(
    confidence.get("mean_spill", 0)
)


# ============================================================
# GIS RISK
# ============================================================

gis_risk = str(
    gis.get("risk_level", "UNKNOWN")
).upper()


# ============================================================
# AIS EVIDENCE
# ============================================================

candidates = ais.get("candidates", [])

top_candidate = None

if candidates:
    top_candidate = sorted(
        candidates,
        key=lambda candidate: candidate.get(
            "evidence_score",
            0
        ),
        reverse=True
    )[0]

top_evidence_score = (
    float(top_candidate.get("evidence_score", 0))
    if top_candidate
    else 0.0
)


# ============================================================
# RULE-BASED DECISION SCORE
# ============================================================

# This score is NOT an oil probability.
# It combines independent prototype signals to prioritize
# incidents for human review.

confidence_component = mean_confidence * 40

coverage_component = min(
    coverage / 10,
    1.0
) * 20

risk_component = {
    "HIGH": 20,
    "MEDIUM": 12,
    "LOW": 5
}.get(
    gis_risk,
    0
)

ais_component = min(
    top_evidence_score / 100,
    1.0
) * 20

decision_score = (
    confidence_component
    + coverage_component
    + risk_component
    + ais_component
)

decision_score = round(
    min(decision_score, 100),
    2
)


# ============================================================
# ALERT PRIORITY
# ============================================================

if (
    spill_detected
    and gis_risk == "HIGH"
    and top_evidence_score >= 80
):
    alert_priority = "HIGH"

elif (
    spill_detected
    and (
        gis_risk in ["HIGH", "MEDIUM"]
        or top_evidence_score >= 60
    )
):
    alert_priority = "MEDIUM"

elif spill_detected:
    alert_priority = "LOW"

else:
    alert_priority = "MONITOR"


# ============================================================
# DECISION EXPLANATION
# ============================================================

explanation_parts = []

if spill_detected:

    explanation_parts.append(
        f"The detection pipeline identified a candidate "
        f"spill covering approximately {area_km2:.2f} km² "
        f"with {mean_confidence * 100:.1f}% mean model "
        f"confidence."
    )

if gis_risk != "UNKNOWN":

    explanation_parts.append(
        f"The GIS module assigns a prototype {gis_risk} "
        f"risk classification using the configured "
        f"spatial-risk rules."
    )

if top_candidate:

    explanation_parts.append(
        f"AIS correlation identified {len(candidates)} "
        f"candidate vessels within the configured "
        f"investigation window. MMSI "
        f"{top_candidate.get('mmsi')} has the highest "
        f"evidence score at "
        f"{top_evidence_score:.2f}."
    )

explanation_parts.append(
    f"The resulting decision-support score is "
    f"{decision_score:.2f}/100. This score prioritizes "
    f"the incident for investigation; it is not an "
    f"oil-detection probability or proof of vessel causation."
)

explanation_parts.append(
    "Human verification is required before operational "
    "action."
)

decision_explanation = " ".join(
    explanation_parts
)


# ============================================================
# BUILD RESULT
# ============================================================

decision_result = {

    "project": "OceanShield-AI",

    "decision_type": (
        "Rule-based decision support"
    ),

    "status": "READY",

    "decision": {

        "risk_level": gis_risk,

        "alert_priority": alert_priority,

        "decision_score": decision_score,

        "human_verification_required": True,

        "explanation": decision_explanation
    },

    "score_components": {

        "detection_confidence_component": round(
            confidence_component,
            2
        ),

        "spatial_coverage_component": round(
            coverage_component,
            2
        ),

        "gis_risk_component": round(
            risk_component,
            2
        ),

        "ais_evidence_component": round(
            ais_component,
            2
        )
    },

    "inputs": {

        "spill_detected": spill_detected,

        "spill_area_km2": area_km2,

        "coverage_percentage": coverage,

        "mean_confidence": mean_confidence,

        "gis_risk": gis_risk,

        "ais_candidate_count": len(
            candidates
        ),

        "top_ais_candidate": (
            top_candidate
            if top_candidate
            else None
        )
    },

    "safety": {

        "human_verification_required": True,

        "not_an_oil_probability": True,

        "not_vessel_causation_proof": True,

        "ais_data_type": ais.get(
            "data_type"
        )
    },

    "limitations": [

        "Decision score is a prototype prioritization score.",

        "The score is not a probability of oil detection.",

        "GIS risk classification is rule-based and not scientifically validated.",

        "AIS data used in this demonstration is synthetic.",

        "AIS evidence ranking does not establish causation.",

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
        decision_result,
        file,
        indent=4
    )


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("DECISION INTELLIGENCE GENERATED")
print("=" * 60)

print(
    f"Risk level: {gis_risk}"
)

print(
    f"Alert priority: {alert_priority}"
)

print(
    f"Decision-support score: "
    f"{decision_score}/100"
)

print(
    "Human verification: REQUIRED"
)

if top_candidate:

    print(
        f"Top AIS candidate: "
        f"{top_candidate.get('mmsi')}"
    )

    print(
        f"AIS evidence score: "
        f"{top_evidence_score}"
    )

print(
    f"Output: {OUTPUT_FILE}"
)