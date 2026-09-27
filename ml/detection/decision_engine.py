from pathlib import Path
import json


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
)

AI_ASSESSMENT_PATH = OUTPUT_DIR / "ai_assessment.json"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_assessment.json"
DETECTION_PATH = OUTPUT_DIR / "detection_result.json"

DECISION_OUTPUT = OUTPUT_DIR / "decision_result.json"


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


print()
print("=" * 70)
print("OCEANSHIELD-AI — DECISION ENGINE")
print("=" * 70)


detection = load_json(
    DETECTION_PATH
)

assessment = load_json(
    AI_ASSESSMENT_PATH
)

candidates = load_json(
    CANDIDATE_PATH
)


# ============================================================
# EXTRACT DETECTION INFORMATION
# ============================================================

spill = detection.get(
    "spill",
    {}
)

confidence = detection.get(
    "confidence",
    {}
)

centroid = detection.get(
    "centroid",
    {}
)

spill_detected = detection.get(
    "spill_detected",
    False
)

area_km2 = float(
    spill.get(
        "area_km2",
        0
    )
)

coverage = float(
    spill.get(
        "percentage",
        0
    )
)

mean_confidence = float(
    confidence.get(
        "mean_spill",
        0
    )
)

max_confidence = float(
    confidence.get(
        "maximum",
        0
    )
)


# ============================================================
# CANDIDATE INFORMATION
# ============================================================

# The AI assessment selects the primary candidate based on
# the largest coherent detected region.
# The decision engine uses that same candidate for consistency.

candidate_analysis = assessment.get(
    "candidate_analysis",
    {}
)

primary_candidate = candidate_analysis.get(
    "primary_candidate",
    {}
)

top_candidates = candidates.get(
    "top_candidates",
    []
)

all_candidates = candidates.get(
    "all_candidates",
    []
)

# Prefer the complete candidate list when available.
candidate_count = len(all_candidates)

# Fallback to top_candidates if the complete list is not stored.
if candidate_count == 0:
    candidate_count = len(top_candidates)


if primary_candidate:

    primary_region_id = primary_candidate.get(
        "region_id"
    )

    primary_area = int(
        primary_candidate.get(
            "area_pixels",
            0
        )
    )

    primary_confidence = float(
        primary_candidate.get(
            "mean_confidence",
            0
        )
    )

    # The AI assessment is the source of truth for
    # the primary region. The old heuristic priority
    # score may not exist here.
    primary_priority_score = 0.0

    primary_classification = "PRIMARY_CANDIDATE"

else:

    primary_region_id = None
    primary_area = 0
    primary_confidence = 0
    primary_priority_score = 0.0
    primary_classification = "REVIEW"


# ============================================================
# DECISION SCORING
# ============================================================

score = 0.0


# Spill detected
if spill_detected:
    score += 25


# Spill area
if area_km2 >= 50:
    score += 25

elif area_km2 >= 10:
    score += 18

elif area_km2 >= 1:
    score += 10


# Model confidence
if mean_confidence >= 0.80:
    score += 25

elif mean_confidence >= 0.65:
    score += 18

elif mean_confidence >= 0.50:
    score += 10


# Candidate confidence
if primary_confidence >= 0.80:
    score += 15

elif primary_confidence >= 0.65:
    score += 10

elif primary_confidence >= 0.50:
    score += 5


# Primary candidate evidence
# The primary candidate is selected by the AI assessment
# using coherent-region analysis. It is not treated as
# a probability or validated risk score.
if primary_area >= 10000:
    score += 10

elif primary_area >= 1000:
    score += 7

elif primary_area >= 100:
    score += 4

# ============================================================
# RISK LEVEL
# ============================================================

if score >= 75:
    risk_level = "HIGH"

elif score >= 50:
    risk_level = "MEDIUM"

else:
    risk_level = "LOW"


# ============================================================
# ALERT PRIORITY
# ============================================================

if risk_level == "HIGH":

    alert_priority = "IMMEDIATE"

elif risk_level == "MEDIUM":

    alert_priority = "PRIORITY_REVIEW"

else:

    alert_priority = "MONITOR"


# ============================================================
# HUMAN VERIFICATION
# ============================================================

if (
    not spill_detected
    or mean_confidence < 0.80
    or assessment.get(
        "validation",
        {}
    ).get(
        "detection_quality",
        "MODERATE"
    ) != "HIGH"
):

    human_verification = True

else:

    human_verification = False


# ============================================================
# RECOMMENDED ACTIONS
# ============================================================

actions = []

if spill_detected:

    actions.append(
        "Verify AI spill detection"
    )

    actions.append(
    "Inspect the primary and secondary candidate regions"
    )

    actions.append(
        "Cross-check nearby vessel activity"
    )

    actions.append(
        "Assess spill movement and affected area"
    )

    if risk_level == "HIGH":

        actions.append(
            "Escalate incident for response assessment"
        )

else:

    actions.append(
        "Continue monitoring satellite observations"
    )


if human_verification:

    actions.insert(
        0,
        "Human verification required before operational action"
    )


# ============================================================
# DECISION EXPLANATION
# ============================================================

if risk_level == "HIGH":

    explanation = (
    "A large potential spill was detected with supporting "
    "AI evidence. The result requires immediate human review "
    "before any operational response."
    )

elif risk_level == "MEDIUM":

    explanation = (
        "AI detected a potential spill with moderate-to-strong "
        "evidence. Human verification is recommended before escalation."
    )

else:

    explanation = (
        "Current AI evidence does not indicate a high-priority "
        "incident. Continue monitoring."
    )


# ============================================================
# CREATE DECISION RESULT
# ============================================================

decision_result = {

    "project": "OceanShield-AI",

    "decision_engine": {
        "version": "1.0",
        "status": "prototype"
    },

    "scene": detection.get(
        "scene"
    ),

    "spill": {
        "detected": bool(
            spill_detected
        ),
        "area_km2": area_km2,
        "coverage_percent": coverage
    },

    "location": {
        "latitude": centroid.get(
            "latitude"
        ),
        "longitude": centroid.get(
            "longitude"
        )
    },

    "ai_evidence": {

        "mean_confidence": mean_confidence,

        "maximum_confidence": max_confidence,

        "candidate_regions": candidate_count,

        "primary_region_id": primary_region_id,

        "primary_region_area_pixels": primary_area,

        "primary_region_confidence": primary_confidence,

        "primary_priority_score": primary_priority_score,

        "primary_classification": primary_classification
    },

    "decision": {

        "risk_level": risk_level,

        "alert_priority": alert_priority,

        "decision_score": round(
            score,
            2
        ),

        "human_verification_required": (
            human_verification
        ),

        "explanation": explanation
    },

    "recommended_actions": actions

}


# ============================================================
# SAVE RESULT
# ============================================================

with open(
    DECISION_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        decision_result,
        f,
        indent=4
    )


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("-" * 70)

print(
    f"Scene                 : "
    f"{decision_result['scene']}"
)

print(
    f"Spill detected        : "
    f"{spill_detected}"
)

print(
    f"Spill area            : "
    f"{area_km2:.2f} km²"
)

print(
    f"AI confidence         : "
    f"{mean_confidence:.3f}"
)

print(
    f"Candidate regions     : "
    f"{candidate_count}"
)

print(
    f"Primary region        : "
    f"{primary_region_id}"
)

print(
    f"Primary candidate     : "
    f"{primary_classification}"
)

print()
print("DECISION")
print("-" * 70)

print(
    f"Decision score        : "
    f"{score:.2f}/100"
)

print(
    f"Risk level            : "
    f"{risk_level}"
)

print(
    f"Alert priority        : "
    f"{alert_priority}"
)

print(
    f"Human verification    : "
    f"{'REQUIRED' if human_verification else 'NOT REQUIRED'}"
)

print()
print("RECOMMENDED ACTIONS")
print("-" * 70)

for index, action in enumerate(
    actions,
    start=1
):

    print(
        f"{index}. {action}"
    )

print()
print(
    f"Decision result saved:"
    f"\n{DECISION_OUTPUT}"
)

print("=" * 70)