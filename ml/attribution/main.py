import json
import os
from pathlib import Path

import pandas as pd

from ais_loader import load_ais_data
from candidate_filter import (
    filter_by_distance,
    filter_by_time
)
from trajectory import analyze_trajectory
from scoring import generate_scores


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

AIS_FILE = (
    PROJECT_ROOT
    / "ml"
    / "attribution"
    / "data"
    / "ais_sample.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "attribution"
    / "outputs"
)

RANKED_CSV = OUTPUT_DIR / "ranked_candidates.csv"
INVESTIGATION_JSON = OUTPUT_DIR / "vessel_investigation.json"


# ============================================================
# LOAD DETECTION RESULT
# ============================================================

def load_detection_result():

    if not DETECTION_FILE.exists():
        raise FileNotFoundError(
            f"Detection result not found:\n{DETECTION_FILE}"
        )

    with open(DETECTION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# CIRCULAR MEAN FOR COG
# ============================================================

def calculate_average_cog(values):
    """
    Calculate circular mean of compass headings.

    Example:
    359° and 1° should average to approximately 0°,
    not 180°.
    """

    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).dropna()

    if values.empty:
        return None

    import math

    radians = [
        math.radians(float(value) % 360)
        for value in values
    ]

    sin_mean = sum(math.sin(x) for x in radians) / len(radians)
    cos_mean = sum(math.cos(x) for x in radians) / len(radians)

    angle = math.degrees(
        math.atan2(sin_mean, cos_mean)
    )

    return round(angle % 360, 2)


# ============================================================
# GET VESSEL INFORMATION
# ============================================================

def get_vessel_information(ais_data, mmsi):

    vessel_data = ais_data[
        ais_data["MMSI"].astype(str) == str(mmsi)
    ].copy()

    if vessel_data.empty:
        return {
            "average_cog": None,
            "destination": None
        }

    # --------------------------------------------------------
    # Average heading
    # --------------------------------------------------------

    average_cog = None

    if "COG" in vessel_data.columns:
        average_cog = calculate_average_cog(
            vessel_data["COG"]
        )

    # --------------------------------------------------------
    # Destination
    # --------------------------------------------------------

    destination = None

    if "destination" in vessel_data.columns:

        destinations = (
            vessel_data["destination"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        destinations = destinations[
            destinations != ""
        ]

        if not destinations.empty:
            destination = destinations.mode().iloc[0]

    return {
        "average_cog": average_cog,
        "destination": destination
    }


# ============================================================
# BUILD INVESTIGATION JSON
# ============================================================

def build_investigation_json(
    detection_result,
    ranked_vessels,
    ais_data,
    spill_time
):

    candidates = []

    for _, row in ranked_vessels.iterrows():

        mmsi = str(row["MMSI"])

        vessel_info = get_vessel_information(
            ais_data,
            mmsi
        )

        candidate = {
            "rank": int(row["rank"]),

            "mmsi": mmsi,

            "ship_type": str(
                row.get("ship_type", "Unknown")
            ),

            "distance_km": round(
                float(row["distance_km"]),
                3
            ),

            "time_difference_minutes": round(
                float(row["time_difference_minutes"]),
                1
            ),

            "trajectory": str(
                row.get("trajectory", "Unknown")
            ),

            "average_speed": round(
                float(row["average_speed"]),
                2
            ),

            "average_cog": vessel_info["average_cog"],

            "destination": vessel_info["destination"],

            "evidence_score": round(
                float(row["attribution_likelihood"]),
                2
            )
        }

        # ----------------------------------------------------
        # Preserve additional scoring evidence if available
        # ----------------------------------------------------

        optional_score_fields = [
            "distance_score",
            "time_score",
            "trajectory_score",
            "approach_score",
            "heading_score",
            "speed_score"
        ]

        evidence = {}

        for field in optional_score_fields:

            if field in ranked_vessels.columns:

                value = row[field]

                if pd.notna(value):

                    evidence[field] = round(
                        float(value),
                        4
                    )

        if evidence:
            candidate["evidence_components"] = evidence

        candidates.append(candidate)

    # ========================================================
    # FINAL JSON STRUCTURE
    # ========================================================

    investigation = {
        "project": "OceanShield-AI",

        "data_type": "Synthetic AIS Demo Scenario",

        "purpose": (
            "AIS-based vessel investigation and evidence ranking "
            "around a detected oil-spill location."
        ),

        "important_note": (
            "The attribution score is an evidence-based ranking "
            "score. It does not prove that a vessel caused the spill."
        ),

        "incident": {
            "scene": detection_result.get(
                "scene",
                "Unknown"
            ),

            "acquisition_date": detection_result.get(
                "acquisition_date",
                None
            ),

            "latitude": detection_result[
                "centroid"
            ]["latitude"],

            "longitude": detection_result[
                "centroid"
            ]["longitude"],

            "detected_area_km2": detection_result.get(
                "spill_area_km2",
                detection_result.get(
                    "spill_area",
                    None
                )
            ),

            "mean_confidence": detection_result.get(
                "mean_spill_confidence",
                detection_result.get(
                    "mean_confidence",
                    None
                )
            ),

            "spill_time": spill_time,

            "spill_time_type": "DEMO_REFERENCE_TIME",

            "spill_time_note": (
                "Reference/demo time used for AIS correlation. "
                "The exact Sentinel-1 acquisition time is not "
                "available in detection_result.json."
            )
        },

        "filtering": {
            "maximum_distance_km": 10,

            "time_window_minutes": 60
        },

        "candidates": candidates
    }

    return investigation


# ============================================================
# SAVE INVESTIGATION JSON
# ============================================================

def save_investigation_json(investigation):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        INVESTIGATION_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            investigation,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("OCEANSHIELD-AI")
    print("AIS VESSEL ATTRIBUTION PIPELINE")
    print("=" * 70)

    # ========================================================
    # LOAD DETECTION RESULT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("LOADING DETECTION RESULT")
    print("=" * 70)

    detection_result = load_detection_result()

    if not detection_result.get("spill_detected", True):
        print("\nNo spill detected.")
        return

    # ========================================================
    # SPILL INFORMATION
    # ========================================================

    centroid = detection_result["centroid"]

    spill_lat = float(
        centroid["latitude"]
    )

    spill_lon = float(
        centroid["longitude"]
    )

    acquisition_date = detection_result.get(
        "acquisition_date",
        "Unknown"
    )

    spill_area = detection_result.get(
        "spill_area_km2",
        detection_result.get(
            "spill_area",
            "Unknown"
        )
    )

    mean_confidence = detection_result.get(
        "mean_spill_confidence",
        detection_result.get(
            "mean_confidence",
            "Unknown"
        )
    )

    # ========================================================
    # DEMO TIME
    # ========================================================

    spill_time = os.getenv(
        "OCEANSHIELD_SPILL_TIME"
    )

    if not spill_time:

        print(
            "\nWARNING:"
        )

        print(
            "Exact Sentinel-1 acquisition time is not "
            "available in detection_result.json."
        )

        print(
            "AIS time filtering requires "
            "OCEANSHIELD_SPILL_TIME."
        )

        print(
            '\nExample PowerShell command:'
        )

        print(
            '$env:OCEANSHIELD_SPILL_TIME = '
            '"2018-09-26 10:30:00"'
        )

        return

    print("\nSPILL INFORMATION")
    print("-" * 70)

    print(
        f"Detection scene : "
        f"{detection_result.get('scene', 'Unknown')}"
    )

    print(
        f"Acquisition date: {acquisition_date}"
    )

    print(
        f"Latitude        : {spill_lat}"
    )

    print(
        f"Longitude       : {spill_lon}"
    )

    print(
        f"Spill time      : {spill_time}"
    )

    print(
        f"Detected area   : {spill_area} km2"
    )

    print(
        f"Mean confidence: {mean_confidence}"
    )

    print(
        "Distance limit  : 10 km"
    )

    print(
        "Time window     : +/-60 minutes"
    )

    # ========================================================
    # STEP 1 — LOAD AIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 1: LOADING AIS DATA")
    print("=" * 70)

    ais_data = load_ais_data(
        str(AIS_FILE)
    )

    print(
        "\nAIS Columns:"
    )

    print(
        list(ais_data.columns)
    )

    print(
        f"Total AIS records loaded: "
        f"{len(ais_data)}"
    )

    # ========================================================
    # STEP 2 — DISTANCE FILTER
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 2: DISTANCE FILTERING")
    print("=" * 70)

    distance_candidates = filter_by_distance(
        ais_data,
        spill_lat,
        spill_lon,
        max_distance_km=10
    )

    print(
        f"Records within 10 km: "
        f"{len(distance_candidates)}"
    )

    # ========================================================
    # STEP 3 — TIME FILTER
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 3: TIME FILTERING")
    print("=" * 70)

    time_candidates = filter_by_time(
    distance_candidates,
    spill_time,
    time_window_minutes=60
)

    print(
        f"Records within +/-60 minutes: "
        f"{len(time_candidates)}"
    )

    if time_candidates.empty:

        print(
            "\nNo AIS candidates found."
        )

        return

    # ========================================================
    # STEP 4 — TRAJECTORY ANALYSIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 4: TRAJECTORY ANALYSIS")
    print("=" * 70)

    trajectory_results = analyze_trajectory(
        time_candidates,
        spill_lat,
        spill_lon
    )

    print(
        f"Vessels analyzed: "
        f"{len(trajectory_results)}"
    )

    # ========================================================
    # STEP 5 — ATTRIBUTION SCORING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 5: ATTRIBUTION SCORING")
    print("=" * 70)

    ranked_vessels = generate_scores(
        time_candidates,
        trajectory_results,
        ais_data,
        spill_lat,
        spill_lon,
        10,
        60
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL VESSEL RANKING")
    print("=" * 70)

    display_columns = [
        "rank",
        "MMSI",
        "ship_type",
        "distance_km",
        "time_difference_minutes",
        "trajectory",
        "average_speed",
        "attribution_likelihood"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in ranked_vessels.columns
    ]

    print(
        ranked_vessels[
            available_columns
        ].to_string(index=False)
    )

    # ========================================================
    # TOP CANDIDATE
    # ========================================================

    if not ranked_vessels.empty:

        top = ranked_vessels.iloc[0]

        print("\n")
        print("=" * 70)
        print("TOP CANDIDATE")
        print("=" * 70)

        print(
            f"MMSI                  : "
            f"{top['MMSI']}"
        )

        print(
            f"Ship type             : "
            f"{top['ship_type']}"
        )

        print(
            f"Closest distance      : "
            f"{top['distance_km']:.2f} km"
        )

        print(
            f"Time difference       : "
            f"{top['time_difference_minutes']:.1f} minutes"
        )

        print(
            f"Trajectory            : "
            f"{top['trajectory']}"
        )

        print(
            f"Average speed         : "
            f"{top['average_speed']:.2f} knots"
        )

        print(
            f"Attribution score     : "
            f"{top['attribution_likelihood']:.2f}%"
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ranked_vessels.to_csv(
        RANKED_CSV,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("RESULT SAVED")
    print("=" * 70)

    print(
        f"CSV file: {RANKED_CSV}"
    )

    # ========================================================
    # GENERATE INVESTIGATION JSON
    # ========================================================

    investigation = build_investigation_json(
        detection_result,
        ranked_vessels,
        ais_data,
        spill_time
    )

    save_investigation_json(
        investigation
    )

    print(
        f"Investigation JSON: "
        f"{INVESTIGATION_JSON}"
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\nNote: Attribution score is an evidence-based "
        "ranking score and does not prove causation."
    )

    print(
        "Note: AIS data in this demonstration is synthetic "
        "demo data and must not be interpreted as real vessel involvement."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()