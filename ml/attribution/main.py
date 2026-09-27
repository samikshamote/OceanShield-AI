import json
import os
from pathlib import Path

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

DETECTION_RESULT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "detection_result.json"
)

AIS_FILE_PATH = (
    PROJECT_ROOT
    / "ml"
    / "attribution"
    / "data"
    / "ais_sample.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "ml"
    / "attribution"
    / "outputs"
)


# ============================================================
# LOAD DETECTION RESULT
# ============================================================

def load_detection_result():

    if not DETECTION_RESULT_PATH.exists():

        raise FileNotFoundError(
            "Detection result not found:\n"
            f"{DETECTION_RESULT_PATH}\n\n"
            "Run the U-Net detection pipeline first:\n"
            "python ml\\detection\\predict.py"
        )

    with open(
        DETECTION_RESULT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        result = json.load(file)

    return result


# ============================================================
# AIS VESSEL ATTRIBUTION - MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("OCEANSHIELD-AI")
    print("AIS VESSEL ATTRIBUTION PIPELINE")
    print("=" * 70)

    # ========================================================
    # 1. LOAD AI DETECTION RESULT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("LOADING DETECTION RESULT")
    print("=" * 70)

    detection_result = load_detection_result()

    # --------------------------------------------------------
    # Verify that a spill was detected
    # --------------------------------------------------------

    if not detection_result.get(
        "spill_detected",
        False
    ):

        print(
            "\nNo oil spill detected by the AI model."
        )

        return

    # --------------------------------------------------------
    # Extract spill coordinates
    # --------------------------------------------------------

    centroid = detection_result.get(
        "centroid",
        {}
    )

    spill_lat = centroid.get(
        "latitude"
    )

    spill_lon = centroid.get(
        "longitude"
    )

    acquisition_date = detection_result.get(
        "acquisition_date"
    )

    if (
        spill_lat is None
        or spill_lon is None
    ):

        raise ValueError(
            "Detection result does not contain "
            "valid spill coordinates."
        )

    # --------------------------------------------------------
    # AIS time limitation
    # --------------------------------------------------------

    # The current detection JSON contains the acquisition DATE,
    # but not an exact acquisition TIME.
    #
    # Therefore, we do NOT invent a timestamp.
    #
    # For the current prototype, the AIS analysis time must be
    # supplied explicitly.

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
            "AIS time filtering therefore requires "
            "OCEANSHIELD_SPILL_TIME."
        )

        print(
            "\nExample:"
        )

        print(
            '$env:OCEANSHIELD_SPILL_TIME = '
            '"2018-09-26 10:30:00"'
        )

        print(
            "\nThe detected spill location will still be "
            "used automatically."
        )

        return

    # ========================================================
    # 2. SPILL INFORMATION
    # ========================================================

    max_distance_km = 10
    max_time_minutes = 60

    print("\nSPILL INFORMATION")
    print("-" * 70)

    print(
        f"Detection scene : "
        f"{detection_result.get('scene')}"
    )

    print(
        f"Acquisition date: "
        f"{acquisition_date}"
    )

    print(
        f"Latitude        : "
        f"{spill_lat:.6f}"
    )

    print(
        f"Longitude       : "
        f"{spill_lon:.6f}"
    )

    print(
        f"Spill time      : "
        f"{spill_time}"
    )

    print(
        f"Detected area   : "
        f"{detection_result['spill']['area_km2']:.2f} km²"
    )

    print(
        f"Mean confidence: "
        f"{detection_result['confidence']['mean_spill']:.3f}"
    )

    print(
        f"Distance limit  : "
        f"{max_distance_km} km"
    )

    print(
        f"Time window     : "
        f"±{max_time_minutes} minutes"
    )

    # ========================================================
    # STEP 1 - LOAD AIS DATA
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 1: LOADING AIS DATA")
    print("=" * 70)

    ais_data = load_ais_data(
        str(AIS_FILE_PATH)
    )

    print(
        f"Total AIS records loaded: "
        f"{len(ais_data)}"
    )

    # ========================================================
    # STEP 2 - DISTANCE FILTERING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 2: DISTANCE FILTERING")
    print("=" * 70)

    distance_candidates = filter_by_distance(
        ais_data,
        spill_lat,
        spill_lon,
        max_distance_km
    )

    print(
        f"Records within "
        f"{max_distance_km} km: "
        f"{len(distance_candidates)}"
    )

    # ========================================================
    # STEP 3 - TIME FILTERING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 3: TIME FILTERING")
    print("=" * 70)

    time_candidates = filter_by_time(
        distance_candidates,
        spill_time,
        max_time_minutes
    )

    print(
        f"Records within "
        f"±{max_time_minutes} minutes: "
        f"{len(time_candidates)}"
    )

    # ========================================================
    # STEP 4 - TRAJECTORY ANALYSIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 4: TRAJECTORY ANALYSIS")
    print("=" * 70)

    trajectory_results = analyze_trajectory(
        ais_data,
        spill_lat,
        spill_lon
    )

    print(
        f"Vessels analyzed: "
        f"{len(trajectory_results)}"
    )

    # ========================================================
    # STEP 5 - ATTRIBUTION SCORING
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
        max_distance_km,
        max_time_minutes
    )

    # ========================================================
    # STEP 6 - DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL VESSEL RANKING")
    print("=" * 70)

    if ranked_vessels.empty:

        print(
            "\nNo candidate vessels found."
        )

        return

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

    print(
        ranked_vessels[
            display_columns
        ].to_string(index=False)
    )

    # ========================================================
    # STEP 7 - TOP CANDIDATE
    # ========================================================

    best = ranked_vessels.iloc[0]

    print("\n")
    print("=" * 70)
    print("TOP CANDIDATE")
    print("=" * 70)

    print(
        f"MMSI                  : "
        f"{best['MMSI']}"
    )

    print(
        f"Ship type             : "
        f"{best['ship_type']}"
    )

    print(
        f"Closest distance      : "
        f"{best['distance_km']:.2f} km"
    )

    print(
        f"Time difference       : "
        f"{best['time_difference_minutes']:.1f} minutes"
    )

    print(
        f"Trajectory            : "
        f"{best['trajectory']}"
    )

    print(
        f"Average speed         : "
        f"{best['average_speed']:.2f} knots"
    )

    print(
        f"Attribution score     : "
        f"{best['attribution_likelihood']:.2f}%"
    )

    # ========================================================
    # STEP 8 - SAVE RESULTS
    # ========================================================

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIRECTORY
        / "ranked_candidates.csv"
    )

    ranked_vessels.to_csv(
        output_file,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("RESULT SAVED")
    print("=" * 70)

    print(
        f"Output file: "
        f"{output_file}"
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


if __name__ == "__main__":
    main()