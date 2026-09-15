import os

from ais_loader import load_ais_data
from candidate_filter import (
    filter_by_distance,
    filter_by_time
)
from trajectory import analyze_trajectory
from scoring import generate_scores


# ============================================================
# AIS VESSEL ATTRIBUTION - MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("OCEANSHIELD-AI")
    print("AIS VESSEL ATTRIBUTION PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. File path
    # --------------------------------------------------------

    file_path = "ml/attribution/data/ais_sample.csv"

    # --------------------------------------------------------
    # 2. Spill information
    # --------------------------------------------------------

    spill_lat = 18.5200
    spill_lon = 72.9100
    spill_time = "2026-09-15 10:30:00"

    # --------------------------------------------------------
    # 3. Filtering parameters
    # --------------------------------------------------------

    max_distance_km = 10
    max_time_minutes = 60

    print("\nSPILL INFORMATION")
    print("-" * 70)

    print(f"Latitude       : {spill_lat}")
    print(f"Longitude      : {spill_lon}")
    print(f"Spill time     : {spill_time}")
    print(f"Distance limit : {max_distance_km} km")
    print(f"Time window    : ±{max_time_minutes} minutes")

    # ========================================================
    # STEP 1 - LOAD AIS DATA
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 1: LOADING AIS DATA")
    print("=" * 70)

    ais_data = load_ais_data(file_path)

    print(
        f"Total AIS records loaded: {len(ais_data)}"
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
        f"Records within {max_distance_km} km: "
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
        f"Records within ±{max_time_minutes} minutes: "
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

        print("\nNo candidate vessels found.")

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
        f"MMSI                  : {best['MMSI']}"
    )

    print(
        f"Ship type             : {best['ship_type']}"
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

    output_directory = "ml/attribution/outputs"

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    output_file = (
        f"{output_directory}/ranked_candidates.csv"
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
        f"Output file: {output_file}"
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