import math
import pandas as pd

from ais_loader import load_ais_data
from candidate_filter import (
    filter_by_distance,
    filter_by_time,
    calculate_distance
)
from trajectory import analyze_trajectory


# ============================================================
# 1. DISTANCE SCORE
# ============================================================

def calculate_distance_score(
    distance_km,
    max_distance_km=10
):
    """
    Closer vessel = higher score.
    """

    if pd.isna(distance_km):
        return 0.0

    score = 1 - (
        distance_km / max_distance_km
    )

    return max(0.0, min(1.0, score))


# ============================================================
# 2. TIME SCORE
# ============================================================

def calculate_time_score(
    time_difference_minutes,
    max_time_minutes=60
):
    """
    Smaller time difference = higher score.
    """

    if pd.isna(time_difference_minutes):
        return 0.0

    score = 1 - (
        time_difference_minutes / max_time_minutes
    )

    return max(0.0, min(1.0, score))


# ============================================================
# 3. TRAJECTORY SCORE
# ============================================================

def calculate_trajectory_score(trajectory):
    """
    Score vessel movement behavior.
    """

    if trajectory == "Approached Then Moved Away":
        return 1.0

    elif trajectory == "Approaching":
        return 0.7

    elif trajectory == "Moving Away":
        return 0.4

    elif trajectory == "Stable":
        return 0.2

    return 0.0


# ============================================================
# 4. BEARING FROM VESSEL TO SPILL
# ============================================================

def calculate_bearing(
    vessel_lat,
    vessel_lon,
    spill_lat,
    spill_lon
):
    """
    Calculate the bearing from vessel position
    toward the spill location.

    0   = North
    90  = East
    180 = South
    270 = West
    """

    lat1 = math.radians(vessel_lat)
    lat2 = math.radians(spill_lat)

    delta_lon = math.radians(
        spill_lon - vessel_lon
    )

    x = (
        math.sin(delta_lon)
        * math.cos(lat2)
    )

    y = (
        math.cos(lat1)
        * math.sin(lat2)
        -
        math.sin(lat1)
        * math.cos(lat2)
        * math.cos(delta_lon)
    )

    bearing = math.degrees(
        math.atan2(x, y)
    )

    return (bearing + 360) % 360


# ============================================================
# 5. ANGLE DIFFERENCE
# ============================================================

def angle_difference(
    angle1,
    angle2
):
    """
    Calculate smallest difference between
    two compass directions.
    """

    difference = abs(
        angle1 - angle2
    )

    if difference > 180:
        difference = 360 - difference

    return difference


# ============================================================
# 6. HEADING ALIGNMENT
# ============================================================

def calculate_heading_alignment(
    vessel_lat,
    vessel_lon,
    spill_lat,
    spill_lon,
    cog
):
    """
    Compare vessel COG with the direction
    from vessel to spill.

    Score:

    0 degree difference   = 1.0
    45 degree difference  = 0.75
    90 degree difference  = 0.50
    180 degree difference = 0.0
    """

    if pd.isna(cog):
        return None

    target_bearing = calculate_bearing(
        vessel_lat,
        vessel_lon,
        spill_lat,
        spill_lon
    )

    difference = angle_difference(
        cog,
        target_bearing
    )

    score = 1 - (
        difference / 180
    )

    return max(
        0.0,
        min(1.0, score)
    )


# ============================================================
# 7. VESSEL HEADING SCORE
# ============================================================

def calculate_vessel_heading_score(
    vessel_data,
    spill_lat,
    spill_lon
):
    """
    Calculate heading alignment using multiple
    AIS observations.

    Only observations where the vessel is not
    extremely close to the spill are considered,
    because heading becomes unstable at the exact
    closest point.
    """

    vessel_data = vessel_data.copy()

    vessel_data = vessel_data.sort_values(
        "timestamp"
    )

    heading_scores = []

    for _, row in vessel_data.iterrows():

        distance = calculate_distance(
            row["latitude"],
            row["longitude"],
            spill_lat,
            spill_lon
        )

        # Ignore points extremely close to spill.
        # Heading at the exact spill location is
        # not useful for determining approach direction.
        if distance < 0.5:
            continue

        score = calculate_heading_alignment(
            row["latitude"],
            row["longitude"],
            spill_lat,
            spill_lon,
            row["COG"]
        )

        if score is not None:
            heading_scores.append(score)

    if not heading_scores:
        return 0.0

    return sum(heading_scores) / len(
        heading_scores
    )


# ============================================================
# 8. APPROACH BEHAVIOR SCORE
# ============================================================

def calculate_approach_behavior_score(
    vessel_data,
    spill_lat,
    spill_lon
):
    """
    Analyze whether the vessel's distance from
    the spill decreases over time.

    Returns a score between 0 and 1.
    """

    vessel_data = vessel_data.sort_values(
        "timestamp"
    ).copy()

    if len(vessel_data) < 2:
        return 0.0

    distances = []

    for _, row in vessel_data.iterrows():

        distance = calculate_distance(
            row["latitude"],
            row["longitude"],
            spill_lat,
            spill_lon
        )

        distances.append(distance)

    # --------------------------------------------------------
    # Count distance decreases
    # --------------------------------------------------------

    decreasing_steps = 0
    total_steps = len(distances) - 1

    for i in range(1, len(distances)):

        if distances[i] < distances[i - 1]:
            decreasing_steps += 1

    if total_steps == 0:
        return 0.0

    approach_ratio = (
        decreasing_steps / total_steps
    )

    # --------------------------------------------------------
    # Find minimum distance
    # --------------------------------------------------------

    minimum_distance = min(distances)

    minimum_index = distances.index(
        minimum_distance
    )

    # --------------------------------------------------------
    # Check whether vessel approached and
    # subsequently moved away.
    # --------------------------------------------------------

    moved_away_after_closest = False

    if (
        minimum_index > 0
        and minimum_index < len(distances) - 1
    ):

        before = distances[
            minimum_index - 1
        ]

        after = distances[
            minimum_index + 1
        ]

        if (
            before > minimum_distance
            and after > minimum_distance
        ):
            moved_away_after_closest = True

    # --------------------------------------------------------
    # Base score
    # --------------------------------------------------------

    score = approach_ratio

    # Strong evidence when vessel approaches,
    # reaches closest point and moves away.
    if moved_away_after_closest:
        score = min(
            1.0,
            score + 0.25
        )

    return max(
        0.0,
        min(1.0, score)
    )


# ============================================================
# 9. APPROACH RATE
# ============================================================

def calculate_approach_rate(
    vessel_data,
    spill_lat,
    spill_lon
):
    """
    Calculate approach rate using the distance
    before the closest approach.

    Positive value = approaching.
    """

    vessel_data = vessel_data.sort_values(
        "timestamp"
    ).copy()

    if len(vessel_data) < 2:
        return 0.0

    distances = []

    for _, row in vessel_data.iterrows():

        distance = calculate_distance(
            row["latitude"],
            row["longitude"],
            spill_lat,
            spill_lon
        )

        distances.append(distance)

    closest_index = distances.index(
        min(distances)
    )

    if closest_index == 0:
        return 0.0

    previous_index = closest_index - 1

    previous_distance = distances[
        previous_index
    ]

    closest_distance = distances[
        closest_index
    ]

    previous_time = vessel_data.iloc[
        previous_index
    ]["timestamp"]

    closest_time = vessel_data.iloc[
        closest_index
    ]["timestamp"]

    time_minutes = (
        closest_time - previous_time
    ).total_seconds() / 60

    if time_minutes <= 0:
        return 0.0

    approach_rate = (
        previous_distance
        - closest_distance
    ) / time_minutes

    return max(
        0.0,
        approach_rate
    )


# ============================================================
# 10. APPROACH RATE SCORE
# ============================================================

def calculate_approach_rate_score(
    approach_rate,
    reference_rate=0.2
):
    """
    Convert approach rate to a normalized score.

    This is a baseline heuristic.
    """

    if approach_rate <= 0:
        return 0.0

    score = (
        approach_rate / reference_rate
    )

    return max(
        0.0,
        min(1.0, score)
    )


# ============================================================
# 11. GENERATE FINAL SCORES
# ============================================================

def generate_scores(
    candidates,
    trajectory_data,
    ais_data,
    spill_lat,
    spill_lon,
    max_distance_km=10,
    max_time_minutes=60
):
    """
    Generate ONE attribution score per vessel.

    Weights:

    Distance          = 30%
    Time              = 20%
    Approach behavior = 20%
    Heading alignment = 15%
    Trajectory        = 15%
    """

    if candidates.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Best candidate observation per vessel
    # --------------------------------------------------------

    best_records = (
        candidates
        .sort_values(
            by=[
                "distance_km",
                "time_difference_minutes"
            ]
        )
        .groupby("MMSI")
        .first()
        .reset_index()
    )

    # --------------------------------------------------------
    # Trajectory data
    # --------------------------------------------------------

    trajectory_df = pd.DataFrame(
        trajectory_data
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    scored = best_records.merge(
        trajectory_df[
            [
                "MMSI",
                "trajectory",
                "average_speed"
            ]
        ],
        on="MMSI",
        how="left"
    )

    # ========================================================
    # DISTANCE SCORE
    # ========================================================

    scored["distance_score"] = (
        scored["distance_km"]
        .apply(
            lambda x:
            calculate_distance_score(
                x,
                max_distance_km
            )
        )
    )

    # ========================================================
    # TIME SCORE
    # ========================================================

    scored["time_score"] = (
        scored["time_difference_minutes"]
        .apply(
            lambda x:
            calculate_time_score(
                x,
                max_time_minutes
            )
        )
    )

    # ========================================================
    # TRAJECTORY SCORE
    # ========================================================

    scored["trajectory_score"] = (
        scored["trajectory"]
        .apply(
            calculate_trajectory_score
        )
    )

    # ========================================================
    # CALCULATE AIS BEHAVIOR FEATURES
    # ========================================================

    approach_rates = {}
    approach_scores = {}
    heading_scores = {}

    for mmsi, vessel in ais_data.groupby("MMSI"):

        # Approach rate
        rate = calculate_approach_rate(
            vessel,
            spill_lat,
            spill_lon
        )

        approach_rates[mmsi] = rate

        # Approach behavior
        behavior_score = (
            calculate_approach_behavior_score(
                vessel,
                spill_lat,
                spill_lon
            )
        )

        approach_scores[mmsi] = (
            behavior_score
        )

        # Heading alignment
        heading_score = (
            calculate_vessel_heading_score(
                vessel,
                spill_lat,
                spill_lon
            )
        )

        heading_scores[mmsi] = (
            heading_score
        )

    # --------------------------------------------------------
    # Add features
    # --------------------------------------------------------

    scored["approach_rate_km_per_min"] = (
        scored["MMSI"].map(
            approach_rates
        )
    )

    scored["approach_score"] = (
        scored["MMSI"].map(
            approach_scores
        )
    )

    scored["heading_score"] = (
        scored["MMSI"].map(
            heading_scores
        )
    )

    # ========================================================
    # FINAL WEIGHTED SCORE
    # ========================================================

    scored["attribution_score"] = (
        0.30 * scored["distance_score"]
        +
        0.20 * scored["time_score"]
        +
        0.20 * scored["approach_score"]
        +
        0.15 * scored["heading_score"]
        +
        0.15 * scored["trajectory_score"]
    )

    # --------------------------------------------------------
    # Convert to percentage-style score
    # --------------------------------------------------------

    scored["attribution_likelihood"] = (
        scored["attribution_score"] * 100
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    scored = scored.sort_values(
        by="attribution_score",
        ascending=False
    ).reset_index(drop=True)

    scored["rank"] = (
        scored.index + 1
    )

    return scored


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AIS VESSEL ATTRIBUTION SCORING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load AIS
    # --------------------------------------------------------

    file_path = (
        "ml/attribution/data/ais_sample.csv"
    )

    ais_data = load_ais_data(
        file_path
    )

    print(
        f"\nTotal AIS records: "
        f"{len(ais_data)}"
    )

    # --------------------------------------------------------
    # Spill information
    # --------------------------------------------------------

    spill_lat = 18.5200
    spill_lon = 72.9100

    spill_time = (
        "2026-09-15 10:30:00"
    )

    max_distance_km = 10
    max_time_minutes = 60

    # --------------------------------------------------------
    # Distance filter
    # --------------------------------------------------------

    distance_candidates = (
        filter_by_distance(
            ais_data,
            spill_lat,
            spill_lon,
            max_distance_km
        )
    )

    print(
        f"\nRecords after distance filtering: "
        f"{len(distance_candidates)}"
    )

    # --------------------------------------------------------
    # Time filter
    # --------------------------------------------------------

    time_candidates = (
        filter_by_time(
            distance_candidates,
            spill_time,
            max_time_minutes
        )
    )

    print(
        f"Records after time filtering: "
        f"{len(time_candidates)}"
    )

    # --------------------------------------------------------
    # Trajectory
    # --------------------------------------------------------

    trajectory_results = (
        analyze_trajectory(
            ais_data,
            spill_lat,
            spill_lon
        )
    )

    print(
        f"Vessels analyzed: "
        f"{len(trajectory_results)}"
    )

    # --------------------------------------------------------
    # Generate scores
    # --------------------------------------------------------

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
    # DISPLAY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL RANKED VESSELS")
    print("=" * 70)

    if ranked_vessels.empty:

        print("\nNo candidate vessels found.")

    else:

        display_columns = [
            "rank",
            "MMSI",
            "ship_type",
            "distance_km",
            "time_difference_minutes",
            "approach_rate_km_per_min",
            "approach_score",
            "heading_score",
            "trajectory",
            "attribution_likelihood"
        ]

        print(
            ranked_vessels[
                display_columns
            ].to_string(index=False)
        )

        # ----------------------------------------------------
        # Top candidate
        # ----------------------------------------------------

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
            f"Approach rate         : "
            f"{best['approach_rate_km_per_min']:.4f} km/min"
        )

        print(
            f"Approach score        : "
            f"{best['approach_score']:.3f}"
        )

        print(
            f"Heading score         : "
            f"{best['heading_score']:.3f}"
        )

        print(
            f"Trajectory            : "
            f"{best['trajectory']}"
        )

        print(
            f"Attribution score     : "
            f"{best['attribution_likelihood']:.2f}%"
        )

        print("\nNote:")
        print(
            "This is an evidence-based AIS attribution "
            "score, not proof of causation."
        )