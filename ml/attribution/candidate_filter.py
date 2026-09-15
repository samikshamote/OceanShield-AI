import math
import pandas as pd

from ais_loader import load_ais_data


# ============================================================
# 1. Calculate distance between two coordinates
# ============================================================

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two latitude/longitude
    points using the Haversine formula.

    Returns:
        Distance in kilometers
    """

    # Earth's radius in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Difference between coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    distance = R * c

    return distance


# ============================================================
# 2. Filter AIS records by distance from spill
# ============================================================

def filter_by_distance(
    ais_data,
    spill_lat,
    spill_lon,
    max_distance_km=10
):
    """
    Keep AIS records that are within the specified
    distance from the detected spill location.
    """

    ais_data = ais_data.copy()

    # Calculate distance for every AIS record
    ais_data["distance_km"] = ais_data.apply(
        lambda row: calculate_distance(
            spill_lat,
            spill_lon,
            row["latitude"],
            row["longitude"]
        ),
        axis=1
    )

    # Keep records within allowed distance
    filtered = ais_data[
        ais_data["distance_km"] <= max_distance_km
    ].copy()

    return filtered


# ============================================================
# 3. Filter AIS records by time
# ============================================================

def filter_by_time(
    ais_data,
    spill_time,
    time_window_minutes=60
):
    """
    Keep AIS records that occurred within the specified
    time window around the spill detection time.
    """

    ais_data = ais_data.copy()

    # Convert spill time to datetime
    spill_time = pd.to_datetime(spill_time)

    # Calculate absolute time difference
    time_difference = (
        ais_data["timestamp"] - spill_time
    ).abs()

    # Store time difference in minutes
    ais_data["time_difference_minutes"] = (
        time_difference.dt.total_seconds() / 60
    )

    # Keep records within allowed time window
    filtered = ais_data[
        ais_data["time_difference_minutes"]
        <= time_window_minutes
    ].copy()

    return filtered


# ============================================================
# 4. Rank candidate vessels
# ============================================================

def rank_candidates(
    candidates,
    max_distance_km=10,
    max_time_minutes=60
):
    """
    Rank candidate vessels using distance and time.

    Distance weight = 60%
    Time weight     = 40%

    Higher score = stronger attribution likelihood.
    """

    candidates = candidates.copy()

    # --------------------------------------------------------
    # Distance score
    # --------------------------------------------------------
    candidates["distance_score"] = (
        1
        - (
            candidates["distance_km"]
            / max_distance_km
        )
    )

    # --------------------------------------------------------
    # Time score
    # --------------------------------------------------------
    candidates["time_score"] = (
        1
        - (
            candidates["time_difference_minutes"]
            / max_time_minutes
        )
    )

    # Make sure scores remain between 0 and 1
    candidates["distance_score"] = (
        candidates["distance_score"]
        .clip(0, 1)
    )

    candidates["time_score"] = (
        candidates["time_score"]
        .clip(0, 1)
    )

    # --------------------------------------------------------
    # Final attribution score
    # --------------------------------------------------------
    candidates["attribution_score"] = (
        0.6 * candidates["distance_score"]
        +
        0.4 * candidates["time_score"]
    )

    # --------------------------------------------------------
    # Convert score to percentage
    # --------------------------------------------------------
    candidates["attribution_likelihood"] = (
        candidates["attribution_score"] * 100
    )

    # --------------------------------------------------------
    # Rank highest score first
    # --------------------------------------------------------
    candidates = candidates.sort_values(
        by="attribution_score",
        ascending=False
    )

    # Add ranking number
    candidates["rank"] = range(
        1,
        len(candidates) + 1
    )

    return candidates


# ============================================================
# 5. Main program
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load AIS data
    # --------------------------------------------------------

    file_path = "ml/attribution/data/ais_sample.csv"

    print("=" * 60)
    print("AIS VESSEL CANDIDATE FILTER")
    print("=" * 60)

    try:

        ais_data = load_ais_data(file_path)

        print(
            f"\nTotal AIS records: {len(ais_data)}"
        )

        # ----------------------------------------------------
        # Detected spill information
        # ----------------------------------------------------

        spill_lat = 18.5200
        spill_lon = 72.9100

        spill_time = "2026-09-15 10:30:00"

        # ----------------------------------------------------
        # Filtering parameters
        # ----------------------------------------------------

        max_distance_km = 10
        time_window_minutes = 60

        print("\nSpill Information")
        print("-" * 60)

        print(f"Spill latitude  : {spill_lat}")
        print(f"Spill longitude : {spill_lon}")
        print(f"Spill time      : {spill_time}")

        print(
            f"Distance limit  : {max_distance_km} km"
        )

        print(
            f"Time window     : ±{time_window_minutes} minutes"
        )

        # ----------------------------------------------------
        # STEP 1: Distance filtering
        # ----------------------------------------------------

        distance_candidates = filter_by_distance(
            ais_data,
            spill_lat,
            spill_lon,
            max_distance_km
        )

        print("\n")
        print("=" * 60)
        print("AFTER DISTANCE FILTERING")
        print("=" * 60)

        print(
            f"Records remaining: "
            f"{len(distance_candidates)}"
        )

        if len(distance_candidates) > 0:

            print(
                distance_candidates[
                    [
                        "MMSI",
                        "timestamp",
                        "latitude",
                        "longitude",
                        "distance_km"
                    ]
                ].to_string(index=False)
            )

        else:

            print("No vessels found within distance limit.")

        # ----------------------------------------------------
        # STEP 2: Time filtering
        # ----------------------------------------------------

        time_candidates = filter_by_time(
            distance_candidates,
            spill_time,
            time_window_minutes
        )

        print("\n")
        print("=" * 60)
        print("AFTER TIME FILTERING")
        print("=" * 60)

        print(
            f"Records remaining: "
            f"{len(time_candidates)}"
        )

        if len(time_candidates) > 0:

            print(
                time_candidates[
                    [
                        "MMSI",
                        "timestamp",
                        "distance_km",
                        "time_difference_minutes"
                    ]
                ].to_string(index=False)
            )

        else:

            print("No vessels found within time window.")

        # ----------------------------------------------------
        # STEP 3: Candidate ranking
        # ----------------------------------------------------

        if len(time_candidates) > 0:

            ranked_candidates = rank_candidates(
                time_candidates,
                max_distance_km,
                time_window_minutes
            )

            print("\n")
            print("=" * 60)
            print("RANKED CANDIDATE VESSELS")
            print("=" * 60)

            print(
                ranked_candidates[
                    [
                        "rank",
                        "MMSI",
                        "ship_type",
                        "distance_km",
                        "time_difference_minutes",
                        "distance_score",
                        "time_score",
                        "attribution_likelihood"
                    ]
                ].to_string(index=False)
            )

            # ------------------------------------------------
            # Best candidate
            # ------------------------------------------------

            best = ranked_candidates.iloc[0]

            print("\n")
            print("=" * 60)
            print("TOP CANDIDATE")
            print("=" * 60)

            print(f"MMSI                 : {best['MMSI']}")
            print(f"Ship type            : {best['ship_type']}")
            print(
                f"Distance from spill  : "
                f"{best['distance_km']:.2f} km"
            )
            print(
                f"Time difference      : "
                f"{best['time_difference_minutes']:.1f} minutes"
            )
            print(
                f"Attribution likelihood: "
                f"{best['attribution_likelihood']:.2f}%"
            )

            print("\nNote:")
            print(
                "This score represents attribution likelihood "
                "based on proximity and timing."
            )
            print(
                "It does NOT prove that the vessel caused the spill."
            )

        else:

            print("\nNo candidate vessels found.")

    except Exception as e:

        print("\nERROR:")
        print(e)