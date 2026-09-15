import matplotlib.pyplot as plt

from ais_loader import load_ais_data
from candidate_filter import calculate_distance


# ============================================================
# 1. Plot vessel trajectories
# ============================================================

def plot_vessel_tracks(ais_data, spill_lat, spill_lon):
    """
    Plot the movement tracks of all vessels
    and mark the detected spill location.
    """

    plt.figure(figsize=(10, 7))

    # Plot each vessel separately
    for mmsi, vessel in ais_data.groupby("MMSI"):

        vessel = vessel.sort_values("timestamp")

        plt.plot(
            vessel["longitude"],
            vessel["latitude"],
            marker="o",
            label=str(mmsi)
        )

    # Mark spill location
    plt.scatter(
        spill_lon,
        spill_lat,
        marker="*",
        s=200,
        label="Spill Location"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.title("AIS Vessel Trajectories and Spill Location")

    plt.legend(title="Vessel MMSI")

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# 2. Analyze vessel trajectory
# ============================================================

def analyze_trajectory(
    ais_data,
    spill_lat,
    spill_lon
):
    """
    Analyze whether each vessel is approaching
    or moving away from the spill.

    Returns one summary row per vessel.
    """

    results = []

    # Process each vessel separately
    for mmsi, vessel in ais_data.groupby("MMSI"):

        vessel = vessel.sort_values("timestamp").copy()

        # ----------------------------------------------------
        # Calculate distance of every point from spill
        # ----------------------------------------------------

        vessel["distance_km"] = vessel.apply(
            lambda row: calculate_distance(
                spill_lat,
                spill_lon,
                row["latitude"],
                row["longitude"]
            ),
            axis=1
        )

        # ----------------------------------------------------
        # Find closest point to spill
        # ----------------------------------------------------

        closest_index = vessel["distance_km"].idxmin()

        closest_point = vessel.loc[closest_index]

        # ----------------------------------------------------
        # First and last distance
        # ----------------------------------------------------

        first_distance = vessel.iloc[0]["distance_km"]

        last_distance = vessel.iloc[-1]["distance_km"]

        # ----------------------------------------------------
        # Determine trajectory direction
        # ----------------------------------------------------

        if last_distance < first_distance:
            direction = "Approaching"
        elif last_distance > first_distance:
            direction = "Moving Away"
        else:
            direction = "Stable"

        # ----------------------------------------------------
        # Check whether vessel approached and then moved away
        # ----------------------------------------------------

        closest_position = vessel["distance_km"].idxmin()

        first_position = vessel.index[0]

        last_position = vessel.index[-1]

        if (
            closest_position != first_position
            and closest_position != last_position
            and first_distance > closest_point["distance_km"]
            and last_distance > closest_point["distance_km"]
        ):
            trajectory_pattern = "Approached Then Moved Away"
        else:
            trajectory_pattern = direction

        # ----------------------------------------------------
        # Average speed
        # ----------------------------------------------------

        average_speed = vessel["SOG"].mean()

        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        average_heading = vessel["COG"].mean()

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({
            "MMSI": mmsi,
            "ship_type": vessel.iloc[0]["ship_type"],
            "first_distance_km": first_distance,
            "closest_distance_km": closest_point["distance_km"],
            "last_distance_km": last_distance,
            "closest_timestamp": closest_point["timestamp"],
            "average_speed": average_speed,
            "average_heading": average_heading,
            "trajectory": trajectory_pattern
        })

    return results


# ============================================================
# 3. Main program
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load AIS data
    # --------------------------------------------------------

    file_path = "ml/attribution/data/ais_sample.csv"

    ais_data = load_ais_data(file_path)

    print("\n")
    print("=" * 70)
    print("AIS VESSEL TRAJECTORY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Spill information
    # --------------------------------------------------------

    spill_lat = 18.5200
    spill_lon = 72.9100

    spill_time = "2026-09-15 10:30:00"

    print("\nSpill Location")
    print("-" * 70)

    print(f"Latitude  : {spill_lat}")
    print(f"Longitude : {spill_lon}")
    print(f"Time      : {spill_time}")

    # --------------------------------------------------------
    # Analyze trajectories
    # --------------------------------------------------------

    trajectory_results = analyze_trajectory(
        ais_data,
        spill_lat,
        spill_lon
    )

    # Convert results to DataFrame
    import pandas as pd

    trajectory_df = pd.DataFrame(
        trajectory_results
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("VESSEL TRAJECTORY RESULTS")
    print("=" * 70)

    print(
        trajectory_df[
            [
                "MMSI",
                "ship_type",
                "first_distance_km",
                "closest_distance_km",
                "last_distance_km",
                "closest_timestamp",
                "average_speed",
                "trajectory"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Plot trajectories
    # --------------------------------------------------------

    plot_vessel_tracks(
        ais_data,
        spill_lat,
        spill_lon
    )