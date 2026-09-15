import os
import matplotlib.pyplot as plt
import pandas as pd

from ais_loader import load_ais_data
from candidate_filter import calculate_distance


# ============================================================
# OCEANSHIELD-AI
# AIS VESSEL TRACK VISUALIZATION
# ============================================================

# Spill information
SPILL_LAT = 18.5200
SPILL_LON = 72.9100
SPILL_TIME = pd.Timestamp("2026-09-15 10:30:00")

# Filtering limits
MAX_DISTANCE_KM = 10
MAX_TIME_MINUTES = 60

# Input and output paths
AIS_FILE = "ml/attribution/data/ais_sample.csv"
OUTPUT_FILE = "ml/attribution/outputs/vessel_tracks.png"


def prepare_data():
    """Load AIS data and calculate distance/time from spill."""

    ais_data = load_ais_data(AIS_FILE)

    # Calculate distance of every AIS point from spill
    ais_data["distance_km"] = ais_data.apply(
        lambda row: calculate_distance(
            row["latitude"],
            row["longitude"],
            SPILL_LAT,
            SPILL_LON
        ),
        axis=1
    )

    # Calculate time difference
    ais_data["time_difference_minutes"] = (
        (ais_data["timestamp"] - SPILL_TIME)
        .abs()
        .dt.total_seconds()
        / 60
    )

    # Candidate records
    candidates = ais_data[
        (ais_data["distance_km"] <= MAX_DISTANCE_KM)
        &
        (ais_data["time_difference_minutes"] <= MAX_TIME_MINUTES)
    ].copy()

    return ais_data, candidates


def find_closest_point(vessel_data):
    """Find the AIS point closest to the spill."""

    index = vessel_data["distance_km"].idxmin()

    return vessel_data.loc[index]


def plot_vessel_tracks(ais_data, candidates):
    """Create AIS vessel track visualization."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Unique candidate vessels
    candidate_mmsi = set(candidates["MMSI"].unique())

    # --------------------------------------------------------
    # Plot all vessel tracks
    # --------------------------------------------------------

    for mmsi, vessel_data in ais_data.groupby("MMSI"):

        vessel_data = vessel_data.sort_values("timestamp")

        is_candidate = mmsi in candidate_mmsi

        # Candidate vessels are highlighted
        if is_candidate:
            ax.plot(
                vessel_data["longitude"],
                vessel_data["latitude"],
                marker="o",
                linewidth=2.5,
                markersize=5,
                label=f"Candidate MMSI {mmsi}"
            )
        else:
            ax.plot(
                vessel_data["longitude"],
                vessel_data["latitude"],
                marker=".",
                linewidth=1,
                alpha=0.45,
                label=f"Vessel {mmsi}"
            )

        # ----------------------------------------------------
        # Mark closest point for every vessel
        # ----------------------------------------------------

        closest = find_closest_point(vessel_data)

        if is_candidate:
            ax.scatter(
                closest["longitude"],
                closest["latitude"],
                s=120,
                marker="X",
                edgecolors="black",
                linewidths=1.2,
                zorder=5
            )

            ax.annotate(
                f"{mmsi}\n{closest['distance_km']:.2f} km",
                (
                    closest["longitude"],
                    closest["latitude"]
                ),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold"
            )

    # --------------------------------------------------------
    # Spill location
    # --------------------------------------------------------

    ax.scatter(
        SPILL_LON,
        SPILL_LAT,
        s=300,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        zorder=10,
        label="Detected Spill"
    )

    ax.annotate(
        "SPILL",
        (SPILL_LON, SPILL_LAT),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Draw approximate 10 km candidate zone
    # --------------------------------------------------------
    #
    # 1 degree latitude ≈ 111 km
    # This is only a visual approximation.
    #

    radius_lat = MAX_DISTANCE_KM / 111

    circle = plt.Circle(
        (SPILL_LON, SPILL_LAT),
        radius_lat,
        fill=False,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="10 km candidate zone"
    )

    ax.add_patch(circle)

    # --------------------------------------------------------
    # Labels and formatting
    # --------------------------------------------------------

    ax.set_title(
        "OceanShield-AI - AIS Vessel Attribution",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend(
        loc="best",
        fontsize=8
    )

    ax.text(
        0.02,
        0.02,
        f"Spill Time: {SPILL_TIME}\n"
        f"Distance Filter: {MAX_DISTANCE_KM} km\n"
        f"Time Filter: ±{MAX_TIME_MINUTES} min",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            alpha=0.85
        )
    )

    plt.tight_layout()

    # Create output directory
    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    # Save image
    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    print("\nVISUALIZATION SAVED")
    print(f"Output file: {OUTPUT_FILE}")

    plt.show()


def main():

    print("=" * 65)
    print("OCEANSHIELD-AI")
    print("AIS VESSEL TRACK VISUALIZATION")
    print("=" * 65)

    print("\nLoading AIS data...")

    ais_data, candidates = prepare_data()

    print(f"Total AIS records: {len(ais_data)}")
    print(f"Candidate records: {len(candidates)}")
    print(f"Candidate vessels: {candidates['MMSI'].nunique()}")

    print("\nCandidate vessels:")

    if len(candidates) > 0:
        print(
            candidates[
                [
                    "MMSI",
                    "ship_type",
                    "distance_km",
                    "time_difference_minutes"
                ]
            ]
            .sort_values("distance_km")
            .to_string(index=False)
        )
    else:
        print("No candidate vessels found.")

    print("\nGenerating vessel track visualization...")

    plot_vessel_tracks(
        ais_data,
        candidates
    )


if __name__ == "__main__":
    main()