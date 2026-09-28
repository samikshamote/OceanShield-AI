import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

from ais_loader import load_ais_data
from candidate_filter import calculate_distance


# ============================================================
# OCEANSHIELD-AI
# AIS VESSEL TRACK VISUALIZATION
# ============================================================

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AIS_FILE = (
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

PNG_OUTPUT = OUTPUT_DIRECTORY / "vessel_tracks.png"
GEOJSON_OUTPUT = OUTPUT_DIRECTORY / "vessel_tracks.geojson"

RANKING_FILE = OUTPUT_DIRECTORY / "ranked_candidates.csv"


# ------------------------------------------------------------
# DETECTION RESULT
# ------------------------------------------------------------

DETECTION_FILE = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "detection_result.json"
)


def load_spill_information():
    """
    Load spill location and acquisition date from the
    U-Net detection result.

    Spill time is a DEMO/REFERENCE time supplied through
    OCEANSHIELD_SPILL_TIME because the exact Sentinel-1
    acquisition time is not available in the detection JSON.
    """

    if not DETECTION_FILE.exists():
        raise FileNotFoundError(
            f"Detection result not found:\n{DETECTION_FILE}"
        )

    with open(
        DETECTION_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        detection = json.load(file)

    centroid = detection.get("centroid", {})

    spill_lat = centroid.get("latitude")
    spill_lon = centroid.get("longitude")

    if spill_lat is None or spill_lon is None:
        raise ValueError(
            "Spill centroid is missing from detection_result.json"
        )

    spill_time_text = os.getenv(
        "OCEANSHIELD_SPILL_TIME"
    )

    if not spill_time_text:
        raise ValueError(
            "\nOCEANSHIELD_SPILL_TIME is not set.\n\n"
            "Set it before running visualization:\n"
            '$env:OCEANSHIELD_SPILL_TIME = '
            '"2018-09-26 10:30:00"'
        )

    spill_time = pd.to_datetime(
        spill_time_text
    )

    return {
        "latitude": float(spill_lat),
        "longitude": float(spill_lon),
        "time": spill_time,
        "scene": detection.get("scene"),
        "acquisition_date": detection.get(
            "acquisition_date"
        )
    }


# ------------------------------------------------------------
# LOAD RANKING INFORMATION
# ------------------------------------------------------------

def load_ranking_data():
    """
    Load the existing ranked candidate results.

    This allows the GeoJSON to contain the same ranking and
    evidence information shown by the AIS attribution pipeline.
    """

    if not RANKING_FILE.exists():
        print(
            "\nWARNING: ranked_candidates.csv not found."
        )
        print(
            "GeoJSON will still be generated, but candidate "
            "ranking information may be unavailable."
        )
        return pd.DataFrame()

    try:
        ranking = pd.read_csv(
            RANKING_FILE
        )

        return ranking

    except Exception as error:
        print(
            f"\nWARNING: Could not load ranking file: {error}"
        )

        return pd.DataFrame()


# ------------------------------------------------------------
# PREPARE AIS DATA
# ------------------------------------------------------------

def prepare_data(
    spill_lat,
    spill_lon,
    spill_time,
    max_distance_km=10,
    max_time_minutes=60
):
    """
    Load AIS records and calculate distance/time from spill.
    """

    ais_data = load_ais_data(
        str(AIS_FILE)
    )

    # --------------------------------------------------------
    # Calculate distance from spill
    # --------------------------------------------------------

    ais_data["distance_km"] = ais_data.apply(
        lambda row: calculate_distance(
            row["latitude"],
            row["longitude"],
            spill_lat,
            spill_lon
        ),
        axis=1
    )

    # --------------------------------------------------------
    # Calculate time difference
    # --------------------------------------------------------

    ais_data["time_difference_minutes"] = (
        (
            ais_data["timestamp"]
            - spill_time
        )
        .abs()
        .dt.total_seconds()
        / 60
    )

    # --------------------------------------------------------
    # Candidate records
    # --------------------------------------------------------

    candidates = ais_data[
        (
            ais_data["distance_km"]
            <= max_distance_km
        )
        &
        (
            ais_data["time_difference_minutes"]
            <= max_time_minutes
        )
    ].copy()

    return ais_data, candidates


# ------------------------------------------------------------
# FIND CLOSEST POINT
# ------------------------------------------------------------

def find_closest_point(
    vessel_data
):
    """
    Find the AIS observation closest to the spill.
    """

    index = vessel_data[
        "distance_km"
    ].idxmin()

    return vessel_data.loc[index]


# ============================================================
# GEOJSON GENERATION
# ============================================================

def create_vessel_tracks_geojson(
    ais_data,
    candidates,
    ranking_data,
    spill_lat,
    spill_lon
):
    """
    Generate vessel_tracks.geojson.

    Each vessel track is represented as a LineString.

    Candidate vessel properties include:
        MMSI
        ship_type
        rank
        distance_km
        time_difference_minutes
        trajectory
        average_speed
        attribution_likelihood
        destination
    """

    features = []

    # --------------------------------------------------------
    # Candidate MMSIs
    # --------------------------------------------------------

    candidate_mmsi = set(
        candidates["MMSI"].astype(str).unique()
    )

    # --------------------------------------------------------
    # Create vessel track features
    # --------------------------------------------------------

    for mmsi, vessel_data in ais_data.groupby(
        "MMSI"
    ):

        vessel_data = vessel_data.sort_values(
            "timestamp"
        )

        # Need at least two points for a line
        if len(vessel_data) < 2:
            continue

        coordinates = [
            [
                float(row["longitude"]),
                float(row["latitude"])
            ]
            for _, row in vessel_data.iterrows()
        ]

        geometry = LineString(
            coordinates
        )

        mmsi_string = str(mmsi)

        # ----------------------------------------------------
        # Default properties
        # ----------------------------------------------------

        properties = {
            "MMSI": mmsi_string,
            "ship_type": str(
                vessel_data.iloc[0].get(
                    "ship_type",
                    "Unknown"
                )
            ),
            "destination": str(
                vessel_data.iloc[0].get(
                    "destination",
                    "Unknown"
                )
            ),
            "is_candidate": (
                mmsi_string in candidate_mmsi
            ),
            "rank": None,
            "distance_km": None,
            "time_difference_minutes": None,
            "trajectory": None,
            "average_speed": None,
            "attribution_likelihood": None
        }

        # ----------------------------------------------------
        # Add ranking information
        # ----------------------------------------------------

        if not ranking_data.empty:

            matching = ranking_data[
                ranking_data["MMSI"].astype(str)
                == mmsi_string
            ]

            if not matching.empty:

                row = matching.iloc[0]

                if "rank" in matching.columns:
                    properties["rank"] = int(
                        row["rank"]
                    )

                if "distance_km" in matching.columns:
                    properties["distance_km"] = round(
                        float(row["distance_km"]),
                        3
                    )

                if (
                    "time_difference_minutes"
                    in matching.columns
                ):
                    properties[
                        "time_difference_minutes"
                    ] = round(
                        float(
                            row[
                                "time_difference_minutes"
                            ]
                        ),
                        2
                    )

                if "trajectory" in matching.columns:
                    properties["trajectory"] = str(
                        row["trajectory"]
                    )

                if "average_speed" in matching.columns:
                    properties["average_speed"] = round(
                        float(
                            row["average_speed"]
                        ),
                        3
                    )

                if (
                    "attribution_likelihood"
                    in matching.columns
                ):
                    properties[
                        "attribution_likelihood"
                    ] = round(
                        float(
                            row[
                                "attribution_likelihood"
                            ]
                        ),
                        3
                    )

        # ----------------------------------------------------
        # Create GeoJSON feature
        # ----------------------------------------------------

        feature = {
            "type": "Feature",
            "geometry": geometry.__geo_interface__,
            "properties": properties
        }

        features.append(
            feature
        )

    # --------------------------------------------------------
    # Add spill point
    # --------------------------------------------------------

    spill_feature = {
        "type": "Feature",
        "geometry": Point(
            spill_lon,
            spill_lat
        ).__geo_interface__,
        "properties": {
            "feature_type": "spill",
            "label": "Detected Spill"
        }
    }

    features.append(
        spill_feature
    )

    # --------------------------------------------------------
    # Final FeatureCollection
    # --------------------------------------------------------

    geojson = {
        "type": "FeatureCollection",
        "name": "OceanShield-AI Vessel Tracks",
        "crs": {
            "type": "name",
            "properties": {
                "name": "EPSG:4326"
            }
        },
        "features": features
    }

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        GEOJSON_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            geojson,
            file,
            indent=2
        )

    print(
        "\nGeoJSON saved successfully:"
    )

    print(
        GEOJSON_OUTPUT
    )

    print(
        f"GeoJSON features: {len(features)}"
    )


# ============================================================
# PNG VISUALIZATION
# ============================================================

def plot_vessel_tracks(
    ais_data,
    candidates,
    ranking_data,
    spill_lat,
    spill_lon,
    spill_time,
    max_distance_km=10,
    max_time_minutes=60
):
    """
    Create the existing vessel_tracks.png visualization.
    """

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    candidate_mmsi = set(
        candidates["MMSI"].astype(str).unique()
    )

    # --------------------------------------------------------
    # Plot all vessel tracks
    # --------------------------------------------------------

    for mmsi, vessel_data in ais_data.groupby(
        "MMSI"
    ):

        vessel_data = vessel_data.sort_values(
            "timestamp"
        )

        mmsi_string = str(mmsi)

        is_candidate = (
            mmsi_string in candidate_mmsi
        )

        # ----------------------------------------------------
        # Determine ranking
        # ----------------------------------------------------

        rank_text = ""

        if not ranking_data.empty:

            matching = ranking_data[
                ranking_data["MMSI"].astype(str)
                == mmsi_string
            ]

            if not matching.empty:

                rank = matching.iloc[0].get(
                    "rank"
                )

                if pd.notna(rank):
                    rank_text = (
                        f"Rank {int(rank)}"
                    )

        # ----------------------------------------------------
        # Plot track
        # ----------------------------------------------------

        if is_candidate:

            ax.plot(
                vessel_data["longitude"],
                vessel_data["latitude"],
                marker="o",
                linewidth=2.5,
                markersize=5,
                label=(
                    f"Candidate {mmsi_string}"
                    + (
                        f" ({rank_text})"
                        if rank_text
                        else ""
                    )
                )
            )

        else:

            ax.plot(
                vessel_data["longitude"],
                vessel_data["latitude"],
                marker=".",
                linewidth=1,
                alpha=0.45,
                label=f"Vessel {mmsi_string}"
            )

        # ----------------------------------------------------
        # Mark closest point
        # ----------------------------------------------------

        closest = find_closest_point(
            vessel_data
        )

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
                (
                    f"{mmsi_string}\n"
                    f"{closest['distance_km']:.2f} km"
                ),
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
        spill_lon,
        spill_lat,
        s=300,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        zorder=10,
        label="Detected Spill"
    )

    ax.annotate(
        "SPILL",
        (
            spill_lon,
            spill_lat
        ),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Approximate 10 km visual zone
    # --------------------------------------------------------

    radius_lat = (
        max_distance_km / 111
    )

    circle = plt.Circle(
        (
            spill_lon,
            spill_lat
        ),
        radius_lat,
        fill=False,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"{max_distance_km} km candidate zone"
    )

    ax.add_patch(
        circle
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        "OceanShield-AI - AIS Vessel Attribution",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    ax.legend(
        loc="best",
        fontsize=8
    )

    # --------------------------------------------------------
    # Information box
    # --------------------------------------------------------

    ax.text(
        0.02,
        0.02,
        f"Reference Time: {spill_time}\n"
        f"Distance Filter: {max_distance_km} km\n"
        f"Time Filter: ±{max_time_minutes} min\n"
        f"AIS Data: Synthetic Demo Scenario",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            alpha=0.85
        )
    )

    plt.tight_layout()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        PNG_OUTPUT,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\nVisualization saved successfully:"
    )

    print(
        PNG_OUTPUT
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OCEANSHIELD-AI")
    print("AIS VESSEL TRACK VISUALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load spill information
    # --------------------------------------------------------

    spill = load_spill_information()

    spill_lat = spill["latitude"]
    spill_lon = spill["longitude"]
    spill_time = spill["time"]

    print(
        f"\nSpill latitude : {spill_lat}"
    )

    print(
        f"Spill longitude: {spill_lon}"
    )

    print(
        f"Reference time : {spill_time}"
    )

    print(
        "\nNOTE: Reference time is supplied for the "
        "synthetic AIS demonstration."
    )

    # --------------------------------------------------------
    # Filtering parameters
    # --------------------------------------------------------

    max_distance_km = 10
    max_time_minutes = 60

    # --------------------------------------------------------
    # Load AIS data
    # --------------------------------------------------------

    print(
        "\nLoading AIS data..."
    )

    ais_data, candidates = prepare_data(
        spill_lat,
        spill_lon,
        spill_time,
        max_distance_km,
        max_time_minutes
    )

    print(
        f"\nTotal AIS records : {len(ais_data)}"
    )

    print(
        f"Candidate records : {len(candidates)}"
    )

    print(
        f"Vessels tracked   : "
        f"{ais_data['MMSI'].nunique()}"
    )

    print(
        f"Candidate vessels : "
        f"{candidates['MMSI'].nunique()}"
    )

    # --------------------------------------------------------
    # Load ranking data
    # --------------------------------------------------------

    ranking_data = load_ranking_data()

    if not ranking_data.empty:

        print(
            "\nRanked candidate vessels:"
        )

        display_columns = [
            "rank",
            "MMSI",
            "ship_type",
            "distance_km",
            "time_difference_minutes",
            "trajectory",
            "attribution_likelihood"
        ]

        available_columns = [
            column
            for column in display_columns
            if column in ranking_data.columns
        ]

        print(
            ranking_data[
                available_columns
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Generate PNG
    # --------------------------------------------------------

    print(
        "\nGenerating vessel track visualization..."
    )

    plot_vessel_tracks(
        ais_data,
        candidates,
        ranking_data,
        spill_lat,
        spill_lon,
        spill_time,
        max_distance_km,
        max_time_minutes
    )

    # --------------------------------------------------------
    # Generate GeoJSON
    # --------------------------------------------------------

    print(
        "\nGenerating vessel track GeoJSON..."
    )

    create_vessel_tracks_geojson(
        ais_data,
        candidates,
        ranking_data,
        spill_lat,
        spill_lon
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print(
        "\nGenerated files:"
    )

    print(
        f"1. {PNG_OUTPUT}"
    )

    print(
        f"2. {GEOJSON_OUTPUT}"
    )

    print(
        "\nGeoJSON is ready for Person 4 / React / Leaflet integration."
    )

    print(
        "Note: AIS data in this demonstration is synthetic demo data."
    )


if __name__ == "__main__":
    main()