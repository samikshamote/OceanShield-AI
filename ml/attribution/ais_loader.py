import pandas as pd
import geopandas as gpd


def load_ais_data(file_path):
    """
    Load AIS data from a CSV file and convert it
    into a GeoDataFrame.
    """

    # Read CSV
    df = pd.read_csv(file_path)

    # Display available columns
    print("AIS Columns:")
    print(df.columns.tolist())

    # Required columns
    required_columns = [
        "MMSI",
        "timestamp",
        "latitude",
        "longitude",
        "SOG",
        "COG"
    ]

    # Check missing columns
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Remove invalid records
    df = df.dropna(
        subset=[
            "MMSI",
            "timestamp",
            "latitude",
            "longitude"
        ]
    )

    # Convert latitude and longitude to numbers
    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # Remove invalid coordinates
    df = df.dropna(
        subset=["latitude", "longitude"]
    )

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["longitude"],
            df["latitude"]
        ),
        crs="EPSG:4326"
    )

    # Sort by vessel and time
    gdf = gdf.sort_values(
        by=["MMSI", "timestamp"]
    )

    return gdf


if __name__ == "__main__":

    # Change this path when you have the AIS CSV
    file_path = "ml/attribution/data/ais_sample.csv"

    try:
        ais_data = load_ais_data(file_path)

        print("\nAIS data loaded successfully!")
        print(f"Number of records: {len(ais_data)}")

        print("\nFirst 5 records:")
        print(ais_data.head())

    except Exception as e:
        print(f"Error: {e}")