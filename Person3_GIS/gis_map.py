import json
import math
from pathlib import Path

import folium
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from shapely.ops import unary_union


# ============================================================
# OCEANSHIELD-AI GIS MODULE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DETECTION_RESULT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "detection_result.json"
)

SPILL_MASK_PATH = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "spill_mask.tif"
)

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

MAP_OUTPUT = OUTPUT_DIRECTORY / "spill_map.html"
GEOJSON_OUTPUT = OUTPUT_DIRECTORY / "spill_footprint.geojson"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("OCEANSHIELD-AI")
print("GIS SPILL IMPACT & MOVEMENT MODULE")
print("=" * 70)


# ============================================================
# CHECK INPUTS
# ============================================================

if not DETECTION_RESULT_PATH.exists():
    raise FileNotFoundError(
        f"Detection result not found:\n{DETECTION_RESULT_PATH}"
    )

if not SPILL_MASK_PATH.exists():
    raise FileNotFoundError(
        f"Spill mask not found:\n{SPILL_MASK_PATH}"
    )


# ============================================================
# LOAD DETECTION RESULT
# ============================================================

print("\n")
print("=" * 70)
print("LOADING DETECTION RESULT")
print("=" * 70)

with open(DETECTION_RESULT_PATH, "r", encoding="utf-8") as file:
    detection = json.load(file)

if not detection.get("spill_detected", False):
    print("\nNo oil spill detected.")
    raise SystemExit(0)


spill = detection["spill"]
confidence_data = detection["confidence"]
centroid = detection["centroid"]
bbox = detection.get("bounding_box", {})

latitude = centroid["latitude"]
longitude = centroid["longitude"]

spill_area_km2 = spill["area_km2"]
spill_percentage = spill["percentage"]

max_confidence = confidence_data["maximum"]
mean_confidence = confidence_data["mean_spill"]

acquisition_date = detection.get(
    "acquisition_date",
    "Unknown"
)

print("\nDETECTED SPILL")
print("-" * 70)
print(f"Scene             : {detection.get('scene', 'Unknown')}")
print(f"Acquisition date  : {acquisition_date}")
print(f"Latitude          : {latitude:.6f}")
print(f"Longitude         : {longitude:.6f}")
print(f"Spill area        : {spill_area_km2:.2f} km²")
print(f"Spill percentage  : {spill_percentage:.2f}%")
print(f"Maximum confidence: {max_confidence:.3f}")
print(f"Mean confidence   : {mean_confidence:.3f}")


# ============================================================
# RISK ASSESSMENT
# ============================================================

if spill_area_km2 >= 50 and mean_confidence >= 0.70:
    risk_level = "HIGH"

elif spill_area_km2 >= 20 and mean_confidence >= 0.60:
    risk_level = "MEDIUM"

else:
    risk_level = "LOW"


if risk_level == "HIGH":
    risk_color = "red"

elif risk_level == "MEDIUM":
    risk_color = "orange"

else:
    risk_color = "green"


# ============================================================
# ENVIRONMENTAL INPUT
# ============================================================

# Prototype input.
# This will later be replaced by real weather/ocean data.

wind_speed = 20.0
wind_direction = "East"

print("\nENVIRONMENTAL INPUT")
print("-" * 70)
print(f"Wind speed        : {wind_speed:.1f} km/h")
print(f"Wind direction    : {wind_direction}")
print("Source            : Prototype environmental input")


# ============================================================
# CONVERT SPILL MASK → GEOJSON
# ============================================================

print("\n")
print("=" * 70)
print("PROCESSING AI SPILL MASK")
print("=" * 70)

with rasterio.open(SPILL_MASK_PATH) as src:

    mask = src.read(1)

    transform = src.transform
    source_crs = src.crs

    print(f"Mask dimensions   : {src.width} x {src.height}")
    print(f"Mask CRS          : {source_crs}")

    # Extract connected raster regions.
    mask_shapes = shapes(
        mask.astype("uint8"),
        mask=(mask > 0),
        transform=transform
    )

    geometries = []

    for geometry, value in mask_shapes:

        if value == 1:

            # IMPORTANT:
            # Convert from the mask CRS (EPSG:32616)
            # to WGS84 latitude/longitude (EPSG:4326)
            geometry_wgs84 = transform_geom(
                source_crs,
                "EPSG:4326",
                geometry,
                precision=6
            )

            geometries.append(
                shape(geometry_wgs84)
            )


if not geometries:
    print("No spill polygons found in mask.")
    raise SystemExit(0)


# ============================================================
# MERGE DETECTED REGIONS
# ============================================================

spill_geometry = unary_union(geometries)

print(f"Detected regions  : {len(geometries)}")

if spill_geometry.geom_type == "Polygon":
    polygon_count = 1
else:
    polygon_count = len(spill_geometry.geoms)

print(f"Merged polygons   : {polygon_count}")


# ============================================================
# CREATE GEOJSON
# ============================================================

geojson_data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "scene": detection.get("scene"),
                "date": acquisition_date,
                "spill_area_km2": spill_area_km2,
                "mean_confidence": mean_confidence,
                "risk_level": risk_level,
                "source_crs": str(source_crs),
                "output_crs": "EPSG:4326"
            },
            "geometry": mapping(spill_geometry)
        }
    ]
}


with open(
    GEOJSON_OUTPUT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        geojson_data,
        file,
        indent=2
    )

print(f"GeoJSON saved     : {GEOJSON_OUTPUT}")
print("GeoJSON CRS       : EPSG:4326")


# ============================================================
# WIND MOVEMENT ESTIMATION
# ============================================================

movement_distance = wind_speed * 0.1

coordinate_change = movement_distance / 111.0

predicted_latitude = latitude
predicted_longitude = longitude


if wind_direction == "North":
    predicted_latitude += coordinate_change

elif wind_direction == "South":
    predicted_latitude -= coordinate_change

elif wind_direction == "East":
    predicted_longitude += coordinate_change

elif wind_direction == "West":
    predicted_longitude -= coordinate_change


# ============================================================
# CREATE FOLIUM MAP
# ============================================================

ocean_map = folium.Map(
    location=[latitude, longitude],
    zoom_start=9,
    tiles="OpenStreetMap"
)


# ============================================================
# ACTUAL AI SPILL FOOTPRINT
# ============================================================

folium.GeoJson(
    geojson_data,
    name="AI Spill Footprint",
    style_function=lambda feature: {
        "color": "red",
        "weight": 2,
        "fillColor": "red",
        "fillOpacity": 0.55
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[
            "scene",
            "date",
            "spill_area_km2",
            "mean_confidence",
            "risk_level"
        ],
        aliases=[
            "Scene",
            "Date",
            "Spill Area (km²)",
            "Mean Confidence",
            "Risk Level"
        ]
    )
).add_to(ocean_map)


# ============================================================
# AI DETECTION CENTROID
# ============================================================

folium.Marker(
    [latitude, longitude],
    popup=f"""
    <b>AI-Detected Oil Spill</b><br><br>
    Scene: {detection.get("scene", "Unknown")}<br>
    Acquisition Date: {acquisition_date}<br>
    Spill Area: {spill_area_km2:.2f} km²<br>
    Coverage: {spill_percentage:.2f}%<br>
    Mean Confidence: {mean_confidence * 100:.1f}%<br>
    Maximum Confidence: {max_confidence * 100:.1f}%<br>
    Risk Level: {risk_level}
    """,
    tooltip="AI Spill Centroid"
).add_to(ocean_map)


# ============================================================
# DETECTION BOUNDING BOX
# ============================================================

if bbox:

    bbox_points = [
        [bbox["min_latitude"], bbox["min_longitude"]],
        [bbox["min_latitude"], bbox["max_longitude"]],
        [bbox["max_latitude"], bbox["max_longitude"]],
        [bbox["max_latitude"], bbox["min_longitude"]],
        [bbox["min_latitude"], bbox["min_longitude"]]
    ]

    folium.PolyLine(
        bbox_points,
        color="orange",
        weight=2,
        dash_array="5, 5",
        tooltip="AI Detection Bounding Box"
    ).add_to(ocean_map)


# ============================================================
# RISK ZONE
# ============================================================

risk_radius_km = math.sqrt(
    spill_area_km2 / math.pi
)

folium.Circle(
    location=[latitude, longitude],
    radius=risk_radius_km * 1000,
    color=risk_color,
    fill=True,
    fill_color=risk_color,
    fill_opacity=0.15,
    popup=f"""
    <b>AI Risk Zone</b><br><br>
    Risk Level: {risk_level}<br>
    Detected Spill Area: {spill_area_km2:.2f} km²<br>
    Equivalent Visualization Radius: {risk_radius_km:.2f} km
    """,
    tooltip=f"{risk_level} Risk Zone"
).add_to(ocean_map)


# ============================================================
# ESTIMATED MOVEMENT
# ============================================================

folium.Marker(
    [predicted_latitude, predicted_longitude],
    popup=f"""
    <b>Estimated Spill Movement</b><br><br>
    Wind Speed: {wind_speed:.1f} km/h<br>
    Wind Direction: {wind_direction}<br>
    Estimated Movement: {movement_distance:.2f} km
    """,
    tooltip="Estimated Future Position",
    icon=folium.Icon(
        color="black",
        icon="arrow-right"
    )
).add_to(ocean_map)


folium.PolyLine(
    [
        [latitude, longitude],
        [predicted_latitude, predicted_longitude]
    ],
    color="black",
    weight=3,
    dash_array="8, 8",
    tooltip="Estimated Spill Movement"
).add_to(ocean_map)


# ============================================================
# INFORMATION PANEL
# ============================================================

info_html = f"""
<div style="
    position: fixed;
    top: 20px;
    right: 20px;
    width: 280px;
    background-color: white;
    border: 2px solid #555;
    border-radius: 8px;
    z-index: 9999;
    font-size: 13px;
    padding: 12px;
">

<b>OCEANSHIELD-AI</b><br>
AI Spill Intelligence Dashboard

<hr>

<b>AI Detection</b><br>
Scene: {detection.get("scene", "Unknown")}<br>
Date: {acquisition_date}<br>
Spill Area: {spill_area_km2:.2f} km²<br>
Coverage: {spill_percentage:.2f}%<br>
Confidence: {mean_confidence * 100:.1f}%<br>
Risk: <b>{risk_level}</b>

<hr>

<b>Movement Estimate</b><br>
Wind: {wind_speed:.1f} km/h<br>
Direction: {wind_direction}<br>
Estimated Movement: {movement_distance:.2f} km

<hr>

<b>GIS Layers</b><br>
🔴 Actual AI Spill Footprint<br>
📍 Detection Centroid<br>
🟠 Detection Bounding Box<br>
⭕ Risk Zone<br>
→ Estimated Movement

</div>
"""

ocean_map.get_root().html.add_child(
    folium.Element(info_html)
)


# ============================================================
# LEGEND
# ============================================================

legend_html = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    width: 180px;
    background-color: white;
    border: 2px solid grey;
    z-index: 9999;
    font-size: 13px;
    padding: 10px;
">

<b>Risk Level</b><br><br>

<span style="color:red;">●</span> HIGH<br>
<span style="color:orange;">●</span> MEDIUM<br>
<span style="color:green;">●</span> LOW

<hr>

<b>AI Spill Footprint</b><br>
<span style="color:red;">■</span> Detected Area

</div>
"""

ocean_map.get_root().html.add_child(
    folium.Element(legend_html)
)


# ============================================================
# LAYER CONTROL
# ============================================================

folium.LayerControl().add_to(ocean_map)


# ============================================================
# FIT MAP
# ============================================================

if bbox:

    ocean_map.fit_bounds([
        [bbox["min_latitude"], bbox["min_longitude"]],
        [bbox["max_latitude"], bbox["max_longitude"]]
    ])

else:

    ocean_map.location = [
        latitude,
        longitude
    ]


# ============================================================
# SAVE
# ============================================================

ocean_map.save(MAP_OUTPUT)

print("\n")
print("=" * 70)
print("GIS RESULTS")
print("=" * 70)

print(f"Risk level              : {risk_level}")
print(f"Spill polygons          : {polygon_count}")
print(f"Detected spill area     : {spill_area_km2:.2f} km²")
print(f"Visualization radius    : {risk_radius_km:.2f} km")
print(f"Estimated movement      : {movement_distance:.2f} km")

print(
    f"Predicted position      : "
    f"{predicted_latitude:.6f}, "
    f"{predicted_longitude:.6f}"
)

print("\nOUTPUT FILES")
print("-" * 70)
print(f"GeoJSON                 : {GEOJSON_OUTPUT}")
print(f"Interactive map         : {MAP_OUTPUT}")

print("\nGIS MAP GENERATED SUCCESSFULLY!")

