import json
import math
import os

import folium
from folium.plugins import Fullscreen

from shapely.geometry import Polygon, mapping


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DETECTION_JSON = os.path.join(
    BASE_DIR,
    "ml",
    "detection",
    "outputs",
    "detection_result.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Person3_GIS",
    "output"
)

GIS_JSON = os.path.join(
    OUTPUT_DIR,
    "gis_analysis.json"
)

GEOJSON_FILE = os.path.join(
    OUTPUT_DIR,
    "spill_footprint.geojson"
)

MAP_FILE = os.path.join(
    OUTPUT_DIR,
    "spill_map.html"
)

VESSEL_TRACKS_FILE = os.path.join(
    BASE_DIR,
    "ml",
    "attribution",
    "outputs",
    "vessel_tracks.geojson"
)

SPILL_MASK_FILE = os.path.join(
    BASE_DIR,
    "ml",
    "detection",
    "outputs",
    "spill_mask.tif"
)


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD AI DETECTION RESULT
# ============================================================

if not os.path.exists(DETECTION_JSON):
    raise FileNotFoundError(
        f"Detection result not found:\n{DETECTION_JSON}"
    )

with open(DETECTION_JSON, "r", encoding="utf-8") as file:
    detection = json.load(file)


# ============================================================
# EXTRACT REAL AI DATA
# ============================================================

scene = detection.get("scene", "Unknown")
acquisition_date = detection.get("acquisition_date", "Unknown")

source_crs = detection.get("crs", "Unknown")

spill_detected = detection.get("spill_detected", False)

spill_info = detection.get("spill", {})
confidence_info = detection.get("confidence", {})
centroid_info = detection.get("centroid", {})
bbox_info = detection.get("bounding_box", {})

spill_area_km2 = float(spill_info.get("area_km2", 0))
spill_percentage = float(spill_info.get("percentage", 0))

maximum_confidence = float(
    confidence_info.get("maximum", 0)
)

mean_confidence = float(
    confidence_info.get("mean_spill", 0)
)

latitude = float(
    centroid_info.get("latitude", 0)
)

longitude = float(
    centroid_info.get("longitude", 0)
)


# ============================================================
# PROTOTYPE RISK CLASSIFICATION
# ============================================================

if spill_area_km2 >= 50 and mean_confidence >= 0.70:
    risk_level = "HIGH"

elif spill_area_km2 >= 20 and mean_confidence >= 0.60:
    risk_level = "MEDIUM"

else:
    risk_level = "LOW"


# ============================================================
# PROTOTYPE WIND-BASED MOVEMENT
# ============================================================

wind_speed_kmh = 20
wind_direction = "East"

estimated_movement_km = 2.0


# Approximate conversion:
# 1 degree latitude ≈ 111 km
# longitude conversion depends on latitude

if wind_direction == "East":
    predicted_latitude = latitude

    predicted_longitude = longitude + (
        estimated_movement_km /
        (111 * math.cos(math.radians(latitude)))
    )

elif wind_direction == "West":
    predicted_latitude = latitude

    predicted_longitude = longitude - (
        estimated_movement_km /
        (111 * math.cos(math.radians(latitude)))
    )

elif wind_direction == "North":
    predicted_latitude = latitude + (
        estimated_movement_km / 111
    )

    predicted_longitude = longitude

elif wind_direction == "South":
    predicted_latitude = latitude - (
        estimated_movement_km / 111
    )

    predicted_longitude = longitude

else:
    predicted_latitude = latitude
    predicted_longitude = longitude


# ============================================================
# GEOJSON
# ============================================================

# IMPORTANT:
# The actual spill_mask.tif is currently missing.
#
# Therefore we do NOT fabricate an AI spill polygon.
#
# Until the real mask is supplied, we create a bounding-box
# fallback using the REAL bounding box from detection_result.json.
#
# This is clearly labelled as a fallback visualization.

spill_mask_available = os.path.exists(SPILL_MASK_FILE)

if spill_mask_available:

    print(
        "spill_mask.tif found."
    )

    print(
        "Actual raster polygon extraction should be performed "
        "with the project Rasterio environment."
    )

    footprint_source = "AI Spill Mask"

else:

    footprint_source = "Detection Bounding Box Fallback"

    min_latitude = float(
        bbox_info.get("min_latitude", latitude)
    )

    min_longitude = float(
        bbox_info.get("min_longitude", longitude)
    )

    max_latitude = float(
        bbox_info.get("max_latitude", latitude)
    )

    max_longitude = float(
        bbox_info.get("max_longitude", longitude)
    )

    polygon = Polygon([
        (min_longitude, min_latitude),
        (max_longitude, min_latitude),
        (max_longitude, max_latitude),
        (min_longitude, max_latitude),
        (min_longitude, min_latitude)
    ])

    geojson_feature = {
        "type": "Feature",
        "properties": {
            "source": footprint_source,
            "scene": scene,
            "risk_level": risk_level,
            "spill_area_km2": spill_area_km2,
            "mean_confidence": mean_confidence
        },
        "geometry": mapping(polygon)
    }

    geojson = {
        "type": "FeatureCollection",
        "features": [geojson_feature]
    }

    with open(
        GEOJSON_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            geojson,
            file,
            indent=4
        )


# ============================================================
# CREATE MAP
# ============================================================

map_object = folium.Map(
    location=[latitude, longitude],
    zoom_start=8,
    control_scale=True,
    tiles=None
)

# Reliable public basemap
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri",
    name="Esri World Street Map",
    overlay=False,
    control=True
).add_to(map_object)


# ============================================================
# FULLSCREEN
# ============================================================

Fullscreen().add_to(map_object)


# ============================================================
# AI SPILL FOOTPRINT / FALLBACK
# ============================================================

if os.path.exists(GEOJSON_FILE):

    footprint_layer = folium.FeatureGroup(
        name="AI Spill Footprint"
    )

    folium.GeoJson(
        GEOJSON_FILE,
        name="Spill Footprint",
        style_function=lambda feature: {
            "color": "red",
            "weight": 2,
            "fillColor": "red",
            "fillOpacity": 0.35
        },
        tooltip="AI Spill Footprint"
    ).add_to(footprint_layer)

    footprint_layer.add_to(map_object)


# ============================================================
# DETECTION CENTROID
# ============================================================

centroid_layer = folium.FeatureGroup(
    name="Detection Centroid"
)

folium.Marker(
    location=[latitude, longitude],
    tooltip="AI Detection Centroid",
    popup=f"""
    <b>AI Detection Centroid</b><br><br>

    Latitude: {latitude:.6f}<br>
    Longitude: {longitude:.6f}<br>

    Spill Area: {spill_area_km2:.2f} km²<br>
    Mean Confidence: {mean_confidence * 100:.2f}%<br>
    Maximum Confidence: {maximum_confidence * 100:.2f}%
    """
).add_to(centroid_layer)

centroid_layer.add_to(map_object)


# ============================================================
# RISK ZONE
# ============================================================

if risk_level == "HIGH":
    risk_color = "red"

elif risk_level == "MEDIUM":
    risk_color = "orange"

else:
    risk_color = "green"


# Visualization radius based on actual spill area
visualization_radius_km = math.sqrt(
    spill_area_km2 / math.pi
)


risk_layer = folium.FeatureGroup(
    name="AI-derived Risk Zone"
)

folium.Circle(
    location=[latitude, longitude],
    radius=visualization_radius_km * 1000,
    color=risk_color,
    fill=True,
    fill_color=risk_color,
    fill_opacity=0.15,
    tooltip=f"AI-derived {risk_level} Risk Zone",
    popup=f"""
    <b>AI-derived Prototype Risk Classification</b><br><br>

    Risk Level: {risk_level}<br>
    Spill Area: {spill_area_km2:.2f} km²<br>
    Mean Confidence: {mean_confidence * 100:.2f}%<br>
    Visualization Radius: {visualization_radius_km:.2f} km
    """
).add_to(risk_layer)

risk_layer.add_to(map_object)


# ============================================================
# PROTOTYPE DRIFT PROJECTION
# ============================================================

movement_layer = folium.FeatureGroup(
    name="Prototype Drift Projection"
)


# Predicted position marker

folium.Marker(
    location=[
        predicted_latitude,
        predicted_longitude
    ],
    tooltip="Predicted Spill Position",
    popup=f"""
    <b>Prototype Drift Projection</b><br><br>

    Wind Speed: {wind_speed_kmh} km/h<br>
    Wind Direction: {wind_direction}<br>
    Estimated Movement: {estimated_movement_km:.2f} km<br><br>

    Predicted Latitude:
    {predicted_latitude:.6f}<br>

    Predicted Longitude:
    {predicted_longitude:.6f}
    """
).add_to(movement_layer)


# Movement line

folium.PolyLine(
    locations=[
        [latitude, longitude],
        [
            predicted_latitude,
            predicted_longitude
        ]
    ],
    color="blue",
    weight=4,
    dash_array="10",
    tooltip="Prototype Wind-based Movement"
).add_to(movement_layer)


movement_layer.add_to(map_object)


# ============================================================
# VESSEL TRACKS
# ============================================================

if os.path.exists(VESSEL_TRACKS_FILE):

    vessel_layer = folium.FeatureGroup(
        name="Vessel Tracks"
    )

    folium.GeoJson(
        VESSEL_TRACKS_FILE,
        name="AIS Vessel Tracks",
        style_function=lambda feature: {
            "color": "purple",
            "weight": 3
        },
        tooltip="AIS Vessel Track"
    ).add_to(vessel_layer)

    vessel_layer.add_to(map_object)

    vessel_tracks_status = "Available"

else:

    vessel_tracks_status = "Not available"


# ============================================================
# LAYER CONTROL
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(map_object)


# ============================================================
# SAVE MAP
# ============================================================

map_object.save(MAP_FILE)


# ============================================================
# BACKEND GIS JSON
# ============================================================

gis_analysis = {

    "project": "OceanShield-AI",

    "scene": scene,

    "date": acquisition_date,

    "spill_detected": spill_detected,

    "risk_level": risk_level,

    "risk_classification": "AI-derived Prototype Risk Classification",

    "spill": {

        "area_km2": spill_area_km2,

        "coverage_percentage": spill_percentage,

        "mean_confidence": mean_confidence,

        "maximum_confidence": maximum_confidence

    },

    "centroid": {

        "latitude": latitude,

        "longitude": longitude

    },

    "bounding_box": {

        "min_latitude":
            bbox_info.get("min_latitude"),

        "min_longitude":
            bbox_info.get("min_longitude"),

        "max_latitude":
            bbox_info.get("max_latitude"),

        "max_longitude":
            bbox_info.get("max_longitude")

    },

    "movement": {

        "type": "Prototype Drift Projection",

        "wind_speed_kmh": wind_speed_kmh,

        "wind_direction": wind_direction,

        "estimated_distance_km":
            estimated_movement_km,

        "predicted_latitude":
            predicted_latitude,

        "predicted_longitude":
            predicted_longitude

    },

    "crs": {

        "source": source_crs,

        "web_map": "EPSG:4326"

    },

    "visualization": {

        "footprint_source":
            footprint_source,

        "visualization_radius_km":
            visualization_radius_km,

        "vessel_tracks":
            vessel_tracks_status

    },

    "limitations": [

        "Prototype Risk Classification is not a scientifically validated environmental risk model.",

        "Prototype Drift Projection is based on configured wind input and is not real-time ocean-current prediction.",

        "Predicted position is for situational awareness and is not an exact future spill boundary.",

        "Actual AI spill geometry requires spill_mask.tif."

    ]

}


with open(
    GIS_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        gis_analysis,
        file,
        indent=4
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print()
print("==============================================")
print("OceanShield-AI GIS Spatial Intelligence")
print("==============================================")

print("Scene:", scene)

print("Acquisition Date:", acquisition_date)

print("Source CRS:", source_crs)

print("Web Map CRS: EPSG:4326")

print()

print("Spill Area:",
      round(spill_area_km2, 2),
      "km²")

print("Spill Coverage:",
      round(spill_percentage, 2),
      "%")

print("Mean Confidence:",
      round(mean_confidence * 100, 2),
      "%")

print("Maximum Confidence:",
      round(maximum_confidence * 100, 2),
      "%")

print()

print("Detection Centroid:",
      latitude,
      longitude)

print("Risk Level:",
      risk_level)

print()

print("Prototype Drift Projection")

print("Wind Speed:",
      wind_speed_kmh,
      "km/h")

print("Wind Direction:",
      wind_direction)

print("Estimated Movement:",
      estimated_movement_km,
      "km")

print("Predicted Position:",
      predicted_latitude,
      predicted_longitude)

print()

print("Footprint Source:",
      footprint_source)

print("Vessel Tracks:",
      vessel_tracks_status)

print()

print("GIS Analysis JSON:",
      GIS_JSON)

print("GeoJSON:",
      GEOJSON_FILE)

print("Map:",
      MAP_FILE)

print()

print("GIS analysis completed successfully.")

print("==============================================")