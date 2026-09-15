import folium
import math

# Simulated data received from Person 1
spill_data = [
    {
        "latitude": 18.5204,
        "longitude": 73.8567,
        "confidence": 0.90,
        "wind_speed": 20,
        "wind_direction": "East"
    },
    {
        "latitude": 19.0760,
        "longitude": 72.8777,
        "confidence": 0.80,
        "wind_speed": 15,
        "wind_direction": "North"
    },
    {
        "latitude": 15.4909,
        "longitude": 73.8278,
        "confidence": 0.60,
        "wind_speed": 10,
        "wind_direction": "West"
    }
]


# Store all spill coordinates
spill_locations = []

for spill in spill_data:
    spill_locations.append([
        spill["latitude"],
        spill["longitude"]
    ])


# Create map
map = folium.Map(
    location=[18.5204, 73.8567],
    zoom_start=6
)


# Process each spill
for spill in spill_data:

    # Extract spill information
    latitude = spill["latitude"]
    longitude = spill["longitude"]
    confidence = spill["confidence"]

    # Extract wind information
    wind_speed = spill["wind_speed"]
    wind_direction = spill["wind_direction"]


    # Determine affected radius based on AI confidence
    if confidence >= 0.90:
        affected_radius = 5
    elif confidence >= 0.75:
        affected_radius = 3
    else:
        affected_radius = 1


    # Calculate potentially affected area
    affected_area = math.pi * affected_radius ** 2


    # Determine risk level
    if affected_area > 50:
        risk_level = "HIGH"
    elif affected_area >= 20:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"


    # Set map color based on risk level
    if risk_level == "HIGH":
        risk_color = "red"
    elif risk_level == "MEDIUM":
        risk_color = "orange"
    else:
        risk_color = "green"


    # -----------------------------------------
    # WIND-BASED SPILL MOVEMENT
    # -----------------------------------------

    # Simplified prototype:
    # 0.1 km movement for every 1 km/h wind speed
    movement_distance = wind_speed * 0.1


    # Convert movement distance into approximate latitude/longitude change
    coordinate_change = movement_distance / 111


    # Calculate predicted location
    predicted_latitude = latitude
    predicted_longitude = longitude


    if wind_direction == "North":
        predicted_latitude = latitude + coordinate_change

    elif wind_direction == "South":
        predicted_latitude = latitude - coordinate_change

    elif wind_direction == "East":
        predicted_longitude = longitude + coordinate_change

    elif wind_direction == "West":
        predicted_longitude = longitude - coordinate_change


    # Add original spill marker
    folium.Marker(
        [latitude, longitude],
        popup=f"""
        <b>Original Oil Spill</b><br><br>
        AI Confidence: {confidence * 100:.1f}%<br>
        Risk Level: {risk_level}<br>
        Affected Radius: {affected_radius} km<br>
        Affected Area: {affected_area:.2f} km²<br>
        Wind Speed: {wind_speed} km/h<br>
        Wind Direction: {wind_direction}
        """,
        tooltip="Original Spill Location"
    ).add_to(map)


    # Add affected area
    folium.Circle(
        location=[latitude, longitude],
        radius=affected_radius * 1000,
        popup=f"""
        <b>Oil Spill Information</b><br><br>
        AI Confidence: {confidence * 100:.1f}%<br>
        Risk Level: {risk_level}<br>
        Affected Radius: {affected_radius} km<br>
        Affected Area: {affected_area:.2f} km²<br>
        Wind Speed: {wind_speed} km/h<br>
        Wind Direction: {wind_direction}
        """,
        tooltip=f"{risk_level} Risk Zone",
        color=risk_color,
        fill=True,
        fill_color=risk_color,
        fill_opacity=0.4
    ).add_to(map)


    # Add predicted spill location
    folium.Marker(
        [predicted_latitude, predicted_longitude],
        popup=f"""
        <b>Predicted Spill Location</b><br><br>
        Wind Speed: {wind_speed} km/h<br>
        Wind Direction: {wind_direction}<br>
        Estimated Movement: {movement_distance:.2f} km
        """,
        tooltip="Predicted Spill Location",
        icon=folium.Icon(
            color="black",
            icon="arrow-right"
        )
    ).add_to(map)


    # Draw line from original spill to predicted location
    folium.PolyLine(
        [
            [latitude, longitude],
            [predicted_latitude, predicted_longitude]
        ],
        tooltip="Predicted Spill Movement"
    ).add_to(map)


    # Display information
    print("Original Spill Location:", latitude, longitude)
    print("AI Confidence:", confidence * 100, "%")
    print("Affected Radius:", affected_radius, "km")
    print("Potentially Affected Area:", round(affected_area, 2), "km²")
    print("Risk Level:", risk_level)
    print("Wind Speed:", wind_speed, "km/h")
    print("Wind Direction:", wind_direction)
    print("Estimated Movement:", round(movement_distance, 2), "km")
    print(
        "Predicted Spill Location:",
        round(predicted_latitude, 4),
        round(predicted_longitude, 4)
    )
    print("-----------------------------")


# Automatically fit map around all original spill locations
map.fit_bounds(spill_locations)


# Add risk legend
legend_html = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    width: 150px;
    background-color: white;
    border: 2px solid grey;
    z-index: 9999;
    font-size: 14px;
    padding: 10px;
">
    <b>Risk Level</b><br><br>

    <span style="color:red;">●</span> HIGH<br>
    <span style="color:orange;">●</span> MEDIUM<br>
    <span style="color:green;">●</span> LOW
</div>
"""

map.get_root().html.add_child(
    folium.Element(legend_html)
)


# Save map
map.save("output/spill_map.html")

print("GIS map generated successfully!")