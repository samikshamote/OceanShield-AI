import folium

# Spill location
latitude = float(input("Enter spill latitude: "))
longitude = float(input("Enter spill longitude: "))
affected_radius = float(input("Enter affected radius (km): "))

# Create map
map = folium.Map(
    location=[latitude, longitude],
    zoom_start=10
)

# Add spill location
folium.Marker(
    [latitude, longitude],
    popup="Oil Spill Detected",
    tooltip="Spill Location"
).add_to(map)

# Add affected area
folium.Circle(
    location=[latitude, longitude],
    radius=affected_radius * 1000,
    popup=f"Potentially Affected Area - {affected_radius} km",
    tooltip="Affected Zone"
).add_to(map)

# Save map
map.save("output/spill_map.html")

print("\nGIS map generated successfully!")
print("Spill Location:", latitude, longitude)
print("Affected Radius:", affected_radius, "km")    