
import { useEffect, useState } from "react";
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  Circle,
  Marker,
  Popup,
  Polyline,
  useMap
} from "react-leaflet";

import L from "leaflet";

import "leaflet/dist/leaflet.css";
import "./App.css";


const API_BASE = "http://127.0.0.1:8000";


/* ============================================================
   MAP AUTO-ZOOM COMPONENT
   ============================================================ */

function MapBounds({ geojson }) {

  const map = useMap();

  useEffect(() => {

    if (!geojson) return;

    try {

      const layer = L.geoJSON(geojson);
      const bounds = layer.getBounds();

      if (bounds.isValid()) {
        map.fitBounds(bounds, {
          padding: [30, 30]
        });
      }

    } catch (error) {

      console.error("Unable to fit GIS bounds:", error);

    }

  }, [geojson, map]);

  return null;
}


/* ============================================================
   MAIN APPLICATION
   ============================================================ */

function App() {

  const [detection, setDetection] = useState(null);
  const [attribution, setAttribution] = useState(null);
  const [gis, setGis] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");


  /* ============================================================
     LOAD BACKEND DATA
     ============================================================ */

  useEffect(() => {

    async function loadDashboard() {

      try {

        const [
          detectionResponse,
          attributionResponse,
          gisResponse
        ] = await Promise.all([

          fetch(`${API_BASE}/api/detection`),

          fetch(`${API_BASE}/api/attribution`),

          fetch(`${API_BASE}/api/gis`)

        ]);


        if (
          !detectionResponse.ok ||
          !attributionResponse.ok ||
          !gisResponse.ok
        ) {

          throw new Error(
            "Backend API request failed"
          );

        }


        const detectionData =
          await detectionResponse.json();

        const attributionData =
          await attributionResponse.json();

        const gisData =
          await gisResponse.json();


        setDetection(detectionData);

        setAttribution(attributionData);

        setGis(gisData);


      } catch (err) {

        console.error(err);

        setError(
          "Unable to connect to OceanShield-AI backend. Make sure FastAPI is running."
        );

      } finally {

        setLoading(false);

      }

    }


    loadDashboard();

  }, []);


  /* ============================================================
     LOADING SCREEN
     ============================================================ */

  if (loading) {

    return (

      <div className="loading-screen">

        <div>

          <h1>🌊 OCEANSHIELD-AI</h1>

          <p>
            Loading intelligence dashboard...
          </p>

        </div>

      </div>

    );

  }


  /* ============================================================
     ERROR SCREEN
     ============================================================ */

  if (error) {

    return (

      <div className="error-screen">

        <h1>🌊 OCEANSHIELD-AI</h1>

        <p>{error}</p>

      </div>

    );

  }


  /* ============================================================
     DATA
     ============================================================ */

  const result =
    detection?.data;

  const vessels =
    attribution?.vessels || [];

  const geojson =
    gis?.data;


  const latitude =
    result?.centroid?.latitude || 0;

  const longitude =
    result?.centroid?.longitude || 0;


  const spillArea =
    result?.spill?.area_km2 || 0;

  const spillPercentage =
    result?.spill?.percentage || 0;

  const confidence =
    result?.confidence?.mean_spill || 0;


  /* ============================================================
     MOVEMENT ESTIMATE
     ============================================================ */

  const movementDistance = 2.0;

  const predictedLongitude =
    longitude + movementDistance / 111;


  /* ============================================================
     MAP CENTER
     ============================================================ */

  const mapCenter = [
    latitude,
    longitude
  ];


  /* ============================================================
     RENDER DASHBOARD
     ============================================================ */

  return (

    <div className="app">


      {/* ======================================================
         HEADER
         ====================================================== */}

      <header className="header">

        <div>

          <h1>
            🌊 OCEANSHIELD-AI
          </h1>

          <p>
            AI Spill Intelligence Dashboard
          </p>

        </div>


        <div className="status">

          <span className="status-dot"></span>

          SYSTEM ONLINE

        </div>

      </header>


      <main className="dashboard">


        {/* ====================================================
           KPI CARDS
           ==================================================== */}

        <section className="kpi-grid">


          <div className="card kpi">

            <span className="label">
              SPILL AREA
            </span>

            <strong>
              {spillArea.toFixed(2)} km²
            </strong>

          </div>


          <div className="card kpi">

            <span className="label">
              COVERAGE
            </span>

            <strong>
              {spillPercentage.toFixed(2)}%
            </strong>

          </div>


          <div className="card kpi">

            <span className="label">
              CONFIDENCE
            </span>

            <strong>
              {(confidence * 100).toFixed(1)}%
            </strong>

          </div>


          <div className="card kpi risk-card">

            <span className="label">
              RISK LEVEL
            </span>

            <strong>
              HIGH
            </strong>

          </div>


        </section>


        {/* ====================================================
           AI DETECTION
           ==================================================== */}

        <section className="card section">


          <div className="section-header">

            <div>

              <h2>
                AI Spill Detection
              </h2>

              <p>
                U-Net satellite image segmentation result
              </p>

            </div>


            <span className="badge high">
              SPILL DETECTED
            </span>

          </div>


          <div className="info-grid">


            <div>

              <span className="label">
                SCENE
              </span>

              <p>
                {result?.scene}
              </p>

            </div>


            <div>

              <span className="label">
                ACQUISITION DATE
              </span>

              <p>
                {result?.acquisition_date}
              </p>

            </div>


            <div>

              <span className="label">
                CENTROID
              </span>

              <p>
                {latitude.toFixed(6)},{" "}
                {longitude.toFixed(6)}
              </p>

            </div>


            <div>

              <span className="label">
                CRS
              </span>

              <p>
                {result?.crs}
              </p>

            </div>


          </div>

        </section>


        {/* ====================================================
           INTERACTIVE GIS MAP
           ==================================================== */}

        <section className="card section map-section">


          <div className="section-header">

            <div>

              <h2>
                🗺️ Live GIS Spill Intelligence
              </h2>

              <p>
                Interactive geospatial visualization generated from
                the AI spill mask
              </p>

            </div>


            <span className="badge">
              GIS ACTIVE
            </span>

          </div>


          <div className="map-container">

            <MapContainer
              center={mapCenter}
              zoom={9}
              scrollWheelZoom={true}
              className="ocean-map"
            >


              {/* BASE MAP */}

              <TileLayer
                attribution='&copy; OpenStreetMap contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />


              {/* ACTUAL AI SPILL FOOTPRINT */}

              {geojson && (

                <GeoJSON
                  data={geojson}
                  style={{
                    color: "red",
                    weight: 2,
                    fillColor: "red",
                    fillOpacity: 0.55
                  }}
                />

              )}


              {/* AI CENTROID */}

              <Marker
                position={[
                  latitude,
                  longitude
                ]}
              >

                <Popup>

                  <strong>
                    AI-Detected Oil Spill
                  </strong>

                  <br />
                  <br />

                  Scene:
                  {" "}
                  {result?.scene}

                  <br />

                  Date:
                  {" "}
                  {result?.acquisition_date}

                  <br />

                  Spill Area:
                  {" "}
                  {spillArea.toFixed(2)}
                  {" km²"}

                  <br />

                  Confidence:
                  {" "}
                  {(confidence * 100).toFixed(1)}
                  %

                </Popup>

              </Marker>


              {/* RISK ZONE */}

              <Circle
                center={[
                  latitude,
                  longitude
                ]}
                radius={5040}
                pathOptions={{
                  color: "red",
                  fillColor: "red",
                  fillOpacity: 0.10
                }}
              />


              {/* ESTIMATED MOVEMENT */}

              <Polyline
                positions={[
                  [
                    latitude,
                    longitude
                  ],

                  [
                    latitude,
                    predictedLongitude
                  ]
                ]}
                pathOptions={{
                  color: "black",
                  weight: 4,
                  dashArray: "8 8"
                }}
              />


              {/* FUTURE POSITION */}

              <Marker
                position={[
                  latitude,
                  predictedLongitude
                ]}
              >

                <Popup>

                  <strong>
                    Estimated Spill Movement
                  </strong>

                  <br />
                  <br />

                  Wind:
                  {" "}
                  20 km/h

                  <br />

                  Direction:
                  {" "}
                  East

                  <br />

                  Estimated movement:
                  {" "}
                  2.00 km

                </Popup>

              </Marker>


              {/* AUTO FIT */}

              <MapBounds
                geojson={geojson}
              />


            </MapContainer>


            {/* MAP LEGEND */}

            <div className="map-legend">

              <strong>
                GIS Layers
              </strong>

              <div>
                🔴 AI Spill Footprint
              </div>

              <div>
                📍 Detection Centroid
              </div>

              <div>
                ⭕ Risk Zone
              </div>

              <div>
                ➜ Estimated Movement
              </div>

            </div>


          </div>


        </section>


        {/* ====================================================
           VESSEL ATTRIBUTION
           ==================================================== */}

        <section className="card section">


          <div className="section-header">

            <div>

              <h2>
                🚢 Vessel Attribution
              </h2>

              <p>
                Evidence-based AIS candidate ranking
              </p>

            </div>


            <span className="badge">
              {vessels.length} CANDIDATES
            </span>

          </div>


          <div className="table-wrapper">

            <table>

              <thead>

                <tr>

                  <th>
                    RANK
                  </th>

                  <th>
                    MMSI
                  </th>

                  <th>
                    TYPE
                  </th>

                  <th>
                    DISTANCE
                  </th>

                  <th>
                    TIME
                  </th>

                  <th>
                    TRAJECTORY
                  </th>

                  <th>
                    SCORE
                  </th>

                </tr>

              </thead>


              <tbody>

                {vessels.map(
                  (vessel, index) => (

                    <tr
                      key={vessel.MMSI}
                    >

                      <td>

                        <span className="rank">

                          #
                          {vessel.rank ||
                            index + 1}

                        </span>

                      </td>


                      <td>
                        {vessel.MMSI}
                      </td>


                      <td>
                        {vessel.ship_type}
                      </td>


                      <td>
                        {Number(
                          vessel.distance_km
                        ).toFixed(2)}
                        {" km"}
                      </td>


                      <td>
                        {Number(
                          vessel.time_difference_minutes
                        ).toFixed(0)}
                        {" min"}
                      </td>


                      <td>
                        {vessel.trajectory}
                      </td>


                      <td>

                        <strong className="score">

                          {Number(
                            vessel.attribution_likelihood
                          ).toFixed(1)}
                          %

                        </strong>

                      </td>

                    </tr>

                  )
                )}

              </tbody>

            </table>

          </div>


          <p className="disclaimer">

            Attribution scores represent evidence-based ranking
            and do not prove causation.

          </p>


        </section>


        {/* ====================================================
           GIS SUMMARY
           ==================================================== */}

        <section className="card section">


          <div className="section-header">

            <div>

              <h2>
                GIS Analysis
              </h2>

              <p>
                Spatial processing generated from the AI prediction
              </p>

            </div>


            <span className="badge">
              EPSG:4326
            </span>

          </div>


          <div className="gis-summary">


            <div>

              <span className="label">
                FOOTPRINT
              </span>

              <p>
                AI detected spill geometry
              </p>

            </div>


            <div>

              <span className="label">
                FEATURES
              </span>

              <p>
                {geojson?.features?.length || 0}
                {" geospatial feature(s)"}
              </p>

            </div>


            <div>

              <span className="label">
                RISK
              </span>

              <p className="risk-text">
                HIGH
              </p>

            </div>


          </div>


        </section>


      </main>


      <footer>

        OceanShield-AI • AI-powered maritime spill intelligence prototype

      </footer>


    </div>

  );

}


export default App;

