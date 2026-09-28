
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
  const [aiAssessment, setAiAssessment] = useState(null);
  const [decision, setDecision] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeView, setActiveView] = useState("overview");


  /* ============================================================
     LOAD BACKEND DATA
     ============================================================ */

  useEffect(() => {

    async function loadDashboard() {

      try {

        const [
                detectionResponse,
                attributionResponse,
                gisResponse,
                aiAssessmentResponse,
                decisionResponse
              ] = await Promise.all([

            fetch(`${API_BASE}/api/detection`),

            fetch(`${API_BASE}/api/attribution`),

            fetch(`${API_BASE}/api/gis`),

            fetch(`${API_BASE}/api/ai-assessment`),

            fetch(`${API_BASE}/api/decision`)

          ]);


          if (
              !detectionResponse.ok ||
              !attributionResponse.ok ||
              !gisResponse.ok ||
              !aiAssessmentResponse.ok ||
              !decisionResponse.ok
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

        const aiAssessmentData =
          await aiAssessmentResponse.json();

        const decisionData =
          await decisionResponse.json();  


          setDetection(detectionData);

          setAttribution(attributionData);

          setGis(gisData);

          setAiAssessment(aiAssessmentData);

          setDecision(decisionData);


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
  
  const assessment =
  aiAssessment?.data;

  const decisionData =
    decision?.data;  


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
     INVESTIGATION CONSOLE NAVIGATION
     ============================================================ */

  const navigateTo = (view, sectionId) => {

    setActiveView(view);

    const section = document.getElementById(sectionId);

    if (section) {

      section.scrollIntoView({
        behavior: "smooth",
        block: "start"
      });

    }

  };


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
           INVESTIGATION CONSOLE
           ==================================================== */}

        <section className="console-shell">

          <div className="console-header">

            <div>

              <span className="console-eyebrow">
                OPERATOR WORKSPACE
              </span>

              <h2>
                Investigation Console
              </h2>

              <p>
                Investigate a detected maritime incident from AI evidence
                through spatial analysis, vessel attribution and decision support.
              </p>

            </div>

            <div className="console-status">

              <span className="console-status-dot"></span>

              INCIDENT ACTIVE

            </div>

          </div>


          {/* INCIDENT CONTROL BAR */}

<div className="console-controls">

  <button
    className={
      activeView === "overview"
        ? "console-button active"
        : "console-button"
    }
    onClick={() =>
      navigateTo("overview", "ai-spill-detection")
    }
  >
    <span>01</span>
    Incident Overview
  </button>


  <button
    className={
      activeView === "evidence"
        ? "console-button active"
        : "console-button"
    }
    onClick={() =>
      navigateTo("evidence", "ai-evidence")
    }
  >
    <span>02</span>
    AI Evidence
  </button>


  <button
    className={
      activeView === "gis"
        ? "console-button active"
        : "console-button"
    }
    onClick={() =>
      navigateTo("gis", "gis-intelligence")
    }
  >
    <span>03</span>
    GIS Intelligence
  </button>


  <button
    className={
      activeView === "vessels"
        ? "console-button active"
        : "console-button"
    }
    onClick={() =>
      navigateTo("vessels", "vessel-investigation")
    }
  >
    <span>04</span>
    Vessel Investigation
  </button>


  <button
    className={
      activeView === "decision"
        ? "console-button active"
        : "console-button"
    }
    onClick={() =>
      navigateTo("decision", "decision-support")
    }
  >
    <span>05</span>
    Decision Support
  </button>

</div>


          {/* CURRENT INCIDENT */}

          <div className="console-incident">

            <div>

              <span className="label">
                ACTIVE INCIDENT
              </span>

              <strong>
                {result?.scene || "Loading incident..."}
              </strong>

            </div>


            <div>

              <span className="label">
                LOCATION
              </span>

              <strong>
                {latitude.toFixed(4)}, {longitude.toFixed(4)}
              </strong>

            </div>


            <div>

              <span className="label">
                STATUS
              </span>

              <strong className="console-high">
                INVESTIGATION REQUIRED
              </strong>

            </div>


            <div>

              <span className="label">
                VERIFICATION
              </span>

              <strong>
                {decisionData?.decision?.human_verification_required
                  ? "HUMAN REQUIRED"
                  : "AUTOMATED"}
              </strong>

            </div>

          </div>

        </section>








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

          <section
            id="ai-spill-detection"
            className="card section"
          >


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
           AI ASSESSMENT + INCIDENT DECISION
           ==================================================== */}

        <section className="assessment-grid">


          {/* AI ASSESSMENT */}

          <div className="card section">

            <div className="section-header">

              <div>

                <h2>
                  🤖 AI Assessment
                </h2>

                <p>
                  AI-assisted interpretation of detected spill candidates
                </p>

              </div>

              <span className="badge high">
                {assessment?.assessment?.confidence_level || "N/A"}
              </span>

            </div>


            <div className="assessment-items">


              <div className="assessment-item">

                <span className="label">
                  CONFIDENCE LEVEL
                </span>

                <strong>
                  {assessment?.assessment?.confidence_level || "N/A"}
                </strong>

              </div>


              <div className="assessment-item">

                <span className="label">
                  SPATIAL EXTENT
                </span>

                <strong>
                  {assessment?.assessment?.spatial_extent || "N/A"}
                </strong>

              </div>


              <div className="assessment-item">

                <span className="label">
                  DETECTION QUALITY
                </span>

                <strong>
                  {assessment?.assessment?.detection_quality || "N/A"}
                </strong>

              </div>


              <div className="assessment-item">

                <span className="label">
                  PRIMARY REGION
                </span>

                <strong>
                  #
                  {assessment?.candidate_analysis?.primary_candidate?.region_id ?? "N/A"}
                </strong>

              </div>


              <div className="assessment-item">

                <span className="label">
                  CANDIDATE REGIONS
                </span>

                <strong>
                  {assessment?.candidate_analysis?.total_regions ?? "N/A"}
                </strong>

              </div>


              <div className="assessment-item">

                <span className="label">
                  PRIMARY AREA
                </span>

                <strong>
                  {assessment?.candidate_analysis?.primary_candidate?.area_pixels?.toLocaleString() || "N/A"}
                  {" px"}
                </strong>

              </div>


              <div className="assessment-item">

                <span className="label">
                  PRIMARY CONFIDENCE
                </span>

                <strong>
                  {(
                    (assessment?.candidate_analysis?.primary_candidate?.mean_confidence || 0)
                    * 100
                  ).toFixed(2)}
                  %
                </strong>

              </div>


            </div>

          </div>


          {/* INCIDENT DECISION */}

          {/* INCIDENT DECISION */}

            <div
              id="decision-support"
              className="card section"
            >

            <div className="section-header">

              <div>

                <h2>
                  🚨 Incident Assessment
                </h2>

                <p>
                  Rule-based decision support requiring human verification
                </p>

              </div>

              <span className="badge high">
                {decisionData?.decision?.risk_level || "N/A"}
              </span>

            </div>


            <div className="incident-panel">


              <div className="incident-main">

                <span className="label">
                  ALERT PRIORITY
                </span>

                <strong>
                  {decisionData?.decision?.alert_priority || "N/A"}
                </strong>

              </div>


              <div className="incident-main">

                <span className="label">
                  DECISION SCORE
                </span>

                <strong>
                  {decisionData?.decision?.decision_score ?? "N/A"}
                  /100
                </strong>

              </div>


              <div className="verification-box">

                <span>
                  👤 HUMAN VERIFICATION
                </span>

                <strong>
                  {decisionData?.decision?.human_verification_required
                    ? "REQUIRED"
                    : "NOT REQUIRED"}
                </strong>

              </div>


              <div className="decision-explanation">

                <span className="label">
                  DECISION EXPLANATION
                </span>

                <p>
                  {decisionData?.decision?.explanation ||
                    "No explanation available."}
                </p>

              </div>


            </div>

          </div>

                </section>


        {/* ====================================================
           AI EXPLANATION + MODEL VALIDATION
           ==================================================== */}

        <section
  id="ai-evidence"
  className="card section"
>

  <div className="section-header">

    <div>

      <h2>
        🧠 Why Did AI Flag This?
              </h2>

              <p>
                Explainable assessment generated from the AI detection pipeline
              </p>

            </div>

            <span className="badge">
              U-NET
            </span>

          </div>


          {/* AI EXPLANATION */}

          <div className="explanation-box">

            <p>
              {assessment?.assessment?.explanation ||
                "AI explanation unavailable."}
            </p>

          </div>


          {/* MODEL DETAILS */}

          <div className="model-details">

            <div>

              <span className="label">
                MODEL
              </span>

              <strong>
                {assessment?.model?.architecture || "N/A"}
              </strong>

            </div>


            <div>

              <span className="label">
                INPUT
              </span>

              <strong>
                {assessment?.model?.input || "N/A"}
              </strong>

            </div>


            <div>

              <span className="label">
                PATCH SIZE
              </span>

              <strong>
                {assessment?.model?.patch_size || "N/A"}
                {" × "}
                {assessment?.model?.patch_size || "N/A"}
              </strong>

            </div>


            <div>

              <span className="label">
                THRESHOLD
              </span>

              <strong>
                {assessment?.model?.threshold ?? "N/A"}
              </strong>

            </div>

          </div>


          {/* VALIDATION */}

          <div className="validation-section">

            <h3>
              📊 Model Validation
            </h3>

            <p className="validation-note">
		Validation performance during model development — not a probability 		of oil detection.         </p>


            <div className="validation-grid">


              <div className="validation-card">

                <span>
                  IoU
                </span>

                <strong>
                  {(
                    (assessment?.validation?.iou || 0) * 100
                  ).toFixed(2)}
                  %
                </strong>

              </div>


              <div className="validation-card">

                <span>
                  Dice / F1
                </span>

                <strong>
                  {(
                    (assessment?.validation?.dice || 0) * 100
                  ).toFixed(2)}
                  %
                </strong>

              </div>


              <div className="validation-card">

                <span>
                  Precision
                </span>

                <strong>
                  {assessment?.validation?.precision != null
  ? `${(assessment.validation.precision * 100).toFixed(2)}%`
  : "N/A"}
                  %
                </strong>

              </div>


              <div className="validation-card">

                <span>
                  Recall
                </span>

                <strong>
                  {assessment?.validation?.recall != null
  ? `${(assessment.validation.recall * 100).toFixed(2)}%`
  : "N/A"}
                  %
                </strong>

              </div>

            </div>


            <div className="validation-footer">

              <span>
                Dataset:
              </span>

              <strong>
                {assessment?.validation?.dataset || "Oil Spill 23 Scenes"}              	      </strong>

              <span>
                •
              </span>

              <span>
                Scene:
              </span>

              <strong>
                {assessment?.assessment?.scene || "N/A"}           
	      </strong>

            </div>

          </div>


          {/* LIMITATIONS */}

          <div className="limitations-box">

            <strong>
              ⚠️ AI Safety Note
            </strong>

            <p>
              AI confidence represents model output strength and does not by itself prove the presence of oil. Human verification is required before operational action.
            </p>

          </div>

        </section>



        {/* ====================================================
   INTERACTIVE GIS MAP
   ==================================================== */}

<section
  id="gis-intelligence"
  className="card section map-section"
>

  <div className="section-header">

    <div>

      <h2>
        🗺️ Live GIS Spill Intelligence
      </h2>

      <p>
        Spatial visualization of AI-detected spill footprint,
        risk zone and prototype movement estimate
      </p>

    </div>

    <span className="badge">
      GIS ACTIVE
    </span>

  </div>


  {/* MAP */}

  <div className="map-container">

    <MapContainer
      center={mapCenter}
      zoom={9}
      scrollWheelZoom={true}
      className="ocean-map"
    >

      {/* ====================================================
         BASE MAP
         ==================================================== */}

      <TileLayer
        attribution='&copy; OpenStreetMap contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />


      {/* ====================================================
         AI SPILL FOOTPRINT
         ==================================================== */}

      {geojson && (

        <GeoJSON
          data={geojson}
          style={{
            color: "#ff3030",
            weight: 2.5,
            fillColor: "#ff3030",
            fillOpacity: 0.48
          }}

          onEachFeature={(feature, layer) => {

            layer.bindPopup(`
              <div style="min-width:220px">

                <strong style="font-size:15px">
                  🔴 AI Spill Footprint
                </strong>

                <br /><br />

                <b>Scene:</b>
                ${result?.scene || "N/A"}

                <br />

                <b>Detected area:</b>
                ${spillArea.toFixed(2)} km²

                <br />

                <b>Mean confidence:</b>
                ${(confidence * 100).toFixed(1)}%

                <br /><br />

                <span style="color:#b45309">
                  AI detection requires human verification.
                </span>

              </div>
            `);

          }}

        />

      )}


      {/* ====================================================
         DETECTION CENTROID
         ==================================================== */}

      <Marker
        position={[
          latitude,
          longitude
        ]}
      >

        <Popup>

          <div style={{ minWidth: "220px" }}>

            <strong style={{ fontSize: "15px" }}>
              📍 AI Detection Centroid
            </strong>

            <br />
            <br />

            <b>Latitude:</b>
            {" "}
            {latitude.toFixed(6)}

            <br />

            <b>Longitude:</b>
            {" "}
            {longitude.toFixed(6)}

            <br /><br />

            <b>Spill area:</b>
            {" "}
            {spillArea.toFixed(2)}
            {" km²"}

            <br />

            <b>Confidence:</b>
            {" "}
            {(confidence * 100).toFixed(1)}%

            <br />

            <b>Date:</b>
            {" "}
            {result?.acquisition_date || "N/A"}

          </div>

        </Popup>

      </Marker>


      {/* ====================================================
         PROTOTYPE RISK ZONE
         ==================================================== */}

      <Circle
        center={[
          latitude,
          longitude
        ]}
        radius={5040}
        pathOptions={{
          color: "#ff8a00",
          weight: 2,
          dashArray: "6 6",
          fillColor: "#ff8a00",
          fillOpacity: 0.08
        }}
      />


      {/* ====================================================
         ESTIMATED MOVEMENT PATH
         ==================================================== */}

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
          color: "#111827",
          weight: 4,
          dashArray: "10 8"
        }}
      />


      {/* ====================================================
         ESTIMATED FUTURE POSITION
         ==================================================== */}

      <Marker
        position={[
          latitude,
          predictedLongitude
        ]}
      >

        <Popup>

          <div style={{ minWidth: "220px" }}>

            <strong style={{ fontSize: "15px" }}>
              ➜ Prototype Movement Estimate
            </strong>

            <br />
            <br />

            <b>Wind:</b>
            {" "}
            20 km/h

            <br />

            <b>Direction:</b>
            {" "}
            East

            <br />

            <b>Estimated movement:</b>
            {" "}
            {movementDistance.toFixed(2)}
            {" km"}

            <br /><br />

            <span style={{ color: "#b45309" }}>
              Prototype wind-based estimate — not a
              physical oil-spread forecast.
            </span>

          </div>

        </Popup>

      </Marker>


      {/* ====================================================
         AUTOMATIC MAP FIT
         ==================================================== */}

      <MapBounds
        geojson={geojson}
      />

    </MapContainer>


    {/* ====================================================
       MAP OVERLAY — INCIDENT SUMMARY
       ==================================================== */}

    <div className="map-status-panel">

      <div className="map-status-title">
        INCIDENT SNAPSHOT
      </div>

      <div className="map-status-row">
        <span>SPILL AREA</span>
        <strong>
          {spillArea.toFixed(2)} km²
        </strong>
      </div>

      <div className="map-status-row">
        <span>CONFIDENCE</span>
        <strong>
          {(confidence * 100).toFixed(1)}%
        </strong>
      </div>

      <div className="map-status-row">
        <span>PRIMARY REGION</span>
        <strong>
          #{assessment?.candidate_analysis?.primary_candidate?.region_id ?? "N/A"}
        </strong>
      </div>

      <div className="map-status-row">
        <span>RISK</span>
        <strong className="map-risk">
          PROTOTYPE HIGH
        </strong>
      </div>

    </div>


    {/* ====================================================
       MAP LEGEND
       ==================================================== */}

    <div className="map-legend">

      <strong>
        GIS Layers
      </strong>

      <div>
        <span className="legend-dot spill-dot"></span>
        AI Spill Footprint
      </div>

      <div>
        <span className="legend-dot centroid-dot"></span>
        Detection Centroid
      </div>

      <div>
        <span className="legend-circle"></span>
        Prototype Risk Zone
      </div>

      <div>
        <span className="legend-line"></span>
        Estimated Movement
      </div>

      <div className="legend-note">
        AI footprint requires human verification.
      </div>

    </div>

  </div>

</section>


        {/* ====================================================
           VESSEL ATTRIBUTION
           ==================================================== */}

        <section
                id="vessel-investigation"
                className="card section"
              >


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

