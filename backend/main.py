from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import pandas as pd


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="OceanShield-AI Backend",
    description="Backend API for oil-spill detection, GIS analysis, AIS vessel attribution and assessment.",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent


# ------------------------------------------------------------
# Detection
# ------------------------------------------------------------

DETECTION_FILE = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "detection_result.json"
)


# ------------------------------------------------------------
# AIS Attribution
# ------------------------------------------------------------

ATTRIBUTION_FILE = (
    PROJECT_ROOT
    / "ml"
    / "attribution"
    / "outputs"
    / "ranked_candidates.csv"
)


# ------------------------------------------------------------
# Vessel Investigation
# ------------------------------------------------------------

VESSEL_INVESTIGATION_FILE = (
    PROJECT_ROOT
    / "ml"
    / "attribution"
    / "outputs"
    / "vessel_investigation.json"
)


# ------------------------------------------------------------
# Vessel Tracks
# ------------------------------------------------------------

VESSEL_TRACKS_FILE = (
    PROJECT_ROOT
    / "ml"
    / "attribution"
    / "outputs"
    / "vessel_tracks.geojson"
)


# ------------------------------------------------------------
# GIS
# ------------------------------------------------------------

GIS_FILE = (
    PROJECT_ROOT
    / "Person3_GIS"
    / "output"
    / "spill_footprint.geojson"
)


# ------------------------------------------------------------
# AI Assessment
# ------------------------------------------------------------

AI_ASSESSMENT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "assessment"
    / "outputs"
    / "ai_assessment.json"
)


# ------------------------------------------------------------
# Decision
# ------------------------------------------------------------

DECISION_FILE = (
    PROJECT_ROOT
    / "ml"
    / "decision"
    / "outputs"
    / "decision_result.json"
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "success": True,
        "project": "OceanShield-AI",
        "message": "OceanShield-AI backend is running",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check():
    return {
        "success": True,
        "status": "healthy",
        "project": "OceanShield-AI"
    }


# ============================================================
# DETECTION API
# ============================================================

@app.get("/api/detection")
def get_detection():

    if not DETECTION_FILE.exists():
        return {
            "success": False,
            "message": "Detection result file not found",
            "file": str(DETECTION_FILE)
        }

    try:

        with open(
            DETECTION_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            detection_data = json.load(file)

        return {
            "success": True,
            "data": detection_data
        }

    except json.JSONDecodeError:

        return {
            "success": False,
            "message": "Invalid detection JSON file"
        }

    except Exception as e:

        return {
            "success": False,
            "message": f"Error reading detection result: {str(e)}"
        }


# ============================================================
# AIS ATTRIBUTION API
# ============================================================

@app.get("/api/attribution")
def get_attribution():

    if not ATTRIBUTION_FILE.exists():
        return {
            "success": False,
            "message": "Attribution result file not found",
            "file": str(ATTRIBUTION_FILE)
        }

    try:

        df = pd.read_csv(ATTRIBUTION_FILE)

        # Convert NaN values to None
        df = df.where(pd.notnull(df), None)

        vessels = df.to_dict(orient="records")

        return {
            "success": True,
            "count": len(vessels),
            "vessels": vessels
        }

    except Exception as e:

        return {
            "success": False,
            "message": f"Error reading attribution results: {str(e)}"
        }


# ============================================================
# VESSEL INVESTIGATION API
# ============================================================

@app.get("/api/vessel-investigation")
def get_vessel_investigation():
    """
    Return detailed AIS vessel investigation results.

    Source:
    ml/attribution/outputs/vessel_investigation.json
    """

    if not VESSEL_INVESTIGATION_FILE.exists():

        return {
            "success": False,
            "message": "Vessel investigation file not found",
            "file": str(VESSEL_INVESTIGATION_FILE)
        }

    try:

        with open(
            VESSEL_INVESTIGATION_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            investigation_data = json.load(file)

        return {
            "success": True,
            "data": investigation_data
        }

    except json.JSONDecodeError:

        return {
            "success": False,
            "message": "Invalid vessel investigation JSON file"
        }

    except Exception as e:

        return {
            "success": False,
            "message": f"Error reading vessel investigation: {str(e)}"
        }


# ============================================================
# VESSEL TRACKS API
# ============================================================

@app.get("/api/vessel-tracks")
def get_vessel_tracks():
    """
    Return vessel track GeoJSON if it exists.
    """

    if not VESSEL_TRACKS_FILE.exists():

        return {
            "success": False,
            "message": "Vessel tracks GeoJSON file not found",
            "file": str(VESSEL_TRACKS_FILE)
        }

    try:

        with open(
            VESSEL_TRACKS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            tracks_data = json.load(file)

        return {
            "success": True,
            "data": tracks_data
        }

    except json.JSONDecodeError:

        return {
            "success": False,
            "message": "Invalid vessel tracks GeoJSON file"
        }

    except Exception as e:

        return {
            "success": False,
            "message": f"Error reading vessel tracks: {str(e)}"
        }


# ============================================================
# GIS API
# ============================================================

@app.get("/api/gis")
def get_gis():

    if not GIS_FILE.exists():
        return {
            "success": False,
            "message": "GIS footprint file not found",
            "file": str(GIS_FILE)
        }

    try:

        with open(
            GIS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            gis_data = json.load(file)

        return {
            "success": True,
            "data": gis_data
        }

    except json.JSONDecodeError:

        return {
            "success": False,
            "message": "Invalid GIS GeoJSON file"
        }

    except Exception as e:

        return {
            "success": False,
            "message": f"Error reading GIS data: {str(e)}"
        }


# ============================================================
# AI ASSESSMENT API
# ============================================================

@app.get("/api/ai-assessment")
def get_ai_assessment():

    if not AI_ASSESSMENT_FILE.exists():
        return {
            "success": False,
            "message": "AI assessment file not found",
            "file": str(AI_ASSESSMENT_FILE)
        }

    try:

        with open(
            AI_ASSESSMENT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            assessment_data = json.load(file)

        return {
            "success": True,
            "data": assessment_data
        }

    except json.JSONDecodeError:

        return {
            "success": False,
            "message": "Invalid AI assessment JSON file"
        }

    except Exception as e:

        return {
            "success": False,
            "message": f"Error reading AI assessment: {str(e)}"
        }


# ============================================================
# DECISION API
# ============================================================

@app.get("/api/decision")
def get_decision():

    if not DECISION_FILE.exists():
        return {
            "success": False,
            "message": "Decision result file not found",
            "file": str(DECISION_FILE)
        }

    try:

        with open(
            DECISION_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            decision_data = json.load(file)

        return {
            "success": True,
            "data": decision_data
        }

    except json.JSONDecodeError:

        return {
            "success": False,
            "message": "Invalid decision JSON file"
        }

    except Exception as e:

        return {
            "success": False,
            "message": f"Error reading decision result: {str(e)}"
        }


# ============================================================
# END
# ============================================================