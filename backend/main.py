from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

DETECTION_FILE = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "detection_result.json"
)

AIS_FILE = (
    PROJECT_ROOT
    / "ml"
    / "attribution"
    / "outputs"
    / "ranked_candidates.csv"
)

GIS_FILE = (
    PROJECT_ROOT
    / "Person3_GIS"
    / "output"
    / "spill_footprint.geojson"
)

AI_ASSESSMENT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "ai_assessment.json"
)

DECISION_FILE = (
    PROJECT_ROOT
    / "ml"
    / "detection"
    / "outputs"
    / "decision_result.json"
)


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="OceanShield-AI API",
    description="AI-powered oil spill detection and intelligence API",
    version="1.0.0"
)

# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "project": "OceanShield-AI",
        "status": "online",
        "message": "OceanShield-AI backend is running"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "OceanShield-AI Backend"
    }


# --------------------------------------------------
# DETECTION RESULT
# --------------------------------------------------

@app.get("/api/detection")
def get_detection():

    if not DETECTION_FILE.exists():
        return {
            "success": False,
            "error": "Detection result file not found",
            "path": str(DETECTION_FILE)
        }

    try:
        with open(DETECTION_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            "success": True,
            "data": data
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# --------------------------------------------------
# AIS VESSEL ATTRIBUTION
# --------------------------------------------------

@app.get("/api/attribution")
def get_attribution():

    if not AIS_FILE.exists():
        return {
            "success": False,
            "error": "AIS attribution result not found",
            "path": str(AIS_FILE)
        }

    try:
        df = pd.read_csv(AIS_FILE)

        records = df.to_dict(orient="records")

        return {
            "success": True,
            "count": len(records),
            "vessels": records
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }

# --------------------------------------------------
# GIS SPILL FOOTPRINT
# --------------------------------------------------

@app.get("/api/gis")
def get_gis_data():

    if not GIS_FILE.exists():
        return {
            "success": False,
            "error": "GIS GeoJSON file not found",
            "path": str(GIS_FILE)
        }

    try:

        with open(GIS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            "success": True,
            "data": data
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# --------------------------------------------------
# AI ASSESSMENT
# --------------------------------------------------

@app.get("/api/ai-assessment")
def get_ai_assessment():

    if not AI_ASSESSMENT_FILE.exists():
        return {
            "success": False,
            "error": "AI assessment file not found",
            "path": str(AI_ASSESSMENT_FILE)
        }

    try:

        with open(
            AI_ASSESSMENT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return {
            "success": True,
            "data": data
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# --------------------------------------------------
# AI DECISION ENGINE
# --------------------------------------------------

@app.get("/api/decision")
def get_decision():

    if not DECISION_FILE.exists():
        return {
            "success": False,
            "error": "Decision result file not found",
            "path": str(DECISION_FILE)
        }

    try:

        with open(
            DECISION_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return {
            "success": True,
            "data": data
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }