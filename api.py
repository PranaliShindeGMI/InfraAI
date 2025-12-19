# backend.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import os
from backend.agents import generate_vm_recommendations 
from backend.preprocessing import preprocess_vm_data
from backend.analysis import analyze_vm_data

# -----------------------------
# LOAD PROCESSED DATA
# -----------------------------
INPUT_DIR = "backend/data/input"
VM_INPUT_FILE = "VM_instance_data.csv"

def load_vm_data():
    path = os.path.join(INPUT_DIR, VM_INPUT_FILE)
    return pd.read_csv(path)

def load_bq_data():
    path = os.path.join(INPUT_DIR, "bq_daily.csv")
    return pd.read_csv(path)

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(
    title="InfraAI Backend",
    description="Service-level recommendations for VM Instances and BigQuery",
    version="1.0"
)

# -----------------------------
# API ROUTES
# -----------------------------
@app.get("/vm/recommendations")
def vm_recommendations():
    try:
        df = load_vm_data()
        df = preprocess_vm_data(df)
        analysis_results = analyze_vm_data(df)
        output = generate_vm_recommendations(df)
        return JSONResponse(content=output)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/vm/analysis")
def vm_analysis():
    """
    Comprehensive analysis endpoint including:
    - Descriptive statistics
    - Correlation analysis
    - Trend analysis
    - Cost analysis
    - Resource utilization insights
    - Anomaly detection
    """
    try:
        df = load_vm_data()
        df = preprocess_vm_data(df)
        analysis_results = analyze_vm_data(df)
        return JSONResponse(content=analysis_results)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/vm/preprocessed")
def vm_preprocessed_data(df):
    """
    Get preprocessed and aggregated VM data
    """
    try:
        preprocessed = preprocess_vm_data(df)
        # Convert to dictionary for JSON response
        result = preprocessed.to_dict(orient='records')
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def root():
    return {"status": "InfraAI backend running"}