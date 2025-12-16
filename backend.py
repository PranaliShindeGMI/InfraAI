# backend.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import os
from agents import generate_vm_recommendations 
from preprocessing import preprocess_vm_data

# -----------------------------
# LOAD PROCESSED DATA
# -----------------------------
INPUT_DIR = "data/input"
VM_INPUT_FILE = "VM_instance_data.csv"

def load_vm_data():
    path = os.path.join(INPUT_DIR, VM_INPUT_FILE)
    return pd.read_csv(path)

def load_bq_data():
    path = os.path.join(INPUT_DIR, "bq_daily.csv")
    return pd.read_csv(path)

# -----------------------------
# PLACEHOLDER AGENT CALLS
# (These will call your ML + LLM agents later)
# -----------------------------




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
        output = generate_vm_recommendations(df)   # <-- function in agents.py
        return JSONResponse(content=output)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def root():
    return {"status": "InfraAI backend running"}
