# Goal: Create a FastAPI app to serve your trained ML model into a web service that anyone 
# (or any system) can call over HTTP.

from fastapi import FastAPI            # Web framework for APIs
from pathlib import Path               # For handling file paths cleanly
from typing import List, Dict, Any     # For type hints (clarity in endpoints)
import pandas as pd                    # To handle incoming JSON as DataFrames
import boto3, os                       # AWS SDK for Python + env variables

# Import inference pipeline
from src.inference_pipeline.inference import predict


# ----------------------------
# Config
# ----------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "rental-bike-data")
REGION = os.getenv("AWS_REGION", "eu-west-2")

s3 = boto3.client("s3", region_name=REGION)

def load_from_s3(key, local_path):
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        print(f"📥 Downloading {key} from S3…")
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = Path(load_from_s3("models/best_xgb_model.pkl", "models/best_xgb_model.pkl"))
TRAIN_FE_PATH = Path(load_from_s3("data_processed/feature_engineered_train.csv",
                                 "data/data_processed/feature_engineered_train.csv"))

# Load schema
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "num_rentals"]
else:
    TRAIN_FEATURE_COLUMNS = None

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Bike Rental Prediction API")

@app.get("/")
def root():
    return {"message": "Bike Rental Prediction API is running 🚀"}

@app.get("/health")
def health():
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH)}
    if not MODEL_PATH.exists():
        status["status"] = "unhealthy"
        status["error"] = "Model not found"
    else:
        status["status"] = "healthy"
        if TRAIN_FEATURE_COLUMNS:
            status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)
    return status

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict_batch(data: List[dict]):
    if not MODEL_PATH.exists():
        return {"error": "Model not found"}

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data provided"}

    preds_df = predict(df, model_path=MODEL_PATH)

    return {
        "predictions": preds_df["num_rentals_prediction"].astype(float).tolist()
    }

    """
🔹 Execution Order / Module Flow

1. Imports (FastAPI, pandas, boto3, your inference function).
2. Config setup (env vars → bucket/region).
3. S3 utility (load_from_s3).
4. Download + load model/artifacts (MODEL_PATH, TRAIN_FE_PATH).
5. Infer schema (TRAIN_FEATURE_COLUMNS).
6. Create FastAPI app (app = FastAPI).
7. Declare endpoints (/, /health, /predict, /run_batch, /latest_predictions).
"""