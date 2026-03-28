# tests/test_inference.py
import sys
import os
from pathlib import Path

import pandas as pd
import pytest

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference_pipeline.inference import predict


@pytest.fixture(scope="session")
def sample_df():
    """Load a small sample from cleaning_eval.csv for inference testing."""
    sample_path = ROOT / "data/data_processed/cleaning_valid_hour.csv"
    df = pd.read_csv(sample_path).sample(5, random_state=42).reset_index(drop=True)
    return df


from src.inference_pipeline.inference import predict

def test_inference_runs_and_returns_predictions(sample_df):
    preds_df = predict(sample_df)

    assert not preds_df.empty
    assert "num_rentals_prediction" in preds_df.columns
    assert pd.api.types.is_numeric_dtype(preds_df["num_rentals_prediction"])

    print(" Inference pipeline test passed. Predictions:")
    print(preds_df[["num_rentals_prediction"]].head())