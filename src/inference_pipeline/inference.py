"""
Inference pipeline for Bike Rental Prediction.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from joblib import load
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_pipeline.feature_engineering import apply_feature_engineering

DEFAULT_MODEL = PROJECT_ROOT / "models" / "best_xgb_model.pkl"

print("📂 Inference using project root:", PROJECT_ROOT)


def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """
    Run inference on a DataFrame.
    """
    # Load model
    model = load(model_path)
    expected_features = list(model.feature_names_in_)

    # Apply feature engineering
    df = apply_feature_engineering(input_df)

    # Drop target if present
    if "num_rentals" in df.columns:
        df = df.drop(columns=["num_rentals"])

    # Align columns to model expectations
    df = df.reindex(columns=expected_features, fill_value=0)

    # Predict
    preds = model.predict(df)

    # Output
    out = input_df.copy()
    out["num_rentals_prediction"] = preds
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"📥 Loaded {len(df)} rows from {args.input}")

    preds_df = predict(df)
    preds_df.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")