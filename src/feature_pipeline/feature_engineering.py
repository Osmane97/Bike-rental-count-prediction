"""
Feature engineering: transforms raw datasets into features ready for modeling.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')
BASE_DIR = Path("data")
INPUT_DIR = BASE_DIR / 'data_processed'
OUTPUT_DIR = BASE_DIR / 'data_featured'


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to a single DataFrame.
    Steps:
        - Convert 'date' to datetime.
        - Create hour_category from 'hour'.
        - Create interaction features: temp_x_humidity, temp_x_wind_speed, temp_feel_diff.
        - One‑hot encode categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or preprocessed DataFrame (must contain columns: date, hour, temp_norm,
        humidity_norm, wind_speed, feels_like_temp_norm).

    Returns
    -------
    pd.DataFrame
        DataFrame with all new features added and categorical columns one‑hot encoded.
    """
    df = df.copy()

    # Ensure datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Hour category
    if "hour" in df.columns:
        df["hour_category"] = "night"
        df.loc[(df["hour"] >= 6) & (df["hour"] < 12), "hour_category"] = "morning"
        df.loc[(df["hour"] >= 12) & (df["hour"] < 17), "hour_category"] = "afternoon"
        df.loc[(df["hour"] >= 17) & (df["hour"] < 21), "hour_category"] = "evening"
    else:
        raise ValueError("Column 'hour' is required but missing.")

    # Interaction features
    required_cols = ["temp_norm", "humidity_norm", "wind_speed", "feels_like_temp_norm"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for interactions: {missing}")

    df["temp_x_humidity"] = df["temp_norm"] * df["humidity_norm"]
    df["temp_x_wind_speed"] = df["temp_norm"] * df["wind_speed"]
    df["temp_feel_diff"] = df["feels_like_temp_norm"] - df["temp_norm"]

    # One‑hot encoding (int dtype to match training)
    df = pd.get_dummies(df, dtype=int)

    return df


def feature_engineering(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR
):
    """
    Load preprocessed train/valid/holdout datasets,
    perform feature engineering,
    and save final datasets ready for modeling.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "train": input_dir / "train_preprocessed.csv",
        "valid": input_dir / "valid_preprocessed.csv",
        "holdout": input_dir / "holdout_preprocessed.csv"
    }

    datasets = {}
    for name, path in file_map.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        datasets[name] = pd.read_csv(path)

    # Apply feature engineering to each dataset
    featured = {}
    for name, df in datasets.items():
        featured[name] = apply_feature_engineering(df)

    # Align columns to training set (important for valid/holdout)
    train_cols = featured["train"].columns
    featured["valid"] = featured["valid"].reindex(columns=train_cols, fill_value=0)
    featured["holdout"] = featured["holdout"].reindex(columns=train_cols, fill_value=0)

    # Save
    for name, df in featured.items():
        df.to_csv(output_dir / f"{name}_featured.csv", index=False)

    return featured["train"], featured["valid"], featured["holdout"]


if __name__ == "__main__":
    feature_engineering()