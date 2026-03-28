# Unit tests for individual pipeline components + one integration test
import pandas as pd
from pathlib import Path
import pytest


# Import functions already created within feature enginneering pipeline


from src.feature_pipeline.load import load_and_split_data
from src.feature_pipeline.preprocess import preprocess_data
from src.feature_pipeline.feature_engineering import feature_engineering


# =========================
# load.py – unit test
# =========================
def test_load_and_split_data_creates_splits(tmp_path):
    dummy_path = tmp_path / "hour.csv"

    # Create raw data
    df = pd.DataFrame({
        "dteday": pd.date_range(start="2011-12-01", periods=320, freq="D"),
        "hr": list(range(320)),  # ✅ FIXED
        "season": [1]*320,
        "yr": [0]*320,
        "mnth": [12]*150 + [1]*170,
        "holiday": [0]*320,
        "weekday": [5]*320,
        "workingday": [1]*320,
        "weathersit": [1]*320,
        "temp": [0.3]*320,
        "atemp": [0.3]*320,
        "hum": [0.8]*320,
        "windspeed": [0.1]*320,
        "casual": [10]*320,
        "registered": [20]*320,
        "cnt": [30]*320,
    })

    df.to_csv(dummy_path, index=False)

    # Run function
    train, valid, holdout = load_and_split_data(
        raw_path=dummy_path,
        output_path=tmp_path
    )

    # Check outputs aren't empty
    assert not train.empty
    assert not valid.empty
    assert not holdout.empty

    # Check date splits
    assert train["dteday"].max() < pd.Timestamp("2012-01-01")
    assert valid["dteday"].min() >= pd.Timestamp("2012-01-01")
    assert holdout["dteday"].min() >= pd.Timestamp("2012-10-01")

    # Check files created
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "valid.csv").exists()
    assert (tmp_path / "holdout.csv").exists()

    print(" Data splitting test passed")



# =========================
# preprocess.py – unit test
# =========================

def test_preprocess_data_creates_clean_files(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    input_dir.mkdir()

    # Create dummy data with anomalies
    df = pd.DataFrame({
        "dteday": pd.date_range(start="2012-01-01", periods=5, freq="D"),
        "hr": [1, 1, 2, 3, 3],
        "weathersit": [1]*5,
        "workingday": [1]*5,
        "windspeed": [0.1]*5,
        "weekday": [2]*5,
        "cnt": [30, 30, 5, 30, 30],
        "temp": [0.3]*5,
        "atemp": [0.3]*5,
        "hum": [0.8]*5,
        "mnth": [1]*5,
        "yr": [1]*5,
        "casual": [10]*5,
        "registered": [20]*5,
    })

    # Add duplicate
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    # Add null
    df.loc[1, "temp"] = None

    # Save correct filenames
    df.to_csv(input_dir / "train_df.csv", index=False)
    df.to_csv(input_dir / "valid_df.csv", index=False)
    df.to_csv(input_dir / "holdout_df.csv", index=False)

    # Run preprocessing
    preprocess_data(input_dir=input_dir, output_dir=output_dir)

    # Check files exist
    assert (output_dir / "train_preprocessed.csv").exists()
    assert (output_dir / "valid_preprocessed.csv").exists()
    assert (output_dir / "holdout_preprocessed.csv").exists()

    processed = pd.read_csv(output_dir / "train_preprocessed.csv")

    # Column checks
    expected_columns = {
        "date", "hour", "weather_situation", "working_day",
        "wind_speed", "week_day", "num_rentals",
        "temp_norm", "feels_like_temp_norm", "humidity_norm",
        "month", "year", "casual", "registered"
    }

    assert expected_columns.issubset(set(processed.columns))
    assert "dteday" not in processed.columns
    assert "cnt" not in processed.columns
    assert "instant" not in processed.columns

    # Date type
    processed["date"] = pd.to_datetime(processed["date"])
    assert pd.api.types.is_datetime64_any_dtype(processed["date"])

    # No duplicates
    assert processed.duplicated().sum() == 0

    # No nulls
    assert processed.isnull().sum().sum() == 0

    # Logical integrity
    assert (processed["num_rentals"] >= processed["casual"] + processed["registered"]).all()
    
    print(" preprocess test passed")

# =========================
# feature_engineering.py – unit test
# =========================
def test_feature_engineering_creates_features_in_memory():
    df = pd.DataFrame({
        "date": pd.date_range(start="2011-01-01", periods=6, freq="h"),
        "hour": [0, 7, 13, 18, 22, 10],
        "temp_norm": [0.3, 0.4, 0.5, 0.6, 0.35, 0.45],
        "humidity_norm": [0.8, 0.7, 0.6, 0.65, 0.75, 0.72],
        "wind_speed": [0.1, 0.2, 0.15, 0.3, 0.12, 0.18],
        "feels_like_temp_norm": [0.32, 0.42, 0.52, 0.61, 0.36, 0.47],
        "num_rentals": [30, 40, 50, 60, 70, 80]
    })

    df_fe = feature_engineering(df)

    # Interaction features
    assert "temp_x_humidity" in df_fe.columns
    assert "temp_x_wind_speed" in df_fe.columns
    assert "temp_feel_diff" in df_fe.columns

    # One-hot hour_category
    assert any(col.startswith("hour_category_") for col in df_fe.columns)

    # No nulls
    assert not df_fe.isnull().any().any()

    # Original columns (except dropped ones) present
    assert all(c in df_fe.columns for c in ["temp_norm", "humidity_norm", "wind_speed", "feels_like_temp_norm", "hour"])

    print(" Feature engineering test passed")


# =========================
# integration test
# =========================
# Confirms the whole feature pipeline works together.

def test_full_feature_pipeline_integration(tmp_path):


    raw_path = tmp_path / "hour.csv"

    df = pd.DataFrame({
        "dteday": pd.date_range(start="2011-12-01", periods=320, freq="D"),
        "hr": list(range(320)),
        "season": [1]*320,
        "yr": [0]*320,
        "mnth": [12]*150 + [1]*170,
        "holiday": [0]*320,
        "weekday": [5]*320,
        "workingday": [1]*320,
        "weathersit": [1]*320,
        "temp": [0.3]*320,
        "atemp": [0.3]*320,
        "hum": [0.8]*320,
        "windspeed": [0.1]*320,
        "casual": [10]*320,
        "registered": [20]*320,
        "cnt": [30]*320,
    })

    df.to_csv(raw_path, index=False)


    # Step 2: Run load/split

    split_dir = tmp_path / "split"
    load_and_split_data(raw_path=raw_path, output_path=split_dir)

    # Fix filename mismatch for preprocess step
    (split_dir / "train.csv").rename(split_dir / "train_df.csv")
    (split_dir / "valid.csv").rename(split_dir / "valid_df.csv")
    (split_dir / "holdout.csv").rename(split_dir / "holdout_df.csv")


    # Step 3: Preprocess

    processed_dir = tmp_path / "processed"
    preprocess_data(input_dir=split_dir, output_dir=processed_dir)


    # Step 4: Feature Engineering
 
    featured_dir = tmp_path / "featured"
    feature_engineering(input_dir=processed_dir, output_dir=featured_dir)

    # Step 5: Final Assertions


    train_file = featured_dir / "train_featured.csv"
    valid_file = featured_dir / "valid_featured.csv"
    holdout_file = featured_dir / "holdout_featured.csv"

    # Files exist
    assert train_file.exists()
    assert valid_file.exists()
    assert holdout_file.exists()

    # Load data
    df_train = pd.read_csv(train_file)
    df_valid = pd.read_csv(valid_file)
    df_holdout = pd.read_csv(holdout_file)

    # Feature checks
    assert "temp_x_humidity" in df_train.columns
    assert "temp_x_wind_speed" in df_train.columns
    assert "temp_feel_diff" in df_train.columns

    # One-hot encoding
    assert any(col.startswith("hour_category_") for col in df_train.columns)

    # Alignment (CRITICAL)
    assert list(df_train.columns) == list(df_valid.columns)
    assert list(df_train.columns) == list(df_holdout.columns)

    # No nulls
    assert not df_train.isnull().any().any()

    print("Full feature pipeline integration test passed")