
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DEFAULT_TRAIN = Path("data/data_processed/cleaning_train_hour.csv")
DEFAULT_EVAL = Path("data/data_processed/cleaning_valid_hour.csv")
DEFAULT_OUT = Path("models/xgb_model.pkl")


def train_model(
    train_path: Path | str = DEFAULT_TRAIN,
    eval_path: Path | str = DEFAULT_EVAL,
    model_output: Path | str = DEFAULT_OUT,
    model_params: Optional[Dict] = None,
    random_state: int = 42,
):
    """Train baseline XGB and save model."""

    # Load datasets
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    # Define features and target
    target = 'num_rentals'
    train_df['hour'] = pd.to_datetime(train_df['date']).dt.hour
    train_df['day_of_week'] = pd.to_datetime(train_df['date']).dt.dayofweek
    train_df['month'] = pd.to_datetime(train_df['date']).dt.month

    eval_df['hour'] = pd.to_datetime(eval_df['date']).dt.hour
    eval_df['day_of_week'] = pd.to_datetime(eval_df['date']).dt.dayofweek
    eval_df['month'] = pd.to_datetime(eval_df['date']).dt.month

    X_train = train_df.drop(columns=[target, 'date'])
    y_train = train_df[target]

    X_eval = eval_df.drop(columns=[target, 'date'])
    y_eval = eval_df[target]

    
    # Default model parameters
    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    # Update with any custom parameters
    if model_params:
        params.update(model_params)

    # Train model
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_eval)
    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    # Save model
    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out)
    print(f"✅ Model trained. Saved to {out}")
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    return model, metrics


if __name__ == "__main__":
    train_model()