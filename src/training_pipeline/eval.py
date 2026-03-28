"""
Evaluate a saved XGBoost model on the eval split.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_EVAL = Path("data/data_processed/cleaning_valid_hour.csv")
DEFAULT_MODEL = Path("models/xgb_model.pkl")



def evaluate_model(
    model_path: Path | str = DEFAULT_MODEL,
    eval_path: Path | str = DEFAULT_EVAL,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, float]:


    eval_df = pd.read_csv(eval_path)
    
    target = 'num_rentals'

    eval_df['hour'] = pd.to_datetime(eval_df['date']).dt.hour
    eval_df['day_of_week'] = pd.to_datetime(eval_df['date']).dt.dayofweek
    eval_df['month'] = pd.to_datetime(eval_df['date']).dt.month

    X_eval = eval_df.drop(columns=[target, 'date'])
    y_eval = eval_df[target]


    model = load(model_path)
    y_pred = model.predict(X_eval)

    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    print(" Evaluation:")
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()
