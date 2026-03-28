from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

import mlflow
import mlflow.xgboost


DEFAULT_TRAIN = Path("data/data_processed/cleaning_train_hour.csv")
DEFAULT_EVAL = Path("data/data_processed/cleaning_valid_hour.csv")
DEFAULT_OUT = Path("models/best_xgb_model.pkl")


# -------------------------
# Load & preprocess data
# -------------------------
def _load_data(train_path: Path | str, eval_path: Path | str):

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    target = "num_rentals"

    # Feature engineering
    if "date" in train_df.columns:
        train_df["hour"] = pd.to_datetime(train_df["date"]).dt.hour
        train_df["day_of_week"] = pd.to_datetime(train_df["date"]).dt.dayofweek
        train_df["month"] = pd.to_datetime(train_df["date"]).dt.month

    if "date" in eval_df.columns:
        eval_df["hour"] = pd.to_datetime(eval_df["date"]).dt.hour
        eval_df["day_of_week"] = pd.to_datetime(eval_df["date"]).dt.dayofweek
        eval_df["month"] = pd.to_datetime(eval_df["date"]).dt.month

    X_train = train_df.drop(columns=[target, "date"], errors="ignore")
    y_train = train_df[target]

    X_eval = eval_df.drop(columns=[target, "date"], errors="ignore")
    y_eval = eval_df[target]

    return X_train, y_train, X_eval, y_eval


# -------------------------
# Main tuning function
# -------------------------
def tune_model(
    train_path: Path | str = DEFAULT_TRAIN,
    eval_path: Path | str = DEFAULT_EVAL,
    model_output: Path | str = DEFAULT_OUT,
    n_trials: int = 15,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "xgboost_optuna_housing",
    random_state: int = 42,
    use_mlflow: bool = True,   #New to fix
) -> Tuple[XGBRegressor, dict]:

    # ✅ FIX: Proper MLflow URI handling
    if tracking_uri:
        tracking_uri = str(tracking_uri)

        if not tracking_uri.startswith("file://"):
            tracking_uri = Path(tracking_uri).absolute().as_uri()

        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    # Load data
    X_train, y_train, X_eval, y_eval = _load_data(train_path, eval_path)

    # -------------------------
    # Objective function
    # -------------------------
    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        with mlflow.start_run(run_name="trial"):

            model = XGBRegressor(**params)
            
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_eval, y_eval)],
                verbose=False,
            )

            y_pred = model.predict(X_eval)
            rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
            r2 = r2_score(y_eval, y_pred)

            mlflow.log_params(params)
            mlflow.log_metrics({"rmse": rmse, "r2": r2})

        return rmse

    # -------------------------
    # Run Optuna
    # -------------------------
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params

    # -------------------------
    # Train final model
    # -------------------------
    model = XGBRegressor(**best_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_eval, y_eval)],

        verbose=False,
    )

    y_pred = model.predict(X_eval)
    rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
    r2 = r2_score(y_eval, y_pred)

    # -------------------------
    # Save model locally
    # -------------------------
    model_output = Path(model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    dump(model, model_output)
    print(f" Model saved locally at {model_output}")

    # -------------------------
    # Log final model
    # -------------------------
    with mlflow.start_run(run_name="final_model"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({"rmse": rmse, "r2": r2})
        mlflow.xgboost.log_model(model, artifact_path="best_xgb_model")

    metrics = {
        "rmse": rmse,
        "r2": r2,
        "best_params": best_params,
    }

    return model, metrics


# -------------------------
# Run script
# -------------------------
if __name__ == "__main__":
    model, metrics = tune_model()
    print("Final metrics:", metrics)