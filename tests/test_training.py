import math
from pathlib import Path
from joblib import load

# Import your functions
from src.training_pipeline.train import train_model
from src.training_pipeline.eval import evaluate_model
from src.training_pipeline.tune import tune_model

# Paths
TRAIN_PATH = Path("data/data_processed/cleaning_train_hour.csv")
EVAL_PATH = Path("data/data_processed/cleaning_valid_hour.csv")


# -------------------------
# Helper: validate metrics
# -------------------------
def _assert_train_metrics(m):
    assert set(m.keys()) == {"mae", "rmse", "r2"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in m.values())


def _assert_tune_metrics(m):
    assert set(m.keys()) == {"rmse", "r2", "best_params"}
    assert isinstance(m["rmse"], float) and math.isfinite(m["rmse"])
    assert isinstance(m["r2"], float) and math.isfinite(m["r2"])
    assert isinstance(m["best_params"], dict) and len(m["best_params"]) > 0


# -------------------------
# TRAIN TEST
# -------------------------
def test_train_creates_model_and_metrics(tmp_path):
    out_path = tmp_path / "xgb_model.pkl"

    _, metrics = train_model(
        train_path=TRAIN_PATH,
        eval_path=EVAL_PATH,
        model_output=out_path,
        model_params={"n_estimators": 20, "max_depth": 4, "learning_rate": 0.1},
    )

    assert out_path.exists()
    _assert_train_metrics(metrics)

    model = load(out_path)
    assert model is not None

    print("✅ train_model test passed")


# -------------------------
# EVAL TEST
# -------------------------
def test_eval_works_with_saved_model(tmp_path):
    model_path = tmp_path / "xgb_model.pkl"

    # Train model first
    train_model(
        train_path=TRAIN_PATH,
        eval_path=EVAL_PATH,
        model_output=model_path,
        model_params={"n_estimators": 20},
    )

    metrics = evaluate_model(
        model_path=model_path,
        eval_path=EVAL_PATH
    )

    _assert_train_metrics(metrics)

    print("✅ evaluate_model test passed")


# -------------------------
# TUNE TEST
# -------------------------
def test_tune_saves_best_model(tmp_path):
    model_out = tmp_path / "xgb_best.pkl"
    tracking_dir = tmp_path / "mlruns"

    model, metrics = tune_model(
        train_path=TRAIN_PATH,
        eval_path=EVAL_PATH,
        model_output=model_out,
        n_trials=2,  # keep fast
        tracking_uri=str(tracking_dir),
        experiment_name="test_xgb_optuna",
    )

    assert model_out.exists()
    _assert_tune_metrics(metrics)
    assert model is not None

    print("✅ tune_model test passed")