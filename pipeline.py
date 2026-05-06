import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import FORECAST_HORIZON, MODEL_DIR, SELECTION_METRIC, VAL_WEEKS
from data_processor import load_and_resample
from feature_engineer import add_features, add_validation_flag, feature_columns
from models import (
    fit_lstm,
    fit_prophet,
    fit_sarima,
    fit_xgboost,
    predict_lstm,
    predict_prophet,
    predict_sarima,
    predict_xgboost,
)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "unknown"


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)[: len(y_true)]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    non_zero = y_true != 0
    mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100) if non_zero.any() else None
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def _score_model(name: str, trainer, predictor, y_true) -> dict:
    try:
        fitted = trainer()
        preds = predictor(fitted)
        return {"model": fitted, "metrics": evaluate(y_true, preds), "error": None}
    except Exception as exc:
        return {
            "model": None,
            "metrics": {"RMSE": float("inf"), "MAE": float("inf"), "MAPE": None},
            "error": f"{type(exc).__name__}: {exc}",
        }


def train_state_models(state: str, state_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    state_df = state_df.sort_values("ds").reset_index(drop=True)
    split_date = state_df["ds"].max() - pd.Timedelta(weeks=VAL_WEEKS)
    train_raw = state_df[state_df["ds"] <= split_date].copy()
    val_raw = state_df[state_df["ds"] > split_date].copy()

    featured = add_validation_flag(add_features(state_df))
    train_feat = featured[~featured["is_val"]].copy()
    val_feat = featured[featured["is_val"]].copy()

    if len(train_raw) < 40 or len(val_raw) != VAL_WEEKS:
        raise ValueError(f"{state} does not have enough weekly history for validation.")
    if train_feat.empty or val_feat.empty:
        raise ValueError(f"{state} does not have enough rows after lag feature creation.")

    scores = {
        "SARIMA": _score_model(
            "SARIMA",
            lambda: fit_sarima(train_raw["y"]),
            lambda model: predict_sarima(model, len(val_raw)),
            val_raw["y"].values,
        ),
        "Prophet": _score_model(
            "Prophet",
            lambda: fit_prophet(train_raw),
            lambda model: predict_prophet(model, len(val_raw)),
            val_raw["y"].values,
        ),
        "XGBoost": _score_model(
            "XGBoost",
            lambda: fit_xgboost(train_feat, feature_cols),
            lambda model: predict_xgboost(model, val_feat, feature_cols),
            val_feat["y"].values,
        ),
        "LSTM": _score_model(
            "LSTM",
            lambda: fit_lstm(train_raw["y"]),
            lambda bundle: predict_lstm(
                {**bundle, "history": train_raw["y"].astype(float).values},
                len(val_raw),
            ),
            val_raw["y"].values,
        ),
    }

    valid_scores = {name: value for name, value in scores.items() if np.isfinite(value["metrics"][SELECTION_METRIC])}
    if not valid_scores:
        raise RuntimeError(f"All model trainings failed for {state}.")

    best_name = min(valid_scores, key=lambda name: valid_scores[name]["metrics"][SELECTION_METRIC])
    return {
        "best_model": best_name,
        "scores": scores,
        "split_date": split_date,
        "train_rows": len(train_raw),
        "validation_rows": len(val_raw),
    }


def refit_best_model(model_name: str, state_df: pd.DataFrame, feature_cols: list[str]):
    if model_name == "SARIMA":
        return fit_sarima(state_df["y"])
    if model_name == "Prophet":
        return fit_prophet(state_df)
    if model_name == "XGBoost":
        return fit_xgboost(add_features(state_df), feature_cols)
    if model_name == "LSTM":
        return fit_lstm(state_df["y"])
    raise ValueError(f"Unsupported model: {model_name}")


def save_artifact(state: str, model_name: str, model, metadata: dict) -> Path:
    state_dir = MODEL_DIR / slugify(state)
    state_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "LSTM":
        model["model"].save(state_dir / "model.keras")
        joblib.dump({"scaler": model["scaler"], "seq_len": model["seq_len"]}, state_dir / "preprocessor.joblib")
        artifact_file = "model.keras"
    else:
        joblib.dump(model, state_dir / "model.joblib")
        artifact_file = "model.joblib"

    metadata = {**metadata, "artifact_file": artifact_file}
    with (state_dir / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, default=str)
    return state_dir


def train_and_save_models(limit_states: int | None = None) -> dict:
    df = load_and_resample()
    feature_cols = feature_columns()
    states = sorted(df["state"].unique())
    if limit_states:
        states = states[:limit_states]

    registry = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "forecast_horizon": FORECAST_HORIZON,
        "selection_metric": SELECTION_METRIC,
        "states": {},
    }

    for state in states:
        state_df = df[df["state"] == state].copy()
        result = train_state_models(state, state_df, feature_cols)
        best_name = result["best_model"]
        best_model = refit_best_model(best_name, state_df, feature_cols)

        serializable_scores = {
            name: {"metrics": item["metrics"], "error": item["error"]}
            for name, item in result["scores"].items()
        }
        metadata = {
            "state": state,
            "state_slug": slugify(state),
            "best_model": best_name,
            "metrics": serializable_scores,
            "feature_columns": feature_cols,
            "last_train_date": state_df["ds"].max().strftime("%Y-%m-%d"),
            "validation_start": (result["split_date"] + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d"),
            "validation_weeks": VAL_WEEKS,
            "train_rows": result["train_rows"],
            "validation_rows": result["validation_rows"],
        }
        artifact_dir = save_artifact(state, best_name, best_model, metadata)
        registry["states"][state] = {**metadata, "artifact_dir": str(artifact_dir)}

        rmse = metadata["metrics"][best_name]["metrics"]["RMSE"]
        print(f"{state}: best={best_name} RMSE={rmse:,.2f}")

    with (MODEL_DIR / "registry.json").open("w", encoding="utf-8") as fp:
        json.dump(registry, fp, indent=2, default=str)

    print(f"Training complete. Saved {len(registry['states'])} state model artifacts to {MODEL_DIR}.")
    return registry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and persist weekly state sales forecasters.")
    parser.add_argument("--limit-states", type=int, default=None, help="Train only the first N states for smoke tests.")
    args = parser.parse_args()
    train_and_save_models(limit_states=args.limit_states)
