import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

from config import FORECAST_HORIZON, FREQUENCY, MODEL_DIR
from data_processor import load_and_resample
from feature_engineer import make_recursive_feature_row
from models import predict_lstm, predict_prophet, predict_sarima
from pipeline import slugify

app = FastAPI(
    title="State Sales Forecasting API",
    version="1.0.0",
    description="Serves the selected best model per state for the next 8 weeks of sales.",
)


class ForecastRequest(BaseModel):
    states: Optional[list[str]] = Field(default=None, description="State names. Omit to forecast every trained state.")
    horizon: int = Field(default=FORECAST_HORIZON, ge=1, le=52)


def registry_path() -> Path:
    return MODEL_DIR / "registry.json"


def load_registry() -> dict:
    path = registry_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="No trained model registry found. Run `python pipeline.py` first.")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_metadata(state: str) -> dict:
    meta_path = MODEL_DIR / slugify(state) / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"No trained artifact found for state: {state}")
    with meta_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_trained_model(state: str, metadata: dict):
    state_dir = MODEL_DIR / slugify(state)
    model_name = metadata["best_model"]
    if model_name == "LSTM":
        keras_model = load_model(state_dir / "model.keras", compile=False)
        preprocessor = joblib.load(state_dir / "preprocessor.joblib")
        return {"model": keras_model, **preprocessor}
    return joblib.load(state_dir / "model.joblib")


def predict_xgboost_recursive(model, state: str, history: pd.DataFrame, dates: pd.DatetimeIndex, feature_cols: list[str]) -> list[float]:
    history = history[["ds", "y"]].sort_values("ds").copy()
    preds = []
    for ds in dates:
        feature_row = make_recursive_feature_row(state, ds, history)
        pred = max(float(model.predict(feature_row[feature_cols])[0]), 0.0)
        preds.append(pred)
        history = pd.concat(
            [history, pd.DataFrame([{"ds": ds, "y": pred}])],
            ignore_index=True,
        )
    return preds


def forecast_state(state: str, horizon: int, data: pd.DataFrame) -> dict:
    metadata = load_metadata(state)
    model = load_trained_model(state, metadata)
    model_name = metadata["best_model"]

    last_train_date = pd.Timestamp(metadata["last_train_date"])
    history = data[(data["state"] == state) & (data["ds"] <= last_train_date)].copy()
    if history.empty:
        raise HTTPException(status_code=404, detail=f"No historical data found for state: {state}")

    dates = pd.date_range(start=last_train_date + pd.Timedelta(weeks=1), periods=horizon, freq=FREQUENCY)

    if model_name == "SARIMA":
        preds = predict_sarima(model, horizon)
    elif model_name == "Prophet":
        preds = predict_prophet(model, horizon)
    elif model_name == "XGBoost":
        preds = predict_xgboost_recursive(model, state, history, dates, metadata["feature_columns"])
    elif model_name == "LSTM":
        preds = predict_lstm({**model, "history": history["y"].astype(float).values}, horizon)
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported saved model: {model_name}")

    return {
        "state": state,
        "best_model": model_name,
        "selection_rmse": metadata["metrics"][model_name]["metrics"]["RMSE"],
        "forecast": [
            {
                "week_ending": ds.strftime("%Y-%m-%d"),
                "forecast_sales": round(float(pred), 2),
            }
            for ds, pred in zip(dates, preds)
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "registry_exists": registry_path().exists()}


@app.get("/states")
def states():
    registry = load_registry()
    return {"count": len(registry["states"]), "states": sorted(registry["states"].keys())}


@app.get("/models/metrics")
def model_metrics():
    registry = load_registry()
    return registry


@app.post("/forecast/next_8_weeks")
def forecast(req: ForecastRequest):
    registry = load_registry()
    trained_states = set(registry["states"])
    requested_states = req.states or sorted(trained_states)
    missing = sorted(set(requested_states).difference(trained_states))
    if missing:
        raise HTTPException(status_code=404, detail=f"States are not trained: {missing}")

    data = load_and_resample()
    results = [forecast_state(state, req.horizon, data) for state in requested_states]
    return {
        "status": "success",
        "horizon_weeks": req.horizon,
        "count": sum(len(item["forecast"]) for item in results),
        "results": results,
    }

