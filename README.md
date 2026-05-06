# End-to-End State Sales Forecasting System

Production-style time series forecasting service for weekly beverage sales by state. The system trains four model families per state, validates them with a time-based split, selects the lowest-RMSE model, persists the selected artifact, and serves the next 8 weeks through FastAPI.

## What It Implements

- Robust Excel ingestion from `data/Forecasting Case- Study.xlsx`
- Weekly state-level resampling with missing date and missing value handling
- Leakage-free feature engineering:
  - lags: `t-1`, `t-7`, `t-30`
  - rolling mean and standard deviation: `7`, `30`
  - day of week, month, US holiday flag
- Time series validation using the final 8 weeks per state
- Mandatory model comparison:
  - SARIMA
  - Prophet
  - XGBoost with engineered lag features
  - LSTM
- Automatic best-model selection per state using RMSE
- Saved model registry and per-state artifacts in `saved_models/`
- REST API for health checks, model metadata, states, and forecasts

## Project Structure

```text
api.py                FastAPI app for serving forecasts
config.py             Paths and forecasting constants
data_processor.py     Dataset loading, cleanup, weekly resampling, imputation
feature_engineer.py   Calendar, lag, rolling, and recursive forecast features
models.py             SARIMA, Prophet, XGBoost, and LSTM training/prediction code
pipeline.py           Train, validate, select, refit, and save artifacts
main.py               Convenience entry point for training and serving
requirements.txt      Python dependencies
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Train Models

```powershell
python pipeline.py
```

For a quick smoke test:

```powershell
python pipeline.py --limit-states 1
```

Training writes:

- `saved_models/registry.json`
- `saved_models/<state_slug>/metadata.json`
- `saved_models/<state_slug>/model.joblib` for SARIMA, Prophet, or XGBoost
- `saved_models/<state_slug>/model.keras` and `preprocessor.joblib` for LSTM

## Run API

```powershell
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive docs:

```text
http://localhost:8000/docs
```

## API Endpoints

- `GET /health`
- `GET /states`
- `GET /models/metrics`
- `POST /forecast/next_8_weeks`

Example request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/forecast/next_8_weeks `
  -ContentType "application/json" `
  -Body '{"states":["Alabama"],"horizon":8}'
```

Example response shape:

```json
{
  "status": "success",
  "horizon_weeks": 8,
  "count": 8,
  "results": [
    {
      "state": "Alabama",
      "best_model": "LSTM",
      "selection_rmse": 8419652.77,
      "forecast": [
        {
          "week_ending": "2023-12-16",
          "forecast_sales": 192653562.99
        }
      ]
    }
  ]
}
```

## Notes

TensorFlow may print native Windows CPU/GPU informational warnings. They do not indicate training failure. Prophet may also warn that Plotly is missing; interactive Prophet plots are not required by this backend service.
