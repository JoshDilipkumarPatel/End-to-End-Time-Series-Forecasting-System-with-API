# Capstone Project Report

# End-to-End Time Series Forecasting System with REST API

## 1. Abstract

This project presents an end-to-end time series forecasting system designed to predict the next 8 weeks of sales for each state using historical sales data provided in an Excel file. The solution follows a production-style machine learning workflow: data ingestion, preprocessing, feature engineering, model training, validation, model comparison, artifact persistence, and forecast serving through a REST API.

The system trains and compares four forecasting approaches: SARIMA, Facebook Prophet, XGBoost, and LSTM. For every state, the best-performing model is selected automatically based on validation RMSE. The selected model artifacts are saved and later used by a FastAPI backend to serve predictions without retraining during API calls.

## 2. Problem Statement

The goal of this project is to forecast the next 8 weeks of sales for each state using historical sales data.

The forecasting system must:

- Handle missing dates and missing values.
- Capture trend and seasonality in sales data.
- Train multiple forecasting algorithms.
- Compare model performance using time-series validation.
- Select the best model automatically.
- Serve future predictions through a REST API.
- Be structured like a real backend service.

## 3. Dataset Description

The dataset is provided as an Excel file:

```text
data/Forecasting Case- Study.xlsx
```

The important columns used in this project are:

| Column | Description |
|---|---|
| `State` | State name |
| `Date` | Sales date |
| `Total` | Sales amount |
| `Category` | Product category |

For forecasting, the system uses:

- `State` as the grouping variable.
- `Date` as the time index.
- `Total` as the target variable.

The final processed dataset contains weekly sales series for 43 states.

## 4. Project Objectives

The main objectives are:

1. Build a complete forecasting pipeline.
2. Train SARIMA, Prophet, XGBoost, and LSTM models.
3. Engineer time-series features such as lags, rolling statistics, and calendar variables.
4. Use time-based train-validation splitting to avoid data leakage.
5. Automatically select the best model per state.
6. Save trained models and metadata.
7. Expose forecasts through a FastAPI REST API.

## 5. System Architecture

The project is divided into separate modules to keep the system maintainable and backend-ready.

```text
forecasting_system/
│
├── data/
│   └── Forecasting Case- Study.xlsx
│
├── saved_models/
│   ├── registry.json
│   └── <state_name>/
│       ├── metadata.json
│       ├── model.joblib or model.keras
│       └── preprocessor.joblib
│
├── config.py
├── data_processor.py
├── feature_engineer.py
├── models.py
├── pipeline.py
├── api.py
├── main.py
├── requirements.txt
└── README.md
```

### Architecture Flow

```text
Excel Dataset
     ↓
Data Loading and Cleaning
     ↓
Weekly Resampling and Missing Value Handling
     ↓
Feature Engineering
     ↓
Time-Based Train/Validation Split
     ↓
Train SARIMA, Prophet, XGBoost, LSTM
     ↓
Evaluate Using RMSE
     ↓
Select Best Model Per State
     ↓
Save Model Artifacts
     ↓
FastAPI Loads Saved Models
     ↓
Serve Next 8 Weeks Forecasts
```

## 6. Data Preprocessing

The preprocessing logic is implemented in:

```text
data_processor.py
```

The preprocessing steps include:

1. Load the Excel dataset using pandas.
2. Validate that required columns are present.
3. Rename columns into a standard internal format:
   - `State` becomes `state`
   - `Date` becomes `ds`
   - `Total` becomes `y`
4. Convert dates into datetime format.
5. Convert sales values into numeric format.
6. Drop unusable rows where state or date is missing.
7. Group data by state.
8. Resample each state into weekly frequency.
9. Handle missing sales values using:
   - time-based interpolation
   - forward fill
   - backward fill
   - fallback zero filling

This ensures every state has a complete weekly time series.

## 7. Feature Engineering

Feature engineering is implemented in:

```text
feature_engineer.py
```

The project creates the required time-series features.

### 7.1 Lag Features

Lag features represent previous sales values.

| Feature | Meaning |
|---|---|
| `lag_1` | Sales from previous week |
| `lag_7` | Sales from 7 weeks ago |
| `lag_30` | Sales from 30 weeks ago |

These features help models learn how past sales influence future sales.

### 7.2 Rolling Features

Rolling features summarize recent sales behavior.

| Feature | Meaning |
|---|---|
| `roll_mean_7` | Average sales over previous 7 weeks |
| `roll_std_7` | Sales variation over previous 7 weeks |
| `roll_mean_30` | Average sales over previous 30 weeks |
| `roll_std_30` | Sales variation over previous 30 weeks |

Rolling mean helps capture local trend, while rolling standard deviation helps capture volatility.

### 7.3 Calendar Features

Calendar features help models learn recurring seasonal patterns.

| Feature | Meaning |
|---|---|
| `day_of_week` | Day number of the week |
| `month` | Month of the year |
| `holiday` | Whether the week contains a US holiday |

### 7.4 Data Leakage Prevention

All lag and rolling features are created using only past sales values. The current week's sales are never used to predict the same week.

This prevents data leakage and makes validation realistic.

## 8. Train and Validation Split

The project uses a time-series validation strategy instead of random splitting.

For each state:

- Historical weeks before the validation period are used for training.
- The last 8 weeks are used for validation.

This approach is correct for forecasting because future data must not influence training.

```text
Past Data                     Future Data
Training Period               Validation Period
|----------------------------|----------------|
                              Last 8 weeks
```

## 9. Models Implemented

The model implementation is located in:

```text
models.py
```

Four mandatory models are implemented and compared.

## 9.1 SARIMA

SARIMA stands for Seasonal AutoRegressive Integrated Moving Average.

It is useful for time series data because it can model:

- trend
- autocorrelation
- seasonality

In this project, SARIMA is trained separately for each state.

## 9.2 Facebook Prophet

Prophet is a forecasting model designed for business time series.

It is useful because it handles:

- trend
- seasonality
- holidays
- missing observations

In this project, Prophet is trained using the `ds` and `y` columns and also includes US holiday information.

## 9.3 XGBoost

XGBoost is a gradient boosting machine learning model.

Since XGBoost is not a native time-series model, lag features, rolling statistics, and calendar features are created and used as input.

Features used by XGBoost include:

- `lag_1`
- `lag_7`
- `lag_30`
- `roll_mean_7`
- `roll_std_7`
- `roll_mean_30`
- `roll_std_30`
- `day_of_week`
- `month`
- `holiday`

For future predictions, XGBoost uses recursive forecasting. Each predicted week is added back into the history so the next week's features can be generated.

## 9.4 LSTM

LSTM stands for Long Short-Term Memory.

It is a deep learning model designed for sequential data. It can learn patterns from previous time steps and use them to forecast future values.

In this project:

- Sales values are scaled using MinMaxScaler.
- Sequences of historical sales are created.
- The LSTM model predicts future sales recursively.

## 10. Model Evaluation and Selection

The training and selection workflow is implemented in:

```text
pipeline.py
```

Each model is evaluated on the last 8 weeks of data for every state.

The main metric used is RMSE.

### RMSE

RMSE stands for Root Mean Squared Error.

It measures the average prediction error, with larger errors penalized more heavily.

The model with the lowest RMSE is selected as the best model for that state.

Example training output:

```text
Alabama: best=LSTM RMSE=8,419,652.77
Florida: best=XGBoost RMSE=40,898,324.86
Nebraska: best=SARIMA RMSE=2,453,637.43
Training complete. Saved 43 state model artifacts to saved_models.
```

This confirms that the system trained all state models and selected the best one automatically.

## 11. Model Persistence

After model selection, the best model for each state is saved in the `saved_models` folder.

The system saves:

- selected model artifact
- model metadata
- validation metrics
- feature column list
- last training date
- model registry

Example saved structure:

```text
saved_models/
├── registry.json
├── alabama/
│   ├── metadata.json
│   ├── model.keras
│   └── preprocessor.joblib
├── florida/
│   ├── metadata.json
│   └── model.joblib
└── nebraska/
    ├── metadata.json
    └── model.joblib
```

This is important because the API can load saved models directly instead of retraining during prediction.

## 12. REST API

The API is implemented using FastAPI in:

```text
api.py
```

The API exposes trained model forecasts through HTTP endpoints.

To start the API:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## 13. API Endpoints

### 13.1 Health Check

```http
GET /health
```

Purpose:

Checks whether the API is running and whether the model registry exists.

Example response:

```json
{
  "status": "ok",
  "registry_exists": true
}
```

### 13.2 List States

```http
GET /states
```

Purpose:

Returns all states for which trained models are available.

Example response:

```json
{
  "count": 43,
  "states": ["Alabama", "Arizona", "Arkansas"]
}
```

### 13.3 Model Metrics

```http
GET /models/metrics
```

Purpose:

Returns model comparison results, selected model, and metrics for every state.

### 13.4 Forecast Next 8 Weeks

```http
POST /forecast/next_8_weeks
```

Purpose:

Returns future sales forecasts.

Example request for one state:

```json
{
  "states": ["Alabama"],
  "horizon": 8
}
```

Example request for all states:

```json
{
  "states": null,
  "horizon": 8
}
```

Example response:

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

## 14. Backend Service Design

The project is structured like a backend service rather than a single notebook.

Important backend design choices:

- Code is split into separate modules.
- Configuration is centralized in `config.py`.
- Training and serving are separated.
- Models are saved as artifacts.
- API loads saved models instead of retraining.
- API includes health, metadata, and prediction endpoints.
- Request body validation is handled using Pydantic.
- Swagger documentation is generated automatically.

This makes the project easier to maintain, test, and deploy.

## 15. How to Run the Project

### Step 1: Create and Activate Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Train Models

```powershell
python pipeline.py
```

Expected final output:

```text
Training complete. Saved 43 state model artifacts to saved_models.
```

### Step 4: Start API

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

### Step 5: Open Swagger UI

```text
http://127.0.0.1:8000/docs
```

### Step 6: Test Forecast Endpoint

Use this request body:

```json
{
  "states": ["Alabama"],
  "horizon": 8
}
```

## 16. Results

The system successfully trained forecasting models for all 43 states.

Example selected models:

| State | Best Model | RMSE |
|---|---:|---:|
| Alabama | LSTM | 8,419,652.77 |
| Arizona | LSTM | 7,857,966.51 |
| Florida | XGBoost | 40,898,324.86 |
| Nebraska | SARIMA | 2,453,637.43 |
| Wyoming | LSTM | 510,976.72 |

The system generated 8-week forecasts through the API.

Example Alabama forecast:

| Week Ending | Forecast Sales |
|---|---:|
| 2023-12-16 | 192,653,562.99 |
| 2023-12-23 | 190,703,107.32 |
| 2023-12-30 | 188,772,632.25 |
| 2024-01-06 | 186,812,318.58 |
| 2024-01-13 | 184,869,702.45 |
| 2024-01-20 | 182,981,207.05 |
| 2024-01-27 | 181,173,042.27 |
| 2024-02-03 | 179,463,555.79 |

## 17. Challenges and Solutions

| Challenge | Solution |
|---|---|
| Missing dates | Weekly resampling and complete time index per state |
| Missing values | Interpolation, forward fill, backward fill |
| Data leakage risk | Lag and rolling features use only past values |
| Multiple states | Models trained independently per state |
| Different model types | Unified training and scoring pipeline |
| API prediction without retraining | Saved model artifacts loaded by FastAPI |
| XGBoost future forecasting | Recursive prediction with generated future features |

## 18. Limitations

The current system is suitable for a capstone-level backend forecasting service, but there are possible improvements:

- Hyperparameter tuning can be added for each model.
- More external features can be included, such as promotions, weather, or economic indicators.
- Docker can be added for containerized deployment.
- A database can be used instead of reading directly from Excel.
- Authentication can be added for production API access.
- Monitoring can be added to track prediction drift.

## 19. Future Scope

Future improvements may include:

- Automated scheduled retraining.
- Model versioning with MLflow.
- Deployment on cloud platforms.
- Dashboard for forecast visualization.
- Confidence intervals for forecasts.
- Batch forecast export to CSV or Excel.
- CI/CD pipeline for testing and deployment.

## 20. Conclusion

This project successfully implements an end-to-end time series forecasting system for state-wise sales prediction.

The solution starts from raw Excel data, performs preprocessing and feature engineering, trains four forecasting model families, evaluates them using time-series validation, selects the best model per state, saves the artifacts, and serves predictions through a FastAPI REST API.

The final system satisfies all assignment requirements and demonstrates a complete machine learning workflow designed in the style of a real backend service.

