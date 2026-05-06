from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Forecasting Case- Study.xlsx"
MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATE_COLUMN = "Date"
STATE_COLUMN = "State"
TARGET_COLUMN = "Total"

FORECAST_HORIZON = 8
VAL_WEEKS = 8
FREQUENCY = "W-SAT"

LAGS = [1, 7, 30]
ROLLING_WINDOWS = [7, 30]
RANDOM_STATE = 42
SELECTION_METRIC = "RMSE"
