import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import xgboost as xgb
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import set_random_seed

from config import FREQUENCY, RANDOM_STATE

warnings.filterwarnings("ignore")


def clip_forecast(values) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def fit_sarima(train_y: pd.Series):
    model = SARIMAX(
        train_y.astype(float),
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 13),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=75)


def predict_sarima(model, steps: int) -> np.ndarray:
    return clip_forecast(model.get_forecast(steps=steps).predicted_mean.values)


def fit_prophet(train_df: pd.DataFrame):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    model.add_country_holidays(country_name="US")
    model.fit(train_df[["ds", "y"]])
    return model


def predict_prophet(model, steps: int) -> np.ndarray:
    future = model.make_future_dataframe(periods=steps, freq=FREQUENCY, include_history=True)
    forecast = model.predict(future).tail(steps)["yhat"].values
    return clip_forecast(forecast)


def fit_xgboost(train_df: pd.DataFrame, feature_cols: list[str]):
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(train_df[feature_cols], train_df["y"])
    return model


def predict_xgboost(model, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return clip_forecast(model.predict(frame[feature_cols]))


def fit_lstm(train_y: pd.Series, seq_len: int = 12, epochs: int = 12):
    values = train_y.astype(float).values.reshape(-1, 1)
    if len(values) <= seq_len + 1:
        raise ValueError("Not enough observations to train LSTM.")

    set_random_seed(RANDOM_STATE)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).ravel()

    x_seq, y_seq = [], []
    for i in range(len(scaled) - seq_len):
        x_seq.append(scaled[i : i + seq_len])
        y_seq.append(scaled[i + seq_len])

    x_seq = np.asarray(x_seq).reshape(-1, seq_len, 1)
    y_seq = np.asarray(y_seq)

    model = Sequential(
        [
            LSTM(48, input_shape=(seq_len, 1)),
            Dense(24, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_seq, y_seq, epochs=epochs, batch_size=16, verbose=0)
    return {"model": model, "scaler": scaler, "seq_len": seq_len}


def predict_lstm(bundle: dict, steps: int) -> np.ndarray:
    model = bundle["model"]
    scaler = bundle["scaler"]
    seq_len = bundle["seq_len"]

    scaled = scaler.transform(bundle["history"].reshape(-1, 1)).ravel()
    last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
    preds = []
    for _ in range(steps):
        pred = float(model.predict(last_seq, verbose=0)[0, 0])
        preds.append(pred)
        last_seq = np.roll(last_seq, -1, axis=1)
        last_seq[0, -1, 0] = pred

    return clip_forecast(scaler.inverse_transform(np.asarray(preds).reshape(-1, 1)).ravel())
