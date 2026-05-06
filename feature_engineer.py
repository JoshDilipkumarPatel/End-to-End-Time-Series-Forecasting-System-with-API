from datetime import timedelta

import holidays
import pandas as pd

from config import LAGS, ROLLING_WINDOWS, VAL_WEEKS

US_HOLIDAYS = holidays.US()


def calendar_features(ds: pd.Timestamp) -> dict:
    week_start = ds - timedelta(days=6)
    has_holiday = any(day in US_HOLIDAYS for day in pd.date_range(week_start, ds, freq="D"))
    return {
        "day_of_week": ds.dayofweek,
        "month": ds.month,
        "holiday": int(has_holiday),
    }


def feature_columns() -> list[str]:
    lag_cols = [f"lag_{lag}" for lag in LAGS]
    rolling_cols = [
        col
        for window in ROLLING_WINDOWS
        for col in (f"roll_mean_{window}", f"roll_std_{window}")
    ]
    return ["day_of_week", "month", "holiday", *lag_cols, *rolling_cols]


def add_features(df: pd.DataFrame, drop_missing: bool = True) -> pd.DataFrame:
    """Create leakage-free lag, rolling, and calendar features."""
    out = df.sort_values(["state", "ds"]).copy()
    out["day_of_week"] = out["ds"].dt.dayofweek
    out["month"] = out["ds"].dt.month
    out["holiday"] = out["ds"].apply(lambda ds: calendar_features(ds)["holiday"])

    grouped_y = out.groupby("state", group_keys=False)["y"]
    for lag in LAGS:
        out[f"lag_{lag}"] = grouped_y.shift(lag)

    for window in ROLLING_WINDOWS:
        shifted = grouped_y.shift(1)
        out[f"roll_mean_{window}"] = shifted.groupby(out["state"]).rolling(window).mean().reset_index(level=0, drop=True)
        out[f"roll_std_{window}"] = shifted.groupby(out["state"]).rolling(window).std().reset_index(level=0, drop=True)

    if drop_missing:
        out = out.dropna(subset=feature_columns())
    return out.reset_index(drop=True)


def add_validation_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    max_dates = out.groupby("state")["ds"].transform("max")
    out["is_val"] = out["ds"] > (max_dates - pd.Timedelta(weeks=VAL_WEEKS))
    return out


def make_recursive_feature_row(state: str, ds: pd.Timestamp, history: pd.DataFrame) -> pd.DataFrame:
    y_history = history.sort_values("ds")["y"].astype(float).reset_index(drop=True)
    row = {"state": state, "ds": ds, **calendar_features(ds)}

    for lag in LAGS:
        row[f"lag_{lag}"] = y_history.iloc[-lag] if len(y_history) >= lag else y_history.iloc[0]

    for window in ROLLING_WINDOWS:
        window_values = y_history.tail(window)
        row[f"roll_mean_{window}"] = float(window_values.mean())
        row[f"roll_std_{window}"] = float(window_values.std(ddof=1) if len(window_values) > 1 else 0.0)

    return pd.DataFrame([row])

