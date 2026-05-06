import pandas as pd

from config import DATA_PATH, DATE_COLUMN, FREQUENCY, STATE_COLUMN, TARGET_COLUMN


def load_and_resample(data_path=DATA_PATH) -> pd.DataFrame:
    """Load sales data and return one complete weekly series per state."""
    raw = pd.read_excel(data_path)
    required = {STATE_COLUMN, DATE_COLUMN, TARGET_COLUMN}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = raw.rename(
        columns={STATE_COLUMN: "state", DATE_COLUMN: "ds", TARGET_COLUMN: "y"}
    )[["state", "ds", "y"]].copy()

    df["state"] = df["state"].astype(str).str.strip()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["state", "ds"])
    df = df[df["state"] != ""]

    weekly_parts = []
    for state, state_df in df.groupby("state", sort=True):
        state_df = state_df.sort_values("ds").set_index("ds")
        weekly = state_df["y"].resample(FREQUENCY).sum(min_count=1)
        weekly = weekly.asfreq(FREQUENCY)
        weekly = weekly.interpolate(method="time", limit_direction="both")
        weekly = weekly.ffill().bfill().fillna(0.0)

        weekly_parts.append(
            pd.DataFrame(
                {
                    "state": state,
                    "ds": weekly.index,
                    "y": weekly.astype(float).values,
                }
            )
        )

    if not weekly_parts:
        raise ValueError("No usable rows were found in the dataset.")

    out = pd.concat(weekly_parts, ignore_index=True)
    return out.sort_values(["state", "ds"]).reset_index(drop=True)

