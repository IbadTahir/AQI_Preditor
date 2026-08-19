"""
Feature engineering shared by BOTH the historical backfill pipeline and the
live feature pipeline. Keeping this in one place guarantees the model sees
the exact same feature schema at training time and at prediction time.
"""

import pandas as pd


# Columns that must always be treated as floating point, even if a given
# batch happens to contain only whole numbers (otherwise pandas infers int,
# Hopsworks locks the feature group schema to bigint on first insert, and a
# later batch with decimals breaks with a schema-mismatch error).
FLOAT_COLUMNS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
    "us_aqi_pm2_5",
    "us_aqi_pm10",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Force known numeric columns to float64 so the Hopsworks feature group
    schema never drifts between historical and live inserts."""
    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    return df


def add_time_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """Adds hour/day/month/weekday features from the datetime column."""
    dt = pd.to_datetime(df[datetime_col])
    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    group_col: str = "city",
    datetime_col: str = "datetime",
    target_col: str = "us_aqi",
) -> pd.DataFrame:
    """
    Adds lag features (aqi 24h/48h ago) and rolling averages, computed
    PER CITY and ALIGNED ON TIMESTAMP (not row position) so gaps in the
    data don't silently shift the lag.
    """
    df = df.sort_values([group_col, datetime_col]).copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    out_frames = []
    for city, g in df.groupby(group_col):
        g = g.set_index(datetime_col).sort_index()

        # Lag features: exact-hour lookup via reindex, not positional shift
        g["aqi_lag_24h"] = g[target_col].reindex(g.index - pd.Timedelta(hours=24)).values
        g["aqi_lag_48h"] = g[target_col].reindex(g.index - pd.Timedelta(hours=48)).values

        # Rolling averages (based on available past rows, min_periods to avoid NaN storms)
        g["aqi_rolling_mean_3h"] = g[target_col].rolling("3h", min_periods=1).mean()
        g["aqi_rolling_mean_24h"] = g[target_col].rolling("24h", min_periods=1).mean()

        # Derived: AQI change rate vs 24h ago
        g["aqi_change_rate_24h"] = g[target_col] - g["aqi_lag_24h"]

        out_frames.append(g.reset_index())

    return pd.concat(out_frames, ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single entry point both pipelines call, so nothing drifts apart."""
    df = enforce_dtypes(df)
    df = add_time_features(df)
    df = add_lag_and_rolling_features(df)
    df = enforce_dtypes(df)  # lag/rolling features derived from float cols are already float, but re-assert to be safe
    return df