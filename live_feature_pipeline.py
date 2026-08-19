"""
Live feature pipeline. Meant to run every hour (via GitHub Actions cron).

Pulls the last N hours of AQI + weather (enough to compute 48h lag features)
from Open-Meteo for every city, engineers the SAME features as the backfill
script, and writes the latest row(s) into the Hopsworks Feature Store.

Run:
    python live_feature_pipeline.py
"""

import time

import pandas as pd
import requests

from config import (
    CITIES,
    AIR_QUALITY_URL,
    WEATHER_FORECAST_URL,
    AIR_QUALITY_HOURLY_VARS,
    WEATHER_HOURLY_VARS,
    HOPSWORKS_PROJECT_NAME,
    HOPSWORKS_API_KEY,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)
from feature_engineering import engineer_features

# Need at least 48h of history in this pull so lag_48h can be computed
# for the newest row, plus a little buffer.
PAST_HOURS = 72


def fetch_air_quality_live(city: str, lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "past_hours": PAST_HOURS,
        "forecast_days": 1,
        "timezone": "auto",
    }
    r = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["city"] = city
    df = df.rename(columns={"time": "datetime"})
    return df


def fetch_weather_live(city: str, lat: float, lon: float) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "past_hours": PAST_HOURS,
        "forecast_days": 1,
        "timezone": "auto",
    }
    r = requests.get(WEATHER_FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["city"] = city
    df = df.rename(columns={"time": "datetime"})
    return df


def fetch_city_live(city: str, lat: float, lon: float) -> pd.DataFrame:
    aq = fetch_air_quality_live(city, lat, lon)
    wx = fetch_weather_live(city, lat, lon)
    merged = pd.merge(aq, wx, on=["datetime", "city"], how="inner")
    return merged


def push_to_hopsworks(df: pd.DataFrame):
    import hopsworks

    project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI + weather features per city (Open-Meteo)",
        primary_key=["city", "datetime"],
        event_time="datetime",
        online_enabled=False,
        time_travel_format="HUDI",
    )
    fg.insert(df, write_options={"wait_for_job": True})


def main():
    all_frames = []
    for city, (lat, lon) in CITIES.items():
        print(f"Fetching live data for {city}...")
        try:
            df = fetch_city_live(city, lat, lon)
            all_frames.append(df)
        except requests.HTTPError as e:
            print(f"  FAILED for {city}: {e}")
        time.sleep(1)

    raw = pd.concat(all_frames, ignore_index=True)
    featured = engineer_features(raw)

    # Only keep the most recent timestamp per city to insert into the
    # feature store — the older rows were only needed to compute lags.
    latest = (
        featured.sort_values("datetime")
        .groupby("city", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    print(f"Pushing {len(latest)} latest rows (one per city).")

    push_to_hopsworks(latest)
    print("Live feature pipeline complete.")


if __name__ == "__main__":
    main()