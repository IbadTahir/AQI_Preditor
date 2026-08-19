"""
Historical backfill pipeline.

Pulls hourly air-quality (pollutants + real US AQI) and weather data from
Open-Meteo for every city in config.CITIES, over a given date range, merges
them, engineers features, and pushes the result into the Hopsworks Feature
Store.

Run:
    python historical_backfill.py --start 2025-08-01 --end 2026-08-06
"""

import argparse
import time

import pandas as pd
import requests

from config import (
    CITIES,
    AIR_QUALITY_URL,
    WEATHER_ARCHIVE_URL,
    AIR_QUALITY_HOURLY_VARS,
    WEATHER_HOURLY_VARS,
    HOPSWORKS_PROJECT_NAME,
    HOPSWORKS_API_KEY,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)
from feature_engineering import engineer_features


def fetch_air_quality(city: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "start_date": start,
        "end_date": end,
        "timezone": "auto",
    }
    r = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["city"] = city
    df = df.rename(columns={"time": "datetime"})
    return df


def fetch_weather(city: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "start_date": start,
        "end_date": end,
        "timezone": "auto",
    }
    r = requests.get(WEATHER_ARCHIVE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["city"] = city
    df = df.rename(columns={"time": "datetime"})
    return df


def backfill_city(city: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    aq = fetch_air_quality(city, lat, lon, start, end)
    wx = fetch_weather(city, lat, lon, start, end)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Skip Hopsworks push, save CSV instead")
    args = parser.parse_args()

    all_frames = []
    for city, (lat, lon) in CITIES.items():
        print(f"Fetching {city}...")
        try:
            df = backfill_city(city, lat, lon, args.start, args.end)
            all_frames.append(df)
        except requests.HTTPError as e:
            print(f"  FAILED for {city}: {e}")
        time.sleep(1)  # be polite to the free API

    raw = pd.concat(all_frames, ignore_index=True)
    print(f"Raw rows: {len(raw)}")

    featured = engineer_features(raw)
    print(f"Featured rows: {len(featured)}, columns: {list(featured.columns)}")

    if args.dry_run:
        featured.to_csv("historical_features.csv", index=False)
        print("Saved to historical_features.csv (dry run, not pushed to Hopsworks)")
    else:
        push_to_hopsworks(featured)
        print("Pushed to Hopsworks feature group.")


if __name__ == "__main__":
    main()