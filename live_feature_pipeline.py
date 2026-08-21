"""
Live feature pipeline. Meant to run every hour (via GitHub Actions cron).

Pulls the last N hours of AQI + weather (enough to compute 48h lag features)
from Open-Meteo for every city, engineers the SAME features as the backfill
script, and writes the latest row(s) into the Hopsworks Feature Store.

Run:
    python live_feature_pipeline.py
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
REQUEST_TIMEOUT = (10, 45)
HOPSWORKS_INSERT_ATTEMPTS = 3


def build_retry_session() -> requests.Session:
    """Retry transient HTTP failures commonly seen in CI runners."""
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_air_quality_live(city: str, lat: float, lon: float, session: requests.Session) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "past_hours": PAST_HOURS,
        "forecast_days": 1,
        "timezone": "auto",
    }
    r = session.get(AIR_QUALITY_URL, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["city"] = city
    df = df.rename(columns={"time": "datetime"})
    return df


def fetch_weather_live(city: str, lat: float, lon: float, session: requests.Session) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "past_hours": PAST_HOURS,
        "forecast_days": 1,
        "timezone": "auto",
    }
    r = session.get(WEATHER_FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["city"] = city
    df = df.rename(columns={"time": "datetime"})
    return df


def fetch_city_live(city: str, lat: float, lon: float, session: requests.Session) -> pd.DataFrame:
    aq = fetch_air_quality_live(city, lat, lon, session)
    wx = fetch_weather_live(city, lat, lon, session)
    merged = pd.merge(aq, wx, on=["datetime", "city"], how="inner")
    return merged


def push_to_hopsworks(df: pd.DataFrame):
    import hopsworks

    for attempt in range(1, HOPSWORKS_INSERT_ATTEMPTS + 1):
        try:
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
            return
        except (requests.RequestException, ConnectionError) as exc:
            if attempt == HOPSWORKS_INSERT_ATTEMPTS:
                raise
            delay = 2 ** attempt
            print(f"Hopsworks insert attempt {attempt} failed: {exc}. Retrying in {delay}s...")
            time.sleep(delay)


def main():
    all_frames = []
    session = build_retry_session()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(fetch_city_live, city, lat, lon, session): city
            for city, (lat, lon) in CITIES.items()
        }
        for future in as_completed(futures):
            city = futures[future]
            print(f"Fetching live data for {city}...")
            try:
                all_frames.append(future.result())
            except requests.RequestException as e:
                print(f"  FAILED for {city}: {e}")

    if not all_frames:
        raise RuntimeError("No city data fetched successfully. Check API availability/network in runner.")

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