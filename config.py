"""
Shared configuration for the AQI Predictor project.
Keep this identical across the backfill and live pipelines so features
always line up.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# City -> (lat, lon), pulled from your existing dataset
CITIES = {
    "Karachi":    (24.8607, 67.0011),
    "Lahore":     (31.5497, 74.3436),
    "Islamabad":  (33.6844, 73.0479),
    "Rawalpindi": (33.5651, 73.0169),
    "Faisalabad": (31.4504, 73.1350),
    "Multan":     (30.1575, 71.5249),
    "Peshawar":   (34.0151, 71.5249),
    "Quetta":     (30.1798, 66.9750),
    "Sialkot":    (32.4945, 74.5229),
    "Hyderabad":  (25.3960, 68.3578),
    "Gujranwala": (32.1877, 74.1945),
    "Bahawalpur": (29.3956, 71.6836),
    "Sargodha":   (32.0836, 72.6711),
    "Sukkur":     (27.7052, 68.8574),
}

# Open-Meteo endpoints (no API key needed for non-commercial use)
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Pollutant + AQI variables (identical for historical and live calls)
AIR_QUALITY_HOURLY_VARS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",          # <- this is the real 0-500 target you'll predict
    "us_aqi_pm2_5",
    "us_aqi_pm10",
]

# Weather variables (identical for historical and live calls)
WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]

# Hopsworks
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_forecast_pakistan")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 2