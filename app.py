"""
Phase 5 dashboard (Streamlit).

Loads latest engineered features from Hopsworks, loads the best available model
artifact from Hopsworks Model Registry, and displays:
1) Current AQI
2) 24h / 48h / 72h forecast
3) Historical AQI trend
4) Hazard alert banner for unhealthy predictions

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    CITIES,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
)

HORIZONS = [24, 48, 72]
TARGET_COL = "us_aqi"
MODEL_CANDIDATES = [
    "aqi_forecast_random_forest",
]
MODEL_NAME = "aqi_forecast_random_forest"
MODEL_VERSION = 3


@st.cache_resource(show_spinner=False)
def connect_hopsworks():
    import hopsworks

    project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    return project, fs


@st.cache_data(show_spinner=False)
def load_feature_data() -> pd.DataFrame:
    _, fs = connect_hopsworks()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["city", "datetime"]).reset_index(drop=True)


def _best_registered_model(project, model_name: str):
    """Get the confirmed model version from the Hopsworks registry."""
    mr = project.get_model_registry()
    model_ref = mr.get_model(name=model_name, version=MODEL_VERSION)
    if model_ref is None:
        raise RuntimeError(f"Model '{model_name}' version {MODEL_VERSION} was not found.")
    return model_ref


@st.cache_resource(show_spinner=False)
def load_best_model_assets():
    project, _ = connect_hopsworks()

    # Prefer latest model trained by Phase 3 by comparing avg_rmse metadata
    # when available; otherwise fall back to first downloadable candidate.
    best = None
    registry_errors = []

    for name in MODEL_CANDIDATES:
        try:
            model_ref = _best_registered_model(project, name)
            candidate = {
                "name": name,
                "model_ref": model_ref,
            }

            if best is None:
                best = candidate
        except Exception as exc:
            registry_errors.append(f"{name}: {exc}")
            continue

    if best is None:
        raise RuntimeError(
            "No registered model could be downloaded. "
            "Expected one of: " + ", ".join(MODEL_CANDIDATES) + ". "
            "Registry details: " + " | ".join(registry_errors)
        )

    model_dir = Path(best["model_ref"].download())
    feature_names_path = model_dir / "feature_names.pkl"
    if not feature_names_path.exists():
        raise RuntimeError(f"feature_names.pkl not found in {model_dir}")
    feature_names = joblib.load(feature_names_path)

    model_pkl = model_dir / "model.pkl"
    model_keras = model_dir / "model.keras"
    scaler_path = model_dir / "scaler.pkl"

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    if model_pkl.exists():
        model = joblib.load(model_pkl)
        model_type = "sklearn"
    elif model_keras.exists():
        from tensorflow.keras.models import load_model

        model = load_model(model_keras)
        model_type = "keras"
    else:
        raise RuntimeError(f"No model artifact found in {model_dir} (model.pkl/model.keras missing).")

    return {
        "model": model,
        "model_type": model_type,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": best["name"],
    }


def build_inference_row(latest_row: pd.Series, feature_names: list[str], city: str) -> pd.DataFrame:
    payload = {}

    for col in feature_names:
        if col in latest_row.index:
            payload[col] = latest_row[col]
        elif col.startswith("city_"):
            payload[col] = 1.0 if col == f"city_{city}" else 0.0
        else:
            payload[col] = 0.0

    X = pd.DataFrame([payload], columns=feature_names)
    return X.astype("float64")


def predict_horizons(model_assets: dict, X: pd.DataFrame) -> np.ndarray:
    model = model_assets["model"]
    model_type = model_assets["model_type"]
    scaler = model_assets["scaler"]

    if model_type == "keras":
        X_in = scaler.transform(X) if scaler is not None else X.values
        preds = model.predict(X_in, verbose=0)
    else:
        preds = model.predict(X)

    preds = np.array(preds).reshape(-1)
    return preds


def aqi_band(aqi_value: float) -> str:
    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Moderate"
    if aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi_value <= 200:
        return "Unhealthy"
    if aqi_value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def render_alert(preds: np.ndarray):
    max_pred = float(np.max(preds))
    band = aqi_band(max_pred)

    if max_pred > 150:
        st.error(f"Hazard Alert: Forecast reaches {max_pred:.1f} AQI ({band}).")
    elif max_pred > 100:
        st.warning(f"Caution: Forecast reaches {max_pred:.1f} AQI ({band}).")
    else:
        st.success(f"Air quality outlook remains at lower-risk levels (max {max_pred:.1f}, {band}).")


def main():
    st.set_page_config(page_title="AQI Forecast Dashboard", layout="wide")
    st.title("AQI Forecast Dashboard")
    st.caption("Current AQI + 3-day forecast (24h / 48h / 72h) from your trained model")

    city_options = list(CITIES.keys())
    selected_city = st.selectbox("Select City", city_options, index=0)

    with st.spinner("Loading feature data and model from Hopsworks..."):
        df = load_feature_data()
        model_assets = load_best_model_assets()

    city_df = df[df["city"] == selected_city].sort_values("datetime").copy()
    if city_df.empty:
        st.error(f"No feature rows found for {selected_city} in feature group.")
        return

    latest = city_df.iloc[-1]
    current_aqi = float(latest[TARGET_COL])
    current_time = pd.to_datetime(latest["datetime"])

    X = build_inference_row(latest, model_assets["feature_names"], selected_city)
    preds = predict_horizons(model_assets, X)

    # Keep forecast non-negative and cast for display
    preds = np.maximum(preds, 0.0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current AQI", f"{current_aqi:.1f}", help=f"Latest timestamp: {current_time}")
    col2.metric("24h Forecast", f"{preds[0]:.1f}")
    col3.metric("72h Forecast", f"{preds[2]:.1f}")

    st.caption(f"Model in use: {model_assets['model_name']}")
    render_alert(preds)

    forecast_df = pd.DataFrame(
        {
            "datetime": [current_time + timedelta(hours=h) for h in HORIZONS],
            "forecast_aqi": preds,
            "horizon": [f"+{h}h" for h in HORIZONS],
        }
    )

    hist_window = city_df[city_df["datetime"] >= (current_time - pd.Timedelta(days=7))]

    st.subheader("3-Day Forecast")
    fig_forecast = px.line(
        forecast_df,
        x="datetime",
        y="forecast_aqi",
        markers=True,
        text="horizon",
        labels={"forecast_aqi": "Predicted AQI", "datetime": "Timestamp"},
    )
    fig_forecast.update_traces(textposition="top center")
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("Recent Historical AQI (Last 7 Days)")
    fig_hist = px.line(
        hist_window,
        x="datetime",
        y=TARGET_COL,
        labels={TARGET_COL: "Observed AQI", "datetime": "Timestamp"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("Latest feature row used for prediction"):
        show_cols = [
            "datetime",
            "city",
            "us_aqi",
            "aqi_lag_24h",
            "aqi_lag_48h",
            "aqi_rolling_mean_3h",
            "aqi_rolling_mean_24h",
            "aqi_change_rate_24h",
        ]
        available = [c for c in show_cols if c in latest.index]
        st.dataframe(pd.DataFrame([latest[available]]), use_container_width=True)


if __name__ == "__main__":
    if os.environ.get("AQI_STREAMLIT_APP") != "1":
        # Allow `python app.py` to start the browser dashboard directly.
        child_env = os.environ.copy()
        child_env["AQI_STREAMLIT_APP"] = "1"
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__), *sys.argv[1:]],
                env=child_env,
            )
        )

    try:
        main()
    except Exception as exc:
        st.error(f"Dashboard failed to load: {exc}")
