"""
Training pipeline.

1. Pulls all historical features from the aqi_features feature group.
2. Builds 3-day-ahead targets (t+24h, t+48h, t+72h) per city.
3. Time-based train/test split (no shuffling - this is forecasting).
4. Trains Ridge, Random Forest, and a small neural net (multi-output:
   predicts all 3 horizons at once).
5. Evaluates each with RMSE / MAE / R^2 per horizon.
6. Saves the best model + its metrics to the Hopsworks Model Registry.

Run:
    python training_pipeline.py
"""

import os
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import (
    HOPSWORKS_PROJECT_NAME,
    HOPSWORKS_API_KEY,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)

HORIZONS = [24, 48, 72]  # hours ahead to predict
TARGET_COL = "us_aqi"
TEST_FRACTION = 0.15  # last 15% of the timeline held out as test

# Columns that should never be used as model inputs
NON_FEATURE_COLS = ["datetime", "city"] + [f"target_{h}h" for h in HORIZONS]


def connect_feature_store():
    import hopsworks

    project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
    return project, project.get_feature_store()


def load_data(fs) -> pd.DataFrame:
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = (
        df.sort_values(["city", "datetime"])
        .drop_duplicates(subset=["city", "datetime"], keep="last")
        .reset_index(drop=True)
    )
    print(f"Training data range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Cities: {df['city'].nunique()} | Unique city/timestamp rows: {len(df)}")
    print("Rows by city:")
    print(df.groupby("city")["datetime"].agg(["min", "max", "count"]).to_string())
    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Adds target_{h}h columns: us_aqi exactly h hours ahead, per city,
    aligned by timestamp (not row position)."""
    out_frames = []
    for city, g in df.groupby("city"):
        g = g.set_index("datetime").sort_index()
        for h in HORIZONS:
            g[f"target_{h}h"] = g[TARGET_COL].reindex(g.index + pd.Timedelta(hours=h)).values
        out_frames.append(g.reset_index())
    return pd.concat(out_frames, ignore_index=True)


def prepare_dataset(df: pd.DataFrame):
    """One-hot encodes city, drops rows with missing lag/target features,
    and returns (X, y, feature_names, datetime_index)."""
    df = pd.get_dummies(df, columns=["city"], prefix="city")

    target_cols = [f"target_{h}h" for h in HORIZONS]
    required_cols = ["aqi_lag_24h", "aqi_lag_48h"] + target_cols
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    feature_cols = [
        c for c in df.columns
        if c not in ["datetime"] + target_cols
    ]

    X = df[feature_cols].astype("float64")
    y = df[target_cols].astype("float64")
    dt = df["datetime"]
    return X, y, feature_cols, dt


def time_based_split(X, y, dt):
    cutoff = dt.quantile(1 - TEST_FRACTION)
    train_mask = dt < cutoff
    test_mask = ~train_mask
    return (
        X[train_mask].reset_index(drop=True),
        X[test_mask].reset_index(drop=True),
        y[train_mask].reset_index(drop=True),
        y[test_mask].reset_index(drop=True),
        cutoff,
    )


def evaluate(y_true: pd.DataFrame, y_pred: np.ndarray, model_name: str) -> dict:
    metrics = {"model": model_name}
    for i, h in enumerate(HORIZONS):
        col = f"target_{h}h"
        rmse = mean_squared_error(y_true[col], y_pred[:, i]) ** 0.5
        mae = mean_absolute_error(y_true[col], y_pred[:, i])
        r2 = r2_score(y_true[col], y_pred[:, i])
        metrics[f"rmse_{h}h"] = round(rmse, 3)
        metrics[f"mae_{h}h"] = round(mae, 3)
        metrics[f"r2_{h}h"] = round(r2, 3)
        print(f"  [{model_name}] {h}h ahead -> RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
    metrics["avg_rmse"] = round(np.mean([metrics[f"rmse_{h}h"] for h in HORIZONS]), 3)
    return metrics


def train_ridge(X_train, y_train):
    model = MultiOutputRegressor(Ridge(alpha=1.0))
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    )
    model.fit(X_train, y_train)
    return model


def train_neural_net(X_train, y_train, X_test, y_test):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Scale inputs - NNs are sensitive to feature scale, tree/linear models aren't
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(HORIZONS)),  # one output per horizon
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=256,
        callbacks=[early_stop],
        verbose=0,
    )
    return model, scaler


def save_model_to_registry(project, model, model_name, metrics, feature_names, extra_files=None):
    mr = project.get_model_registry()

    tmp_dir = f"model_export_{model_name}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    if extra_files:  # Keras model: save via its own format + scaler
        model.save(os.path.join(tmp_dir, "model.keras"))
        joblib.dump(extra_files["scaler"], os.path.join(tmp_dir, "scaler.pkl"))
    else:  # sklearn model
        joblib.dump(model, os.path.join(tmp_dir, "model.pkl"))

    joblib.dump(feature_names, os.path.join(tmp_dir, "feature_names.pkl"))

    hw_model = mr.python.create_model(
        name=f"aqi_forecast_{model_name}",
        metrics={"avg_rmse": metrics["avg_rmse"]},
        description=f"AQI 24/48/72h forecast model ({model_name})",
    )
    hw_model.save(tmp_dir)
    shutil.rmtree(tmp_dir)
    print(f"Saved '{model_name}' to Model Registry with avg_rmse={metrics['avg_rmse']}")


def main():
    project, fs = connect_feature_store()

    print("Loading data from feature store...")
    df = load_data(fs)
    print(f"Loaded {len(df)} rows.")

    print("Building multi-horizon targets...")
    df = build_targets(df)

    X, y, feature_names, dt = prepare_dataset(df)
    print(f"Usable rows after dropping NaN lag/target rows: {len(X)}")

    X_train, X_test, y_train, y_test, cutoff = time_based_split(X, y, dt)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | Split cutoff: {cutoff}")

    results = []

    print("\nTraining Ridge Regression...")
    ridge = train_ridge(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    results.append((evaluate(y_test, ridge_pred, "ridge"), ridge, None))

    print("\nTraining Random Forest...")
    rf = train_random_forest(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results.append((evaluate(y_test, rf_pred, "random_forest"), rf, None))

    print("\nTraining Neural Network...")
    nn, scaler = train_neural_net(X_train, y_train, X_test, y_test)
    nn_pred = nn.predict(scaler.transform(X_test), verbose=0)
    results.append((evaluate(y_test, nn_pred, "neural_net"), nn, scaler))

    # Pick the best by lowest average RMSE across horizons
    best_metrics, best_model, best_scaler = min(results, key=lambda r: r[0]["avg_rmse"])
    print(f"\nBest model: {best_metrics['model']} (avg_rmse={best_metrics['avg_rmse']})")

    print("\nSaving best model to Hopsworks Model Registry...")
    extra_files = {"scaler": best_scaler} if best_scaler is not None else None
    save_model_to_registry(project, best_model, best_metrics["model"], best_metrics, feature_names, extra_files)

    print("\nAll model results:")
    for m, _, _ in results:
        print(f"  {m['model']}: avg_rmse={m['avg_rmse']}")


if __name__ == "__main__":
    main()