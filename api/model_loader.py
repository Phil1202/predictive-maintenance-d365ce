# Hilfsmodul zum Laden der Modelle
import joblib
import tensorflow as tf
import json
from pathlib import Path

# Modellpfade
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

def load_scaler():
    return joblib.load(MODEL_DIR / "scaler" / "standard_scaler.pkl")

def load_xgboost_model():
    return joblib.load(MODEL_DIR / "classification" / "xgboost_with_ae.pkl")

def load_autoencoder_model():
    return tf.keras.models.load_model(MODEL_DIR / "anomaly_detection" / "autoencoder_model.h5")

def load_anomaly_threshold():
    with open(MODEL_DIR / "anomaly_detection" / "thresholds_anomaly.json") as f:
        return json.load(f)["autoencoder"]
