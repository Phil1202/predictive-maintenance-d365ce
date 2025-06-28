from fastapi import FastAPI, HTTPException
from api.schema import PredictionInput
from api.model_loader import load_scaler, load_xgboost_model, load_autoencoder_model, load_anomaly_threshold
from api.constants import FEATURE_NAMES
import numpy as np
import pandas as pd

app = FastAPI(
    title="Predictive Maintenance API",
    version="1.0.0",
    description="Stellt Vorhersagen für Klassifikation und Anomalieerkennung bereit"
)

# Modellkomponenten laden
scaler = load_scaler()
xgb_model = load_xgboost_model()
autoencoder = load_autoencoder_model()
threshold = load_anomaly_threshold()

@app.post("/predict")
def predict_failure(input: PredictionInput):
    try:
        # Feature-Vektor mit Spaltennamen
        X_df = pd.DataFrame([input.data], columns=FEATURE_NAMES)
        X_scaled = scaler.transform(X_df)

        # Klassifikationsmodell aufrufen
        prediction = xgb_model.predict_proba(X_scaled)[0][1]  # Wahrscheinlichkeit für "Ausfall"
        predicted_class = int(xgb_model.predict(X_scaled)[0])  # 0 oder 1

        return {
            "model": "XGBoost",
            "failure_probability": float(prediction),
            "predicted_class": predicted_class
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/anomaly")
def detect_anomaly(input: PredictionInput):
    try:
        # Feature-Vektor mit Spaltennamen
        X_df = pd.DataFrame([input.data], columns=FEATURE_NAMES)
        X_scaled = scaler.transform(X_df)
        X_pred = autoencoder.predict(X_scaled, verbose=0)
        reconstruction_error = np.mean((X_scaled - X_pred) ** 2)
        is_anomaly = reconstruction_error > threshold

        return {
            "model": "Autoencoder",
            "reconstruction_error": float(reconstruction_error),
            "is_anomaly": bool(is_anomaly)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
