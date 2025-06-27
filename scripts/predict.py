# === Importe ===
import argparse
import pandas as pd
import numpy as np
import joblib
import json
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# === 0. Scaler auf Basis von vollständigem X_train fitten und speichern ===
# Es stellt sicher, dass das Skalierungsmodell dieselbe Feature-Struktur nutzt wie die trainierten Modelle.
X_train = pd.read_pickle("data/X_train.pkl")  # Eingabedaten mit allen Features (inkl. Type_H und active_errors)
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, "models/scaler/standard_scaler.pkl")
print("Neuer Scaler gespeichert mit allen 12 Features.")

# === 1. Argumentparser definieren ===
def parse_args():
    parser = argparse.ArgumentParser(description="Predictive Maintenance - Vorhersage-Skript")
    parser.add_argument('--input', type=str, required=True, help='Pfad zur Eingabedatei (.pkl, .csv, .json)')
    parser.add_argument('--mode', type=str, choices=['classification', 'anomaly'], required=True, help='Anwendungsmodus: Klassifikation oder Anomalieerkennung')
    return parser.parse_args()

# === 2. Modelle und Konfigurationen laden ===
def load_model(mode):
    """Lädt das passende Modell je nach Modus (Klassifikation oder Anomalieerkennung)"""
    if mode == 'classification':
        return joblib.load('models/classification/xgboost_with_ae.pkl')
    elif mode == 'anomaly':
        return tf.keras.models.load_model('models/anomaly_detection/autoencoder_model.h5')

def load_scaler():
    """Lädt den gespeicherten StandardScaler"""
    return joblib.load('models/scaler/standard_scaler.pkl')

def load_threshold():
    """Lädt den gespeicherten Schwellenwert für den Autoencoder (MSE-Grenzwert zur Anomalieerkennung)"""
    with open('models/anomaly_detection/thresholds_anomaly.json', 'r') as f:
        return json.load(f)['autoencoder']

# === 3. Eingabedaten einlesen ===
def load_input(path):
    """Lädt die Eingabedatei anhand der Dateiendung"""
    ext = os.path.splitext(path)[1]
    if ext == '.pkl':
        return pd.read_pickle(path)
    elif ext == '.csv':
        return pd.read_csv(path)
    elif ext == '.json':
        return pd.read_json(path)
    else:
        raise ValueError("Ungültiges Eingabeformat. Bitte .pkl, .csv oder .json verwenden.")

# === 4. Feature-Vorverarbeitung ===
def prepare_features(df, scaler):
    """Bereitet die Features für die Vorhersage vor (inkl. Validierung und Skalierung)"""
    required_features = list(scaler.feature_names_in_)

    print("✅ Erwartete Features:", required_features)
    print("📥 Vorhandene Features im Input:", list(df.columns))

    if 'active_errors' not in df.columns:
        raise ValueError("Input muss die Spalte 'active_errors' enthalten!")

    missing = set(required_features) - set(df.columns)
    if missing:
        raise ValueError(f"❌ Fehlende Spalten im Input: {missing}")

    X = df[required_features].copy()
    X_scaled = scaler.transform(X)
    return X_scaled

# === 5. Vorhersagefunktionen ===
def predict_classification(model, X_scaled):
    """Berechnet die Ausfallwahrscheinlichkeit für jedes Sample"""
    y_prob = model.predict_proba(X_scaled)[:, 1]
    return pd.DataFrame({'prediction_probability': y_prob})

def predict_anomaly(model, X_scaled, threshold):
    """Berechnet den Rekonstruktionsfehler und entscheidet, ob eine Anomalie vorliegt"""
    X_pred = model.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
    anomaly = (mse > threshold).astype(int)
    return pd.DataFrame({'reconstruction_error': mse, 'anomaly': anomaly})

# === 6. Hauptfunktion ===
def main():
    args = parse_args()
    df_input = load_input(args.input)
    scaler = load_scaler()
    X_scaled = prepare_features(df_input, scaler)

    if args.mode == 'classification':
        model = load_model('classification')
        result = predict_classification(model, X_scaled)
    else:
        model = load_model('anomaly')
        threshold = load_threshold()
        result = predict_anomaly(model, X_scaled, threshold)

    print("\n🔍 Vorhersage-Ergebnis:")
    print(result.head())

    result.to_csv('outputs/prediction_result.csv', index=False)

# === Skriptausführung ===
if __name__ == '__main__':
    main()
