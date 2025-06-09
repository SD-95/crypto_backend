from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import logging
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["https://sd-95.github.io"])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

try:
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    meta_model = joblib.load(os.path.join(MODELS_DIR, "meta_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    lstm_model = load_model(os.path.join(MODELS_DIR, "lstm_model.h5"))
    logger.info("All models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    rf_model = meta_model = scaler = le = lstm_model = None

@app.route("/")
def home():
    return "Welcome to the Flask API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        price = float(data.get("price", 0))
        price_1h = float(data.get("price_1h", 0))
        price_24h = float(data.get("price_24h", 0))
        price_7d = float(data.get("price_7d", 0))
        volume_24h = float(data.get("volume_24h", 0))
        market_cap = float(data.get("market_cap", 0))

        price_lag1 = price / (1 + price_1h / 100) if price_1h != -100 else 0
        volume_lag1 = volume_24h
        mktcap_lag1 = market_cap
        price_2d_avg = (price + price_lag1) / 2
        volume_2d_avg = (volume_24h + volume_lag1) / 2
        vol_to_mcap = volume_24h / market_cap if market_cap else 0
        vol_price_ratio = volume_24h / price if price else 0

        final_features = [
            price_1h, price_24h, price_7d,
            price_lag1, volume_lag1, mktcap_lag1,
            price_2d_avg, volume_2d_avg,
            vol_to_mcap, vol_price_ratio
        ]

        if not all([rf_model, meta_model, scaler, le, lstm_model]):
            return jsonify({"error": "Model not loaded"}), 503

        input_arr = np.array(final_features).reshape(1, -1)
        input_scaled = scaler.transform(input_arr)
        input_seq = input_scaled.reshape((1, 1, input_scaled.shape[1]))

        rf_probs = rf_model.predict_proba(input_scaled)
        lstm_probs = lstm_model.predict(input_seq)
        stacked_input = np.hstack((rf_probs, lstm_probs))

        final_pred_class = meta_model.predict(stacked_input)[0]
        final_pred_proba = meta_model.predict_proba(stacked_input)[0][final_pred_class]
        liquidity_level = le.inverse_transform([final_pred_class])[0]

        if liquidity_level.lower() == "high" and final_pred_proba < 0.80:
            liquidity_level = "Medium"

        if liquidity_level.lower() == "high" and final_pred_proba >= 0.80:
            advice = "Buy"
        elif liquidity_level.lower() == "medium" or (liquidity_level.lower() == "high" and final_pred_proba >= 0.60):
            advice = "Hold"
        else:
            advice = "Avoid"

        return jsonify({
            "liquidity_level": liquidity_level,
            "confidence_score": round(final_pred_proba * 100, 2),
            "investment_advice": advice
        })

    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
