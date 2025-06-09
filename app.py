from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, os
from tensorflow.keras.models import load_model

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
app = Flask(__name__)

# 🚦 Enable CORS for specific origins: local dev and GitHub Pages
:contentReference[oaicite:2]{index=2}
    :contentReference[oaicite:3]{index=3}
    :contentReference[oaicite:4]{index=4}
]}})

# Load models and scaler
:contentReference[oaicite:5]{index=5}
:contentReference[oaicite:6]{index=6}
:contentReference[oaicite:7]{index=7}
:contentReference[oaicite:8]{index=8}
:contentReference[oaicite:9]{index=9}
:contentReference[oaicite:10]{index=10}
:contentReference[oaicite:11]{index=11}

:contentReference[oaicite:12]{index=12}
def home():
    :contentReference[oaicite:13]{index=13}

:contentReference[oaicite:14]{index=14}
def predict():
    try:
        :contentReference[oaicite:15]{index=15}
        :contentReference[oaicite:16]{index=16}

        # Extract and prepare features as before...
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

        print(f"Liquidity Level: {liquidity_level}")
        print(f"Confidence Score: {round(final_pred_proba * 100, 2)}%")
        print(f"Investment Advice: {advice}")

        return jsonify({
            "liquidity_level": liquidity_level,
            "confidence_score": round(final_pred_proba * 100, 2),
            "investment_advice": advice
        })

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 400

:contentReference[oaicite:17]{index=17}
    :contentReference[oaicite:18]{index=18}
    :contentReference[oaicite:19]{index=19}
