from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import joblib, numpy as np, os
from tensorflow.keras.models import load_model

# 🔧 Disable oneDNN optimizations for TensorFlow
:contentReference[oaicite:1]{index=1}

:contentReference[oaicite:2]{index=2}

# 🌐 Enable CORS globally for your frontend
CORS(app,
     :contentReference[oaicite:3]{index=3}
     :contentReference[oaicite:4]{index=4}
     :contentReference[oaicite:5]{index=5}
     supports_credentials=False)

# ✨ Add CORS headers to every response (including preflight)
@app.after_request
:contentReference[oaicite:6]{index=6}
    :contentReference[oaicite:7]{index=7}
    :contentReference[oaicite:8]{index=8}
        :contentReference[oaicite:9]{index=9}
        :contentReference[oaicite:10]{index=10}
        :contentReference[oaicite:11]{index=11}
    return response

# 🗂 Load ML models from your 'models' directory
:contentReference[oaicite:12]{index=12}
:contentReference[oaicite:13]{index=13}

:contentReference[oaicite:14]{index=14}
:contentReference[oaicite:15]{index=15}
:contentReference[oaicite:16]{index=16}
:contentReference[oaicite:17]{index=17}
:contentReference[oaicite:18]{index=18}

:contentReference[oaicite:19]{index=19}
def home():
    :contentReference[oaicite:20]{index=20}

:contentReference[oaicite:21]{index=21}
def predict():
    # 🛑 Preflight response
    :contentReference[oaicite:22]{index=22}
        :contentReference[oaicite:23]{index=23}

    try:
        data = request.get_json()
        # --- Feature extraction ---
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

        # --- Scaling and model predictions ---
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

        advice = (
            "Buy" if liquidity_level.lower() == "high" and final_pred_proba >= 0.80 else
            "Hold" if liquidity_level.lower() == "medium" or final_pred_proba >= 0.60 else
            "Avoid"
        )

        return jsonify({
            "liquidity_level": liquidity_level,
            "confidence_score": round(final_pred_proba * 100, 2),
            "investment_advice": advice
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

:contentReference[oaicite:24]{index=24}
    :contentReference[oaicite:25]{index=25}
    :contentReference[oaicite:26]{index=26}
