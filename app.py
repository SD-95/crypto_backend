from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import logging # Import logging module
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO) # Set logging level to INFO
logger = logging.getLogger(__name__)

# Set environment variable to disable oneDNN optimizations
# This should be done before TensorFlow is imported/loaded if possible,
# though placing it here is common practice.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for specific origins
# Use the exact URL of your GitHub Pages deployment.
# If you are debugging locally, you might temporarily add "http://localhost:XXXX" here
# or use CORS(app) for broader access during development (NOT production).
CORS(app, resources={r"/predict": {"origins": "https://sd-95.github.io"}})
logger.info("CORS configured for origin: https://sd-95.github.io for /predict route")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load models - add error handling for clarity in logs
try:
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    logger.info("Random Forest model loaded.")
    meta_model = joblib.load(os.path.join(MODELS_DIR, "meta_model.pkl"))
    logger.info("Meta model loaded.")
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    logger.info("Scaler loaded.")
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    logger.info("Label Encoder loaded.")
    lstm_model = load_model(os.path.join(MODELS_DIR, "lstm_model.h5"))
    logger.info("LSTM model loaded.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    # Consider exiting or making the app unhealthy if models fail to load
    # For now, let it continue, but be aware it will fail on /predict
    # You might want to raise an exception or handle this more robustly in production.
    # raise e # uncomment this if you want the app to fail fast on model loading errors


@app.route("/")
def home():
    logger.info("Home route accessed.")
    return "Welcome to the Flask API!"

@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Predict route accessed (POST request).")
    try:
        data = request.get_json()
        if not data:
            logger.warning("Received empty or non-JSON data for prediction.")
            return jsonify({"error": "Invalid JSON data provided"}), 400

        logger.info(f"Received data: {data}")

        # Extract and prepare features with more robust defaults and error handling
        # Using .get() with default 0 is good, but converting to float might fail
        # if the value is not numeric. Adding a try-except for conversion.
        try:
            price = float(data.get("price", 0))
            price_1h = float(data.get("price_1h", 0))
            price_24h = float(data.get("price_24h", 0))
            price_7d = float(data.get("price_7d", 0))
            volume_24h = float(data.get("volume_24h", 0))
            market_cap = float(data.get("market_cap", 0))
        except ValueError as ve:
            logger.error(f"Data conversion error: {ve} - Raw data: {data}")
            return jsonify({"error": f"Invalid numeric input for prediction: {ve}"}), 400


        # Feature engineering (as per your original code)
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
        logger.info(f"Processed features: {final_features}")

        # Ensure models are loaded before proceeding
        if not all([rf_model, meta_model, scaler, le, lstm_model]):
             logger.error("One or more models failed to load at app startup.")
             return jsonify({"error": "Prediction service not ready, models not loaded."}), 503 # Service Unavailable

        # Prediction logic
        input_arr = np.array(final_features).reshape(1, -1)
        input_scaled = scaler.transform(input_arr)
        input_seq = input_scaled.reshape((1, 1, input_scaled.shape[1]))

        rf_probs = rf_model.predict_proba(input_scaled)
        lstm_probs = lstm_model.predict(input_seq)
        stacked_input = np.hstack((rf_probs, lstm_probs))

        final_pred_class = meta_model.predict(stacked_input)[0]
        final_pred_proba = meta_model.predict_proba(stacked_input)[0][final_pred_class]
        liquidity_level = le.inverse_transform([final_pred_class])[0]

        # Adjust liquidity level and advice based on confidence
        if liquidity_level.lower() == "high" and final_pred_proba < 0.80:
            liquidity_level = "Medium"

        if liquidity_level.lower() == "high" and final_pred_proba >= 0.80:
            advice = "Buy"
        elif liquidity_level.lower() == "medium" or (liquidity_level.lower() == "high" and final_pred_proba >= 0.60):
            advice = "Hold"
        else:
            advice = "Avoid"

        response = {
            "liquidity_level": liquidity_level,
            "confidence_score": round(final_pred_proba * 100, 2),
            "investment_advice": advice
        }
        logger.info(f"Prediction successful: {response}")
        return jsonify(response)

    except Exception as e:
        logger.exception("An unexpected error occurred during prediction.") # Logs traceback
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500 # Changed to 500 for unhandled exceptions

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting Flask app on {os.getenv('HOST', '0.0.0.0')}:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)