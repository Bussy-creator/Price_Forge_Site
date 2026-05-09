from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app) 

# ── Load model ────────────────────────────────────────────────────────────────
# Ensure this file is in the main directory of your repository
with open("laptop_price_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
encoders = data["encoders"]
features = data["features"]

def safe_transform(encoder, value):
    value = str(value).lower().strip()
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return encoder.transform([encoder.classes_[0]])[0]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.json
        encoded = {
            "brand_name_enc":      safe_transform(encoders["brand_name"], payload["brand_name"]),
            "series_name_enc":     safe_transform(encoders["series_name"], payload["series_name"]),
            "Touchscreen":         bool(payload["Touchscreen"]),
            "RAM_GB":              float(payload["RAM_GB"]),
            "storage_gb":          float(payload["storage_gb"]),
            "is_ssd":              int(payload["is_ssd"]),
            "Processor_clean_enc": safe_transform(encoders["Processor_clean"], payload["Processor_clean"]),
            "GPU":                 bool(payload["GPU"]),
        }
        X_new = pd.DataFrame([encoded])[features]
        log_pred = model.predict(X_new)[0]
        price = np.expm1(log_pred)
        return jsonify({"price": float(price)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Standard Flask Route ──────────────────────────────────────────────────────
@app.route('/')
def index():
    # This looks for 'index.html' inside the 'templates' folder
    return render_template('index.html')

if __name__ == '__main__':
    # Use environment variable for port to support various hosting platforms
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
