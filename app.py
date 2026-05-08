from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, static_folder='.')

# ── Load model ────────────────────────────────────────────────────────────────
with open("laptop_price_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
encoders = data["encoders"]

# ── Helper to handle unseen labels ─────────────────────────────────────────────
def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        print(f"Warning: '{value}' not seen in training data. Using most common category.")
        return encoder.transform([encoder.classes_[0]])[0]

# ── Prediction endpoint ────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.json
        
        # Encode the input
        encoded = {
            "brand_name_enc":      safe_transform(encoders["brand_name"], payload["brand_name"]),
            "series_name_enc":     safe_transform(encoders["series_name"], payload["series_name"]),
            "Touchscreen":         payload["Touchscreen"],
            "RAM_GB":              payload["RAM_GB"],
            "storage_gb":          payload["storage_gb"],
            "is_ssd":              payload["is_ssd"],
            "Processor_clean_enc": safe_transform(encoders["Processor_clean"], payload["Processor_clean"]),
            "GPU":                 payload["GPU"],
        }
        
        X_new = pd.DataFrame([encoded])
        log_pred = model.predict(X_new)[0]
        price = np.expm1(log_pred)  # reverse the log transform
        
        return jsonify({"price": float(price)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Serve HTML ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'price_forge(1).html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
