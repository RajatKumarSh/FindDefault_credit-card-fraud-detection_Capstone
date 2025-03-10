from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load the trained model

import os

# Get the correct path for deployment
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get current directory
model_path = os.path.join(base_dir, "..", "models", "final_fraud_detection_model.pkl")  # Move up to 'models' folder

# Load the model
model = joblib.load(model_path)

# Get the expected feature names from the trained model
expected_features = model.feature_names_in_.tolist()  # Ensure it's a list

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running! Use /predict for fraud detection."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Step 1: Get JSON Data from API Request
        data = request.get_json()

        # Step 2: Convert JSON Data to Pandas DataFrame
        input_data = pd.DataFrame([data["features"]], columns=expected_features[:len(data["features"])])

        # Step 3: Ensure All 30 Features Are Present
        for feature in expected_features:
            if feature not in input_data.columns:
                input_data[feature] = 0  # Fill missing features with 0

        # Step 4: Convert Data to Model-Readable Format
        input_array = input_data[expected_features].values  # Ensure correct feature order

        # Step 5: Make Prediction
        fraud_probability = model.predict_proba(input_array)[:, 1][0]
        fraud_label = "Fraud Alert!" if fraud_probability >= 0.19 else "Safe Transaction"

        return jsonify({
            "fraud_probability": round(fraud_probability, 4),
            "prediction": fraud_label
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
