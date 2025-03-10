from flask import Flask, request, jsonify
import joblib
import numpy as np

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Load the trained model
model_path = "models/final_fraud_detection_model.pkl"  # Keep relative path
model = joblib.load(model_path)

# ✅ Define feature names (Ensure it matches the trained model)
final_features = ["V14", "V10", "V12", "V4", "V17", "V3", "V7", "V16", "V11", "V18", "V9", "V21", "Amount"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running on Render! Use /predict for fraud detection."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Get JSON data from request
        data = request.get_json()

        # ✅ Ensure the "features" key exists
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request."}), 400
        
        # ✅ Convert input data to NumPy array
        features = np.array(data["features"]).reshape(1, -1)

        # ✅ Ensure correct number of features
        if features.shape[1] != len(final_features):
            return jsonify({
                "error": f"Expected {len(final_features)} features, but got {features.shape[1]}."
            }), 400

        # ✅ Get fraud probability
        fraud_probability = model.predict_proba(features)[:, 1][0]

        # ✅ Apply threshold for classification
        threshold = 0.19  # Adjusted based on model optimization
        prediction = "Fraud Alert!" if fraud_probability >= threshold else "Safe Transaction"

        return jsonify({
            "fraud_probability": round(float(fraud_probability), 4),
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask app on Render (Removes localhost issue)
if __name__ == "__main__":
    from waitress import serve  # Production WSGI server
    serve(app, host="0.0.0.0", port=10000)  # Render uses dynamic ports
