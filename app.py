from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained components
model = joblib.load("catboost_model.pkl")          # Your CatBoost model
scaler = joblib.load("scaler.pkl")                 # Scaler used during training
label_encoder = joblib.load("label_encoder.pkl")   # For decoding labels
feature_columns = joblib.load("feature_columns.pkl")  # Ordered list of feature names

# Serve the HTML file
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in correct order
        input_values = [float(data.get(col, 0)) for col in feature_columns]
        input_array = np.array(input_values).reshape(1, -1)
        
        # Apply scaling
        scaled_input = scaler.transform(input_array)

        # Predict
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]

        # Decode class
        predicted_label = label_encoder.inverse_transform([int(prediction)])[0]

        return jsonify({
            "prediction": predicted_label,
            "probabilities": list(map(float, probabilities)),
            "classes": label_encoder.classes_.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the server
if __name__ == "__main__":
    app.run(debug=True)
