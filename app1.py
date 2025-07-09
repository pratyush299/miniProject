from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model, encoder, and feature columns
model = joblib.load("depression_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Serve the HTML file
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data
        data = request.get_json()
        
        # Create a list of features in the same order as during training
        features = []
        for column in feature_columns:
            features.append(float(data.get(column, 0)))  # Default to 0 if feature is missing
            
        # Reshape features for prediction
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)  # Get probability scores
        
        # Convert prediction back to label
        result = label_encoder.inverse_transform(prediction)[0]
        
        # Return the result with probability scores
        return jsonify({
            'prediction': result,
            'probabilities': prediction_proba.tolist(),
            'classes': label_encoder.classes_.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)