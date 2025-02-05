from flask import Flask, request, jsonify, render_template
import numpy as np
import onnxruntime as ort
import joblib
from test_task import CurveData, ONNXPredictor
import os

app = Flask(__name__)

# Load model and scalers
model_path = 'curve_model.onnx'
scaler_dir = os.path.dirname(model_path)

try:
    # Initialize predictor
    model = ort.InferenceSession(model_path)
    x_scaler = joblib.load(os.path.join(scaler_dir, 'X_scaler.joblib'))
    y_scaler = joblib.load(os.path.join(scaler_dir, 'y_scaler.joblib'))
    predictor = ONNXPredictor(model, x_scaler, y_scaler)
    print("Model and scalers loaded successfully")
except Exception as e:
    print(f"Error loading model or scalers: {str(e)}")
    print("Please ensure curve_model.onnx, X_scaler.joblib, and y_scaler.joblib are in the root directory")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input points from request
        data = request.json
        points = np.array(data['points'])
        
        # Create CurveData object
        curve_a = CurveData(points)
        
        # Get prediction
        predicted_points = predictor.predict(curve_a)
        
        return jsonify({
            'predicted_points': predicted_points.tolist()
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 