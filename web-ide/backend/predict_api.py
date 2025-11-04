#!/usr/bin/env python3
"""
Flask API for model predictions
Integrates with the web IDE for testing trained models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import sys
import traceback

app = Flask(__name__)
CORS(app)

# Path configuration
COMPILER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(COMPILER_DIR, 'model.pkl')

@app.route('/api/predict/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'prediction-api'})

@app.route('/api/predict/check-model', methods=['GET'])
def check_model():
    """Check if trained model exists"""
    try:
        model_exists = os.path.exists(MODEL_PATH)
        
        if model_exists:
            model = joblib.load(MODEL_PATH)
            model_type = type(model).__name__
            
            return jsonify({
                'exists': True,
                'path': MODEL_PATH,
                'model_type': model_type
            })
        else:
            return jsonify({
                'exists': False,
                'message': 'No trained model found. Please run train.py first.'
            })
    except Exception as e:
        return jsonify({
            'exists': False,
            'error': str(e)
        }), 500

@app.route('/api/predict/from-csv', methods=['POST'])
def predict_from_csv():
    """Make predictions from uploaded CSV data"""
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            return jsonify({
                'success': False,
                'error': 'Model not found. Please train the model first by running train.py'
            }), 404
        
        # Load model
        model = joblib.load(MODEL_PATH)
        
        # Get CSV data from request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        # Read CSV
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to read CSV: {str(e)}'
            }), 400
        
        # Check if dataset has labels (last column)
        has_labels = df.shape[1] > 1
        
        if has_labels:
            X = df.iloc[:, :-1].values
            y_true = df.iloc[:, -1].values
        else:
            X = df.values
            y_true = None
        
        # Make predictions
        predictions = model.predict(X)
        
        # Calculate accuracy if labels exist
        accuracy = None
        if y_true is not None:
            accuracy = float((predictions == y_true).mean())
        
        # Get probabilities if available (for classifiers)
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X)
                probabilities = proba.tolist()
            except:
                probabilities = None
        
        # Prepare response
        result = {
            'success': True,
            'model_type': type(model).__name__,
            'num_samples': int(X.shape[0]),
            'num_features': int(X.shape[1]),
            'predictions': predictions.tolist(),
            'accuracy': accuracy,
            'has_labels': has_labels
        }
        
        if probabilities is not None:
            result['probabilities'] = probabilities
        
        if y_true is not None:
            result['actual_labels'] = y_true.tolist()
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/predict/from-input', methods=['POST'])
def predict_from_input():
    """Make prediction from manual feature input"""
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            return jsonify({
                'success': False,
                'error': 'Model not found. Please train the model first by running train.py'
            }), 404
        
        # Load model
        model = joblib.load(MODEL_PATH)
        
        # Get features from request
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({
                'success': False,
                'error': 'No features provided'
            }), 400
        
        features = data['features']
        
        # Convert to numpy array
        try:
            X = np.array([features])
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid feature format: {str(e)}'
            }), 400
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X)[0]
                probability = proba.tolist()
            except:
                probability = None
        
        result = {
            'success': True,
            'model_type': type(model).__name__,
            'features': features,
            'prediction': float(prediction) if isinstance(prediction, (np.integer, np.floating)) else str(prediction)
        }
        
        if probability is not None:
            result['probability'] = probability
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    print(f"🧠 Prediction API starting...")
    print(f"📂 Compiler directory: {COMPILER_DIR}")
    print(f"🎯 Model path: {MODEL_PATH}")
    app.run(host='0.0.0.0', port=5001, debug=True)
