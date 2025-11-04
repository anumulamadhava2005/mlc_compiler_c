#!/usr/bin/env python3
"""
Prediction script for trained models
Loads model.pkl and makes predictions on new data
"""

import joblib
import pandas as pd
import numpy as np

def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('model.pkl')
        print("✅ Model loaded successfully!")
        print(f"📦 Model type: {type(model).__name__}")
        return model
    except FileNotFoundError:
        print("❌ Error: model.pkl not found!")
        print("💡 Run train.py first to generate the model")
        exit(1)

def predict_from_csv(model, csv_path):
    """Make predictions from a CSV file"""
    print(f"\n📂 Loading data from: {csv_path}")
    data = pd.read_csv(csv_path)
    
    # Assume last column is the label (for comparison), rest are features
    X = data.iloc[:, :-1].values
    y_true = data.iloc[:, -1].values if data.shape[1] > 1 else None
    
    print(f"📊 Data shape: {X.shape}")
    print(f"🔢 Number of samples: {X.shape[0]}")
    
    # Make predictions
    predictions = model.predict(X)
    
    print("\n🎯 Predictions:")
    for i, pred in enumerate(predictions[:10]):  # Show first 10
        if y_true is not None:
            print(f"  Sample {i+1}: Predicted = {pred}, Actual = {y_true[i]}")
        else:
            print(f"  Sample {i+1}: Predicted = {pred}")
    
    if len(predictions) > 10:
        print(f"  ... and {len(predictions) - 10} more predictions")
    
    # Calculate accuracy if we have true labels
    if y_true is not None:
        accuracy = (predictions == y_true).mean()
        print(f"\n📈 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return predictions

def predict_from_input(model):
    """Make predictions from manual input"""
    print("\n📝 Enter feature values (comma-separated):")
    print("   Example: 5.1,3.5,1.4,0.2")
    
    user_input = input("➜ ")
    
    try:
        # Parse input
        features = [float(x.strip()) for x in user_input.split(',')]
        X = np.array([features])
        
        print(f"\n🔢 Input features: {features}")
        
        # Make prediction
        prediction = model.predict(X)[0]
        print(f"🎯 Prediction: {prediction}")
        
        # Try to get probability if it's a classifier
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            print(f"📊 Probabilities: {proba}")
        
        return prediction
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("=" * 60)
    print("🤖 MODEL PREDICTION TOOL")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    print("\n📋 Choose prediction mode:")
    print("  1. Predict from CSV file")
    print("  2. Predict from manual input")
    print("  3. Exit")
    
    choice = input("\n➜ Enter choice (1-3): ").strip()
    
    if choice == '1':
        csv_path = input("📂 Enter CSV file path: ").strip()
        predict_from_csv(model, csv_path)
    elif choice == '2':
        predict_from_input(model)
    elif choice == '3':
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
