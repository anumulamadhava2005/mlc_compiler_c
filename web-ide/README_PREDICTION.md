# MLC Web IDE - Prediction Feature Guide

## 🎯 Overview

The MLC Web IDE now includes **integrated model testing** that allows you to:
- Upload CSV files for batch predictions
- Enter manual features for single predictions
- View accuracy metrics and prediction results
- Test trained models directly in the browser

---

## 🚀 Quick Start

### 1. Start the Full Stack

```bash
cd web-ide
chmod +x start_with_predict.sh stop.sh
./start_with_predict.sh
```

This starts:
- **Frontend** (Port 5173) - React UI
- **Backend** (Port 5000) - Node.js API
- **Prediction API** (Port 5001) - Flask Python API

### 2. Open the IDE

Visit: http://localhost:5173

---

## 📋 Complete Workflow

### Step 1: Write MLC Code
```mlc
dataset "./data.csv"

model SVM {
    kernel = linear
    C = 1.0
}
```

### Step 2: Compile
Click the **"Compile"** button to generate `train.py`

### Step 3: Train the Model
Run training (in a terminal):
```bash
cd /home/madhava/mlc_compiler_c
venv/bin/python train.py
```

This creates `model.pkl`

### Step 4: Test the Model
Back in the web IDE:
1. Click **"Test Model"** button (becomes green when model.pkl exists)
2. Choose testing method:
   - **Upload CSV**: Upload a test dataset
   - **Manual Input**: Enter feature values

### Step 5: View Results
- See predictions
- View accuracy (if labels provided)
- Compare predicted vs actual values

---

## 📊 Testing Methods

### Method 1: Upload CSV File

**CSV Format (with labels):**
```csv
feature1,feature2,feature3,label
5.1,3.5,1.4,0
4.9,3.0,1.4,0
6.2,3.4,5.4,2
```

**CSV Format (without labels):**
```csv
feature1,feature2,feature3
5.1,3.5,1.4
4.9,3.0,1.4
```

**What you get:**
- Predictions for all rows
- Accuracy (if labels exist)
- First 10 results displayed
- Model type and statistics

### Method 2: Manual Input

**Input format:**
```
5.1, 3.5, 1.4, 0.2
```

**What you get:**
- Single prediction
- Confidence scores (for classifiers)
- Probability distribution

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  React Frontend                     │
│                  (Port 5173)                        │
│  • Code Editor                                      │
│  • Prediction Panel                                 │
│  • Results Display                                  │
└──────────────┬──────────────────────────────────────┘
               │
               │ HTTP Requests
               ▼
┌─────────────────────────────────────────────────────┐
│              Node.js Backend                        │
│                 (Port 5000)                         │
│  • /api/compile                                     │
│  • /api/predict/check-model                         │
│  • /api/predict/from-csv   ────────┐                │
│  • /api/predict/from-input         │                │
└────────────────────────────────────┼────────────────┘
                                     │
                                     │ Proxies to
                                     ▼
┌─────────────────────────────────────────────────────┐
│            Flask Prediction API                     │
│                (Port 5001)                          │
│  • Loads model.pkl                                  │
│  • Processes CSV uploads                            │
│  • Makes predictions                                │
│  • Calculates accuracy                              │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 API Endpoints

### Check if Model Exists
```
GET /api/predict/check-model
```

**Response:**
```json
{
  "exists": true,
  "path": "/path/to/model.pkl",
  "model_type": "SVC"
}
```

### Predict from CSV
```
POST /api/predict/from-csv
Content-Type: multipart/form-data
```

**Request:**
- `file`: CSV file

**Response:**
```json
{
  "success": true,
  "model_type": "SVC",
  "num_samples": 100,
  "num_features": 4,
  "accuracy": 0.95,
  "predictions": [0, 1, 0, ...],
  "actual_labels": [0, 1, 0, ...]
}
```

### Predict from Manual Input
```
POST /api/predict/from-input
Content-Type: application/json
```

**Request:**
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Response:**
```json
{
  "success": true,
  "model_type": "SVC",
  "prediction": 0,
  "probability": [0.95, 0.03, 0.02]
}
```

---

## 🐛 Troubleshooting

### "Prediction API not running"
**Solution:**
```bash
cd web-ide/backend
python3 predict_api.py
```

### "Model not found"
**Solution:** Train your model first:
```bash
cd /home/madhava/mlc_compiler_c
venv/bin/python train.py
```

### Dependencies Missing
**Backend:**
```bash
cd web-ide
npm install
```

**Flask API:**
```bash
pip3 install -r backend/requirements.txt
```

---

## 📦 Installation

### First Time Setup

1. **Install Node dependencies:**
```bash
cd web-ide
npm install
cd frontend && npm install && cd ..
```

2. **Install Python dependencies:**
```bash
pip3 install flask flask-cors joblib pandas numpy scikit-learn
# Or use requirements.txt
pip3 install -r backend/requirements.txt
```

3. **Build the compiler:**
```bash
cd ..
make
```

4. **Start everything:**
```bash
cd web-ide
./start_with_predict.sh
```

---

## 🎨 UI Features

### Visual Indicators
- 🟢 **Green "Test Model" button**: Model is ready
- 🔴 **Gray "Test Model" button**: No model found
- ⚡ **Loading spinners**: Operations in progress
- ✅ **Success indicators**: Green checkmarks
- ❌ **Error indicators**: Red X marks

### Real-time Updates
- Model status checks every 5 seconds
- Automatic detection when training completes
- Live prediction results

---

## 💡 Tips

1. **Keep terminal open** when running training to see progress
2. **Model persists** between IDE sessions (until overwritten)
3. **CSV format matters** - ensure proper structure
4. **Feature count must match** training data dimensions
5. **Check logs** at `/tmp/mlc-ide/*.log` if issues occur

---

## 🛑 Stopping Services

```bash
cd web-ide
./stop.sh
```

Or manually:
```bash
pkill -f "node.*server.js"
pkill -f "predict_api.py"
pkill -f "vite"
```

---

## 📝 Example Session

1. Write code: Define SVM classifier
2. Compile: Generate train.py
3. Train: Run `venv/bin/python train.py`
4. Test: Upload CSV with test data
5. Result: See 95% accuracy! 🎉

---

## 🔗 Related Files

- `backend/server.js` - Main backend server
- `backend/predict_api.py` - Flask prediction API
- `frontend/src/App.jsx` - React UI with prediction panel
- `start_with_predict.sh` - Startup script
- `stop.sh` - Shutdown script

---

**Happy Testing! 🚀**
