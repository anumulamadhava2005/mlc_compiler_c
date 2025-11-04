# 🚀 MLC Web IDE - Quick Start Testing

## One-Time Setup (Run Once)
```bash
cd web-ide
./SETUP_PREDICTION.sh
```

## Start the IDE
```bash
./start_with_predict.sh
```
Opens:
- Frontend: http://localhost:5173
- Backend: http://localhost:5000  
- Prediction API: http://localhost:5001

## Complete Example (5 Minutes)

### 1. Write Code
In the web IDE editor:
```mlc
dataset "./data.csv"

model SVM {
    kernel = linear
    C = 1.0
}
```

### 2. Compile
Click **"Compile"** button → Generates `train.py`

### 3. Train Model
Open terminal:
```bash
cd /home/madhava/mlc_compiler_c
venv/bin/python train.py
```
Wait for: "✅ Training completed!"

### 4. Test Model
Back in web IDE:
1. Click green **"Test Model"** button
2. Choose mode:
   - **Upload CSV**: Select test file
   - **Manual Input**: Type `5.1, 3.5, 1.4`
3. See results instantly!

## Sample CSV Format
```csv
feature1,feature2,feature3,label
5.1,3.5,1.4,0
4.9,3.0,1.4,0
6.2,3.4,5.4,2
```

## Stop Services
```bash
./stop.sh
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Gray "Test Model" button | Train model first |
| "Prediction API not running" | Check if Flask started (port 5001) |
| Upload fails | Check CSV format |
| Port in use | Run `./stop.sh` first |

## Files to Check
- **Logs**: `/tmp/mlc-ide/*.log`
- **Model**: `../model.pkl`
- **Training script**: `../train.py`

## Key Features
✅ Real-time model detection  
✅ Batch CSV predictions  
✅ Single feature predictions  
✅ Accuracy metrics  
✅ Confidence scores  
✅ Predicted vs Actual comparison

---

**Need help?** Read `README_PREDICTION.md` for complete documentation.
