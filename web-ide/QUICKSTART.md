# Quick Start Guide - MLC Web IDE

## 📦 Installation (One-time setup)

### Step 1: Install Dependencies
```bash
cd ~/mlc_compiler_c/web-ide
npm run install:all
```

This installs all packages for both frontend and backend.

### Step 2: Ensure Compiler is Built
```bash
cd ~/mlc_compiler_c
make clean && make
```

## 🚀 Running the IDE

### Option 1: Using the Start Script (Recommended)
```bash
cd ~/mlc_compiler_c/web-ide
./start.sh
```

### Option 2: Using npm
```bash
cd ~/mlc_compiler_c/web-ide
npm run dev
```

### Option 3: Manual (Two Terminals)
```bash
# Terminal 1 - Backend
cd ~/mlc_compiler_c/web-ide
npm run dev:backend

# Terminal 2 - Frontend
cd ~/mlc_compiler_c/web-ide
npm run dev:frontend
```

## 🌐 Access the IDE

Open your browser and go to: **http://localhost:3000**

## ✨ Features to Try

1. **Load Examples**: Click the example buttons (Scikit-Learn, TensorFlow, PyTorch, Transformers)
2. **Write Code**: Edit MLC code in the Monaco editor
3. **Compile**: Click the "Compile" button (play icon)
4. **View Output**: Check compilation results in the output panel
5. **Download**: Download generated `train.py` and `venv.zip` files

## 🎯 Keyboard Shortcuts

The editor supports standard VS Code shortcuts:
- `Ctrl+F` - Find
- `Ctrl+H` - Find and Replace
- `Ctrl+/` - Toggle Comment
- `Ctrl+Z` - Undo
- `Ctrl+Shift+Z` - Redo

## 📝 Example Workflow

1. Open http://localhost:3000
2. Click "TensorFlow" example button
3. Modify parameters if needed (epochs, batch_size, etc.)
4. Click "Compile"
5. Wait for success message
6. Click "Download train.py"
7. Run the training script: `venv/bin/python train.py`

## 🔧 Troubleshooting

**Port 3000 already in use:**
```bash
# Change port in frontend/vite.config.js, line 6:
port: 3001  # or any other port
```

**Port 5000 already in use:**
```bash
# Change port in backend/server.js, line 15:
const PORT = 5001  # or any other port
```

**Compiler not found:**
```bash
cd ~/mlc_compiler_c
make clean && make
```

**Can't download files:**
- Ensure compilation was successful
- Check that train.py exists in ~/mlc_compiler_c/
- Check backend logs in terminal

## 🎨 UI Overview

```
┌─────────────────────────────────────────────────────────┐
│  MLC Compiler IDE           [Examples: SK | TF | PT | T]│
├────────────────────────┬────────────────────────────────┤
│                        │  Output                        │
│   Monaco Editor        │  ┌──────────────────────────┐ │
│   (Code Input)         │  │  Compilation results... │ │
│                        │  └──────────────────────────┘ │
│   dataset "..."        │                                │
│                        │  [Download train.py]           │
│   model ResNet50 {     │  [Download venv.zip]           │
│     epochs = 10        │                                │
│   }                    │                                │
│                        │                                │
│ [▶ Compile]            │                                │
└────────────────────────┴────────────────────────────────┘
```

## 📊 Supported Models

- **Scikit-Learn**: LinearRegression, LogisticRegression, DecisionTree, RandomForest, KNeighbors, SVC, GaussianNB, KMeans, LinearSVC, SGDClassifier
- **TensorFlow**: ResNet, VGG, EfficientNet, MobileNet, DenseNet, InceptionV3
- **PyTorch**: UNet, GAN, AutoEncoder, VAE
- **Transformers**: BERT, GPT, T5, RoBERTa, DistilBERT

## 📞 Need Help?

Check the main README: `~/mlc_compiler_c/web-ide/README.md`
