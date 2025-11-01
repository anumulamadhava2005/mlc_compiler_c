# MLC Compiler Web IDE

A modern, web-based IDE for the MLC (Machine Learning Configuration) Compiler. Write MLC code in your browser, compile it, and download the generated `train.py` and `venv` files.

## Features

- 🎨 **Modern IDE Interface** - Built with React, Monaco Editor (VS Code's editor), and TailwindCSS
- ⚡ **Real-time Compilation** - Compile MLC code with a single click
- 📥 **File Downloads** - Download generated `train.py` and `venv.zip` files
- 📝 **Example Templates** - Quick-start templates for Scikit-Learn, TensorFlow, PyTorch, and Transformers
- 🎯 **Syntax Highlighting** - Clean code editor with line numbers and syntax highlighting
- 📊 **Live Output** - See compilation output and errors in real-time

## Architecture

```
web-ide/
├── frontend/          # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx   # Main IDE component
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
├── backend/           # Express.js API server
│   ├── server.js     # API endpoints for compilation
│   └── temp/         # Temporary compilation files
├── package.json       # Root package for running both servers
└── README.md
```

## Prerequisites

1. **MLC Compiler** - Must be built first:
   ```bash
   cd ~/mlc_compiler_c
   make clean && make
   ```

2. **Node.js** - Version 16+ required:
   ```bash
   node --version  # Should be v16 or higher
   ```

## Installation

1. Navigate to the web-ide directory:
   ```bash
   cd ~/mlc_compiler_c/web-ide
   ```

2. Install all dependencies:
   ```bash
   npm run install:all
   ```

## Running the IDE

Start both frontend and backend servers:

```bash
npm run dev
```

This will start:
- **Frontend** at http://localhost:3000
- **Backend** at http://localhost:5000

Open your browser and navigate to http://localhost:3000

## Usage

1. **Write MLC Code** - Use the editor on the left to write or modify MLC configuration
2. **Select Example** - Click example buttons (Scikit-Learn, TensorFlow, PyTorch, Transformers) to load templates
3. **Compile** - Click the "Compile" button to generate training scripts
4. **View Output** - Check the output panel on the right for compilation results
5. **Download Files** - Once compiled successfully, download `train.py` or `venv.zip`

## Example MLC Code

### Scikit-Learn
```mlc
dataset "/home/madhava/datasets/classification.csv"

model RandomForestClassifier {
    n_estimators = 100
    max_depth = 4
}
```

### TensorFlow
```mlc
dataset "/home/madhava/datasets/flowers"

model ResNet50 {
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
}
```

### PyTorch
```mlc
dataset "/home/madhava/datasets/images"

model UNet {
    epochs = 20
    batch_size = 16
    learning_rate = 0.0001
}
```

### Transformers
```mlc
dataset "imdb"

model BERT {
    epochs = 3
    batch_size = 8
    learning_rate = 0.00002
}
```

## API Endpoints

- `POST /api/compile` - Compile MLC code
  - Body: `{ "code": "dataset ... model ... { ... }" }`
  - Returns: `{ "success": true/false, "output": "...", "files": {...} }`

- `GET /api/download/train` - Download train.py file
- `GET /api/download/venv` - Download venv as zip file
- `GET /api/health` - Health check endpoint

## Development

Run frontend and backend separately:

```bash
# Terminal 1 - Frontend
npm run dev:frontend

# Terminal 2 - Backend
npm run dev:backend
```

## Technology Stack

- **Frontend**: React 18, Vite, Monaco Editor, TailwindCSS, Lucide Icons
- **Backend**: Express.js, Node.js
- **Build**: Vite (for fast development and HMR)

## Troubleshooting

**Compiler not found error:**
- Make sure the MLC compiler is built: `cd ~/mlc_compiler_c && make`

**Port already in use:**
- Frontend (3000): Change in `frontend/vite.config.js`
- Backend (5000): Change `PORT` in `backend/server.js`

**Dependencies not installing:**
```bash
cd ~/mlc_compiler_c/web-ide
rm -rf node_modules frontend/node_modules
npm run install:all
```

## License

MIT License - Part of the MLC Compiler project
