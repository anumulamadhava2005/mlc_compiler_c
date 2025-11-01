# MLC Web IDE - Project Structure

## 📁 Complete File Tree

```
mlc_compiler_c/
├── web-ide/                          # New web IDE directory
│   ├── package.json                  # Root package (runs both servers)
│   ├── start.sh                      # Quick start script
│   ├── README.md                     # Full documentation
│   ├── QUICKSTART.md                 # Quick start guide
│   ├── .gitignore                    # Git ignore file
│   │
│   ├── frontend/                     # React frontend
│   │   ├── package.json              # Frontend dependencies
│   │   ├── vite.config.js            # Vite configuration
│   │   ├── tailwind.config.js        # TailwindCSS config
│   │   ├── postcss.config.js         # PostCSS config
│   │   ├── index.html                # HTML entry point
│   │   └── src/
│   │       ├── main.jsx              # React entry point
│   │       ├── App.jsx               # Main IDE component (Monaco editor)
│   │       └── index.css             # Global styles (Tailwind)
│   │
│   └── backend/                      # Express.js backend
│       └── server.js                 # API server (compilation endpoints)
│
├── mlc_compiler                      # Compiled MLC compiler binary
├── lexer.l                           # Flex lexer
├── parser.y                          # Bison parser
├── main.c                            # Compiler main
├── ast.h                             # AST definitions
└── Makefile                          # Build configuration
```

## 🏗️ Architecture

```
┌─────────────┐         HTTP          ┌─────────────┐
│   Browser   │ ◄──────────────────► │   Express   │
│  (React UI) │   POST /api/compile   │   Backend   │
│   Port 3000 │   GET /api/download   │   Port 5000 │
└─────────────┘                       └──────┬──────┘
                                             │
                                             │ executes
                                             ▼
                                      ┌─────────────┐
                                      │     MLC     │
                                      │  Compiler   │
                                      │   Binary    │
                                      └──────┬──────┘
                                             │
                                             │ generates
                                             ▼
                                      ┌─────────────┐
                                      │  train.py   │
                                      │    venv/    │
                                      └─────────────┘
```

## 🔄 Data Flow

1. **User writes MLC code** in Monaco editor (frontend)
2. **Click Compile** → sends code to backend via POST /api/compile
3. **Backend**:
   - Saves code to temp.mlc
   - Executes `./mlc_compiler temp.mlc`
   - Captures output and checks for generated files
   - Returns result to frontend
4. **Frontend displays**:
   - Compilation output
   - Success/error status
   - Download buttons (if successful)
5. **User downloads**:
   - GET /api/download/train → train.py file
   - GET /api/download/venv → venv.zip file

## 🎨 UI Components

### Main IDE Interface (App.jsx)
- **Header**: Title + Example buttons
- **Left Panel**: Monaco Editor (code input)
- **Right Panel**: Output console + Download buttons
- **Footer**: Framework info

### Key Features
- ✅ Syntax highlighting (Monaco Editor)
- ✅ Real-time compilation
- ✅ Error handling and display
- ✅ File download (train.py & venv.zip)
- ✅ Example templates
- ✅ Modern, responsive UI

## 📦 Dependencies

### Frontend
- **react** (18.2.0) - UI framework
- **@monaco-editor/react** (4.6.0) - Code editor
- **lucide-react** (0.294.0) - Icons
- **axios** (1.6.2) - HTTP client
- **vite** (5.0.8) - Build tool
- **tailwindcss** (3.3.6) - Styling

### Backend
- **express** (4.18.2) - Web server
- **cors** (2.8.5) - CORS middleware
- **archiver** (6.0.1) - Zip creation

## 🚀 API Endpoints

| Method | Endpoint              | Description                      |
|--------|-----------------------|----------------------------------|
| POST   | /api/compile          | Compile MLC code                |
| GET    | /api/download/train   | Download train.py                |
| GET    | /api/download/venv    | Download venv.zip                |
| GET    | /api/health           | Health check                     |

## 🎯 Key Files

| File                     | Purpose                              |
|--------------------------|--------------------------------------|
| frontend/src/App.jsx     | Main IDE UI component               |
| backend/server.js        | API server with compilation logic   |
| start.sh                 | Quick start script                  |
| QUICKSTART.md            | User guide                          |

## 🔧 Configuration

### Ports
- Frontend: 3000 (configurable in `frontend/vite.config.js`)
- Backend: 5000 (configurable in `backend/server.js`)

### Paths
- Compiler: `../mlc_compiler` (relative to backend)
- Temp files: `backend/temp/`
- Generated files: `../../` (compiler root directory)

## 💡 Usage Example

```bash
# 1. Install dependencies
cd ~/mlc_compiler_c/web-ide
npm run install:all

# 2. Start the IDE
./start.sh

# 3. Open browser
# http://localhost:3000

# 4. Write/select MLC code → Compile → Download files
```

## 🎨 Tech Stack Summary

- **Frontend**: React 18 + Vite + TailwindCSS + Monaco Editor
- **Backend**: Node.js + Express
- **Editor**: Monaco (VS Code's editor component)
- **Icons**: Lucide React
- **Styling**: TailwindCSS (utility-first CSS)
- **HTTP**: Axios (frontend), built-in (backend)
- **Build**: Vite (fast HMR and builds)
