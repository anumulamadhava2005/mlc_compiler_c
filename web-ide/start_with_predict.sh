#!/bin/bash

# MLC Compiler Web IDE Startup Script with Prediction Support
# This script starts both the Node.js backend and Flask prediction API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   MLC Compiler Web IDE - Full Stack Startup${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if MLC compiler is built
COMPILER_PATH="../mlc_compiler"
if [ ! -f "$COMPILER_PATH" ]; then
    echo -e "${YELLOW}⚠️  MLC compiler not found${NC}"
    echo -e "Building compiler..."
    cd ..
    make clean && make
    cd web-ide
    echo -e "${GREEN}✅ Compiler built successfully${NC}\n"
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Installing npm dependencies...${NC}"
    npm install
    echo -e "${GREEN}✅ Dependencies installed${NC}\n"
fi

# Check if Flask dependencies are installed
echo -e "${BLUE}🐍 Checking Python dependencies...${NC}"

# Use venv_web for Flask API
PYTHON_CMD="python3"
if [ -d "venv_web" ] && [ -f "venv_web/bin/python" ]; then
    PYTHON_CMD="venv_web/bin/python"
    echo -e "${GREEN}Using web IDE virtual environment${NC}"
elif [ -d "../venv" ] && [ -f "../venv/bin/python" ]; then
    PYTHON_CMD="../venv/bin/python"
    echo -e "${GREEN}Using compiler virtual environment${NC}"
else
    echo -e "${YELLOW}Using system Python (venv recommended)${NC}"
fi

# Check Flask installation
if ! $PYTHON_CMD -c "import flask" 2>/dev/null; then
    echo -e "${RED}❌ Flask not installed!${NC}"
    echo -e "${YELLOW}Creating venv and installing dependencies...${NC}"
    python3 -m venv venv_web
    venv_web/bin/pip install flask flask-cors joblib pandas numpy scikit-learn
    PYTHON_CMD="venv_web/bin/python"
fi

echo -e "${GREEN}✅ Python dependencies ready${NC}\n"

# Create PID file directory
mkdir -p /tmp/mlc-ide

# Kill any existing processes
echo -e "${YELLOW}🧹 Cleaning up existing processes...${NC}"
pkill -f "node.*server.js" 2>/dev/null || true
pkill -f "predict_api.py" 2>/dev/null || true
sleep 1

# Start Flask Prediction API
echo -e "${BLUE}🧠 Starting Flask Prediction API (Port 5001)...${NC}"
# Save current directory
WEB_IDE_DIR="$(pwd)"

cd backend
$WEB_IDE_DIR/$PYTHON_CMD predict_api.py > /tmp/mlc-ide/predict-api.log 2>&1 &
PREDICT_PID=$!
echo $PREDICT_PID > /tmp/mlc-ide/predict-api.pid
cd ..
echo -e "${GREEN}✅ Prediction API started (PID: $PREDICT_PID)${NC}"

# Wait for Flask to start
sleep 2

# Start Node.js Backend
echo -e "${BLUE}🚀 Starting Node.js Backend (Port 5000)...${NC}"
cd backend
node server.js > /tmp/mlc-ide/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > /tmp/mlc-ide/backend.pid
cd ..
echo -e "${GREEN}✅ Backend started (PID: $BACKEND_PID)${NC}"

# Wait for backend to start
sleep 2

# Start Vite Frontend
echo -e "${BLUE}⚡ Starting Vite Frontend (Port 5173)...${NC}"
cd frontend
npm run dev > /tmp/mlc-ide/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > /tmp/mlc-ide/frontend.pid
cd ..
echo -e "${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}\n"

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 3

echo -e "\n${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   🎉 MLC Compiler Web IDE is Running!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}\n"

echo -e "${BLUE}📍 Service URLs:${NC}"
echo -e "   • Frontend:        ${GREEN}http://localhost:5173${NC}"
echo -e "   • Backend API:     ${GREEN}http://localhost:5000${NC}"
echo -e "   • Prediction API:  ${GREEN}http://localhost:5001${NC}\n"

echo -e "${BLUE}📋 Process IDs:${NC}"
echo -e "   • Frontend:    ${YELLOW}$FRONTEND_PID${NC}"
echo -e "   • Backend:     ${YELLOW}$BACKEND_PID${NC}"
echo -e "   • Predict API: ${YELLOW}$PREDICT_PID${NC}\n"

echo -e "${BLUE}📂 Log Files:${NC}"
echo -e "   • Frontend:    /tmp/mlc-ide/frontend.log"
echo -e "   • Backend:     /tmp/mlc-ide/backend.log"
echo -e "   • Predict API: /tmp/mlc-ide/predict-api.log\n"

echo -e "${BLUE}🛑 To stop all services:${NC}"
echo -e "   ./stop.sh\n"

echo -e "${BLUE}💡 Usage:${NC}"
echo -e "   1. Write MLC code in the editor"
echo -e "   2. Click 'Compile' to generate train.py"
echo -e "   3. Run training (outside IDE or via terminal)"
echo -e "   4. Click 'Test Model' to upload CSV or enter features"
echo -e "   5. View predictions and accuracy!\n"

echo -e "${YELLOW}Press Ctrl+C to view logs or use './stop.sh' to stop all services${NC}\n"

# Follow logs
trap "echo -e '\n${YELLOW}Stopping services...${NC}'; ./stop.sh 2>/dev/null; exit 0" INT TERM

echo -e "${BLUE}📜 Tailing logs (Ctrl+C to exit):${NC}\n"
tail -f /tmp/mlc-ide/*.log
