#!/bin/bash

# Stop all MLC Web IDE services

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🛑 Stopping MLC Web IDE services...${NC}\n"

# Kill processes by PID files
if [ -f /tmp/mlc-ide/frontend.pid ]; then
    FRONTEND_PID=$(cat /tmp/mlc-ide/frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        kill $FRONTEND_PID 2>/dev/null
        echo -e "${GREEN}✅ Stopped Frontend (PID: $FRONTEND_PID)${NC}"
    fi
    rm -f /tmp/mlc-ide/frontend.pid
fi

if [ -f /tmp/mlc-ide/backend.pid ]; then
    BACKEND_PID=$(cat /tmp/mlc-ide/backend.pid)
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        kill $BACKEND_PID 2>/dev/null
        echo -e "${GREEN}✅ Stopped Backend (PID: $BACKEND_PID)${NC}"
    fi
    rm -f /tmp/mlc-ide/backend.pid
fi

if [ -f /tmp/mlc-ide/predict-api.pid ]; then
    PREDICT_PID=$(cat /tmp/mlc-ide/predict-api.pid)
    if ps -p $PREDICT_PID > /dev/null 2>&1; then
        kill $PREDICT_PID 2>/dev/null
        echo -e "${GREEN}✅ Stopped Prediction API (PID: $PREDICT_PID)${NC}"
    fi
    rm -f /tmp/mlc-ide/predict-api.pid
fi

# Fallback: kill by process name
pkill -f "node.*server.js" 2>/dev/null
pkill -f "predict_api.py" 2>/dev/null
pkill -f "vite" 2>/dev/null

echo -e "\n${GREEN}All services stopped!${NC}\n"
