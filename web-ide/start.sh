#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting MLC Compiler Web IDE...${NC}\n"

# Check if compiler exists
if [ ! -f "../mlc_compiler" ]; then
    echo -e "${YELLOW}⚠️  MLC compiler not found. Building it first...${NC}"
    cd .. && make clean && make
    cd web-ide
fi

# Check if node_modules exists
if [ ! -d "node_modules" ] || [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}📦 Installing dependencies...${NC}"
    npm run install:all
fi

echo -e "${GREEN}✅ Starting servers...${NC}"
echo -e "${BLUE}Frontend: http://localhost:3001${NC}"
echo -e "${BLUE}Backend:  http://localhost:5000${NC}\n"

npm run dev
