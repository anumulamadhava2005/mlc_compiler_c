#!/bin/bash

# Quick setup script for MLC Web IDE with Prediction Support

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   MLC Web IDE - Prediction Feature Setup${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"

# 1. Install Node dependencies
echo -e "${YELLOW}📦 Installing Node.js dependencies...${NC}"
npm install
cd frontend && npm install && cd ..
echo -e "${GREEN}✅ Node dependencies installed${NC}\n"

# 2. Install Python dependencies
echo -e "${YELLOW}🐍 Installing Python dependencies...${NC}"
pip3 install -r backend/requirements.txt
echo -e "${GREEN}✅ Python dependencies installed${NC}\n"

# 3. Make scripts executable
echo -e "${YELLOW}🔧 Making scripts executable...${NC}"
chmod +x start_with_predict.sh stop.sh backend/predict_api.py
echo -e "${GREEN}✅ Scripts are executable${NC}\n"

# 4. Build compiler if needed
if [ ! -f "../mlc_compiler" ]; then
    echo -e "${YELLOW}🔨 Building MLC compiler...${NC}"
    cd .. && make && cd web-ide
    echo -e "${GREEN}✅ Compiler built${NC}\n"
else
    echo -e "${GREEN}✅ Compiler already built${NC}\n"
fi

echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✅ Setup Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}\n"

echo -e "${BLUE}🚀 To start the IDE with prediction support:${NC}"
echo -e "   ./start_with_predict.sh\n"

echo -e "${BLUE}📖 Documentation:${NC}"
echo -e "   • README_PREDICTION.md - Complete prediction guide"
echo -e "   • README.md - General web IDE documentation\n"

echo -e "${YELLOW}💡 Next Steps:${NC}"
echo -e "   1. Run: ./start_with_predict.sh"
echo -e "   2. Open: http://localhost:5173"
echo -e "   3. Write MLC code and compile"
echo -e "   4. Train your model"
echo -e "   5. Test with the prediction panel!\n"
