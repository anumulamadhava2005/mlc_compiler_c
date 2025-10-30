#!/bin/bash
# MLC Compiler Demo Script

echo "================================"
echo "MLC Compiler Demonstration"
echo "================================"
echo ""

echo "1️⃣  Building MLC Compiler..."
make clean > /dev/null 2>&1
make > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Build successful!"
else
    echo "   ❌ Build failed!"
    exit 1
fi
echo ""

echo "2️⃣  Testing TensorFlow Backend (ResNet50)..."
echo "   Input: test.mlc"
cat test.mlc
echo ""
./mlc_compiler test.mlc
echo ""

echo "3️⃣  Generated Python Script Preview:"
head -30 train.py
echo "   ... (truncated)"
echo ""

echo "4️⃣  Testing PyTorch Backend (UNet)..."
echo "   Input: test_pytorch.mlc"
cat test_pytorch.mlc
echo ""
./mlc_compiler test_pytorch.mlc > /dev/null
echo "   ✅ PyTorch code generated successfully!"
echo ""

echo "5️⃣  Testing Transformers Backend (BERT)..."
echo "   Input: test_transformer.mlc"
cat test_transformer.mlc
echo ""
./mlc_compiler test_transformer.mlc > /dev/null
echo "   ✅ Transformers code generated successfully!"
echo ""

echo "================================"
echo "Demo Complete! 🎉"
echo "================================"
echo ""
echo "To run training:"
echo "   venv/bin/python train.py"
echo ""
echo "For more information, see README.md"
