#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     MLC COMPILER - MULTI-PHASE COMPILATION DEMONSTRATION     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Build the verbose compiler
echo "🔧 Building verbose compiler..."
make -f Makefile.verbose clean
make -f Makefile.verbose

if [ ! -f mlc_compiler_verbose ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build successful!"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Running MLC Compiler with ALL PHASES displayed..."
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Run with verbose mode
./mlc_compiler_verbose -v example_verbose.mlc

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🎉 Demonstration Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Generated files:"
echo "  ✓ train.py - Executable Python training script"
echo ""
echo "To see the generated code:"
echo "  cat train.py"
echo ""
