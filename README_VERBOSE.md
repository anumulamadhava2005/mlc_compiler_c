# MLC Compiler - Verbose Mode (All 7 Compilation Phases)

## 🎯 Overview

This is an **enhanced version** of the MLC Compiler that displays **all 7 compilation phases** in detail, making it perfect for:

- 📚 **Learning compiler design**
- 🔍 **Debugging compilation issues**
- 🎓 **Teaching compiler theory**
- 🧪 **Understanding code transformations**

---

## 🚀 Quick Start

### 1️⃣ Build the Compiler
```bash
make -f Makefile.verbose
```

### 2️⃣ Run the Demo
```bash
./RUN_VERBOSE_DEMO.sh
```

### 3️⃣ Compile Your Own Code
```bash
./mlc_compiler_verbose -v your_file.mlc
```

---

## 📋 The 7 Compilation Phases

### 🔹 **Phase 1: Lexical Analysis**
- **Purpose:** Break source code into tokens
- **Output:** List of tokens with types and line numbers
- **Example:** `[DATASET, "dataset", line 1]`

### 🔹 **Phase 2: Syntax Analysis**
- **Purpose:** Build Abstract Syntax Tree (AST)
- **Output:** Visual parse tree showing program structure
- **Example:** Tree showing model hierarchy and parameters

### 🔹 **Phase 3: Semantic Analysis**
- **Purpose:** Type checking and symbol table construction
- **Output:** Symbol table with types, values, and scopes
- **Example:** `epochs: int, learning_rate: float`

### 🔹 **Phase 4: Intermediate Code Generation**
- **Purpose:** Generate platform-independent IR
- **Output:** 3-Address Code (TAC)
- **Example:** `t1 = LOAD_DATASET(imdb)`

### 🔹 **Phase 5: Code Optimization**
- **Purpose:** Optimize the IR for efficiency
- **Output:** Before/after comparison
- **Example:** Constant propagation, dead code elimination

### 🔹 **Phase 6: Code Generation**
- **Purpose:** Convert IR to target language (Python)
- **Output:** IR-to-Python mapping
- **Example:** `SET_PARAM(epochs, 3)` → `epochs = 3`

### 🔹 **Phase 7: Code Linking & Assembly**
- **Purpose:** Link libraries and create final executable
- **Output:** Library list and environment setup
- **Example:** Links transformers, datasets, torch

---

## 📂 Project Files

| File | Purpose |
|------|---------|
| `lexer_verbose.l` | Lexer with token printing |
| `parser_verbose.y` | Parser integrated with all phases |
| `compiler_phases.h` | Phase function declarations |
| `compiler_phases.c` | Phase implementations |
| `main_verbose.c` | Entry point with `-v` flag |
| `Makefile.verbose` | Build configuration |
| `example_verbose.mlc` | Example MLC code |
| `RUN_VERBOSE_DEMO.sh` | Quick demo script |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| `COMPILATION_PHASES_GUIDE.md` | Complete guide with examples |
| `QUICK_REFERENCE.md` | Quick command reference |
| `ACTUAL_OUTPUT_DEMO.md` | Real compilation output |
| `README_VERBOSE.md` | This file |

---

## 🎮 Usage Examples

### Example 1: BERT Model (Transformers)
```mlc
dataset "imdb"

model BERT {
    epochs = 3
    batch_size = 8
    learning_rate = 0.00002
}
```

**Compile:**
```bash
./mlc_compiler_verbose -v example_verbose.mlc
```

**Output:** Shows all 7 phases, generates `train.py` with HuggingFace Transformers code

---

### Example 2: Random Forest (Scikit-Learn)
```mlc
dataset "/data/classification.csv"

model RandomForestClassifier {
    n_estimators = 100
    max_depth = 10
}
```

**Compile:**
```bash
./mlc_compiler_verbose -v forest.mlc
```

**Output:** Shows all 7 phases, generates scikit-learn code

---

### Example 3: ResNet (TensorFlow)
```mlc
dataset "/data/images"

model ResNet50 {
    epochs = 20
    batch_size = 32
    learning_rate = 0.001
}
```

**Compile:**
```bash
./mlc_compiler_verbose -v resnet.mlc
```

**Output:** Shows all 7 phases, generates TensorFlow/Keras code

---

## 🎓 Educational Value

### For Students
- See how **text transforms** into **executable code**
- Understand **each compilation phase** in detail
- Learn **compiler design patterns**

### For Teachers
- **Demonstrate compiler theory** with real examples
- Show **practical applications** of abstract concepts
- Use as a **teaching tool** for CS courses

### For Developers
- **Debug compilation issues** efficiently
- Understand **code generation** process
- Learn **IR optimization** techniques

---

## 🔍 What You'll See

When you run with `-v` flag:

```
╔═══════════════════════════════════════════════════════════════╗
║          MLC COMPILER - MULTI-PHASE COMPILATION              ║
║        Machine Learning Configuration Compiler v2.0          ║
╚═══════════════════════════════════════════════════════════════╝

🔹 PHASE 1: LEXICAL ANALYSIS
  [DATASET, "dataset", line 1]
  [STRING, ""imdb"", line 1]
  ...

🔹 PHASE 2: SYNTAX ANALYSIS
  Parse Tree:
  program
  ├── dataset_decl
  └── model_def_list
  ...

🔹 PHASE 3: SEMANTIC ANALYSIS
  Symbol Table:
  ┌──────────┬──────┬────────┬────────┐
  │ Name     │ Type │ Value  │ Scope  │
  ...

🔹 PHASE 4: INTERMEDIATE CODE GENERATION
  3-Address Code:
  1: t0 = LOAD_DATASET(imdb)
  ...

🔹 PHASE 5: CODE OPTIMIZATION
  Before: 8 instructions
  After: 8 instructions (optimal)
  ...

🔹 PHASE 6: CODE GENERATION
  IR → Python mapping
  ...

🔹 PHASE 7: CODE LINKING & ASSEMBLY
  Libraries linked: transformers, datasets, torch
  ...

🎉 COMPILATION COMPLETE
```

---

## 📊 Comparison: Regular vs Verbose

| Feature | Regular (`./mlc_compiler`) | Verbose (`./mlc_compiler_verbose -v`) |
|---------|---------------------------|---------------------------------------|
| **Tokens** | Hidden | ✅ All shown with line numbers |
| **Parse Tree** | Hidden | ✅ Visual tree structure |
| **Symbol Table** | Hidden | ✅ Complete table display |
| **IR Code** | Hidden | ✅ 3-address code shown |
| **Optimization** | Silent | ✅ Before/after comparison |
| **Code Mapping** | Hidden | ✅ IR→Python mapping |
| **Linking** | Brief | ✅ Detailed library info |
| **Use Case** | Production | Learning & Debugging |

---

## 🛠️ Command Reference

```bash
# Build
make -f Makefile.verbose

# Clean
make -f Makefile.verbose clean

# Run verbose (all phases shown)
./mlc_compiler_verbose -v file.mlc

# Run normal (minimal output)
./mlc_compiler_verbose file.mlc

# Show help
./mlc_compiler_verbose -h

# Quick demo
./RUN_VERBOSE_DEMO.sh

# View generated code
cat train.py

# Run training script
venv/bin/python train.py
```

---

## 🔧 Build Requirements

```bash
sudo apt install flex bison gcc python3-venv
```

---

## 📈 Workflow

```
1. Write .mlc file
    ↓
2. Run compiler with -v flag
    ↓
3. Review all 7 phases
    ↓
4. Check generated train.py
    ↓
5. Run the training script
```

---

## 🎯 Supported ML Frameworks

| Framework | Models | Example |
|-----------|--------|---------|
| **Scikit-Learn** | LinearRegression, RandomForest, etc. | `model RandomForestClassifier { ... }` |
| **TensorFlow** | ResNet, VGG, EfficientNet, etc. | `model ResNet50 { ... }` |
| **PyTorch** | UNet, GAN, VAE, etc. | `model UNet { ... }` |
| **Transformers** | BERT, GPT, T5, etc. | `model BERT { ... }` |

---

## 💡 Tips

1. **Always use `-v`** when learning or debugging
2. **Check symbol table** to verify variable scoping
3. **Review IR code** to understand optimizations
4. **Compare phases** to see transformations
5. **Read generated Python** to validate output

---

## 🐛 Troubleshooting

### Build fails
```bash
make -f Makefile.verbose clean
make -f Makefile.verbose
```

### Parser warnings
The "1 shift/reduce conflict" is expected and harmless.

### Missing libraries
```bash
sudo apt update
sudo apt install flex bison gcc
```

---

## 📞 Support

- **Full Guide:** `COMPILATION_PHASES_GUIDE.md`
- **Quick Ref:** `QUICK_REFERENCE.md`
- **Example Output:** `ACTUAL_OUTPUT_DEMO.md`
- **Main README:** `README.md`

---

## 🌟 Features

✅ **7 compilation phases** fully displayed  
✅ **Educational tool** for learning compilers  
✅ **Debugging aid** for identifying errors  
✅ **Multi-framework** code generation  
✅ **Symbol table** visualization  
✅ **IR optimization** tracking  
✅ **Production-ready** Python output  

---

## 🎉 Try It Now!

```bash
# Quick start
./RUN_VERBOSE_DEMO.sh

# Or manually
make -f Makefile.verbose
./mlc_compiler_verbose -v example_verbose.mlc
cat train.py
```

---

**Experience compiler design in action! 🚀**
