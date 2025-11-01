# MLC Compiler - Quick Reference Guide

## 🎯 7-Phase Compilation Output

### ✨ What You Get

When you run `./mlc_compiler_verbose -v example.mlc`, you'll see **all 7 compilation phases** clearly displayed:

---

## 📋 Command Cheat Sheet

| Command | Description |
|---------|-------------|
| **Build** | `make -f Makefile.verbose` |
| **Run Verbose** | `./mlc_compiler_verbose -v file.mlc` |
| **Run Normal** | `./mlc_compiler_verbose file.mlc` |
| **Quick Demo** | `./RUN_VERBOSE_DEMO.sh` |
| **Clean** | `make -f Makefile.verbose clean` |

---

## 🔍 Phase-by-Phase Breakdown

### 🔹 **PHASE 1: LEXICAL ANALYSIS**
**What it does:** Breaks source code into tokens

**Output:**
```
[DATASET     , "dataset"            , line 1]
[STRING      , ""imdb""             , line 1]
[MODEL       , "model"              , line 3]
[IDENTIFIER  , "BERT"               , line 3]
[LBRACE      , "{"                  , line 3]
...
```

**Key Points:**
- Each token has: **type**, **lexeme**, **line number**
- Recognizes: keywords, identifiers, operators, literals

---

### 🔹 **PHASE 2: SYNTAX ANALYSIS**
**What it does:** Builds Abstract Syntax Tree (AST)

**Output:**
```
Parse Tree (AST):
program
├── dataset_decl
│   └── path: "imdb"
└── model_def_list
    └── model_def_1
        ├── model_name: BERT
        └── parameters {
            ├── epochs = 3
            ├── batch_size = 8
            └── learning_rate = 0.00002
```

**Key Points:**
- Shows grammar rules applied
- Visual tree structure
- Validates syntax correctness

---

### 🔹 **PHASE 3: SEMANTIC ANALYSIS**
**What it does:** Type checking and symbol table construction

**Output:**
```
Symbol Table:
┌────────────────┬──────────┬────────────┬────────────┐
│ Name           │ Type     │ Value      │ Scope      │
├────────────────┼──────────┼────────────┼────────────┤
│ dataset        │ string   │ imdb       │ global     │
│ epochs         │ int      │ 3          │ model_BERT │
│ learning_rate  │ float    │ 0.00002    │ model_BERT │
└────────────────┴──────────┴────────────┴────────────┘

Type Checking:
  ✓ Variable 'epochs' in scope 'model_BERT': type=int, value=3
```

**Key Points:**
- All variables tracked in symbol table
- Type inference (int, float, string)
- Scope management (global, model-specific)

---

### 🔹 **PHASE 4: INTERMEDIATE CODE GENERATION**
**What it does:** Generates 3-Address Code (TAC)

**Output:**
```
3-Address Code (TAC):
  1: t0 = LOAD_DATASET(imdb)
  2: t2 = INIT_MODEL(BERT)
  3: t3 = SET_PARAM(epochs, 3)
  4: t4 = SET_PARAM(batch_size, 8)
  5: t5 = SET_PARAM(learning_rate, 0.000020)
  6: t6 = COMPILE_MODEL(optimizer, loss_fn)
  7: t7 = TRAIN(t0, epochs)
  8: t8 = SAVE_MODEL(model_path)
```

**Key Points:**
- Platform-independent representation
- Each instruction: `result = operation(args)`
- Temporary variables (t0, t1, t2, ...)

---

### 🔹 **PHASE 5: CODE OPTIMIZATION**
**What it does:** Optimizes intermediate representation

**Output:**
```
Before Optimization:
  [8 instructions shown]

Optimizations Applied:
  ✓ Constant propagation
  ✓ Dead code elimination (none found)
  ✓ Common subexpression elimination (none found)

After Optimization:
  Instructions: 8 → 8 (no change - code already optimal)
```

**Key Points:**
- Shows before/after IR
- Lists optimization techniques applied
- Reduces instruction count when possible

---

### 🔹 **PHASE 6: CODE GENERATION**
**What it does:** Converts IR to target language (Python)

**Output:**
```
Backend Framework: transformers

Mapping IR to Target Code:
  Model: BERT
    IR: SET_PARAM(epochs, 3)
    → Python: epochs = 3
    
    IR: SET_PARAM(batch_size, 8)
    → Python: batch_size = 8

✅ Target code written to: train.py
```

**Key Points:**
- Maps IR instructions to Python code
- Selects appropriate ML framework
- Generates executable script

---

### 🔹 **PHASE 7: CODE LINKING & ASSEMBLY**
**What it does:** Links libraries and creates final executable

**Output:**
```
External Libraries Linked:
  ✓ transformers (NLP framework)
  ✓ datasets (dataset library)
  ✓ torch (backend)

Virtual Environment Setup:
  ✓ Python venv created
  ✓ Dependencies installed
  ✓ Environment activated

✅ Final Output: train.py (executable Python script)
✅ Virtual Environment: venv/ (ready to use)
```

**Key Points:**
- Links external ML frameworks
- Sets up virtual environment
- Ready-to-run output

---

## 📊 Compilation Pipeline

```
Source Code (.mlc)
    ↓
🔹 Phase 1: Lexical Analysis  → Tokens
    ↓
🔹 Phase 2: Syntax Analysis   → AST
    ↓
🔹 Phase 3: Semantic Analysis → Symbol Table
    ↓
🔹 Phase 4: IR Generation     → 3-Address Code
    ↓
🔹 Phase 5: Optimization      → Optimized IR
    ↓
🔹 Phase 6: Code Generation   → Python Code
    ↓
🔹 Phase 7: Linking           → Executable + venv
    ↓
Final Output (train.py + venv/)
```

---

## 🎓 Use Cases

### For Learning
```bash
# See all phases
./mlc_compiler_verbose -v example.mlc
```

### For Production
```bash
# Minimal output
./mlc_compiler_verbose example.mlc
```

### For Debugging
```bash
# Identify which phase has errors
./mlc_compiler_verbose -v buggy_code.mlc
```

---

## 📝 Example Workflows

### Workflow 1: BERT Sentiment Analysis
```mlc
dataset "imdb"

model BERT {
    epochs = 3
    batch_size = 8
    learning_rate = 0.00002
}
```

**Run:** `./mlc_compiler_verbose -v example.mlc`  
**Result:** Generates BERT training code with HuggingFace Transformers

---

### Workflow 2: Random Forest Classifier
```mlc
dataset "/data/classification.csv"

model RandomForestClassifier {
    n_estimators = 100
    max_depth = 10
}
```

**Run:** `./mlc_compiler_verbose -v forest.mlc`  
**Result:** Generates scikit-learn Random Forest code

---

### Workflow 3: ResNet Image Classification
```mlc
dataset "/data/images"

model ResNet50 {
    epochs = 20
    batch_size = 32
    learning_rate = 0.001
}
```

**Run:** `./mlc_compiler_verbose -v resnet.mlc`  
**Result:** Generates TensorFlow/Keras ResNet50 training code

---

## 🎯 Key Features

✅ **Educational** - See how compilers work phase-by-phase  
✅ **Transparent** - Every transformation is visible  
✅ **Debugging** - Identify errors at any stage  
✅ **Multi-Framework** - Supports Scikit-Learn, TensorFlow, PyTorch, Transformers  
✅ **Production Ready** - Generates runnable Python code  

---

## 🔧 Troubleshooting

### Build Errors
```bash
make -f Makefile.verbose clean
make -f Makefile.verbose
```

### Parser Conflicts
- The "1 shift/reduce conflict" warning is expected and harmless

### Missing Dependencies
```bash
sudo apt install flex bison gcc python3-venv
```

---

## 📚 Files Overview

| File | Purpose |
|------|---------|
| `lexer_verbose.l` | Lexical analyzer with token printing |
| `parser_verbose.y` | Parser with all phase hooks |
| `compiler_phases.c` | Phase display implementations |
| `main_verbose.c` | Entry point with verbose flag |
| `example_verbose.mlc` | Example input file |
| `train.py` | Generated output (after compilation) |

---

## 🚀 Next Steps

1. **Build:** `make -f Makefile.verbose`
2. **Try the demo:** `./RUN_VERBOSE_DEMO.sh`
3. **Create your own .mlc file**
4. **Compile with:** `./mlc_compiler_verbose -v yourfile.mlc`
5. **Run generated code:** `venv/bin/python train.py`

---

## 💡 Pro Tips

- Use **verbose mode** (`-v`) for learning and debugging
- Use **normal mode** for production builds
- Check the **symbol table** to verify variable scoping
- Review the **IR code** to understand optimization opportunities
- Compare **before/after optimization** to see performance improvements

---

**Happy Compiling! 🎉**
