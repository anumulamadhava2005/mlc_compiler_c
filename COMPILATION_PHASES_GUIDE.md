# MLC Compiler - Multi-Phase Compilation Guide

## Overview

This guide demonstrates all **7 phases** of the MLC compiler with detailed output for each stage.

---

## Usage

### Build the Verbose Compiler
```bash
make -f Makefile.verbose
```

### Run with Verbose Mode
```bash
./mlc_compiler_verbose -v example_verbose.mlc
```

### Quick Demo
```bash
./RUN_VERBOSE_DEMO.sh
```

---

## Example Input File (example_verbose.mlc)

```mlc
dataset "imdb"

model BERT {
    epochs = 3
    batch_size = 8
    learning_rate = 0.00002
}
```

---

## Expected Output - All 7 Phases

### 🔹 PHASE 1: LEXICAL ANALYSIS

```
═══════════════════════════════════════════════════════════════
🔹 PHASE 1: LEXICAL ANALYSIS
═══════════════════════════════════════════════════════════════
Tokens extracted:

[DATASET     , "dataset"            , line 1]
[STRING      , ""imdb""             , line 1]
[MODEL       , "model"              , line 3]
[IDENTIFIER  , "BERT"               , line 3]
[LBRACE      , "{"                  , line 3]
[IDENTIFIER  , "epochs"             , line 4]
[ASSIGN      , "="                  , line 4]
[INT         , "3"                  , line 4]
[IDENTIFIER  , "batch_size"         , line 5]
[ASSIGN      , "="                  , line 5]
[INT         , "8"                  , line 5]
[IDENTIFIER  , "learning_rate"      , line 6]
[ASSIGN      , "="                  , line 6]
[FLOAT       , "0.00002"            , line 6]
[RBRACE      , "}"                  , line 7]
```

**Explanation:**
- Each token includes: **token type**, **lexeme** (actual text), and **line number**
- The lexer recognizes keywords (`dataset`, `model`), identifiers (`BERT`, `epochs`), operators (`=`, `{`, `}`), and literals (strings, integers, floats)

---

### 🔹 PHASE 2: SYNTAX ANALYSIS

```
═══════════════════════════════════════════════════════════════
🔹 PHASE 2: SYNTAX ANALYSIS
═══════════════════════════════════════════════════════════════
Grammar Rules Applied:
  program → dataset_decl model_def_list
  dataset_decl → DATASET STRING
  model_def_list → model_def_list model_def | model_def
  model_def → MODEL ID { param_list }
  param_list → param_list param | param | ε
  param → ID = value
  value → INT | FLOAT | STRING

Parse Tree (AST):
program
├── dataset_decl
│   ├── DATASET
│   └── path: "imdb"
└── model_def_list
    └── model_def_1
      ├── model_name: BERT
      ├── parameters {
      │   ├── epochs = 3
      │   ├── batch_size = 8
      │   ├── learning_rate = 0.00002
      │   └── }
```

**Explanation:**
- Shows the context-free grammar rules applied during parsing
- Displays the **Abstract Syntax Tree (AST)** structure
- Verifies syntactic correctness

---

### 🔹 PHASE 3: SEMANTIC ANALYSIS

```
═══════════════════════════════════════════════════════════════
🔹 PHASE 3: SEMANTIC ANALYSIS
═══════════════════════════════════════════════════════════════

Symbol Table:
┌────────────────────┬──────────┬────────────────┬────────────────┐
│ Name               │ Type     │ Value          │ Scope          │
├────────────────────┼──────────┼────────────────┼────────────────┤
│ dataset            │ string   │ imdb           │ global         │
│ model_name         │ identifier│ BERT          │ model_BERT     │
│ epochs             │ int      │ 3              │ model_BERT     │
│ batch_size         │ int      │ 8              │ model_BERT     │
│ learning_rate      │ float    │ 0.00002        │ model_BERT     │
└────────────────────┴──────────┴────────────────┴────────────────┘

Type Checking:
  ✓ Variable 'dataset' in scope 'global': type=string, value=imdb
  ✓ Variable 'model_name' in scope 'model_BERT': type=identifier, value=BERT
  ✓ Variable 'epochs' in scope 'model_BERT': type=int, value=3
  ✓ Variable 'batch_size' in scope 'model_BERT': type=int, value=8
  ✓ Variable 'learning_rate' in scope 'model_BERT': type=float, value=0.00002

✅ No type errors detected.
```

**Explanation:**
- **Symbol Table**: Stores all identifiers with their types, values, and scopes
- **Type Checking**: Validates that all variables have consistent types
- **Scope Analysis**: Ensures variables are defined in appropriate scopes

---

### 🔹 PHASE 4: INTERMEDIATE CODE GENERATION

```
═══════════════════════════════════════════════════════════════
🔹 PHASE 4: INTERMEDIATE CODE GENERATION (3-Address Code)
═══════════════════════════════════════════════════════════════

3-Address Code (TAC):
    1: t1 = LOAD_DATASET(imdb)
    2: t2 = INIT_MODEL(BERT)
    3: t3 = SET_PARAM(epochs, 3)
    4: t4 = SET_PARAM(batch_size, 8)
    5: t5 = SET_PARAM(learning_rate, 0.00002)
    6: t6 = COMPILE_MODEL(optimizer, loss_fn)
    7: t7 = TRAIN(t1, epochs)
    8: t8 = SAVE_MODEL(model_path)
```

**Explanation:**
- **3-Address Code (TAC)**: Intermediate representation where each instruction has at most 3 operands
- Each instruction is in the form: `result = operation(arg1, arg2)`
- Temporary variables (`t1`, `t2`, etc.) hold intermediate results
- This IR is platform-independent and easier to optimize

---

### 🔹 PHASE 5: CODE OPTIMIZATION

```
═══════════════════════════════════════════════════════════════
🔹 PHASE 5: CODE OPTIMIZATION
═══════════════════════════════════════════════════════════════
Before Optimization:
    1: t1 = LOAD_DATASET(imdb)
    2: t2 = INIT_MODEL(BERT)
    3: t3 = SET_PARAM(epochs, 3)
    4: t4 = SET_PARAM(batch_size, 8)
    5: t5 = SET_PARAM(learning_rate, 0.00002)
    6: t6 = COMPILE_MODEL(optimizer, loss_fn)
    7: t7 = TRAIN(t1, epochs)
    8: t8 = SAVE_MODEL(model_path)

Optimizations Applied:
  ✓ Constant propagation
  ✓ Dead code elimination (none found)
  ✓ Common subexpression elimination (none found)

After Optimization:
  Instructions: 8 → 8 (no change - code already optimal)
```

**Explanation:**
- **Constant Propagation**: Replace variables with their constant values where possible
- **Dead Code Elimination**: Remove unreachable or unused code
- **Common Subexpression Elimination**: Compute repeated expressions only once
- In this case, code is already optimal, so no changes are made

---

### 🔹 PHASE 6: CODE GENERATION

```
═══════════════════════════════════════════════════════════════
🔹 PHASE 6: CODE GENERATION (Target: Python)
═══════════════════════════════════════════════════════════════
Backend Framework: transformers

Mapping IR to Target Code:

  Model: BERT
    IR: SET_PARAM(epochs, 3)
    → Python: epochs = 3
    
    IR: SET_PARAM(batch_size, 8)
    → Python: batch_size = 8
    
    IR: SET_PARAM(learning_rate, 0.00002)
    → Python: learning_rate = 0.00002

✅ Target code written to: train.py
```

**Explanation:**
- Converts intermediate representation to **target language** (Python)
- Maps IR instructions to actual Python constructs
- Selects appropriate backend framework (transformers for BERT)
- Generates executable code

---

### 🔹 PHASE 7: CODE LINKING & ASSEMBLY

```
═══════════════════════════════════════════════════════════════
🔹 PHASE 7: CODE LINKING & ASSEMBLY
═══════════════════════════════════════════════════════════════
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

═══════════════════════════════════════════════════════════════
🎉 COMPILATION COMPLETE
═══════════════════════════════════════════════════════════════

To run the generated code:
  $ venv/bin/python train.py
```

**Explanation:**
- **Library Linking**: Links external ML frameworks and dependencies
- **Environment Setup**: Creates isolated Python virtual environment
- **Final Assembly**: Produces ready-to-run executable script
- All dependencies are resolved and installed

---

## Generated Output File (train.py)

```python
#!/usr/bin/env python3
# Generated by MLC Compiler
# Auto-generated machine learning training script

# =====================================
# Model 1: BERT
# Backend: transformers
# =====================================

epochs = 3
batch_size = 8
learning_rate = 0.00002

# Dataset Loading
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset('imdb')
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Model: BERT (transformers backend)
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Model
model_name = 'BERT'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    evaluation_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

trainer.train()
print('Training complete!')

# Save model
trainer.save_model('./model')
print('Model saved!')
```

---

## Compilation Flow Diagram

```
┌─────────────────┐
│  Source Code    │
│  (.mlc file)    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐      🔹 PHASE 1: LEXICAL ANALYSIS
│  Lexer (Flex)   │ ───► Tokens: [DATASET, "imdb", line 1], ...
└────────┬────────┘
         │
         ↓
┌─────────────────┐      🔹 PHASE 2: SYNTAX ANALYSIS
│ Parser (Bison)  │ ───► Parse Tree / AST
└────────┬────────┘
         │
         ↓
┌─────────────────┐      🔹 PHASE 3: SEMANTIC ANALYSIS
│ Semantic Check  │ ───► Symbol Table, Type Checking
└────────┬────────┘
         │
         ↓
┌─────────────────┐      🔹 PHASE 4: IR GENERATION
│  IR Generator   │ ───► 3-Address Code (TAC)
└────────┬────────┘
         │
         ↓
┌─────────────────┐      🔹 PHASE 5: OPTIMIZATION
│   Optimizer     │ ───► Optimized IR
└────────┬────────┘
         │
         ↓
┌─────────────────┐      🔹 PHASE 6: CODE GENERATION
│ Code Generator  │ ───► Python Code (train.py)
└────────┬────────┘
         │
         ↓
┌─────────────────┐      🔹 PHASE 7: LINKING & ASSEMBLY
│ Linker/Assembly │ ───► Executable + venv
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Final Output   │
│  train.py       │
│  venv/          │
└─────────────────┘
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `make -f Makefile.verbose` | Build verbose compiler |
| `./mlc_compiler_verbose -v file.mlc` | Compile with all phases shown |
| `./mlc_compiler_verbose file.mlc` | Compile without verbose output |
| `./RUN_VERBOSE_DEMO.sh` | Run full demonstration |
| `cat train.py` | View generated Python code |
| `venv/bin/python train.py` | Run the generated training script |

---

## Key Differences: Regular vs Verbose Mode

| Feature | Regular Mode | Verbose Mode (`-v`) |
|---------|--------------|---------------------|
| Token Display | ❌ Hidden | ✅ All tokens shown |
| Parse Tree | ❌ Hidden | ✅ Visual tree shown |
| Symbol Table | ❌ Hidden | ✅ Complete table shown |
| IR Code | ❌ Hidden | ✅ 3-address code shown |
| Optimization | ❌ Silent | ✅ Before/after shown |
| Code Mapping | ❌ Hidden | ✅ IR→Python mapping shown |
| Library Linking | ❌ Brief message | ✅ Detailed linking info |

---

## Educational Value

This verbose compilation mode is perfect for:

✅ **Learning compiler design** - See how each phase transforms the code  
✅ **Debugging** - Identify which phase causes errors  
✅ **Understanding ML frameworks** - See how high-level configs map to framework code  
✅ **Teaching** - Demonstrate compiler theory with real examples  

---

## Next Steps

1. **Build the compiler**: `make -f Makefile.verbose`
2. **Run the demo**: `./RUN_VERBOSE_DEMO.sh`
3. **Try your own code**: Create a `.mlc` file and compile with `-v` flag
4. **Compare outputs**: Run with and without `-v` to see the difference

---

## Files Created

- `lexer_verbose.l` - Lexer with token printing
- `parser_verbose.y` - Parser with phase integration
- `compiler_phases.h` - Phase function declarations
- `compiler_phases.c` - Phase implementations
- `main_verbose.c` - Main program with verbose flag
- `Makefile.verbose` - Build configuration
- `example_verbose.mlc` - Example input
- `RUN_VERBOSE_DEMO.sh` - Quick demo script
- `COMPILATION_PHASES_GUIDE.md` - This guide

---

**Enjoy exploring the compilation phases! 🚀**
