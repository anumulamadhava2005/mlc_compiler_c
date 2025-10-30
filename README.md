# MLC Compiler - Machine Learning Configuration Compiler

A C-based compiler that translates ML configuration files (`.mlc`) into executable Python training scripts with automatic framework detection and virtual environment setup.

## Features

- ✅ **Multi-Framework Support**: Automatically selects and generates code for:
  - **Scikit-Learn**: LinearRegression, LogisticRegression, DecisionTree, RandomForest, KNeighbors, SVC, GaussianNB, KMeans, LinearSVC, SGDClassifier (10 models!)
  - **TensorFlow/Keras**: ResNet, VGG, EfficientNet, MobileNet, DenseNet, InceptionV3
  - **PyTorch**: UNet, GAN, AutoEncoder, VAE
  - **Transformers**: BERT, GPT, T5, RoBERTa, DistilBERT

- ✅ **Automatic Dataset Loading**: Generates appropriate dataset loading code based on backend
- ✅ **Virtual Environment Management**: Creates venv and installs required packages automatically
- ✅ **Flexible Configuration**: Support for custom hyperparameters (epochs, batch_size, learning_rate, etc.)

## Project Structure

```
~/mlc_compiler_c/
├── ast.h              # AST structure definitions
├── lexer.l            # Flex lexer for tokenization
├── parser.y           # Bison parser with code generation
├── main.c             # Main compiler driver
├── Makefile           # Build configuration
├── test.mlc           # TensorFlow example (ResNet50)
├── test_pytorch.mlc   # PyTorch example (UNet)
├── test_transformer.mlc # Transformers example (BERT)
└── README.md          # This file
```

## Installation

### Prerequisites
```bash
sudo apt update
sudo apt install flex bison python3-venv python3-pip gcc -y
```

### Build
```bash
cd ~/mlc_compiler_c
make clean
make
```

## Usage

### 1. Create an MLC configuration file

**Example: Scikit-Learn (test_sklearn_forest.mlc)**
```mlc
dataset "/home/madhava/datasets/classification.csv"

model RandomForestClassifier {
    n_estimators = 100
    max_depth = 4
}
```

**Example: TensorFlow (test.mlc)**
```mlc
dataset "/home/madhava/datasets/flowers"

model ResNet50 {
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
}
```

**Example: PyTorch (test_pytorch.mlc)**
```mlc
dataset "/home/madhava/datasets/images"

model UNet {
    epochs = 20
    batch_size = 16
    learning_rate = 0.0001
}
```

**Example: Transformers (test_transformer.mlc)**
```mlc
dataset "imdb"

model BERT {
    epochs = 3
    batch_size = 8
    learning_rate = 0.00002
}
```

### 2. Compile the configuration
```bash
./mlc_compiler test.mlc
```

**Expected Output:**
```
📂 Dataset path set to: /home/madhava/datasets/flowers
✅ Parsing completed successfully!
✅ Python script 'train.py' generated successfully!
🔧 Setting up virtual environment and installing packages...
   Creating virtual environment...
   Upgrading pip...
   Installing tensorflow and dependencies...
✅ Virtual environment ready with tensorflow installed!

📋 To run training:
   venv/bin/python train.py
```

### 3. Run the generated training script
```bash
venv/bin/python train.py
```

## Backend Selection Logic

The compiler automatically detects which framework to use based on model name:

| Model Type | Backend | Examples |
|------------|---------|----------|
| **Classical ML** | **Scikit-Learn** | **LinearRegression, LogisticRegression, DecisionTree, RandomForest, KNeighbors, SVC, GaussianNB, KMeans, LinearSVC, SGDClassifier** |
| CNNs | TensorFlow | ResNet, VGG, EfficientNet, MobileNet, DenseNet, InceptionV3 |
| Generative Models | PyTorch | UNet, GAN, AutoEncoder, VAE |
| Language Models | Transformers | BERT, GPT, T5, RoBERTa, DistilBERT |

## Configuration Parameters

Supported hyperparameters (all optional with defaults):
- `epochs` (default: 10)
- `batch_size` (default: 32)
- `learning_rate` (default: 0.001)

## Generated Code Features

### TensorFlow Backend
- Image dataset loading from directory
- Transfer learning with pretrained weights
- Model compilation with Adam optimizer
- Training loop with progress tracking
- Model saving in H5 format

### PyTorch Backend
- ImageFolder dataset with transforms
- Custom model architecture
- Training loop with loss tracking
- Model saving as state dict

### Transformers Backend
- HuggingFace dataset loading
- Tokenization pipeline
- Trainer API integration
- Model checkpointing

## Makefile Commands

```bash
make           # Build the compiler
make clean     # Remove generated files
```

## Architecture

1. **Lexer (lexer.l)**: Tokenizes `.mlc` files into tokens (DATASET, MODEL, ID, INT, FLOAT, STRING, etc.)
2. **Parser (parser.y)**: Parses tokens into an AST and generates Python code
3. **AST (ast.h)**: Defines data structures for storing program configuration
4. **Main (main.c)**: Entry point that reads input and invokes the parser

## Code Generation Pipeline

```
.mlc file → Lexer → Parser → AST → Backend Detection → Python Code Generation → venv Setup
```

## Examples of Generated Code

The compiler generates complete, runnable Python scripts with:
- Proper imports for the detected framework
- Dataset loading with appropriate transformations
- Model initialization (with pretrained weights where applicable)
- Compilation/optimizer setup
- Full training loop
- Model saving/checkpointing

## Troubleshooting

**Virtual environment issues:**
```bash
# Manual setup if auto-setup fails
python3 -m venv venv
source venv/bin/activate
pip install tensorflow  # or torch, or transformers
python train.py
```

**Parser errors:**
- Ensure proper syntax in `.mlc` files
- Check for matching braces `{}`
- Verify quotes around strings

## License

MIT License - Educational project for compiler design and ML automation.

## Author

Built as a demonstration of compiler design principles applied to machine learning workflow automation.
