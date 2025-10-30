# Scikit-Learn Models Support in MLC Compiler

The MLC compiler now supports **10 scikit-learn models** with automatic code generation!

## ✅ Supported Models

### 🧩 1. LinearRegression
**Type:** Regression  
**Library:** `sklearn.linear_model`  
**Use Case:** Predicting continuous values (house price, temperature, etc.)

**Example:**
```mlc
dataset "/home/madhava/datasets/housing.csv"

model LinearRegression {
    fit_intercept = true
    normalize = false
}
```

**Generated Metrics:** MSE, R² Score

---

### 🧠 2. LogisticRegression
**Type:** Classification  
**Library:** `sklearn.linear_model`  
**Use Case:** Binary / Multiclass classification

**Example:**
```mlc
dataset "/home/madhava/datasets/classification.csv"

model LogisticRegression {
    max_iter = 200
    solver = "lbfgs"
}
```

**Generated Metrics:** Accuracy, Classification Report

---

### 🌳 3. DecisionTreeClassifier
**Type:** Classification  
**Library:** `sklearn.tree`  
**Use Case:** Simple classification with interpretable rules

**Example:**
```mlc
dataset "/home/madhava/datasets/iris.csv"

model DecisionTreeClassifier {
    max_depth = 5
    criterion = "entropy"
}
```

**Generated Metrics:** Accuracy, Classification Report

---

### 🌲 4. RandomForestClassifier
**Type:** Classification  
**Library:** `sklearn.ensemble`  
**Use Case:** Robust ensemble classification

**Example:**
```mlc
dataset "/home/madhava/datasets/classification.csv"

model RandomForestClassifier {
    n_estimators = 100
    max_depth = 4
}
```

**Generated Metrics:** Accuracy, Classification Report

---

### 📈 5. KNeighborsClassifier
**Type:** Classification  
**Library:** `sklearn.neighbors`  
**Use Case:** Classification using nearest-neighbor distances

**Example:**
```mlc
dataset "/home/madhava/datasets/iris.csv"

model KNeighborsClassifier {
    n_neighbors = 3
    metric = "euclidean"
}
```

**Generated Metrics:** Accuracy, Classification Report

---

### ⚙️ 6. SVC (Support Vector Classifier)
**Type:** Classification  
**Library:** `sklearn.svm`  
**Use Case:** Classification with kernels

**Example:**
```mlc
dataset "/home/madhava/datasets/iris.csv"

model SVC {
    kernel = "rbf"
    C = 1.0
}
```

**Generated Metrics:** Accuracy, Classification Report

---

### 📊 7. GaussianNB
**Type:** Classification  
**Library:** `sklearn.naive_bayes`  
**Use Case:** Text or categorical classification (simple model!)

**Example:**
```mlc
dataset "/home/madhava/datasets/classification.csv"

model GaussianNB { }
```

**Generated Metrics:** Accuracy, Classification Report

---

### 🔢 8. KMeans
**Type:** Clustering (unsupervised)  
**Library:** `sklearn.cluster`  
**Use Case:** Grouping unlabeled data

**Example:**
```mlc
dataset "/home/madhava/datasets/clustering.csv"

model KMeans {
    n_clusters = 3
    max_iter = 300
}
```

**Generated Metrics:** Silhouette Score, Inertia

---

### 🧮 9. LinearSVC
**Type:** Classification  
**Library:** `sklearn.svm`  
**Use Case:** Faster linear SVM

**Example:**
```mlc
dataset "/home/madhava/datasets/classification.csv"

model LinearSVC {
    C = 1.0
    max_iter = 2000
}
```

**Generated Metrics:** Accuracy, Classification Report

---

### 🧩 10. SGDClassifier
**Type:** Classification (incremental training)  
**Library:** `sklearn.linear_model`  
**Use Case:** Fast online training on large datasets

**Example:**
```mlc
dataset "/home/madhava/datasets/classification.csv"

model SGDClassifier {
    loss = "log_loss"
    max_iter = 1000
    alpha = 0.0001
}
```

**Generated Metrics:** Accuracy, Classification Report

---

## 🚀 Quick Start

### 1. Create an `.mlc` file
```mlc
dataset "/path/to/your/data.csv"

model RandomForestClassifier {
    n_estimators = 100
    max_depth = 4
}
```

### 2. Compile
```bash
./mlc_compiler your_model.mlc
```

### 3. Run
```bash
venv/bin/python train.py
```

---

## 📋 Generated Code Features

For sklearn models, the compiler automatically generates:

1. **Data Loading**
   - CSV file reading with pandas
   - Feature/target separation (last column = target)
   - Train/test split (80/20)

2. **Model Initialization**
   - Correct import statements
   - Parameter passing with proper Python formatting
   - Boolean values: `true` → `True`, `false` → `False`
   - String values: Automatically quoted

3. **Training**
   - `model.fit(X_train, y_train)` call
   - Progress indicators

4. **Evaluation**
   - Model-specific metrics (MSE/R² for regression, Accuracy for classification)
   - Classification reports
   - Clustering metrics (Silhouette score, Inertia)

5. **Model Saving**
   - Serialization with `joblib`
   - Saved as `model.pkl`

---

## 🎯 Dataset Format

For sklearn models, the compiler expects CSV files with:
- **Features:** All columns except the last
- **Target:** Last column
- **Header:** Optional (pandas will auto-detect)

**Example CSV:**
```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
```

---

## 🔧 Parameter Handling

The compiler intelligently converts parameter values:

| MLC Input | Python Output | Type |
|-----------|---------------|------|
| `true` | `True` | Boolean |
| `false` | `False` | Boolean |
| `"lbfgs"` | `"lbfgs"` | String |
| `lbfgs` | `"lbfgs"` | String (auto-quoted) |
| `100` | `100` | Integer |
| `0.001` | `0.001` | Float |

---

## 📦 Automatic Dependencies

When using sklearn models, the compiler installs:
- `scikit-learn`
- `pandas`
- `joblib`

---

## 🎓 Model Selection Guide

| Task | Recommended Model | Why |
|------|------------------|-----|
| **Simple Regression** | LinearRegression | Fast, interpretable |
| **Binary Classification** | LogisticRegression | Probabilistic outputs |
| **Complex Classification** | RandomForestClassifier | Handles non-linearity |
| **Large Datasets** | SGDClassifier | Memory efficient |
| **Clustering** | KMeans | Simple, fast |
| **Text Classification** | GaussianNB | Works well with high dimensions |

---

## ✨ Full Backend Support

MLC Compiler now supports **4 backends**:

1. **sklearn** - 10 models (this document)
2. **tensorflow** - Deep learning CNNs
3. **pytorch** - Custom neural networks
4. **transformers** - NLP models

---

**Status:** ✅ All 10 sklearn models fully implemented and tested!
