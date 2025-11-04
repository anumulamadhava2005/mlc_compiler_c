# 🎨 MLC Compiler - Custom Models Guide

## 🆕 What's New?

You can now create **ANY scikit-learn model** without being limited to pre-defined ones! Just specify the `backend` parameter and use any valid scikit-learn model name.

---

## 🚀 Quick Example

### Old Way (Limited to Pre-defined Models)
```mlc
dataset "./data.csv"

model RandomForestClassifier {
    n_estimators = 100
    max_depth = 10
}
```

### New Way (ANY Model You Want!)
```mlc
dataset "./data.csv"

model GradientBoostingClassifier {
    backend = sklearn
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 3
}
```

---

## 📋 Syntax

```mlc
dataset "path/to/data.csv"

model YourCustomModel {
    backend = sklearn          # Required for custom models
    param1 = value1
    param2 = value2
    # ... any parameters the model accepts
}
```

### Key Points:
- ✅ **`backend = sklearn`** - Required for custom models
- ✅ **Model name** must be a valid scikit-learn class name
- ✅ **Parameters** are passed directly to the model constructor
- ✅ **Automatic detection** of classifier vs regressor
- ✅ **Works with predict.py** and web IDE testing

---

## 🎓 Supported Model Categories

The compiler searches these scikit-learn modules:

### 1. **Ensemble Methods** (`sklearn.ensemble`)
```mlc
model AdaBoostClassifier {
    backend = sklearn
    n_estimators = 50
    learning_rate = 1.0
}

model GradientBoostingRegressor {
    backend = sklearn
    n_estimators = 100
    learning_rate = 0.1
}

model ExtraTreesClassifier {
    backend = sklearn
    n_estimators = 100
    max_depth = 10
}
```

### 2. **Linear Models** (`sklearn.linear_model`)
```mlc
model Ridge {
    backend = sklearn
    alpha = 1.0
    max_iter = 1000
}

model Lasso {
    backend = sklearn
    alpha = 0.1
}

model ElasticNet {
    backend = sklearn
    alpha = 1.0
    l1_ratio = 0.5
}

model LogisticRegression {
    backend = sklearn
    C = 1.0
    max_iter = 100
}

model SGDClassifier {
    backend = sklearn
    loss = hinge
    max_iter = 1000
}
```

### 3. **Tree-Based Models** (`sklearn.tree`)
```mlc
model DecisionTreeClassifier {
    backend = sklearn
    max_depth = 10
    min_samples_split = 2
}

model DecisionTreeRegressor {
    backend = sklearn
    max_depth = 5
}
```

### 4. **Support Vector Machines** (`sklearn.svm`)
```mlc
model SVR {
    backend = sklearn
    kernel = rbf
    C = 1.0
}

model NuSVC {
    backend = sklearn
    nu = 0.5
    kernel = rbf
}
```

### 5. **Nearest Neighbors** (`sklearn.neighbors`)
```mlc
model KNeighborsClassifier {
    backend = sklearn
    n_neighbors = 5
    weights = uniform
}

model KNeighborsRegressor {
    backend = sklearn
    n_neighbors = 5
}

model RadiusNeighborsClassifier {
    backend = sklearn
    radius = 1.0
}
```

### 6. **Naive Bayes** (`sklearn.naive_bayes`)
```mlc
model GaussianNB {
    backend = sklearn
}

model MultinomialNB {
    backend = sklearn
    alpha = 1.0
}

model BernoulliNB {
    backend = sklearn
    alpha = 1.0
}
```

### 7. **Clustering** (`sklearn.cluster`)
```mlc
model KMeans {
    backend = sklearn
    n_clusters = 8
    random_state = 42
}

model DBSCAN {
    backend = sklearn
    eps = 0.5
    min_samples = 5
}

model AgglomerativeClustering {
    backend = sklearn
    n_clusters = 2
}
```

### 8. **Neural Networks** (`sklearn.neural_network`)
```mlc
model MLPClassifier {
    backend = sklearn
    hidden_layer_sizes = 100
    activation = relu
    solver = adam
    max_iter = 500
}

model MLPRegressor {
    backend = sklearn
    hidden_layer_sizes = 100
    activation = relu
}
```

### 9. **Discriminant Analysis** (`sklearn.discriminant_analysis`)
```mlc
model QuadraticDiscriminantAnalysis {
    backend = sklearn
}
```

---

## 🔧 How It Works

### 1. **Compile Your MLC File**
```bash
./mlc_compiler your_custom_model.mlc
```

### 2. **Generated Python Code**

The compiler creates smart code that:

```python
# Dynamic import - searches for your model
import importlib

model_class = None
sklearn_modules = [
    'sklearn.ensemble',
    'sklearn.linear_model',
    # ... etc
]

for module_name in sklearn_modules:
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, 'YourModel'):
            model_class = getattr(module, 'YourModel')
            print(f'✓ Found {model_class.__name__}')
            break
    except:
        continue

# Create instance with your parameters
model = model_class(param1=value1, param2=value2)

# Train
model.fit(X_train, y_train)

# Auto-detect classifier vs regressor
is_classifier = hasattr(model, 'predict_proba') or 'Classifier' in model_class.__name__

if is_classifier:
    # Show accuracy metrics
    accuracy = accuracy_score(y_test, y_pred)
else:
    # Show regression metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
```

### 3. **Train the Model**
```bash
venv/bin/python train.py
```

Output:
```
✓ Found GradientBoostingClassifier in sklearn.ensemble
🚀 Starting training...
✅ Training completed!
📊 Accuracy: 0.9600
💾 Model saved as model.pkl
```

### 4. **Test the Model**
```bash
python3 predict.py
```

Or use the web IDE prediction panel!

---

## 💡 Pro Tips

### Tip 1: Check Parameter Names
Visit scikit-learn docs to find valid parameters:
https://scikit-learn.org/stable/modules/classes.html

### Tip 2: String vs Number Values
```mlc
model SVC {
    backend = sklearn
    kernel = rbf       # String (no quotes needed)
    C = 1.0            # Float
    gamma = auto       # String
    random_state = 42  # Integer
}
```

### Tip 3: Boolean Values
```mlc
model Ridge {
    backend = sklearn
    fit_intercept = true    # Becomes Python True
    normalize = false       # Becomes Python False
}
```

### Tip 4: Mix Pre-defined and Custom
```mlc
dataset "./data.csv"

# Pre-defined model (auto-detected)
model SVM {
    kernel = linear
    C = 1.0
}

# Custom model (explicit backend)
model GradientBoostingClassifier {
    backend = sklearn
    n_estimators = 100
}
```

---

## 📊 Complete Examples

### Example 1: Gradient Boosting
```mlc
dataset "./train.csv"

model GradientBoostingClassifier {
    backend = sklearn
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 3
    min_samples_split = 2
    min_samples_leaf = 1
    subsample = 1.0
    random_state = 42
}
```

**Compile & Run:**
```bash
./mlc_compiler example_gradient_boost.mlc
venv/bin/python train.py
```

### Example 2: Neural Network (MLP)
```mlc
dataset "./neural_data.csv"

model MLPClassifier {
    backend = sklearn
    hidden_layer_sizes = 100
    activation = relu
    solver = adam
    alpha = 0.0001
    batch_size = auto
    learning_rate = constant
    learning_rate_init = 0.001
    max_iter = 500
    random_state = 42
}
```

### Example 3: K-Nearest Neighbors
```mlc
dataset "./knn_data.csv"

model KNeighborsClassifier {
    backend = sklearn
    n_neighbors = 5
    weights = distance
    algorithm = auto
    leaf_size = 30
    p = 2
}
```

### Example 4: Ridge Regression
```mlc
dataset "./regression_data.csv"

model Ridge {
    backend = sklearn
    alpha = 1.0
    fit_intercept = true
    max_iter = 1000
    tol = 0.0001
    solver = auto
}
```

---

## 🐛 Troubleshooting

### Error: "Could not find model: XYZ"
**Cause:** Model name doesn't exist in scikit-learn  
**Fix:** Check spelling, use exact scikit-learn class name

```mlc
# ❌ Wrong
model RandomForest {
    backend = sklearn
}

# ✅ Correct
model RandomForestClassifier {
    backend = sklearn
}
```

### Error: "TypeError: __init__() got unexpected keyword"
**Cause:** Invalid parameter name  
**Fix:** Check scikit-learn docs for valid parameters

### Model not saving
**Cause:** Training failed  
**Fix:** Check dataset path and format

---

## 🎯 Comparison

| Feature | Pre-defined Models | Custom Models |
|---------|-------------------|---------------|
| **Syntax** | `model SVM { ... }` | `model SVC { backend=sklearn ... }` |
| **Flexibility** | Limited to ~10 models | ALL sklearn models |
| **Auto-import** | Yes | Yes (dynamic) |
| **Parameters** | Model-specific | Any valid params |
| **Backend** | Auto-detected | Explicit |
| **Testing** | Full support | Full support |

---

## 📚 Resources

- **Scikit-learn Models**: https://scikit-learn.org/stable/supervised_learning.html
- **Parameter Reference**: https://scikit-learn.org/stable/modules/classes.html
- **MLC Compiler Docs**: README.md
- **Web IDE Guide**: web-ide/README_PREDICTION.md

---

## 🚀 Next Steps

1. **Explore**: Try different scikit-learn models
2. **Tune**: Experiment with parameters
3. **Compare**: Test multiple models on same dataset
4. **Deploy**: Use web IDE for easy testing

---

**Happy modeling! You now have access to 100+ scikit-learn models! 🎉**
