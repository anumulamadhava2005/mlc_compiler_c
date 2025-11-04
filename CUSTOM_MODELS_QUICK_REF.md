# 🎨 Custom Models - Quick Reference

## Basic Syntax
```mlc
dataset "path/to/data.csv"

model AnySklearnModel {
    backend = sklearn
    parameter1 = value1
    parameter2 = value2
}
```

## Common Examples

### Gradient Boosting
```mlc
model GradientBoostingClassifier {
    backend = sklearn
    n_estimators = 100
    learning_rate = 0.1
    max_depth = 3
}
```

### K-Nearest Neighbors
```mlc
model KNeighborsClassifier {
    backend = sklearn
    n_neighbors = 5
    weights = uniform
}
```

### Neural Network
```mlc
model MLPClassifier {
    backend = sklearn
    hidden_layer_sizes = 100
    activation = relu
    max_iter = 500
}
```

### Ridge Regression
```mlc
model Ridge {
    backend = sklearn
    alpha = 1.0
    fit_intercept = true
}
```

### AdaBoost
```mlc
model AdaBoostClassifier {
    backend = sklearn
    n_estimators = 50
    learning_rate = 1.0
}
```

## Workflow

1. **Create** `.mlc` file with custom model
2. **Compile**: `./mlc_compiler your_file.mlc`
3. **Train**: `venv/bin/python train.py`
4. **Test**: Web IDE or `python3 predict.py`

## Value Types

- **Integers**: `n_estimators = 100`
- **Floats**: `learning_rate = 0.1`
- **Strings**: `kernel = rbf` (no quotes)
- **Booleans**: `fit_intercept = true`

## All Supported Models

✅ **100+ models** from:
- sklearn.ensemble
- sklearn.linear_model
- sklearn.tree
- sklearn.svm
- sklearn.neighbors
- sklearn.naive_bayes
- sklearn.cluster
- sklearn.neural_network
- sklearn.discriminant_analysis

See `CUSTOM_MODELS_GUIDE.md` for complete list!
