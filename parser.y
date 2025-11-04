%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"
#include "compiler_phases.h"

int yylex(void);
void yyerror(const char *s);

// Global program storage
Program prog = { .dataset_path = "", .model_count = 0 };

// Symbol table and IR
SymbolTable symtab;
IRCode ir_code;

// Forward declarations
void generate_python();
const char* detect_backend(const char* model_name);
char* strip_quotes(char* str);

extern int verbose_mode;

%}

%union {
    int intVal;
    float floatVal;
    char *strVal;
}

%token DATASET MODEL LBRACE RBRACE ASSIGN LBRACKET RBRACKET COMMA
%token <intVal> INT
%token <floatVal> FLOAT
%token <strVal> ID STRING

%type <strVal> value

%%

program:
    dataset_decl model_def_list
    {
        if (verbose_mode) {
            printf("\n");  // Close lexical analysis output
            phase2_syntax_analysis(&prog);
            phase3_semantic_analysis(&prog, &symtab);
            phase4_ir_generation(&prog, &ir_code);
            phase5_optimization(&ir_code);
        }
        
        printf("\n✅ Parsing completed successfully!\n");
        generate_python();
    }
    | model_def_list
    {
        if (verbose_mode) {
            printf("\n");
            phase2_syntax_analysis(&prog);
            phase3_semantic_analysis(&prog, &symtab);
            phase4_ir_generation(&prog, &ir_code);
            phase5_optimization(&ir_code);
        }
        
        printf("\n✅ Parsing completed successfully!\n");
        generate_python();
    }
    ;

dataset_decl:
    DATASET STRING
    {
        char* path = strip_quotes($2);
        strncpy(prog.dataset_path, path, 255);
        prog.dataset_path[255] = '\0';
        if (!verbose_mode) {
            printf("📂 Dataset path set to: %s\n", prog.dataset_path);
        }
        free(path);
        free($2);
    }
    ;

model_def_list:
    model_def_list model_def
    | model_def
    ;

model_def:
    MODEL ID {
        Model *m = &prog.models[prog.model_count];
        strncpy(m->name, $2, 63);
        m->name[63] = '\0';
        m->param_count = 0;
        m->backend[0] = '\0';  // Initialize backend as empty
        free($2);
    } LBRACE param_list RBRACE
    {
        prog.model_count++;
    }
    ;

param_list:
    param_list param
    | param
    | /* empty */
    ;

param:
    ID ASSIGN value
    {
        Model *m = &prog.models[prog.model_count];
        
        // Check if this is the "backend" parameter
        if (strcmp($1, "backend") == 0) {
            strncpy(m->backend, $3, 63);
            m->backend[63] = '\0';
        } else {
            // Regular parameter
            int idx = m->param_count;
            strncpy(m->param_names[idx], $1, 63);
            m->param_names[idx][63] = '\0';
            
            strncpy(m->param_values[idx], $3, 63);
            m->param_values[idx][63] = '\0';
            
            m->param_count++;
        }
        
        free($1);
        free($3);
    }
    ;

value:
    INT    { 
        char buf[64]; 
        sprintf(buf, "%d", $1);
        $$ = strdup(buf);
    }
    | FLOAT  { 
        char buf[64]; 
        sprintf(buf, "%.6f", $1);
        $$ = strdup(buf);
    }
    | STRING { 
        $$ = strip_quotes($1);
        free($1);
    }
    | ID {
        $$ = strdup($1);
        free($1);
    }
    ;

%%

void yyerror(const char *s) {
    fprintf(stderr, "❌ Parse error: %s\n", s);
}

char* strip_quotes(char* str) {
    int len = strlen(str);
    char* result = malloc(len + 1);
    if (len >= 2 && str[0] == '"' && str[len-1] == '"') {
        strncpy(result, str + 1, len - 2);
        result[len - 2] = '\0';
    } else {
        strcpy(result, str);
    }
    return result;
}

const char* detect_backend(const char* model_name) {
    // Scikit-learn models
    if (strstr(model_name, "LinearRegression") || strstr(model_name, "LogisticRegression") ||
        strstr(model_name, "DecisionTree") || strstr(model_name, "RandomForest") ||
        strstr(model_name, "KNeighbors") || strstr(model_name, "SVC") ||
        strstr(model_name, "GaussianNB") || strstr(model_name, "KMeans") ||
        strstr(model_name, "LinearSVC") || strstr(model_name, "SGDClassifier") ||
        strstr(model_name, "SVM") || strstr(model_name, "LDA")) {
        return "sklearn";
    }
    
    // TensorFlow models
    if (strstr(model_name, "ResNet") || strstr(model_name, "VGG") ||
        strstr(model_name, "EfficientNet") || strstr(model_name, "MobileNet") ||
        strstr(model_name, "DenseNet") || strstr(model_name, "InceptionV3")) {
        return "tensorflow";
    }
    
    // PyTorch models
    if (strstr(model_name, "UNet") || strstr(model_name, "GAN") || 
        strstr(model_name, "AutoEncoder") || strstr(model_name, "VAE")) {
        return "pytorch";
    }
    
    // Transformers models
    if (strstr(model_name, "BERT") || strstr(model_name, "GPT") || 
        strstr(model_name, "T5") || strstr(model_name, "RoBERTa") ||
        strstr(model_name, "DistilBERT")) {
        return "transformers";
    }
    
    return "tensorflow";
}

void generate_python() {
    FILE *fp = fopen("train.py", "w");
    if (!fp) {
        fprintf(stderr, "Error creating train.py\n");
        return;
    }

    fprintf(fp, "#!/usr/bin/env python3\n");
    fprintf(fp, "# Generated by MLC Compiler\n");
    fprintf(fp, "# Auto-generated machine learning training script\n\n");

    // Determine primary backend for verbose mode
    const char* primary_backend = "sklearn";
    if (prog.model_count > 0) {
        if (strlen(prog.models[0].backend) > 0) {
            primary_backend = prog.models[0].backend;
        } else {
            primary_backend = detect_backend(prog.models[0].name);
        }
    }

    // If multiple models, add comparison tracking
    if (prog.model_count > 1) {
        fprintf(fp, "# Multiple models - tracking for comparison\n");
        fprintf(fp, "import pandas as pd\n");
        fprintf(fp, "from sklearn.model_selection import train_test_split\n");
        fprintf(fp, "import joblib\n\n");
        fprintf(fp, "model_results = []\n\n");
    }

    // Generate code for each model
    for (int i = 0; i < prog.model_count; i++) {
        Model *m = &prog.models[i];
        
        // Determine backend: user-specified or auto-detect
        const char* backend;
        if (strlen(m->backend) > 0) {
            backend = m->backend;  // Use user-specified backend
        } else {
            backend = detect_backend(m->name);  // Auto-detect from model name
        }
        
        if (verbose_mode && i == 0) {
            phase6_code_generation(&prog, backend);
        }
        
        fprintf(fp, "# =====================================\n");
        fprintf(fp, "# Model %d: %s\n", i + 1, m->name);
        fprintf(fp, "# Backend: %s\n", backend);
        fprintf(fp, "# Parameters: ");
        for (int j = 0; j < m->param_count; j++) {
            fprintf(fp, "%s=%s", m->param_names[j], m->param_values[j]);
            if (j < m->param_count - 1) fprintf(fp, ", ");
        }
        fprintf(fp, "\n# =====================================\n\n");

        // Simple sklearn code generation
        if (strcmp(backend, "sklearn") == 0) {
            fprintf(fp, "# Imports\n");
            fprintf(fp, "import pandas as pd\n");
            fprintf(fp, "from sklearn.model_selection import train_test_split\n");
            fprintf(fp, "import joblib\n");
            
            // Add model-specific imports
            if (strstr(m->name, "RandomForest")) {
                fprintf(fp, "from sklearn.ensemble import RandomForestClassifier\n");
                fprintf(fp, "from sklearn.metrics import accuracy_score, classification_report\n");
            } else if (strstr(m->name, "LinearRegression")) {
                fprintf(fp, "from sklearn.linear_model import LinearRegression\n");
                fprintf(fp, "from sklearn.metrics import mean_squared_error, r2_score\n");
            } else if (strcmp(m->name, "SVM") == 0 || strcmp(m->name, "SVC") == 0) {
                fprintf(fp, "from sklearn.svm import SVC\n");
                fprintf(fp, "from sklearn.metrics import accuracy_score, classification_report\n");
            } else if (strcmp(m->name, "LDA") == 0) {
                fprintf(fp, "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n");
                fprintf(fp, "from sklearn.metrics import accuracy_score, classification_report\n");
            }
            fprintf(fp, "\n");
            
            fprintf(fp, "# Load dataset\n");
            fprintf(fp, "dataset = pd.read_csv('%s')\n", prog.dataset_path);
            fprintf(fp, "X = dataset.iloc[:, :-1].values\n");
            fprintf(fp, "y = dataset.iloc[:, -1].values\n\n");
            
            fprintf(fp, "# Split dataset\n");
            fprintf(fp, "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n");
            
            fprintf(fp, "# Model: %s\n", m->name);
            
            if (strstr(m->name, "RandomForest")) {
                fprintf(fp, "model = RandomForestClassifier(");
                for (int j = 0; j < m->param_count; j++) {
                    fprintf(fp, "%s=%s", m->param_names[j], m->param_values[j]);
                    if (j < m->param_count - 1) fprintf(fp, ", ");
                }
                fprintf(fp, ")\n\n");
                fprintf(fp, "print('🚀 Starting training...')\n");
                fprintf(fp, "model.fit(X_train, y_train)\n");
                fprintf(fp, "print('✅ Training completed!')\n\n");
                fprintf(fp, "y_pred = model.predict(X_test)\n");
                fprintf(fp, "accuracy = accuracy_score(y_test, y_pred)\n");
                fprintf(fp, "print(f'📊 Accuracy: {accuracy:.4f}')\n");
                fprintf(fp, "print('\\n📋 Classification Report:')\n");
                fprintf(fp, "print(classification_report(y_test, y_pred))\n");
                if (prog.model_count > 1) {
                    fprintf(fp, "model_results.append({'name': '%s', 'accuracy': accuracy, 'type': 'classifier'})\n", m->name);
                }
            } else if (strstr(m->name, "LinearRegression")) {
                fprintf(fp, "model = LinearRegression(");
                for (int j = 0; j < m->param_count; j++) {
                    // Convert 'true'/'false' to 'True'/'False' for Python
                    char value_buf[64];
                    if (strcmp(m->param_values[j], "true") == 0) {
                        strcpy(value_buf, "True");
                    } else if (strcmp(m->param_values[j], "false") == 0) {
                        strcpy(value_buf, "False");
                    } else {
                        strcpy(value_buf, m->param_values[j]);
                    }
                    fprintf(fp, "%s=%s", m->param_names[j], value_buf);
                    if (j < m->param_count - 1) fprintf(fp, ", ");
                }
                fprintf(fp, ")\n\n");
                fprintf(fp, "print('🚀 Starting training...')\n");
                fprintf(fp, "model.fit(X_train, y_train)\n");
                fprintf(fp, "print('✅ Training completed!')\n\n");
                fprintf(fp, "y_pred = model.predict(X_test)\n");
                fprintf(fp, "mse = mean_squared_error(y_test, y_pred)\n");
                fprintf(fp, "r2 = r2_score(y_test, y_pred)\n");
                fprintf(fp, "print(f'📊 Mean Squared Error: {mse:.4f}')\n");
                fprintf(fp, "print(f'📊 R² Score: {r2:.4f}')\n");
                if (prog.model_count > 1) {
                    fprintf(fp, "model_results.append({'name': '%s', 'r2': r2, 'mse': mse, 'type': 'regressor'})\n", m->name);
                }
            } else if (strcmp(m->name, "SVM") == 0 || strcmp(m->name, "SVC") == 0) {
                fprintf(fp, "model = SVC(");
                for (int j = 0; j < m->param_count; j++) {
                    // Add quotes around string values like kernel
                    char value_buf[64];
                    if (m->param_values[j][0] >= 'a' && m->param_values[j][0] <= 'z' && 
                        strchr(m->param_values[j], '.') == NULL) {
                        snprintf(value_buf, sizeof(value_buf), "\"%s\"", m->param_values[j]);
                    } else {
                        strcpy(value_buf, m->param_values[j]);
                    }
                    fprintf(fp, "%s=%s", m->param_names[j], value_buf);
                    if (j < m->param_count - 1) fprintf(fp, ", ");
                }
                fprintf(fp, ")\n\n");
                fprintf(fp, "print('🚀 Starting training...')\n");
                fprintf(fp, "model.fit(X_train, y_train)\n");
                fprintf(fp, "print('✅ Training completed!')\n\n");
                fprintf(fp, "y_pred = model.predict(X_test)\n");
                fprintf(fp, "accuracy = accuracy_score(y_test, y_pred)\n");
                fprintf(fp, "print(f'📊 Accuracy: {accuracy:.4f}')\n");
                fprintf(fp, "print('\\n📋 Classification Report:')\n");
                fprintf(fp, "print(classification_report(y_test, y_pred))\n");
                if (prog.model_count > 1) {
                    fprintf(fp, "model_results.append({'name': '%s', 'accuracy': accuracy, 'type': 'classifier'})\n", m->name);
                }
            } else if (strcmp(m->name, "LDA") == 0) {
                fprintf(fp, "model = LinearDiscriminantAnalysis(");
                for (int j = 0; j < m->param_count; j++) {
                    // Add quotes around string values like solver
                    char value_buf[64];
                    if (m->param_values[j][0] >= 'a' && m->param_values[j][0] <= 'z' && 
                        strchr(m->param_values[j], '.') == NULL) {
                        snprintf(value_buf, sizeof(value_buf), "\"%s\"", m->param_values[j]);
                    } else {
                        strcpy(value_buf, m->param_values[j]);
                    }
                    fprintf(fp, "%s=%s", m->param_names[j], value_buf);
                    if (j < m->param_count - 1) fprintf(fp, ", ");
                }
                fprintf(fp, ")\n\n");
                fprintf(fp, "print('🚀 Starting training...')\n");
                fprintf(fp, "model.fit(X_train, y_train)\n");
                fprintf(fp, "print('✅ Training completed!')\n\n");
                fprintf(fp, "y_pred = model.predict(X_test)\n");
                fprintf(fp, "accuracy = accuracy_score(y_test, y_pred)\n");
                fprintf(fp, "print(f'📊 Accuracy: {accuracy:.4f}')\n");
                fprintf(fp, "print('\\n📋 Classification Report:')\n");
                fprintf(fp, "print(classification_report(y_test, y_pred))\n");
                if (prog.model_count > 1) {
                    fprintf(fp, "model_results.append({'name': '%s', 'accuracy': accuracy, 'type': 'classifier'})\n", m->name);
                }
            } else {
                // Generic sklearn model handler for custom models
                fprintf(fp, "# Custom model - using dynamic import\n");
                fprintf(fp, "# Trying to import: %s\n", m->name);
                fprintf(fp, "import importlib\n");
                fprintf(fp, "from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score\n\n");
                
                fprintf(fp, "# Try to find and import the model\n");
                fprintf(fp, "model_class = None\n");
                fprintf(fp, "sklearn_modules = [\n");
                fprintf(fp, "    'sklearn.ensemble',\n");
                fprintf(fp, "    'sklearn.linear_model',\n");
                fprintf(fp, "    'sklearn.tree',\n");
                fprintf(fp, "    'sklearn.svm',\n");
                fprintf(fp, "    'sklearn.neighbors',\n");
                fprintf(fp, "    'sklearn.naive_bayes',\n");
                fprintf(fp, "    'sklearn.cluster',\n");
                fprintf(fp, "    'sklearn.neural_network',\n");
                fprintf(fp, "    'sklearn.discriminant_analysis',\n");
                fprintf(fp, "]\n\n");
                
                fprintf(fp, "for module_name in sklearn_modules:\n");
                fprintf(fp, "    try:\n");
                fprintf(fp, "        module = importlib.import_module(module_name)\n");
                fprintf(fp, "        if hasattr(module, '%s'):\n", m->name);
                fprintf(fp, "            model_class = getattr(module, '%s')\n", m->name);
                fprintf(fp, "            print(f'✓ Found {model_class.__name__} in {module_name}')\n");
                fprintf(fp, "            break\n");
                fprintf(fp, "    except:\n");
                fprintf(fp, "        continue\n\n");
                
                fprintf(fp, "if model_class is None:\n");
                fprintf(fp, "    raise ImportError(f'Could not find model: %s in scikit-learn')\n\n", m->name);
                
                fprintf(fp, "# Create model instance with parameters\n");
                fprintf(fp, "model = model_class(");
                for (int j = 0; j < m->param_count; j++) {
                    char value_buf[64];
                    // Smart value formatting
                    if (strcmp(m->param_values[j], "true") == 0) {
                        strcpy(value_buf, "True");
                    } else if (strcmp(m->param_values[j], "false") == 0) {
                        strcpy(value_buf, "False");
                    } else if (m->param_values[j][0] >= 'a' && m->param_values[j][0] <= 'z' && 
                               strchr(m->param_values[j], '.') == NULL &&
                               strcmp(m->param_values[j], "true") != 0 && 
                               strcmp(m->param_values[j], "false") != 0) {
                        snprintf(value_buf, sizeof(value_buf), "'%s'", m->param_values[j]);
                    } else {
                        strcpy(value_buf, m->param_values[j]);
                    }
                    fprintf(fp, "%s=%s", m->param_names[j], value_buf);
                    if (j < m->param_count - 1) fprintf(fp, ", ");
                }
                fprintf(fp, ")\n\n");
                
                fprintf(fp, "print('🚀 Starting training...')\n");
                fprintf(fp, "model.fit(X_train, y_train)\n");
                fprintf(fp, "print('✅ Training completed!')\n\n");
                
                fprintf(fp, "# Make predictions\n");
                fprintf(fp, "y_pred = model.predict(X_test)\n\n");
                
                fprintf(fp, "# Try to determine if it's a classifier or regressor\n");
                fprintf(fp, "is_classifier = hasattr(model, 'predict_proba') or 'Classifier' in model_class.__name__\n\n");
                
                fprintf(fp, "if is_classifier:\n");
                fprintf(fp, "    accuracy = accuracy_score(y_test, y_pred)\n");
                fprintf(fp, "    print(f'📊 Accuracy: {accuracy:.4f}')\n");
                fprintf(fp, "    print('\\n📋 Classification Report:')\n");
                fprintf(fp, "    print(classification_report(y_test, y_pred))\n");
                if (prog.model_count > 1) {
                    fprintf(fp, "    model_results.append({'name': '%s', 'accuracy': accuracy, 'type': 'classifier'})\n", m->name);
                }
                fprintf(fp, "else:\n");
                fprintf(fp, "    mse = mean_squared_error(y_test, y_pred)\n");
                fprintf(fp, "    r2 = r2_score(y_test, y_pred)\n");
                fprintf(fp, "    print(f'📊 Mean Squared Error: {mse:.4f}')\n");
                fprintf(fp, "    print(f'📊 R² Score: {r2:.4f}')\n");
                if (prog.model_count > 1) {
                    fprintf(fp, "    model_results.append({'name': '%s', 'r2': r2, 'mse': mse, 'type': 'regressor'})\n", m->name);
                }
            }
            
            fprintf(fp, "\n# Save model\n");
            if (prog.model_count > 1) {
                fprintf(fp, "model_filename = 'model_%s.pkl'\n", m->name);
                fprintf(fp, "joblib.dump(model, model_filename)\n");
                fprintf(fp, "print(f'💾 Model saved as {model_filename}')\n");
            } else {
                fprintf(fp, "joblib.dump(model, 'model.pkl')\n");
                fprintf(fp, "print('💾 Model saved as model.pkl')\n");
            }
        }
        // Transformers backend
        else if (strcmp(backend, "transformers") == 0) {
            fprintf(fp, "# Dataset Loading\n");
            fprintf(fp, "from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments\n");
            fprintf(fp, "from datasets import load_dataset\n\n");
            fprintf(fp, "# Load dataset\n");
            fprintf(fp, "dataset = load_dataset('%s')\n", prog.dataset_path);
            fprintf(fp, "tokenizer = AutoTokenizer.from_pretrained(model_name)\n\n");
            fprintf(fp, "def tokenize_function(examples):\n");
            fprintf(fp, "    return tokenizer(examples['text'], padding='max_length', truncation=True)\n\n");
            fprintf(fp, "tokenized_datasets = dataset.map(tokenize_function, batched=True)\n\n");
            fprintf(fp, "# Model: %s (transformers backend)\n", m->name);
            fprintf(fp, "from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer\n\n");
            fprintf(fp, "# Model\n");
            fprintf(fp, "model_name = '%s'\n", m->name);
            fprintf(fp, "model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)\n\n");
            fprintf(fp, "# Training\n");
            fprintf(fp, "training_args = TrainingArguments(\n");
            fprintf(fp, "    output_dir='./results',\n");
            fprintf(fp, "    num_train_epochs=epochs,\n");
            fprintf(fp, "    per_device_train_batch_size=batch_size,\n");
            fprintf(fp, "    learning_rate=learning_rate,\n");
            fprintf(fp, "    evaluation_strategy='epoch',\n");
            fprintf(fp, ")\n\n");
            fprintf(fp, "trainer = Trainer(\n");
            fprintf(fp, "    model=model,\n");
            fprintf(fp, "    args=training_args,\n");
            fprintf(fp, "    train_dataset=tokenized_datasets['train'],\n");
            fprintf(fp, "    eval_dataset=tokenized_datasets['test'],\n");
            fprintf(fp, ")\n\n");
            fprintf(fp, "trainer.train()\n");
            fprintf(fp, "print('Training complete!')\n\n");
            fprintf(fp, "# Save model\n");
            fprintf(fp, "trainer.save_model('./model')\n");
            fprintf(fp, "print('Model saved!')\n");
        }
    }

    // Add model comparison if multiple models
    if (prog.model_count > 1) {
        fprintf(fp, "\n");
        fprintf(fp, "# =====================================\n");
        fprintf(fp, "# MODEL COMPARISON\n");
        fprintf(fp, "# =====================================\n\n");
        fprintf(fp, "print('\\n' + '='*60)\n");
        fprintf(fp, "print('📊 MODEL COMPARISON RESULTS')\n");
        fprintf(fp, "print('='*60 + '\\n')\n\n");
        
        fprintf(fp, "# Display all models\n");
        fprintf(fp, "for i, result in enumerate(model_results, 1):\n");
        fprintf(fp, "    print(f\"{i}. {result['name']}:\")\n");
        fprintf(fp, "    if result['type'] == 'classifier':\n");
        fprintf(fp, "        print(f\"   Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%%)\")\n");
        fprintf(fp, "    else:\n");
        fprintf(fp, "        print(f\"   R² Score: {result['r2']:.4f}\")\n");
        fprintf(fp, "        print(f\"   MSE: {result['mse']:.4f}\")\n");
        fprintf(fp, "    print()\n\n");
        
        fprintf(fp, "# Find best model\n");
        fprintf(fp, "classifiers = [r for r in model_results if r['type'] == 'classifier']\n");
        fprintf(fp, "regressors = [r for r in model_results if r['type'] == 'regressor']\n\n");
        
        fprintf(fp, "best_model = None\n");
        fprintf(fp, "if classifiers:\n");
        fprintf(fp, "    best_model = max(classifiers, key=lambda x: x['accuracy'])\n");
        fprintf(fp, "    print(f\"🏆 BEST CLASSIFIER: {best_model['name']}\")\n");
        fprintf(fp, "    print(f\"   Accuracy: {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.2f}%%)\\n\")\n");
        fprintf(fp, "elif regressors:\n");
        fprintf(fp, "    best_model = max(regressors, key=lambda x: x['r2'])\n");
        fprintf(fp, "    print(f\"🏆 BEST REGRESSOR: {best_model['name']}\")\n");
        fprintf(fp, "    print(f\"   R² Score: {best_model['r2']:.4f}\\n\")\n\n");
        
        fprintf(fp, "# Copy best model to model.pkl for easy prediction\n");
        fprintf(fp, "if best_model:\n");
        fprintf(fp, "    import shutil\n");
        fprintf(fp, "    best_model_file = f\"model_{best_model['name']}.pkl\"\n");
        fprintf(fp, "    shutil.copy(best_model_file, 'model.pkl')\n");
        fprintf(fp, "    print(f\"✅ Best model copied to model.pkl for prediction\\n\")\n");
        fprintf(fp, "    print(f\"📁 All models saved:\")\n");
        fprintf(fp, "    for result in model_results:\n");
        fprintf(fp, "        print(f\"   - model_{result['name']}.pkl\")\n");
        fprintf(fp, "    print(f\"   - model.pkl (best model)\\n\")\n");
    }

    fclose(fp);

    if (!verbose_mode) {
        printf("✅ Python script 'train.py' generated successfully!\n");
    }

    // Phase 7 - Linking and venv setup
    if (verbose_mode) {
        phase7_linking(primary_backend);
    } else {
        printf("🔧 Setting up virtual environment and installing packages...\n");
    }
}
