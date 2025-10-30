%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"

int yylex(void);
void yyerror(const char *s);

// Global program storage
Program prog = { .dataset_path = "", .model_count = 0 };

// Forward declarations
void generate_python();
const char* detect_backend(const char* model_name);
void generate_dataset_loading(FILE *fp, const char* backend, const char* dataset_path);
void generate_model_code(FILE *fp, Model *m, const char* backend);
void setup_venv_and_install(const char* backend);
char* strip_quotes(char* str);

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
        printf("\n✅ Parsing completed successfully!\n");
        generate_python();
    }
    | model_def_list
    {
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
        printf("📂 Dataset path set to: %s\n", prog.dataset_path);
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
        // Initialize current model
        Model *m = &prog.models[prog.model_count];
        strncpy(m->name, $2, 63);
        m->name[63] = '\0';
        m->param_count = 0;
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
        int idx = m->param_count;
        
        strncpy(m->param_names[idx], $1, 63);
        m->param_names[idx][63] = '\0';
        
        strncpy(m->param_values[idx], $3, 63);
        m->param_values[idx][63] = '\0';
        
        m->param_count++;
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
        // Support for boolean values (true/false) and identifiers
        $$ = strdup($1);
        free($1);
    }
    ;

%%

// Error handling
void yyerror(const char *s) {
    fprintf(stderr, "❌ Parse error: %s\n", s);
}

// Strip quotes from strings
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

// Detect backend based on model name
const char* detect_backend(const char* model_name) {
    // Scikit-learn models
    if (strstr(model_name, "LinearRegression") || strstr(model_name, "LogisticRegression") ||
        strstr(model_name, "DecisionTree") || strstr(model_name, "RandomForest")) {
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
    
    // Default to TensorFlow
    return "tensorflow";
}

// Convert value to Python format (capitalize booleans)
const char* to_python_value(const char* value) {
    static char buffer[256];
    if (strcmp(value, "true") == 0) {
        return "True";
    } else if (strcmp(value, "false") == 0) {
        return "False";
    } else {
        // Check if it's a string that needs quotes
        // If it starts with a letter and contains only alphanumeric/underscore, add quotes
        int needs_quotes = 0;
        if (value[0] >= 'a' && value[0] <= 'z') {
            // Check if it's not a number and not a boolean
            if (strcmp(value, "True") != 0 && strcmp(value, "False") != 0 &&
                strcmp(value, "None") != 0) {
                // Check if it contains non-numeric characters (excluding . for floats)
                int has_alpha = 0;
                for (int i = 0; value[i]; i++) {
                    if ((value[i] >= 'a' && value[i] <= 'z') || 
                        (value[i] >= 'A' && value[i] <= 'Z') || value[i] == '_') {
                        has_alpha = 1;
                        break;
                    }
                }
                if (has_alpha) {
                    snprintf(buffer, sizeof(buffer), "\"%s\"", value);
                    return buffer;
                }
            }
        }
        return value;
    }
}

// Generate dataset loading code
void generate_dataset_loading(FILE *fp, const char* backend, const char* dataset_path) {
    fprintf(fp, "# Dataset Loading\n");
    
    if (strcmp(backend, "sklearn") == 0) {
        fprintf(fp, "import pandas as pd\n");
        fprintf(fp, "from sklearn.model_selection import train_test_split\n\n");
        fprintf(fp, "# Load dataset (assuming CSV format)\n");
        fprintf(fp, "dataset = pd.read_csv('%s')\n", dataset_path);
        fprintf(fp, "# Separate features and target (assuming last column is target)\n");
        fprintf(fp, "X = dataset.iloc[:, :-1].values\n");
        fprintf(fp, "y = dataset.iloc[:, -1].values\n\n");
        fprintf(fp, "# Split into train and test sets\n");
        fprintf(fp, "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n");
    } else if (strcmp(backend, "tensorflow") == 0) {
        fprintf(fp, "import tensorflow as tf\n\n");
        fprintf(fp, "# Load dataset from directory\n");
        fprintf(fp, "train_ds = tf.keras.utils.image_dataset_from_directory(\n");
        fprintf(fp, "    '%s',\n", dataset_path);
        fprintf(fp, "    image_size=(224, 224),\n");
        fprintf(fp, "    batch_size=batch_size,\n");
        fprintf(fp, "    label_mode='categorical'\n");
        fprintf(fp, ")\n\n");
        fprintf(fp, "# Normalize pixel values\n");
        fprintf(fp, "normalization_layer = tf.keras.layers.Rescaling(1./255)\n");
        fprintf(fp, "train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))\n\n");
    } else if (strcmp(backend, "pytorch") == 0) {
        fprintf(fp, "import torch\n");
        fprintf(fp, "import torch.nn as nn\n");
        fprintf(fp, "import torch.optim as optim\n");
        fprintf(fp, "from torchvision import datasets, transforms\n");
        fprintf(fp, "from torch.utils.data import DataLoader\n\n");
        fprintf(fp, "# Dataset transforms and loading\n");
        fprintf(fp, "transform = transforms.Compose([\n");
        fprintf(fp, "    transforms.Resize((224, 224)),\n");
        fprintf(fp, "    transforms.ToTensor(),\n");
        fprintf(fp, "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n");
        fprintf(fp, "])\n\n");
        fprintf(fp, "train_ds = datasets.ImageFolder('%s', transform=transform)\n", dataset_path);
        fprintf(fp, "train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)\n\n");
    } else if (strcmp(backend, "transformers") == 0) {
        fprintf(fp, "from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments\n");
        fprintf(fp, "from datasets import load_dataset\n\n");
        fprintf(fp, "# Load dataset\n");
        fprintf(fp, "dataset = load_dataset('%s')\n", dataset_path);
        fprintf(fp, "tokenizer = AutoTokenizer.from_pretrained(model_name)\n\n");
        fprintf(fp, "def tokenize_function(examples):\n");
        fprintf(fp, "    return tokenizer(examples['text'], padding='max_length', truncation=True)\n\n");
        fprintf(fp, "tokenized_datasets = dataset.map(tokenize_function, batched=True)\n\n");
    }
}

// Generate model-specific code
void generate_model_code(FILE *fp, Model *m, const char* backend) {
    fprintf(fp, "# Model: %s (%s backend)\n", m->name, backend);
    
    if (strcmp(backend, "tensorflow") == 0) {
        fprintf(fp, "import tensorflow as tf\n");
        fprintf(fp, "from tensorflow.keras.applications import %s\n", m->name);
        fprintf(fp, "from tensorflow.keras.layers import Dense, GlobalAveragePooling2D\n");
        fprintf(fp, "from tensorflow.keras.models import Model\n\n");
        
        fprintf(fp, "# Build model\n");
        fprintf(fp, "base_model = %s(weights='imagenet', include_top=False, input_shape=(224, 224, 3))\n", m->name);
        fprintf(fp, "x = base_model.output\n");
        fprintf(fp, "x = GlobalAveragePooling2D()(x)\n");
        fprintf(fp, "x = Dense(1024, activation='relu')(x)\n");
        fprintf(fp, "predictions = Dense(len(train_ds.class_names), activation='softmax')(x)\n");
        fprintf(fp, "model = Model(inputs=base_model.input, outputs=predictions)\n\n");
        
        fprintf(fp, "# Compile model\n");
        fprintf(fp, "model.compile(\n");
        fprintf(fp, "    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),\n");
        fprintf(fp, "    loss='categorical_crossentropy',\n");
        fprintf(fp, "    metrics=['accuracy']\n");
        fprintf(fp, ")\n\n");
        
        fprintf(fp, "# Training\n");
        fprintf(fp, "print('🚀 Starting training...')\n");
        fprintf(fp, "history = model.fit(\n");
        fprintf(fp, "    train_ds,\n");
        fprintf(fp, "    epochs=epochs,\n");
        fprintf(fp, "    verbose=1\n");
        fprintf(fp, ")\n\n");
        fprintf(fp, "print('✅ Training completed!')\n");
        fprintf(fp, "model.save('trained_model.h5')\n");
        fprintf(fp, "print('💾 Model saved as trained_model.h5')\n");
        
    } else if (strcmp(backend, "pytorch") == 0) {
        fprintf(fp, "import torch\n");
        fprintf(fp, "import torch.nn as nn\n");
        fprintf(fp, "import torch.optim as optim\n\n");
        
        fprintf(fp, "# Define model (example: simple CNN)\n");
        fprintf(fp, "class %s(nn.Module):\n", m->name);
        fprintf(fp, "    def __init__(self):\n");
        fprintf(fp, "        super(%s, self).__init__()\n", m->name);
        fprintf(fp, "        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)\n");
        fprintf(fp, "        self.pool = nn.MaxPool2d(2, 2)\n");
        fprintf(fp, "        self.fc1 = nn.Linear(64 * 112 * 112, 10)\n");
        fprintf(fp, "    def forward(self, x):\n");
        fprintf(fp, "        x = self.pool(torch.relu(self.conv1(x)))\n");
        fprintf(fp, "        x = x.view(-1, 64 * 112 * 112)\n");
        fprintf(fp, "        x = self.fc1(x)\n");
        fprintf(fp, "        return x\n\n");
        
        fprintf(fp, "model = %s()\n", m->name);
        fprintf(fp, "criterion = nn.CrossEntropyLoss()\n");
        fprintf(fp, "optimizer = optim.Adam(model.parameters(), lr=learning_rate)\n\n");
        
        fprintf(fp, "# Training loop\n");
        fprintf(fp, "print('🚀 Starting training...')\n");
        fprintf(fp, "for epoch in range(epochs):\n");
        fprintf(fp, "    running_loss = 0.0\n");
        fprintf(fp, "    for i, (inputs, labels) in enumerate(train_loader):\n");
        fprintf(fp, "        optimizer.zero_grad()\n");
        fprintf(fp, "        outputs = model(inputs)\n");
        fprintf(fp, "        loss = criterion(outputs, labels)\n");
        fprintf(fp, "        loss.backward()\n");
        fprintf(fp, "        optimizer.step()\n");
        fprintf(fp, "        running_loss += loss.item()\n");
        fprintf(fp, "    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')\n\n");
        fprintf(fp, "print('✅ Training completed!')\n");
        fprintf(fp, "torch.save(model.state_dict(), 'model.pth')\n");
        fprintf(fp, "print('💾 Model saved as model.pth')\n");
        
    } else if (strcmp(backend, "transformers") == 0) {
        fprintf(fp, "from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer\n\n");
        
        fprintf(fp, "# Model\n");
        fprintf(fp, "model_name = '%s'\n", m->name);
        fprintf(fp, "model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)\n\n");
        
        fprintf(fp, "# Training arguments\n");
        fprintf(fp, "training_args = TrainingArguments(\n");
        fprintf(fp, "    output_dir='./results',\n");
        fprintf(fp, "    num_train_epochs=epochs,\n");
        fprintf(fp, "    per_device_train_batch_size=batch_size,\n");
        fprintf(fp, "    learning_rate=learning_rate,\n");
        fprintf(fp, "    logging_dir='./logs',\n");
        fprintf(fp, ")\n\n");
        
        fprintf(fp, "# Trainer\n");
        fprintf(fp, "trainer = Trainer(\n");
        fprintf(fp, "    model=model,\n");
        fprintf(fp, "    args=training_args,\n");
        fprintf(fp, "    train_dataset=tokenized_datasets['train'],\n");
        fprintf(fp, ")\n\n");
        
        fprintf(fp, "print('🚀 Starting training...')\n");
        fprintf(fp, "trainer.train()\n");
        fprintf(fp, "print('✅ Training completed!')\n");
        fprintf(fp, "model.save_pretrained('./trained_model')\n");
        fprintf(fp, "print('💾 Model saved to ./trained_model')\n");
    
    } else if (strcmp(backend, "sklearn") == 0) {
        fprintf(fp, "# Scikit-learn model\n");
        
        // Determine which sklearn model and import appropriately
        if (strcmp(m->name, "LinearRegression") == 0) {
            fprintf(fp, "from sklearn.linear_model import LinearRegression\n");
            fprintf(fp, "from sklearn.metrics import mean_squared_error, r2_score\n\n");
            fprintf(fp, "# Initialize model\n");
            fprintf(fp, "model = LinearRegression(");
            // Add parameters
            int first = 1;
            for (int i = 0; i < m->param_count; i++) {
                if (!first) fprintf(fp, ", ");
                fprintf(fp, "%s=%s", m->param_names[i], to_python_value(m->param_values[i]));
                first = 0;
            }
            fprintf(fp, ")\n\n");
            fprintf(fp, "# Training\n");
            fprintf(fp, "print('🚀 Starting training...')\n");
            fprintf(fp, "model.fit(X_train, y_train)\n");
            fprintf(fp, "print('✅ Training completed!')\n\n");
            fprintf(fp, "# Evaluation\n");
            fprintf(fp, "y_pred = model.predict(X_test)\n");
            fprintf(fp, "mse = mean_squared_error(y_test, y_pred)\n");
            fprintf(fp, "r2 = r2_score(y_test, y_pred)\n");
            fprintf(fp, "print(f'📊 Mean Squared Error: {mse:.4f}')\n");
            fprintf(fp, "print(f'📊 R² Score: {r2:.4f}')\n");
            
        } else if (strcmp(m->name, "LogisticRegression") == 0) {
            fprintf(fp, "from sklearn.linear_model import LogisticRegression\n");
            fprintf(fp, "from sklearn.metrics import accuracy_score, classification_report\n\n");
            fprintf(fp, "# Initialize model\n");
            fprintf(fp, "model = LogisticRegression(");
            int first = 1;
            for (int i = 0; i < m->param_count; i++) {
                if (!first) fprintf(fp, ", ");
                fprintf(fp, "%s=%s", m->param_names[i], to_python_value(m->param_values[i]));
                first = 0;
            }
            fprintf(fp, ")\n\n");
            fprintf(fp, "# Training\n");
            fprintf(fp, "print('🚀 Starting training...')\n");
            fprintf(fp, "model.fit(X_train, y_train)\n");
            fprintf(fp, "print('✅ Training completed!')\n\n");
            fprintf(fp, "# Evaluation\n");
            fprintf(fp, "y_pred = model.predict(X_test)\n");
            fprintf(fp, "accuracy = accuracy_score(y_test, y_pred)\n");
            fprintf(fp, "print(f'📊 Accuracy: {accuracy:.4f}')\n");
            fprintf(fp, "print('\\n📋 Classification Report:')\n");
            fprintf(fp, "print(classification_report(y_test, y_pred))\n");
            
        } else if (strcmp(m->name, "DecisionTreeClassifier") == 0) {
            fprintf(fp, "from sklearn.tree import DecisionTreeClassifier\n");
            fprintf(fp, "from sklearn.metrics import accuracy_score, classification_report\n\n");
            fprintf(fp, "# Initialize model\n");
            fprintf(fp, "model = DecisionTreeClassifier(");
            int first = 1;
            for (int i = 0; i < m->param_count; i++) {
                if (!first) fprintf(fp, ", ");
                fprintf(fp, "%s=%s", m->param_names[i], to_python_value(m->param_values[i]));
                first = 0;
            }
            fprintf(fp, ")\n\n");
            fprintf(fp, "# Training\n");
            fprintf(fp, "print('🚀 Starting training...')\n");
            fprintf(fp, "model.fit(X_train, y_train)\n");
            fprintf(fp, "print('✅ Training completed!')\n\n");
            fprintf(fp, "# Evaluation\n");
            fprintf(fp, "y_pred = model.predict(X_test)\n");
            fprintf(fp, "accuracy = accuracy_score(y_test, y_pred)\n");
            fprintf(fp, "print(f'📊 Accuracy: {accuracy:.4f}')\n");
            fprintf(fp, "print('\\n📋 Classification Report:')\n");
            fprintf(fp, "print(classification_report(y_test, y_pred))\n");
            
        } else if (strcmp(m->name, "RandomForestClassifier") == 0) {
            fprintf(fp, "from sklearn.ensemble import RandomForestClassifier\n");
            fprintf(fp, "from sklearn.metrics import accuracy_score, classification_report\n\n");
            fprintf(fp, "# Initialize model\n");
            fprintf(fp, "model = RandomForestClassifier(");
            int first = 1;
            for (int i = 0; i < m->param_count; i++) {
                if (!first) fprintf(fp, ", ");
                fprintf(fp, "%s=%s", m->param_names[i], to_python_value(m->param_values[i]));
                first = 0;
            }
            fprintf(fp, ")\n\n");
            fprintf(fp, "# Training\n");
            fprintf(fp, "print('🚀 Starting training...')\n");
            fprintf(fp, "model.fit(X_train, y_train)\n");
            fprintf(fp, "print('✅ Training completed!')\n\n");
            fprintf(fp, "# Evaluation\n");
            fprintf(fp, "y_pred = model.predict(X_test)\n");
            fprintf(fp, "accuracy = accuracy_score(y_test, y_pred)\n");
            fprintf(fp, "print(f'📊 Accuracy: {accuracy:.4f}')\n");
            fprintf(fp, "print('\\n📋 Classification Report:')\n");
            fprintf(fp, "print(classification_report(y_test, y_pred))\n");
            
        } else if (strcmp(m->name, "KNeighborsClassifier") == 0) {
            fprintf(fp, "from sklearn.neighbors import KNeighborsClassifier\n");
            fprintf(fp, "from sklearn.metrics import accuracy_score, classification_report\n\n");
            fprintf(fp, "# Initialize model\n");
            fprintf(fp, "model = KNeighborsClassifier(");
            int first = 1;
            for (int i = 0; i < m->param_count; i++) {
                if (!first) fprintf(fp, ", ");
                fprintf(fp, "%s=%s", m->param_names[i], to_python_value(m->param_values[i]));
                first = 0;
            }
            fprintf(fp, ")\n\n");
            fprintf(fp, "# Training\n");
            fprintf(fp, "print('🚀 Starting training...')\n");
            fprintf(fp, "model.fit(X_train, y_train)\n");
            fprintf(fp, "print('✅ Training completed!')\n\n");
            fprintf(fp, "# Evaluation\n");
            fprintf(fp, "y_pred = model.predict(X_test)\n");
            fprintf(fp, "accuracy = accuracy_score(y_test, y_pred)\n");
            fprintf(fp, "print(f'📊 Accuracy: {accuracy:.4f}')\n");
            fprintf(fp, "print('\\n📋 Classification Report:')\n");
            fprintf(fp, "print(classification_report(y_test, y_pred))\n");
            
        } else if (strcmp(m->name, "SVC") == 0) {
            fprintf(fp, "from sklearn.svm import SVC\n");
            fprintf(fp, "from sklearn.metrics import accuracy_score, classification_report\n\n");
            fprintf(fp, "# Initialize model\n");
            fprintf(fp, "model = SVC(");
            int first = 1;
            for (int i = 0; i < m->param_count; i++) {
                if (!first) fprintf(fp, ", ");
                fprintf(fp, "%s=%s", m->param_names[i], to_python_value(m->param_values[i]));
                first = 0;
            }
            fprintf(fp, ")\n\n");
            fprintf(fp, "# Training\n");
            fprintf(fp, "print('🚀 Starting training...')\n");
            fprintf(fp, "model.fit(X_train, y_train)\n");
            fprintf(fp, "print('✅ Training completed!')\n\n");
            fprintf(fp, "# Evaluation\n");
            fprintf(fp, "y_pred = model.predict(X_test)\n");
            fprintf(fp, "accuracy = accuracy_score(y_test, y_pred)\n");
            fprintf(fp, "print(f'📊 Accuracy: {accuracy:.4f}')\n");
            fprintf(fp, "print('\\n📋 Classification Report:')\n");
            fprintf(fp, "print(classification_report(y_test, y_pred))\n");
            
        } else if (strcmp(m->name, "GaussianNB") == 0) {
            fprintf(fp, "from sklearn.naive_bayes import GaussianNB\n");
            fprintf(fp, "from sklearn.metrics import accuracy_score, classification_report\n\n");
            fprintf(fp, "# Initialize model\n");
            fprintf(fp, "model = GaussianNB()\n\n");
            fprintf(fp, "# Training\n");
            fprintf(fp, "print('🚀 Starting training...')\n");
            fprintf(fp, "model.fit(X_train, y_train)\n");
            fprintf(fp, "print('✅ Training completed!')\n\n");
            fprintf(fp, "# Evaluation\n");
            fprintf(fp, "y_pred = model.predict(X_test)\n");
            fprintf(fp, "accuracy = accuracy_score(y_test, y_pred)\n");
            fprintf(fp, "print(f'📊 Accuracy: {accuracy:.4f}')\n");
            fprintf(fp, "print('\\n📋 Classification Report:')\n");
            fprintf(fp, "print(classification_report(y_test, y_pred))\n");
            
        } else if (strcmp(m->name, "KMeans") == 0) {
            fprintf(fp, "from sklearn.cluster import KMeans\n");
            fprintf(fp, "from sklearn.metrics import silhouette_score\n\n");
            fprintf(fp, "# Initialize model\n");
            fprintf(fp, "model = KMeans(");
            int first = 1;
            for (int i = 0; i < m->param_count; i++) {
                if (!first) fprintf(fp, ", ");
                fprintf(fp, "%s=%s", m->param_names[i], to_python_value(m->param_values[i]));
                first = 0;
            }
            fprintf(fp, ")\n\n");
            fprintf(fp, "# Training (clustering)\n");
            fprintf(fp, "print('🚀 Starting clustering...')\n");
            fprintf(fp, "model.fit(X_train)\n");
            fprintf(fp, "print('✅ Clustering completed!')\n\n");
            fprintf(fp, "# Evaluation\n");
            fprintf(fp, "labels = model.labels_\n");
            fprintf(fp, "silhouette = silhouette_score(X_train, labels)\n");
            fprintf(fp, "print(f'📊 Silhouette Score: {silhouette:.4f}')\n");
            fprintf(fp, "print(f'📊 Inertia: {model.inertia_:.4f}')\n");
            
        } else if (strcmp(m->name, "LinearSVC") == 0) {
            fprintf(fp, "from sklearn.svm import LinearSVC\n");
            fprintf(fp, "from sklearn.metrics import accuracy_score, classification_report\n\n");
            fprintf(fp, "# Initialize model\n");
            fprintf(fp, "model = LinearSVC(");
            int first = 1;
            for (int i = 0; i < m->param_count; i++) {
                if (!first) fprintf(fp, ", ");
                fprintf(fp, "%s=%s", m->param_names[i], to_python_value(m->param_values[i]));
                first = 0;
            }
            fprintf(fp, ")\n\n");
            fprintf(fp, "# Training\n");
            fprintf(fp, "print('🚀 Starting training...')\n");
            fprintf(fp, "model.fit(X_train, y_train)\n");
            fprintf(fp, "print('✅ Training completed!')\n\n");
            fprintf(fp, "# Evaluation\n");
            fprintf(fp, "y_pred = model.predict(X_test)\n");
            fprintf(fp, "accuracy = accuracy_score(y_test, y_pred)\n");
            fprintf(fp, "print(f'📊 Accuracy: {accuracy:.4f}')\n");
            fprintf(fp, "print('\\n📋 Classification Report:')\n");
            fprintf(fp, "print(classification_report(y_test, y_pred))\n");
            
        } else if (strcmp(m->name, "SGDClassifier") == 0) {
            fprintf(fp, "from sklearn.linear_model import SGDClassifier\n");
            fprintf(fp, "from sklearn.metrics import accuracy_score, classification_report\n\n");
            fprintf(fp, "# Initialize model\n");
            fprintf(fp, "model = SGDClassifier(");
            int first = 1;
            for (int i = 0; i < m->param_count; i++) {
                if (!first) fprintf(fp, ", ");
                fprintf(fp, "%s=%s", m->param_names[i], to_python_value(m->param_values[i]));
                first = 0;
            }
            fprintf(fp, ")\n\n");
            fprintf(fp, "# Training\n");
            fprintf(fp, "print('🚀 Starting training...')\n");
            fprintf(fp, "model.fit(X_train, y_train)\n");
            fprintf(fp, "print('✅ Training completed!')\n\n");
            fprintf(fp, "# Evaluation\n");
            fprintf(fp, "y_pred = model.predict(X_test)\n");
            fprintf(fp, "accuracy = accuracy_score(y_test, y_pred)\n");
            fprintf(fp, "print(f'📊 Accuracy: {accuracy:.4f}')\n");
            fprintf(fp, "print('\\n📋 Classification Report:')\n");
            fprintf(fp, "print(classification_report(y_test, y_pred))\n");
        }
        
        fprintf(fp, "\n# Save model\n");
        fprintf(fp, "import joblib\n");
        fprintf(fp, "joblib.dump(model, 'model.pkl')\n");
        fprintf(fp, "print('💾 Model saved as model.pkl')\n");
    }
}

// Main code generation function
void generate_python() {
    FILE *fp = fopen("train.py", "w");
    if (!fp) {
        perror("train.py");
        exit(1);
    }

    fprintf(fp, "#!/usr/bin/env python3\n");
    fprintf(fp, "# Generated by MLC Compiler\n");
    fprintf(fp, "# Auto-generated machine learning training script\n\n");

    if (prog.model_count == 0) {
        fprintf(fp, "print('No models defined')\n");
        fclose(fp);
        printf("⚠️  No models found, empty train.py generated\n");
        return;
    }

    // Process each model
    for (int i = 0; i < prog.model_count; i++) {
        Model *m = &prog.models[i];
        const char* backend = detect_backend(m->name);

        fprintf(fp, "# =====================================\n");
        fprintf(fp, "# Model %d: %s\n", i+1, m->name);
        fprintf(fp, "# Backend: %s\n", backend);
        fprintf(fp, "# =====================================\n\n");

        // Set hyperparameters with defaults (skip for sklearn since params go in constructor)
        if (strcmp(backend, "sklearn") != 0) {
            int has_epochs = 0, has_lr = 0, has_batch_size = 0;
            
            for (int j = 0; j < m->param_count; j++) {
                fprintf(fp, "%s = %s\n", m->param_names[j], m->param_values[j]);
                if (strcmp(m->param_names[j], "epochs") == 0) has_epochs = 1;
                if (strcmp(m->param_names[j], "learning_rate") == 0) has_lr = 1;
                if (strcmp(m->param_names[j], "batch_size") == 0) has_batch_size = 1;
            }
            
            if (!has_epochs) fprintf(fp, "epochs = 10\n");
            if (!has_lr) fprintf(fp, "learning_rate = 0.001\n");
            if (!has_batch_size) fprintf(fp, "batch_size = 32\n");
            fprintf(fp, "\n");
        }

        // Generate dataset loading if path is provided
        if (strlen(prog.dataset_path) > 0) {
            generate_dataset_loading(fp, backend, prog.dataset_path);
        } else {
            fprintf(fp, "# No dataset path specified\n");
            fprintf(fp, "print('⚠️  Warning: No dataset path provided')\n\n");
        }

        // Generate model code
        generate_model_code(fp, m, backend);
        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("✅ Python script 'train.py' generated successfully!\n");

    // Setup virtual environment
    printf("🔧 Setting up virtual environment and installing packages...\n");
    
    // Determine which backend to install
    const char* backend = detect_backend(prog.models[0].name);
    setup_venv_and_install(backend);
}

// Setup virtual environment and install packages
void setup_venv_and_install(const char* backend) {
    int ret;
    
    // Create virtual environment
    printf("   Creating virtual environment...\n");
    ret = system("python3 -m venv venv 2>/dev/null");
    if (ret != 0) {
        fprintf(stderr, "   ⚠️  Warning: Could not create venv (might already exist)\n");
    }

    // Upgrade pip
    printf("   Upgrading pip...\n");
    system("venv/bin/python -m pip install --upgrade pip -q 2>/dev/null");

    // Install packages based on backend
    printf("   Installing %s and dependencies...\n", backend);
    
    if (strcmp(backend, "sklearn") == 0) {
        ret = system("venv/bin/python -m pip install scikit-learn pandas joblib -q 2>/dev/null");
    } else if (strcmp(backend, "tensorflow") == 0) {
        ret = system("venv/bin/python -m pip install tensorflow keras pillow -q 2>/dev/null");
    } else if (strcmp(backend, "pytorch") == 0) {
        ret = system("venv/bin/python -m pip install torch torchvision -q 2>/dev/null");
    } else if (strcmp(backend, "transformers") == 0) {
        ret = system("venv/bin/python -m pip install transformers datasets torch -q 2>/dev/null");
    }

    if (ret == 0) {
        printf("✅ Virtual environment ready with %s installed!\n", backend);
        printf("\n📋 To run training:\n");
        printf("   venv/bin/python train.py\n\n");
    } else {
        fprintf(stderr, "⚠️  Warning: Package installation may have encountered issues\n");
        printf("\n📋 To install manually:\n");
        printf("   source venv/bin/activate\n");
        printf("   pip install %s\n", backend);
        printf("   python train.py\n\n");
    }
}
