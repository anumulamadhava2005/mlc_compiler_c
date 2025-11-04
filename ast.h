#ifndef AST_H
#define AST_H

#define MAX_PARAMS 50
#define MAX_MODELS 10

typedef struct {
    char name[64];
    char backend[64];  // User-specified backend (e.g., "sklearn", "tensorflow")
    char param_names[MAX_PARAMS][64];
    char param_values[MAX_PARAMS][64];
    int param_count;
} Model;

typedef struct {
    char dataset_path[256];
    Model models[MAX_MODELS];
    int model_count;
} Program;

#endif
