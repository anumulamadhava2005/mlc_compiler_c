#ifndef AST_H
#define AST_H

#define MAX_PARAMS 20
#define MAX_MODELS 5

typedef struct {
    char *name;
    char *value;
} Param;

typedef struct {
    char *name;
    Param params[MAX_PARAMS];
    int param_count;
} Model;

typedef struct {
    Model models[MAX_MODELS];
    int model_count;
} Program;

#endif
