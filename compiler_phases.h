#ifndef COMPILER_PHASES_H
#define COMPILER_PHASES_H

#include "ast.h"

// Symbol Table Entry
typedef struct {
    char name[64];
    char type[32];
    char value[64];
    char scope[64];
} SymbolEntry;

// Symbol Table
typedef struct {
    SymbolEntry entries[100];
    int count;
} SymbolTable;

// Intermediate Representation (3-Address Code)
typedef struct {
    char op[32];
    char arg1[64];
    char arg2[64];
    char result[64];
} IRInstruction;

typedef struct {
    IRInstruction instructions[200];
    int count;
} IRCode;

// Function declarations
void phase1_lexical_analysis();
void phase2_syntax_analysis(Program *prog);
void phase3_semantic_analysis(Program *prog, SymbolTable *symtab);
void phase4_ir_generation(Program *prog, IRCode *ir);
void phase5_optimization(IRCode *ir);
void phase6_code_generation(Program *prog, const char* backend);
void phase7_linking(const char* backend);

void print_symbol_table(SymbolTable *symtab);
void print_ir_code(IRCode *ir);
void print_parse_tree(Program *prog, int indent);

#endif
