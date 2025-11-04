#include <stdio.h>
#include <string.h>
#include "parser.tab.h"
#include "compiler_phases.h"

extern int yyparse(void);
extern FILE *yyin;

int verbose_mode = 1;  // Default to verbose to show all compilation phases
int line_number = 1;

void print_header() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║          MLC COMPILER - MULTI-PHASE COMPILATION              ║\n");
    printf("║        Machine Learning Configuration Compiler v2.0          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
}

void print_usage() {
    printf("Usage: ./mlc_compiler [options] <input.mlc>\n");
    printf("Options:\n");
    printf("  -v, --verbose    Show all compilation phases (default)\n");
    printf("  -q, --quiet      Suppress compilation phase output\n");
    printf("  -h, --help       Show this help message\n");
}

int main(int argc, char **argv) {
    char *input_file = NULL;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose_mode = 1;
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            verbose_mode = 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        } else {
            input_file = argv[i];
        }
    }
    
    if (!input_file) {
        printf("Error: No input file specified\n");
        print_usage();
        return 1;
    }
    
    yyin = fopen(input_file, "r");
    if (!yyin) {
        perror(input_file);
        return 1;
    }
    
    if (verbose_mode) {
        print_header();
        printf("\nInput file: %s\n", input_file);
        phase1_lexical_analysis();
    }
    
    yyparse();
    
    fclose(yyin);
    return 0;
}
