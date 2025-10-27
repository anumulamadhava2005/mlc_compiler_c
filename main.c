    #include <stdio.h>
#include "parser.tab.h"

extern int yyparse(void);
extern FILE *yyin;

int main(int argc, char **argv) {
    if (argc > 1) {
        yyin = fopen(argv[1], "r");
        if (!yyin) {
            perror(argv[1]);
            return 1;
        }
    } else {
        yyin = stdin;
    }

    yyparse();

    return 0;
}
