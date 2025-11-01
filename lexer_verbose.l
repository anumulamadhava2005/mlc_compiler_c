%{
#include "parser.tab.h"
#include <stdio.h>
#include <string.h>

extern int verbose_mode;
extern int line_number;

void print_token(const char* token_type, const char* lexeme) {
    if (verbose_mode) {
        printf("[%-12s, \"%-20s\", line %d]\n", token_type, lexeme, line_number);
    }
}
%}

%%
"dataset"               { print_token("DATASET", yytext); return DATASET; }
"model"                 { print_token("MODEL", yytext); return MODEL; }
"{"                     { print_token("LBRACE", yytext); return LBRACE; }
"}"                     { print_token("RBRACE", yytext); return RBRACE; }
"["                     { print_token("LBRACKET", yytext); return LBRACKET; }
"]"                     { print_token("RBRACKET", yytext); return RBRACKET; }
"="                     { print_token("ASSIGN", yytext); return ASSIGN; }
","                     { print_token("COMMA", yytext); return COMMA; }
[0-9]+                  { print_token("INT", yytext); yylval.intVal = atoi(yytext); return INT; }
[0-9]+"."[0-9]+         { print_token("FLOAT", yytext); yylval.floatVal = atof(yytext); return FLOAT; }
\"[^\"]*\"              { print_token("STRING", yytext); yylval.strVal = strdup(yytext); return STRING; }
[a-zA-Z_][a-zA-Z0-9_]*  { print_token("IDENTIFIER", yytext); yylval.strVal = strdup(yytext); return ID; }
\n                      { line_number++; }
[ \t\r]+                { /* ignore whitespace */ }
.                       { return *yytext; }

%%
int yywrap(void) { return 1; }
