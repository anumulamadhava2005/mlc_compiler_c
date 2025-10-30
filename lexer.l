%{
#include "parser.tab.h"
#include <stdio.h>
%}

%%
"dataset"               { return DATASET; }
"model"                 { return MODEL; }
"{"                     { return LBRACE; }
"}"                     { return RBRACE; }
"["                     { return LBRACKET; }
"]"                     { return RBRACKET; }
"="                     { return ASSIGN; }
","                     { return COMMA; }
[0-9]+                  { yylval.intVal = atoi(yytext); return INT; }
[0-9]+"."[0-9]+         { yylval.floatVal = atof(yytext); return FLOAT; }
\"[^\"]*\"              { yylval.strVal = strdup(yytext); return STRING; }
[a-zA-Z_][a-zA-Z0-9_]*  { yylval.strVal = strdup(yytext); return ID; }
[ \t\r\n]+              { /* ignore whitespace */ }
.                       { return *yytext; }

%%
int yywrap(void) { return 1; }
