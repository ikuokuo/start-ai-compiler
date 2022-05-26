/* recognize tokens for the calculator and print them out */
%option noinput nounput

%{
#include "calc.tab.h"
%}

%%

"+"     { return ADD; }
"-"     { return SUB; }
"*"     { return MUL; }
"/"     { return DIV; }
"|"     { return ABS; }
"("     { return OP; }
")"     { return CP; }
[0-9]+  { yylval = atoi(yytext); return NUMBER; }
\n      { return EOL; }
"//".*
[ \t]   { /* ignore whitespaces and comments starting with // */ }
.       { printf("Mystery character %c\n", *yytext); }

%%