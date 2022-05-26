/* recognize tokens for the calculator and print them out */
%option noinput nounput

%{
enum yytokentype {
  NUMBER = 258,
  ADD    = 259,
  SUB    = 260,
  MUL    = 261,
  DIV    = 262,
  ABS    = 263,
  EOL    = 264,  // end-of-line
};

int yylval;
%}

%%

"+"     { return ADD; }
"-"     { return SUB; }
"*"     { return MUL; }
"/"     { return DIV; }
"|"     { return ABS; }
[0-9]+  { yylval = atoi(yytext); return NUMBER; }
\n      { return EOL; }
[ \t]   { /* ignore whitespaces */ }
.       { printf("Mystery character %c\n", *yytext); }

%%

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
  int tok;

  while ((tok = yylex())) {
    printf("%d", tok);
    if (tok == NUMBER)
      printf(" = %d\n", yylval);
    else
      printf("\n");
  }

  return 0;
}