/* simplest version of calculator */

%{
#include <stdio.h>

/*
[solution] forward declaration of function ‘yylex’ ‘yyerror’
[issue]
  warning: implicit declaration of function ‘yylex’ [-Wimplicit-function-declaration]
  warning: implicit declaration of function ‘yyerror’; did you mean ‘yyerrok’? [-Wimplicit-function-declaration]
*/
int yylex(void);
void yyerror(const char *s);
%}

/* declare tokens */

%token NUMBER
%token ADD SUB MUL DIV ABS
%token OP CP
%token EOL

%%

calclist: /* nothing, matches at beginning of input */
        | calclist exp EOL { printf("= %d\n> ", $2); }  /* EOL is end of an expression */
        | calclist EOL { printf("> "); }                /* blank line or a comment */
        ;

exp     : factor    /* default $$ = $1 */
        | exp ADD factor { $$ = $1 + $3; }
        | exp SUB factor { $$ = $1 - $3; }
        ;

factor  : term      /* default $$ = $1 */
        | factor MUL term { $$ = $1 * $3; }
        | factor DIV term { $$ = $1 / $3; }
        ;

term    : NUMBER    /* default $$ = $1 */
        | ABS term  { $$ = $2 > 0 ? $2 : -$2; }
        | OP exp CP { $$ = $2; }
        ;

%%

int main(void)
{
  printf("> ");
  yyparse();
}

void yyerror(const char *s)
{
  fprintf(stderr, "error: %s\n", s);
}
