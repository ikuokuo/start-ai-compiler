/* just like Unix wc */

/*
[solution] https://stackoverflow.com/q/39075510/1480131
[issue]
  warning: ‘input’ defined but not used [-Wunused-function]
  warning: ‘yyunput’ defined but not used [-Wunused-function]
*/
%option noinput nounput

%{
int chars = 0;
int words = 0;
int lines = 0;
%}

%%

[a-zA-Z]+   { words++; chars += strlen(yytext); }
\n          { chars++; lines++; }
.           { chars++; }

  /* [^ \t\n\r\f\v]+     { words++; chars += strlen(yytext); } */

%%

int main(void)
{
  yylex();
  printf("%8d%8d%8d\n", lines, words, chars);
  return 0;
}