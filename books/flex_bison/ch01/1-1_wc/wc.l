%{
int chars = 0;
int words = 0;
int lines = 0;
%}

%%
[a-zA-Z]+    { words++; chars += strlen(yytext); }
\n           { chars++; lines++; }
.            { chars++; }
%%

int main(void) {
  yylex();
  printf("%8d%8d%8d\n", lines, words, chars);
  return 0;
}