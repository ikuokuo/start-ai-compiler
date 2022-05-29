/* skeleton for include files */
%option nodefault nounput noyywrap
%x IFILE

%{
struct bufstack {
  struct bufstack *prev;  /* previous entry */
  YY_BUFFER_STATE bs;     /* saved buffer */
  int lineno;             /* saved line number */
  char *filename;         /* name of this file */
  FILE *f;                /* current file */
} *curbs = 0;

char *curfilename;        /* name of current input file */

int newfile(char *fn);
int popfile(void);
%}

%%

  /* match #include statement up through the quote or < */
^"#"[ \t]*include[ \t]*[\"<]  { BEGIN IFILE; }

  /* handle filename up to the closing quote, >, or end of line */
<IFILE>[^ \t\n\">]+     {
                            { int c;
                              while ((c = input()) && c != '\n');
                            }
                            yylineno++;
                            if (!newfile(yytext))
                              yyterminate(); /* no such file */
                            BEGIN INITIAL;
                        }

  /* handle bad input in IFILE state */
<IFILE>.|\n             { fprintf(stderr, "%4d bad include line\n", yylineno);
                          yyterminate();
                        }

  /* pop the file stack at end of file, terminate if it's the outermost file */
<<EOF>>                 { if (!popfile()) yyterminate(); }

  /* print the line number at the beginning of each line
     and bump the line number each time a \n is read */
^.                      { fprintf(yyout, "%4d %s", yylineno, yytext); }
^\n                     { fprintf(yyout, "%4d %s", yylineno++, yytext); }
\n                      { ECHO; yylineno++; }
.                       { ECHO; }

%%

int main(int argc, char **argv)
{
  if (argc < 2) {
    fprintf(stderr, "need filename\n");
    return 1;
  }
  if (newfile(argv[1]))
    yylex();
  return 0;
}

int newfile(char *fn)
{
  FILE *f = fopen(fn, "r");
  struct bufstack *bs = malloc(sizeof(struct bufstack));

  /* if no file, try find it in same dir of prev file */
  if (!f && curbs && curbs->filename) {
    char *dir_end = strrchr(curbs->filename, '/');
    if (dir_end != NULL) {
      size_t dir_len = strlen(curbs->filename) - strlen(dir_end);
      char dir[dir_len + 1]; dir[dir_len] = '\0';
      strncpy(dir, curbs->filename, dir_len);
      char fn_new[strlen(dir) + 1 + strlen(fn) + 1];
      sprintf(fn_new, "%s/%s", dir, fn);
      f = fopen(fn_new, "r");
      if (f) fn = strdup(fn_new);
    }
  }

  /* die if no file or no room */
  if (!f) { perror(fn); return 0; }
  if (!bs) { perror("malloc"); exit(1); }

  /* remember state */
  if (curbs) curbs->lineno = yylineno;
  bs->prev = curbs;

  /* set up current entry */
  bs->bs = yy_create_buffer(f, YY_BUF_SIZE);
  bs->f = f;
  bs->filename = fn;
  yy_switch_to_buffer(bs->bs);
  curbs = bs;
  yylineno = 1;
  curfilename = fn;
  return 1;
}

int popfile(void)
{
  struct bufstack *bs = curbs;
  struct bufstack *prevbs;

  if (!bs) return 0;

  /* get rid of current entry */
  fclose(bs->f);
  yy_delete_buffer(bs->bs);

  /* switch back to previous */
  prevbs = bs->prev;
  free(bs);

  if (!prevbs) return 0;

  yy_switch_to_buffer(prevbs->bs);
  curbs = prevbs;
  yylineno = curbs->lineno;
  curfilename = curbs->filename;
  return 1;
}