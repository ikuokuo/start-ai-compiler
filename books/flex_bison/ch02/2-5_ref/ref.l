/* C cross-ref */
%option noyywrap nodefault yylineno nounput

%x COMMENT
%x IFILE

/* some complex named patterns */
/* Universal Character Name */
UCN     (\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})
/* float exponent */
EXP     ([Ee][-+]?[0-9]+)
/* integer length */
ILEN    ([Uu](L|l|LL|ll)?|(L|l|LL|ll)[Uu]?)


/* the symbol table */
%{
struct symbol {           /* a variable name */
  struct ref *reflist;
  char *name;
};

struct ref {
  struct ref *next;
  char *filename;
  int flags;              /* 01 - definition */
  int lineno;
};

/* simple symtab of fixed size */
#define NHASH 9997
struct symbol symtab[NHASH];

struct symbol *lookup(char*);
void addref(int, char*, char*, int);

char *curfilename;        /* name of current input file */

/* include file stack */
struct bufstack {
  struct bufstack *prev;  /* previous entry */
  YY_BUFFER_STATE bs;     /* saved buffer */
  int lineno;             /* saved line number in this file */
  char *filename;         /* name of this file */
  FILE *f;                /* current file */
} *curbs;

int newfile(char *fn);
int popfile(void);

int defining;             /* names are probably definitions */
%}

%%

  /* comments */
"/*"                  { BEGIN(COMMENT); }
<COMMENT>"*/"         { BEGIN(INITIAL); }
<COMMENT>([^*]|\n)+|.
<COMMENT><<EOF>>      { printf("%s:%d: Unterminated comment\n",
                            curfilename, yylineno); return 0; }

  /* C++ comment, a common extension */
"//".*\n

  /* declaration keywords */
_Bool |
_Complex |
_Imaginary |
auto |
char |
const |
double |
enum |
extern |
float |
inline |
int |
long |
register |
restrict |
short |
signed |
static |
struct |
typedef |
union |
unsigned |
void |
volatile { defining = 1; }

  /* keywords */
break
case
continue
default
do
else
for
goto
if
return
sizeof
switch
while

  /* constants */

  /* integers */
0[0-7]*{ILEN}?
[1-9][0-9]*{ILEN}?
0[Xx][0-9a-fA-F]+{ILEN}?

  /* decimal float */
([0-9]*\.[0-9]+|[0-9]+\.){EXP}?[flFL]?
[0-9]+{EXP}[flFL]?

  /* hex float */
0[Xx]([0-9a-fA-F]*\.[0-9a-fA-F]+|[0-9a-fA-F]+\.?)[Pp][-+]?[0-9]+[flFL]?

  /* char const */
\'([^'\\]|\\['"?\\abfnrtv]|\\[0-7]{1,3}|\\[Xx][0-9a-fA-F]+|{UCN})+\'

  /* string literal */
L?\"([^"\\]|\\['"?\\abfnrtv]|\\[0-7]{1,3}|\\[Xx][0-9a-fA-F]+|{UCN})*\"

  /* punctuators */
"{"|"<%"|";"        { defining = 0; }

"["|"]"|"("|")"|"{"|"}"|"."|"->"
"++"|"--"|"&"|"*"|"+"|"-"|"~"|"!"
"/"|"%"|"<<"|">>"|"<"|">"|"<="|">="|"=="|"!="|"^"|"|"|"&&"|"||"
"?"|":"|";"|"..."
"="|"*="|"/="|"%="|"+="|"-="|"<<="|">>="|"&="|"^=""|="
","|"#"|"##"
"<:"|":>"|"%>"|"%:"|"%:%:"

  /* identifier */
([_a-zA-Z]|{UCN})([_a-zA-Z0-9]|{UCN})* {
                          addref(yylineno, curfilename, yytext, defining); }

  /* whitespace */
[ \t\n\r\f\v]+

  /* continued line */
\\$

  /* some preprocessor stuff */
"#"" "*if.*\n
"#"" "*else.*\n
"#"" "*endif.*\n
"#"" "*define.*\n
"#"" "*line.*\n

  /* recognize an include */
^"#"[ \t]*include[ \t]*[\"<] { BEGIN IFILE; }

<IFILE>[^>\"]+  {
                  { int c;
                    while ((c = input()) && c != '\n') ;
                  }
                  newfile(strdup(yytext));
                  BEGIN INITIAL;
                }

<IFILE>.|\n     { fprintf(stderr, "%s:%d bad include line\n",
                      curfilename, yylineno);
                  BEGIN INITIAL;
                }

<<EOF>>         { if (!popfile()) yyterminate(); }

  /* invalid character */
.               { printf("%s:%d: Mystery character '%s'\n",
                      curfilename, yylineno, yytext);
                }

%%

/* hash a symbol */
static unsigned
symhash(char *sym)
{
  unsigned int hash = 0;
  unsigned c;

  while ((c = *sym++)) { hash = hash*9 ^ c; };

  return hash;
}

struct symbol *
lookup(char* sym)
{
  struct symbol *sp = &symtab[symhash(sym)%NHASH];
  int scount = NHASH; /* how many have we looked at */

  while (--scount >= 0) {
    if (sp->name && !strcmp(sp->name, sym)) return sp;

    if (!sp->name) { /* new entry */
      sp->name = strdup(sym);
      sp->reflist = 0;
      return sp;
    }

    if (++sp >= symtab+NHASH) sp = symtab; /* try the next entry */
  }

  fputs("symbol table overflow\n", stderr);
  abort(); /* tried them all, table is full */
}

void
addref(int lineno, char *filename, char *word, int flags)
{
  struct ref *r;
  struct symbol *sp = lookup(word);

  /* don't do dups of same line and file */
  if (sp->reflist &&
      sp->reflist->lineno == lineno &&
      sp->reflist->filename == filename) return;

  r = malloc(sizeof(struct ref));
  if (!r) { fputs("out of space\n", stderr); abort(); }
  r->next = sp->reflist;
  r->filename = filename;
  r->lineno = lineno;
  r->flags = flags;
  sp->reflist = r;
}

/* aux function for sorting */
static int
symcompare(const void *xa, const void *xb)
{
  const struct symbol *a = xa;
  const struct symbol *b = xb;
  if (!a->name) {
    if (!b->name) return 0; /* both empty */
    return 1;               /* put empties at the end */
  }
  if (!b->name) return -1;
  return strcmp(a->name, b->name);
}

void
printrefs()
{
  struct symbol *sp;

  qsort(symtab, NHASH, sizeof(struct symbol), symcompare); /* sort the symbol table */

  for (sp = symtab; sp->name && sp < symtab+NHASH; sp++) {
    char *prevfn = NULL;  /* last printed filename, to skip dups */

    /* reverse the list of references */
    struct ref *rp = sp->reflist;
    struct ref *rpp = 0;  /* previous ref */
    struct ref *rpn;      /* next ref */

    do {
      rpn = rp->next;
      rp->next = rpp;
      rpp = rp;
      rp = rpn;
    } while (rp);

    /* now print the word and its references */
    printf("%10s", sp->name);
    for (rp = rpp; rp; rp = rp->next) {
      if (rp->filename == prevfn) {
        printf(" %d", rp->lineno);
      } else {
        printf(" %s:%d", rp->filename, rp->lineno);
        prevfn = rp->filename;
      }
      if (rp->flags & 01) printf("*");
    }
    printf("\n");
  }
}

int
newfile(char *fn)
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

int
popfile(void)
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

int
main(int argc, char **argv)
{
  if (argc < 2) {
    fprintf(stderr, "need filename\n");
    return 1;
  }

  for (int i = 1; i < argc; i++) {
    if (newfile(argv[i]))
    yylex();
  }

  printrefs();
  return 0;
}