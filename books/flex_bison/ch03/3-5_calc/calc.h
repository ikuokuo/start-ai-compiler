/* 3-5 declarations for a calculator */

/* interface to the lexer */
extern int yylineno; /* from lexer */
extern int yylex(void);
extern int yyparse(void);
void yyerror(char *s, ...);

/* symbol table */
struct symbol {         /* a variable name */
  char *name;
  double value;
  struct ast *func;     /* stmt for the function */
  struct symlist *syms; /* list of dummy args */
};

/* simple symtab of fixed size */
#define NHASH 9997
struct symbol symtab[NHASH];

struct symbol *lookup(char *);

/* list of symbols, for an argument list */
struct symlist {
  struct symbol *sym;
  struct symlist *next;
};

struct symlist *newsymlist(struct symbol *sym, struct symlist *next);
void symlistfree(struct symlist *sl);

/* node types
 * + - * / |
 * 0-7 comparison ops, bit coded 04 equal, 02 less, 01 greater
 * M unary minus
 * L expression or statement list
 * I IF statement
 * W WHILE statement
 * N symbol ref
 * = assignment
 * S list of symbols
 * F built in function call
 * C user function call
 */

enum bifs {           /* built-in functions */
  B_sqrt = 1,
  B_exp,
  B_log,
  B_print
};

/* nodes in the abstract syntax tree */
/* all have common initial nodetype */

struct ast {
  int nodetype;
  struct ast *l;
  struct ast *r;
};

struct fncall {       /* built-in function */
  int nodetype;       /* type F */
  struct ast *l;
  enum bifs functype;
};

struct ufncall {      /* user function */
  int nodetype;       /* type C */
  struct ast *l;      /* list of arguments */
  struct symbol *s;
};

struct flow {
  int nodetype;       /* type I or W */
  struct ast *cond;   /* condition */
  struct ast *tl;     /* then branch or do list */
  struct ast *el;     /* optional else branch */
};

struct numval {
  int nodetype;       /* type K */
  double number;
};

struct symref {
  int nodetype;       /* type N */
  struct symbol *s;
};

struct symasgn {
  int nodetype;       /* type = */
  struct symbol *s;
  struct ast *v;      /* value */
};

/* build an AST */
struct ast *newast(int nodetype, struct ast *l, struct ast *r);
struct ast *newcmp(int cmptype, struct ast *l, struct ast *r);
struct ast *newfunc(int functype, struct ast *l);
struct ast *newcall(struct symbol *s, struct ast *l);
struct ast *newref(struct symbol *s);
struct ast *newasgn(struct symbol *s, struct ast *v);
struct ast *newnum(double d);
struct ast *newflow(int nodetype, struct ast *cond, struct ast *tl,
                    struct ast *tr);

/* define a function */
void dodef(struct symbol *name, struct symlist *syms, struct ast *stmts);

/* evaluate an AST */
double eval(struct ast *);

/* delete and free an AST */
void treefree(struct ast *);
