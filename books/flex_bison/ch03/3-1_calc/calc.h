/* 3-1 declarations for a calculator */

/* interface to the lexer */
extern int yylineno; /* from lexer */
extern int yylex(void);
extern int yyparse(void);
void yyerror(char *s, ...);

/* nodes in the abstract syntax tree */
struct ast {
  int nodetype;
  struct ast *l;
  struct ast *r;
};

struct numval {
  int nodetype; /* type K for constant */
  double number;
};

/* build an AST */
struct ast *newast(int nodetype, struct ast *l, struct ast *r);
struct ast *newnum(double d);

/* evaluate an AST */
double eval(struct ast *);

/* delete and free an AST */
void treefree(struct ast *);
