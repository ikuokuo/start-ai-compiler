<!-- markdownlint-disable MD033 -->
# [Flex](https://github.com/westes/flex) 与 [Bison](https://github.com/akimd/bison)

- [2011-03 / flex与bison（中文版）](https://book.douban.com/subject/6109479/) / [阅读](http://home.ustc.edu.cn/~guoxing/ebooks/flex%E4%B8%8Ebison%E4%B8%AD%E6%96%87%E7%89%88.pdf)
- [2009 / flex & bison - Text Processing Tools](https://book.douban.com/subject/3568327/) / [阅读](https://web.iitd.ac.in/~sumeet/flex__bison.pdf)

## 简介

Flex 与 Bison 是为编译器和解释器的编程人员特别设计的工具，不过后来它们在其他应用领域被证明也非常有效。任何应用程序，尤其文本处理，只要在其输入中寻找特定的模式，或者它使用命令语言作为输入，都适合使用 Flex 与 Bison。

Flex 用于词法分析（lexical analysis，或称 scanning），把输入分割成一个个有意义的词块，称为记号（token）。Bison 用于语法分析（syntax analysis，或称 parsing），确定这些记号是如何彼此关联的。例如，如下代码片段：

```c
alpha = beta + gamma;
```

词法分析把这段代码分解为这样一些记号：`alpha`, `=`, `beta`, `+`, `gamma`, `;`。接着语法分析确定了 `beta + gamma` 是一个表达式，而这个表达式被赋给了 `alpha`。

## 安装

大多数 Linux 和 BSD 系统自带 flex 和 bison 作为系统的基础部分。如果你的系统没有包含它们，安装它们也很容易。

例如在 Ubuntu/Debian 系统，可以直接 apt 安装：

```bash
# Ubuntu 20
$ sudo apt install flex bison -y

$ flex -V
flex 2.6.4
$ bison -V
bison (GNU Bison) 3.5.1
```

## 范例

当前路径下有我的范例，如下编译所有：

```bash
cd books/flex_bison/

# 编译 release
make
# 编译 debug
make debug

# 清理
make clean
```

范例程序会输出进 `_build` 目录，如下执行：

```bash
$ ./_build/linux-x86_64/release/1-5_calc/bin/1-5_calc
> (1+2)*3 + 4/2
= 11
```

如果只编译某一范例：

```bash
cd ch01/1-1_wc/

# 编译 release
make -j8
# 编译 debug
make -j8 args="debug"

# 清理
make clean
```

注：找原书范例，可[参考这儿](https://github.com/shaoran/flex_and_bison_updated_examples)。

<hr />

## 笔记：Flex

### 正则表达式

Flex 的正则表达式语言本质上是扩展的 POSIX 正则表达式。

Flex 会匹配尽可能多的字符串、匹配在程序中更早出现的模式。

<hr />

## 笔记：Bison

### BNF 文法

为了编写一个语法分析器，我们需要一定的方法来描述语法分析器所使用的把一系列记号转化为语法分析树的规则。在计算机分析程序里最常用的语言就是上下文无关文法 (Context-Free Grammar, CFG)。书写上下文无关文法的标准格式就是 Backus-Naur 范式 (Backus-Naur Form, BNF)。

```txt
<exp> ::= <factor>
    | <exp> + <factor>
<factor> ::= NUMBER
    | <factor> * NUMBER
```

每一行就是一条规则，且规则总是带有递归性的。

- `::=` 是、变成
- `|` 或者

Bison 的规则基本上就是 BNF，但是做了一点点简化以易于输入。

### 分析方法

Bison 可以使用两种分析方法：

- GLR (Generalized Left to Right)
- LALR(1) (Look Ahead Left to Right with a one-token lookahead)
  - LALR 不如 GLR 强大，但认为比 GLR 更快更容易使用
  - LALR 其实也很强大，只是不能处理有歧义的语法
    - 二义性文法让人迷惑要规避，但也有技巧解决

### 抽象语法树

AST (abstract syntax tree)

- 移进 (shift)
- 归约 (reduction)

### 指针模型

一个指针将会在每次读到一个记号（token）时在 Bison 语法中移动。

- 当指针达到规则结束的位置时，该规则将被归约。
- 在有多个指针的情况下归约一个规则会存在冲突。

归约过程：

```
start: x
     | y;
x: A ↑;  /* 读到 A 后，指针 ↑ 达到规则 x 的末尾。仅存一个，规则 x 将被归约 */
y: B;
```

归约/归约冲突：

```
start: x
     | y;
x: A ↑;  /* 读到 A 后，指针 ↑ 达到规则 x 的末尾 */
y: A ↑;  /* 读到 A 后，指针 ↑ 达到规则 y 的末尾。存在两个，归约冲突 */
```

移进/归约冲突：

```
start: x
     | y R;
x: A ↑ R;  /* 读到 A 后，指针 ↑ 还在移进 */
y: A ↑;    /* 读到 A 后，指针 ↑ 正在规约 */
```

归约正常：

```
start: x B
     | y C;
x: A ↑;
y: A ↑;  /* Bison 在 A 之后会（也只能）预读一个记号，看见是 B 还是 C */
```

## 笔记：SQL

SQL (Structured Query Language)

- MySQL: C++ 词法分析, Bison 语法分析
    - [sql/sql_yacc.yy](https://github.com/mysql/mysql-server/blob/8.0/sql/sql_yacc.yy)
- PostgreSQL: Flex 词法分析, Bison 语法分析
    - [parser/scan.l](https://github.com/postgres/postgres/blob/master/src/backend/parser/scan.l)
    - [parser/gram.y](https://github.com/postgres/postgres/blob/master/src/backend/parser/gram.y)

### 逆波兰式

本书记号化的 SQL 版本将使用逆波兰式 (Reverse Polish Notation，RPN)。
