# Flex 与 Bison

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
$ ./_build/linux-x86_64/release/1-1_wc/bin/1-1_wc
hello flex
       1       2      11
```

如果只编译某一范例：

```bash
cd ch01/1-1_wc/

# 编译 release
make
# 编译 debug
make args="debug"

# 清理
make clean
```

注：找原书范例，可[参考这儿](https://github.com/shaoran/flex_and_bison_updated_examples)。
