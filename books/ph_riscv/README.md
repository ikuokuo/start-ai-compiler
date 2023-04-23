# 计算机组成与设计 - 硬件/软件接口 RISC-V

- [2020-06 / 计算机组成与设计 - 硬件/软件接口（第5版·RISC-V版）](https://book.douban.com/subject/35088440/)
- [2017-05 / Computer Organization and Design RISC-V Edition](https://book.douban.com/subject/27103952/) / [READ](https://www.cs.sfu.ca/~ashriram/Courses/CS295/assets/books/HandP_RISCV.pdf)

访问在线的[配套资源](https://booksite.elsevier.com/9780128122754/)。

## 计算机体系结构中的 8 个伟大思想

- 面向摩尔定律的设计
- 使用抽象简化设计
- 加速经常性事件
- 通过并行提高性能
- 通过流水线提高性能
- 通过预测提高性能
- 存储层次
- 通过冗余提高可靠性

## 硬件设计 3 条基本原则

- 简单源于规则
- 更少则更快
- 优秀的设计需要适当的折中

## 处理器

- 流水线：取指、译码、执行、访存、写回

## 存储

局部性原理：

- 时间局部性（temporal locality）
  - 如果某个数据项被访问，那么在不久的将来它可能再次被访问
- 空间局部性（spatial locality）
  - 如果某个数据项被访问，与它地址相邻的数据项可能很快也将被访问

存储层次结构（memory hierarchy）：

多级存储采用的结构，即与处理器的距离越远，存储的容量越大，但访问速度越慢。

存储技术：

- 静态随机访问存储（Static Random Access Memory, SRAM）
- 动态随机访问存储（Dynamic Random Access Memory, DRAM）
  - 同步 DRAM （Synchronous DRAM）
  - 双倍数据传输率 DDR （Double Data Rate）
  - 双列直插式内存模块 DIMM （Dual Inline Memory Modules）
- 闪存（flash memory）
  - 电可擦除的可编程只读存储器（Electrically Erasable Programmable Read-only Memory, EEPROM）
- 磁盘（magnetic disk）

缓存（cache）：

- 衡量指标：平均存储访问时间（Average Memory Access Time, AMAT）

虚拟存储：虚拟地址 <> 物理地址

- 页式存储
  - 进程、页表、交换区；TLB
- 段式存储

cache, TLB, 虚拟存储，4 个问题：

- 块可以被放哪里？
  - 一个位置（直接映射）、一些位置（组相联）、任何位置（全相联）
- 如何找到块？
  - 索引，有限的检索，全部检索，单独的查找表（页表）
- 失效时替换哪一块？
  - LRU、随机
- 如何处理写操作？
  - 写穿透、写返回

3C 模型：

- 强制失效
- 容量失效
- 冲突失效

cache 一致性

## 并行处理器

并行硬件分类：

- SISD (Single Instruction stream, Single Data stream)
- SIMD (Single Instruction stream, Multiple Data streams)
- MISD (Multiple Instruction streams, Single Data stream)
- MIMD (Multiple Instruction streams, Multiple Data streams)
  - SPMD (Single Program Multiple Data)

OpenMP, GPU, OpenCL

## RISC-V

### 操作数

- 32 个寄存器（x0~x31, 64 bits）
- 2^61 个存储字（双字 64 bits, 8 Byte, 2^64 Bytes）

### 寻址模式

1. 立即数寻址
2. 寄存器寻址
3. 基址或偏移寻址
4. PC 相对寻址

## 附录A 逻辑设计基础

- 门、真值表和逻辑方程
  - 组合逻辑、时序逻辑
- 组合逻辑电路
- 使用硬件描述语言
  - Verilog、VHDL
- 构建基本算数逻辑单元
  - ALU
- 快速加法：超前进位
- 时钟
- 存储元件：触发器、锁存器和寄存器
- 存储元件：SRAM 和 DRAM
- 有限状态自动机
- 定时方法
- 现场可编程设备
  - FPD: PLD、FPGA
