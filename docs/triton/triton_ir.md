# Triton IR

流程：

```txt
Python AST → Triton IR → Triton GPU IR → LLVM IR → PTX → SASS
```

代码：

- [vecadd_ir.py](../../frameworks/triton/debug/vecadd_ir.py)

运行：

```bash
conda activate triton
python3 frameworks/triton/debug/vecadd_ir.py
```

输出：

| 编译阶段 | 对应文件 | 核心作用 | 关键信息/优化 |
| --- | --- | --- | --- |
| Python AST | `01_source.txt` | 解析 Python 语法树 | 控制流结构、变量作用域、函数定义 |
| Triton IR<br>(Triton Dialect) | `02_ttir.mlir` | 平台无关的高级 IR | 块级并行、张量操作、内存访问模式 |
| Triton GPU IR<br>(TritonGPU Dialect) | `03_ttgir.mlir` | GPU 特定优化 | 线程布局；内存层级、向量化、合并访问 |
| LLVM IR | `04_llir.ll` | 低级虚拟机器码 | 寄存器分配、指令选择、平台无关优化 |
| PTX<br>(Parallel Thread Execution) | `05_ptx.ptx` | NVIDIA 汇编中间表示 | 线程级指令、寄存器使用、内存访问指令 |
| SASS/CUBIN<br>(Shader Assembly) | `06_cubin.cubin` | 最终 GPU 机器码 | 二进制可执行代码、资源占用、启动配置 |
