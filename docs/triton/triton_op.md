# Triton 算子开发

## 官方教程

- [Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

环境确认，运行向量加法，见 [01_vecadd.py](../../frameworks/triton/tutorials/01_vecadd.py)：

```bash
conda activate triton
python3 frameworks/triton/tutorials/01_vecadd.py
```

## 关键概念

| 关键概念 | 简要说明 |
| --- | --- |
| `triton.jit` 装饰器 | 定义 Triton 的内核函数 |
| 块（Block）与网格 | 块级并行，块组成网格来完成计算任务 |
| 核融合（Kernel Fusion） | 提升性能，多个步骤合并到一个内核函数 |
| 内存层次与数据搬运 | 优化关键，减少层级移动（全局内存 -> 共享内存 -> 寄存器） |
| 规约操作 | 聚合结果，需要高效的并行规约实现 |

动手实现一下 GEMM 来学相关概念吧。

## GEMM 示例

GEMM (General Matrix Multiplication) 指通用矩阵乘法，即 matmul（Matrix Multiplication）。

Triton GEMM 官方实现，见 [03_matmul.py](../../frameworks/triton/tutorials/03_matmul.py)：

```bash
python3 frameworks/triton/tutorials/03_matmul.py
```

其用了 `@triton.autotune`，性能接近 `cuBLAS`。

Triton GEMM 用 AI 写的示例，见 [gemm_demo.py](../../frameworks/triton/gemm/gemm_demo.py)：

```bash
python3 frameworks/triton/gemm/gemm_demo.py
```

其清晰展示了上述几个关键概念，用于教学。

## GEMM 实现

### 01. `triton.jit`

代码：

- [gemm_01_jit.py](../../frameworks/triton/gemm/gemm_01_jit.py)

运行：

```bash
python3 frameworks/triton/gemm/gemm_01_jit.py
```

### 02. Block & Grid

代码：

- [gemm_02_blk.py](../../frameworks/triton/gemm/gemm_02_blk.py)

运行：

```bash
python3 frameworks/triton/gemm/gemm_02_blk.py
```

### 03. Kernel Fusion

代码：

- [gemm_03_fusion.py](../../frameworks/triton/gemm/gemm_03_fusion.py)

运行：

```bash
python3 frameworks/triton/gemm/gemm_03_fusion.py
```
