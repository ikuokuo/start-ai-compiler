import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,  # 指针：A、B、C 矩阵
    M, N, K,  # 矩阵维度
    stride_am, stride_ak,  # A 的 strides
    stride_bk, stride_bn,  # B 的 strides
    stride_cm, stride_cn,  # C 的 strides
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """矩阵乘法 kernel: C = A @ B (A: MxK, B: KxN, C: MxN)"""

    # 程序 ID，用于确定当前 block 处理的是 C 中的哪个子块
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 当前 block 在 C 矩阵中的起始位置
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # 用于加载 A、B 的 K 维度的偏移
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 对 K 维度进行分块计算
    for k in range(0, K, BLOCK_SIZE_K):
        # A 子块: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B 子块: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_mask = (offs_k[:, None] < K - k) & (offs_n[None, :] < N)
        b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 累加矩阵乘法结果
        acc += tl.dot(a, b)

    if ACTIVATION == "leaky_relu":
        acc = leaky_relu(acc)

    # 将结果写回 C
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def gemm(a: torch.Tensor, b: torch.Tensor, activation="") -> torch.Tensor:
    """封装函数，用于启动 gemm kernel"""
    # 检查设备
    assert a.device == DEVICE and b.device == DEVICE

    # 矩阵维度
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "Incompatible dimensions"

    # 分配输出矩阵
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 网格大小：每个 block 负责一个 BLOCK_SIZE_M x BLOCK_SIZE_N 的输出块
    #   meta 特指 kernel 函数所有标记为 tl.constexpr 的参数
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    # print(grid({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}))

    # 启动 kernel
    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),  # stride_am, stride_ak
        b.stride(0), b.stride(1),  # stride_bk, stride_bn
        c.stride(0), c.stride(1),  # stride_cm, stride_cn
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
        ACTIVATION=activation,
    )
    return c


def main():
    torch.manual_seed(0)

    # 矩阵维度
    M, N, K = 256, 256, 256
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float32)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float32)

    # Triton 计算结果
    c_triton = gemm(a, b, "leaky_relu")

    # PyTorch 计算结果
    c_mm = torch.matmul(a, b)
    c_torch = torch.where(c_mm >= 0, c_mm, 0.01 * c_mm)  # leaky_relu

    # 比较结果
    print("Triton 结果 C:\n", c_triton)
    print("Torch 结果 C:\n", c_torch)
    diff = torch.max(torch.abs(c_triton - c_torch))
    print(f"最大差异: {diff.item()}")


if __name__ == "__main__":
    main()
