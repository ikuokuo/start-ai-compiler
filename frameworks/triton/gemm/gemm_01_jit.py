import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
):
    """最简单的 GEMM: 每个线程计算 C 中的一个元素"""

    # 获取当前程序（线程）的 ID
    pid = tl.program_id(axis=0)

    # 将一维 ID 映射到二维索引 (i, j)
    i = pid // N  # 行索引
    j = pid % N   # 列索引

    # 检查边界
    if i < M and j < N:
        # 计算 C[i,j] = sum_{k=0}^{K-1} A[i,k] * B[k,j]
        acc = 0.0
        for k in range(K):
            a = tl.load(a_ptr + i * stride_am + k * stride_ak)
            b = tl.load(b_ptr + k * stride_bk + j * stride_bn)
            acc += a * b

        # 存储结果
        tl.store(c_ptr + i * stride_cm + j * stride_cn, acc)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """启动简单 GEMM kernel"""
    assert a.device == DEVICE and b.device == DEVICE

    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "维度不匹配"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # 网格大小：每个输出元素一个线程
    grid = (M * N,)

    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def main():
    # 使用小矩阵便于观察
    M, N, K = 4, 4, 4

    a = torch.arange(M * K, device=DEVICE, dtype=torch.float32).reshape(M, K)
    b = torch.arange(K * N, device=DEVICE, dtype=torch.float32).reshape(K, N)

    print("矩阵 A:")
    print(a)
    print("\n矩阵 B:")
    print(b)

    c_triton = gemm(a, b)
    c_torch = torch.matmul(a, b)

    print("\nTriton 结果 C:")
    print(c_triton)
    print("\nTorch 结果 C:")
    print(c_torch)
    print(f"\n最大差异: {torch.max(torch.abs(c_triton - c_torch))}")


if __name__ == "__main__":
    main()
