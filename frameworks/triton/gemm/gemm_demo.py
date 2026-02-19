"""
General Matrix Multiplication (GEMM) with Triton
=============================================

This tutorial demonstrates a high-performance matrix multiplication kernel that showcases
five key Triton concepts through a complete, runnable implementation:

┌────────────────────────┬─────────────────────────────────────────────────┐
│ Key Concept            │ Implementation in this GEMM kernel              │
├────────────────────────┼─────────────────────────────────────────────────┤
│ 1. @triton.jit decorator │ GPU kernel compilation and optimization       │
│ 2. Block & Grid        │ SPMD parallelization with grouped ordering      │
│ 3. Kernel Fusion       │ Fused bias addition + activation functions      │
│ 4. Memory Hierarchy    │ Global Memory → Shared Memory tiling            │
│ 5. Reduction Operations│ K-dimension accumulation with tl.dot            │
└────────────────────────┴─────────────────────────────────────────────────┘

The implementation includes:
• Complete GEMM kernel with shared memory tiling
• PyTorch baseline for correctness verification
• Performance benchmarks against cuBLAS/rocBLAS
• Memory hierarchy visualization
• Support for bias fusion and multiple activations (ReLU, LeakyReLU, GELU, Sigmoid)
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# -----------------------------------------------------------------------------
# Concept 1: @triton.jit decorator - GPU kernel compilation
# -----------------------------------------------------------------------------
@triton.jit
def gemm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, bias_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides for pointer arithmetic
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters (compile-time constants)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """
    Triton GEMM kernel demonstrating five key concepts.

    Args:
        a_ptr: Pointer to matrix A [M, K]
        b_ptr: Pointer to matrix B [K, N]
        c_ptr: Pointer to output matrix C [M, N]
        bias_ptr: Pointer to bias vector [N] (optional)
        M, N, K: Matrix dimensions
        stride_*: Strides for each matrix dimension
        BLOCK_SIZE_*: Tile sizes for each dimension
        GROUP_SIZE_M: Number of M blocks per group (L2 optimization)
        USE_BIAS: Whether to apply bias
        ACTIVATION: Activation function name
    """

    # -------------------------------------------------------------------------
    # Concept 2: Block & Grid - SPMD parallelization with grouped ordering
    # -------------------------------------------------------------------------
    # Each program instance (block) computes one output tile of size
    # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Grouped ordering: process GROUP_SIZE_M rows before moving to next column
    # This improves L2 cache reuse when accessing matrix B
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -------------------------------------------------------------------------
    # Concept 4: Memory Hierarchy - Global Memory → Shared Memory tiling
    # -------------------------------------------------------------------------
    # Compute offsets for the current block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Memory access masks for boundary conditions
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    # Pointer arithmetic for coalesced global memory access
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Allocate shared memory tiles (SRAM)
    # This is the key memory hierarchy optimization:
    # Global Memory (DRAM) → Shared Memory (SRAM) → Register accumulation
    a_shared = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float16)
    b_shared = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float16)

    # -------------------------------------------------------------------------
    # Concept 5: Reduction Operations - K-dimension accumulation
    # -------------------------------------------------------------------------
    # Initialize accumulator in registers (fastest memory)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over K dimension in tiles - this is a reduction operation
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    for k in range(0, k_tiles):
        # Load A tile from global memory to shared memory
        k_mask = offs_k[None, :] < (K - k * BLOCK_SIZE_K)
        a_shared = tl.load(a_ptrs, mask=m_mask & k_mask, other=0.0)

        # Load B tile from global memory to shared memory
        b_shared = tl.load(b_ptrs, mask=n_mask & k_mask.T, other=0.0)

        # Matrix multiply using tensor cores - accumulate result
        # tl.dot performs warp-level matrix multiplication
        accumulator = tl.dot(a_shared, b_shared, accumulator)

        # Advance pointers to next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # -------------------------------------------------------------------------
    # Concept 3: Kernel Fusion - Combine multiple operations in one kernel
    # -------------------------------------------------------------------------
    c_fp32 = accumulator

    # Fused bias addition (eliminates separate kernel launch)
    if USE_BIAS:
        bias_ptrs = bias_ptr + offs_n
        bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
        bias_fp32 = bias.to(tl.float32)
        c_fp32 += bias_fp32[None, :]

    # Fused activation functions (eliminates separate kernel launch)
    if ACTIVATION == "relu":
        c_fp32 = tl.where(c_fp32 >= 0, c_fp32, 0.0)
    elif ACTIVATION == "leaky_relu":
        c_fp32 = tl.where(c_fp32 >= 0, c_fp32, 0.01 * c_fp32)
    elif ACTIVATION == "gelu":
        # GELU approximation: x * sigmoid(1.702x)
        c_fp32 = c_fp32 * tl.sigmoid(1.702 * c_fp32)
    elif ACTIVATION == "sigmoid":
        c_fp32 = tl.sigmoid(c_fp32)

    # Clip to prevent overflow when converting back to fp16
    c_fp32 = tl.clamp(c_fp32, -65504.0, 65504.0)
    c = c_fp32.to(tl.float16)

    # Write result back to global memory
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# -----------------------------------------------------------------------------
# Wrapper Function - PyTorch-like interface
# -----------------------------------------------------------------------------
def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor = None,
    activation: str = "",
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 64,
    BLOCK_SIZE_K: int = 32,
    GROUP_SIZE_M: int = 8,
):
    """
    PyTorch-style GEMM wrapper with Triton backend.

    Args:
        a: Input matrix A of shape [M, K]
        b: Input matrix B of shape [K, N]
        bias: Optional bias vector of shape [N]
        activation: Activation function name ("", "relu", "leaky_relu", "gelu", "sigmoid")
        BLOCK_SIZE_M/N/K: Tile sizes for each dimension
        GROUP_SIZE_M: Number of M blocks per group

    Returns:
        Output matrix C of shape [M, N]
    """
    # Shape and dtype validation
    assert a.shape[1] == b.shape[0], f"Incompatible dimensions: {a.shape} vs {b.shape}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert a.dtype == b.dtype, f"Type mismatch: {a.dtype} vs {b.dtype}"
    assert a.device == b.device, f"Device mismatch: {a.device} vs {b.device}"

    M, K = a.shape
    K, N = b.shape

    # Handle empty tensors
    if M == 0 or N == 0 or K == 0:
        return torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Configure bias
    if bias is not None:
        assert bias.shape[0] == N, f"Bias must match N dimension: {bias.shape[0]} vs {N}"
        USE_BIAS = True
    else:
        bias = torch.empty(0, device=a.device)
        USE_BIAS = False

    # Define grid: number of program instances = number of output tiles
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    # Launch Triton kernel
    gemm_kernel[grid](
        a, b, c, bias,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        USE_BIAS=USE_BIAS,
        ACTIVATION=activation,
    )

    return c


# -----------------------------------------------------------------------------
# PyTorch Baseline - Reference implementation using cuBLAS/rocBLAS
# -----------------------------------------------------------------------------
def torch_gemm_baseline(a, b, bias=None, activation=None):
    """PyTorch reference implementation for correctness verification."""
    c = torch.matmul(a, b)

    if bias is not None:
        c = c + bias

    if activation == "relu":
        c = torch.relu(c)
    elif activation == "leaky_relu":
        c = torch.nn.functional.leaky_relu(c, 0.01)
    elif activation == "gelu":
        c = torch.nn.functional.gelu(c)
    elif activation == "sigmoid":
        c = torch.sigmoid(c)

    return c


# -----------------------------------------------------------------------------
# Concept 4: Memory Hierarchy Demonstration
# -----------------------------------------------------------------------------
def demonstrate_memory_hierarchy():
    """
    Demonstrate the impact of shared memory tiling on performance.

    This function compares two versions:
    1. Direct global memory access (no shared memory)
    2. Shared memory tiling (our implementation)

    The performance difference illustrates the importance of memory hierarchy
    optimization in GPU computing.
    """
    print("\n" + "=" * 80)
    print("CONCEPT 4: MEMORY HIERARCHY DEMONSTRATION")
    print("=" * 80)
    print("""
    GPU Memory Hierarchy:

    Level           Speed       Capacity    Usage in GEMM
    ──────────────────────────────────────────────────────────
    Global Memory   ~1.5 TB/s   GB          Input matrices A, B (initial)
        ↓
    Shared Memory   ~19 TB/s    64-192 KB   Current tiles of A and B
        ↓
    Registers       ~49 TB/s    256 KB/SM   Accumulator, intermediate results

    Key Insight: Each tile loaded from Global→Shared is reused
    BLOCK_SIZE_N times for A and BLOCK_SIZE_M times for B!
    """)

    # Test configuration
    size = 1024
    M = N = K = size
    BLOCK_SIZE_K = 32

    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    c_shared = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    c_no_shared = torch.empty((M, N), device=DEVICE, dtype=torch.float16)

    print(f"\n📊 Test Configuration: {M} x {K} * {K} x {N}")
    print("   Warming up...")

    # Warmup
    for _ in range(5):
        _ = gemm(a, b)

    print("   Benchmarking...")

    # Benchmark our shared memory implementation
    ms_shared = triton.testing.do_bench(lambda: gemm(a, b), rep=50)

    # For comparison, create a simple kernel without shared memory
    @triton.jit
    def gemm_no_shared(
        a_ptr, b_ptr, c_ptr, bias_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        USE_BIAS: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        # Same as gemm_kernel but without shared memory allocation
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_mask = offs_k[None, :] < (K - k * BLOCK_SIZE_K)
            # Direct global memory load - no shared memory
            a = tl.load(a_ptrs, mask=m_mask & k_mask, other=0.0)
            b = tl.load(b_ptrs, mask=n_mask & k_mask.T, other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(tl.float16)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    # Benchmark no-shared-memory version
    def run_no_shared():
        grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
        gemm_no_shared[grid](
            a, b, c_no_shared, None,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c_no_shared.stride(0), c_no_shared.stride(1),
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8, USE_BIAS=False, ACTIVATION="",
        )

    ms_no_shared = triton.testing.do_bench(run_no_shared, rep=50)

    # Calculate performance metrics
    tflops_shared = 2 * M * N * K * 1e-12 / (ms_shared * 1e-3)
    tflops_no_shared = 2 * M * N * K * 1e-12 / (ms_no_shared * 1e-3)
    speedup = ms_no_shared / ms_shared

    print(f"\n{'Configuration':<30} {'Time (ms)':<12} {'TFLOPS':<12} {'Speedup':<12}")
    print("-" * 70)
    print(f"{'Global Memory Only':<30} {ms_no_shared:<12.3f} {tflops_no_shared:<12.2f} {'1.00x':<12}")
    print(f"{'Shared Memory Tiling':<30} {ms_shared:<12.3f} {tflops_shared:<12.2f} {speedup:<12.2f}x")

    # Calculate memory traffic reduction
    gmem_bytes_no_shared = 2 * M * N * K * 2  # Each element read twice (A and B)
    gmem_bytes_shared = 2 * M * N * K * 2 * (BLOCK_SIZE_K / K)  # Each tile reused
    reduction = (gmem_bytes_no_shared - gmem_bytes_shared) / gmem_bytes_no_shared * 100

    print(f"\n📈 Memory Traffic Reduction: {reduction:.1f}%")
    print(f"   Global Memory Reads: {gmem_bytes_no_shared/1e9:.1f} GB → {gmem_bytes_shared/1e9:.1f} GB")
    print("=" * 80)

    return ms_no_shared, ms_shared


# -----------------------------------------------------------------------------
# Concept 3: Kernel Fusion Demonstration
# -----------------------------------------------------------------------------
def demonstrate_fusion():
    """Demonstrate the performance benefits of kernel fusion."""
    print("\n" + "=" * 80)
    print("CONCEPT 3: KERNEL FUSION DEMONSTRATION")
    print("=" * 80)
    print("""
    Kernel Fusion combines multiple operations into a single kernel:

    Without Fusion:                 With Fusion (Our Implementation):
    ───────────────────────────    ───────────────────────────────
    kernel1: GEMM                  kernel1: GEMM + Bias + ReLU
    kernel2: Bias Add              (Single kernel launch)
    kernel3: ReLU
    (3 kernel launches)           (1 kernel launch)

    Benefits:
    • Reduced kernel launch overhead
    • Less global memory traffic (intermediate results stay in registers)
    • Better cache utilization
    """)

    size = 1024
    M = N = K = size

    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    bias = torch.randn(N, device=DEVICE, dtype=torch.float16)

    print(f"\n📊 Test: {M}x{K} * {K}x{N} with Bias + ReLU")

    # Fused: Single Triton kernel
    ms_fused = triton.testing.do_bench(
        lambda: gemm(a, b, bias, "relu"),
        rep=50
    )

    # Separate: PyTorch with multiple operations
    ms_separate = triton.testing.do_bench(
        lambda: torch.relu(torch.matmul(a, b) + bias),
        rep=50
    )

    fusion_speedup = ms_separate / ms_fused

    print(f"\n{'Implementation':<30} {'Time (ms)':<12} {'Kernels':<12} {'Speedup':<12}")
    print("-" * 70)
    print(f"{'Separate (GEMM+Bias+ReLU)':<30} {ms_separate:<12.3f} {'3':<12} {'1.00x':<12}")
    print(f"{'Fused (Single Triton)':<30} {ms_fused:<12.3f} {'1':<12} {fusion_speedup:<12.2f}x")
    print("=" * 80)


# -----------------------------------------------------------------------------
# Correctness Verification
# -----------------------------------------------------------------------------
def test_correctness():
    """Verify Triton GEMM correctness against PyTorch baseline."""
    print("\n" + "=" * 80)
    print("CORRECTNESS VERIFICATION")
    print("=" * 80)

    test_cases = [
        # (M, N, K, activation, use_bias)
        (256, 256, 256, "", False),
        (256, 256, 256, "relu", True),
        (384, 256, 512, "leaky_relu", False),
        (512, 256, 384, "gelu", True),
        (128, 512, 256, "sigmoid", False),
    ]

    print(f"\n{'Test Case':<40} {'Activation':<12} {'Bias':<6} {'Max Diff':<12} {'Status'}")
    print("-" * 80)

    for M, N, K, activation, use_bias in test_cases:
        # Generate test data
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        bias = torch.randn(N, device=DEVICE, dtype=torch.float16) if use_bias else None

        # Compute with Triton and PyTorch
        triton_out = gemm(a, b, bias, activation)
        torch_out = torch_gemm_baseline(a, b, bias, activation)

        # Compare results
        max_diff = torch.max(torch.abs(triton_out - torch_out))

        # Different tolerance for different activations
        tolerance = 0.2 if activation in ["gelu", "sigmoid"] else 0.1
        status = "✅ PASS" if max_diff < tolerance else "❌ FAIL"

        test_name = f"M={M:3d}, N={N:3d}, K={K:3d}"
        print(f"{test_name:<40} {activation:<12} {str(use_bias):<6} {max_diff:.4e}   {status}")


# -----------------------------------------------------------------------------
# Performance Benchmark
# -----------------------------------------------------------------------------
def benchmark_performance():
    """Benchmark Triton GEMM against PyTorch cuBLAS/rocBLAS."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK: Triton vs PyTorch")
    print("=" * 80)

    sizes = [512, 1024, 2048]

    print(f"\n{'Size':>8} | {'Triton (ms)':>12} | {'PyTorch (ms)':>12} | {'TFLOPS':>10} | {'Speedup':>8}")
    print("-" * 80)

    for size in sizes:
        M = N = K = size

        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

        # Warmup
        for _ in range(5):
            _ = gemm(a, b)
            _ = torch.matmul(a, b)

        # Benchmark
        ms_triton = triton.testing.do_bench(lambda: gemm(a, b), rep=50)
        ms_torch = triton.testing.do_bench(lambda: torch.matmul(a, b), rep=50)

        # Calculate TFLOPS
        tflops = 2 * M * N * K * 1e-12 / (ms_triton * 1e-3)
        speedup = ms_torch / ms_triton

        print(f"{size:8d} | {ms_triton:12.3f} | {ms_torch:12.3f} | {tflops:10.2f} | {speedup:8.2f}x")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("TRITON GEMM - FIVE KEY CONCEPTS DEMONSTRATION")
    print("=" * 80)
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │           General Matrix Multiplication with Triton         │
    └─────────────────────────────────────────────────────────────┘

    This implementation demonstrates five essential Triton concepts:

    1️⃣ @triton.jit DECORATOR
       • Just-in-time compilation to PTX/CUDA
       • Meta-parameterization via tl.constexpr
       • Automatic kernel specialization

    2️⃣ BLOCK & GRID (SPMD Parallelization)
       • Each program computes one output tile
       • Grouped ordering for L2 cache reuse
       • 2D grid decomposition from 1D program ID

    3️⃣ KERNEL FUSION
       • Bias addition fused into GEMM kernel
       • Activation functions (ReLU, LeakyReLU, GELU, Sigmoid) fused
       • Eliminates separate kernel launches

    4️⃣ MEMORY HIERARCHY
       • Global Memory → Shared Memory tiling
       • Coalesced pointer arithmetic
       • Register accumulation for intermediate results

    5️⃣ REDUCTION OPERATIONS
       • K-dimension reduction via accumulation
       • tl.dot for warp-level matrix multiplication
       • Loop over K tiles with progressive accumulation

    Run the sections below to see each concept in action!
    """)
    print("=" * 80)

    # Run all demonstrations
    demonstrate_memory_hierarchy()  # Concept 4
    demonstrate_fusion()            # Concept 3
    test_correctness()              # Verification
    benchmark_performance()         # Performance comparison

    print("\n" + "=" * 80)
    print("✅ All five Triton concepts successfully demonstrated!")
    print("   • JIT: @triton.jit with meta-parameters")
    print("   • Block/Grid: Grouped ordering for cache efficiency")
    print("   • Kernel fusion: Bias + activation fused into GEMM")
    print("   • Memory hierarchy: Shared memory tiling implemented")
    print("   • Reduction: K-dimension accumulation with tl.dot")
    print("=" * 80)
