import torch
import triton
import triton.language as tl
import os

# 设置一些有用的环境变量（可选，但能获取更完整信息）
os.environ["TRITON_ALWAYS_COMPILE"] = "1"  # 强制重新编译，避免使用缓存
os.environ["MLIR_ENABLE_DUMP"] = "1"       # 启用 MLIR 调试输出

@triton.jit
def vecadd_kernel(
    x_ptr, y_ptr, z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 计算
    z = x + y

    # 存储结果
    tl.store(z_ptr + offsets, z, mask=mask)

def debug_vecadd_all_ir():
    """编译向量加法内核并输出所有IR层次"""

    # 准备数据
    n_elements = 1024 * 10  # 10个block的数据
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    x = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    y = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    z = torch.empty_like(x)

    print("=" * 80)
    print("开始编译向量加法内核...")
    print("=" * 80)

    # 编译内核
    compiled = vecadd_kernel[grid](
        x, y, z, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 打印所有可用的asm层次
    print(f"\n可用的IR层次: {list(compiled.asm.keys())}")

    # 1. Triton IR (高级中间表示)
    if 'ttir' in compiled.asm:
        print("\n" + "=" * 80)
        print("1. Triton IR (TTIR) - 初始中间表示")
        print("=" * 80)
        print(compiled.asm['ttir'])

    # 2. TritonGPU IR (经过GPU优化)
    if 'ttgir' in compiled.asm:
        print("\n" + "=" * 80)
        print("2. TritonGPU IR (TTGIR) - GPU特定优化后")
        print("=" * 80)
        print(compiled.asm['ttgir'])

    # 3. LLVM IR (低级虚拟机器码)
    if 'llir' in compiled.asm:
        print("\n" + "=" * 80)
        print("3. LLVM IR - LLVM中间表示")
        print("=" * 80)
        print(compiled.asm['llir'])

    # 4. PTX (NVIDIA并行线程执行)
    if 'ptx' in compiled.asm:
        print("\n" + "=" * 80)
        print("4. PTX - NVIDIA并行线程执行代码")
        print("=" * 80)
        # 只打印前50行，避免太长
        ptx_lines = compiled.asm['ptx'].split('\n')
        print('\n'.join(ptx_lines[:50]))
        if len(ptx_lines) > 50:
            print(f"... (还有 {len(ptx_lines)-50} 行)")

    # 5. CUBIN (CUDA二进制，如果有)
    if 'cubin' in compiled.asm:
        print("\n" + "=" * 80)
        print("5. CUBIN - CUDA二进制（不可读，显示大小）")
        print("=" * 80)
        print(f"大小: {len(compiled.asm['cubin'])} 字节")

    # 6. 内核元数据
    print("\n" + "=" * 80)
    print("内核元数据")
    print("=" * 80)
    for attr in dir(compiled.metadata):
        if not attr.startswith('_'):  # 只打印非私有属性
            try:
                value = getattr(compiled.metadata, attr)
                print(f"  {attr}: {value}")
            except:
                pass

    return compiled

def debug_specific_ir_levels(levels=None):
    """只输出指定的IR层次"""
    if levels is None:
        levels = ['ttir', 'ttgir', 'ptx']

    # 准备数据
    n_elements = 1024 * 10
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    x = torch.randn(n_elements, device='cuda')
    y = torch.randn(n_elements, device='cuda')
    z = torch.empty_like(x)

    compiled = vecadd_kernel[grid](
        x, y, z, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    for level in levels:
        if level in compiled.asm:
            print(f"\n{'='*60}")
            print(f"IR层次: {level}")
            print(f"{'='*60}")
            print(compiled.asm[level])
        else:
            print(f"警告: {level} 不可用")

    return compiled

def save_ir_to_files():
    """将各层次IR保存到单独的文件"""
    import datetime
    import os
    import inspect

    n_elements = 1024 * 10
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    x = torch.randn(n_elements, device='cuda')
    y = torch.randn(n_elements, device='cuda')
    z = torch.empty_like(x)

    compiled = vecadd_kernel[grid](
        x, y, z, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 获取当前脚本名称（不含.py）
    frame = inspect.currentframe()
    script_name = os.path.splitext(os.path.basename(inspect.getfile(frame)))[0]

    # 创建带脚本名称和时间戳的目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_dir = f"{script_name}_{timestamp}"
    os.makedirs(dump_dir, exist_ok=True)

    # 获取所有可用的IR层次
    available_irs = compiled.asm.keys()
    print(f"可用的IR层次: {list(available_irs)}")

    # 定义IR层次的文件扩展名映射
    extension_map = {
        'ttir': 'mlir',
        'ttgir': 'mlir',
        'llir': 'll',
        'ptx': 'ptx',
        'cubin': 'cubin',
        'amdgcn': 's',
        'hsaco': 'hsaco',
    }

    # 保存所有可用的IR
    saved_count = 0
    for i, ir_type in enumerate(available_irs, 1):
        # 确定文件扩展名
        ext = extension_map.get(ir_type, 'txt')
        filename = f"{dump_dir}/{i:02d}_{ir_type}.{ext}"

        try:
            content = compiled.asm[ir_type]
            # 二进制文件需要以二进制模式写入
            if ir_type in ['cubin', 'hsaco']:
                with open(filename, 'wb') as f:
                    f.write(content)
                size_info = f"{len(content)} 字节"
            else:
                with open(filename, 'w') as f:
                    f.write(content)
                lines = content.count('\n')
                size_info = f"{lines} 行"

            print(f"✓ 已保存 {ir_type} 到 {filename} ({size_info})")
            saved_count += 1
        except Exception as e:
            print(f"✗ 保存 {ir_type} 失败: {e}")

    # 保存内核元数据
    meta_file = f'{dump_dir}/00_metadata.txt'
    with open(meta_file, 'w') as f:
        f.write("内核元数据:\n")
        f.write("=" * 50 + "\n")

        # 打印所有非私有属性
        for attr in dir(compiled.metadata):
            if not attr.startswith('_'):
                try:
                    value = getattr(compiled.metadata, attr)
                    f.write(f"{attr}: {value}\n")
                except:
                    f.write(f"{attr}: <无法访问>\n")

        f.write("\n编译配置:\n")
        f.write("=" * 50 + "\n")
        f.write(f"grid: {grid}\n")
        f.write(f"BLOCK_SIZE: {BLOCK_SIZE}\n")
        f.write(f"n_elements: {n_elements}\n")

    print(f"✓ 已保存元数据到 {meta_file}")
    print(f"\n✅ 共保存 {saved_count + 1} 个文件到目录: {dump_dir}/")

    return compiled


if __name__ == "__main__":
    print("选项:")
    print("1. 输出所有IR层次")
    print("2. 只输出TTIR、TTGIR和PTX")
    print("3. 保存IR到文件")

    choice = input("\n请选择 (1/2/3): ").strip()

    if choice == '1':
        debug_vecadd_all_ir()
    elif choice == '2':
        debug_specific_ir_levels(['ttir', 'ttgir', 'ptx'])
    elif choice == '3':
        save_ir_to_files()
    else:
        print("无效选择，运行默认版本")
        debug_vecadd_all_ir()
