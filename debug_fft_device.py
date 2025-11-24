#!/usr/bin/env python
"""
深度排查 FFT 计算位置和开销
重点检查:
1. FFT 是在 CPU 还是 GPU 上计算
2. FFT 计算是否影响 DataLoader (CPU 资源竞争)
3. torch.fft 在不同设备上的性能对比
"""

import torch
import time
import numpy as np
import os

print("=" * 80)
print("FFT 设备和性能深度排查")
print("=" * 80)
print()

# 环境信息
print("🔍 环境检查:")
print(f"  PyTorch 版本: {torch.__version__}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA 设备: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA 版本: {torch.version.cuda}")
print(f"  CPU 线程:")
print(f"    - torch.get_num_threads(): {torch.get_num_threads()}")
print(f"    - OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '未设置')}")
print()

# 测试 1: torch.fft 在 CPU vs GPU 的性能
print("=" * 80)
print("测试 1: torch.fft.fft2 性能 (CPU vs GPU)")
print("=" * 80)
print()

def benchmark_fft(shape, device, num_iters=100):
    """测试 FFT 性能"""
    N, C, H, W = shape
    x = torch.randn(N, C, H, W, device=device)

    # 预热
    for _ in range(5):
        X = torch.fft.fft2(x, dim=(-2, -1))
        if device == 'cuda':
            torch.cuda.synchronize()

    # 测试
    times = []
    for _ in range(num_iters):
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        X = torch.fft.fft2(x, dim=(-2, -1))
        if device == 'cuda':
            torch.cuda.synchronize()

        times.append(time.time() - start)

    return np.mean(times) * 1000, np.std(times) * 1000

# 测试不同分辨率
test_shapes = [
    (1, 24, 128, 128, "Stage2 单样本"),
    (1, 48, 64, 64, "Stage3 单样本"),
    (8, 24, 128, 128, "Stage2 batch=8"),
    (8, 48, 64, 64, "Stage3 batch=8"),
    (32, 48, 64, 64, "Stage3 batch=32"),
]

print("格式: (N, C, H, W) → CPU时间 | GPU时间 | GPU加速比")
print("-" * 80)

results = []
for N, C, H, W, desc in test_shapes:
    # CPU 测试
    cpu_time, cpu_std = benchmark_fft((N, C, H, W), 'cpu', num_iters=20)

    # GPU 测试
    if torch.cuda.is_available():
        gpu_time, gpu_std = benchmark_fft((N, C, H, W), 'cuda', num_iters=100)
        speedup = cpu_time / gpu_time
    else:
        gpu_time = 0
        speedup = 0

    results.append({
        'desc': desc,
        'shape': (N, C, H, W),
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': speedup
    })

    print(f"{desc:25s} ({N},{C},{H},{W})")
    print(f"  CPU: {cpu_time:7.2f}±{cpu_std:5.2f}ms | GPU: {gpu_time:6.2f}ms | 加速: {speedup:5.1f}×")

print()
print("⚠️  关键发现:")
for r in results:
    if r['cpu_time'] > 50:  # CPU 上 >50ms 就算慢
        print(f"  ❌ {r['desc']}: CPU耗时 {r['cpu_time']:.1f}ms - 如果在CPU上跑会很慢!")
    if r['speedup'] > 10:
        print(f"  ✅ {r['desc']}: GPU加速 {r['speedup']:.0f}× - 必须在GPU上跑!")

print()

# 测试 2: 完整 FFT 流程 (fft2 + fftshift + ifftshift + ifft2)
print("=" * 80)
print("测试 2: 完整 FFT 循环 (模拟 FDM 的一次调用)")
print("=" * 80)
print()

def benchmark_full_fft_cycle(shape, device, num_iters=50):
    """测试完整 FFT 循环: fft2 → fftshift → ifftshift → ifft2"""
    N, C, H, W = shape
    x = torch.randn(N, C, H, W, device=device)

    # 预热
    for _ in range(5):
        X = torch.fft.fft2(x, dim=(-2, -1))
        X = torch.fft.fftshift(X, dim=(-2, -1))
        A = torch.abs(X)
        phase = torch.angle(X)
        X_mod = torch.complex(A * torch.cos(phase), A * torch.sin(phase))
        X_mod = torch.fft.ifftshift(X_mod, dim=(-2, -1))
        x_mod = torch.fft.ifft2(X_mod, dim=(-2, -1))
        x_mod = torch.real(x_mod)
        if device == 'cuda':
            torch.cuda.synchronize()

    # 测试
    times = []
    for _ in range(num_iters):
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()

        # 完整 FDM 流程
        X = torch.fft.fft2(x, dim=(-2, -1))
        X = torch.fft.fftshift(X, dim=(-2, -1))
        A = torch.abs(X)
        phase = torch.angle(X)
        # 简单的幅度调制 (不做 band 分解)
        A_mod = A * 1.5
        X_mod = torch.complex(A_mod * torch.cos(phase), A_mod * torch.sin(phase))
        X_mod = torch.fft.ifftshift(X_mod, dim=(-2, -1))
        x_mod = torch.fft.ifft2(X_mod, dim=(-2, -1))
        x_mod = torch.real(x_mod)

        if device == 'cuda':
            torch.cuda.synchronize()

        times.append(time.time() - start)

    return np.mean(times) * 1000, np.std(times) * 1000

print("测试完整 FFT 循环 (fft2 → shift → 调制 → ifft2):")
print("-" * 80)

for N, C, H, W, desc in test_shapes[:3]:  # 只测试前3个
    cpu_time, cpu_std = benchmark_full_fft_cycle((N, C, H, W), 'cpu', num_iters=10)

    if torch.cuda.is_available():
        gpu_time, gpu_std = benchmark_full_fft_cycle((N, C, H, W), 'cuda', num_iters=50)
        speedup = cpu_time / gpu_time
    else:
        gpu_time = 0
        speedup = 0

    print(f"{desc:25s}")
    print(f"  CPU: {cpu_time:7.2f}ms | GPU: {gpu_time:6.2f}ms | 加速: {speedup:5.1f}×")

print()

# 测试 3: 检查 FFT 是否抢占 CPU 资源
print("=" * 80)
print("测试 3: FFT 计算时的 CPU 占用 (模拟 DataLoader 竞争)")
print("=" * 80)
print()

def test_cpu_contention():
    """测试 FFT 在 GPU 上计算时是否占用 CPU"""
    import threading
    import queue

    if not torch.cuda.is_available():
        print("  跳过 (CUDA 不可用)")
        return

    # 创建一个 GPU tensor
    x = torch.randn(8, 48, 64, 64, device='cuda')

    # 监控变量
    cpu_task_times = queue.Queue()

    def cpu_intensive_task(task_id):
        """模拟 DataLoader 的 CPU 密集任务"""
        start = time.time()
        # 模拟 numpy 操作 (类似 LSA)
        arr = np.random.randn(100, 100)
        for _ in range(100):
            arr = np.dot(arr, arr.T)
        elapsed = (time.time() - start) * 1000
        cpu_task_times.put((task_id, elapsed))

    print("场景 A: 无 GPU FFT 时的 CPU 任务耗时")
    print("  (模拟 DataLoader 单独运行)")

    # 启动 CPU 任务
    threads = []
    for i in range(4):
        t = threading.Thread(target=cpu_intensive_task, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    baseline_times = []
    while not cpu_task_times.empty():
        task_id, elapsed = cpu_task_times.get()
        baseline_times.append(elapsed)

    baseline_avg = np.mean(baseline_times)
    print(f"    CPU任务平均耗时: {baseline_avg:.1f}ms")
    print()

    print("场景 B: GPU FFT + CPU 任务并发")
    print("  (模拟 FDM forward + DataLoader 同时运行)")

    # 启动 CPU 任务
    threads = []
    for i in range(4):
        t = threading.Thread(target=cpu_intensive_task, args=(i,))
        threads.append(t)

    # 启动 CPU 任务同时做 GPU FFT
    for t in threads:
        t.start()

    # 做一些 GPU FFT
    fft_start = time.time()
    for _ in range(50):
        X = torch.fft.fft2(x, dim=(-2, -1))
        X = torch.fft.fftshift(X, dim=(-2, -1))
        X = torch.fft.ifftshift(X, dim=(-2, -1))
        x_out = torch.fft.ifft2(X, dim=(-2, -1))
    torch.cuda.synchronize()
    fft_time = (time.time() - fft_start) * 1000

    for t in threads:
        t.join()

    concurrent_times = []
    while not cpu_task_times.empty():
        task_id, elapsed = cpu_task_times.get()
        concurrent_times.append(elapsed)

    concurrent_avg = np.mean(concurrent_times)

    print(f"    GPU FFT 总耗时: {fft_time:.1f}ms")
    print(f"    CPU任务平均耗时: {concurrent_avg:.1f}ms")
    print()

    # 判断干扰
    slowdown = concurrent_avg / baseline_avg
    print(f"  CPU任务减速: {slowdown:.2f}×")
    if slowdown > 1.2:
        print(f"  ❌ GPU FFT 显著拖慢了 CPU 任务 ({slowdown:.1f}× slower)!")
        print(f"     这会干扰 DataLoader 的性能!")
    elif slowdown > 1.05:
        print(f"  ⚠️  轻微影响 ({slowdown:.2f}×)")
    else:
        print(f"  ✅ 几乎无干扰 (GPU FFT 不抢占 CPU)")

test_cpu_contention()

print()

# 测试 4: 实际 FDM 层的设备和耗时
print("=" * 80)
print("测试 4: 真实 FDM 层在不同设备上的性能")
print("=" * 80)
print()

try:
    from util.freq_domain_mod import FreqDomainModLayer

    def benchmark_real_fdm(C, H, W, num_bands, device, num_iters=20):
        """测试真实 FDM 层"""
        fdm = FreqDomainModLayer(
            in_channels=C,
            num_bands=num_bands,
            init_from_stats=False,
            update_stats_interval=999999
        ).to(device)
        fdm.eval()

        x = torch.randn(1, C, H, W, device=device)

        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = fdm(x)
                if device == 'cuda':
                    torch.cuda.synchronize()

        # 测试
        times = []
        with torch.no_grad():
            for _ in range(num_iters):
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                y = fdm(x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)

        return np.mean(times) * 1000, np.std(times) * 1000

    print("真实 FDM 层性能:")
    print("-" * 80)

    fdm_configs = [
        (24, 128, 128, 8, "Stage2, 8bands"),
        (24, 128, 128, 3, "Stage2, 3bands"),
        (48, 64, 64, 8, "Stage3, 8bands"),
        (48, 64, 64, 3, "Stage3, 3bands"),
    ]

    for C, H, W, bands, desc in fdm_configs:
        cpu_time, _ = benchmark_real_fdm(C, H, W, bands, 'cpu', num_iters=5)

        if torch.cuda.is_available():
            gpu_time, gpu_std = benchmark_real_fdm(C, H, W, bands, 'cuda', num_iters=20)
            speedup = cpu_time / gpu_time
        else:
            gpu_time = 0
            gpu_std = 0
            speedup = 0

        print(f"{desc:20s}: CPU {cpu_time:7.1f}ms | GPU {gpu_time:6.2f}±{gpu_std:4.2f}ms | {speedup:4.1f}×")

        # 判断
        if gpu_time > 50:
            print(f"  ❌ GPU耗时 >50ms, FDM 太重!")
        if cpu_time > 500:
            print(f"  ❌ CPU耗时 >500ms, 绝对不能在CPU上跑!")

except ImportError as e:
    print(f"  ⚠️  无法导入 FreqDomainModLayer: {e}")

print()

# 测试 5: 检查 band 循环是否是瓶颈
print("=" * 80)
print("测试 5: FDM 的 band 循环开销")
print("=" * 80)
print()

def benchmark_band_loop(num_bands, device='cuda'):
    """测试不同 band 数量的影响"""
    if device == 'cuda' and not torch.cuda.is_available():
        return 0, 0

    N, C, H, W = 1, 48, 64, 64
    x = torch.randn(N, C, H, W, device=device)

    # 模拟 FDM 的 band 循环
    X = torch.fft.fft2(x, dim=(-2, -1))
    X = torch.fft.fftshift(X, dim=(-2, -1))
    A = torch.abs(X)
    phase = torch.angle(X)

    # 创建 band masks
    h_center, w_center = H // 2, W // 2
    y_coords = torch.arange(H, device=device) - h_center
    x_coords = torch.arange(W, device=device) - w_center
    Y, X_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    radius = torch.sqrt(X_grid.float()**2 + Y.float()**2)
    max_radius = torch.sqrt(torch.tensor([h_center**2 + w_center**2], device=device))

    band_masks = []
    for b in range(num_bands):
        r_low = (b / num_bands) * max_radius
        r_high = ((b + 1) / num_bands) * max_radius
        mask = ((radius >= r_low) & (radius < r_high)).float()
        band_masks.append(mask)

    # 测试
    times = []
    for _ in range(50):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()

        # Band 循环 (这是关键!)
        A_mod = torch.zeros_like(A)
        for b in range(num_bands):
            mask_b = band_masks[b].unsqueeze(0).unsqueeze(0)
            w_b = 1.0  # 简化
            A_mod = A_mod + A * mask_b * w_b

        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)

    return np.mean(times) * 1000, np.std(times) * 1000

print("Band 循环开销 (仅调制部分, 不含 FFT):")
print("-" * 80)

if torch.cuda.is_available():
    for num_bands in [2, 3, 4, 8, 16]:
        avg_time, std_time = benchmark_band_loop(num_bands, 'cuda')
        print(f"  {num_bands:2d} bands: {avg_time:5.2f}±{std_time:4.2f}ms")

print()

# 总结和建议
print("=" * 80)
print("🎯 总结和建议")
print("=" * 80)
print()

print("1. FFT 设备检查:")
print("   ✅ 确保所有 feature 在 GPU 上,避免 CPU FFT")
print("   ✅ 在 freq_domain_mod.py 的 forward 加入:")
print("      assert x.is_cuda, 'FDM input must be on CUDA'")
print()

print("2. FDM 减肥建议:")
if torch.cuda.is_available():
    # 找出最重的配置
    heavy_configs = [r for r in results if r['gpu_time'] > 30]
    if heavy_configs:
        print("   ❌ 以下配置在 GPU 上仍然较重:")
        for r in heavy_configs:
            print(f"      - {r['desc']}: {r['gpu_time']:.1f}ms")
        print()
        print("   建议:")
        print("      • 避免在 stage2 (128×128) 使用 FDM")
        print("      • 只在 stage3 (64×64) 使用")
        print("      • 降低 num_bands 到 3 或 4")

print()
print("3. DataLoader 干扰检查:")
print("   • 如果上面测试显示 GPU FFT 干扰 CPU 任务:")
print("     → 可能是 CUDA kernel launch overhead")
print("     → 或者是 PyTorch 的调度线程")
print("   • 解决方法:")
print("     → 确保 torch.set_num_threads(1)")
print("     → 降低 num_workers")

print()
print("4. 下一步:")
print("   • 查看上面的测试结果")
print("   • 如果 GPU FFT 很快 (<30ms) 但训练慢:")
print("     → 问题在 DataLoader,不在 FDM")
print("   • 如果 GPU FFT 很慢 (>50ms):")
print("     → FDM 确实太重,需要减肥")

print()
print("=" * 80)
print("排查完成! 请根据上面的结果判断瓶颈")
print("=" * 80)
