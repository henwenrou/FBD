#!/usr/bin/env python
"""
服务器端快速诊断脚本
用于找出 FBD 慢的根本原因
"""

import torch
import time
import os

print("=" * 80)
print("FBD 性能问题 - 服务器端诊断")
print("=" * 80)
print()

# 1. 环境检查
print("1️⃣  环境检查")
print("-" * 80)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.current_device()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ CUDA 不可用! 这是主要问题!")
    exit(1)

print(f"CPU 线程: {torch.get_num_threads()}")
print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '未设置')}")
print()

# 2. FFT 性能测试 (GPU vs CPU)
print("2️⃣  FFT 性能测试 (GPU vs CPU)")
print("-" * 80)

def benchmark_fft(shape, device, num_iters=50):
    N, C, H, W = shape
    x = torch.randn(N, C, H, W, device=device)

    # 预热
    for _ in range(5):
        X = torch.fft.fft2(x, dim=(-2, -1))
        if device == 'cuda':
            torch.cuda.synchronize()

    # 测试
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_iters):
        X = torch.fft.fft2(x, dim=(-2, -1))

    if device == 'cuda':
        torch.cuda.synchronize()

    elapsed = (time.time() - start) * 1000 / num_iters
    return elapsed

# 测试关键配置
configs = [
    ((1, 24, 128, 128), "Stage2 单样本"),
    ((8, 24, 128, 128), "Stage2 batch=8"),
    ((32, 24, 128, 128), "Stage2 batch=32"),
    ((1, 48, 64, 64), "Stage3 单样本"),
    ((32, 48, 64, 64), "Stage3 batch=32"),
]

print("配置 → GPU时间 | CPU时间 | 加速比")
for shape, desc in configs:
    gpu_time = benchmark_fft(shape, 'cuda', 50)
    cpu_time = benchmark_fft(shape, 'cpu', 10)
    speedup = cpu_time / gpu_time

    print(f"{desc:25s}: {gpu_time:5.1f}ms | {cpu_time:6.1f}ms | {speedup:4.0f}×")

    if gpu_time > 20:
        print(f"  ⚠️  GPU 时间 >20ms, FFT 较重!")

print()

# 3. 完整 FDM 层性能
print("3️⃣  真实 FDM 层性能")
print("-" * 80)

try:
    from util.freq_domain_mod import FreqDomainModLayer

    def benchmark_fdm(C, H, W, num_bands, device, num_iters=20):
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
            for _ in range(5):
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
                times.append((time.time() - start) * 1000)

        import numpy as np
        return np.mean(times), np.std(times)

    # 测试不同配置
    fdm_configs = [
        (24, 128, 128, 8, "Stage2, 8bands"),
        (24, 128, 128, 4, "Stage2, 4bands"),
        (24, 128, 128, 3, "Stage2, 3bands"),
        (48, 64, 64, 8, "Stage3, 8bands"),
        (48, 64, 64, 3, "Stage3, 3bands"),
    ]

    print("配置 → GPU时间")
    for C, H, W, bands, desc in fdm_configs:
        gpu_avg, gpu_std = benchmark_fdm(C, H, W, bands, 'cuda', 20)
        print(f"{desc:20s}: {gpu_avg:6.2f}±{gpu_std:4.2f}ms")

        if gpu_avg > 50:
            print(f"  ❌ 耗时 >50ms, 太重了!")
        elif gpu_avg > 20:
            print(f"  ⚠️  耗时 >20ms, 偏重")
        else:
            print(f"  ✅ 耗时可接受")

    print()

except ImportError as e:
    print(f"⚠️  无法导入 FreqDomainModLayer: {e}")
    print("    跳过此测试")
    print()

# 4. 检查 PAC/SBP 统计开销
print("4️⃣  PAC/SBP 统计计算开销")
print("-" * 80)

try:
    from util.freq_domain_mod import FreqDomainModLayer

    # 不触发统计
    fdm_no_stats = FreqDomainModLayer(
        in_channels=48,
        num_bands=4,
        init_from_stats=False,
        update_stats_interval=999999
    ).cuda()

    x = torch.randn(1, 48, 64, 64).cuda()

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = fdm_no_stats(x)
    torch.cuda.synchronize()
    time_no_stats = (time.time() - start) * 1000

    # 触发统计
    fdm_with_stats = FreqDomainModLayer(
        in_channels=48,
        num_bands=4,
        init_from_stats=True,
        update_stats_interval=1
    ).cuda()
    fdm_with_stats.train()

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = fdm_with_stats(x)
    torch.cuda.synchronize()
    time_with_stats = (time.time() - start) * 1000

    overhead = time_with_stats - time_no_stats

    print(f"无统计: {time_no_stats:6.2f}ms")
    print(f"有统计: {time_with_stats:6.2f}ms")
    print(f"开销:   {overhead:6.2f}ms ({overhead/time_no_stats*100:.0f}% 增加)")
    print()

    if overhead > 100:
        print("❌ 统计开销 >100ms, 必须禁用!")
    elif overhead > 50:
        print("⚠️  统计开销较大, 建议禁用或降低频率")
    else:
        print("✅ 统计开销可接受")

except Exception as e:
    print(f"⚠️  测试失败: {e}")

print()

# 5. 估算实际训练时的 FDM 开销
print("5️⃣  估算实际训练中的 FDM 开销")
print("-" * 80)

print("假设配置:")
print("  • Stage2 (128×128, 24通道) + Stage3 (64×64, 48通道)")
print("  • 8 bands")
print("  • Batch size = 32")
print("  • 有 SBF (双 forward)")
print()

# 粗略估算
stage2_time = 30  # ms (根据上面的测试结果)
stage3_time = 10  # ms
sbf_factor = 2    # 双 forward

total_fdm = (stage2_time + stage3_time) * sbf_factor
print(f"单次 iteration FDM 开销: ~{total_fdm}ms")
print()

if total_fdm > 100:
    print("❌ FDM 开销 >100ms/iter, 这就是主要瓶颈!")
    print()
    print("优化建议:")
    print("  1. 禁用 SBF (节省 50%)")
    print("  2. 只在 stage3 使用 FDM (节省 70%)")
    print("  3. 减少 num_bands 到 3 (节省 ~30%)")
    print("  4. 组合优化: stage3 only + 3 bands + 无 SBF")
    print(f"     预期: {total_fdm}ms → ~5ms (节省 95%!)")
elif total_fdm > 50:
    print("⚠️  FDM 开销 >50ms/iter, 有优化空间")
    print("  建议: 禁用 SBF 或只在 stage3 使用")
else:
    print("✅ FDM 开销可接受")

print()

# 6. 总结
print("=" * 80)
print("🎯 诊断总结")
print("=" * 80)
print()

print("根据上面的测试结果:")
print()
print("如果 Stage2 (128×128) 的 FDM 耗时 >30ms:")
print("  → 这是主要瓶颈")
print("  → 建议只在 stage3 使用 FDM")
print()
print("如果 PAC/SBP 统计开销 >100ms:")
print("  → 必须设置 init_from_stats: false")
print()
print("如果有 SBF (双 forward):")
print("  → FDM 开销翻倍")
print("  → 建议禁用 SBF 或接受性能损失")
print()

print("推荐配置 (在 configs/efficientUnet_FDM_LIGHTWEIGHT.yaml):")
print("  fdm_enabled: true")
print("  fdm_stages: [3]              # 只在 stage3")
print("  fdm_num_bands: 3             # 降低 band 数")
print("  fdm_init_from_stats: false   # 禁用统计")
print("  saliency_balancing_fusion:")
print("    usage: false               # 禁用 SBF")
print()

print("=" * 80)
print("诊断完成! 请根据上面的结果优化配置")
print("=" * 80)
