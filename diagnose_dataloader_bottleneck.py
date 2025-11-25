#!/usr/bin/env python
"""
深度排查 DataLoader 为什么这么慢 (43.96s!)
"""

import time
import sys
import os

# 必须先设置环境变量
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

print("=" * 80)
print("DataLoader 瓶颈深度诊断")
print("=" * 80)
print()

print(f"✓ OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
print(f"✓ torch threads: {torch.get_num_threads()}")
print()

# 测试单个数据样本的加载时间
print("测试 1: 单个样本加载时间")
print("-" * 80)

try:
    from dataloaders.AbdominalDataset import AbdominalDataset
    import dataloaders.transform_utils as trans

    # 创建最简单的 dataset
    transforms_train = [
        trans.RandomRotation((-10, 10)),
        trans.RandomCrop((224, 224)),
    ]

    print("创建 dataset (不使用 LSA)...")
    dataset_no_lsa = AbdominalDataset(
        mode='train',
        transforms=transforms_train,
        base_dir='./data/abdominal/',
        domains=['CHAOST2'],
        idx_pct=[0.7, 0.1, 0.2],
        tile_z_dim=1,
        location_scale=False  # 关键: 禁用 LSA
    )

    print(f"Dataset 大小: {len(dataset_no_lsa)}")
    print()

    # 测试单个样本
    print("测试单个样本加载 (无 LSA):")
    times = []
    for i in range(5):
        start = time.time()
        sample = dataset_no_lsa[0]
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  样本 {i}: {elapsed:.0f}ms")

    import numpy as np
    avg_no_lsa = np.mean(times)
    print(f"  平均: {avg_no_lsa:.0f}ms")
    print()

    # 测试带 LSA
    print("创建 dataset (使用 LSA)...")
    dataset_with_lsa = AbdominalDataset(
        mode='train',
        transforms=transforms_train,
        base_dir='./data/abdominal/',
        domains=['CHAOST2'],
        idx_pct=[0.7, 0.1, 0.2],
        tile_z_dim=1,
        location_scale=True  # 开启 LSA
    )

    print("测试单个样本加载 (有 LSA):")
    times = []
    for i in range(5):
        start = time.time()
        sample = dataset_with_lsa[0]
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  样本 {i}: {elapsed:.0f}ms")

    avg_with_lsa = np.mean(times)
    print(f"  平均: {avg_with_lsa:.0f}ms")
    print()

    lsa_overhead = avg_with_lsa - avg_no_lsa
    print(f"⚠️  LSA 开销: {lsa_overhead:.0f}ms ({lsa_overhead/avg_no_lsa*100:.0f}% 增加)")
    print()

    if lsa_overhead > 1000:
        print("❌❌❌ LSA 开销 >1s, 这就是主要瓶颈!")
        print()
        print("检查 location_scale_augmentation.py 中的 nTimes:")
        import dataloaders.location_scale_augmentation as lsa_module
        import inspect
        source = inspect.getsource(lsa_module.LocationScaleAugmentation.__init__)
        for line in source.split('\n'):
            if 'nTimes' in line:
                print(f"  {line.strip()}")
        print()

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 测试 DataLoader 的 worker 性能
print("测试 2: DataLoader 性能 (不同 num_workers)")
print("-" * 80)

try:
    from torch.utils.data import DataLoader

    # 测试不同的 num_workers
    for num_workers in [0, 1, 2, 4]:
        print(f"\nnum_workers = {num_workers}:")

        loader = DataLoader(
            dataset_with_lsa,
            batch_size=8,
            num_workers=num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True
        )

        # 测试前 3 个 batch
        times = []
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            start = time.time()
            # 模拟一些简单处理
            _ = batch
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            print(f"  Batch {i}: {elapsed:.0f}ms")

        avg_time = np.mean(times) if times else 0
        print(f"  平均: {avg_time:.0f}ms")

        if avg_time > 10000:
            print(f"  ❌ 平均时间 >{avg_time/1000:.1f}s, 太慢了!")

        del loader

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 检查 LSA 的 nTimes 设置
print("测试 3: 检查 LSA 配置")
print("-" * 80)

try:
    # 读取文件检查 nTimes
    with open('dataloaders/location_scale_augmentation.py', 'r') as f:
        content = f.read()

    # 查找 nTimes 的默认值
    import re
    matches = re.findall(r'nTimes\s*=\s*(\d+)', content)

    if matches:
        print(f"LSA nTimes 当前设置:")
        for i, val in enumerate(matches):
            print(f"  出现 {i+1}: nTimes = {val}")
            if int(val) > 1000:
                print(f"    ❌ 值太大! 建议降到 100 或 50")
    else:
        print("⚠️  未找到 nTimes 设置")

except Exception as e:
    print(f"❌ 无法读取文件: {e}")

print()

# 总结
print("=" * 80)
print("🎯 诊断总结")
print("=" * 80)
print()

print("根据你的训练日志: data: 43.96s")
print()
print("可能的原因:")
print()
print("1. LSA 的 nTimes 太大 (>1000)")
print("   → 检查 location_scale_augmentation.py")
print("   → 建议改为 50 或 100")
print()
print("2. 磁盘 I/O 太慢")
print("   → NIfTI 文件读取慢")
print("   → 考虑预加载到内存或用 SSD")
print()
print("3. num_workers 太少或太多")
print("   → 当前设置为 2")
print("   → 尝试 0 (单进程) 或 4")
print()
print("4. Saliency 计算太重")
print("   → 如果开启了 SBF")
print("   → 考虑禁用")
print()

print("下一步:")
print("  1. 运行: python diagnose_dataloader_bottleneck.py")
print("  2. 查看哪个环节最慢")
print("  3. 针对性优化")
print()
