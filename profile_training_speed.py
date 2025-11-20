#!/usr/bin/env python
"""
分析训练速度瓶颈

根据你的log：
- 总时间：158.8秒/iter
- 数据加载：138.0秒（87%！）
- 模型计算：~20秒（13%）

这说明主要瓶颈在DataLoader，而不是FDM！
"""

import torch
import torch.nn as nn
import sys
import os
import time
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def profile_fdm_overhead():
    """测量FDM的实际开销"""
    print("=" * 70)
    print("测试1: FDM模块的实际开销")
    print("=" * 70)

    from util.unet_fdm_wrapper import UNetWithFDM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = [
        ("无FDM", {'fdm_enabled': False}),
        ("FDM (4 bands, 不更新统计)", {
            'fdm_enabled': True,
            'fdm_num_bands': 4,
            'fdm_init_from_stats': False,
            'fdm_update_stats_interval': 999999
        }),
        ("FDM (8 bands, 不更新统计)", {
            'fdm_enabled': True,
            'fdm_num_bands': 8,
            'fdm_init_from_stats': False,
            'fdm_update_stats_interval': 999999
        }),
        ("FDM (4 bands, 每1000次更新)", {
            'fdm_enabled': True,
            'fdm_num_bands': 4,
            'fdm_init_from_stats': True,
            'fdm_update_stats_interval': 1000
        }),
        ("FDM (8 bands, 每100次更新)", {
            'fdm_enabled': True,
            'fdm_num_bands': 8,
            'fdm_init_from_stats': True,
            'fdm_update_stats_interval': 100
        }),
    ]

    batch_size = 32
    x = torch.randn(batch_size, 1, 256, 256).to(device)

    for name, config in configs:
        model = UNetWithFDM(
            encoder_name='efficientnet-b2',
            encoder_weights=None,
            in_channels=1,
            classes=5,
            **config
        ).to(device)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        target = torch.randint(0, 5, (batch_size, 256, 256)).to(device)

        # Warmup
        for _ in range(3):
            y = model(x)
            loss = criterion(y, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 测量
        times = []
        for i in range(10):
            # 在第5次触发统计更新（如果enabled）
            if config.get('fdm_enabled') and config.get('fdm_init_from_stats'):
                interval = config.get('fdm_update_stats_interval', 100)
                if model.fdm is not None:
                    for layer in model.fdm.fdm_layers:
                        layer.num_updates.fill_(i * interval)

            start = time.time()
            y = model(x)
            loss = criterion(y, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        print(f"{name:35s}: {avg_time*1000:7.2f} ms/iter ({batch_size/avg_time:5.1f} samples/s)")

        del model, optimizer
        torch.cuda.empty_cache()

    print()


def profile_dataloader_speed():
    """测量DataLoader速度"""
    print("=" * 70)
    print("测试2: DataLoader加载速度")
    print("=" * 70)

    try:
        from dataloaders.AbdominalDataset import get_training
        from torch.utils.data import DataLoader

        train_dataset = get_training(
            location_scale=True,
            modality=['CHAOST2'],
            tile_z_dim=1
        )

        print(f"数据集大小: {len(train_dataset)}")

        # 测试不同num_workers
        for num_workers in [0, 1, 2, 4, 8]:
            print(f"\nnum_workers = {num_workers}:")

            loader = DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=False if num_workers == 0 else True,
                drop_last=True,
                pin_memory=True
            )

            times = []
            for i, batch in enumerate(loader):
                start = time.time()
                # 模拟简单处理
                _ = batch
                times.append(time.time() - start)

                if i >= 10:  # 只测试前10个batch
                    break

            if times:
                avg_time = sum(times) / len(times)
                print(f"  平均加载时间: {avg_time*1000:.2f} ms/batch")

            del loader

    except Exception as e:
        print(f"无法测试DataLoader: {e}")
        print("这可能需要在实际数据目录下运行")

    print()


def profile_forward_backward_breakdown():
    """分解前向/反向传播时间"""
    print("=" * 70)
    print("测试3: 前向/反向传播时间分解")
    print("=" * 70)

    from util.unet_fdm_wrapper import UNetWithFDM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNetWithFDM(
        encoder_name='efficientnet-b2',
        encoder_weights=None,
        in_channels=1,
        classes=5,
        fdm_enabled=True,
        fdm_num_bands=4,
        fdm_init_from_stats=False
    ).to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = 32
    x = torch.randn(batch_size, 1, 256, 256).to(device)
    target = torch.randint(0, 5, (batch_size, 256, 256)).to(device)

    # Warmup
    for _ in range(3):
        y = model(x)
        loss = criterion(y, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 详细计时
    n_runs = 10
    forward_times = []
    loss_times = []
    backward_times = []
    optim_times = []

    for _ in range(n_runs):
        # 前向
        start = time.time()
        y = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_times.append(time.time() - start)

        # 损失
        start = time.time()
        loss = criterion(y, target)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        loss_times.append(time.time() - start)

        # 反向
        start = time.time()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        backward_times.append(time.time() - start)

        # 优化
        start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        optim_times.append(time.time() - start)

    avg_forward = sum(forward_times) / n_runs
    avg_loss = sum(loss_times) / n_runs
    avg_backward = sum(backward_times) / n_runs
    avg_optim = sum(optim_times) / n_runs
    total = avg_forward + avg_loss + avg_backward + avg_optim

    print(f"前向传播:  {avg_forward*1000:7.2f} ms ({avg_forward/total*100:5.1f}%)")
    print(f"损失计算:  {avg_loss*1000:7.2f} ms ({avg_loss/total*100:5.1f}%)")
    print(f"反向传播:  {avg_backward*1000:7.2f} ms ({avg_backward/total*100:5.1f}%)")
    print(f"优化器:    {avg_optim*1000:7.2f} ms ({avg_optim/total*100:5.1f}%)")
    print(f"总计:      {total*1000:7.2f} ms")
    print(f"吞吐量:    {batch_size/total:5.1f} samples/s")

    print()


def main():
    print("\n" + "=" * 70)
    print("训练速度性能分析")
    print("=" * 70)
    print()
    print("根据你的log分析：")
    print("  总时间: 158.8秒/iter")
    print("  数据加载: 138.0秒 (87%！！！)")
    print("  模型计算: ~20秒 (13%)")
    print()
    print("结论：主要瓶颈是DataLoader，不是FDM！")
    print("=" * 70)
    print()

    profile_fdm_overhead()
    profile_forward_backward_breakdown()
    profile_dataloader_speed()

    print("=" * 70)
    print("优化建议：")
    print("=" * 70)
    print()
    print("1. DataLoader优化（最重要！占87%时间）：")
    print("   - 检查数据预处理是否太复杂")
    print("   - 尝试减少num_workers（当前8可能过多）")
    print("   - 考虑预先缓存预处理后的数据")
    print("   - 检查Location Scale Augmentation是否太慢")
    print()
    print("2. FDM优化：")
    print("   - 使用 fdm_num_bands=4 替代 8（减少50%计算量）")
    print("   - 使用 fdm_update_stats_interval=1000 或更大")
    print("   - 或完全禁用统计：fdm_init_from_stats=false")
    print()
    print("3. 配置文件：")
    print("   - 快速模式: configs/efficientUnet_FDM_fast_CHAOS_to_SABSCT.yaml")
    print("   - 极速模式: configs/efficientUnet_FDM_ultra_fast_CHAOS_to_SABSCT.yaml")
    print()


if __name__ == "__main__":
    main()
