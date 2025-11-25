#!/bin/bash
# 紧急修复 data: 43.96s 问题

set -e

echo "=========================================="
echo "🚨 紧急诊断: data: 43.96s 问题"
echo "=========================================="
echo ""

# 1. 检查 LSA nTimes
echo "1️⃣  检查 LSA nTimes 设置"
echo "------------------------------------------------------------------------"

if [ -f "dataloaders/location_scale_augmentation.py" ]; then
    echo "当前 nTimes 设置:"
    grep "nTimes=" dataloaders/location_scale_augmentation.py | head -5
    echo ""

    # 提取数值
    ntimes_value=$(grep "nTimes=" dataloaders/location_scale_augmentation.py | head -1 | grep -o "nTimes=[0-9]*" | cut -d= -f2)

    if [ -n "$ntimes_value" ]; then
        echo "检测到 nTimes = $ntimes_value"
        echo ""

        if [ "$ntimes_value" -gt 1000 ]; then
            echo "❌❌❌ nTimes = $ntimes_value 太大了! 这就是 43s 的原因!"
            echo ""
            echo "立即修复..."

            # 备份
            cp dataloaders/location_scale_augmentation.py dataloaders/location_scale_augmentation.py.backup_emergency

            # 修改为 50
            sed -i 's/nTimes=[0-9]*/nTimes=50/g' dataloaders/location_scale_augmentation.py

            echo "✓ 已修改为 nTimes=50"
            echo ""
            echo "验证:"
            grep "nTimes=" dataloaders/location_scale_augmentation.py | head -3
            echo ""

        elif [ "$ntimes_value" -gt 200 ]; then
            echo "⚠️  nTimes = $ntimes_value 偏大, 建议改为 50"
            echo ""
            read -p "是否修改为 50? (y/n) " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                cp dataloaders/location_scale_augmentation.py dataloaders/location_scale_augmentation.py.backup_emergency
                sed -i 's/nTimes=[0-9]*/nTimes=50/g' dataloaders/location_scale_augmentation.py
                echo "✓ 已修改为 nTimes=50"
            fi
        else
            echo "✅ nTimes = $ntimes_value 合理"
        fi
    else
        echo "⚠️  未能提取 nTimes 数值"
    fi
else
    echo "❌ 文件不存在!"
fi

echo ""

# 2. 检查磁盘速度
echo "2️⃣  检查磁盘 I/O 性能"
echo "------------------------------------------------------------------------"

if [ -d "./data/abdominal/" ]; then
    echo "数据目录: ./data/abdominal/"

    # 检查挂载点
    mount_point=$(df ./data/abdominal/ | tail -1 | awk '{print $1}')
    echo "挂载点: $mount_point"

    # 简单读取测试
    echo ""
    echo "测试文件读取速度 (读取一个样本文件)..."

    sample_file=$(find ./data/abdominal/CHAOST2/processed/ -name "*.nii.gz" -type f | head -1)

    if [ -n "$sample_file" ]; then
        echo "测试文件: $sample_file"
        file_size=$(du -h "$sample_file" | cut -f1)
        echo "文件大小: $file_size"
        echo ""

        echo "读取 3 次测试..."
        for i in {1..3}; do
            # 清除缓存（需要 root，可能失败）
            sync 2>/dev/null || true
            echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

            start=$(date +%s%N)
            cat "$sample_file" > /dev/null
            end=$(date +%s%N)

            elapsed_ms=$(( (end - start) / 1000000 ))
            echo "  读取 $i: ${elapsed_ms}ms"

            if [ "$elapsed_ms" -gt 1000 ]; then
                echo "    ⚠️  >1s, 磁盘较慢"
            fi
        done
    else
        echo "⚠️  未找到样本文件"
    fi
else
    echo "❌ 数据目录不存在!"
fi

echo ""

# 3. 检查配置文件
echo "3️⃣  检查训练配置"
echo "------------------------------------------------------------------------"

config_file="configs/efficientUnet_FDM_CHAOS_to_SABSCT.yaml"

if [ -f "$config_file" ]; then
    echo "配置文件: $config_file"
    echo ""

    echo "num_workers:"
    grep "num_workers:" "$config_file" | head -1

    echo ""
    echo "location_scale:"
    grep "location_scale:" "$config_file" | head -1

    echo ""
    echo "saliency_balancing_fusion:"
    grep -A 2 "saliency_balancing_fusion:" "$config_file" | head -3
else
    echo "⚠️  配置文件不存在"
fi

echo ""

# 4. 给出建议
echo "=========================================="
echo "🎯 诊断总结和建议"
echo "=========================================="
echo ""

echo "根据检查结果:"
echo ""

echo "如果 nTimes > 1000:"
echo "  → 这就是 data: 43s 的主要原因"
echo "  → 已自动修复为 50"
echo "  → 重新训练应该会看到 data: 2-3s"
echo ""

echo "如果磁盘读取 >1s/文件:"
echo "  → 磁盘 I/O 是瓶颈"
echo "  → 考虑将数据复制到本地 SSD"
echo "  → 或者预加载到内存"
echo ""

echo "如果 location_scale: true 且 nTimes 已经很小:"
echo "  → 尝试禁用 LSA: location_scale: false"
echo "  → 测试是否加速"
echo ""

echo "=========================================="
echo "✅ 诊断完成"
echo "=========================================="
echo ""

echo "下一步:"
echo "  1. 如果修改了 nTimes, 重新训练:"
echo "     python main.py -b configs/efficientUnet_FDM_CHAOS_to_SABSCT.yaml"
echo ""
echo "  2. 观察新的 data 时间:"
echo "     预期: data: 2-4s (从 43s 降到 2-4s!)"
echo ""
echo "  3. 如果还是慢, 运行详细诊断:"
echo "     python diagnose_dataloader_bottleneck.py"
echo ""

echo "恢复原文件 (如需要):"
echo "  cp dataloaders/location_scale_augmentation.py.backup_emergency \\"
echo "     dataloaders/location_scale_augmentation.py"
echo ""
