#!/bin/bash
# 服务器端快速修复脚本
# 解决 CPU 线程竞争导致的 DataLoader 慢问题

set -e

echo "=========================================="
echo "FBD 性能快速修复 (服务器端)"
echo "=========================================="
echo ""

# Step 1: 备份 main.py
echo "1. 备份 main.py..."
if [ ! -f "main.py.backup_thread_fix" ]; then
    cp main.py main.py.backup_thread_fix
    echo "   ✓ 已备份到 main.py.backup_thread_fix"
else
    echo "   ⚠️  备份已存在,跳过"
fi
echo ""

# Step 2: 检查是否已经添加线程限制
if grep -q "OMP_NUM_THREADS" main.py; then
    echo "2. CPU 线程限制..."
    echo "   ✓ 已存在,跳过"
else
    echo "2. 添加 CPU 线程限制到 main.py..."

    # 创建临时文件，在开头添加线程限制
    cat > /tmp/thread_limit_code.py << 'EOFCODE'
# ============================================================
# CPU 线程限制 - 解决 DataLoader 慢的关键!
# 必须在所有 import 之前设置
# ============================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

print("=" * 60)
print("✓ CPU 线程限制已应用:")
print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
print(f"  torch threads: {torch.get_num_threads()}")
print(f"  torch interop: {torch.get_num_interop_threads()}")
print("=" * 60)
print()
# ============================================================

EOFCODE

    # 将线程限制代码添加到 main.py 开头
    cat /tmp/thread_limit_code.py main.py > main.py.tmp
    mv main.py.tmp main.py
    rm /tmp/thread_limit_code.py

    echo "   ✓ 已添加"
fi
echo ""

# Step 3: 检查 num_workers 配置
echo "3. 检查 num_workers 配置..."

# 查找所有配置文件
config_files=$(find configs -name "*.yaml" 2>/dev/null | grep -E "FDM|CHAOS" | head -5)

if [ -z "$config_files" ]; then
    echo "   ⚠️  未找到配置文件"
else
    for cfg in $config_files; do
        if grep -q "num_workers:" "$cfg"; then
            num_workers=$(grep "num_workers:" "$cfg" | head -1 | awk '{print $2}')
            echo "   $cfg: num_workers = $num_workers"

            if [ "$num_workers" -gt 4 ] 2>/dev/null; then
                echo "      ⚠️  num_workers > 4, 建议改为 2 或 4"
            fi
        fi
    done
fi
echo ""

# Step 4: 提供优化建议
echo "4. 优化建议:"
echo "------------------------------------------------------------------------"
echo ""
echo "  已完成:"
echo "    ✅ CPU 线程限制已添加到 main.py"
echo ""
echo "  建议检查配置文件:"
echo ""
echo "    如果 num_workers > 4, 修改为 2 或 4:"
echo "    ---"
echo "    data:"
echo "      params:"
echo "        num_workers: 2  # 改小"
echo "    ---"
echo ""
echo "  可选优化 (如果还想更快):"
echo ""
echo "    禁用 SBF (避免双 forward):"
echo "    ---"
echo "    saliency_balancing_fusion:"
echo "      usage: false"
echo "    ---"
echo ""
echo "    或使用轻量级配置:"
echo "    ---"
echo "    fdm_stages: [3]       # 只在 stage3"
echo "    fdm_num_bands: 3      # 降低 bands"
echo "    ---"
echo ""

# Step 5: 运行建议
echo "=========================================="
echo "✅ 修复完成!"
echo "=========================================="
echo ""
echo "下一步: 运行训练并观察效果"
echo ""
echo "  python main.py -b configs/你的配置.yaml"
echo ""
echo "预期效果:"
echo "  • 原来: time ~7s, data ~6s"
echo "  • 现在: time ~3s, data ~2s  (2-3× 加速!)"
echo ""
echo "如果还是慢,检查 num_workers 和 SBF 设置"
echo ""
echo "恢复原文件:"
echo "  cp main.py.backup_thread_fix main.py"
echo ""
