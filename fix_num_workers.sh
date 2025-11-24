#!/bin/bash
# 批量修改配置文件的 num_workers

echo "=========================================="
echo "批量修改 num_workers (8 → 2)"
echo "=========================================="
echo ""

# 查找并修改所有包含 FDM 或 CHAOS 的配置文件
config_files=$(find configs -name "*.yaml" | grep -E "FDM|CHAOS")

for cfg in $config_files; do
    if grep -q "num_workers:" "$cfg"; then
        echo "修改: $cfg"

        # 备份
        cp "$cfg" "${cfg}.backup"

        # 修改 num_workers: 8 → num_workers: 2
        sed -i 's/num_workers: 8/num_workers: 2/g' "$cfg"

        # 验证
        new_value=$(grep "num_workers:" "$cfg" | head -1 | awk '{print $2}')
        echo "  原值: 8 → 新值: $new_value"
        echo "  ✓ 备份: ${cfg}.backup"
        echo ""
    fi
done

echo "=========================================="
echo "✅ 修改完成!"
echo "=========================================="
echo ""
echo "下一步: 运行训练"
echo "  python main.py -b configs/efficientUnet_FDM_CHAOS_to_SABSCT.yaml"
echo ""
echo "如需恢复原配置:"
echo "  cp configs/你的配置.yaml.backup configs/你的配置.yaml"
echo ""
