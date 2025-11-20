#!/bin/bash
# One-command script to train FDM with optimal settings

set -e

echo "=========================================="
echo "FDM Training - Fast Configuration"
echo "=========================================="
echo ""

# Check if in FBD directory
if [ ! -f "main.py" ]; then
    echo "Error: Please run this script from FBD directory"
    exit 1
fi

# Pull latest code
echo "1. Pulling latest code..."
git pull origin || {
    echo "Warning: git pull failed, continuing with local code"
}
echo ""

# Check if fix is applied
echo "2. Checking fixes..."
if grep -q "torch.real(x_mod)" util/freq_domain_mod.py; then
    echo "   ✓ Complex warning fix applied"
else
    echo "   ✗ Complex warning fix NOT found - may see warnings"
fi

if [ -f "configs/efficientUnet_FDM_zero_workers_CHAOS_to_SABSCT.yaml" ]; then
    echo "   ✓ Zero workers config found"
else
    echo "   ✗ Zero workers config NOT found"
    exit 1
fi
echo ""

# Show configuration
echo "3. Configuration:"
echo "   - num_workers: 0 (no multiprocessing)"
echo "   - fdm_num_bands: 4 (reduced for speed)"
echo "   - fdm_init_from_stats: false (no PAC/SBP overhead)"
echo ""

# Start training
echo "4. Starting training..."
echo "   Expected: time < 15s/iter, data < 5s"
echo ""

python main.py -b configs/efficientUnet_FDM_zero_workers_CHAOS_to_SABSCT.yaml

echo ""
echo "Training completed or interrupted"
