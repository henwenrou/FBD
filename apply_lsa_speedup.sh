#!/bin/bash
# Apply Location Scale Augmentation speedup patch

set -e

echo "=========================================="
echo "Applying LSA Speed Optimization Patch"
echo "=========================================="
echo ""

# Check if in FBD directory
if [ ! -f "dataloaders/location_scale_augmentation.py" ]; then
    echo "Error: Please run this script from FBD directory"
    exit 1
fi

# Backup original file
echo "1. Creating backup..."
cp dataloaders/location_scale_augmentation.py dataloaders/location_scale_augmentation.py.backup
echo "   ✓ Backup saved to: dataloaders/location_scale_augmentation.py.backup"
echo ""

# Show current setting
echo "2. Current setting:"
grep "nTimes=" dataloaders/location_scale_augmentation.py | head -1
echo ""

# Apply patch
echo "3. Applying patch (nTimes: 100000 → 10000)..."
sed -i.tmp 's/nTimes=100000/nTimes=10000/g' dataloaders/location_scale_augmentation.py
rm -f dataloaders/location_scale_augmentation.py.tmp
echo "   ✓ Patch applied"
echo ""

# Verify
echo "4. Verification:"
grep "nTimes=" dataloaders/location_scale_augmentation.py | head -1
echo ""

echo "=========================================="
echo "Patch applied successfully!"
echo "=========================================="
echo ""
echo "Expected speedup: 10x for LSA, 2x overall"
echo "From: time ~16s, data ~13s"
echo "To:   time ~8s,  data ~4s"
echo ""
echo "Now run:"
echo "  python main.py -b configs/efficientUnet_FDM_zero_workers_CHAOS_to_SABSCT.yaml"
echo ""
echo "To restore original (if needed):"
echo "  cp dataloaders/location_scale_augmentation.py.backup dataloaders/location_scale_augmentation.py"
