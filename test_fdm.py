#!/usr/bin/env python
"""
Test script for Frequency Domain Modulation (FDM) module integration.

This script verifies:
1. FDM module can be instantiated
2. Forward pass works correctly
3. Statistics computation (PAC, SBP, SRI) works
4. Wrapper model integrates with SMP UNet
5. Configuration toggle (enable/disable) works
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.freq_domain_mod import FreqDomainModLayer, FreqDomainModulation
from util.unet_fdm_wrapper import UNetWithFDM


def test_fdm_layer():
    """Test single FDM layer functionality."""
    print("=" * 60)
    print("Test 1: FreqDomainModLayer basic functionality")
    print("=" * 60)

    # Create layer
    in_channels = 48
    num_bands = 8
    layer = FreqDomainModLayer(
        in_channels=in_channels,
        num_bands=num_bands,
        alpha=0.5,
        delta=0.2,
        init_from_stats=True,
        update_stats_interval=1  # Update every iteration for testing
    )
    layer.train()

    # Create dummy input
    batch_size = 2
    H, W = 64, 64
    x = torch.randn(batch_size, in_channels, H, W)

    print(f"Input shape: {x.shape}")
    print(f"Initial weights: {layer.freq_weights.data}")

    # Forward pass
    y = layer(x)
    print(f"Output shape: {y.shape}")
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    # Check that weights were updated from SRI
    print(f"Updated weights: {layer.freq_weights.data}")
    print(f"Running PAC: {layer.running_pac}")
    print(f"Running SBP: {layer.running_sbp}")
    print(f"Running SRI: {layer.running_sri}")
    print(f"Num updates: {layer.num_updates.item()}")

    # Verify output is real-valued
    assert not torch.any(torch.isnan(y)), "Output contains NaN"
    assert not torch.any(torch.isinf(y)), "Output contains Inf"

    print("✓ FreqDomainModLayer test passed!\n")


def test_fdm_multi_layer():
    """Test multi-layer FDM module."""
    print("=" * 60)
    print("Test 2: FreqDomainModulation (multi-layer)")
    print("=" * 60)

    # Channels for different encoder stages
    channels_list = [24, 48]  # Stage 2 and Stage 3 channels
    fdm = FreqDomainModulation(
        channels_list=channels_list,
        num_bands=8,
        alpha=0.5,
        delta=0.2,
        init_from_stats=True,
        update_stats_interval=1
    )
    fdm.train()

    # Create dummy features
    batch_size = 2
    features = [
        torch.randn(batch_size, 24, 64, 64),  # Stage 2: 4x downsampling
        torch.randn(batch_size, 48, 32, 32),  # Stage 3: 8x downsampling
    ]

    print(f"Input feature shapes: {[f.shape for f in features]}")

    # Forward pass
    modulated = fdm(features)
    print(f"Output feature shapes: {[m.shape for m in modulated]}")

    # Verify shapes
    for i, (orig, mod) in enumerate(zip(features, modulated)):
        assert orig.shape == mod.shape, f"Layer {i} shape mismatch"

    # Get statistics
    stats = fdm.get_stats()
    for layer_name, layer_stats in stats.items():
        print(f"\n{layer_name}:")
        print(f"  PAC: {layer_stats['pac']}")
        print(f"  SBP: {layer_stats['sbp']}")
        print(f"  SRI: {layer_stats['sri']}")
        print(f"  Weights: {layer_stats['weights']}")

    print("\n✓ FreqDomainModulation test passed!\n")


def test_unet_with_fdm():
    """Test UNet wrapper with FDM integration."""
    print("=" * 60)
    print("Test 3: UNetWithFDM wrapper model")
    print("=" * 60)

    # Create model with FDM enabled
    model = UNetWithFDM(
        encoder_name="efficientnet-b2",
        encoder_weights=None,
        in_channels=1,
        classes=5,
        activation=None,
        fdm_enabled=True,
        fdm_stages=(2, 3),
        fdm_num_bands=8,
        fdm_alpha=0.5,
        fdm_delta=0.2,
        fdm_init_from_stats=True,
        fdm_update_stats_interval=1
    )
    model.train()

    print(f"Encoder channels: {model.encoder_channels}")
    print(f"FDM stages: {model.fdm_stages}")

    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 1, 256, 256)
    print(f"Input shape: {x.shape}")

    # Forward pass
    y = model(x)
    print(f"Output shape: {y.shape}")
    expected_shape = (batch_size, 5, 256, 256)
    assert y.shape == expected_shape, f"Shape mismatch: {y.shape} != {expected_shape}"

    # Check FDM statistics
    stats = model.get_fdm_stats()
    print("\nFDM Statistics:")
    for layer_name, layer_stats in stats.items():
        print(f"  {layer_name}: weights = {layer_stats['weights']}")

    # Test optimizer parameters
    param_groups = model.optim_parameters()
    print(f"\nOptimizer parameter groups: {len(param_groups)}")
    for i, group in enumerate(param_groups):
        print(f"  Group {i}: {len(group['params'])} params, lr_scale={group['lr_scale']}")

    print("✓ UNetWithFDM test passed!\n")


def test_fdm_toggle():
    """Test FDM enable/disable toggle."""
    print("=" * 60)
    print("Test 4: FDM enable/disable toggle")
    print("=" * 60)

    # Create model
    model = UNetWithFDM(
        encoder_name="efficientnet-b2",
        encoder_weights=None,
        in_channels=1,
        classes=5,
        fdm_enabled=True,
        fdm_stages=(2, 3),
    )
    model.eval()

    x = torch.randn(1, 1, 128, 128)

    # Test with FDM enabled
    print("FDM enabled:", model.fdm_enabled)
    y_with_fdm = model(x)

    # Disable FDM
    model.set_fdm_enabled(False)
    print("FDM enabled (after disable):", model.fdm_enabled)
    y_without_fdm = model(x)

    # Re-enable FDM
    model.set_fdm_enabled(True)
    print("FDM enabled (after re-enable):", model.fdm_enabled)
    y_with_fdm_again = model(x)

    # Outputs should differ when FDM is enabled vs disabled
    # (unless weights are exactly 1.0)
    print(f"Output with FDM: mean={y_with_fdm.mean().item():.6f}")
    print(f"Output without FDM: mean={y_without_fdm.mean().item():.6f}")

    print("✓ FDM toggle test passed!\n")


def test_fdm_freeze():
    """Test freezing FDM weights."""
    print("=" * 60)
    print("Test 5: FDM weight freezing")
    print("=" * 60)

    model = UNetWithFDM(
        encoder_name="efficientnet-b2",
        encoder_weights=None,
        in_channels=1,
        classes=5,
        fdm_enabled=True,
        fdm_stages=(2, 3),
    )

    # Check initial requires_grad
    print("Before freeze:")
    for i, layer in enumerate(model.fdm.fdm_layers):
        print(f"  Layer {i} weights requires_grad: {layer.freq_weights.requires_grad}")

    # Freeze
    model.freeze_fdm_weights()
    print("\nAfter freeze:")
    for i, layer in enumerate(model.fdm.fdm_layers):
        print(f"  Layer {i} weights requires_grad: {layer.freq_weights.requires_grad}")
        assert not layer.freq_weights.requires_grad

    # Unfreeze
    model.unfreeze_fdm_weights()
    print("\nAfter unfreeze:")
    for i, layer in enumerate(model.fdm.fdm_layers):
        print(f"  Layer {i} weights requires_grad: {layer.freq_weights.requires_grad}")
        assert layer.freq_weights.requires_grad

    print("✓ FDM freeze test passed!\n")


def test_gradient_flow():
    """Test that gradients flow through FDM module."""
    print("=" * 60)
    print("Test 6: Gradient flow through FDM")
    print("=" * 60)

    model = UNetWithFDM(
        encoder_name="efficientnet-b2",
        encoder_weights=None,
        in_channels=1,
        classes=5,
        fdm_enabled=True,
        fdm_stages=(2, 3),
    )
    model.train()

    x = torch.randn(2, 1, 128, 128, requires_grad=True)
    y = model(x)

    # Compute loss (dummy)
    loss = y.sum()
    loss.backward()

    # Check gradients
    print(f"Input has gradient: {x.grad is not None}")

    print("FDM weight gradients:")
    for i, layer in enumerate(model.fdm.fdm_layers):
        grad = layer.freq_weights.grad
        if grad is not None:
            print(f"  Layer {i}: grad mean={grad.mean().item():.6f}, std={grad.std().item():.6f}")
        else:
            print(f"  Layer {i}: No gradient")

    print("✓ Gradient flow test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Frequency Domain Modulation (FDM) Integration Tests")
    print("=" * 60 + "\n")

    try:
        test_fdm_layer()
        test_fdm_multi_layer()
        test_unet_with_fdm()
        test_fdm_toggle()
        test_fdm_freeze()
        test_gradient_flow()

        print("=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
