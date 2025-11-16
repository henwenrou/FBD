#!/usr/bin/env python
"""
Test that FDM-enabled model can be instantiated from YAML config
using the same mechanism as main.py
"""

import sys
import os
import importlib

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf


def get_obj_from_str(string, reload=False):
    """Same as in main.py"""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """Same as in main.py"""
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def test_config_instantiation():
    """Test model instantiation from FDM config file."""
    print("=" * 60)
    print("Testing model instantiation from config")
    print("=" * 60)

    # Load config
    config_path = "configs/efficientUnet_FDM_CHAOS_to_SABSCT.yaml"
    print(f"Loading config from: {config_path}")

    config = OmegaConf.load(config_path)
    model_config = config.model

    print(f"\nModel target: {model_config.target}")
    print(f"Model params: {OmegaConf.to_yaml(model_config.params)}")

    # Instantiate model
    print("Instantiating model...")
    model = instantiate_from_config(model_config)

    print(f"\nModel type: {type(model)}")
    print(f"FDM enabled: {model.fdm_enabled}")
    print(f"FDM stages: {model.fdm_stages}")
    print(f"Encoder channels: {model.encoder_channels}")

    # Test forward pass
    import torch
    x = torch.randn(1, 1, 256, 256)
    print(f"\nTest input shape: {x.shape}")

    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {y.shape}")

    # Test optim_parameters (used in main.py)
    print("\nTesting optim_parameters()...")
    param_groups = model.optim_parameters()
    print(f"Number of parameter groups: {len(param_groups)}")
    total_params = sum(len(group['params']) for group in param_groups)
    print(f"Total parameters: {total_params}")

    print("\n✓ Config instantiation test passed!")


def test_disabled_config():
    """Test that FDM-disabled config also works."""
    print("\n" + "=" * 60)
    print("Testing FDM-disabled config")
    print("=" * 60)

    config_path = "configs/efficientUnet_FDM_disabled_CHAOS_to_SABSCT.yaml"
    print(f"Loading config from: {config_path}")

    config = OmegaConf.load(config_path)
    model_config = config.model

    model = instantiate_from_config(model_config)

    print(f"FDM enabled: {model.fdm_enabled}")
    assert not model.fdm_enabled, "FDM should be disabled"

    # Test forward pass
    import torch
    x = torch.randn(1, 1, 256, 256)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {y.shape}")

    print("✓ FDM-disabled config test passed!")


if __name__ == "__main__":
    test_config_instantiation()
    test_disabled_config()
    print("\n" + "=" * 60)
    print("All config tests passed!")
    print("=" * 60)
