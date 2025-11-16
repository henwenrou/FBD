"""
U-Net with Frequency Domain Modulation (FDM) Wrapper

This module wraps segmentation_models_pytorch.Unet to inject FDM layers
at specified encoder stages (default: stages 2 and 3, corresponding to
4x and 8x downsampling).
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .freq_domain_mod import FreqDomainModulation


class UNetWithFDM(nn.Module):
    """
    U-Net model wrapped with Frequency Domain Modulation.

    Injects FDM layers after specified encoder stages to perform adaptive
    frequency band amplitude modulation based on structure correlation metrics.
    """

    def __init__(
        self,
        encoder_name="efficientnet-b2",
        encoder_weights=None,
        in_channels=1,
        classes=5,
        activation=None,
        # FDM specific parameters
        fdm_enabled=True,
        fdm_stages=(2, 3),  # Encoder stages to apply FDM (0-indexed, stage 2=4x, stage 3=8x)
        fdm_num_bands=8,
        fdm_alpha=0.5,
        fdm_delta=0.2,
        fdm_eps=1e-6,
        fdm_init_from_stats=True,
        fdm_update_stats_interval=100,
        # Additional SMP Unet parameters
        **kwargs
    ):
        """
        Args:
            encoder_name: Name of encoder (e.g., 'efficientnet-b2', 'resnet50')
            encoder_weights: Pre-trained weights ('imagenet' or None)
            in_channels: Number of input channels
            classes: Number of segmentation classes
            activation: Final activation function
            fdm_enabled: Whether to enable FDM module
            fdm_stages: Tuple of encoder stage indices to apply FDM
            fdm_num_bands: Number of frequency bands for FDM
            fdm_alpha: Weight for PAC-SBP fusion
            fdm_delta: Perturbation strength for statistics computation
            fdm_eps: Numerical stability constant
            fdm_init_from_stats: Initialize weights from SRI statistics
            fdm_update_stats_interval: Interval for updating statistics
        """
        super().__init__()

        # Create base U-Net model
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs
        )

        self.fdm_enabled = fdm_enabled
        self.fdm_stages = fdm_stages

        # Get encoder output channels for each stage
        # segmentation_models_pytorch encoder outputs features at multiple scales
        self.encoder_channels = self.unet.encoder.out_channels
        # encoder_channels is a list like [3, 16, 24, 48, 120, 352] for efficientnet-b2
        # Index 0 is input, indices 1+ are feature stages

        if fdm_enabled and fdm_stages:
            # Validate stages
            max_stage = len(self.encoder_channels) - 1
            for stage in fdm_stages:
                if stage < 1 or stage > max_stage:
                    raise ValueError(f"FDM stage {stage} out of range [1, {max_stage}]")

            # Get channels for FDM stages
            fdm_channels = [self.encoder_channels[s] for s in fdm_stages]

            # Create FDM module
            self.fdm = FreqDomainModulation(
                channels_list=fdm_channels,
                num_bands=fdm_num_bands,
                alpha=fdm_alpha,
                delta=fdm_delta,
                eps=fdm_eps,
                init_from_stats=fdm_init_from_stats,
                update_stats_interval=fdm_update_stats_interval
            )
        else:
            self.fdm = None

    def forward(self, x):
        """
        Forward pass with optional FDM applied to encoder features.

        Args:
            x: Input tensor of shape [N, C, H, W]

        Returns:
            Output segmentation logits of shape [N, classes, H, W]
        """
        # Get encoder features
        features = self.unet.encoder(x)
        # features is a list of tensors at different scales

        # Apply FDM if enabled
        if self.fdm_enabled and self.fdm is not None:
            # Extract features at FDM stages
            fdm_features = [features[s] for s in self.fdm_stages]

            # Apply FDM
            modulated_features = self.fdm(fdm_features)

            # Replace original features with modulated ones
            features = list(features)  # Convert to mutable list
            for i, stage in enumerate(self.fdm_stages):
                features[stage] = modulated_features[i]
            features = tuple(features)

        # Decode
        decoder_output = self.unet.decoder(*features)

        # Segmentation head
        masks = self.unet.segmentation_head(decoder_output)

        return masks

    def get_fdm_stats(self):
        """
        Get FDM statistics (PAC, SBP, SRI) for monitoring.

        Returns:
            Dictionary of statistics or None if FDM is disabled
        """
        if self.fdm is not None:
            return self.fdm.get_stats()
        return None

    def optim_parameters(self):
        """
        Get optimizer parameter groups with potentially different learning rates
        for FDM weights vs. backbone.

        Returns:
            List of parameter group dictionaries
        """
        # Separate FDM parameters from backbone
        if self.fdm is not None:
            fdm_params = list(self.fdm.parameters())
            fdm_param_ids = {id(p) for p in fdm_params}

            backbone_params = [p for p in self.unet.parameters()
                               if id(p) not in fdm_param_ids and p.requires_grad]

            return [
                {"params": backbone_params, "lr_scale": 1.0},
                {"params": fdm_params, "lr_scale": 1.0}  # Can adjust FDM learning rate
            ]
        else:
            return [{"params": [p for p in self.parameters() if p.requires_grad],
                     "lr_scale": 1.0}]

    def set_fdm_enabled(self, enabled):
        """
        Enable or disable FDM module at runtime.

        Args:
            enabled: Boolean to enable/disable FDM
        """
        self.fdm_enabled = enabled

    def freeze_fdm_weights(self):
        """
        Freeze FDM frequency band weights (stop learning them).
        """
        if self.fdm is not None:
            for layer in self.fdm.fdm_layers:
                layer.freq_weights.requires_grad = False

    def unfreeze_fdm_weights(self):
        """
        Unfreeze FDM frequency band weights (resume learning).
        """
        if self.fdm is not None:
            for layer in self.fdm.fdm_layers:
                layer.freq_weights.requires_grad = True
