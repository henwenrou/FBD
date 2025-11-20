"""
Frequency Domain Modulation Module (FDM)

Phase-Amplitude Coupling Driven Multi-Layer Frequency Domain Adaptive Modulation

This module implements frequency band structure correlation analysis and adaptive
amplitude modulation based on Phase-Amplitude Coupling (PAC) and Structural Band
Persistence (SBP) metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FreqDomainModLayer(nn.Module):
    """
    Frequency Domain Modulation Layer for single feature map.

    Performs adaptive frequency band modulation based on structure correlation metrics:
    - Phase-Amplitude Coupling (PAC): measures how amplitude perturbations affect phase
    - Structural Band Persistence (SBP): measures phase distribution stability under perturbation
    - Structural Relevance Index (SRI): weighted combination of PAC and SBP

    The learned band weights are initialized from SRI statistics and updated during training.
    """

    def __init__(self, in_channels, num_bands=8, alpha=0.5, delta=0.2, eps=1e-6,
                 init_from_stats=True, update_stats_interval=100):
        """
        Args:
            in_channels: Number of input channels C
            num_bands: Number of frequency bands B for radial decomposition
            alpha: Weight for combining PAC and SBP (SRI = alpha*PAC + (1-alpha)*SBP)
            delta: Amplitude perturbation strength for computing PAC/SBP
            eps: Numerical stability constant
            init_from_stats: Whether to initialize weights from SRI statistics
            update_stats_interval: Iteration interval for updating statistics
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_bands = num_bands
        self.alpha = alpha
        self.delta = delta
        self.eps = eps
        self.init_from_stats = init_from_stats
        self.update_stats_interval = update_stats_interval

        # Learnable frequency band weights W_l ∈ R^B
        # Initialized uniformly, later updated from SRI statistics
        self.freq_weights = nn.Parameter(torch.ones(num_bands) / num_bands)

        # Running statistics for PAC / SBP / SRI (buffers, not trained)
        self.register_buffer("running_pac", torch.zeros(num_bands))
        self.register_buffer("running_sbp", torch.zeros(num_bands))
        self.register_buffer("running_sri", torch.zeros(num_bands))
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("stats_initialized", torch.tensor(False, dtype=torch.bool))

        # Band masks will be built on first forward pass based on spatial dimensions
        self.register_buffer("band_mask_cache", torch.empty(0))
        self.register_buffer("cached_H", torch.tensor(0, dtype=torch.long))
        self.register_buffer("cached_W", torch.tensor(0, dtype=torch.long))

    def _build_band_masks(self, H, W, device):
        """
        Build radial frequency band masks Ω_b, dividing frequency domain into
        B concentric annular bands.

        Args:
            H, W: Spatial dimensions of frequency domain
            device: Target device

        Returns:
            band_masks: [B, H, W] binary masks for each band
        """
        # Check cache
        if (self.band_mask_cache.numel() != 0 and
            self.cached_H.item() == H and
            self.cached_W.item() == W):
            return self.band_mask_cache

        # Compute radius from center (DC component)
        cy, cx = H // 2, W // 2
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij"
        )
        rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_max = rr.max() + self.eps

        # Uniformly divide radial range [0, r_max] into B bands
        radius_bins = torch.linspace(0.0, float(r_max), self.num_bands + 1, device=device)
        band_masks = []
        for b in range(self.num_bands):
            r_min = radius_bins[b]
            r_max_b = radius_bins[b + 1] + self.eps
            mask = ((rr >= r_min) & (rr < r_max_b)).float()
            band_masks.append(mask.unsqueeze(0))  # [1, H, W]

        band_masks = torch.cat(band_masks, dim=0)  # [B, H, W]

        # Update cache
        self.band_mask_cache = band_masks
        self.cached_H.fill_(H)
        self.cached_W.fill_(W)

        return band_masks

    @torch.no_grad()
    def _estimate_band_stats(self, A, phase, band_masks):
        """
        Estimate PAC and SBP for each frequency band based on current batch.

        Args:
            A: [N, C, H, W] amplitude spectrum
            phase: [N, C, H, W] phase spectrum
            band_masks: [B, H, W] frequency band masks
        """
        N, C, H, W = A.shape
        device = A.device
        B = self.num_bands

        pac_vals = []
        sbp_vals = []

        for b in range(B):
            mask_b = band_masks[b:b+1]  # [1, H, W]

            # Perturb amplitude only in band b: A' = A * (1 + delta * mask_b)
            A_pert = A * (1 + self.delta * mask_b.unsqueeze(0))  # [N, C, H, W]

            # Reconstruct perturbed complex spectrum
            real_pert = A_pert * torch.cos(phase)
            imag_pert = A_pert * torch.sin(phase)
            F_pert = torch.complex(real_pert, imag_pert)

            # IFFT + FFT to propagate perturbation effects through spatial domain
            f_spatial = torch.fft.ifft2(F_pert, dim=(-2, -1))
            F_pert2 = torch.fft.fft2(f_spatial, dim=(-2, -1))
            phase_pert = torch.angle(F_pert2)  # [N, C, H, W]

            # Compute phase offset Δφ_{l,b} within band b
            diff = phase_pert - phase
            # Wrap to [-π, π]
            diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi

            # Mean absolute phase difference in band b
            mask_expanded = mask_b.unsqueeze(0)  # [1, 1, H, W]
            band_count = mask_expanded.sum() * C * N + self.eps
            delta_phi = (diff.abs() * mask_expanded).sum() / band_count
            pac_vals.append(delta_phi)

            # Estimate SBP via KL divergence of phase distributions
            num_bins = 32
            phase_band = phase * mask_expanded  # [N, C, H, W]
            phase_pert_band = phase_pert * mask_expanded

            # Extract valid phase values in band
            valid_mask = mask_expanded.expand_as(phase) > 0
            phase_vals = phase_band[valid_mask]
            phase_pert_vals = phase_pert_band[valid_mask]

            if phase_vals.numel() < num_bins:
                sbp_vals.append(torch.tensor(1.0, device=device))
            else:
                # Histogram-based distribution estimation
                hist_p = torch.histc(phase_vals, bins=num_bins,
                                     min=-math.pi, max=math.pi) + self.eps
                hist_q = torch.histc(phase_pert_vals, bins=num_bins,
                                     min=-math.pi, max=math.pi) + self.eps
                p = hist_p / hist_p.sum()
                q = hist_q / hist_q.sum()

                # KL divergence
                kl = (p * (p / q).log()).sum()
                sbp = torch.exp(-kl)
                sbp_vals.append(sbp)

        pac = torch.stack(pac_vals)  # [B]
        sbp = torch.stack(sbp_vals)  # [B]

        # Normalize PAC
        pac_norm = pac / (pac.max() + self.eps)

        # Compute SRI
        sri = self.alpha * pac_norm + (1 - self.alpha) * sbp

        # Update running statistics with exponential moving average
        if not self.stats_initialized.item():
            self.running_pac = pac_norm
            self.running_sbp = sbp
            self.running_sri = sri
            self.stats_initialized.fill_(True)
        else:
            momentum = 0.9
            self.running_pac = momentum * self.running_pac + (1 - momentum) * pac_norm
            self.running_sbp = momentum * self.running_sbp + (1 - momentum) * sbp
            self.running_sri = momentum * self.running_sri + (1 - momentum) * sri

        self.num_updates += 1

        # Initialize/update learnable weights from SRI
        if self.init_from_stats:
            w = self.running_sri / (self.running_sri.sum() + self.eps)
            # Scale to have mean=1 for stable training
            w = w * self.num_bands
            self.freq_weights.data = w

    def forward(self, x):
        """
        Apply frequency domain modulation to input features.

        Args:
            x: Tensor of shape [N, C, H, W]

        Returns:
            x_mod: Modulated features with same shape [N, C, H, W]
        """
        N, C, H, W = x.shape
        device = x.device

        # 2D FFT (shift to center DC component)
        X = torch.fft.fft2(x, dim=(-2, -1))
        X = torch.fft.fftshift(X, dim=(-2, -1))

        A = torch.abs(X)
        phase = torch.angle(X)

        # Build frequency band masks
        band_masks = self._build_band_masks(H, W, device)  # [B, H, W]

        # Update statistics during training
        if self.training and self.init_from_stats:
            if (self.num_updates.item() % self.update_stats_interval) == 0:
                self._estimate_band_stats(A, phase, band_masks)

        # Apply adaptive amplitude modulation based on learned band weights
        # A'(u,v) = A(u,v) * W_{l,b} for (u,v) ∈ Ω_b
        A_mod = torch.zeros_like(A)

        for b in range(self.num_bands):
            mask_b = band_masks[b:b+1].unsqueeze(0)  # [1, 1, H, W]
            w_b = self.freq_weights[b]
            A_mod = A_mod + A * mask_b * w_b

        # Reconstruct with modulated amplitude and original phase
        real = A_mod * torch.cos(phase)
        imag = A_mod * torch.sin(phase)
        X_mod = torch.complex(real, imag)

        # Inverse FFT shift and transform back to spatial domain
        X_mod = torch.fft.ifftshift(X_mod, dim=(-2, -1))
        x_mod = torch.fft.ifft2(X_mod, dim=(-2, -1))
        # Use torch.real() to avoid warning, input is real so output should be real
        x_mod = torch.real(x_mod)

        return x_mod


class FreqDomainModulation(nn.Module):
    """
    Multi-layer Frequency Domain Modulation module.

    Wraps multiple FDM layers for different encoder feature levels.
    """

    def __init__(self, channels_list, num_bands=8, alpha=0.5, delta=0.2, eps=1e-6,
                 init_from_stats=True, update_stats_interval=100):
        """
        Args:
            channels_list: List of channel counts for each modulation layer
            num_bands: Number of frequency bands
            alpha: PAC-SBP fusion weight
            delta: Perturbation strength
            eps: Numerical stability
            init_from_stats: Whether to use SRI for weight initialization
            update_stats_interval: Statistics update interval
        """
        super().__init__()

        self.fdm_layers = nn.ModuleList([
            FreqDomainModLayer(
                in_channels=c,
                num_bands=num_bands,
                alpha=alpha,
                delta=delta,
                eps=eps,
                init_from_stats=init_from_stats,
                update_stats_interval=update_stats_interval
            )
            for c in channels_list
        ])

    def forward(self, features_list):
        """
        Apply FDM to multiple feature maps.

        Args:
            features_list: List of tensors [N, C_i, H_i, W_i]

        Returns:
            modulated_features: List of modulated tensors with same shapes
        """
        return [fdm(f) for fdm, f in zip(self.fdm_layers, features_list)]

    def get_stats(self):
        """
        Retrieve current PAC/SBP/SRI statistics for all layers.

        Returns:
            Dictionary containing statistics for each layer
        """
        stats = {}
        for i, fdm in enumerate(self.fdm_layers):
            stats[f'layer_{i}'] = {
                'pac': fdm.running_pac.cpu().numpy(),
                'sbp': fdm.running_sbp.cpu().numpy(),
                'sri': fdm.running_sri.cpu().numpy(),
                'weights': fdm.freq_weights.data.cpu().numpy(),
                'num_updates': fdm.num_updates.item()
            }
        return stats
