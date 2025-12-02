import torch
from typing import List, Sequence, Tuple, Optional, Union

try:
    from omegaconf import ListConfig
except Exception:  # pragma: no cover - optional dependency
    ListConfig = ()  # type: ignore


def build_radial_band_masks(height: int, width: int, num_bands: int) -> List[torch.Tensor]:
    """
    Build annular band masks by uniformly splitting the frequency radius.

    Args:
        height: Spatial height (H) of the image.
        width: Spatial width (W) of the image.
        num_bands: Number of frequency bands.

    Returns:
        List of boolean masks with shape [H, W] for each band.
    """
    cy, cx = height // 2, width // 2
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    dist = torch.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    r_max = dist.max()

    masks: List[torch.Tensor] = []
    for b in range(num_bands):
        r1 = r_max * b / num_bands
        r2 = r_max * (b + 1) / num_bands
        mask = (dist >= r1) & (dist < r2)
        masks.append(mask)
    return masks


def _match_stats(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Match mean/std of src to ref."""
    mean_ref, std_ref = ref.mean(), ref.std()
    mean_src, std_src = src.mean(), src.std()
    if std_src > 1e-6:
        src = (src - mean_src) / std_src * std_ref + mean_ref
    return src


def fourier_augment(
    x: torch.Tensor,
    mode: str,
    p_fourier: float = 0.5,
    sigma: float = 0.08,
    alpha_range: Tuple[float, float] = (0.7, 1.0),
    band_masks: Optional[Sequence[torch.Tensor]] = None,
    band_weights: Optional[Sequence[float]] = None,
    eps_max: float = 0.3,
) -> torch.Tensor:
    """
    Fourier-based augmentation with baseline/naive/FBD modes.

    Args:
        x: Input tensor [C, H, W] (float32), already normalized.
        mode: One of {"baseline", "naive", "fbd"}.
        p_fourier: Probability to apply Fourier perturbation.
        sigma: Global noise std for naive mode.
        alpha_range: Range for linear mixing coefficient.
        band_masks: List of [H, W] masks for FBD mode.
        band_weights: Importance weights per band for FBD mode.
        eps_max: Clamp range for multiplicative amplitude factor.
    """
    mode = mode.lower()
    if mode == "baseline" or torch.rand(1, device=x.device) > p_fourier:
        return x

    if mode not in {"naive", "fbd"}:
        raise ValueError(f"Unsupported Fourier mode {mode}")

    C, H, W = x.shape
    x_aug = x.clone()

    if mode == "fbd":
        if band_masks is None or band_weights is None:
            raise ValueError("band_masks and band_weights are required for FBD mode.")
        B = len(band_masks)
        if len(band_weights) != B:
            raise ValueError("band_weights length must match band_masks length.")

        w = torch.tensor(band_weights, dtype=torch.float32, device=x.device)
        w = w / (w.mean() + 1e-6)
        sigma_b = sigma * w

        masks = torch.stack([bm.to(x.device).to(torch.bool) for bm in band_masks], dim=0)
    else:
        sigma_b = None
        masks = None

    for c in range(C):
        xc = x[c]
        F = torch.fft.fft2(xc)
        A = torch.abs(F)
        phase = torch.angle(F)

        if mode == "naive":
            eps = torch.randn_like(A) * sigma
        else:  # fbd
            eps = torch.zeros_like(A)
            for b in range(len(band_masks)):
                noise_b = torch.randn_like(A) * sigma_b[b]
                eps = torch.where(masks[b], noise_b, eps)

        factor = 1.0 + eps
        factor = torch.clamp(factor, 1.0 - eps_max, 1.0 + eps_max)
        A_perturbed = torch.clamp(A * factor, min=0.0)

        F_perturbed = A_perturbed * torch.exp(1j * phase)
        x_fourier = torch.fft.ifft2(F_perturbed).real
        x_fourier = _match_stats(x_fourier, xc)

        alpha = torch.empty(1, device=x.device).uniform_(*alpha_range).item()
        x_aug[c] = (1 - alpha) * xc + alpha * x_fourier

    return x_aug


def _expand_to_band_params(value: Union[float, Sequence[float]], num_bands: int) -> List[float]:
    """Expand scalar or single-entry list to per-band list."""
    # Accept Python list/tuple, OmegaConf ListConfig, or scalar
    if isinstance(value, (list, tuple, ListConfig)):
        value_list = list(value)
        if len(value_list) == 1:
            return [float(value_list[0]) for _ in range(num_bands)]
        if len(value_list) != num_bands:
            raise ValueError(f"Expected {num_bands} values, got {len(value_list)}.")
        return [float(v) for v in value_list]
    return [float(value) for _ in range(num_bands)]


def core_guided_fourier_augment(
    x: torch.Tensor,
    core_mask: torch.Tensor,
    p_fourier: float = 0.5,
    sigma: float = 0.08,
    alpha_range: Tuple[float, float] = (0.7, 1.0),
    band_masks: Optional[Sequence[torch.Tensor]] = None,
    band_weights: Optional[Sequence[float]] = None,
    eps_max: float = 0.3,
    lambda_core: Union[float, Sequence[float]] = 0.05,
    lambda_noncore: Union[float, Sequence[float]] = 1.0,
) -> torch.Tensor:
    """
    Core-guided Fourier augmentation that preserves core structure while perturbing non-core regions.

    Args:
        x: Input tensor [B, C, H, W] (float32).
        core_mask: Core mask [B, 1, H, W] or [B, H, W] with values in [0, 1].
        p_fourier: Probability to apply Fourier perturbation per sample.
        sigma: Base noise std for amplitude perturbation.
        alpha_range: Range for linear mixing coefficient with the perturbed view.
        band_masks: Optional list of [H, W] masks for frequency bands.
        band_weights: Optional importance weights per band.
        eps_max: Clamp range for multiplicative amplitude factor.
        lambda_core: Multiplicative factor inside core (can be scalar or list per band).
        lambda_noncore: Multiplicative factor outside core (scalar or list per band).

    Returns:
        Augmented tensor with the same shape as x.
    """
    if x.ndimension() != 4:
        raise ValueError(f"core_guided_fourier_augment expects [B, C, H, W], got {x.shape}")

    device = x.device
    B, C, H, W = x.shape

    # Build frequency band resources
    if band_masks is None:
        nb = len(band_weights) if band_weights is not None else 4
        band_masks = build_radial_band_masks(H, W, nb)
    num_bands = len(band_masks)
    if band_weights is None:
        band_weights = [1.0 for _ in range(num_bands)]
    if len(band_weights) != num_bands:
        raise ValueError("band_masks and band_weights must have the same length.")

    lambda_core_list = _expand_to_band_params(lambda_core, num_bands)
    lambda_non_list = _expand_to_band_params(lambda_noncore, num_bands)

    masks = torch.stack([bm.to(device).to(torch.bool) for bm in band_masks], dim=0)  # [num_bands, H, W]
    w = torch.tensor(band_weights, dtype=torch.float32, device=device)
    w = w / (w.mean() + 1e-6)
    sigma_b = sigma * w

    core_mask = core_mask.to(device)
    if core_mask.ndimension() == 3:
        core_mask = core_mask.unsqueeze(1)
    if core_mask.shape[1] != 1:
        core_mask = core_mask[:, :1]
    core_mask = core_mask.clamp(0.0, 1.0)

    outputs: List[torch.Tensor] = []
    for b_idx in range(B):
        if torch.rand(1, device=device) > p_fourier:
            outputs.append(x[b_idx])
            continue

        cm = core_mask[b_idx, 0]  # [H, W]
        sample = x[b_idx]
        aug_channels = []

        for c_idx in range(C):
            xc = sample[c_idx]
            F = torch.fft.fft2(xc)
            A = torch.abs(F)
            phase = torch.angle(F)

            eps = torch.zeros_like(A)
            for band in range(num_bands):
                spatial_weight = lambda_core_list[band] * cm + lambda_non_list[band] * (1 - cm)
                noise_b = torch.randn_like(A) * sigma_b[band] * spatial_weight
                eps = torch.where(masks[band], noise_b, eps)

            factor = torch.clamp(1.0 + eps, 1.0 - eps_max, 1.0 + eps_max)
            A_perturbed = torch.clamp(A * factor, min=0.0)

            F_perturbed = A_perturbed * torch.exp(1j * phase)
            x_fourier = torch.fft.ifft2(F_perturbed).real
            x_fourier = _match_stats(x_fourier, xc)

            alpha = torch.empty(1, device=device).uniform_(*alpha_range).item()
            aug_channels.append((1 - alpha) * xc + alpha * x_fourier)

        outputs.append(torch.stack(aug_channels, dim=0))

    return torch.stack(outputs, dim=0)
