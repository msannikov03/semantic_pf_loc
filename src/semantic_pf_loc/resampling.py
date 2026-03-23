"""Resampling strategies for particle filter."""

import torch


def systematic_resample(
    weights: torch.Tensor,
    num_samples: int | None = None,
) -> torch.Tensor:
    """Low-variance systematic resampling.

    Args:
        weights: [N] normalized weights (sum to 1)
        num_samples: number of samples (default: N)
    Returns:
        [num_samples] indices into particle array
    """
    N = weights.shape[0]
    if num_samples is None:
        num_samples = N

    cumsum = torch.cumsum(weights, dim=0)
    u0 = torch.rand(1, device=weights.device) / num_samples
    u = u0 + torch.arange(num_samples, device=weights.device, dtype=weights.dtype) / num_samples
    indices = torch.searchsorted(cumsum, u)
    return indices.clamp(max=N - 1)


def multinomial_resample(
    weights: torch.Tensor,
    num_samples: int | None = None,
) -> torch.Tensor:
    """Standard multinomial resampling."""
    N = weights.shape[0]
    if num_samples is None:
        num_samples = N
    return torch.multinomial(weights, num_samples, replacement=True)


def effective_sample_size(weights: torch.Tensor) -> float:
    """N_eff = 1 / sum(w_i^2). Range: [1, N]."""
    return 1.0 / (weights**2).sum().item()


def effective_sample_size_log(log_weights: torch.Tensor) -> float:
    """N_eff from log-weights (numerically stable)."""
    return torch.exp(-torch.logsumexp(2 * log_weights, dim=0)).item()


def normalize_log_weights(log_weights: torch.Tensor) -> torch.Tensor:
    """Normalize log-weights so they sum to 1 in probability space."""
    return log_weights - torch.logsumexp(log_weights, dim=0)
