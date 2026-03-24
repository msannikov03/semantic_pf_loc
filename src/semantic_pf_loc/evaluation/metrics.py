"""Evaluation metrics for camera pose localization."""

import torch
import math
from typing import Optional


def translation_error(estimated: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """Per-frame translation error in meters.
    Args: estimated [T,4,4], ground_truth [T,4,4] (camera-to-world)
    Returns: [T]
    """
    t_est = estimated[:, :3, 3]
    t_gt = ground_truth[:, :3, 3]
    return (t_est - t_gt).norm(dim=-1)


def rotation_error(estimated: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """Per-frame rotation error in degrees (geodesic distance on SO(3))."""
    R_est = estimated[:, :3, :3]
    R_gt = ground_truth[:, :3, :3]
    R_diff = torch.bmm(R_est.transpose(1, 2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cos_angle) * (180.0 / math.pi)


def absolute_trajectory_error(estimated: torch.Tensor, ground_truth: torch.Tensor) -> dict:
    """ATE statistics in meters."""
    errors = translation_error(estimated, ground_truth)
    return {
        "rmse": (errors**2).mean().sqrt().item(),
        "mean": errors.mean().item(),
        "median": errors.median().item(),
        "std": errors.std().item(),
        "min": errors.min().item(),
        "max": errors.max().item(),
    }


def absolute_rotation_error(estimated: torch.Tensor, ground_truth: torch.Tensor) -> dict:
    """Rotation error statistics in degrees."""
    errors = rotation_error(estimated, ground_truth)
    return {
        "rmse": (errors**2).mean().sqrt().item(),
        "mean": errors.mean().item(),
        "median": errors.median().item(),
        "std": errors.std().item(),
        "min": errors.min().item(),
        "max": errors.max().item(),
    }


def success_rate(
    estimated: torch.Tensor,
    ground_truth: torch.Tensor,
    trans_threshold: float = 0.05,
    rot_threshold: float = 2.0,
) -> float:
    """Fraction of frames within (trans_threshold m, rot_threshold deg)."""
    t_err = translation_error(estimated, ground_truth)
    r_err = rotation_error(estimated, ground_truth)
    success = (t_err < trans_threshold) & (r_err < rot_threshold)
    return success.float().mean().item()


def convergence_time(trans_errors: torch.Tensor, threshold: float = 0.1) -> int:
    """Frames until error stays below threshold permanently."""
    T = len(trans_errors)
    exceeds = (trans_errors >= threshold).nonzero(as_tuple=True)[0]
    if len(exceeds) == 0:
        return 0
    last_exceed = exceeds[-1].item()
    if last_exceed >= T - 1:
        return T
    return last_exceed + 1


def compute_all_metrics(
    estimated: torch.Tensor,
    ground_truth: torch.Tensor,
    per_step_times_ms: Optional[torch.Tensor] = None,
    trans_threshold: float = 0.05,
    rot_threshold: float = 2.0,
    convergence_threshold: float = 0.1,
) -> dict:
    """Compute all metrics in one call."""
    t_err = translation_error(estimated, ground_truth)
    r_err = rotation_error(estimated, ground_truth)

    result = {
        "ate": absolute_trajectory_error(estimated, ground_truth),
        "are": absolute_rotation_error(estimated, ground_truth),
        "success_rate": success_rate(estimated, ground_truth, trans_threshold, rot_threshold),
        "convergence_frame": convergence_time(t_err, convergence_threshold),
        "trans_errors": t_err,
        "rot_errors": r_err,
    }

    if per_step_times_ms is not None:
        result["runtime_mean_ms"] = per_step_times_ms.mean().item()
        result["runtime_std_ms"] = per_step_times_ms.std().item()

    return result
