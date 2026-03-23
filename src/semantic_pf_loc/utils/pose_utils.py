"""SE(3) pose utilities for particle filter localization."""

import torch
import pypose as pp
import math


def matrix_to_se3(matrix: torch.Tensor) -> pp.LieTensor:
    """Convert 4x4 homogeneous matrix to PyPose SE3.
    Args: matrix [4,4] or [B,4,4] camera-to-world
    Returns: pp.SE3 [7] or [B,7] (tx,ty,tz,qx,qy,qz,qw)
    """
    if matrix.dim() == 2:
        return pp.mat2SE3(matrix.unsqueeze(0)).squeeze(0)
    return pp.mat2SE3(matrix)


def se3_to_matrix(pose: pp.LieTensor) -> torch.Tensor:
    """Convert PyPose SE3 to 4x4 homogeneous matrix."""
    return pose.matrix()


def se3_to_viewmat(pose: pp.LieTensor) -> torch.Tensor:
    """Convert camera-to-world SE3 to world-to-camera viewmat for gsplat."""
    return pose.Inv().matrix()


def weighted_se3_mean(
    poses: pp.LieTensor,
    weights: torch.Tensor,
    max_iterations: int = 10,
    tol: float = 1e-6,
) -> pp.LieTensor:
    """Weighted Frechet mean on SE(3).

    Iterative: init from best particle, then refine in tangent space.
    Falls back to best particle when weights are near-uniform (unconverged PF).
    """
    best_idx = weights.argmax()
    mean = poses[best_idx : best_idx + 1]  # [1, 7]

    # If weights are near-uniform, just return best particle
    # (Frechet mean diverges with scattered particles and flat weights)
    n_eff = 1.0 / (weights**2).sum()
    if n_eff > 0.8 * weights.shape[0]:
        return mean.squeeze(0)

    for _ in range(max_iterations):
        rel = mean.Inv() @ poses  # [N, 7]
        deltas = rel.Log()  # [N, 6]

        # Clamp large tangent vectors to prevent divergence
        delta_norms = deltas.tensor().norm(dim=-1, keepdim=True)
        scale = torch.clamp(delta_norms, max=10.0) / delta_norms.clamp(min=1e-8)
        clamped_deltas = deltas.tensor() * scale

        weighted_delta = (weights.unsqueeze(-1) * clamped_deltas).sum(
            dim=0, keepdim=True
        )  # [1, 6]

        if weighted_delta.norm() < tol:
            break

        # Clamp the update itself
        if weighted_delta.norm() > 1.0:
            weighted_delta = weighted_delta / weighted_delta.norm()

        mean = pp.se3(weighted_delta).Exp() @ mean

    return mean.squeeze(0)


def uniform_se3(
    n: int,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    device: str = "cuda",
) -> pp.LieTensor:
    """Sample N uniform SE(3) poses within spatial bounds.
    Translation: uniform in bounds. Rotation: uniform SO(3).
    """
    trans = (
        torch.rand(n, 3, device=device) * (bounds_max - bounds_min).to(device)
        + bounds_min.to(device)
    )

    # Uniform SO(3) via normalized Gaussian quaternion
    quat = torch.randn(n, 4, device=device)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    # Canonical: ensure qw > 0 (last element in PyPose convention)
    quat[quat[:, -1] < 0] *= -1

    se3_data = torch.cat([trans, quat], dim=-1)  # [N, 7]
    return pp.SE3(se3_data)


def pose_error(
    estimated: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[float, float]:
    """Translation (m) and rotation (deg) error between two 4x4 poses."""
    t_err = (estimated[:3, 3] - ground_truth[:3, 3]).norm().item()

    R_est = estimated[:3, :3]
    R_gt = ground_truth[:3, :3]
    R_diff = R_est.T @ R_gt
    trace = R_diff.diagonal().sum()
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    r_err = math.degrees(torch.acos(cos_angle).item())

    return t_err, r_err


def interpolate_poses(
    poses: torch.Tensor,
    timestamps: torch.Tensor,
    query_timestamps: torch.Tensor,
) -> torch.Tensor:
    """Nearest-neighbor pose interpolation for query timestamps."""
    diffs = torch.abs(query_timestamps.unsqueeze(1) - timestamps.unsqueeze(0))  # [Q, M]
    indices = diffs.argmin(dim=1)
    return poses[indices]


def scale_intrinsics(
    K: torch.Tensor,
    original_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Scale camera intrinsics for different resolution.
    Args: K [3,3] or [B,3,3], sizes as (width, height)
    """
    sx = target_size[0] / original_size[0]
    sy = target_size[1] / original_size[1]

    K_scaled = K.clone()
    if K.dim() == 2:
        K_scaled[0, 0] *= sx
        K_scaled[1, 1] *= sy
        K_scaled[0, 2] *= sx
        K_scaled[1, 2] *= sy
    else:
        K_scaled[:, 0, 0] *= sx
        K_scaled[:, 1, 1] *= sy
        K_scaled[:, 0, 2] *= sx
        K_scaled[:, 1, 2] *= sy

    return K_scaled
