"""SE(3) motion model for particle filter."""

import torch
import pypose as pp


class MotionModel:
    """Constant-velocity motion model with SE(3) Gaussian noise."""

    def __init__(
        self,
        translation_std: float = 0.02,
        rotation_std: float = 0.01,
        device: str = "cuda",
    ):
        self.trans_std = translation_std
        self.rot_std = rotation_std
        self.device = device

    def predict(
        self,
        particles: pp.LieTensor,
        velocity: pp.LieTensor | None = None,
    ) -> pp.LieTensor:
        """Propagate particles with noise (and optional velocity).

        Args:
            particles: SE3 [N, 7]
            velocity: se3 [6] or [N, 6] velocity in Lie algebra (optional)
        Returns:
            SE3 [N, 7] propagated particles
        """
        N = particles.shape[0]

        # Apply velocity if provided
        if velocity is not None:
            if velocity.dim() == 1:
                velocity = velocity.unsqueeze(0).expand(N, -1)
            particles = pp.se3(velocity).Exp() @ particles

        # Sample noise in se(3) Lie algebra
        noise = torch.zeros(N, 6, device=self.device)
        noise[:, :3] = torch.randn(N, 3, device=self.device) * self.trans_std
        noise[:, 3:] = torch.randn(N, 3, device=self.device) * self.rot_std

        # Left-perturbation: T_new = Exp(noise) @ T_old
        particles = pp.se3(noise).Exp() @ particles

        return particles
