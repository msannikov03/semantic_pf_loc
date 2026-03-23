"""Particle filter for 6-DoF localization in 3DGS maps.

Follows standard MCL (Monte Carlo Localization) as in NuRF/Loc-NeRF:
  1. Resample from previous weights
  2. Predict (motion model with noise)
  3. Update weights from current observation ONLY
  4. Estimate pose
"""

import torch
import pypose as pp
import time

from .gaussian_map import GaussianMap
from .batch_renderer import BatchRenderer
from .observation.base import ObservationModel
from .motion_model import MotionModel
from .gradient_refiner import GradientRefiner
from .resampling import (
    systematic_resample,
    effective_sample_size,
    normalize_log_weights,
)
from .utils.pose_utils import (
    weighted_se3_mean,
    uniform_se3,
    scale_intrinsics,
    se3_to_viewmat,
)


class ParticleFilter:
    """6-DoF particle filter for localization in 3DGS maps."""

    def __init__(
        self,
        gaussian_map: GaussianMap,
        renderer: BatchRenderer,
        observation_model: ObservationModel,
        motion_model: MotionModel,
        num_particles: int = 400,
        gradient_refiner: GradientRefiner | None = None,
        top_k_refine: int = 10,
        render_width: int = 80,
        render_height: int = 60,
        render_width_hires: int = 320,
        render_height_hires: int = 240,
        convergence_threshold: float = 0.1,
        roughening_trans: float = 0.005,
        roughening_rot: float = 0.002,
        device: str = "cuda",
    ):
        self.gmap = gaussian_map
        self.renderer = renderer
        self.obs_model = observation_model
        self.motion = motion_model
        self.N = num_particles
        self.refiner = gradient_refiner
        self.top_k = top_k_refine
        self.render_w = render_width
        self.render_h = render_height
        self.render_w_hi = render_width_hires
        self.render_h_hi = render_height_hires
        self.conv_threshold = convergence_threshold
        self.roughening_trans = roughening_trans
        self.roughening_rot = roughening_rot
        self.device = device

        # State
        self.particles: pp.LieTensor | None = None
        self.weights: torch.Tensor | None = None
        self._converged = False
        self._prev_estimate: pp.LieTensor | None = None
        self._step_count = 0

    def initialize_global(
        self,
        bounds_min: torch.Tensor,
        bounds_max: torch.Tensor,
    ) -> None:
        """Uniform initialization over scene bounds."""
        self.particles = uniform_se3(self.N, bounds_min, bounds_max, self.device)
        self.weights = torch.ones(self.N, device=self.device) / self.N
        self._converged = False
        self._prev_estimate = None
        self._step_count = 0

    def initialize_around_pose(
        self,
        pose: pp.LieTensor,
        trans_spread: float = 0.5,
        rot_spread: float = 0.3,
    ) -> None:
        """Initialize particles around a known pose."""
        noise = torch.zeros(self.N, 6, device=self.device)
        noise[:, :3] = torch.randn(self.N, 3, device=self.device) * trans_spread
        noise[:, 3:] = torch.randn(self.N, 3, device=self.device) * rot_spread

        base = pose.unsqueeze(0).expand(self.N, -1)
        self.particles = pp.se3(noise).Exp() @ pp.SE3(base)
        self.weights = torch.ones(self.N, device=self.device) / self.N
        self._converged = False
        self._prev_estimate = None
        self._step_count = 0

    @torch.no_grad()
    def step(
        self,
        observation: dict,
        K: torch.Tensor,
    ) -> tuple[pp.LieTensor, dict]:
        """One MCL cycle: resample -> predict -> update -> estimate.

        Args:
            observation: {"image": [H,W,3]} or {"text": str}
            K: [3,3] native camera intrinsics
        Returns:
            (estimated_pose SE3 [7], info_dict)
        """
        t_start = time.time()
        info = {}

        # 1. RESAMPLE (standard MCL: resample every step)
        if self._step_count > 0:
            indices = systematic_resample(self.weights, self.N)
            self.particles = pp.SE3(self.particles.tensor()[indices])

            # Roughening: add small noise after resampling to maintain diversity
            rough_noise = torch.zeros(self.N, 6, device=self.device)
            rough_noise[:, :3] = torch.randn(self.N, 3, device=self.device) * self.roughening_trans
            rough_noise[:, 3:] = torch.randn(self.N, 3, device=self.device) * self.roughening_rot
            self.particles = pp.se3(rough_noise).Exp() @ self.particles

        # 2. PREDICT (motion model)
        self.particles = self.motion.predict(self.particles)

        # 3. Choose render resolution
        if self._converged:
            rw, rh = self.render_w_hi, self.render_h_hi
        else:
            rw, rh = self.render_w, self.render_h
        self.renderer.update_resolution(rw, rh)

        # Scale intrinsics
        native_size = (K[0, 2].item() * 2, K[1, 2].item() * 2)
        K_scaled = scale_intrinsics(K, (int(native_size[0]), int(native_size[1])), (rw, rh))
        Ks = K_scaled.unsqueeze(0).expand(self.N, -1, -1).to(self.device)

        # 4. RENDER all particles
        viewmats = se3_to_viewmat(self.particles)  # [N, 4, 4]
        rendered, _, _ = self.renderer.render_batch(viewmats, Ks)

        # Resize observation to match render resolution
        obs_for_model = self._prepare_observation(observation, rw, rh)

        # 5. UPDATE weights (current observation only — standard MCL)
        log_w = self.obs_model.compute_log_weights(rendered, obs_for_model)
        self.weights = normalize_log_weights(log_w).exp()  # normalize to probabilities

        # 6. GRADIENT REFINEMENT (optional, on converged filter)
        if self.refiner is not None and self._converged and "image" in observation:
            top_k_idx = torch.topk(self.weights, min(self.top_k, self.N)).indices
            top_k_poses = pp.SE3(self.particles.tensor()[top_k_idx])

            with torch.enable_grad():
                refined = self.refiner.refine(
                    top_k_poses, obs_for_model["image"], K_scaled
                )

            self.particles.tensor()[top_k_idx] = refined.tensor()

            # Re-weight refined particles
            refined_viewmats = se3_to_viewmat(refined)
            refined_Ks = Ks[:len(top_k_idx)]
            refined_rendered, _, _ = self.renderer.render_batch(
                refined_viewmats, refined_Ks
            )
            refined_log_w = self.obs_model.compute_log_weights(
                refined_rendered, obs_for_model
            )
            log_w_updated = log_w.clone()
            log_w_updated[top_k_idx] = refined_log_w
            self.weights = normalize_log_weights(log_w_updated).exp()

        # 7. ESTIMATE pose (weighted mean, or best particle if unconverged)
        n_eff = effective_sample_size(self.weights)
        info["n_eff"] = n_eff

        estimated = weighted_se3_mean(self.particles, self.weights)

        # 8. Track convergence
        trans_var = self._translation_variance()
        self._converged = trans_var < self.conv_threshold
        info["trans_variance"] = trans_var
        info["converged"] = self._converged

        # 9. Store for velocity (unused in standard MCL but available)
        self._prev_estimate = pp.SE3(estimated.tensor().detach().clone())
        self._step_count += 1

        info["step_time_ms"] = (time.time() - t_start) * 1000
        info["render_resolution"] = (rw, rh)

        return estimated, info

    def _translation_variance(self) -> float:
        """Weighted variance of particle translations."""
        trans = self.particles.tensor()[:, :3]  # [N, 3]
        mean_trans = (self.weights.unsqueeze(-1) * trans).sum(dim=0)
        var = (self.weights.unsqueeze(-1) * (trans - mean_trans) ** 2).sum().item()
        return var

    def _prepare_observation(
        self, observation: dict, rw: int, rh: int
    ) -> dict:
        """Resize observation image to match render resolution."""
        if "image" not in observation:
            return observation

        img = observation["image"]  # [H, W, 3]
        if img.shape[0] != rh or img.shape[1] != rw:
            img = img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            img = torch.nn.functional.interpolate(
                img, size=(rh, rw), mode="bilinear", align_corners=False
            )
            img = img.squeeze(0).permute(1, 2, 0)  # [rh, rw, 3]
        return {**observation, "image": img.to(self.device)}
