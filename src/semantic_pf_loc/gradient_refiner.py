"""Differentiable pose refinement of top-k particles."""

import torch
import torch.nn.functional as F
import pypose as pp
from pytorch_msssim import ssim

from .batch_renderer import BatchRenderer


class GradientRefiner:
    """Refine particle poses via gradient descent through gsplat."""

    def __init__(
        self,
        renderer: BatchRenderer,
        num_iterations: int = 50,
        lr_init: float = 1e-2,
        lr_final: float = 1e-5,
        loss_type: str = "l1+ssim",
        ssim_weight: float = 0.2,
        blur_schedule: bool = True,
        blur_sigma_init: float = 10.0,
        blur_sigma_final: float = 0.1,
    ):
        self.renderer = renderer
        self.num_iterations = num_iterations
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.loss_type = loss_type
        self.ssim_weight = ssim_weight
        self.blur_schedule = blur_schedule
        self.blur_sigma_init = blur_sigma_init
        self.blur_sigma_final = blur_sigma_final

    def refine(
        self,
        poses: pp.LieTensor,
        query_image: torch.Tensor,
        K: torch.Tensor,
    ) -> pp.LieTensor:
        """Optimize K poses via gradient descent on rendering loss.

        Args:
            poses: SE3 [K, 7] initial particle poses (camera-to-world)
            query_image: [H, W, 3] target image
            K: [3, 3] camera intrinsics (already scaled for render resolution)
        Returns:
            SE3 [K, 7] refined poses
        """
        num_poses = poses.shape[0]
        device = poses.device

        # Perturbation in se(3) Lie algebra
        delta = torch.zeros(num_poses, 6, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr_init)

        # LR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_iterations, eta_min=self.lr_final
        )

        # Expand intrinsics
        Ks = K.unsqueeze(0).expand(num_poses, -1, -1).clone()

        # Resize target to match renderer resolution
        rh, rw = self.renderer.height, self.renderer.width
        if query_image.shape[0] != rh or query_image.shape[1] != rw:
            query_resized = F.interpolate(
                query_image.permute(2, 0, 1).unsqueeze(0),
                size=(rh, rw), mode="bilinear", align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        else:
            query_resized = query_image
        target = query_resized.unsqueeze(0).expand(num_poses, -1, -1, -1)  # [K, H, W, 3]

        # Detach base poses
        base_poses = pp.SE3(poses.tensor().detach().clone())

        for i in range(self.num_iterations):
            optimizer.zero_grad()

            # Compose perturbation: T_new = Exp(delta) @ T_base
            perturbed = pp.se3(delta).Exp() @ base_poses  # [K, 7]

            # Convert to world-to-camera viewmats
            viewmats = perturbed.Inv().matrix()  # [K, 4, 4]

            # Render
            rendered, _, _ = self.renderer.render_batch(viewmats, Ks)  # [K, H, W, 3]

            # Optional blur (coarse-to-fine)
            if self.blur_schedule:
                t = i / max(self.num_iterations - 1, 1)
                sigma = self.blur_sigma_init * (1 - t) + self.blur_sigma_final * t
                if sigma > 0.5:
                    ksize = int(6 * sigma) | 1  # ensure odd
                    ksize = max(ksize, 3)
                    rendered = self._gaussian_blur(rendered, ksize, sigma)
                    target_blurred = self._gaussian_blur(target, ksize, sigma)
                else:
                    target_blurred = target
            else:
                target_blurred = target

            # Compute loss
            loss = self._compute_loss(rendered, target_blurred)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Return refined poses
        with torch.no_grad():
            refined = pp.se3(delta.detach()).Exp() @ base_poses
        return refined

    def _compute_loss(
        self, rendered: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute rendering loss."""
        if self.loss_type == "l1":
            return F.l1_loss(rendered, target)
        elif self.loss_type == "ssim":
            r = rendered.permute(0, 3, 1, 2)
            t = target.permute(0, 3, 1, 2)
            return 1.0 - ssim(r, t, data_range=1.0, size_average=True)
        elif self.loss_type == "l1+ssim":
            l1 = F.l1_loss(rendered, target)
            r = rendered.permute(0, 3, 1, 2)
            t = target.permute(0, 3, 1, 2)
            ssim_loss = 1.0 - ssim(r, t, data_range=1.0, size_average=True)
            return (1.0 - self.ssim_weight) * l1 + self.ssim_weight * ssim_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    @staticmethod
    def _gaussian_blur(
        images: torch.Tensor, kernel_size: int, sigma: float
    ) -> torch.Tensor:
        """Apply Gaussian blur to [B, H, W, 3] images."""
        x = images.permute(0, 3, 1, 2)  # [B, 3, H, W]
        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        # Separable 2D convolution
        kernel_h = kernel_1d.view(1, 1, 1, -1).expand(3, -1, -1, -1)
        kernel_v = kernel_1d.view(1, 1, -1, 1).expand(3, -1, -1, -1)
        pad_h = kernel_size // 2
        pad_v = kernel_size // 2
        x = F.pad(x, [pad_h, pad_h, 0, 0], mode="reflect")
        x = F.conv2d(x, kernel_h, groups=3)
        x = F.pad(x, [0, 0, pad_v, pad_v], mode="reflect")
        x = F.conv2d(x, kernel_v, groups=3)
        return x.permute(0, 2, 3, 1)
