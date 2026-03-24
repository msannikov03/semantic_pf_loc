"""Train 3D Gaussian Splatting maps for localization."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import math
import torch
import torch.nn.functional as F
from torch import optim
from pathlib import Path
from tqdm import tqdm
from pytorch_msssim import ssim
from omegaconf import OmegaConf

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

from semantic_pf_loc.datasets.tum import TUMDataset
from semantic_pf_loc.datasets.replica import ReplicaDataset


def initialize_from_depth(dataset, num_points=200000, num_frames=50,
                          max_depth=10.0, pixel_stride=4):
    """Back-project depth maps to 3D point cloud and subsample.

    Args:
        dataset: Dataset object with __getitem__ returning image, depth, pose, K.
        num_points: Target number of Gaussians to initialize.
        num_frames: How many frames to sample for back-projection.
        max_depth: Maximum valid depth in meters.
        pixel_stride: Skip every N pixels to reduce memory during collection.

    Returns:
        means: [num_points, 3] float32 world-frame positions.
        colors: [num_points, 3] float32 RGB in [0, 1].
        scales: [num_points, 3] float32 log-scale initial values.
    """
    all_points = []
    all_colors = []
    n_dataset = len(dataset)
    stride = max(1, n_dataset // num_frames)

    print(f"Depth init: sampling {min(num_frames, n_dataset)} of {n_dataset} "
          f"frames (stride={stride}, pixel_stride={pixel_stride})")

    for i in tqdm(range(0, n_dataset, stride), desc="Back-projecting depth"):
        sample = dataset[i]
        depth = sample.get("depth", None)
        if depth is None:
            continue

        image = sample["image"]     # [H, W, 3] float32
        pose = sample["pose"]       # [4, 4] float64 c2w
        K_frame = sample["K"]       # [3, 3] float64

        H, W = depth.shape
        fx = K_frame[0, 0].float()
        fy = K_frame[1, 1].float()
        cx = K_frame[0, 2].float()
        cy = K_frame[1, 2].float()

        # Create pixel grid with stride for efficiency
        u_coords = torch.arange(0, W, pixel_stride, dtype=torch.float32)
        v_coords = torch.arange(0, H, pixel_stride, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")
        # u_grid, v_grid are [H_sub, W_sub]

        # Sample depth and image at strided locations
        v_idx = torch.arange(0, H, pixel_stride)
        u_idx = torch.arange(0, W, pixel_stride)
        depth_sub = depth[v_idx][:, u_idx]       # [H_sub, W_sub]
        image_sub = image[v_idx][:, u_idx]       # [H_sub, W_sub, 3]

        # Valid depth mask
        valid = (depth_sub > 0.01) & (depth_sub < max_depth)
        if valid.sum() == 0:
            continue

        d = depth_sub[valid]
        u_valid = u_grid[valid]
        v_valid = v_grid[valid]

        # Back-project to camera frame
        x = (u_valid - cx) * d / fx
        y = (v_valid - cy) * d / fy
        z = d

        # Points in camera frame [M, 4]
        ones = torch.ones_like(z)
        pts_cam = torch.stack([x, y, z, ones], dim=-1)

        # Transform to world frame
        pose_f32 = pose.float()
        pts_world = (pose_f32 @ pts_cam.T).T[:, :3]  # [M, 3]

        # Get colors
        colors = image_sub[valid]  # [M, 3]

        all_points.append(pts_world)
        all_colors.append(colors)

    if len(all_points) == 0:
        print("WARNING: No valid depth maps found. Falling back to random init.")
        return None, None, None

    all_points = torch.cat(all_points, dim=0)
    all_colors = torch.cat(all_colors, dim=0)
    print(f"Collected {len(all_points)} points from depth back-projection")

    # Subsample to num_points via random permutation
    if len(all_points) > num_points:
        indices = torch.randperm(len(all_points))[:num_points]
        all_points = all_points[indices]
        all_colors = all_colors[indices]
    elif len(all_points) < num_points:
        print(f"Only {len(all_points)} points available (target: {num_points}). "
              f"Using all points.")

    # Estimate initial scales from local point density
    # Use a random subset for kNN distance estimation (full pairwise is too expensive)
    n_pts = len(all_points)
    sample_size = min(n_pts, 50000)
    sample_idx = torch.randperm(n_pts)[:sample_size]
    sample_pts = all_points[sample_idx]

    # For each sampled point, find distance to k-th nearest neighbor
    # Use chunked computation to avoid OOM
    k = 4
    avg_dists = []
    chunk_size = 5000
    for c_start in range(0, sample_size, chunk_size):
        c_end = min(c_start + chunk_size, sample_size)
        chunk = sample_pts[c_start:c_end]  # [chunk, 3]
        dists = torch.cdist(chunk, sample_pts)  # [chunk, sample_size]
        # Get k-th nearest (excluding self which is 0)
        topk_dists, _ = dists.topk(k + 1, largest=False)  # +1 for self
        avg_dist = topk_dists[:, 1:].mean(dim=1)  # mean of k nearest
        avg_dists.append(avg_dist)

    avg_dists = torch.cat(avg_dists)
    median_dist = avg_dists.median().item()
    # Initial log-scale: make Gaussians roughly cover the spacing between points
    init_log_scale = math.log(max(median_dist * 0.5, 1e-6))
    print(f"Median neighbor distance: {median_dist:.5f}m, "
          f"init log-scale: {init_log_scale:.2f}")

    scales = torch.full((n_pts, 3), init_log_scale, dtype=torch.float32)

    return all_points, all_colors, scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to scene config YAML")
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory")
    parser.add_argument("--init_gaussians", type=int, default=None)
    parser.add_argument("--depth_weight", type=float, default=0.0,
                        help="Weight for depth supervision loss (0=disabled, 0.5=recommended)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max_steps from config")
    parser.add_argument("--init_from_depth", action="store_true",
                        help="Initialize Gaussians from depth back-projections instead of random")
    parser.add_argument("--init_num_frames", type=int, default=50,
                        help="Number of frames to use for depth back-projection (default: 50)")
    args = parser.parse_args()

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
    )

    device = "cuda"
    tc = cfg.train_gs

    # CLI overrides
    depth_weight = args.depth_weight
    if args.max_steps is not None:
        tc.max_steps = args.max_steps

    if cfg.scene.type == "tum":
        dataset = TUMDataset(cfg.scene.data_dir)
    else:
        dataset = ReplicaDataset(cfg.scene.data_dir)

    train_indices = dataset.get_train_indices(stride=1)
    print(f"Training on {len(train_indices)} frames from {cfg.scene.name}")

    K = dataset.get_intrinsics().float().to(device)
    W, H = dataset.image_size
    W_train = W // tc.data_factor
    H_train = H // tc.data_factor

    K_train = K.clone()
    K_train[0, 0] /= tc.data_factor
    K_train[1, 1] /= tc.data_factor
    K_train[0, 2] /= tc.data_factor
    K_train[1, 2] /= tc.data_factor

    bounds_min, bounds_max = dataset.get_bounds()

    N = args.init_gaussians or tc.init_num_gaussians
    sh_degree = tc.sh_degree
    num_sh = (sh_degree + 1) ** 2

    if args.init_from_depth:
        print("Initializing Gaussians from depth back-projections...")
        init_pts, init_colors, init_scales = initialize_from_depth(
            dataset, num_points=N, num_frames=args.init_num_frames,
        )

        if init_pts is not None:
            N = len(init_pts)
            print(f"Depth init: {N} Gaussians placed on surfaces")

            means = torch.nn.Parameter(init_pts.to(device))
            scales = torch.nn.Parameter(init_scales.to(device))
            quats = torch.nn.Parameter(torch.randn(N, 4, device=device))
            # Start with higher opacity since points are on real surfaces
            opacities = torch.nn.Parameter(torch.full((N,), 2.0, device=device))

            # Initialize SH DC from observed RGB colors
            # SH DC coefficient = (color - 0.5) / C0 where C0 = 0.28209479
            C0 = 0.28209479177387814
            sh0_init = ((init_colors.to(device) - 0.5) / C0).unsqueeze(1)  # [N, 1, 3]
            sh0 = torch.nn.Parameter(sh0_init)
            shN = torch.nn.Parameter(torch.zeros(N, num_sh - 1, 3, device=device))
        else:
            # Fallback to random
            means = torch.nn.Parameter(
                torch.rand(N, 3, device=device) * (bounds_max - bounds_min).to(device)
                + bounds_min.to(device)
            )
            scales = torch.nn.Parameter(torch.full((N, 3), -5.0, device=device))
            quats = torch.nn.Parameter(torch.randn(N, 4, device=device))
            opacities = torch.nn.Parameter(torch.zeros(N, device=device))
            sh0 = torch.nn.Parameter(torch.zeros(N, 1, 3, device=device))
            shN = torch.nn.Parameter(torch.zeros(N, num_sh - 1, 3, device=device))
    else:
        means = torch.nn.Parameter(
            torch.rand(N, 3, device=device) * (bounds_max - bounds_min).to(device)
            + bounds_min.to(device)
        )
        scales = torch.nn.Parameter(torch.full((N, 3), -5.0, device=device))
        quats = torch.nn.Parameter(torch.randn(N, 4, device=device))
        opacities = torch.nn.Parameter(torch.zeros(N, device=device))

        # IMPORTANT: separate SH DC and SH rest as independent parameters
        # gsplat's DefaultStrategy modifies params in-place during densification
        sh0 = torch.nn.Parameter(torch.zeros(N, 1, 3, device=device))
        shN = torch.nn.Parameter(torch.zeros(N, num_sh - 1, 3, device=device))

    # Separate optimizers (required by DefaultStrategy)
    opt_means = optim.Adam([means], lr=tc.lr_means)
    opt_scales = optim.Adam([scales], lr=tc.lr_scales)
    opt_quats = optim.Adam([quats], lr=tc.lr_quats)
    opt_opacities = optim.Adam([opacities], lr=tc.lr_opacities)
    opt_sh0 = optim.Adam([sh0], lr=tc.lr_sh)
    opt_shN = optim.Adam([shN], lr=tc.lr_sh)

    params_dict = {
        "means": means, "scales": scales, "quats": quats,
        "opacities": opacities, "sh0": sh0, "shN": shN,
    }
    opt_dict = {
        "means": opt_means, "scales": opt_scales, "quats": opt_quats,
        "opacities": opt_opacities, "sh0": opt_sh0, "shN": opt_shN,
    }

    use_densify = (N <= 50000)  # only densify if starting small
    if use_densify:
        strategy = DefaultStrategy(absgrad=True, verbose=True)
        strategy_state = strategy.initialize_state()
    else:
        print(f"Skipping densification (N={N} >= 50000)")

    pbar = tqdm(range(tc.max_steps), desc=f"Training {cfg.scene.name}")
    for step in pbar:
        idx = train_indices[torch.randint(len(train_indices), (1,)).item()]
        sample = dataset[idx]
        gt_image = sample["image"].float().to(device)
        pose = sample["pose"].float().to(device)

        # Load and downsample GT depth if depth supervision is enabled
        gt_depth = None
        if depth_weight > 0:
            gt_depth = sample.get("depth", None)
            if gt_depth is not None:
                gt_depth = gt_depth.float().to(device)
                if tc.data_factor > 1:
                    # Nearest-neighbor downsampling for depth (no blending at edges)
                    gt_depth = F.interpolate(
                        gt_depth.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                        size=(H_train, W_train),
                        mode="nearest",
                    ).squeeze(0).squeeze(0)  # [H_train, W_train]

        if tc.data_factor > 1:
            gt_image = F.interpolate(
                gt_image.permute(2, 0, 1).unsqueeze(0),
                size=(H_train, W_train),
                mode="bilinear", align_corners=False,
            ).squeeze(0).permute(1, 2, 0)

        viewmat = torch.linalg.inv(pose).unsqueeze(0)
        K_batch = K_train.unsqueeze(0)

        # Use params_dict refs (strategy may have replaced Parameters)
        _means = params_dict["means"]
        _quats = params_dict["quats"]
        _scales = params_dict["scales"]
        _opacities = params_dict["opacities"]
        sh_coeffs = torch.cat([params_dict["sh0"], params_dict["shN"]], dim=1)

        renders, alphas, meta = rasterization(
            means=_means,
            quats=_quats,
            scales=torch.exp(_scales),
            opacities=torch.sigmoid(_opacities),
            colors=sh_coeffs,
            viewmats=viewmat,
            Ks=K_batch,
            width=W_train,
            height=H_train,
            sh_degree=sh_degree,
            packed=False,
            near_plane=tc.near_plane,
            far_plane=tc.far_plane,
            absgrad=True,
        )

        rendered = renders[0]
        l1_loss = F.l1_loss(rendered, gt_image)
        ssim_loss = 1.0 - ssim(
            rendered.permute(2, 0, 1).unsqueeze(0),
            gt_image.permute(2, 0, 1).unsqueeze(0),
            data_range=1.0, size_average=True,
        )
        loss = (1.0 - tc.ssim_lambda) * l1_loss + tc.ssim_lambda * ssim_loss

        # Depth supervision
        depth_loss = torch.tensor(0.0, device=device)
        if depth_weight > 0 and gt_depth is not None:
            # Compute per-gaussian z-depth in camera frame
            N_cur = _means.shape[0]
            means_homo = torch.cat(
                [_means, torch.ones(N_cur, 1, device=device)], dim=-1
            )  # [N, 4]
            means_cam = (viewmat[0] @ means_homo.T).T[:, :3]  # [N, 3]
            depths_per_gaussian = means_cam[:, 2:3]  # [N, 1] z-depth

            # Render depth via separate rasterization (no SH, raw 1-channel color)
            C = 1  # single camera
            depth_renders, _, _ = rasterization(
                means=_means,
                quats=_quats,
                scales=torch.exp(_scales),
                opacities=torch.sigmoid(_opacities),
                colors=depths_per_gaussian.unsqueeze(0).expand(C, -1, -1),  # [1, N, 1]
                viewmats=viewmat,
                Ks=K_batch,
                width=W_train,
                height=H_train,
                sh_degree=None,  # raw colors, no SH
                packed=False,
                near_plane=tc.near_plane,
                far_plane=tc.far_plane,
            )
            rendered_depth = depth_renders[0, :, :, 0]  # [H, W]

            # Masked L1 loss (only where GT depth is valid)
            valid_mask = (gt_depth > 0) & (gt_depth < 10.0)
            if valid_mask.sum() > 100:
                depth_loss = F.l1_loss(rendered_depth[valid_mask], gt_depth[valid_mask])
                loss = loss + depth_weight * depth_loss

        if use_densify:
            strategy.step_pre_backward(
                params=params_dict, optimizers=opt_dict,
                state=strategy_state, step=step, info=meta,
            )

        loss.backward()

        if use_densify:
            strategy.step_post_backward(
                params=params_dict, optimizers=opt_dict,
                state=strategy_state, step=step, info=meta, packed=False,
            )
            N = params_dict["means"].shape[0]

        for opt in opt_dict.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 500 == 0:
            postfix = dict(
                loss=f"{loss.item():.4f}",
                l1=f"{l1_loss.item():.4f}",
                ssim=f"{ssim_loss.item():.4f}",
                n_gs=N,
            )
            if depth_weight > 0:
                postfix["depth"] = f"{depth_loss.item():.4f}"
            pbar.set_postfix(**postfix)

    # --- Quick PSNR evaluation on 50 training frames ---
    print("\nEvaluating PSNR on training frames...")
    eval_indices = dataset.get_eval_indices(stride=max(1, len(train_indices) // 50))
    psnr_values = []
    with torch.no_grad():
        _means_e = params_dict["means"]
        _quats_e = params_dict["quats"]
        _scales_e = params_dict["scales"]
        _opacities_e = params_dict["opacities"]
        sh_coeffs_e = torch.cat([params_dict["sh0"], params_dict["shN"]], dim=1)

        for ei in eval_indices[:50]:
            sample_e = dataset[ei]
            gt_e = sample_e["image"].float().to(device)
            pose_e = sample_e["pose"].float().to(device)

            if tc.data_factor > 1:
                gt_e = F.interpolate(
                    gt_e.permute(2, 0, 1).unsqueeze(0),
                    size=(H_train, W_train),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).permute(1, 2, 0)

            viewmat_e = torch.linalg.inv(pose_e).unsqueeze(0)
            renders_e, _, _ = rasterization(
                means=_means_e, quats=_quats_e,
                scales=torch.exp(_scales_e),
                opacities=torch.sigmoid(_opacities_e),
                colors=sh_coeffs_e,
                viewmats=viewmat_e, Ks=K_train.unsqueeze(0),
                width=W_train, height=H_train,
                sh_degree=sh_degree, packed=False,
                near_plane=tc.near_plane, far_plane=tc.far_plane,
            )
            rendered_e = renders_e[0].clamp(0, 1)
            mse = F.mse_loss(rendered_e, gt_e)
            psnr = -10.0 * torch.log10(mse + 1e-8)
            psnr_values.append(psnr.item())

    avg_psnr = sum(psnr_values) / len(psnr_values)
    print(f"Average PSNR: {avg_psnr:.2f} dB ({len(psnr_values)} frames)")

    # Save checkpoint — combine sh0 + shN back
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{cfg.scene.name}.ckpt"

    # Use params_dict (strategy may have replaced Parameter objects during densification)
    final_sh = torch.cat([params_dict["sh0"].data, params_dict["shN"].data], dim=1)
    torch.save({
        "means": params_dict["means"].data.cpu(),
        "quats": params_dict["quats"].data.cpu(),
        "scales": params_dict["scales"].data.cpu(),
        "opacities": params_dict["opacities"].data.cpu(),
        "sh_coeffs": final_sh.cpu(),
        "sh_degree": sh_degree,
        "scene_bounds": (bounds_min, bounds_max),
        "step": tc.max_steps,
        "init_from_depth": args.init_from_depth,
        "psnr": avg_psnr,
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path} ({N} Gaussians, PSNR={avg_psnr:.2f} dB)")


if __name__ == "__main__":
    main()
