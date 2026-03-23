"""Train 3D Gaussian Splatting maps for localization."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to scene config YAML")
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory")
    parser.add_argument("--init_gaussians", type=int, default=None)
    parser.add_argument("--depth_weight", type=float, default=0.0,
                        help="Weight for depth supervision loss (0=disabled, 0.5=recommended)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max_steps from config")
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
    means = torch.nn.Parameter(
        torch.rand(N, 3, device=device) * (bounds_max - bounds_min).to(device)
        + bounds_min.to(device)
    )
    scales = torch.nn.Parameter(torch.full((N, 3), -5.0, device=device))
    quats = torch.nn.Parameter(torch.randn(N, 4, device=device))
    opacities = torch.nn.Parameter(torch.zeros(N, device=device))

    sh_degree = tc.sh_degree
    num_sh = (sh_degree + 1) ** 2

    # IMPORTANT: separate SH DC and SH rest as independent parameters
    # gsplat's DefaultStrategy modifies params in-place during densification
    sh0 = torch.nn.Parameter(torch.zeros(N, 1, 3, device=device))   # DC
    shN = torch.nn.Parameter(torch.zeros(N, num_sh - 1, 3, device=device))  # rest

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
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path} ({N} Gaussians)")


if __name__ == "__main__":
    main()
