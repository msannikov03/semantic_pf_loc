"""Map Quality vs Localization Accuracy Ablation.

Investigates how 3DGS map quality affects PF localization performance.
Trains maps at different quality levels and measures both PSNR and
localization ATE, producing a scatter plot of PSNR vs ATE.

Ablation axes:
  1. Training steps:   5K, 10K, 20K, 50K
  2. Initial Gaussians: 50K, 100K, 200K, 500K
  3. Depth supervision: depth_weight=0 vs 0.5

For each configuration, measures:
  - PSNR (rendering quality on held-out frames)
  - Localization ATE median (SSIM + gradient refinement)
  - Localization success rate (5cm / 2deg)

Run on office0 (best scene). Total: ~14 configs x ~15 min = ~3.5 hours.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import torch
import torch.nn.functional as F
import pypose as pp
import json
import time
import gc
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from pytorch_msssim import ssim as compute_ssim

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

from semantic_pf_loc.gaussian_map import GaussianMap
from semantic_pf_loc.batch_renderer import BatchRenderer
from semantic_pf_loc.particle_filter import ParticleFilter
from semantic_pf_loc.motion_model import MotionModel
from semantic_pf_loc.gradient_refiner import GradientRefiner
from semantic_pf_loc.observation.ssim import SSIMObservation
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.evaluation.metrics import (
    compute_all_metrics,
    absolute_trajectory_error,
    absolute_rotation_error,
    success_rate as compute_success_rate,
)
from semantic_pf_loc.utils.pose_utils import scale_intrinsics


# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda"

# Scene config (office0 only)
SCENE = {
    "name": "office0",
    "type": "replica",
    "path": "data/replica/office0",
    "native_w": 1200,
    "native_h": 680,
    "data_factor": 4,  # training resolution: 300x170
    "trans_std": 0.003,
    "rot_std": 0.002,
}

# Ablation grid
STEP_COUNTS = [5000, 10000, 20000, 50000]
GAUSSIAN_COUNTS = [50000, 100000, 200000, 500000]
DEPTH_WEIGHTS = [0.0, 0.5]

# Training hyperparameters (matching default.yaml)
TRAIN_PARAMS = {
    "sh_degree": 3,
    "ssim_lambda": 0.2,
    "lr_means": 1.6e-4,
    "lr_scales": 5.0e-3,
    "lr_opacities": 5.0e-2,
    "lr_sh": 2.5e-3,
    "lr_quats": 1.0e-3,
    "near_plane": 0.01,
    "far_plane": 100.0,
}

# Localization parameters
LOC_PARAMS = {
    "n_particles": 200,
    "n_frames": 100,
    "pf_width": 160,
    "pf_height": 120,
    "refine_width": 320,
    "refine_height": 240,
    "refine_iters": 100,
    "refine_lr": 0.01,
    "ssim_temperature": 3.0,
    "trans_spread": 0.03,
    "rot_spread": 0.01,
}

# PSNR measurement
PSNR_FRAMES = 50
PSNR_STRIDE = 10


# ============================================================================
# Training function
# ============================================================================

def train_map(dataset, scene_cfg, max_steps, init_gaussians, depth_weight,
              ckpt_path):
    """Train a 3DGS map with given parameters, save checkpoint."""
    print(f"\n  Training: steps={max_steps}, N_init={init_gaussians}, "
          f"depth_w={depth_weight}")

    K = dataset.get_intrinsics().float().to(DEVICE)
    W, H = dataset.image_size
    df = scene_cfg["data_factor"]
    W_train = W // df
    H_train = H // df

    K_train = K.clone()
    K_train[0, 0] /= df
    K_train[1, 1] /= df
    K_train[0, 2] /= df
    K_train[1, 2] /= df

    bounds_min, bounds_max = dataset.get_bounds()

    N = init_gaussians
    means = torch.nn.Parameter(
        torch.rand(N, 3, device=DEVICE) * (bounds_max - bounds_min).to(DEVICE)
        + bounds_min.to(DEVICE)
    )
    scales = torch.nn.Parameter(torch.full((N, 3), -5.0, device=DEVICE))
    quats = torch.nn.Parameter(torch.randn(N, 4, device=DEVICE))
    opacities = torch.nn.Parameter(torch.zeros(N, device=DEVICE))

    tp = TRAIN_PARAMS
    sh_degree = tp["sh_degree"]
    num_sh = (sh_degree + 1) ** 2

    sh0 = torch.nn.Parameter(torch.zeros(N, 1, 3, device=DEVICE))
    shN = torch.nn.Parameter(torch.zeros(N, num_sh - 1, 3, device=DEVICE))

    opt_means = torch.optim.Adam([means], lr=tp["lr_means"])
    opt_scales = torch.optim.Adam([scales], lr=tp["lr_scales"])
    opt_quats = torch.optim.Adam([quats], lr=tp["lr_quats"])
    opt_opacities = torch.optim.Adam([opacities], lr=tp["lr_opacities"])
    opt_sh0 = torch.optim.Adam([sh0], lr=tp["lr_sh"])
    opt_shN = torch.optim.Adam([shN], lr=tp["lr_sh"])

    params_dict = {
        "means": means, "scales": scales, "quats": quats,
        "opacities": opacities, "sh0": sh0, "shN": shN,
    }
    opt_dict = {
        "means": opt_means, "scales": opt_scales, "quats": opt_quats,
        "opacities": opt_opacities, "sh0": opt_sh0, "shN": opt_shN,
    }

    use_densify = (N <= 50000)
    if use_densify:
        strategy = DefaultStrategy(absgrad=True, verbose=False)
        strategy_state = strategy.initialize_state()

    train_indices = dataset.get_train_indices(stride=1)

    pbar = tqdm(range(max_steps), desc=f"  Train", ncols=100, leave=False)
    for step in pbar:
        idx = train_indices[torch.randint(len(train_indices), (1,)).item()]
        sample = dataset[idx]
        gt_image = sample["image"].float().to(DEVICE)
        pose = sample["pose"].float().to(DEVICE)

        gt_depth = None
        if depth_weight > 0:
            gt_depth = sample.get("depth", None)
            if gt_depth is not None:
                gt_depth = gt_depth.float().to(DEVICE)
                if df > 1:
                    gt_depth = F.interpolate(
                        gt_depth.unsqueeze(0).unsqueeze(0),
                        size=(H_train, W_train), mode="nearest",
                    ).squeeze(0).squeeze(0)

        if df > 1:
            gt_image = F.interpolate(
                gt_image.permute(2, 0, 1).unsqueeze(0),
                size=(H_train, W_train), mode="bilinear", align_corners=False,
            ).squeeze(0).permute(1, 2, 0)

        viewmat = torch.linalg.inv(pose).unsqueeze(0)
        K_batch = K_train.unsqueeze(0)

        _means = params_dict["means"]
        _quats = params_dict["quats"]
        _scales = params_dict["scales"]
        _opacities = params_dict["opacities"]
        sh_coeffs = torch.cat([params_dict["sh0"], params_dict["shN"]], dim=1)

        renders, alphas, meta = rasterization(
            means=_means, quats=_quats,
            scales=torch.exp(_scales),
            opacities=torch.sigmoid(_opacities),
            colors=sh_coeffs, viewmats=viewmat, Ks=K_batch,
            width=W_train, height=H_train,
            sh_degree=sh_degree, packed=False,
            near_plane=tp["near_plane"], far_plane=tp["far_plane"],
            absgrad=True,
        )

        rendered = renders[0]
        l1_loss = F.l1_loss(rendered, gt_image)
        ssim_loss = 1.0 - compute_ssim(
            rendered.permute(2, 0, 1).unsqueeze(0),
            gt_image.permute(2, 0, 1).unsqueeze(0),
            data_range=1.0, size_average=True,
        )
        loss = (1.0 - tp["ssim_lambda"]) * l1_loss + tp["ssim_lambda"] * ssim_loss

        # Depth supervision
        if depth_weight > 0 and gt_depth is not None:
            N_cur = _means.shape[0]
            means_homo = torch.cat(
                [_means, torch.ones(N_cur, 1, device=DEVICE)], dim=-1
            )
            means_cam = (viewmat[0] @ means_homo.T).T[:, :3]
            depths_per_gaussian = means_cam[:, 2:3]

            C = 1
            depth_renders, _, _ = rasterization(
                means=_means, quats=_quats,
                scales=torch.exp(_scales),
                opacities=torch.sigmoid(_opacities),
                colors=depths_per_gaussian.unsqueeze(0).expand(C, -1, -1),
                viewmats=viewmat, Ks=K_batch,
                width=W_train, height=H_train,
                sh_degree=None, packed=False,
                near_plane=tp["near_plane"], far_plane=tp["far_plane"],
            )
            rendered_depth = depth_renders[0, :, :, 0]

            valid_mask = (gt_depth > 0) & (gt_depth < 10.0)
            if valid_mask.sum() > 100:
                depth_loss = F.l1_loss(rendered_depth[valid_mask],
                                       gt_depth[valid_mask])
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

        for opt in opt_dict.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        N_cur = params_dict["means"].shape[0]

        if step % 2000 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", n_gs=N_cur)

    # Save checkpoint
    final_sh = torch.cat(
        [params_dict["sh0"].data, params_dict["shN"].data], dim=1
    )
    torch.save({
        "means": params_dict["means"].data.cpu(),
        "quats": params_dict["quats"].data.cpu(),
        "scales": params_dict["scales"].data.cpu(),
        "opacities": params_dict["opacities"].data.cpu(),
        "sh_coeffs": final_sh.cpu(),
        "sh_degree": sh_degree,
        "scene_bounds": (bounds_min, bounds_max),
        "step": max_steps,
    }, ckpt_path)

    final_N = params_dict["means"].shape[0]
    print(f"    Saved: {ckpt_path} ({final_N} Gaussians)")

    # Cleanup
    del params_dict, opt_dict, renders, meta
    if use_densify:
        del strategy, strategy_state
    torch.cuda.empty_cache()
    gc.collect()

    return final_N


# ============================================================================
# PSNR measurement
# ============================================================================

def measure_psnr(gmap, dataset, scene_cfg):
    """Compute PSNR on held-out frames at half-native resolution."""
    psnr_w = scene_cfg["native_w"] // 2
    psnr_h = scene_cfg["native_h"] // 2
    native_size = (scene_cfg["native_w"], scene_cfg["native_h"])

    renderer = BatchRenderer(gmap, width=psnr_w, height=psnr_h)
    K_native = dataset.get_intrinsics().float().to(DEVICE)
    K_psnr = scale_intrinsics(K_native, native_size, (psnr_w, psnr_h))

    psnrs = []
    indices = list(range(0, min(500, len(dataset)), PSNR_STRIDE))[:PSNR_FRAMES]

    for i in indices:
        sample = dataset[i]
        gt_img = sample["image"].float().to(DEVICE)
        gt_pose_mat = sample["pose"].double().to(DEVICE)

        gt_se3 = pp.mat2SE3(
            gt_pose_mat.unsqueeze(0), check=False
        ).squeeze(0).float()
        viewmat = gt_se3.Inv().matrix().unsqueeze(0)

        with torch.no_grad():
            rendered, _, _ = renderer.render_batch(
                viewmat, K_psnr.unsqueeze(0)
            )

        gt_resized = F.interpolate(
            gt_img.permute(2, 0, 1).unsqueeze(0),
            size=(psnr_h, psnr_w), mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0)

        mse = ((rendered[0] - gt_resized) ** 2).mean()
        if mse > 0:
            psnr = -10 * torch.log10(mse)
            psnrs.append(psnr.item())

    del renderer
    torch.cuda.empty_cache()
    return float(np.mean(psnrs)) if psnrs else 0.0


# ============================================================================
# Localization evaluation
# ============================================================================

def evaluate_localization(gmap, dataset, scene_cfg):
    """Run PF + gradient refinement, return ATE/ARE/SR metrics."""
    lp = LOC_PARAMS
    K = dataset.get_intrinsics().float().to(DEVICE)
    native_size = (scene_cfg["native_w"], scene_cfg["native_h"])

    renderer = BatchRenderer(gmap, width=lp["pf_width"], height=lp["pf_height"])
    obs_model = SSIMObservation(temperature=lp["ssim_temperature"])
    motion = MotionModel(
        translation_std=scene_cfg["trans_std"],
        rotation_std=scene_cfg["rot_std"],
        device=DEVICE,
    )

    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=lp["n_particles"],
        render_width=lp["pf_width"],
        render_height=lp["pf_height"],
        render_width_hires=lp["refine_width"],
        render_height_hires=lp["refine_height"],
        convergence_threshold=0.02,
        roughening_trans=0.002,
        roughening_rot=0.001,
        gradient_refiner=None,
        device=DEVICE,
    )

    # High-res renderer + refiner for post-hoc refinement
    K_hires = scale_intrinsics(K, native_size,
                                (lp["refine_width"], lp["refine_height"]))
    hires_renderer = BatchRenderer(
        gmap, width=lp["refine_width"], height=lp["refine_height"]
    )
    refiner = GradientRefiner(
        hires_renderer,
        num_iterations=lp["refine_iters"],
        lr_init=lp["refine_lr"],
        blur_schedule=True,
        blur_sigma_init=10.0,
        blur_sigma_final=0.1,
    )

    # Initialize around GT pose 0
    sample0 = dataset[0]
    gt_pose0 = pp.mat2SE3(
        sample0["pose"].double().to(DEVICE).unsqueeze(0), check=False
    ).squeeze(0).float()
    pf.initialize_around_pose(
        gt_pose0,
        trans_spread=lp["trans_spread"],
        rot_spread=lp["rot_spread"],
    )

    est_poses, gt_poses = [], []
    n_frames = min(lp["n_frames"], len(dataset))

    for i in range(n_frames):
        sample = dataset[i]
        obs = {"image": sample["image"].float().to(DEVICE)}

        est, info = pf.step(obs, K)

        # Post-hoc refinement on converged frames
        if info["converged"]:
            est_for_refine = pp.SE3(est.tensor().unsqueeze(0))
            with torch.enable_grad():
                refined = refiner.refine(
                    est_for_refine,
                    sample["image"].float().to(DEVICE),
                    K_hires,
                )
            est = refined.squeeze(0)

        est_poses.append(est.matrix().cpu())
        gt_poses.append(sample["pose"].float())

    est_stack = torch.stack(est_poses)
    gt_stack = torch.stack(gt_poses)

    ate = absolute_trajectory_error(est_stack, gt_stack)
    are = absolute_rotation_error(est_stack, gt_stack)
    sr = compute_success_rate(est_stack, gt_stack,
                               trans_threshold=0.05, rot_threshold=2.0)

    # Cleanup
    del renderer, hires_renderer, refiner, pf
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "ate_median_cm": ate["median"] * 100,
        "ate_rmse_cm": ate["rmse"] * 100,
        "are_median_deg": are["median"],
        "success_rate": sr * 100,
    }


# ============================================================================
# Main ablation loop
# ============================================================================

def build_configs(mode="full"):
    """Build list of (max_steps, init_gaussians, depth_weight) configs.

    Modes:
      full:   full grid (STEP_COUNTS x GAUSSIAN_COUNTS x DEPTH_WEIGHTS)
      quick:  reduced grid for faster testing
      steps:  vary steps only (200K init, depth=0.5)
      gauss:  vary gaussians only (50K steps, depth=0.5)
    """
    configs = []

    if mode == "full":
        # Primary axis: training steps (fixed 200K Gaussians)
        for steps in STEP_COUNTS:
            for dw in DEPTH_WEIGHTS:
                configs.append((steps, 200000, dw))

        # Secondary axis: Gaussian count (fixed 50K steps)
        for n_gauss in GAUSSIAN_COUNTS:
            for dw in DEPTH_WEIGHTS:
                key = (50000, n_gauss, dw)
                # Avoid duplicates
                if key not in [(c[0], c[1], c[2]) for c in configs]:
                    configs.append(key)

    elif mode == "quick":
        # Reduced grid for testing
        for steps in [5000, 20000, 50000]:
            configs.append((steps, 200000, 0.5))
        for n_gauss in [50000, 200000, 500000]:
            if (50000, n_gauss, 0.5) not in configs:
                configs.append((50000, n_gauss, 0.5))

    elif mode == "steps":
        for steps in STEP_COUNTS:
            configs.append((steps, 200000, 0.5))

    elif mode == "gauss":
        for n_gauss in GAUSSIAN_COUNTS:
            configs.append((50000, n_gauss, 0.5))

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Map quality vs localization accuracy ablation"
    )
    parser.add_argument(
        "--mode", default="full",
        choices=["full", "quick", "steps", "gauss"],
        help="Ablation mode: full grid, quick test, steps-only, gaussians-only"
    )
    parser.add_argument(
        "--output_dir", default="results/map_quality_ablation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--ckpt_dir", default="checkpoints_ablation",
        help="Directory to save intermediate checkpoints"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip configs that already have results"
    )
    parser.add_argument(
        "--train_only", action="store_true",
        help="Only train maps, skip localization evaluation"
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Only evaluate existing checkpoints, skip training"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args.mode)

    start_time = datetime.now()
    print("=" * 70)
    print("  MAP QUALITY vs LOCALIZATION ACCURACY ABLATION")
    print(f"  Scene: {SCENE['name']}")
    print(f"  Mode: {args.mode} ({len(configs)} configurations)")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load dataset once
    dataset = ReplicaDataset(SCENE["path"], stride=1)
    print(f"  Dataset: {len(dataset)} frames")

    # Load existing results if any
    results_path = output_dir / "ablation_results.json"
    if results_path.exists() and args.skip_existing:
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"  Loaded {len(all_results)} existing results")
    else:
        all_results = []

    existing_keys = set()
    for r in all_results:
        key = f"s{r['max_steps']}_n{r['init_gaussians']}_d{r['depth_weight']}"
        existing_keys.add(key)

    for ci, (max_steps, init_gaussians, depth_weight) in enumerate(configs):
        config_key = f"s{max_steps}_n{init_gaussians}_d{depth_weight}"
        ckpt_name = f"office0_{config_key}.ckpt"
        ckpt_path = ckpt_dir / ckpt_name

        print(f"\n{'='*70}")
        print(f"  Config {ci+1}/{len(configs)}: {config_key}")
        print(f"    steps={max_steps}, gaussians={init_gaussians}, "
              f"depth_weight={depth_weight}")
        print(f"{'='*70}")

        if config_key in existing_keys and args.skip_existing:
            print("  SKIP (already evaluated)")
            continue

        # ---- Train ----
        if not args.eval_only:
            if ckpt_path.exists() and args.skip_existing:
                print(f"  Checkpoint exists, skipping training")
            else:
                t0 = time.time()
                final_N = train_map(
                    dataset, SCENE, max_steps, init_gaussians,
                    depth_weight, str(ckpt_path),
                )
                train_time = time.time() - t0
                print(f"    Training time: {train_time:.1f}s "
                      f"({train_time/60:.1f} min)")

        if args.train_only:
            continue

        # ---- Load checkpoint ----
        if not ckpt_path.exists():
            print(f"  ERROR: checkpoint not found: {ckpt_path}")
            continue

        gmap = GaussianMap.from_checkpoint(str(ckpt_path), device=DEVICE)
        print(f"  Loaded {gmap.num_gaussians:,} Gaussians")

        # ---- Measure PSNR ----
        print("  Measuring PSNR...")
        t0 = time.time()
        psnr_val = measure_psnr(gmap, dataset, SCENE)
        psnr_time = time.time() - t0
        print(f"    PSNR: {psnr_val:.2f} dB ({psnr_time:.1f}s)")

        # ---- Evaluate localization ----
        print("  Running localization (PF + refine)...")
        t0 = time.time()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        loc_metrics = evaluate_localization(gmap, dataset, SCENE)
        loc_time = time.time() - t0
        print(f"    ATE median: {loc_metrics['ate_median_cm']:.2f} cm")
        print(f"    ARE median: {loc_metrics['are_median_deg']:.2f} deg")
        print(f"    Success rate: {loc_metrics['success_rate']:.0f}%")
        print(f"    Localization time: {loc_time:.1f}s")

        # ---- Store result ----
        result = {
            "config_key": config_key,
            "max_steps": max_steps,
            "init_gaussians": init_gaussians,
            "depth_weight": depth_weight,
            "final_gaussians": gmap.num_gaussians,
            "psnr_db": psnr_val,
            **loc_metrics,
        }
        all_results.append(result)

        # Save incrementally (in case of crash)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        del gmap
        torch.cuda.empty_cache()
        gc.collect()

    # ================================================================
    # Generate figures and summary
    # ================================================================
    if args.train_only or len(all_results) == 0:
        print("\nNo evaluation results to plot.")
        return

    print(f"\n{'='*70}")
    print("  GENERATING FIGURES")
    print(f"{'='*70}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # ---- Figure 1: PSNR vs ATE scatter plot ----
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Color by depth_weight, marker by gaussian count
        markers = {50000: "o", 100000: "s", 200000: "^", 500000: "D"}
        colors = {0.0: "#2196F3", 0.5: "#FF5722"}

        for r in all_results:
            marker = markers.get(r["init_gaussians"], "x")
            color = colors.get(r["depth_weight"], "gray")
            ax.scatter(
                r["psnr_db"], r["ate_median_cm"],
                marker=marker, c=color, s=100, edgecolors="black",
                linewidths=0.5, zorder=5,
            )
            # Annotate with step count
            ax.annotate(
                f'{r["max_steps"]//1000}K',
                (r["psnr_db"], r["ate_median_cm"]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, alpha=0.7,
            )

        ax.set_xlabel("PSNR (dB)", fontsize=12)
        ax.set_ylabel("Localization ATE Median (cm)", fontsize=12)
        ax.set_title("Map Quality vs Localization Accuracy (office0)", fontsize=13)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = []
        # Colors for depth weight
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#2196F3", markersize=10,
                   label="No depth supervision")
        )
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#FF5722", markersize=10,
                   label="Depth weight = 0.5")
        )
        # Markers for gaussian count
        for n_gauss, mk in markers.items():
            legend_elements.append(
                Line2D([0], [0], marker=mk, color="w",
                       markerfacecolor="gray", markersize=8,
                       label=f"{n_gauss//1000}K init Gaussians")
            )
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # lower ATE is better = top

        fig.tight_layout()
        fig.savefig(figures_dir / "psnr_vs_ate.png", dpi=150)
        fig.savefig(figures_dir / "psnr_vs_ate.pdf")
        plt.close(fig)
        print(f"  Saved: {figures_dir / 'psnr_vs_ate.png'}")

        # ---- Figure 2: PSNR vs Success Rate ----
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for r in all_results:
            marker = markers.get(r["init_gaussians"], "x")
            color = colors.get(r["depth_weight"], "gray")
            ax.scatter(
                r["psnr_db"], r["success_rate"],
                marker=marker, c=color, s=100, edgecolors="black",
                linewidths=0.5, zorder=5,
            )
            ax.annotate(
                f'{r["max_steps"]//1000}K',
                (r["psnr_db"], r["success_rate"]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, alpha=0.7,
            )

        ax.set_xlabel("PSNR (dB)", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title(
            "Map Quality vs Localization Success Rate (office0)", fontsize=13
        )
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

        fig.tight_layout()
        fig.savefig(figures_dir / "psnr_vs_sr.png", dpi=150)
        fig.savefig(figures_dir / "psnr_vs_sr.pdf")
        plt.close(fig)
        print(f"  Saved: {figures_dir / 'psnr_vs_sr.png'}")

        # ---- Figure 3: Training steps vs ATE (line plot) ----
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for dw in DEPTH_WEIGHTS:
            # Filter for default 200K Gaussians
            step_results = sorted(
                [r for r in all_results
                 if r["init_gaussians"] == 200000 and r["depth_weight"] == dw],
                key=lambda x: x["max_steps"],
            )
            if not step_results:
                continue
            steps = [r["max_steps"] for r in step_results]
            ates = [r["ate_median_cm"] for r in step_results]
            psnrs = [r["psnr_db"] for r in step_results]
            label = f"depth_w={dw}"
            color = colors[dw]

            ax1.plot(steps, ates, "o-", color=color, label=label, markersize=8)
            ax2.plot(steps, psnrs, "o-", color=color, label=label, markersize=8)

        ax1.set_xlabel("Training Steps", fontsize=12)
        ax1.set_ylabel("ATE Median (cm)", fontsize=12)
        ax1.set_title("Training Steps vs Localization Error", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")

        ax2.set_xlabel("Training Steps", fontsize=12)
        ax2.set_ylabel("PSNR (dB)", fontsize=12)
        ax2.set_title("Training Steps vs Map Quality", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")

        fig.tight_layout()
        fig.savefig(figures_dir / "steps_vs_ate_psnr.png", dpi=150)
        fig.savefig(figures_dir / "steps_vs_ate_psnr.pdf")
        plt.close(fig)
        print(f"  Saved: {figures_dir / 'steps_vs_ate_psnr.png'}")

        # ---- Figure 4: Gaussian count vs ATE ----
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for dw in DEPTH_WEIGHTS:
            gauss_results = sorted(
                [r for r in all_results
                 if r["max_steps"] == 50000 and r["depth_weight"] == dw],
                key=lambda x: x["init_gaussians"],
            )
            if not gauss_results:
                continue
            ns = [r["init_gaussians"] for r in gauss_results]
            ates = [r["ate_median_cm"] for r in gauss_results]
            psnrs = [r["psnr_db"] for r in gauss_results]
            label = f"depth_w={dw}"
            color = colors[dw]

            ax1.plot(ns, ates, "o-", color=color, label=label, markersize=8)
            ax2.plot(ns, psnrs, "o-", color=color, label=label, markersize=8)

        ax1.set_xlabel("Initial Gaussians", fontsize=12)
        ax1.set_ylabel("ATE Median (cm)", fontsize=12)
        ax1.set_title("Gaussian Count vs Localization Error", fontsize=13)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")

        ax2.set_xlabel("Initial Gaussians", fontsize=12)
        ax2.set_ylabel("PSNR (dB)", fontsize=12)
        ax2.set_title("Gaussian Count vs Map Quality", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")

        fig.tight_layout()
        fig.savefig(figures_dir / "gaussians_vs_ate_psnr.png", dpi=150)
        fig.savefig(figures_dir / "gaussians_vs_ate_psnr.pdf")
        plt.close(fig)
        print(f"  Saved: {figures_dir / 'gaussians_vs_ate_psnr.png'}")

    except ImportError:
        print("  matplotlib not available, skipping figure generation")

    # ================================================================
    # Print summary table
    # ================================================================
    elapsed_total = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*90}")
    print(f"  MAP QUALITY ABLATION RESULTS")
    print(f"{'='*90}")
    header = (
        f"{'Config':<30} | {'PSNR':>6} | {'ATE':>8} | {'ARE':>8} | "
        f"{'SR%':>5} | {'N_final':>8}"
    )
    print(header)
    print("-" * 90)

    # Sort by PSNR
    for r in sorted(all_results, key=lambda x: x["psnr_db"]):
        config = (f"s={r['max_steps']//1000}K n={r['init_gaussians']//1000}K "
                  f"d={r['depth_weight']}")
        print(
            f"{config:<30} | {r['psnr_db']:>5.1f} | "
            f"{r['ate_median_cm']:>6.2f}cm | "
            f"{r['are_median_deg']:>6.2f}d | "
            f"{r['success_rate']:>4.0f}% | "
            f"{r['final_gaussians']:>8,}"
        )

    print(f"\nTotal time: {elapsed_total/60:.1f} minutes")
    print(f"Results: {results_path}")
    print(f"Figures: {output_dir / 'figures'}/")
    print("=" * 90)


if __name__ == "__main__":
    main()
