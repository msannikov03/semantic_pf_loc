"""Test observation model improvements on room0 (the failing scene).

Configurations tested:
  1. baseline_ssim_160x120     — current system (SSIM, 160x120, temp=3.0)
  2. msssim_160x120            — MS-SSIM at same resolution
  3. ssim_320x240              — SSIM at higher resolution (4x pixels)
  4. ssim_160x120_anneal       — SSIM with temperature annealing 1->5
  5. ssim_320x240_anneal       — higher res + annealing
  6. msssim_320x240_anneal     — MS-SSIM + higher res + annealing

Each config: 200 particles, 100 frames, 1 trial, post-hoc refinement (320x240).
"""

import sys
import os
# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pypose as pp
import numpy as np
import json
import time
import gc
from pathlib import Path
from datetime import datetime

from semantic_pf_loc.gaussian_map import GaussianMap
from semantic_pf_loc.batch_renderer import BatchRenderer
from semantic_pf_loc.particle_filter import ParticleFilter
from semantic_pf_loc.motion_model import MotionModel
from semantic_pf_loc.gradient_refiner import GradientRefiner
from semantic_pf_loc.observation.ssim import SSIMObservation
from semantic_pf_loc.observation.ms_ssim import MSSSIMObservation
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.datasets.tum import TUMDataset
from semantic_pf_loc.evaluation.metrics import (
    compute_all_metrics, translation_error, rotation_error,
)
from semantic_pf_loc.utils.pose_utils import scale_intrinsics

# ============================================================================
# Test configurations
# ============================================================================

CONFIGS = [
    {
        "name": "baseline_ssim_160x120",
        "obs": "ssim",
        "res": (160, 120),
        "temp": 3.0,
        "anneal": False,
    },
    {
        "name": "msssim_160x120",
        "obs": "ms_ssim",
        "res": (160, 120),
        "temp": 3.0,
        "anneal": False,
    },
    {
        "name": "ssim_320x240",
        "obs": "ssim",
        "res": (320, 240),
        "temp": 3.0,
        "anneal": False,
    },
    {
        "name": "ssim_160x120_anneal",
        "obs": "ssim",
        "res": (160, 120),
        "temp": 1.0,  # start temp for annealing
        "anneal": True,
    },
    {
        "name": "ssim_320x240_anneal",
        "obs": "ssim",
        "res": (320, 240),
        "temp": 1.0,
        "anneal": True,
    },
    {
        "name": "msssim_320x240_anneal",
        "obs": "ms_ssim",
        "res": (320, 240),
        "temp": 1.0,
        "anneal": True,
    },
]

N_FRAMES = 100
N_PARTICLES = 200
DEVICE = "cuda"
SEED = 42


# ============================================================================
# Scene definitions
# ============================================================================

ROOM0 = {
    "name": "room0", "type": "replica",
    "path": "data/replica/room0",
    "ckpt": "checkpoints_depthinit/room0.ckpt",
    "native_w": 1200, "native_h": 680,
    "trans_std": 0.003, "rot_std": 0.002,
}

REGRESSION_SCENES = [
    {
        "name": "office0", "type": "replica",
        "path": "data/replica/office0",
        "ckpt": "checkpoints_depthinit/office0.ckpt",
        "native_w": 1200, "native_h": 680,
        "trans_std": 0.003, "rot_std": 0.002,
    },
    {
        "name": "fr3_office", "type": "tum",
        "path": "data/tum/rgbd_dataset_freiburg3_long_office_household",
        "ckpt": "checkpoints_depthinit/fr3_office.ckpt",
        "native_w": 640, "native_h": 480,
        "trans_std": 0.005, "rot_std": 0.003,
    },
]


def make_obs_model(obs_type, temperature):
    """Create observation model by type string."""
    if obs_type == "ssim":
        return SSIMObservation(temperature=temperature)
    elif obs_type == "ms_ssim":
        return MSSSIMObservation(temperature=temperature)
    else:
        raise ValueError(f"Unknown obs type: {obs_type}")


def run_single_config(dataset, gmap, scene_cfg, config, seed=42):
    """Run PF + post-hoc refinement for a single config on a single scene."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    rw, rh = config["res"]
    K = dataset.get_intrinsics().float().to(DEVICE)
    native_size = (scene_cfg["native_w"], scene_cfg["native_h"])

    # Observation model
    obs_model = make_obs_model(config["obs"], config["temp"])

    # PF renderer at config resolution
    renderer = BatchRenderer(gmap, width=rw, height=rh)
    motion = MotionModel(
        translation_std=scene_cfg["trans_std"],
        rotation_std=scene_cfg["rot_std"],
        device=DEVICE,
    )

    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=N_PARTICLES,
        render_width=rw, render_height=rh,
        render_width_hires=rw, render_height_hires=rh,  # same res (no convergence switch)
        convergence_threshold=0.02,
        roughening_trans=0.002, roughening_rot=0.001,
        gradient_refiner=None,
        device=DEVICE,
    )

    # Post-hoc refiner always at 320x240
    hires_renderer = BatchRenderer(gmap, width=320, height=240)
    K_hires = scale_intrinsics(K, native_size, (320, 240))
    refiner = GradientRefiner(
        hires_renderer,
        num_iterations=100,
        lr_init=0.01,
        blur_schedule=True,
        blur_sigma_init=10.0,
        blur_sigma_final=0.1,
    )

    # Init around GT pose 0
    sample0 = dataset[0]
    gt_pose0 = pp.mat2SE3(
        sample0["pose"].double().to(DEVICE).unsqueeze(0), check=False
    ).squeeze(0).float()
    pf.initialize_around_pose(gt_pose0, trans_spread=0.03, rot_spread=0.01)

    raw_est_poses, refined_est_poses, gt_poses = [], [], []
    step_times = []

    for i in range(min(N_FRAMES, len(dataset))):
        sample = dataset[i]
        obs = {"image": sample["image"].float().to(DEVICE)}

        # Temperature annealing: update before each step
        if config["anneal"]:
            annealed_temp = 1.0 + (5.0 - 1.0) * min(i / 30.0, 1.0)
            obs_model.temperature = annealed_temp

        t0 = time.time()
        est, info = pf.step(obs, K)
        t_pf = time.time() - t0

        raw_est_poses.append(est.matrix().cpu())

        # Post-hoc refinement
        t0_ref = time.time()
        est_for_refine = pp.SE3(est.tensor().unsqueeze(0))
        with torch.enable_grad():
            refined = refiner.refine(
                est_for_refine, sample["image"].float().to(DEVICE), K_hires
            )
        refined_pose = refined.squeeze(0)
        t_ref = time.time() - t0_ref

        refined_est_poses.append(refined_pose.matrix().cpu())
        gt_poses.append(sample["pose"].float())
        step_times.append((t_pf + t_ref) * 1000)

        if i % 20 == 0:
            t_err = (est.matrix().cpu()[:3, 3] - sample["pose"].float()[:3, 3]).norm().item()
            temp_str = f" temp={obs_model.temperature:.1f}" if config["anneal"] else ""
            print(f"    frame {i:3d}: raw_err={t_err*100:.1f}cm  "
                  f"neff={info['n_eff']:.0f}  conv={info['converged']}{temp_str}")

    # Compute metrics
    raw_stack = torch.stack(raw_est_poses)
    refined_stack = torch.stack(refined_est_poses)
    gt_stack = torch.stack(gt_poses)
    time_tensor = torch.tensor(step_times)

    raw_metrics = compute_all_metrics(raw_stack, gt_stack)
    refined_metrics = compute_all_metrics(refined_stack, gt_stack, time_tensor)

    raw_trans = translation_error(raw_stack, gt_stack)
    ref_trans = translation_error(refined_stack, gt_stack)

    result = {
        "config_name": config["name"],
        "scene": scene_cfg["name"],
        "raw_ate_median_cm": raw_metrics["ate"]["median"] * 100,
        "raw_ate_mean_cm": raw_metrics["ate"]["mean"] * 100,
        "raw_sr": raw_metrics["success_rate"] * 100,
        "refined_ate_median_cm": refined_metrics["ate"]["median"] * 100,
        "refined_ate_mean_cm": refined_metrics["ate"]["mean"] * 100,
        "refined_sr": refined_metrics["success_rate"] * 100,
        "refined_are_median": refined_metrics["are"]["median"],
        "runtime_mean_ms": refined_metrics.get("runtime_mean_ms", 0),
        "convergence_frame": refined_metrics["convergence_frame"],
        "raw_trans_errors": raw_trans.tolist(),
        "refined_trans_errors": ref_trans.tolist(),
    }

    # Cleanup
    del renderer, pf, hires_renderer, refiner, obs_model
    torch.cuda.empty_cache()
    gc.collect()

    return result


def load_scene(scene_cfg):
    """Load dataset and Gaussian map for a scene."""
    if scene_cfg["type"] == "tum":
        dataset = TUMDataset(scene_cfg["path"], stride=1)
    else:
        dataset = ReplicaDataset(scene_cfg["path"], stride=1)

    gmap = GaussianMap.from_checkpoint(scene_cfg["ckpt"])
    return dataset, gmap


def main():
    torch.set_grad_enabled(False)

    output_dir = Path("results/observation_fixes")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    print("=" * 78)
    print("  OBSERVATION MODEL IMPROVEMENTS — room0 TEST")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Particles: {N_PARTICLES}, Frames: {N_FRAMES}, Seed: {SEED}")
    print(f"  Configs: {len(CONFIGS)}")
    print("=" * 78)

    # ================================================================
    # Phase 1: Test all configs on room0
    # ================================================================
    print(f"\n{'='*78}")
    print("  PHASE 1: room0 (all configs)")
    print(f"{'='*78}")

    dataset, gmap = load_scene(ROOM0)
    print(f"  Loaded: {gmap.num_gaussians} Gaussians, {len(dataset)} frames")

    room0_results = []
    for cfg in CONFIGS:
        print(f"\n  --- {cfg['name']} ---")
        print(f"  obs={cfg['obs']}, res={cfg['res']}, temp={cfg['temp']}, anneal={cfg['anneal']}")
        t0 = time.time()

        result = run_single_config(dataset, gmap, ROOM0, cfg, seed=SEED)
        elapsed = time.time() - t0

        print(f"  => Raw PF:  ATE={result['raw_ate_median_cm']:.2f}cm  SR={result['raw_sr']:.0f}%")
        print(f"  => Refined: ATE={result['refined_ate_median_cm']:.2f}cm  SR={result['refined_sr']:.0f}%  "
              f"ARE={result['refined_are_median']:.2f}deg  [{elapsed:.1f}s]")
        room0_results.append(result)

    # Cleanup room0
    del dataset, gmap
    torch.cuda.empty_cache()
    gc.collect()

    # ================================================================
    # Phase 1 summary
    # ================================================================
    print(f"\n{'='*78}")
    print("  ROOM0 RESULTS SUMMARY")
    print(f"{'='*78}")
    print(f"{'Config':<30} | {'Raw ATE':>10} | {'Ref ATE':>10} | {'Ref SR':>8} | {'ms/step':>8}")
    print("-" * 78)
    for r in room0_results:
        print(f"{r['config_name']:<30} | {r['raw_ate_median_cm']:>8.2f}cm | "
              f"{r['refined_ate_median_cm']:>8.2f}cm | {r['refined_sr']:>6.0f}% | "
              f"{r['runtime_mean_ms']:>6.0f}ms")

    # Find best config (lowest refined ATE)
    best = min(room0_results, key=lambda r: r["refined_ate_median_cm"])
    print(f"\n  BEST CONFIG: {best['config_name']}  "
          f"(ATE={best['refined_ate_median_cm']:.2f}cm, SR={best['refined_sr']:.0f}%)")

    # ================================================================
    # Phase 2: Test best config on office0 and fr3_office for regression
    # ================================================================
    best_cfg = next(c for c in CONFIGS if c["name"] == best["config_name"])

    print(f"\n{'='*78}")
    print(f"  PHASE 2: Regression test — {best_cfg['name']} on office0, fr3_office")
    print(f"{'='*78}")

    regression_results = []
    for scene_cfg in REGRESSION_SCENES:
        if not Path(scene_cfg["ckpt"]).exists():
            print(f"  SKIP {scene_cfg['name']}: checkpoint not found")
            continue

        print(f"\n  --- {scene_cfg['name']} with {best_cfg['name']} ---")
        dataset, gmap = load_scene(scene_cfg)
        print(f"  Loaded: {gmap.num_gaussians} Gaussians, {len(dataset)} frames")

        t0 = time.time()
        result = run_single_config(dataset, gmap, scene_cfg, best_cfg, seed=SEED)
        elapsed = time.time() - t0

        print(f"  => Raw PF:  ATE={result['raw_ate_median_cm']:.2f}cm  SR={result['raw_sr']:.0f}%")
        print(f"  => Refined: ATE={result['refined_ate_median_cm']:.2f}cm  SR={result['refined_sr']:.0f}%  "
              f"ARE={result['refined_are_median']:.2f}deg  [{elapsed:.1f}s]")
        regression_results.append(result)

        del dataset, gmap
        torch.cuda.empty_cache()
        gc.collect()

    # Also run baseline on regression scenes for comparison
    baseline_cfg = CONFIGS[0]  # baseline_ssim_160x120
    baseline_regression = []
    if best_cfg["name"] != baseline_cfg["name"]:
        print(f"\n  --- Baseline comparison on regression scenes ---")
        for scene_cfg in REGRESSION_SCENES:
            if not Path(scene_cfg["ckpt"]).exists():
                continue

            print(f"\n  --- {scene_cfg['name']} with {baseline_cfg['name']} ---")
            dataset, gmap = load_scene(scene_cfg)

            t0 = time.time()
            result = run_single_config(dataset, gmap, scene_cfg, baseline_cfg, seed=SEED)
            elapsed = time.time() - t0

            print(f"  => Raw PF:  ATE={result['raw_ate_median_cm']:.2f}cm  SR={result['raw_sr']:.0f}%")
            print(f"  => Refined: ATE={result['refined_ate_median_cm']:.2f}cm  SR={result['refined_sr']:.0f}%  "
                  f"[{elapsed:.1f}s]")
            baseline_regression.append(result)

            del dataset, gmap
            torch.cuda.empty_cache()
            gc.collect()

    # ================================================================
    # Final summary
    # ================================================================
    elapsed_total = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*78}")
    print("  FINAL SUMMARY")
    print(f"{'='*78}")

    print(f"\n  room0 (baseline ATE ~38.5cm):")
    print(f"  {'Config':<30} | {'Refined ATE':>12} | {'SR':>6} | {'vs baseline':>12}")
    print(f"  {'-'*68}")
    baseline_ate = room0_results[0]["refined_ate_median_cm"]
    for r in room0_results:
        diff = r["refined_ate_median_cm"] - baseline_ate
        diff_str = f"{diff:+.2f}cm" if r["config_name"] != "baseline_ssim_160x120" else "---"
        print(f"  {r['config_name']:<30} | {r['refined_ate_median_cm']:>10.2f}cm | "
              f"{r['refined_sr']:>4.0f}% | {diff_str:>12}")

    if regression_results:
        print(f"\n  Regression check ({best_cfg['name']}):")
        print(f"  {'Scene':<15} | {'Best cfg ATE':>14} | {'Baseline ATE':>14} | {'Delta':>10}")
        print(f"  {'-'*60}")
        for i, r in enumerate(regression_results):
            if i < len(baseline_regression):
                bl_ate = baseline_regression[i]["refined_ate_median_cm"]
                delta = r["refined_ate_median_cm"] - bl_ate
                bl_str = f"{bl_ate:.2f}cm"
                delta_str = f"{delta:+.2f}cm"
            else:
                bl_str = "N/A"
                delta_str = "N/A"
            print(f"  {r['scene']:<15} | {r['refined_ate_median_cm']:>12.2f}cm | "
                  f"{bl_str:>14} | {delta_str:>10}")

    print(f"\n  Total time: {elapsed_total/60:.1f} minutes")

    # Save all results
    all_results = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Observation model improvements test",
            "n_particles": N_PARTICLES,
            "n_frames": N_FRAMES,
            "seed": SEED,
            "best_config": best_cfg["name"],
        },
        "room0_results": room0_results,
        "regression_results": regression_results,
        "baseline_regression": baseline_regression,
    }

    json_path = output_dir / "observation_fixes_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {json_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
