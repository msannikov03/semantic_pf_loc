"""Regression test: LPIPS t=3.0 vs SSIM t=3.0 on office0 and fr3_office.

LPIPS t=3.0 @ 160x120 won room0 (31.74cm vs SSIM 38.54cm).
Now check for regressions on scenes where SSIM already works well.

Usage:
    cd /home/anywherevla/semantic_pf_loc && source .env
    python3 -u scripts/test_lpips_regression.py
"""

import sys
import os
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
from semantic_pf_loc.observation.lpips_obs import LPIPSObservation
from semantic_pf_loc.datasets.tum import TUMDataset
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.evaluation.metrics import (
    compute_all_metrics, translation_error, rotation_error,
)
from semantic_pf_loc.utils.pose_utils import scale_intrinsics


SCENES = [
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

N_TRIALS = 3
N_FRAMES = 100
N_PARTICLES = 200
DEVICE = "cuda"
PF_W, PF_H = 160, 120
REF_W, REF_H = 320, 240


def run_trial(dataset, gmap, scene_cfg, obs_model, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    K = dataset.get_intrinsics().float().to(DEVICE)
    native_size = (scene_cfg["native_w"], scene_cfg["native_h"])

    renderer = BatchRenderer(gmap, width=PF_W, height=PF_H)
    motion = MotionModel(
        translation_std=scene_cfg["trans_std"],
        rotation_std=scene_cfg["rot_std"],
        device=DEVICE,
    )

    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=N_PARTICLES,
        render_width=PF_W, render_height=PF_H,
        render_width_hires=REF_W, render_height_hires=REF_H,
        convergence_threshold=0.02,
        roughening_trans=0.002, roughening_rot=0.001,
        gradient_refiner=None,
        device=DEVICE,
    )

    hires_renderer = BatchRenderer(gmap, width=REF_W, height=REF_H)
    K_hires = scale_intrinsics(K, native_size, (REF_W, REF_H))
    refiner = GradientRefiner(
        hires_renderer,
        num_iterations=100,
        lr_init=0.01,
        blur_schedule=True,
        blur_sigma_init=10.0,
        blur_sigma_final=0.1,
    )

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

        t0 = time.time()
        est, info = pf.step(obs, K)
        t_pf = time.time() - t0

        raw_est_poses.append(est.matrix().cpu())

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

    raw_stack = torch.stack(raw_est_poses)
    refined_stack = torch.stack(refined_est_poses)
    gt_stack = torch.stack(gt_poses)
    time_tensor = torch.tensor(step_times)

    raw_metrics = compute_all_metrics(raw_stack, gt_stack)
    refined_metrics = compute_all_metrics(refined_stack, gt_stack, time_tensor)

    result = {
        "raw_ate_cm": raw_metrics["ate"]["median"] * 100,
        "refined_ate_cm": refined_metrics["ate"]["median"] * 100,
        "refined_are_deg": refined_metrics["are"]["median"],
        "refined_sr": refined_metrics["success_rate"] * 100,
        "runtime_ms": refined_metrics.get("runtime_mean_ms", 0),
    }

    del renderer, pf, hires_renderer, refiner
    torch.cuda.empty_cache()
    gc.collect()

    return result


def main():
    torch.set_grad_enabled(False)

    start_time = datetime.now()
    print("=" * 80)
    print("  LPIPS vs SSIM REGRESSION CHECK (office0, fr3_office)")
    print(f"  LPIPS t=3.0, SSIM t=3.0, both at {PF_W}x{PF_H}")
    print(f"  {N_TRIALS} trials, {N_FRAMES} frames, {N_PARTICLES} particles")
    print("=" * 80)

    all_results = {}

    for scene_cfg in SCENES:
        scene_name = scene_cfg["name"]
        print(f"\n{'='*70}")
        print(f"  SCENE: {scene_name}")
        print(f"{'='*70}")

        if not Path(scene_cfg["ckpt"]).exists():
            print(f"  SKIP — checkpoint not found")
            continue

        if scene_cfg["type"] == "tum":
            dataset = TUMDataset(scene_cfg["path"], stride=1)
        else:
            dataset = ReplicaDataset(scene_cfg["path"], stride=1)
        gmap = GaussianMap.from_checkpoint(scene_cfg["ckpt"])
        print(f"  Loaded: {gmap.num_gaussians} Gaussians, {len(dataset)} frames")

        scene_results = {}

        # SSIM baseline
        print(f"\n  --- SSIM temp=3.0 ---")
        ssim_model = SSIMObservation(temperature=3.0)
        ssim_ates = []
        for trial in range(N_TRIALS):
            seed = 42 + trial * 1000
            print(f"    Trial {trial+1}/{N_TRIALS} (seed={seed})...", end=" ", flush=True)
            t0 = time.time()
            r = run_trial(dataset, gmap, scene_cfg, ssim_model, seed=seed)
            elapsed = time.time() - t0
            ssim_ates.append(r["refined_ate_cm"])
            print(f"ATE={r['refined_ate_cm']:.2f}cm SR={r['refined_sr']:.0f}% [{elapsed:.0f}s]")
        sorted_idx = sorted(range(N_TRIALS), key=lambda i: ssim_ates[i])
        median_ate = ssim_ates[sorted_idx[1]]
        print(f"  >> SSIM median ATE: {median_ate:.2f}cm")
        scene_results["ssim"] = {"median_ate_cm": median_ate, "all_ates": ssim_ates}
        del ssim_model

        # LPIPS
        print(f"\n  --- LPIPS temp=3.0 ---")
        lpips_model = LPIPSObservation(net="vgg", temperature=3.0, device=DEVICE)
        lpips_ates = []
        for trial in range(N_TRIALS):
            seed = 42 + trial * 1000
            print(f"    Trial {trial+1}/{N_TRIALS} (seed={seed})...", end=" ", flush=True)
            t0 = time.time()
            r = run_trial(dataset, gmap, scene_cfg, lpips_model, seed=seed)
            elapsed = time.time() - t0
            lpips_ates.append(r["refined_ate_cm"])
            print(f"ATE={r['refined_ate_cm']:.2f}cm SR={r['refined_sr']:.0f}% [{elapsed:.0f}s]")
        sorted_idx = sorted(range(N_TRIALS), key=lambda i: lpips_ates[i])
        median_ate = lpips_ates[sorted_idx[1]]
        print(f"  >> LPIPS median ATE: {median_ate:.2f}cm")
        scene_results["lpips"] = {"median_ate_cm": median_ate, "all_ates": lpips_ates}
        del lpips_model
        torch.cuda.empty_cache()

        all_results[scene_name] = scene_results

        del gmap, dataset
        torch.cuda.empty_cache()
        gc.collect()

    # Final summary
    elapsed_total = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON (all at {PF_W}x{PF_H}, median of {N_TRIALS} trials)")
    print(f"{'='*70}")
    print(f"  {'Scene':<14} | {'SSIM ATE':>10} | {'LPIPS ATE':>10} | {'Delta':>10}")
    print(f"  {'-'*54}")

    # Include room0 results from phase 1
    print(f"  {'room0':<14} | {'38.54cm':>10} | {'31.74cm':>10} | {'-6.80cm':>10}")

    for scene_name, sr in all_results.items():
        ssim_ate = sr["ssim"]["median_ate_cm"]
        lpips_ate = sr["lpips"]["median_ate_cm"]
        delta = lpips_ate - ssim_ate
        sign = "+" if delta > 0 else ""
        print(f"  {scene_name:<14} | {ssim_ate:>8.2f}cm | {lpips_ate:>8.2f}cm | {sign}{delta:>7.2f}cm")

    # Save JSON
    output_dir = Path("results/lpips_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "regression_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print(f"Total time: {elapsed_total/60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    main()
