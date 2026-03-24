"""Evaluation of depth-initialized 3DGS checkpoints for localization.

These checkpoints have dramatically better PSNR:
  office0:    29.5 -> 37.9 dB
  room0:      23.7 -> 32.3 dB
  room1:      28.4 -> 34.7 dB
  fr3_office: 23.0 -> 25.0 dB
  fr1_desk:   22.2 -> 22.4 dB

All 5 scenes, SSIM + tuned post-hoc refinement (320x240, 100 iters, lr=0.01).
3 trials per scene for stability.
"""

import sys
import os
# Force unbuffered stdout/stderr
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
from semantic_pf_loc.datasets.tum import TUMDataset
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.evaluation.metrics import (
    compute_all_metrics, translation_error, rotation_error,
)
from semantic_pf_loc.utils.pose_utils import scale_intrinsics

# ============================================================================
# Scene configurations — ALL use depthinit checkpoints
# ============================================================================

SCENES = [
    {
        "name": "office0", "type": "replica",
        "path": "data/replica/office0",
        "ckpt": "checkpoints_depthinit/office0.ckpt",
        "native_w": 1200, "native_h": 680,
        "trans_std": 0.003, "rot_std": 0.002,
        "old_ate": 1.4, "old_sr": 75, "psnr_old": 29.5, "psnr_new": 37.9,
    },
    {
        "name": "room0", "type": "replica",
        "path": "data/replica/room0",
        "ckpt": "checkpoints_depthinit/room0.ckpt",
        "native_w": 1200, "native_h": 680,
        "trans_std": 0.003, "rot_std": 0.002,
        "old_ate": 41.8, "old_sr": 29, "psnr_old": 23.7, "psnr_new": 32.3,
    },
    {
        "name": "room1", "type": "replica",
        "path": "data/replica/room1",
        "ckpt": "checkpoints_depthinit/room1.ckpt",
        "native_w": 1200, "native_h": 680,
        "trans_std": 0.003, "rot_std": 0.002,
        "old_ate": 80.6, "old_sr": 7, "psnr_old": 28.4, "psnr_new": 34.7,
    },
    {
        "name": "fr3_office", "type": "tum",
        "path": "data/tum/rgbd_dataset_freiburg3_long_office_household",
        "ckpt": "checkpoints_depthinit/fr3_office.ckpt",
        "native_w": 640, "native_h": 480,
        "trans_std": 0.005, "rot_std": 0.003,
        "old_ate": 0.6, "old_sr": 94, "psnr_old": 23.0, "psnr_new": 25.0,
    },
    {
        "name": "fr1_desk", "type": "tum",
        "path": "data/tum/rgbd_dataset_freiburg1_desk",
        "ckpt": "checkpoints_depthinit/fr1_desk.ckpt",
        "native_w": 640, "native_h": 480,
        "trans_std": 0.005, "rot_std": 0.003,
        "old_ate": 66.4, "old_sr": 10, "psnr_old": 22.2, "psnr_new": 22.4,
    },
]

N_TRIALS = 3
N_FRAMES = 100
N_PARTICLES = 200
DEVICE = "cuda"


# ============================================================================
# Core evaluation function
# ============================================================================

def run_pf_with_tuned_refine(dataset, gmap, scene_cfg, seed=None):
    """Run a single PF trial with post-hoc single-pose refinement.

    Settings: SSIM temp=3.0, 200 particles, 160x120 PF, 320x240 refiner,
    100 iterations, lr=0.01, blur schedule.
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    K = dataset.get_intrinsics().float().to(DEVICE)
    native_size = (scene_cfg["native_w"], scene_cfg["native_h"])

    # Low-res renderer for PF
    renderer = BatchRenderer(gmap, width=160, height=120)
    obs_model = SSIMObservation(temperature=3.0)
    motion = MotionModel(
        translation_std=scene_cfg["trans_std"],
        rotation_std=scene_cfg["rot_std"],
        device=DEVICE,
    )

    # PF WITHOUT any built-in refiner
    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=N_PARTICLES,
        render_width=160, render_height=120,
        render_width_hires=320, render_height_hires=240,
        convergence_threshold=0.02,
        roughening_trans=0.002, roughening_rot=0.001,
        gradient_refiner=None,
        device=DEVICE,
    )

    # Separate hi-res renderer + refiner for post-hoc refinement
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

    # Initialize around GT pose 0
    sample0 = dataset[0]
    gt_pose0 = pp.mat2SE3(
        sample0["pose"].double().to(DEVICE).unsqueeze(0), check=False
    ).squeeze(0).float()
    pf.initialize_around_pose(gt_pose0, trans_spread=0.03, rot_spread=0.01)

    # Storage for per-frame results
    raw_est_poses, refined_est_poses, gt_poses = [], [], []
    step_times = []

    for i in range(min(N_FRAMES, len(dataset))):
        sample = dataset[i]
        obs = {"image": sample["image"].float().to(DEVICE)}

        t0 = time.time()
        est, info = pf.step(obs, K)
        t_pf = time.time() - t0

        # Record raw PF estimate
        raw_est_poses.append(est.matrix().cpu())

        # Post-hoc single-pose refinement
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
        step_times.append((t_pf + t_ref) * 1000)  # ms

    # Stack tensors
    raw_stack = torch.stack(raw_est_poses)
    refined_stack = torch.stack(refined_est_poses)
    gt_stack = torch.stack(gt_poses)
    time_tensor = torch.tensor(step_times)

    # Compute metrics for both raw and refined
    raw_metrics = compute_all_metrics(raw_stack, gt_stack)
    refined_metrics = compute_all_metrics(refined_stack, gt_stack, time_tensor)

    # Per-frame errors
    raw_trans_err = translation_error(raw_stack, gt_stack)
    raw_rot_err = rotation_error(raw_stack, gt_stack)
    ref_trans_err = translation_error(refined_stack, gt_stack)
    ref_rot_err = rotation_error(refined_stack, gt_stack)

    result = {
        "raw": {
            "ate": raw_metrics["ate"],
            "are": raw_metrics["are"],
            "success_rate": raw_metrics["success_rate"],
            "trans_errors": raw_trans_err.tolist(),
            "rot_errors": raw_rot_err.tolist(),
        },
        "refined": {
            "ate": refined_metrics["ate"],
            "are": refined_metrics["are"],
            "success_rate": refined_metrics["success_rate"],
            "runtime_mean_ms": refined_metrics.get("runtime_mean_ms", 0),
            "trans_errors": ref_trans_err.tolist(),
            "rot_errors": ref_rot_err.tolist(),
        },
    }

    # Cleanup
    del renderer, pf, hires_renderer, refiner, obs_model
    torch.cuda.empty_cache()
    gc.collect()

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    torch.set_grad_enabled(False)

    output_dir = Path("results/depthinit_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    print("=" * 78)
    print("  DEPTH-INITIALIZED 3DGS LOCALIZATION EVALUATION")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Trials per scene: {N_TRIALS}")
    print(f"  Frames per trial: {N_FRAMES}")
    print(f"  Particles: {N_PARTICLES}")
    print(f"  Observation: SSIM (temperature=3.0)")
    print(f"  Refiner: 320x240, 100 iters, lr=0.01, blur schedule")
    print("=" * 78)

    all_scene_results = {}

    for scene_cfg in SCENES:
        scene_name = scene_cfg["name"]
        print(f"\n{'='*78}")
        print(f"  SCENE: {scene_name}")
        print(f"  Checkpoint: {scene_cfg['ckpt']}")
        print(f"  PSNR: {scene_cfg['psnr_old']:.1f} -> {scene_cfg['psnr_new']:.1f} dB")
        print(f"  Old results: ATE={scene_cfg['old_ate']:.1f}cm, SR={scene_cfg['old_sr']}%")
        print(f"{'='*78}")

        # Verify checkpoint
        if not Path(scene_cfg["ckpt"]).exists():
            print(f"  SKIP - checkpoint not found: {scene_cfg['ckpt']}")
            continue

        # Load dataset
        if scene_cfg["type"] == "tum":
            dataset = TUMDataset(scene_cfg["path"], stride=1)
        else:
            dataset = ReplicaDataset(scene_cfg["path"], stride=1)

        # Load 3DGS map
        gmap = GaussianMap.from_checkpoint(scene_cfg["ckpt"])
        print(f"  Loaded: {gmap.num_gaussians} Gaussians, dataset: {len(dataset)} frames")

        trial_results = []
        trial_ate_refined = []
        trial_sr_refined = []

        for trial in range(N_TRIALS):
            seed = 42 + trial * 1000
            print(f"\n  Trial {trial+1}/{N_TRIALS} (seed={seed})...")
            t0 = time.time()

            result = run_pf_with_tuned_refine(dataset, gmap, scene_cfg, seed=seed)
            elapsed = time.time() - t0

            raw_ate = result["raw"]["ate"]["median"] * 100
            raw_sr = result["raw"]["success_rate"] * 100
            ref_ate = result["refined"]["ate"]["median"] * 100
            ref_are = result["refined"]["are"]["median"]
            ref_sr = result["refined"]["success_rate"] * 100
            ref_rt = result["refined"]["runtime_mean_ms"]

            print(f"    Raw PF:   ATE={raw_ate:.2f}cm  SR={raw_sr:.0f}%")
            print(f"    Refined:  ATE={ref_ate:.2f}cm  ARE={ref_are:.2f}deg  "
                  f"SR={ref_sr:.0f}%  ({ref_rt:.0f}ms/step)  [{elapsed:.1f}s total]")

            trial_results.append(result)
            trial_ate_refined.append(result["refined"]["ate"]["median"])
            trial_sr_refined.append(result["refined"]["success_rate"])

        # Take median across trials (index 1 of sorted list of 3)
        sorted_idx = sorted(range(N_TRIALS), key=lambda i: trial_ate_refined[i])
        median_idx = sorted_idx[1]

        median_result = trial_results[median_idx]
        median_ate = trial_ate_refined[median_idx] * 100
        median_sr = trial_sr_refined[median_idx] * 100

        print(f"\n  >> MEDIAN TRIAL (trial {median_idx+1}):")
        print(f"     Raw PF:   ATE={median_result['raw']['ate']['median']*100:.2f}cm  "
              f"SR={median_result['raw']['success_rate']*100:.0f}%")
        print(f"     Refined:  ATE={median_ate:.2f}cm  SR={median_sr:.0f}%")
        print(f"     Old:      ATE={scene_cfg['old_ate']:.1f}cm  SR={scene_cfg['old_sr']}%")

        improvement = scene_cfg['old_ate'] - median_ate
        if improvement > 0:
            print(f"     IMPROVEMENT: {improvement:.1f}cm better!")
        else:
            print(f"     REGRESSION: {-improvement:.1f}cm worse")

        all_scene_results[scene_name] = {
            "trials": [
                {
                    "raw_ate_median": r["raw"]["ate"]["median"],
                    "raw_are_median": r["raw"]["are"]["median"],
                    "raw_sr": r["raw"]["success_rate"],
                    "refined_ate_median": r["refined"]["ate"]["median"],
                    "refined_are_median": r["refined"]["are"]["median"],
                    "refined_sr": r["refined"]["success_rate"],
                    "refined_runtime_ms": r["refined"]["runtime_mean_ms"],
                    "raw_trans_errors": r["raw"]["trans_errors"],
                    "raw_rot_errors": r["raw"]["rot_errors"],
                    "refined_trans_errors": r["refined"]["trans_errors"],
                    "refined_rot_errors": r["refined"]["rot_errors"],
                }
                for r in trial_results
            ],
            "median_trial_idx": median_idx,
            "median_refined_ate": trial_ate_refined[median_idx],
            "median_refined_sr": trial_sr_refined[median_idx],
            "psnr_old": scene_cfg["psnr_old"],
            "psnr_new": scene_cfg["psnr_new"],
            "old_ate_cm": scene_cfg["old_ate"],
            "old_sr_pct": scene_cfg["old_sr"],
        }

        # Cleanup scene
        del gmap, dataset
        torch.cuda.empty_cache()
        gc.collect()

    # ================================================================
    # Save JSON results
    # ================================================================
    json_results = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Depth-initialized 3DGS checkpoint evaluation",
            "n_trials": N_TRIALS,
            "n_frames": N_FRAMES,
            "n_particles": N_PARTICLES,
            "observation": "SSIM (temperature=3.0)",
            "refiner": "320x240, 100 iters, lr=0.01, blur_schedule",
            "init": "local (trans_spread=0.03, rot_spread=0.01)",
        },
        "scenes": all_scene_results,
    }

    json_path = output_dir / "depthinit_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ================================================================
    # Print comparison table
    # ================================================================
    elapsed_total = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*96}")
    print(f"  DEPTH-INIT vs OLD CHECKPOINTS — COMPARISON TABLE")
    print(f"  (median of {N_TRIALS} trials, SSIM+Refine, 200 particles, 100 frames)")
    print(f"{'='*96}")
    print(f"{'Scene':<14} | {'Old ATE':>10} | {'New ATE':>10} | {'Old SR':>8} | {'New SR':>8} | "
          f"{'Raw ATE':>10} | {'PSNR old->new':>16}")
    print("-" * 96)

    for scene_cfg in SCENES:
        sn = scene_cfg["name"]
        if sn not in all_scene_results:
            print(f"{sn:<14} | {'SKIPPED':>10} |")
            continue

        sr = all_scene_results[sn]
        median_idx = sr["median_trial_idx"]
        new_ate = sr["median_refined_ate"] * 100
        new_sr = sr["median_refined_sr"] * 100
        raw_ate = sr["trials"][median_idx]["raw_ate_median"] * 100

        old_ate_str = f"{scene_cfg['old_ate']:.1f} cm"
        new_ate_str = f"{new_ate:.1f} cm"
        old_sr_str = f"{scene_cfg['old_sr']}%"
        new_sr_str = f"{new_sr:.0f}%"
        raw_ate_str = f"{raw_ate:.1f} cm"
        psnr_str = f"{scene_cfg['psnr_old']:.1f} -> {scene_cfg['psnr_new']:.1f}"

        print(f"{sn:<14} | {old_ate_str:>10} | {new_ate_str:>10} | {old_sr_str:>8} | "
              f"{new_sr_str:>8} | {raw_ate_str:>10} | {psnr_str:>16}")

    print("-" * 96)

    # Summary statistics
    n_success_old = sum(1 for s in SCENES if s["old_sr"] >= 50)
    n_success_new = sum(1 for s in SCENES
                        if s["name"] in all_scene_results
                        and all_scene_results[s["name"]]["median_refined_sr"] >= 0.5)
    print(f"\nScenes with SR >= 50%:  Old: {n_success_old}/5  New: {n_success_new}/5")

    # Per-trial detail
    print(f"\n{'='*78}")
    print(f"  PER-TRIAL DETAIL (Refined ATE in cm)")
    print(f"{'='*78}")
    print(f"{'Scene':<14} | {'Trial 1':>10} | {'Trial 2':>10} | {'Trial 3':>10} | {'Median':>10}")
    print("-" * 62)
    for scene_cfg in SCENES:
        sn = scene_cfg["name"]
        if sn not in all_scene_results:
            continue
        sr = all_scene_results[sn]
        ates = [t["refined_ate_median"] * 100 for t in sr["trials"]]
        median_ate = sr["median_refined_ate"] * 100
        print(f"{sn:<14} | {ates[0]:>8.2f}cm | {ates[1]:>8.2f}cm | {ates[2]:>8.2f}cm | {median_ate:>8.2f}cm")

    print(f"\nTotal time: {elapsed_total/60:.1f} minutes")
    print(f"Output: {output_dir}/")
    print("=" * 78)


if __name__ == "__main__":
    main()
