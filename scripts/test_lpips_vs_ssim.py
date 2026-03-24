"""Compare LPIPS vs SSIM observation models on room0, office0, fr3_office.

Runs the PF with post-hoc refinement (same setup as run_depthinit_evaluation.py)
and compares the two observation models head-to-head.

Usage:
    cd /home/anywherevla/semantic_pf_loc && source .env
    python scripts/test_lpips_vs_ssim.py
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
from semantic_pf_loc.observation.lpips_obs import LPIPSObservation
from semantic_pf_loc.datasets.tum import TUMDataset
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.evaluation.metrics import (
    compute_all_metrics, translation_error, rotation_error,
)
from semantic_pf_loc.utils.pose_utils import scale_intrinsics


# ============================================================================
# Scene configurations
# ============================================================================

SCENES = {
    "room0": {
        "type": "replica",
        "path": "data/replica/room0",
        "ckpt": "checkpoints_depthinit/room0.ckpt",
        "native_w": 1200, "native_h": 680,
        "trans_std": 0.003, "rot_std": 0.002,
        "ssim_baseline_ate_cm": 38.5,
    },
    "office0": {
        "type": "replica",
        "path": "data/replica/office0",
        "ckpt": "checkpoints_depthinit/office0.ckpt",
        "native_w": 1200, "native_h": 680,
        "trans_std": 0.003, "rot_std": 0.002,
        "ssim_baseline_ate_cm": 1.4,
    },
    "fr3_office": {
        "type": "tum",
        "path": "data/tum/rgbd_dataset_freiburg3_long_office_household",
        "ckpt": "checkpoints_depthinit/fr3_office.ckpt",
        "native_w": 640, "native_h": 480,
        "trans_std": 0.005, "rot_std": 0.003,
        "ssim_baseline_ate_cm": 0.6,
    },
}

N_TRIALS = 3
N_FRAMES = 100
N_PARTICLES = 200
DEVICE = "cuda"


# ============================================================================
# Core evaluation function
# ============================================================================

def run_single_trial(dataset, gmap, scene_cfg, obs_model, pf_width, pf_height,
                     refine_width=320, refine_height=240, seed=None):
    """Run one PF trial with given observation model and resolution.

    Returns dict with raw and refined metrics.
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    K = dataset.get_intrinsics().float().to(DEVICE)
    native_size = (scene_cfg["native_w"], scene_cfg["native_h"])

    # Low-res renderer for PF
    renderer = BatchRenderer(gmap, width=pf_width, height=pf_height)
    motion = MotionModel(
        translation_std=scene_cfg["trans_std"],
        rotation_std=scene_cfg["rot_std"],
        device=DEVICE,
    )

    # PF WITHOUT any built-in refiner
    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=N_PARTICLES,
        render_width=pf_width, render_height=pf_height,
        render_width_hires=refine_width, render_height_hires=refine_height,
        convergence_threshold=0.02,
        roughening_trans=0.002, roughening_rot=0.001,
        gradient_refiner=None,
        device=DEVICE,
    )

    # Separate hi-res renderer + refiner for post-hoc refinement
    hires_renderer = BatchRenderer(gmap, width=refine_width, height=refine_height)
    K_hires = scale_intrinsics(K, native_size, (refine_width, refine_height))
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

    raw_est_poses, refined_est_poses, gt_poses = [], [], []
    step_times = []

    for i in range(min(N_FRAMES, len(dataset))):
        sample = dataset[i]
        obs = {"image": sample["image"].float().to(DEVICE)}

        t0 = time.time()
        est, info = pf.step(obs, K)
        t_pf = time.time() - t0

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
        step_times.append((t_pf + t_ref) * 1000)

    raw_stack = torch.stack(raw_est_poses)
    refined_stack = torch.stack(refined_est_poses)
    gt_stack = torch.stack(gt_poses)
    time_tensor = torch.tensor(step_times)

    raw_metrics = compute_all_metrics(raw_stack, gt_stack)
    refined_metrics = compute_all_metrics(refined_stack, gt_stack, time_tensor)

    result = {
        "raw_ate_cm": raw_metrics["ate"]["median"] * 100,
        "raw_are_deg": raw_metrics["are"]["median"],
        "raw_sr": raw_metrics["success_rate"] * 100,
        "refined_ate_cm": refined_metrics["ate"]["median"] * 100,
        "refined_are_deg": refined_metrics["are"]["median"],
        "refined_sr": refined_metrics["success_rate"] * 100,
        "runtime_mean_ms": refined_metrics.get("runtime_mean_ms", 0),
    }

    del renderer, pf, hires_renderer, refiner
    torch.cuda.empty_cache()
    gc.collect()

    return result


def run_multi_trial(dataset, gmap, scene_cfg, obs_model, pf_width, pf_height,
                    label="", n_trials=N_TRIALS):
    """Run multiple trials and return median result."""
    results = []
    for trial in range(n_trials):
        seed = 42 + trial * 1000
        print(f"    {label} trial {trial+1}/{n_trials} (seed={seed})...", end=" ")
        t0 = time.time()
        r = run_single_trial(dataset, gmap, scene_cfg, obs_model,
                             pf_width, pf_height, seed=seed)
        elapsed = time.time() - t0
        print(f"raw={r['raw_ate_cm']:.1f}cm  refined={r['refined_ate_cm']:.2f}cm  "
              f"SR={r['refined_sr']:.0f}%  [{elapsed:.1f}s]")
        results.append(r)

    # Median by refined ATE
    sorted_idx = sorted(range(n_trials), key=lambda i: results[i]["refined_ate_cm"])
    median_idx = sorted_idx[n_trials // 2]
    median_result = results[median_idx]
    median_result["all_trials"] = results
    median_result["median_trial_idx"] = median_idx
    return median_result


# ============================================================================
# Main
# ============================================================================

def main():
    torch.set_grad_enabled(False)

    output_dir = Path("results/lpips_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    print("=" * 80)
    print("  LPIPS vs SSIM OBSERVATION MODEL COMPARISON")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Trials: {N_TRIALS}, Frames: {N_FRAMES}, Particles: {N_PARTICLES}")
    print("=" * 80)

    # Pre-load LPIPS model once (it downloads VGG weights on first use)
    print("\nLoading LPIPS model...", end=" ")
    t0 = time.time()
    lpips_model_t5 = LPIPSObservation(net="vgg", temperature=5.0, device=DEVICE)
    print(f"done ({time.time()-t0:.1f}s)")

    all_results = {}

    # ================================================================
    # Phase 1: Room0 — temperature sweep for LPIPS
    # ================================================================
    scene_name = "room0"
    scene_cfg = SCENES[scene_name]
    print(f"\n{'='*80}")
    print(f"  PHASE 1: {scene_name} — LPIPS temperature sweep + SSIM baseline")
    print(f"  SSIM baseline ATE: {scene_cfg['ssim_baseline_ate_cm']} cm")
    print(f"{'='*80}")

    if not Path(scene_cfg["ckpt"]).exists():
        print(f"  SKIP — checkpoint not found: {scene_cfg['ckpt']}")
    else:
        if scene_cfg["type"] == "tum":
            dataset = TUMDataset(scene_cfg["path"], stride=1)
        else:
            dataset = ReplicaDataset(scene_cfg["path"], stride=1)
        gmap = GaussianMap.from_checkpoint(scene_cfg["ckpt"])
        print(f"  Loaded: {gmap.num_gaussians} Gaussians, {len(dataset)} frames")

        scene_results = {}

        # --- SSIM baseline (160x120) ---
        print(f"\n  [SSIM temp=3.0, 160x120]")
        ssim_model = SSIMObservation(temperature=3.0)
        r = run_multi_trial(dataset, gmap, scene_cfg, ssim_model, 160, 120,
                            label="SSIM-160x120")
        scene_results["ssim_t3_160x120"] = r
        print(f"  >> SSIM median: ATE={r['refined_ate_cm']:.2f}cm SR={r['refined_sr']:.0f}%")
        del ssim_model

        # --- LPIPS temperature sweep (160x120) ---
        for temp in [3.0, 5.0, 10.0]:
            label = f"LPIPS-t{temp}-160x120"
            print(f"\n  [LPIPS temp={temp}, 160x120]")
            lpips_model = LPIPSObservation(net="vgg", temperature=temp, device=DEVICE)
            r = run_multi_trial(dataset, gmap, scene_cfg, lpips_model, 160, 120,
                                label=label)
            scene_results[f"lpips_t{temp}_160x120"] = r
            print(f"  >> LPIPS t={temp} median: ATE={r['refined_ate_cm']:.2f}cm "
                  f"SR={r['refined_sr']:.0f}%")
            del lpips_model
            torch.cuda.empty_cache()

        # --- Best LPIPS temperature at 320x240 ---
        # Find best temp from 160x120 results
        lpips_results = {k: v for k, v in scene_results.items() if k.startswith("lpips")}
        best_key = min(lpips_results, key=lambda k: lpips_results[k]["refined_ate_cm"])
        best_temp = float(best_key.split("_t")[1].split("_")[0])
        print(f"\n  Best LPIPS temperature at 160x120: {best_temp} "
              f"(ATE={lpips_results[best_key]['refined_ate_cm']:.2f}cm)")

        print(f"\n  [LPIPS temp={best_temp}, 320x240]")
        lpips_hires = LPIPSObservation(net="vgg", temperature=best_temp, device=DEVICE)
        r = run_multi_trial(dataset, gmap, scene_cfg, lpips_hires, 320, 240,
                            label=f"LPIPS-t{best_temp}-320x240")
        scene_results[f"lpips_t{best_temp}_320x240"] = r
        print(f"  >> LPIPS hires median: ATE={r['refined_ate_cm']:.2f}cm "
              f"SR={r['refined_sr']:.0f}%")
        del lpips_hires
        torch.cuda.empty_cache()

        all_results[scene_name] = scene_results

        # Print room0 summary table
        print(f"\n  {'='*70}")
        print(f"  ROOM0 SUMMARY")
        print(f"  {'='*70}")
        print(f"  {'Config':<30} | {'ATE (cm)':>10} | {'SR':>6} | {'Runtime':>10}")
        print(f"  {'-'*70}")
        for key, res in scene_results.items():
            print(f"  {key:<30} | {res['refined_ate_cm']:>8.2f}cm | {res['refined_sr']:>5.0f}% | "
                  f"{res['runtime_mean_ms']:>8.0f}ms")

        # Decide if we should continue to other scenes
        # Continue if any LPIPS config improved over SSIM
        ssim_ate = scene_results["ssim_t3_160x120"]["refined_ate_cm"]
        any_lpips_better = any(
            v["refined_ate_cm"] < ssim_ate
            for k, v in scene_results.items() if k.startswith("lpips")
        )

        del gmap, dataset
        torch.cuda.empty_cache()
        gc.collect()

    # ================================================================
    # Phase 2: If LPIPS improved room0, test office0 and fr3_office
    # ================================================================
    if any_lpips_better:
        # Use the overall best LPIPS config from room0
        best_lpips_key = min(
            (k for k in all_results["room0"] if k.startswith("lpips")),
            key=lambda k: all_results["room0"][k]["refined_ate_cm"]
        )
        best_temp = float(best_lpips_key.split("_t")[1].split("_")[0])
        best_res = best_lpips_key.split("_")[-1]  # e.g., "160x120" or "320x240"
        pf_w = int(best_res.split("x")[0])
        pf_h = int(best_res.split("x")[1])

        print(f"\n{'='*80}")
        print(f"  PHASE 2: Regression check on office0 & fr3_office")
        print(f"  Using best config: LPIPS temp={best_temp}, {pf_w}x{pf_h}")
        print(f"{'='*80}")

        for scene_name in ["office0", "fr3_office"]:
            scene_cfg = SCENES[scene_name]
            print(f"\n  --- {scene_name} ---")

            if not Path(scene_cfg["ckpt"]).exists():
                print(f"  SKIP — checkpoint not found: {scene_cfg['ckpt']}")
                continue

            if scene_cfg["type"] == "tum":
                dataset = TUMDataset(scene_cfg["path"], stride=1)
            else:
                dataset = ReplicaDataset(scene_cfg["path"], stride=1)
            gmap = GaussianMap.from_checkpoint(scene_cfg["ckpt"])
            print(f"  Loaded: {gmap.num_gaussians} Gaussians, {len(dataset)} frames")

            scene_results = {}

            # SSIM baseline
            print(f"\n  [SSIM temp=3.0, 160x120]")
            ssim_model = SSIMObservation(temperature=3.0)
            r = run_multi_trial(dataset, gmap, scene_cfg, ssim_model, 160, 120,
                                label="SSIM-160x120")
            scene_results["ssim_t3_160x120"] = r
            print(f"  >> SSIM median: ATE={r['refined_ate_cm']:.2f}cm SR={r['refined_sr']:.0f}%")
            del ssim_model

            # LPIPS with best config
            print(f"\n  [LPIPS temp={best_temp}, {pf_w}x{pf_h}]")
            lpips_model = LPIPSObservation(net="vgg", temperature=best_temp, device=DEVICE)
            r = run_multi_trial(dataset, gmap, scene_cfg, lpips_model, pf_w, pf_h,
                                label=f"LPIPS-t{best_temp}")
            scene_results[f"lpips_t{best_temp}_{pf_w}x{pf_h}"] = r
            print(f"  >> LPIPS median: ATE={r['refined_ate_cm']:.2f}cm SR={r['refined_sr']:.0f}%")
            del lpips_model
            torch.cuda.empty_cache()

            all_results[scene_name] = scene_results

            del gmap, dataset
            torch.cuda.empty_cache()
            gc.collect()
    else:
        print(f"\n  LPIPS did NOT improve room0 over SSIM. Skipping other scenes.")

    # ================================================================
    # Final summary
    # ================================================================
    elapsed_total = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"  {'Scene':<12} | {'Model':<30} | {'ATE (cm)':>10} | {'SR':>6} | {'ms/step':>8}")
    print(f"  {'-'*76}")

    for scene_name, scene_results in all_results.items():
        for key, res in scene_results.items():
            print(f"  {scene_name:<12} | {key:<30} | {res['refined_ate_cm']:>8.2f}cm | "
                  f"{res['refined_sr']:>5.0f}% | {res['runtime_mean_ms']:>6.0f}ms")
        print(f"  {'-'*76}")

    # Save results
    json_path = output_dir / "lpips_vs_ssim.json"
    # Convert for JSON serialization (remove per-trial detail for cleaner output)
    json_results = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_trials": N_TRIALS,
            "n_frames": N_FRAMES,
            "n_particles": N_PARTICLES,
        },
        "scenes": {},
    }
    for scene_name, scene_results in all_results.items():
        json_results["scenes"][scene_name] = {}
        for key, res in scene_results.items():
            json_results["scenes"][scene_name][key] = {
                "refined_ate_cm": res["refined_ate_cm"],
                "refined_are_deg": res["refined_are_deg"],
                "refined_sr": res["refined_sr"],
                "raw_ate_cm": res["raw_ate_cm"],
                "raw_sr": res["raw_sr"],
                "runtime_mean_ms": res["runtime_mean_ms"],
                "per_trial": [
                    {
                        "refined_ate_cm": t["refined_ate_cm"],
                        "refined_sr": t["refined_sr"],
                        "raw_ate_cm": t["raw_ate_cm"],
                    }
                    for t in res["all_trials"]
                ],
            }

    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print(f"Total time: {elapsed_total/60:.1f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    main()
