"""5-trial evaluation on fr3_office (SSIM+Refine) for statistical robustness.

Runs 5 independent PF+Refine evaluations to show the spread is tight.
Reports: all 5 ATE medians, mean, std, min, max.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pypose as pp
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
from semantic_pf_loc.evaluation.metrics import compute_all_metrics
from semantic_pf_loc.utils.pose_utils import scale_intrinsics


SCENE_CFG = {
    "name": "fr3_office",
    "type": "tum",
    "path": "data/tum/rgbd_dataset_freiburg3_long_office_household",
    "ckpt": "checkpoints_depth/fr3_office.ckpt",
    "native_w": 640,
    "native_h": 480,
    "trans_std": 0.005,
    "rot_std": 0.003,
}

DEVICE = "cuda"
N_PARTICLES = 200
N_FRAMES = 100
N_TRIALS = 5


def run_single_trial(dataset, gmap, scene_cfg, seed):
    """Run a single PF+Refine trial. Returns metrics dict."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    K = dataset.get_intrinsics().float().to(DEVICE)
    native_size = (scene_cfg["native_w"], scene_cfg["native_h"])

    # Low-res renderer for PF
    renderer = BatchRenderer(gmap, width=160, height=120)
    motion = MotionModel(
        translation_std=scene_cfg["trans_std"],
        rotation_std=scene_cfg["rot_std"],
        device=DEVICE,
    )
    obs_model = SSIMObservation(temperature=3.0)

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

    # Refiner (post-hoc)
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

    est_poses, gt_poses, times = [], [], []
    n_frames = min(N_FRAMES, len(dataset))

    for i in range(n_frames):
        sample = dataset[i]
        obs = {"image": sample["image"].float().to(DEVICE)}

        est, info = pf.step(obs, K)

        # Post-hoc refinement
        if info["converged"]:
            est_for_refine = pp.SE3(est.tensor().unsqueeze(0))
            with torch.enable_grad():
                refined = refiner.refine(
                    est_for_refine, sample["image"].float().to(DEVICE), K_hires
                )
            est = refined.squeeze(0)

        est_poses.append(est.matrix().cpu())
        gt_poses.append(sample["pose"].float())
        times.append(info["step_time_ms"])

    est_stack = torch.stack(est_poses)
    gt_stack = torch.stack(gt_poses)
    metrics = compute_all_metrics(est_stack, gt_stack, torch.tensor(times))

    # Cleanup
    del renderer, pf, hires_renderer, refiner
    torch.cuda.empty_cache()
    gc.collect()

    return metrics


def main():
    torch.set_grad_enabled(False)

    output_dir = Path("results/five_trials")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    print("=" * 70)
    print(f"  5-TRIAL EVALUATION: fr3_office (SSIM + Refine)")
    print(f"  {N_PARTICLES} particles, {N_FRAMES} frames, {N_TRIALS} trials")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load once
    dataset = TUMDataset(SCENE_CFG["path"], stride=1)
    gmap = GaussianMap.from_checkpoint(SCENE_CFG["ckpt"])
    print(f"Loaded: {gmap.num_gaussians} Gaussians, {len(dataset)} frames\n")

    trial_results = []

    for trial in range(N_TRIALS):
        seed = 42 + trial * 1000
        t0 = time.time()

        print(f"  Trial {trial + 1}/{N_TRIALS} (seed={seed})...")
        m = run_single_trial(dataset, gmap, SCENE_CFG, seed)
        elapsed = time.time() - t0

        ate_med = m["ate"]["median"]
        are_med = m["are"]["median"]
        sr = m["success_rate"]
        rt = m.get("runtime_mean_ms", 0)

        result = {
            "trial": trial + 1,
            "seed": seed,
            "ate_median_m": ate_med,
            "ate_median_cm": ate_med * 100,
            "ate_mean_m": m["ate"]["mean"],
            "ate_rmse_m": m["ate"]["rmse"],
            "are_median_deg": are_med,
            "are_mean_deg": m["are"]["mean"],
            "success_rate": sr,
            "runtime_mean_ms": rt,
            "wall_time_s": elapsed,
            "convergence_frame": m["convergence_frame"],
        }
        trial_results.append(result)

        print(f"    ATE median: {ate_med*100:.2f}cm  "
              f"ARE median: {are_med:.2f}deg  "
              f"SR: {sr*100:.0f}%  "
              f"Time: {elapsed:.1f}s")

    # Summary statistics
    ate_medians = [r["ate_median_m"] for r in trial_results]
    are_medians = [r["are_median_deg"] for r in trial_results]
    success_rates = [r["success_rate"] for r in trial_results]

    import statistics
    ate_mean = statistics.mean(ate_medians)
    ate_std = statistics.stdev(ate_medians) if len(ate_medians) > 1 else 0
    ate_min = min(ate_medians)
    ate_max = max(ate_medians)

    are_mean = statistics.mean(are_medians)
    are_std = statistics.stdev(are_medians) if len(are_medians) > 1 else 0

    sr_mean = statistics.mean(success_rates)
    sr_std = statistics.stdev(success_rates) if len(success_rates) > 1 else 0

    elapsed_total = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*70}")
    print(f"  5-TRIAL SUMMARY: fr3_office (SSIM + Refine)")
    print(f"{'='*70}")
    print(f"\n  Individual ATE medians (cm):")
    for r in trial_results:
        print(f"    Trial {r['trial']}: {r['ate_median_cm']:.2f}cm  "
              f"(ARE: {r['are_median_deg']:.2f}deg, SR: {r['success_rate']*100:.0f}%)")

    print(f"\n  ATE Statistics:")
    print(f"    Mean:   {ate_mean*100:.2f}cm")
    print(f"    Std:    {ate_std*100:.2f}cm")
    print(f"    Min:    {ate_min*100:.2f}cm")
    print(f"    Max:    {ate_max*100:.2f}cm")
    print(f"    Range:  {(ate_max-ate_min)*100:.2f}cm")

    print(f"\n  ARE Statistics:")
    print(f"    Mean:   {are_mean:.2f}deg")
    print(f"    Std:    {are_std:.2f}deg")

    print(f"\n  Success Rate Statistics:")
    print(f"    Mean:   {sr_mean*100:.1f}%")
    print(f"    Std:    {sr_std*100:.1f}%")

    print(f"\n  Total wall time: {elapsed_total/60:.1f} minutes")

    # Save results
    output = {
        "metadata": {
            "scene": "fr3_office",
            "observation_model": "SSIM+Refine",
            "n_particles": N_PARTICLES,
            "n_frames": N_FRAMES,
            "n_trials": N_TRIALS,
            "refiner_settings": "320x240, 100 iters, lr=0.01, blur_schedule",
            "init": "local (trans_spread=0.03, rot_spread=0.01)",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_wall_time_min": round(elapsed_total / 60, 1),
        },
        "trials": trial_results,
        "summary": {
            "ate_median": {
                "values_cm": [round(a * 100, 2) for a in ate_medians],
                "mean_cm": round(ate_mean * 100, 2),
                "std_cm": round(ate_std * 100, 2),
                "min_cm": round(ate_min * 100, 2),
                "max_cm": round(ate_max * 100, 2),
                "range_cm": round((ate_max - ate_min) * 100, 2),
            },
            "are_median": {
                "values_deg": [round(a, 2) for a in are_medians],
                "mean_deg": round(are_mean, 2),
                "std_deg": round(are_std, 2),
            },
            "success_rate": {
                "values_pct": [round(s * 100, 1) for s in success_rates],
                "mean_pct": round(sr_mean * 100, 1),
                "std_pct": round(sr_std * 100, 1),
            },
        },
    }

    json_path = output_dir / "five_trial_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
