"""Tune gradient refiner hyperparameters for office0.

Tests different configurations of resolution, learning rate, iterations, and
top-k refinement to find the best settings for sub-1cm ATE.

Key idea: after the PF converges, take the weighted mean pose and refine JUST
that single pose at high resolution for many iterations, rather than spreading
compute across 5 particles at low resolution.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pypose as pp
import time
import itertools
from pathlib import Path
from tqdm import tqdm

from semantic_pf_loc.gaussian_map import GaussianMap
from semantic_pf_loc.batch_renderer import BatchRenderer
from semantic_pf_loc.particle_filter import ParticleFilter
from semantic_pf_loc.motion_model import MotionModel
from semantic_pf_loc.gradient_refiner import GradientRefiner
from semantic_pf_loc.observation.ssim import SSIMObservation
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.evaluation.metrics import (
    compute_all_metrics,
    translation_error,
    rotation_error,
    absolute_trajectory_error,
    success_rate,
)
from semantic_pf_loc.utils.pose_utils import (
    weighted_se3_mean,
    scale_intrinsics,
    se3_to_viewmat,
)


NATIVE_W, NATIVE_H = 1200, 680
N_FRAMES = 100
N_PARTICLES = 200
DEVICE = "cuda"


def run_pf_collect_state(dataset, gmap, n_frames=N_FRAMES):
    """Run PF (SSIM, no refinement) and collect per-frame PF state for offline refinement.

    Returns:
        pf_estimates: list of SE3 [7] estimated poses from PF (no refinement)
        pf_particles: list of SE3 [N, 7] particles at each frame
        pf_weights: list of [N] weight tensors at each frame
        gt_poses: list of [4, 4] ground truth pose matrices
        observations: list of {"image": [H, W, 3]} observation dicts
        K: [3, 3] native intrinsics
        converged_flags: list of bool indicating whether PF was converged at each frame
    """
    K = dataset.get_intrinsics().float().to(DEVICE)
    renderer = BatchRenderer(gmap, width=160, height=120)
    motion = MotionModel(translation_std=0.003, rotation_std=0.002, device=DEVICE)
    obs_model = SSIMObservation(temperature=3.0)

    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=N_PARTICLES, render_width=160, render_height=120,
        render_width_hires=320, render_height_hires=240,
        convergence_threshold=0.02, roughening_trans=0.002, roughening_rot=0.001,
        gradient_refiner=None, top_k_refine=5, device=DEVICE,
    )

    sample0 = dataset[0]
    gt_pose0 = pp.mat2SE3(
        sample0["pose"].double().to(DEVICE).unsqueeze(0), check=False
    ).squeeze(0).float()
    pf.initialize_around_pose(gt_pose0, trans_spread=0.03, rot_spread=0.01)

    pf_estimates = []
    pf_particles = []
    pf_weights = []
    gt_poses = []
    observations = []
    converged_flags = []

    for i in tqdm(range(min(n_frames, len(dataset))), desc="PF baseline"):
        sample = dataset[i]
        obs = {"image": sample["image"].float().to(DEVICE)}
        est, info = pf.step(obs, K)

        pf_estimates.append(est)
        pf_particles.append(pp.SE3(pf.particles.tensor().clone()))
        pf_weights.append(pf.weights.clone())
        gt_poses.append(sample["pose"].float())
        observations.append(obs)
        converged_flags.append(info["converged"])

    del renderer
    torch.cuda.empty_cache()

    return pf_estimates, pf_particles, pf_weights, gt_poses, observations, K, converged_flags


def refine_single_pose(gmap, pose_se3, query_image, K_native, render_w, render_h,
                        num_iterations, lr_init, blur_schedule=True,
                        blur_sigma_init=10.0, blur_sigma_final=0.1):
    """Refine a single SE3 pose at given resolution.

    Args:
        pose_se3: SE3 [7] single pose
        query_image: [H, W, 3] native-resolution observation image
        K_native: [3, 3] native intrinsics
        render_w, render_h: resolution for refinement rendering
    Returns:
        refined SE3 [7] pose
    """
    renderer = BatchRenderer(gmap, width=render_w, height=render_h)
    K_scaled = scale_intrinsics(K_native, (NATIVE_W, NATIVE_H), (render_w, render_h))

    refiner = GradientRefiner(
        renderer,
        num_iterations=num_iterations,
        lr_init=lr_init,
        blur_schedule=blur_schedule,
        blur_sigma_init=blur_sigma_init,
        blur_sigma_final=blur_sigma_final,
    )

    pose_batch = pp.SE3(pose_se3.tensor().unsqueeze(0))  # [1, 7]
    with torch.enable_grad():
        refined = refiner.refine(pose_batch, query_image, K_scaled)

    result = refined.squeeze(0)  # [7]

    del renderer, refiner
    torch.cuda.empty_cache()
    return result


def refine_topk_poses(gmap, particles, weights, query_image, K_native,
                      render_w, render_h, top_k, num_iterations, lr_init,
                      blur_schedule=True, blur_sigma_init=10.0, blur_sigma_final=0.1):
    """Refine top-k particles and return best refined pose.

    Args:
        particles: SE3 [N, 7]
        weights: [N]
    Returns:
        refined SE3 [7] best pose (highest weight after re-scoring)
    """
    renderer = BatchRenderer(gmap, width=render_w, height=render_h)
    K_scaled = scale_intrinsics(K_native, (NATIVE_W, NATIVE_H), (render_w, render_h))
    obs_model = SSIMObservation(temperature=3.0)

    refiner = GradientRefiner(
        renderer,
        num_iterations=num_iterations,
        lr_init=lr_init,
        blur_schedule=blur_schedule,
        blur_sigma_init=blur_sigma_init,
        blur_sigma_final=blur_sigma_final,
    )

    top_k_idx = torch.topk(weights, min(top_k, len(weights))).indices
    top_k_poses = pp.SE3(particles.tensor()[top_k_idx])

    with torch.enable_grad():
        refined = refiner.refine(top_k_poses, query_image, K_scaled)

    # Take weighted mean of refined poses (using original top-k weights, renormalized)
    top_k_w = weights[top_k_idx]
    top_k_w = top_k_w / top_k_w.sum()
    result = weighted_se3_mean(refined, top_k_w)

    del renderer, refiner
    torch.cuda.empty_cache()
    return result


def compute_metrics_from_lists(est_se3_list, gt_pose_list):
    """Compute ATE metrics from lists of SE3 estimates and [4,4] GT poses."""
    est_mats = torch.stack([e.matrix().cpu() for e in est_se3_list])
    gt_mats = torch.stack(gt_pose_list)
    ate = absolute_trajectory_error(est_mats, gt_mats)
    sr = success_rate(est_mats, gt_mats, trans_threshold=0.05, rot_threshold=2.0)
    t_err = translation_error(est_mats, gt_mats)
    return ate, sr, t_err


def main():
    torch.set_grad_enabled(False)

    print("=" * 70)
    print("  REFINER TUNING — office0")
    print("=" * 70)

    # Load scene
    print("\nLoading checkpoint and dataset...")
    gmap = GaussianMap.from_checkpoint("checkpoints/office0.ckpt")
    dataset = ReplicaDataset("data/replica/office0", stride=1)

    # Run PF baseline (no refinement) and collect state
    print("\nRunning PF baseline (200 particles, 160x120, SSIM, no refinement)...")
    (pf_estimates, pf_particles, pf_weights,
     gt_poses, observations, K, converged_flags) = run_pf_collect_state(dataset, gmap)

    # Compute baseline metrics
    ate_base, sr_base, t_err_base = compute_metrics_from_lists(pf_estimates, gt_poses)
    print(f"\n--- PF Baseline (no refinement) ---")
    print(f"  ATE median: {ate_base['median']*100:.2f} cm")
    print(f"  ATE mean:   {ate_base['mean']*100:.2f} cm")
    print(f"  Success:    {sr_base*100:.0f}%")

    # =========================================================================
    # Experiment 1: Vary top-k and resolution with inline PF refinement
    # (replicates what the PF does, but with different hyperparams applied
    #  post-hoc to the saved particle states)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Top-K refinement (varying k, resolution, iters, lr)")
    print("=" * 70)

    configs_topk = [
        # (name, top_k, render_w, render_h, num_iters, lr)
        ("baseline_k5_160x120_i20_lr005",    5, 160, 120, 20,  0.005),
        ("k5_160x120_i50_lr005",             5, 160, 120, 50,  0.005),
        ("k5_160x120_i50_lr01",              5, 160, 120, 50,  0.01),
        ("k3_240x180_i50_lr01",              3, 240, 180, 50,  0.01),
        ("k3_320x240_i50_lr01",              3, 320, 240, 50,  0.01),
        ("k1_320x240_i50_lr01",              1, 320, 240, 50,  0.01),
        ("k1_320x240_i100_lr01",             1, 320, 240, 100, 0.01),
        ("k1_320x240_i100_lr02",             1, 320, 240, 100, 0.02),
    ]

    results_topk = []
    for cfg_name, top_k, rw, rh, n_iters, lr in configs_topk:
        print(f"\n  Config: {cfg_name}")
        refined_estimates = []
        t_start = time.time()

        for i in tqdm(range(N_FRAMES), desc=f"    {cfg_name}", leave=False):
            if converged_flags[i]:
                refined = refine_topk_poses(
                    gmap, pf_particles[i], pf_weights[i],
                    observations[i]["image"], K,
                    rw, rh, top_k, n_iters, lr,
                )
                refined_estimates.append(refined)
            else:
                refined_estimates.append(pf_estimates[i])

        elapsed = time.time() - t_start
        ate, sr, t_err = compute_metrics_from_lists(refined_estimates, gt_poses)
        results_topk.append({
            "name": cfg_name, "ate_median": ate["median"], "ate_mean": ate["mean"],
            "success_rate": sr, "time_s": elapsed,
        })
        print(f"    ATE median: {ate['median']*100:.2f} cm | mean: {ate['mean']*100:.2f} cm | SR: {sr*100:.0f}% | time: {elapsed:.1f}s")

    # =========================================================================
    # Experiment 2: Single-pose refinement of the PF weighted mean estimate
    # This is the key insight: take the PF's best guess and refine JUST that
    # one pose at high resolution
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Single-pose refinement of PF weighted mean")
    print("=" * 70)

    configs_single = [
        # (name, render_w, render_h, num_iters, lr)
        ("single_160x120_i50_lr01",    160, 120, 50,  0.01),
        ("single_240x180_i50_lr01",    240, 180, 50,  0.01),
        ("single_320x240_i50_lr005",   320, 240, 50,  0.005),
        ("single_320x240_i50_lr01",    320, 240, 50,  0.01),
        ("single_320x240_i50_lr02",    320, 240, 50,  0.02),
        ("single_320x240_i100_lr005",  320, 240, 100, 0.005),
        ("single_320x240_i100_lr01",   320, 240, 100, 0.01),
        ("single_320x240_i100_lr02",   320, 240, 100, 0.02),
        ("single_480x360_i100_lr01",   480, 360, 100, 0.01),
        ("single_480x360_i100_lr005",  480, 360, 100, 0.005),
    ]

    results_single = []
    for cfg_name, rw, rh, n_iters, lr in configs_single:
        print(f"\n  Config: {cfg_name}")
        refined_estimates = []
        t_start = time.time()

        for i in tqdm(range(N_FRAMES), desc=f"    {cfg_name}", leave=False):
            if converged_flags[i]:
                refined = refine_single_pose(
                    gmap, pf_estimates[i], observations[i]["image"], K,
                    rw, rh, n_iters, lr,
                )
                refined_estimates.append(refined)
            else:
                refined_estimates.append(pf_estimates[i])

        elapsed = time.time() - t_start
        ate, sr, t_err = compute_metrics_from_lists(refined_estimates, gt_poses)
        results_single.append({
            "name": cfg_name, "ate_median": ate["median"], "ate_mean": ate["mean"],
            "success_rate": sr, "time_s": elapsed,
        })
        print(f"    ATE median: {ate['median']*100:.2f} cm | mean: {ate['mean']*100:.2f} cm | SR: {sr*100:.0f}% | time: {elapsed:.1f}s")

    # =========================================================================
    # Experiment 3: Two-stage — refine top-k first, then single-pose refine the mean
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Two-stage (top-k at low-res, then single at high-res)")
    print("=" * 70)

    configs_twostage = [
        # (name, k, low_rw, low_rh, low_iters, low_lr, hi_rw, hi_rh, hi_iters, hi_lr)
        ("2stage_k3_160_i20+320_i50",  3, 160, 120, 20, 0.005, 320, 240, 50, 0.01),
        ("2stage_k5_160_i20+320_i80",  5, 160, 120, 20, 0.005, 320, 240, 80, 0.01),
        ("2stage_k3_240_i30+480_i50",  3, 240, 180, 30, 0.01,  480, 360, 50, 0.005),
    ]

    results_twostage = []
    for (cfg_name, top_k, low_rw, low_rh, low_iters, low_lr,
         hi_rw, hi_rh, hi_iters, hi_lr) in configs_twostage:
        print(f"\n  Config: {cfg_name}")
        refined_estimates = []
        t_start = time.time()

        for i in tqdm(range(N_FRAMES), desc=f"    {cfg_name}", leave=False):
            if converged_flags[i]:
                # Stage 1: refine top-k at low resolution
                stage1 = refine_topk_poses(
                    gmap, pf_particles[i], pf_weights[i],
                    observations[i]["image"], K,
                    low_rw, low_rh, top_k, low_iters, low_lr,
                )
                # Stage 2: refine the stage1 result at high resolution
                stage2 = refine_single_pose(
                    gmap, stage1, observations[i]["image"], K,
                    hi_rw, hi_rh, hi_iters, hi_lr,
                )
                refined_estimates.append(stage2)
            else:
                refined_estimates.append(pf_estimates[i])

        elapsed = time.time() - t_start
        ate, sr, t_err = compute_metrics_from_lists(refined_estimates, gt_poses)
        results_twostage.append({
            "name": cfg_name, "ate_median": ate["median"], "ate_mean": ate["mean"],
            "success_rate": sr, "time_s": elapsed,
        })
        print(f"    ATE median: {ate['median']*100:.2f} cm | mean: {ate['mean']*100:.2f} cm | SR: {sr*100:.0f}% | time: {elapsed:.1f}s")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    all_results = (
        [{"name": "PF_baseline_no_refine", "ate_median": ate_base["median"],
          "ate_mean": ate_base["mean"], "success_rate": sr_base, "time_s": 0}]
        + results_topk + results_single + results_twostage
    )

    # Sort by ATE median
    all_results.sort(key=lambda x: x["ate_median"])

    print(f"\n{'Rank':<5} {'Config':<42} {'ATE Med':>9} {'ATE Mean':>9} {'SR':>6} {'Time':>7}")
    print("-" * 80)
    for rank, r in enumerate(all_results, 1):
        print(f"{rank:<5} {r['name']:<42} {r['ate_median']*100:>7.2f}cm {r['ate_mean']*100:>7.2f}cm {r['success_rate']*100:>5.0f}% {r['time_s']:>6.1f}s")

    # Save results
    output_path = Path("results/refiner_tuning")
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(all_results, output_path / "tune_results_office0.pt")
    print(f"\nResults saved to {output_path / 'tune_results_office0.pt'}")

    # Highlight best
    best = all_results[0]
    print(f"\n>>> BEST CONFIG: {best['name']}")
    print(f"    ATE median: {best['ate_median']*100:.2f} cm")
    print(f"    ATE mean:   {best['ate_mean']*100:.2f} cm")
    print(f"    Success:    {best['success_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
