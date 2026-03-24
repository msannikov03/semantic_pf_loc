"""Timing breakdown of the particle filter pipeline.

Instruments each PF component to measure average time:
  - Resampling
  - Roughening
  - Motion prediction
  - Rendering (the big one)
  - Observation model weight computation
  - Pose estimation (weighted mean)
  - Gradient refinement (when applied)

Runs on fr3_office with SSIM+Refine for 100 frames.
Saves results as JSON table.
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

from semantic_pf_loc.gaussian_map import GaussianMap
from semantic_pf_loc.batch_renderer import BatchRenderer
from semantic_pf_loc.motion_model import MotionModel
from semantic_pf_loc.gradient_refiner import GradientRefiner
from semantic_pf_loc.observation.ssim import SSIMObservation
from semantic_pf_loc.datasets.tum import TUMDataset
from semantic_pf_loc.resampling import (
    systematic_resample,
    effective_sample_size,
    normalize_log_weights,
)
from semantic_pf_loc.utils.pose_utils import (
    weighted_se3_mean,
    scale_intrinsics,
    se3_to_viewmat,
)


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
SEED = 42


def run_instrumented_pf():
    """Run PF with timing for each component."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Load
    dataset = TUMDataset(SCENE_CFG["path"], stride=1)
    gmap = GaussianMap.from_checkpoint(SCENE_CFG["ckpt"])
    print(f"Loaded: {gmap.num_gaussians} Gaussians, {len(dataset)} frames")

    K = dataset.get_intrinsics().float().to(DEVICE)
    native_size = (SCENE_CFG["native_w"], SCENE_CFG["native_h"])

    # PF components
    renderer = BatchRenderer(gmap, width=160, height=120)
    motion = MotionModel(
        translation_std=SCENE_CFG["trans_std"],
        rotation_std=SCENE_CFG["rot_std"],
        device=DEVICE,
    )
    obs_model = SSIMObservation(temperature=3.0)

    # Refiner
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

    # State
    N = N_PARTICLES
    convergence_threshold = 0.02
    roughening_trans = 0.002
    roughening_rot = 0.001

    # Timing accumulators
    timings = {
        "resampling": [],
        "roughening": [],
        "motion_prediction": [],
        "rendering": [],
        "observation_weights": [],
        "pose_estimation": [],
        "gradient_refinement": [],
        "total_step": [],
        "obs_prepare": [],
    }

    n_refine_steps = 0

    # Initialize
    sample0 = dataset[0]
    gt_pose0 = pp.mat2SE3(
        sample0["pose"].double().to(DEVICE).unsqueeze(0), check=False
    ).squeeze(0).float()

    # Initialize particles
    noise = torch.zeros(N, 6, device=DEVICE)
    noise[:, :3] = torch.randn(N, 3, device=DEVICE) * 0.03
    noise[:, 3:] = torch.randn(N, 3, device=DEVICE) * 0.01
    base = gt_pose0.unsqueeze(0).expand(N, -1)
    particles = pp.se3(noise).Exp() @ pp.SE3(base)
    weights = torch.ones(N, device=DEVICE) / N
    converged = False

    n_frames = min(N_FRAMES, len(dataset))
    print(f"\nRunning {n_frames} instrumented PF steps...")

    for step_i in range(n_frames):
        t_total_start = time.time()
        torch.cuda.synchronize()

        sample = dataset[step_i]
        query_image = sample["image"].float().to(DEVICE)
        obs = {"image": query_image}

        # 1. RESAMPLE
        torch.cuda.synchronize()
        t0 = time.time()
        if step_i > 0:
            indices = systematic_resample(weights, N)
            particles = pp.SE3(particles.tensor()[indices])
        torch.cuda.synchronize()
        timings["resampling"].append((time.time() - t0) * 1000)

        # 2. ROUGHENING
        torch.cuda.synchronize()
        t0 = time.time()
        if step_i > 0:
            rough_noise = torch.zeros(N, 6, device=DEVICE)
            rough_noise[:, :3] = torch.randn(N, 3, device=DEVICE) * roughening_trans
            rough_noise[:, 3:] = torch.randn(N, 3, device=DEVICE) * roughening_rot
            particles = pp.se3(rough_noise).Exp() @ particles
        torch.cuda.synchronize()
        timings["roughening"].append((time.time() - t0) * 1000)

        # 3. MOTION PREDICTION
        torch.cuda.synchronize()
        t0 = time.time()
        particles = motion.predict(particles)
        torch.cuda.synchronize()
        timings["motion_prediction"].append((time.time() - t0) * 1000)

        # 4. Choose render resolution
        if converged:
            rw, rh = 320, 240
        else:
            rw, rh = 160, 120
        renderer.update_resolution(rw, rh)

        K_native_size = (K[0, 2].item() * 2, K[1, 2].item() * 2)
        K_scaled = scale_intrinsics(K, (int(K_native_size[0]), int(K_native_size[1])), (rw, rh))
        Ks = K_scaled.unsqueeze(0).expand(N, -1, -1).to(DEVICE)

        # Prepare observation (resize)
        torch.cuda.synchronize()
        t0 = time.time()
        img = query_image
        if img.shape[0] != rh or img.shape[1] != rw:
            img = img.permute(2, 0, 1).unsqueeze(0)
            img = torch.nn.functional.interpolate(
                img, size=(rh, rw), mode="bilinear", align_corners=False
            )
            img = img.squeeze(0).permute(1, 2, 0)
        obs_for_model = {"image": img.to(DEVICE)}
        torch.cuda.synchronize()
        timings["obs_prepare"].append((time.time() - t0) * 1000)

        # 5. RENDERING
        torch.cuda.synchronize()
        t0 = time.time()
        viewmats = se3_to_viewmat(particles)
        rendered, _, _ = renderer.render_batch(viewmats, Ks)
        torch.cuda.synchronize()
        timings["rendering"].append((time.time() - t0) * 1000)

        # 6. OBSERVATION WEIGHTS
        torch.cuda.synchronize()
        t0 = time.time()
        log_w = obs_model.compute_log_weights(rendered, obs_for_model)
        weights = normalize_log_weights(log_w).exp()
        torch.cuda.synchronize()
        timings["observation_weights"].append((time.time() - t0) * 1000)

        # 7. POSE ESTIMATION
        torch.cuda.synchronize()
        t0 = time.time()
        estimated = weighted_se3_mean(particles, weights)
        # Track convergence
        trans = particles.tensor()[:, :3]
        mean_trans = (weights.unsqueeze(-1) * trans).sum(dim=0)
        trans_var = (weights.unsqueeze(-1) * (trans - mean_trans) ** 2).sum().item()
        converged = trans_var < convergence_threshold
        torch.cuda.synchronize()
        timings["pose_estimation"].append((time.time() - t0) * 1000)

        # 8. GRADIENT REFINEMENT (post-hoc, only on converged frames)
        torch.cuda.synchronize()
        t0 = time.time()
        if converged:
            est_for_refine = pp.SE3(estimated.tensor().unsqueeze(0))
            with torch.enable_grad():
                refined = refiner.refine(
                    est_for_refine, query_image, K_hires
                )
            estimated = refined.squeeze(0)
            n_refine_steps += 1
        torch.cuda.synchronize()
        t_refine = (time.time() - t0) * 1000
        timings["gradient_refinement"].append(t_refine)

        torch.cuda.synchronize()
        timings["total_step"].append((time.time() - t_total_start) * 1000)

        if (step_i + 1) % 20 == 0 or step_i == 0:
            print(f"  Step {step_i:3d}/{n_frames}: "
                  f"total={timings['total_step'][-1]:.1f}ms  "
                  f"render={timings['rendering'][-1]:.1f}ms  "
                  f"refine={timings['gradient_refinement'][-1]:.1f}ms  "
                  f"conv={converged}")

    # Cleanup
    del renderer, hires_renderer, refiner, gmap, dataset
    torch.cuda.empty_cache()
    gc.collect()

    return timings, n_refine_steps, n_frames


def main():
    torch.set_grad_enabled(False)

    output_dir = Path("results/timing")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  TIMING BREAKDOWN: fr3_office (SSIM + Refine)")
    print(f"  {N_PARTICLES} particles, {N_FRAMES} frames")
    print("=" * 70)

    timings, n_refine_steps, n_frames = run_instrumented_pf()

    # Compute statistics
    results = {}
    print(f"\n{'='*70}")
    print(f"  TIMING RESULTS (averaged over {n_frames} frames)")
    print(f"{'='*70}")
    print(f"{'Component':<28} {'Mean (ms)':>10} {'Std (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'% Total':>8}")
    print("-" * 78)

    total_mean = sum(t for t in timings["total_step"]) / len(timings["total_step"])

    for component in ["resampling", "roughening", "motion_prediction",
                       "obs_prepare", "rendering", "observation_weights",
                       "pose_estimation", "gradient_refinement", "total_step"]:
        vals = timings[component]
        mean_val = sum(vals) / len(vals)
        std_val = (sum((v - mean_val) ** 2 for v in vals) / len(vals)) ** 0.5
        min_val = min(vals)
        max_val = max(vals)
        pct = (mean_val / total_mean) * 100 if total_mean > 0 else 0

        results[component] = {
            "mean_ms": round(mean_val, 3),
            "std_ms": round(std_val, 3),
            "min_ms": round(min_val, 3),
            "max_ms": round(max_val, 3),
            "pct_of_total": round(pct, 1),
        }

        label = component.replace("_", " ").title()
        if component == "total_step":
            print("-" * 78)
            label = "TOTAL"
        print(f"  {label:<26} {mean_val:>10.2f} {std_val:>10.2f} {min_val:>10.2f} {max_val:>10.2f} {pct:>7.1f}%")

    # Also compute refine-only stats (only frames where refinement ran)
    refine_times_nonzero = [t for t in timings["gradient_refinement"] if t > 0.1]
    if refine_times_nonzero:
        mean_refine = sum(refine_times_nonzero) / len(refine_times_nonzero)
        print(f"\n  Gradient refinement (converged frames only):")
        print(f"    Ran on {n_refine_steps}/{n_frames} frames")
        print(f"    Mean when active: {mean_refine:.2f}ms")
        results["gradient_refinement_active_only"] = {
            "mean_ms": round(mean_refine, 3),
            "n_active_frames": n_refine_steps,
            "n_total_frames": n_frames,
        }

    # Save JSON
    output = {
        "metadata": {
            "scene": "fr3_office",
            "observation_model": "SSIM",
            "n_particles": N_PARTICLES,
            "n_frames": n_frames,
            "pf_resolution": "160x120 (low) / 320x240 (hi)",
            "refiner_resolution": "320x240",
            "refiner_iterations": 100,
            "device": DEVICE,
        },
        "timing_breakdown": results,
        "raw_timings": {k: [round(v, 3) for v in vals] for k, vals in timings.items()},
    }

    json_path = output_dir / "timing_breakdown.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
