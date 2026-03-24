"""Compare old (no depth supervision) vs new (depth-supervised) 3DGS checkpoints.

Uses TRACKING mode: initialize PF around the first GT pose (0.5m trans, 0.3 rad rot spread).
Runs PF with SSIM (200 particles, 160x120, temp=3.0) for 100 frames per scene,
then applies tuned gradient refinement (single-pose, 320x240, 100 iters, lr=0.01).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pypose as pp
import json
import time
from pathlib import Path
from tqdm import tqdm

from semantic_pf_loc.gaussian_map import GaussianMap
from semantic_pf_loc.batch_renderer import BatchRenderer
from semantic_pf_loc.particle_filter import ParticleFilter
from semantic_pf_loc.motion_model import MotionModel
from semantic_pf_loc.gradient_refiner import GradientRefiner
from semantic_pf_loc.observation.ssim import SSIMObservation
from semantic_pf_loc.datasets.replica import ReplicaDataset, REPLICA_WIDTH, REPLICA_HEIGHT
from semantic_pf_loc.evaluation.metrics import (
    translation_error,
    rotation_error,
    absolute_trajectory_error,
    absolute_rotation_error,
    success_rate,
)
from semantic_pf_loc.utils.pose_utils import scale_intrinsics


# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "replica"
CKPT_OLD_DIR = PROJECT_ROOT / "checkpoints"
CKPT_DEPTH_DIR = PROJECT_ROOT / "checkpoints_depth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "depth_comparison"

SCENES = ["office0", "room0", "room1"]
NUM_FRAMES = 100
NUM_PARTICLES = 200
PF_WIDTH, PF_HEIGHT = 160, 120          # PF render resolution
REFINE_WIDTH, REFINE_HEIGHT = 320, 240  # Gradient refiner resolution
REFINE_ITERS = 100
REFINE_LR = 0.01
TEMPERATURE = 3.0
DEVICE = "cuda"


def run_one(scene: str, ckpt_path: str, label: str) -> dict:
    """Run PF + single-pose refinement on one scene/checkpoint combo."""
    print(f"\n{'='*60}")
    print(f"  Scene: {scene}  |  Checkpoint: {label}")
    print(f"  Checkpoint path: {ckpt_path}")
    print(f"{'='*60}")

    # Load dataset (stride=1, we'll just use first NUM_FRAMES)
    data_dir = str(DATA_ROOT / scene)
    dataset = ReplicaDataset(data_dir, stride=1)
    n_frames = min(NUM_FRAMES, len(dataset))

    # Load 3DGS map
    gmap = GaussianMap.from_checkpoint(str(ckpt_path), device=DEVICE)
    print(f"  Loaded {gmap.num_gaussians:,} Gaussians")

    # PF renderer (low-res for particle weighting)
    renderer = BatchRenderer(gmap, width=PF_WIDTH, height=PF_HEIGHT)

    # High-res renderer + refiner (for post-PF single-pose refinement)
    hires_renderer = BatchRenderer(gmap, width=REFINE_WIDTH, height=REFINE_HEIGHT)
    refiner = GradientRefiner(
        hires_renderer,
        num_iterations=REFINE_ITERS,
        lr_init=REFINE_LR,
        blur_schedule=True,
    )

    # Observation model
    obs_model = SSIMObservation(temperature=TEMPERATURE)

    # Motion model
    motion = MotionModel(translation_std=0.003, rotation_std=0.002, device=DEVICE)

    # Particle filter — NO built-in gradient refiner
    pf = ParticleFilter(
        gaussian_map=gmap,
        renderer=renderer,
        observation_model=obs_model,
        motion_model=motion,
        num_particles=NUM_PARTICLES,
        gradient_refiner=None,          # refine separately after PF
        render_width=PF_WIDTH,
        render_height=PF_HEIGHT,
        device=DEVICE,
    )

    # Native intrinsics
    K_native = dataset.get_intrinsics().float().to(DEVICE)  # [3, 3]
    native_size = (REPLICA_WIDTH, REPLICA_HEIGHT)  # (1200, 680)

    # High-res intrinsics for refinement
    K_hires = scale_intrinsics(K_native, native_size, (REFINE_WIDTH, REFINE_HEIGHT))

    # Initialize PF in TRACKING mode (around first GT pose)
    first_sample = dataset[0]
    first_gt = first_sample["pose"].double().to(DEVICE)  # [4, 4]
    init_pose = pp.mat2SE3(first_gt.unsqueeze(0), check=False).squeeze(0).float()
    pf.initialize_around_pose(init_pose, trans_spread=0.5, rot_spread=0.3)

    # Run
    est_poses_raw = []
    est_poses_refined = []
    gt_poses = []
    step_times = []

    pbar = tqdm(range(n_frames), desc=f"{scene}/{label}", ncols=100)
    for i in pbar:
        sample = dataset[i]
        gt_pose = sample["pose"].float()  # [4, 4]
        query_image = sample["image"].float().to(DEVICE)  # [H, W, 3]

        observation = {"image": query_image}

        t0 = time.time()

        # PF step (raw, no gradient refiner)
        est_pose, info = pf.step(observation, K_native)

        # Record raw PF estimate
        est_matrix_raw = est_pose.matrix().cpu()
        est_poses_raw.append(est_matrix_raw)

        # Single-pose gradient refinement on the PF estimate
        with torch.enable_grad():
            refined = refiner.refine(
                pp.SE3(est_pose.tensor().unsqueeze(0)),  # [1, 7]
                query_image,  # [H, W, 3] — refiner internally resizes
                K_hires,      # [3, 3]
            )

        est_matrix_refined = refined.squeeze(0).matrix().cpu()
        est_poses_refined.append(est_matrix_refined)

        elapsed = (time.time() - t0) * 1000
        step_times.append(elapsed)

        gt_poses.append(gt_pose)

        if i % 20 == 0:
            t_err_raw = (est_matrix_raw[:3, 3] - gt_pose[:3, 3]).norm().item()
            t_err_ref = (est_matrix_refined[:3, 3] - gt_pose[:3, 3]).norm().item()
            pbar.set_postfix(
                raw=f"{t_err_raw:.3f}m",
                ref=f"{t_err_ref:.3f}m",
                conv=info.get("converged", False),
            )

    # Stack results
    gt_stack = torch.stack(gt_poses)              # [T, 4, 4]
    raw_stack = torch.stack(est_poses_raw)        # [T, 4, 4]
    ref_stack = torch.stack(est_poses_refined)    # [T, 4, 4]

    # Compute metrics on REFINED poses (the ones we care about)
    ate = absolute_trajectory_error(ref_stack, gt_stack)
    are = absolute_rotation_error(ref_stack, gt_stack)
    sr = success_rate(ref_stack, gt_stack, trans_threshold=0.05, rot_threshold=2.0)

    # Also compute raw PF metrics for reference
    ate_raw = absolute_trajectory_error(raw_stack, gt_stack)
    are_raw = absolute_rotation_error(raw_stack, gt_stack)
    sr_raw = success_rate(raw_stack, gt_stack, trans_threshold=0.05, rot_threshold=2.0)

    result = {
        "scene": scene,
        "checkpoint": label,
        "n_frames": n_frames,
        "ate_median_cm": ate["median"] * 100,
        "are_median_deg": are["median"],
        "success_rate": sr * 100,
        "ate_rmse_cm": ate["rmse"] * 100,
        "are_rmse_deg": are["rmse"],
        "ate_raw_median_cm": ate_raw["median"] * 100,
        "are_raw_median_deg": are_raw["median"],
        "sr_raw": sr_raw * 100,
        "mean_step_time_ms": sum(step_times) / len(step_times),
    }

    print(f"\n  Results ({label}):")
    print(f"    Raw PF:   ATE med {ate_raw['median']*100:.2f} cm | ARE med {are_raw['median']:.2f}° | SR {sr_raw*100:.1f}%")
    print(f"    Refined:  ATE med {ate['median']*100:.2f} cm | ARE med {are['median']:.2f}° | SR {sr*100:.1f}%")

    # Cleanup GPU
    del gmap, renderer, hires_renderer, refiner, pf
    torch.cuda.empty_cache()

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for scene in SCENES:
        for label, ckpt_dir in [("old", CKPT_OLD_DIR), ("depth", CKPT_DEPTH_DIR)]:
            ckpt_path = ckpt_dir / f"{scene}.ckpt"
            if not ckpt_path.exists():
                print(f"WARNING: {ckpt_path} not found, skipping")
                continue
            result = run_one(scene, str(ckpt_path), label)
            all_results.append(result)

    # ── Print comparison table ──────────────────────────────────────────
    print("\n")
    print("=" * 85)
    print("  DEPTH SUPERVISION COMPARISON  (PF + single-pose refinement @ 320x240, 100 iters)")
    print("=" * 85)

    header = f"{'Scene':<12} | {'Ckpt':<6} | {'ATE Med (cm)':>12} | {'ARE Med (deg)':>13} | {'Success %':>9} | {'ATE RMSE':>8}"
    print(header)
    print("-" * 85)

    for r in all_results:
        row = (
            f"{r['scene']:<12} | {r['checkpoint']:<6} | "
            f"{r['ate_median_cm']:>12.2f} | {r['are_median_deg']:>13.2f} | "
            f"{r['success_rate']:>8.1f}% | {r['ate_rmse_cm']:>7.2f}"
        )
        print(row)

    # Also print raw PF results for reference
    print("\n")
    print("=" * 85)
    print("  RAW PF (no refinement) for reference")
    print("=" * 85)
    header2 = f"{'Scene':<12} | {'Ckpt':<6} | {'ATE Med (cm)':>12} | {'ARE Med (deg)':>13} | {'Success %':>9}"
    print(header2)
    print("-" * 85)

    for r in all_results:
        row = (
            f"{r['scene']:<12} | {r['checkpoint']:<6} | "
            f"{r['ate_raw_median_cm']:>12.2f} | {r['are_raw_median_deg']:>13.2f} | "
            f"{r['sr_raw']:>8.1f}%"
        )
        print(row)

    # ── Compute improvement summary ─────────────────────────────────────
    print("\n")
    print("=" * 85)
    print("  IMPROVEMENT (depth vs old)")
    print("=" * 85)
    for scene in SCENES:
        old = next((r for r in all_results if r["scene"] == scene and r["checkpoint"] == "old"), None)
        new = next((r for r in all_results if r["scene"] == scene and r["checkpoint"] == "depth"), None)
        if old and new:
            ate_diff = old["ate_median_cm"] - new["ate_median_cm"]
            are_diff = old["are_median_deg"] - new["are_median_deg"]
            sr_diff = new["success_rate"] - old["success_rate"]
            ate_pct = (ate_diff / old["ate_median_cm"] * 100) if old["ate_median_cm"] > 0 else 0
            print(
                f"  {scene:<12}: ATE {ate_diff:+.2f} cm ({ate_pct:+.1f}%)  |  "
                f"ARE {are_diff:+.2f}°  |  SR {sr_diff:+.1f}pp"
            )

    # ── Save results ────────────────────────────────────────────────────
    results_path = OUTPUT_DIR / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Also save raw tensors
    torch.save(all_results, OUTPUT_DIR / "comparison_results.pt")
    print(f"Tensors saved to {OUTPUT_DIR / 'comparison_results.pt'}")


if __name__ == "__main__":
    main()
