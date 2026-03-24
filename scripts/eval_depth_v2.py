"""Compare old (no depth supervision) vs depth-supervised 3DGS checkpoints — V2.

CORRECTED: uses trans_spread=0.03, rot_spread=0.01 (local init around GT).
Previous v1 used 0.5m / 0.3rad which made results garbage.

For each scene + checkpoint:
  1. PF (200 particles, 160x120, SSIM temp=3.0, 100 frames)
  2. Tuned gradient refinement (separate 320x240 renderer, 100 iters, lr=0.01)
  3. PSNR measurement (50 frames at half-native resolution)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
import pypose as pp
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from semantic_pf_loc.gaussian_map import GaussianMap
from semantic_pf_loc.batch_renderer import BatchRenderer
from semantic_pf_loc.particle_filter import ParticleFilter
from semantic_pf_loc.motion_model import MotionModel
from semantic_pf_loc.gradient_refiner import GradientRefiner
from semantic_pf_loc.observation.ssim import SSIMObservation
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.datasets.tum import TUMDataset
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
CKPT_OLD_DIR = PROJECT_ROOT / "checkpoints"
CKPT_DEPTH_DIR = PROJECT_ROOT / "checkpoints_depth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "depth_comparison_v2"

NUM_FRAMES = 100
NUM_PARTICLES = 200
PF_WIDTH, PF_HEIGHT = 160, 120
REFINE_WIDTH, REFINE_HEIGHT = 320, 240
REFINE_ITERS = 100
REFINE_LR = 0.01
TEMPERATURE = 3.0
DEVICE = "cuda"

# CORRECTED spreads (local tracking, NOT global init)
TRANS_SPREAD = 0.03
ROT_SPREAD = 0.01

SCENES = [
    {
        "name": "office0",
        "type": "replica",
        "path": "data/replica/office0",
        "native_w": 1200,
        "native_h": 680,
        "trans_std": 0.003,
        "rot_std": 0.002,
    },
    {
        "name": "room0",
        "type": "replica",
        "path": "data/replica/room0",
        "native_w": 1200,
        "native_h": 680,
        "trans_std": 0.003,
        "rot_std": 0.002,
    },
    {
        "name": "room1",
        "type": "replica",
        "path": "data/replica/room1",
        "native_w": 1200,
        "native_h": 680,
        "trans_std": 0.003,
        "rot_std": 0.002,
    },
    {
        "name": "fr3_office",
        "type": "tum",
        "path": "data/tum/rgbd_dataset_freiburg3_long_office_household",
        "native_w": 640,
        "native_h": 480,
        "trans_std": 0.005,
        "rot_std": 0.003,
    },
    {
        "name": "fr1_desk",
        "type": "tum",
        "path": "data/tum/rgbd_dataset_freiburg1_desk",
        "native_w": 640,
        "native_h": 480,
        "trans_std": 0.005,
        "rot_std": 0.003,
    },
]


def compute_psnr(gmap, dataset, scene_cfg):
    """Compute PSNR of a checkpoint on ~50 random frames at half-native res."""
    psnr_w = scene_cfg["native_w"] // 2
    psnr_h = scene_cfg["native_h"] // 2
    native_size = (scene_cfg["native_w"], scene_cfg["native_h"])

    psnr_renderer = BatchRenderer(gmap, width=psnr_w, height=psnr_h)
    K_native = dataset.get_intrinsics().float().to(DEVICE)
    K_psnr = scale_intrinsics(K_native, native_size, (psnr_w, psnr_h))

    psnrs = []
    # Sample every 10th frame, up to 50 samples
    indices = list(range(0, min(500, len(dataset)), 10))[:50]

    for i in tqdm(indices, desc="  PSNR", ncols=80, leave=False):
        sample = dataset[i]
        gt_img = sample["image"].float().to(DEVICE)  # [H, W, 3]
        gt_pose_mat = sample["pose"].double().to(DEVICE)  # [4, 4]

        # Convert to SE3, then to viewmat (world-to-camera)
        gt_se3 = pp.mat2SE3(gt_pose_mat.unsqueeze(0), check=False).squeeze(0).float()
        viewmat = gt_se3.Inv().matrix().unsqueeze(0)  # [1, 4, 4]

        with torch.no_grad():
            rendered, _, _ = psnr_renderer.render_batch(
                viewmat, K_psnr.unsqueeze(0)
            )  # [1, psnr_h, psnr_w, 3]

        # Resize GT to match
        gt_resized = F.interpolate(
            gt_img.permute(2, 0, 1).unsqueeze(0),
            size=(psnr_h, psnr_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)  # [psnr_h, psnr_w, 3]

        mse = ((rendered[0] - gt_resized) ** 2).mean()
        if mse > 0:
            psnr = -10 * torch.log10(mse)
            psnrs.append(psnr.item())

    del psnr_renderer
    torch.cuda.empty_cache()

    return float(np.mean(psnrs)) if psnrs else 0.0


def run_one(scene_cfg, ckpt_path, label):
    """Run PF + single-pose refinement on one scene/checkpoint combo."""
    scene_name = scene_cfg["name"]
    print(f"\n{'='*70}")
    print(f"  Scene: {scene_name}  |  Checkpoint: {label}")
    print(f"  Path: {ckpt_path}")
    print(f"  Init spread: trans={TRANS_SPREAD}m, rot={ROT_SPREAD}rad (CORRECTED)")
    print(f"{'='*70}")

    # Load dataset
    data_path = str(PROJECT_ROOT / scene_cfg["path"])
    if scene_cfg["type"] == "replica":
        dataset = ReplicaDataset(data_path, stride=1)
    else:
        dataset = TUMDataset(data_path, stride=1)
    n_frames = min(NUM_FRAMES, len(dataset))

    # Load 3DGS map
    gmap = GaussianMap.from_checkpoint(str(ckpt_path), device=DEVICE)
    print(f"  Loaded {gmap.num_gaussians:,} Gaussians")

    # ── PSNR measurement ──────────────────────────────────────────────
    print("  Computing PSNR...")
    psnr_val = compute_psnr(gmap, dataset, scene_cfg)
    print(f"  PSNR: {psnr_val:.2f} dB")

    # ── PF setup ──────────────────────────────────────────────────────
    native_size = (scene_cfg["native_w"], scene_cfg["native_h"])
    K_native = dataset.get_intrinsics().float().to(DEVICE)

    # PF renderer (low-res)
    renderer = BatchRenderer(gmap, width=PF_WIDTH, height=PF_HEIGHT)

    # Observation & motion models
    obs_model = SSIMObservation(temperature=TEMPERATURE)
    motion = MotionModel(
        translation_std=scene_cfg["trans_std"],
        rotation_std=scene_cfg["rot_std"],
        device=DEVICE,
    )

    # Particle filter (NO built-in gradient refiner)
    pf = ParticleFilter(
        gaussian_map=gmap,
        renderer=renderer,
        observation_model=obs_model,
        motion_model=motion,
        num_particles=NUM_PARTICLES,
        gradient_refiner=None,
        render_width=PF_WIDTH,
        render_height=PF_HEIGHT,
        render_width_hires=REFINE_WIDTH,
        render_height_hires=REFINE_HEIGHT,
        convergence_threshold=0.02,
        roughening_trans=0.002,
        roughening_rot=0.001,
        device=DEVICE,
    )

    # Initialize PF with CORRECT spread
    first_sample = dataset[0]
    first_gt = first_sample["pose"].double().to(DEVICE)
    init_pose = pp.mat2SE3(first_gt.unsqueeze(0), check=False).squeeze(0).float()
    pf.initialize_around_pose(init_pose, trans_spread=TRANS_SPREAD, rot_spread=ROT_SPREAD)

    # ── Separate high-res renderer + refiner ──────────────────────────
    K_hires = scale_intrinsics(K_native, native_size, (REFINE_WIDTH, REFINE_HEIGHT))
    hires_renderer = BatchRenderer(gmap, width=REFINE_WIDTH, height=REFINE_HEIGHT)
    refiner = GradientRefiner(
        hires_renderer,
        num_iterations=REFINE_ITERS,
        lr_init=REFINE_LR,
        blur_schedule=True,
    )

    # ── Run PF + refinement ───────────────────────────────────────────
    est_poses_raw = []
    est_poses_refined = []
    gt_poses = []

    pbar = tqdm(range(n_frames), desc=f"{scene_name}/{label}", ncols=100)
    for i in pbar:
        sample = dataset[i]
        gt_pose = sample["pose"].float()  # [4, 4]
        query_image = sample["image"].float().to(DEVICE)

        observation = {"image": query_image}

        # PF step
        est_pose, info = pf.step(observation, K_native)

        # Raw PF estimate
        est_matrix_raw = est_pose.matrix().cpu()
        est_poses_raw.append(est_matrix_raw)

        # Single-pose gradient refinement at 320x240
        with torch.enable_grad():
            refined = refiner.refine(
                pp.SE3(est_pose.tensor().unsqueeze(0)),
                query_image,
                K_hires,
            )

        est_matrix_refined = refined.squeeze(0).matrix().cpu()
        est_poses_refined.append(est_matrix_refined)
        gt_poses.append(gt_pose)

        if i % 20 == 0:
            t_err_raw = (est_matrix_raw[:3, 3] - gt_pose[:3, 3]).norm().item()
            t_err_ref = (est_matrix_refined[:3, 3] - gt_pose[:3, 3]).norm().item()
            pbar.set_postfix(
                raw=f"{t_err_raw:.3f}m",
                ref=f"{t_err_ref:.4f}m",
                conv=info.get("converged", False),
            )

    # ── Compute metrics ───────────────────────────────────────────────
    gt_stack = torch.stack(gt_poses)
    raw_stack = torch.stack(est_poses_raw)
    ref_stack = torch.stack(est_poses_refined)

    # Raw PF
    ate_raw = absolute_trajectory_error(raw_stack, gt_stack)
    are_raw = absolute_rotation_error(raw_stack, gt_stack)
    sr_raw = success_rate(raw_stack, gt_stack, trans_threshold=0.05, rot_threshold=2.0)

    # PF + refinement
    ate_ref = absolute_trajectory_error(ref_stack, gt_stack)
    are_ref = absolute_rotation_error(ref_stack, gt_stack)
    sr_ref = success_rate(ref_stack, gt_stack, trans_threshold=0.05, rot_threshold=2.0)

    result = {
        "scene": scene_name,
        "checkpoint": label,
        "n_frames": n_frames,
        "psnr_db": psnr_val,
        # Raw PF
        "pf_ate_median_cm": ate_raw["median"] * 100,
        "pf_are_median_deg": are_raw["median"],
        "pf_sr": sr_raw * 100,
        # PF + refinement
        "ref_ate_median_cm": ate_ref["median"] * 100,
        "ref_are_median_deg": are_ref["median"],
        "ref_sr": sr_ref * 100,
        "ref_ate_rmse_cm": ate_ref["rmse"] * 100,
    }

    print(f"\n  Results ({label}):")
    print(f"    PSNR:       {psnr_val:.2f} dB")
    print(f"    Raw PF:     ATE med {ate_raw['median']*100:.2f} cm | ARE med {are_raw['median']:.2f} deg | SR {sr_raw*100:.1f}%")
    print(f"    PF+Refine:  ATE med {ate_ref['median']*100:.2f} cm | ARE med {are_ref['median']:.2f} deg | SR {sr_ref*100:.1f}%")

    # Cleanup
    del gmap, renderer, hires_renderer, refiner, pf
    torch.cuda.empty_cache()

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DEPTH SUPERVISION COMPARISON V2 (CORRECTED init spread)")
    print(f"  trans_spread={TRANS_SPREAD}, rot_spread={ROT_SPREAD}")
    print(f"  PF: {NUM_PARTICLES} particles, {PF_WIDTH}x{PF_HEIGHT}, SSIM temp={TEMPERATURE}")
    print(f"  Refine: {REFINE_WIDTH}x{REFINE_HEIGHT}, {REFINE_ITERS} iters, lr={REFINE_LR}")
    print("=" * 70)

    all_results = []

    for scene_cfg in SCENES:
        for label, ckpt_dir in [("old", CKPT_OLD_DIR), ("depth", CKPT_DEPTH_DIR)]:
            ckpt_path = ckpt_dir / f"{scene_cfg['name']}.ckpt"
            if not ckpt_path.exists():
                print(f"WARNING: {ckpt_path} not found, skipping")
                continue
            result = run_one(scene_cfg, str(ckpt_path), label)
            all_results.append(result)

    # ── Print final comparison table ──────────────────────────────────
    print("\n\n")
    print("=" * 100)
    print("  DEPTH SUPERVISION COMPARISON V2  (trans_spread=0.03, rot_spread=0.01)")
    print("=" * 100)

    header = (
        f"{'Scene':<12} | {'Ckpt':<6} | {'PSNR':>6} | "
        f"{'PF ATE':>7} | {'PF+Ref ATE':>10} | {'PF+Ref ARE':>10} | {'SR%':>5}"
    )
    print(header)
    print("-" * 100)

    for r in all_results:
        row = (
            f"{r['scene']:<12} | {r['checkpoint']:<6} | "
            f"{r['psnr_db']:>5.1f} | "
            f"{r['pf_ate_median_cm']:>6.2f}cm | "
            f"{r['ref_ate_median_cm']:>8.2f}cm | "
            f"{r['ref_are_median_deg']:>8.2f}° | "
            f"{r['ref_sr']:>4.0f}%"
        )
        print(row)

    # ── Improvement summary ───────────────────────────────────────────
    print("\n")
    print("=" * 100)
    print("  IMPROVEMENT (depth vs old)")
    print("=" * 100)
    for scene_cfg in SCENES:
        sn = scene_cfg["name"]
        old = next((r for r in all_results if r["scene"] == sn and r["checkpoint"] == "old"), None)
        new = next((r for r in all_results if r["scene"] == sn and r["checkpoint"] == "depth"), None)
        if old and new:
            psnr_diff = new["psnr_db"] - old["psnr_db"]
            ate_diff = old["ref_ate_median_cm"] - new["ref_ate_median_cm"]
            are_diff = old["ref_are_median_deg"] - new["ref_are_median_deg"]
            sr_diff = new["ref_sr"] - old["ref_sr"]
            ate_pct = (ate_diff / old["ref_ate_median_cm"] * 100) if old["ref_ate_median_cm"] > 0 else 0
            print(
                f"  {sn:<12}: PSNR {psnr_diff:+.1f} dB  |  "
                f"ATE {ate_diff:+.2f} cm ({ate_pct:+.1f}%)  |  "
                f"ARE {are_diff:+.2f} deg  |  SR {sr_diff:+.1f}pp"
            )

    # ── Save results ──────────────────────────────────────────────────
    results_path = OUTPUT_DIR / "comparison_results_v2.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    torch.save(all_results, OUTPUT_DIR / "comparison_results_v2.pt")
    print(f"Tensors saved to {OUTPUT_DIR / 'comparison_results_v2.pt'}")


if __name__ == "__main__":
    main()
