"""Regenerate all figures with corrected data.

Fixes:
1. Trajectory plots: re-run PF + post-hoc refinement so trajectories show REFINED poses
2. GSLoc baseline plots: use correct PF+Refine reference values from final evaluation
3. HLoc comparison plot: use correct PF+Refine values
4. Convergence plots: regenerate with consistent titles from saved data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Flush print output immediately
import functools
print = functools.partial(print, flush=True)

import torch
import pypose as pp
import json
import gc
import time
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from semantic_pf_loc.utils.visualization import (
    plot_trajectory_2d, plot_convergence_comparison, plot_error_over_time,
)


# ============================================================================
# Scene configurations (same as final evaluation)
# ============================================================================

SCENES = [
    {
        "name": "office0", "type": "replica",
        "path": "data/replica/office0",
        "ckpt": "checkpoints_depthinit/office0.ckpt",
        "native_w": 1200, "native_h": 680,
        "trans_std": 0.003, "rot_std": 0.002,
    },
    {
        "name": "room0", "type": "replica",
        "path": "data/replica/room0",
        "ckpt": "checkpoints_depthinit/room0.ckpt",
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

# CORRECTED PF+Refine reference values from depth-initialized evaluation (3-trial median)
PF_REFINE_CORRECT = {
    "office0":    {"ate_median_cm": 0.41, "are_median_deg": 0.10, "success_rate": 0.75},
    "room0":      {"ate_median_cm": 38.5, "are_median_deg": 10.49, "success_rate": 0.30},
    "fr3_office": {"ate_median_cm": 0.43, "are_median_deg": 0.60, "success_rate": 0.99},
}

N_FRAMES = 100
N_PARTICLES = 100  # Reduced from 200 to fit GPU alongside other processes
DEVICE = "cuda"


# ============================================================================
# Part 1: Re-run PF + post-hoc refinement to get REFINED trajectory poses
# ============================================================================

def run_pf_with_posthoc_refine(dataset, gmap, scene_cfg, seed=42):
    """Run PF + post-hoc refinement for every frame, returning refined poses."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    K = dataset.get_intrinsics().float().to(DEVICE)
    native_size = (scene_cfg["native_w"], scene_cfg["native_h"])

    # Low-res renderer for PF (small chunk_size to fit alongside other GPU processes)
    renderer = BatchRenderer(gmap, width=160, height=120, chunk_size=50)
    motion = MotionModel(
        translation_std=scene_cfg["trans_std"],
        rotation_std=scene_cfg["rot_std"],
        device=DEVICE,
    )
    obs_model = SSIMObservation(temperature=3.0)

    # PF without built-in refiner
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

    # Hi-res renderer + refiner for post-hoc refinement
    hires_renderer = BatchRenderer(gmap, width=320, height=240, chunk_size=1)
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
    trans_errors_list, rot_errors_list = [], []
    converged_mask = []  # track which frames have converged estimates

    n_frames = min(N_FRAMES, len(dataset))
    for i in range(n_frames):
        sample = dataset[i]
        obs = {"image": sample["image"].float().to(DEVICE)}

        est, info = pf.step(obs, K)

        # Apply post-hoc refinement only on converged frames (matching final evaluation)
        if info["converged"]:
            est_for_refine = pp.SE3(est.tensor().unsqueeze(0))
            with torch.enable_grad():
                refined = refiner.refine(
                    est_for_refine, sample["image"].float().to(DEVICE), K_hires
                )
            est_mat = refined.squeeze(0).matrix().cpu()
        else:
            est_mat = est.matrix().cpu()
        gt_mat = sample["pose"].float()

        est_poses.append(est_mat)
        gt_poses.append(gt_mat)
        times.append(info["step_time_ms"])
        converged_mask.append(info["converged"])

        # Per-frame errors
        te = (est_mat[:3, 3] - gt_mat[:3, 3]).norm().item()
        R_diff = est_mat[:3, :3].T @ gt_mat[:3, :3]
        trace = R_diff[0, 0] + R_diff[1, 1] + R_diff[2, 2]
        cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        re = torch.acos(cos_angle).item() * 180.0 / 3.14159265
        trans_errors_list.append(te)
        rot_errors_list.append(re)

        if (i + 1) % 10 == 0:
            print(f"    Frame {i+1}/{n_frames}: TE={te*100:.2f}cm RE={re:.2f}deg")

    est_stack = torch.stack(est_poses)
    gt_stack = torch.stack(gt_poses)

    # Cleanup
    del renderer, pf, hires_renderer, refiner, obs_model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "estimated_poses": est_stack,
        "gt_poses": gt_stack,
        "trans_errors": torch.tensor(trans_errors_list),
        "rot_errors": torch.tensor(rot_errors_list),
        "converged_mask": torch.tensor(converged_mask),
    }


# ============================================================================
# Part 2: GSLoc baseline plots with corrected reference values
# ============================================================================

def regenerate_gsloc_plots(gsloc_results_path, output_dir):
    """Regenerate GSLoc noise_vs_ate and noise_vs_success with corrected PF+Refine values."""
    with open(gsloc_results_path) as f:
        gsloc_data = json.load(f)

    scene_names = ["office0", "room0", "fr3_office"]

    # Parse noise levels from keys
    def parse_noise_key(key):
        parts = key.split("_")
        return float(parts[0])

    # --- noise_vs_ate ---
    fig, axes = plt.subplots(1, len(scene_names), figsize=(5 * len(scene_names), 5), squeeze=False)
    colors_gsloc = "#d62728"
    colors_pf = "#2196F3"

    for j, name in enumerate(scene_names):
        ax = axes[0][j]
        if name not in gsloc_data:
            continue

        noise_vals = []
        ate_vals = []
        for key, metrics in sorted(gsloc_data[name].items()):
            noise_cm = parse_noise_key(key) * 100
            noise_vals.append(noise_cm)
            ate_vals.append(metrics["ate_median"] * 100)

        ax.plot(noise_vals, ate_vals, "o-", color=colors_gsloc, linewidth=2,
                markersize=7, label="GSLoc (gradient-only)")

        # CORRECTED PF+Refine reference
        if name in PF_REFINE_CORRECT:
            pf_ate = PF_REFINE_CORRECT[name]["ate_median_cm"]
            ax.axhline(y=pf_ate, color=colors_pf, linestyle="--", linewidth=2,
                       label=f"PF+Refine ({pf_ate:.1f} cm)")

        # 5cm threshold
        ax.axhline(y=5.0, color="gray", linestyle=":", alpha=0.5, label="5 cm threshold")

        ax.set_xlabel("Init. Trans. Noise (cm)", fontsize=12)
        ax.set_ylabel("ATE Median (cm)", fontsize=12)
        ax.set_title(name, fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("GSLoc vs PF+Refine: Noise Robustness", fontsize=15, y=1.02)
    fig.tight_layout()
    save_path = output_dir / "noise_vs_ate.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    # --- noise_vs_success ---
    fig, axes = plt.subplots(1, len(scene_names), figsize=(5 * len(scene_names), 5), squeeze=False)

    for j, name in enumerate(scene_names):
        ax = axes[0][j]
        if name not in gsloc_data:
            continue

        noise_vals = []
        sr_vals = []
        for key, metrics in sorted(gsloc_data[name].items()):
            noise_cm = parse_noise_key(key) * 100
            noise_vals.append(noise_cm)
            sr_vals.append(metrics["success_rate"] * 100)

        ax.plot(noise_vals, sr_vals, "s-", color=colors_gsloc, linewidth=2,
                markersize=7, label="GSLoc (gradient-only)")

        # CORRECTED PF+Refine reference
        if name in PF_REFINE_CORRECT:
            pf_sr = PF_REFINE_CORRECT[name]["success_rate"] * 100
            ax.axhline(y=pf_sr, color=colors_pf, linestyle="--", linewidth=2,
                       label=f"PF+Refine ({pf_sr:.0f}%)")

        ax.set_xlabel("Init. Trans. Noise (cm)", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title(name, fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    fig.suptitle("GSLoc vs PF+Refine: Success Rate", fontsize=15, y=1.02)
    fig.tight_layout()
    save_path = output_dir / "noise_vs_success.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# Part 3: HLoc comparison plot with corrected PF+Refine values
# ============================================================================

def regenerate_hloc_plot(hloc_results_path, gsloc_results_path, output_dir):
    """Regenerate HLoc comparison bar chart with corrected PF+Refine values."""
    with open(hloc_results_path) as f:
        hloc_data = json.load(f)
    with open(gsloc_results_path) as f:
        gsloc_data = json.load(f)

    scenes = ["office0", "room0", "fr3_office"]
    n_scenes = len(scenes)

    # GSLoc at 3cm noise level
    gsloc_3cm_key = "0.030_0.0100"
    gsloc_results = {}
    for s in scenes:
        if s in gsloc_data and gsloc_3cm_key in gsloc_data[s]:
            m = gsloc_data[s][gsloc_3cm_key]
            gsloc_results[s] = {
                "ate_median_cm": m["ate_median"] * 100,
                "are_median_deg": m["are_median"],
                "success_rate": m["success_rate"],
            }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(n_scenes)
    w = 0.25

    # --- ATE ---
    ax = axes[0]
    hloc_ate = [hloc_data[s]["ate_median"] * 100 for s in scenes]
    pf_ate = [PF_REFINE_CORRECT[s]["ate_median_cm"] for s in scenes]
    gsloc_ate = [gsloc_results[s]["ate_median_cm"] for s in scenes]

    ax.bar(x - w, hloc_ate, w, label="HLoc (SIFT+PnP)", color="#e74c3c", alpha=0.85)
    ax.bar(x, pf_ate, w, label="PF+Refine (Ours)", color="#2196F3", alpha=0.85)
    ax.bar(x + w, gsloc_ate, w, label="GSLoc (Grad-only)", color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=11)
    ax.set_ylabel("ATE Median (cm)", fontsize=12)
    ax.set_title("Translation Error", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=5.0, color="gray", linestyle=":", alpha=0.5)

    # --- ARE ---
    ax = axes[1]
    hloc_are = [hloc_data[s]["are_median"] for s in scenes]
    pf_are = [PF_REFINE_CORRECT[s]["are_median_deg"] for s in scenes]
    gsloc_are = [gsloc_results[s]["are_median_deg"] for s in scenes]

    ax.bar(x - w, hloc_are, w, label="HLoc (SIFT+PnP)", color="#e74c3c", alpha=0.85)
    ax.bar(x, pf_are, w, label="PF+Refine (Ours)", color="#2196F3", alpha=0.85)
    ax.bar(x + w, gsloc_are, w, label="GSLoc (Grad-only)", color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=11)
    ax.set_ylabel("ARE Median (deg)", fontsize=12)
    ax.set_title("Rotation Error", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Success Rate ---
    ax = axes[2]
    hloc_sr = [hloc_data[s]["success_rate"] * 100 for s in scenes]
    pf_sr = [PF_REFINE_CORRECT[s]["success_rate"] * 100 for s in scenes]
    gsloc_sr = [gsloc_results[s]["success_rate"] * 100 for s in scenes]

    ax.bar(x - w, hloc_sr, w, label="HLoc (SIFT+PnP)", color="#e74c3c", alpha=0.85)
    ax.bar(x, pf_sr, w, label="PF+Refine (Ours)", color="#2196F3", alpha=0.85)
    ax.bar(x + w, gsloc_sr, w, label="GSLoc (Grad-only)", color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Success Rate (5cm / 2deg)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    fig.suptitle("HLoc Baseline vs 3DGS-based Methods", fontsize=14, y=1.02)
    fig.tight_layout()
    save_path = output_dir / "hloc_comparison.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# Part 4: Convergence plots with consistent titles
# ============================================================================

def regenerate_convergence_plots(convergence_data, figures_dir):
    """Regenerate convergence plots for all scenes with consistent titles."""
    for scene_name, scene_conv in convergence_data.items():
        # Translation convergence
        plot_convergence_comparison(
            scene_conv, metric="trans",
            title=f"Translation Convergence: {scene_name}",
            save_path=str(figures_dir / f"conv_trans_{scene_name}.png"),
        )
        plt.close("all")
        print(f"  Saved: conv_trans_{scene_name}.png")

        # Rotation convergence
        plot_convergence_comparison(
            scene_conv, metric="rot",
            title=f"Rotation Convergence: {scene_name}",
            save_path=str(figures_dir / f"conv_rot_{scene_name}.png"),
        )
        plt.close("all")
        print(f"  Saved: conv_rot_{scene_name}.png")


# ============================================================================
# Main
# ============================================================================

def main():
    torch.set_grad_enabled(False)

    base_dir = Path(".")
    figures_dir = base_dir / "results" / "final_evaluation" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    gsloc_dir = base_dir / "results" / "gsloc_baseline"
    hloc_dir = base_dir / "results" / "hloc_baseline"

    print("=" * 70)
    print("  REGENERATING FIGURES WITH CORRECTED DATA")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load existing convergence data from final evaluation
    # ------------------------------------------------------------------
    print("\n--- Loading existing results ---")
    all_results_path = base_dir / "results" / "final_evaluation" / "all_results.pt"
    saved_data = torch.load(str(all_results_path), weights_only=False)
    convergence_data = saved_data["convergence_data"]

    # ------------------------------------------------------------------
    # Part 1: Re-run PF + post-hoc refinement for trajectory plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PART 1: Re-running PF + post-hoc refinement for trajectory plots")
    print("=" * 70)

    # Use the median trial seed for each scene (seed=42 for trial 0, which
    # corresponds to the median trial index from the original evaluation)
    # We use seed=1042 (trial 1, the median of 3) to match the original median trial
    for scene_cfg in SCENES:
        scene_name = scene_cfg["name"]
        print(f"\n--- {scene_name} ---")

        if not Path(scene_cfg["ckpt"]).exists():
            print(f"  SKIP - checkpoint not found: {scene_cfg['ckpt']}")
            continue

        # Load dataset
        if scene_cfg["type"] == "tum":
            dataset = TUMDataset(scene_cfg["path"], stride=1)
        else:
            dataset = ReplicaDataset(scene_cfg["path"], stride=1)

        # Load map
        gmap = GaussianMap.from_checkpoint(scene_cfg["ckpt"])
        print(f"  Loaded: {gmap.num_gaussians} Gaussians, {len(dataset)} frames")

        try:
            t0 = time.time()
            result = run_pf_with_posthoc_refine(dataset, gmap, scene_cfg, seed=1042)
            elapsed = time.time() - t0

            # Compute summary
            te_median = result["trans_errors"].median().item() * 100
            re_median = result["rot_errors"].median().item()
            sr = ((result["trans_errors"] < 0.05) & (result["rot_errors"] < 2.0)).float().mean().item() * 100
            print(f"  Done ({elapsed:.1f}s): ATE={te_median:.2f}cm ARE={re_median:.2f}deg SR={sr:.0f}%")

            # Save trajectory plot — use only converged frames so that
            # pre-convergence outliers don't dominate the XZ scale
            conv = result["converged_mask"]
            if conv.any():
                # Find first converged frame; include everything from there onward
                first_conv = conv.nonzero(as_tuple=True)[0][0].item()
                est_traj = result["estimated_poses"][first_conv:]
                gt_traj = result["gt_poses"][first_conv:]
            else:
                est_traj = result["estimated_poses"]
                gt_traj = result["gt_poses"]

            plot_trajectory_2d(
                est_traj, gt_traj,
                title=f"Trajectory: {scene_name} (SSIM+Refine)",
                save_path=str(figures_dir / f"traj_{scene_name}.png"),
            )
            plt.close("all")
            print(f"  Saved: traj_{scene_name}.png")

            # Save error over time
            plot_error_over_time(
                result["trans_errors"], result["rot_errors"],
                title=f"Error: {scene_name} (SSIM+Refine)",
                save_path=str(figures_dir / f"errors_{scene_name}.png"),
            )
            plt.close("all")
            print(f"  Saved: errors_{scene_name}.png")

            # Update convergence data for SSIM+Refine with refined trajectory data
            if scene_name in convergence_data:
                convergence_data[scene_name]["SSIM+Refine"] = {
                    "trans_errors": result["trans_errors"],
                    "rot_errors": result["rot_errors"],
                }

            del result

        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping trajectory for {scene_name}")

        # Cleanup
        del gmap, dataset
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # Part 4: Convergence plots with consistent titles (do before GSLoc/HLoc
    #          since those don't need GPU)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PART 4: Convergence plots")
    print("=" * 70)
    regenerate_convergence_plots(convergence_data, figures_dir)

    # ------------------------------------------------------------------
    # Part 2: GSLoc baseline plots with corrected values
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PART 2: GSLoc baseline plots with corrected PF+Refine values")
    print("=" * 70)
    gsloc_results_json = gsloc_dir / "results.json"
    if gsloc_results_json.exists():
        regenerate_gsloc_plots(gsloc_results_json, gsloc_dir)
    else:
        print(f"  SKIP - {gsloc_results_json} not found")

    # ------------------------------------------------------------------
    # Part 3: HLoc comparison plot with corrected values
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PART 3: HLoc comparison plot with corrected PF+Refine values")
    print("=" * 70)
    hloc_results_json = hloc_dir / "results.json"
    gsloc_results_json = gsloc_dir / "results.json"
    if hloc_results_json.exists() and gsloc_results_json.exists():
        regenerate_hloc_plot(hloc_results_json, gsloc_results_json, hloc_dir)
    else:
        print(f"  SKIP - results files not found")

    print("\n" + "=" * 70)
    print("  ALL FIGURES REGENERATED")
    print("=" * 70)
    print(f"  Trajectory plots: {figures_dir}/traj_*.png")
    print(f"  Error plots:      {figures_dir}/errors_*.png")
    print(f"  Convergence:      {figures_dir}/conv_*.png")
    print(f"  GSLoc plots:      {gsloc_dir}/noise_vs_*.png")
    print(f"  HLoc plot:        {hloc_dir}/hloc_comparison.png")


if __name__ == "__main__":
    main()
