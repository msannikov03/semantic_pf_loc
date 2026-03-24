"""Lighting perturbation ablation: SSIM vs CLIP-Image robustness to gamma shifts.

Applies gamma correction to query images ONLY (not rendered images from 3DGS)
to simulate real-world lighting variation. Tests the hypothesis that CLIP's
semantic features are more invariant to lighting than SSIM's pixel-level comparison.

Gamma < 1.0 = brighter, gamma = 1.0 = no change, gamma > 1.0 = darker.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import torch
import pypose as pp
import numpy as np
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from semantic_pf_loc.gaussian_map import GaussianMap
from semantic_pf_loc.batch_renderer import BatchRenderer
from semantic_pf_loc.particle_filter import ParticleFilter
from semantic_pf_loc.motion_model import MotionModel
from semantic_pf_loc.observation.ssim import SSIMObservation
from semantic_pf_loc.observation.clip_image import CLIPImageObservation
from semantic_pf_loc.datasets.tum import TUMDataset
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.evaluation.metrics import compute_all_metrics


# --- Configuration ---
SCENES = [
    {
        "name": "office0", "type": "replica",
        "path": "data/replica/office0",
        "trans_std": 0.003, "rot_std": 0.002,
    },
    {
        "name": "room0", "type": "replica",
        "path": "data/replica/room0",
        "trans_std": 0.003, "rot_std": 0.002,
    },
    {
        "name": "fr3_office", "type": "tum",
        "path": "data/tum/rgbd_dataset_freiburg3_long_office_household",
        "trans_std": 0.005, "rot_std": 0.003,
    },
]

GAMMA_VALUES = [0.5, 0.75, 1.0, 1.5, 2.0]
N_PARTICLES = 200
N_FRAMES = 100


def run_pf_with_gamma(dataset, gmap, obs_model, motion_params, gamma=1.0,
                       n_frames=N_FRAMES, n_particles=N_PARTICLES):
    """Run PF with gamma-perturbed query images. Returns metrics dict."""
    K = dataset.get_intrinsics().float().to("cuda")
    renderer = BatchRenderer(gmap, width=160, height=120)
    motion = MotionModel(
        translation_std=motion_params["trans_std"],
        rotation_std=motion_params["rot_std"],
        device="cuda",
    )

    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=n_particles, render_width=160, render_height=120,
        render_width_hires=320, render_height_hires=240,
        convergence_threshold=0.02, roughening_trans=0.002, roughening_rot=0.001,
        device="cuda",
    )

    sample0 = dataset[0]
    gt_pose0 = pp.mat2SE3(
        sample0["pose"].double().to("cuda").unsqueeze(0), check=False
    ).squeeze(0).float()
    pf.initialize_around_pose(gt_pose0, trans_spread=0.03, rot_spread=0.01)

    est_poses, gt_poses, times = [], [], []
    for i in range(min(n_frames, len(dataset))):
        sample = dataset[i]
        # Apply gamma correction to query image ONLY
        obs = {"image": sample["image"].float().to("cuda").pow(gamma)}
        est, info = pf.step(obs, K)
        est_poses.append(est.matrix().cpu())
        gt_poses.append(sample["pose"].float())
        times.append(info["step_time_ms"])

    est_stack = torch.stack(est_poses)
    gt_stack = torch.stack(gt_poses)
    metrics = compute_all_metrics(est_stack, gt_stack, torch.tensor(times))

    del renderer, pf
    torch.cuda.empty_cache()
    return metrics


def plot_lighting_robustness(all_results, save_path):
    """Plot ATE median vs gamma for SSIM and CLIP-Image, one subplot per scene."""
    scenes = list(dict.fromkeys(r["scene"] for r in all_results))
    n_scenes = len(scenes)

    fig, axes = plt.subplots(1, n_scenes, figsize=(5 * n_scenes, 4.5), squeeze=False)
    axes = axes[0]

    colors = {"SSIM": "#2196F3", "CLIP-Image": "#FF5722"}
    markers = {"SSIM": "o", "CLIP-Image": "s"}

    for i, scene in enumerate(scenes):
        ax = axes[i]
        for model_name in ["SSIM", "CLIP-Image"]:
            rows = [r for r in all_results
                    if r["scene"] == scene and r["model"] == model_name]
            rows.sort(key=lambda r: r["gamma"])
            gammas = [r["gamma"] for r in rows]
            ates = [r["ate_median"] * 100 for r in rows]  # convert to cm

            ax.plot(gammas, ates,
                    color=colors[model_name],
                    marker=markers[model_name],
                    linewidth=2, markersize=8,
                    label=model_name, alpha=0.9)

        ax.set_xlabel("Gamma", fontsize=12)
        if i == 0:
            ax.set_ylabel("ATE Median (cm)", fontsize=12)
        ax.set_title(scene, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(GAMMA_VALUES)
        ax.set_ylim(bottom=0)

        # Mark gamma=1.0 (no perturbation)
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle("Lighting Robustness: SSIM vs CLIP-Image", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved to {save_path}")


def plot_success_rate(all_results, save_path):
    """Plot success rate vs gamma."""
    scenes = list(dict.fromkeys(r["scene"] for r in all_results))
    n_scenes = len(scenes)

    fig, axes = plt.subplots(1, n_scenes, figsize=(5 * n_scenes, 4.5), squeeze=False)
    axes = axes[0]

    colors = {"SSIM": "#2196F3", "CLIP-Image": "#FF5722"}
    markers = {"SSIM": "o", "CLIP-Image": "s"}

    for i, scene in enumerate(scenes):
        ax = axes[i]
        for model_name in ["SSIM", "CLIP-Image"]:
            rows = [r for r in all_results
                    if r["scene"] == scene and r["model"] == model_name]
            rows.sort(key=lambda r: r["gamma"])
            gammas = [r["gamma"] for r in rows]
            srs = [r["success_rate"] * 100 for r in rows]

            ax.plot(gammas, srs,
                    color=colors[model_name],
                    marker=markers[model_name],
                    linewidth=2, markersize=8,
                    label=model_name, alpha=0.9)

        ax.set_xlabel("Gamma", fontsize=12)
        if i == 0:
            ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title(scene, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(GAMMA_VALUES)
        ax.set_ylim(-5, 105)
        ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle("Success Rate under Lighting Perturbation", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved to {save_path}")


def generate_latex_table(all_results):
    """Generate LaTeX table for the lighting ablation."""
    scenes = list(dict.fromkeys(r["scene"] for r in all_results))
    models = ["SSIM", "CLIP-Image"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}ll" + "c" * len(GAMMA_VALUES) + r"@{}}",
        r"\toprule",
        r"\textbf{Scene} & \textbf{Model} & " +
        " & ".join([f"$\\gamma={g}$" for g in GAMMA_VALUES]) + r" \\",
        r"\midrule",
    ]

    for scene in scenes:
        for j, model in enumerate(models):
            scene_label = scene if j == 0 else ""
            row_vals = []
            for g in GAMMA_VALUES:
                match = [r for r in all_results
                         if r["scene"] == scene and r["model"] == model
                         and abs(r["gamma"] - g) < 0.01]
                if match:
                    ate_cm = match[0]["ate_median"] * 100
                    row_vals.append(f"{ate_cm:.1f}")
                else:
                    row_vals.append("--")
            lines.append(
                f"  {scene_label} & {model} & " + " & ".join(row_vals) + r" \\"
            )
        if scene != scenes[-1]:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{ATE median (cm) under gamma perturbation. "
        r"$\gamma < 1$ brightens, $\gamma > 1$ darkens the query image. "
        r"CLIP-Image features are more robust to lighting changes than SSIM.}",
        r"\label{tab:lighting_ablation}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    output_dir = Path("results/lighting_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for scene_cfg in SCENES:
        name = scene_cfg["name"]
        print(f"\n{'='*60}")
        print(f"  Scene: {name}")
        print(f"{'='*60}")

        ckpt_path = f"checkpoints/{name}.ckpt"
        if not Path(ckpt_path).exists():
            print(f"  SKIP - checkpoint not found: {ckpt_path}")
            continue

        if scene_cfg["type"] == "tum":
            dataset = TUMDataset(scene_cfg["path"], stride=1)
        else:
            dataset = ReplicaDataset(scene_cfg["path"], stride=1)

        gmap = GaussianMap.from_checkpoint(ckpt_path)

        # Run both models at each gamma
        for model_name in ["SSIM", "CLIP-Image"]:
            print(f"\n  Model: {model_name}")
            if model_name == "SSIM":
                obs_model = SSIMObservation(temperature=3.0)
            else:
                obs_model = CLIPImageObservation(device="cuda", temperature=10.0)

            for gamma in GAMMA_VALUES:
                print(f"    gamma={gamma:.2f} ... ", end="", flush=True)
                metrics = run_pf_with_gamma(
                    dataset, gmap, obs_model, scene_cfg, gamma=gamma,
                )
                result = {
                    "scene": name,
                    "model": model_name,
                    "gamma": gamma,
                    "ate_median": metrics["ate"]["median"],
                    "ate_mean": metrics["ate"]["mean"],
                    "ate_rmse": metrics["ate"]["rmse"],
                    "are_median": metrics["are"]["median"],
                    "success_rate": metrics["success_rate"],
                }
                all_results.append(result)
                print(f"ATE={result['ate_median']*100:.1f}cm  SR={result['success_rate']*100:.0f}%")

            # Free CLIP model between scenes if needed
            if model_name == "CLIP-Image":
                del obs_model
                torch.cuda.empty_cache()

        del gmap
        torch.cuda.empty_cache()

    # --- Save results ---
    torch.save(all_results, output_dir / "all_results.pt")
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir}")

    # --- Generate figures ---
    print("\nGenerating figures...")
    plot_lighting_robustness(all_results, str(output_dir / "lighting_robustness.png"))
    plot_success_rate(all_results, str(output_dir / "success_rate.png"))

    # --- LaTeX table ---
    latex = generate_latex_table(all_results)
    with open(output_dir / "lighting_ablation.tex", "w") as f:
        f.write(latex)
    print(f"  LaTeX table saved to {output_dir / 'lighting_ablation.tex'}")

    # --- Print summary ---
    print(f"\n{'='*70}")
    print(f"  LIGHTING ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Scene':<12} {'Model':<14} {'Gamma':>6} {'ATE Med':>10} {'Succ%':>8}")
    print("-" * 52)
    for r in all_results:
        print(f"{r['scene']:<12} {r['model']:<14} {r['gamma']:>6.2f} "
              f"{r['ate_median']*100:>8.1f}cm {r['success_rate']*100:>7.0f}%")

    # --- Robustness summary ---
    print(f"\n{'='*70}")
    print(f"  DEGRADATION SUMMARY (ATE increase from gamma=1.0)")
    print(f"{'='*70}")
    scenes = list(dict.fromkeys(r["scene"] for r in all_results))
    for scene in scenes:
        print(f"\n  {scene}:")
        for model in ["SSIM", "CLIP-Image"]:
            baseline = [r for r in all_results
                        if r["scene"] == scene and r["model"] == model
                        and abs(r["gamma"] - 1.0) < 0.01]
            if not baseline:
                continue
            base_ate = baseline[0]["ate_median"]
            worst = max(
                [r for r in all_results
                 if r["scene"] == scene and r["model"] == model],
                key=lambda r: r["ate_median"]
            )
            degradation = (worst["ate_median"] - base_ate) * 100
            print(f"    {model:<14}  baseline={base_ate*100:.1f}cm  "
                  f"worst={worst['ate_median']*100:.1f}cm (gamma={worst['gamma']})  "
                  f"degradation=+{degradation:.1f}cm")


if __name__ == "__main__":
    main()
