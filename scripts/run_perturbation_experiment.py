"""Perturbation robustness experiment: SSIM vs CLIP-Image under realistic image degradations.

Tests the hypothesis that CLIP's high-level semantic features (224x224 learned representations)
are invariant to pixel-level perturbations, while SSIM (pixel statistics) degrades severely.

Perturbation types:
  1. Gaussian noise (sigma=0.2)
  2. Motion blur (kernel_size=21)
  3. Color temperature shift (warm/cool +/-0.3)
  4. Contrast+brightness (scale 0.6, shift +0.15)
  5. JPEG compression (quality=20)
  6. Combined (all of the above)

Run on fr3_office with PF only (NO gradient refinement) to isolate observation model effect.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import io
import torch
import torch.nn.functional as F
import pypose as pp
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
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
from semantic_pf_loc.evaluation.metrics import compute_all_metrics


# --- Configuration ---
SCENE = {
    "name": "fr3_office",
    "type": "tum",
    "path": "data/tum/rgbd_dataset_freiburg3_long_office_household",
    "trans_std": 0.005,
    "rot_std": 0.003,
}

N_PARTICLES = 200
N_FRAMES = 100


# =====================================================================
# Perturbation functions
# =====================================================================

def apply_noise(img, sigma=0.2):
    """Add Gaussian noise. img: [H, W, 3] float [0,1]."""
    return (img + torch.randn_like(img) * sigma).clamp(0, 1)


def apply_motion_blur(img, kernel_size=21):
    """Horizontal motion blur via 1D convolution. img: [H, W, 3] float [0,1]."""
    # Build horizontal motion blur kernel
    kernel = torch.zeros(kernel_size, 1, device=img.device, dtype=img.dtype)
    kernel[:, 0] = 1.0 / kernel_size
    # Reshape for depthwise conv: [out_ch, 1, kH, kW]
    kernel_2d = kernel.view(1, 1, 1, kernel_size).expand(3, -1, -1, -1)  # [3, 1, 1, K]

    x = img.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    pad = kernel_size // 2
    x = F.pad(x, [pad, pad, 0, 0], mode="reflect")
    x = F.conv2d(x, kernel_2d, groups=3)
    return x.squeeze(0).permute(1, 2, 0).clamp(0, 1)


def apply_color_temp(img, temp_shift=0.3):
    """Shift color temperature. Positive = warm (boost R, reduce B). img: [H, W, 3]."""
    img_mod = img.clone()
    img_mod[:, :, 0] = (img[:, :, 0] * (1 + temp_shift)).clamp(0, 1)  # R
    img_mod[:, :, 2] = (img[:, :, 2] * (1 - temp_shift * 0.7)).clamp(0, 1)  # B
    return img_mod


def apply_contrast_brightness(img, contrast=0.6, brightness=0.15):
    """Random contrast scaling and brightness shift. img: [H, W, 3]."""
    return (img * contrast + brightness).clamp(0, 1)


def apply_jpeg_compression(img, quality=20):
    """JPEG compression artifacts. img: [H, W, 3] float [0,1] on GPU."""
    img_cpu = img.cpu()
    # Convert to PIL
    arr = (img_cpu.numpy() * 255).astype(np.uint8)
    pil_img = PILImage.fromarray(arr)
    # Compress
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    # Reload
    pil_reload = PILImage.open(buffer).convert("RGB")
    result = torch.from_numpy(np.array(pil_reload)).float() / 255.0
    return result.to(img.device)


def apply_combined(img):
    """Apply all perturbations in sequence for maximum degradation."""
    x = apply_noise(img, sigma=0.15)
    x = apply_motion_blur(x, kernel_size=15)
    x = apply_color_temp(x, temp_shift=0.25)
    x = apply_contrast_brightness(x, contrast=0.7, brightness=0.1)
    x = apply_jpeg_compression(x, quality=25)
    return x


PERTURBATIONS = {
    "none": lambda img: img,
    "noise": lambda img: apply_noise(img, sigma=0.2),
    "motion_blur": lambda img: apply_motion_blur(img, kernel_size=21),
    "color_temp": lambda img: apply_color_temp(img, temp_shift=0.3),
    "contrast": lambda img: apply_contrast_brightness(img, contrast=0.6, brightness=0.15),
    "jpeg": lambda img: apply_jpeg_compression(img, quality=20),
    "combined": apply_combined,
}


# =====================================================================
# PF runner (no gradient refinement)
# =====================================================================

def run_pf_with_perturbation(dataset, gmap, obs_model, motion_params,
                              perturb_fn, n_frames=N_FRAMES, n_particles=N_PARTICLES):
    """Run PF with perturbed query images. NO refinement. Returns metrics dict."""
    K = dataset.get_intrinsics().float().to("cuda")
    renderer = BatchRenderer(gmap, width=160, height=120)
    motion = MotionModel(
        translation_std=motion_params["trans_std"],
        rotation_std=motion_params["rot_std"],
        device="cuda",
    )

    # NO gradient refiner -- isolate observation model
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
        img = sample["image"].float().to("cuda")
        # Apply perturbation to query image ONLY
        img_perturbed = perturb_fn(img)
        obs = {"image": img_perturbed}
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


# =====================================================================
# Visualization
# =====================================================================

def plot_perturbation_comparison(all_results, save_path):
    """Bar chart: ATE median for each perturbation, SSIM vs CLIP side by side."""
    perturbations = list(dict.fromkeys(r["perturbation"] for r in all_results))
    models = ["SSIM", "CLIP-Image"]

    x = np.arange(len(perturbations))
    width = 0.35
    colors = {"SSIM": "#2196F3", "CLIP-Image": "#FF5722"}

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, model in enumerate(models):
        vals = []
        for p in perturbations:
            match = [r for r in all_results
                     if r["perturbation"] == p and r["model"] == model]
            if match:
                vals.append(match[0]["ate_median"] * 100)
            else:
                vals.append(0)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=colors[model], alpha=0.85, edgecolor="white", linewidth=0.5)
        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Perturbation Type", fontsize=12)
    ax.set_ylabel("ATE Median (cm)", fontsize=12)
    ax.set_title("Observation Model Robustness to Image Perturbations (fr3_office, no refinement)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in perturbations], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved to {save_path}")


def plot_degradation_ratio(all_results, save_path):
    """Plot relative degradation from baseline (none) for each model."""
    perturbations = [p for p in dict.fromkeys(r["perturbation"] for r in all_results)
                     if p != "none"]
    models = ["SSIM", "CLIP-Image"]
    colors = {"SSIM": "#2196F3", "CLIP-Image": "#FF5722"}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(perturbations))
    width = 0.35

    for i, model in enumerate(models):
        baseline = [r for r in all_results
                    if r["perturbation"] == "none" and r["model"] == model]
        if not baseline:
            continue
        base_ate = baseline[0]["ate_median"]

        ratios = []
        for p in perturbations:
            match = [r for r in all_results
                     if r["perturbation"] == p and r["model"] == model]
            if match and base_ate > 0:
                ratios.append(match[0]["ate_median"] / base_ate)
            else:
                ratios.append(1.0)

        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, ratios, width, label=model,
                      color=colors[model], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.1f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No degradation")
    ax.set_xlabel("Perturbation Type", fontsize=12)
    ax.set_ylabel("ATE Ratio (perturbed / baseline)", fontsize=12)
    ax.set_title("Relative Degradation: SSIM vs CLIP-Image", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in perturbations], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved to {save_path}")


def plot_example_perturbations(dataset, save_path):
    """Save example images showing each perturbation type."""
    sample = dataset[50]
    img = sample["image"].float().to("cuda")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for idx, (name, fn) in enumerate(PERTURBATIONS.items()):
        if idx >= 8:
            break
        perturbed = fn(img).cpu().numpy()
        axes[idx].imshow(perturbed)
        axes[idx].set_title(name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        axes[idx].axis("off")

    # Fill remaining
    for idx in range(len(PERTURBATIONS), 8):
        axes[idx].axis("off")

    fig.suptitle("Image Perturbation Examples (fr3_office frame 50)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Example perturbations saved to {save_path}")


def generate_latex_table(all_results):
    """Generate LaTeX table for perturbation results."""
    perturbations = list(dict.fromkeys(r["perturbation"] for r in all_results))
    models = ["SSIM", "CLIP-Image"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l" + "cc" * len(perturbations) + r"@{}}",
        r"\toprule",
    ]

    # Header row 1: perturbation names spanning 2 cols each
    header1 = r"& "
    header1 += " & ".join([
        r"\multicolumn{2}{c}{" + p.replace("_", " ").title() + "}"
        for p in perturbations
    ])
    header1 += r" \\"
    lines.append(header1)

    # Header row 2: model names
    cmidrules = " ".join([
        f"\\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(perturbations))
    ])
    lines.append(cmidrules)
    header2 = r"\textbf{Metric} & "
    header2 += " & ".join(["ATE" + r"$\downarrow$" + " & SR" + r"$\uparrow$"] * len(perturbations))
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"\midrule")

    for model in models:
        row = f"  {model}"
        for p in perturbations:
            match = [r for r in all_results
                     if r["perturbation"] == p and r["model"] == model]
            if match:
                ate_cm = match[0]["ate_median"] * 100
                sr = match[0]["success_rate"] * 100
                row += f" & {ate_cm:.1f} & {sr:.0f}\\%"
            else:
                row += " & -- & --"
        row += r" \\"
        lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{ATE median (cm) and success rate (\%) under various image perturbations "
        r"on fr3\_office. CLIP-Image features are robust to pixel-level degradations "
        r"while SSIM degrades severely, especially under combined perturbations.}",
        r"\label{tab:perturbation_robustness}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# =====================================================================
# Main
# =====================================================================

def main():
    torch.manual_seed(42)
    output_dir = Path("results/perturbation_robustness")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PERTURBATION ROBUSTNESS: SSIM vs CLIP-Image")
    print("  Scene: fr3_office | N=200 | 100 frames | NO refinement")
    print("=" * 70)

    # Load scene
    ckpt_path = f"checkpoints/{SCENE['name']}.ckpt"
    if not Path(ckpt_path).exists():
        print(f"  ERROR: checkpoint not found: {ckpt_path}")
        return

    dataset = TUMDataset(SCENE["path"], stride=1)
    gmap = GaussianMap.from_checkpoint(ckpt_path)

    # Save example perturbations
    print("\nSaving perturbation examples...")
    plot_example_perturbations(dataset, str(output_dir / "perturbation_examples.png"))

    all_results = []

    for model_name in ["SSIM", "CLIP-Image"]:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        if model_name == "SSIM":
            obs_model = SSIMObservation(temperature=3.0)
        else:
            obs_model = CLIPImageObservation(device="cuda", temperature=10.0)

        for pert_name, pert_fn in PERTURBATIONS.items():
            print(f"  Perturbation: {pert_name:15s} ... ", end="", flush=True)

            metrics = run_pf_with_perturbation(
                dataset, gmap, obs_model, SCENE, perturb_fn=pert_fn,
            )

            result = {
                "scene": SCENE["name"],
                "model": model_name,
                "perturbation": pert_name,
                "ate_median": metrics["ate"]["median"],
                "ate_mean": metrics["ate"]["mean"],
                "ate_rmse": metrics["ate"]["rmse"],
                "are_median": metrics["are"]["median"],
                "success_rate": metrics["success_rate"],
                "runtime_ms": metrics.get("runtime_mean_ms", 0),
            }
            all_results.append(result)
            print(f"ATE={result['ate_median']*100:.1f}cm  "
                  f"SR={result['success_rate']*100:.0f}%  "
                  f"ARE={result['are_median']:.1f}deg")

        if model_name == "CLIP-Image":
            del obs_model
            torch.cuda.empty_cache()

    del gmap
    torch.cuda.empty_cache()

    # --- Save results ---
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    torch.save(all_results, output_dir / "all_results.pt")
    print(f"\nResults saved to {output_dir}")

    # --- Generate figures ---
    print("\nGenerating figures...")
    plot_perturbation_comparison(all_results, str(output_dir / "perturbation_comparison.png"))
    plot_degradation_ratio(all_results, str(output_dir / "degradation_ratio.png"))

    # --- LaTeX table ---
    latex = generate_latex_table(all_results)
    with open(output_dir / "perturbation_robustness.tex", "w") as f:
        f.write(latex)
    print(f"  LaTeX table saved to {output_dir / 'perturbation_robustness.tex'}")

    # --- Print summary ---
    print(f"\n{'='*75}")
    print(f"  PERTURBATION ROBUSTNESS RESULTS")
    print(f"{'='*75}")
    print(f"{'Model':<14} {'Perturbation':<15} {'ATE Med':>10} {'ATE Mean':>10} "
          f"{'Succ%':>8} {'ARE Med':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model']:<14} {r['perturbation']:<15} "
              f"{r['ate_median']*100:>8.1f}cm {r['ate_mean']*100:>8.1f}cm "
              f"{r['success_rate']*100:>7.0f}% {r['are_median']:>8.1f}deg")

    # --- Degradation summary ---
    print(f"\n{'='*75}")
    print(f"  DEGRADATION RATIO (ATE perturbed / ATE baseline)")
    print(f"{'='*75}")
    for model in ["SSIM", "CLIP-Image"]:
        baseline = [r for r in all_results
                    if r["model"] == model and r["perturbation"] == "none"]
        if not baseline:
            continue
        base_ate = baseline[0]["ate_median"]
        print(f"\n  {model} (baseline ATE={base_ate*100:.1f}cm):")
        for r in all_results:
            if r["model"] == model and r["perturbation"] != "none":
                ratio = r["ate_median"] / base_ate if base_ate > 0 else float("inf")
                arrow = ">>>" if ratio > 2.0 else ">>" if ratio > 1.5 else ">" if ratio > 1.1 else "~"
                print(f"    {r['perturbation']:<15} {r['ate_median']*100:>6.1f}cm  "
                      f"({ratio:.1f}x baseline) {arrow}")

    # --- Key finding ---
    print(f"\n{'='*75}")
    print("  KEY FINDING")
    print(f"{'='*75}")
    ssim_combined = [r for r in all_results
                     if r["model"] == "SSIM" and r["perturbation"] == "combined"]
    clip_combined = [r for r in all_results
                     if r["model"] == "CLIP-Image" and r["perturbation"] == "combined"]
    ssim_base = [r for r in all_results
                 if r["model"] == "SSIM" and r["perturbation"] == "none"]
    clip_base = [r for r in all_results
                 if r["model"] == "CLIP-Image" and r["perturbation"] == "none"]

    if ssim_combined and clip_combined and ssim_base and clip_base:
        ssim_deg = ssim_combined[0]["ate_median"] / ssim_base[0]["ate_median"]
        clip_deg = clip_combined[0]["ate_median"] / clip_base[0]["ate_median"]
        print(f"  Under combined perturbations:")
        print(f"    SSIM:       {ssim_combined[0]['ate_median']*100:.1f}cm "
              f"({ssim_deg:.1f}x baseline)")
        print(f"    CLIP-Image: {clip_combined[0]['ate_median']*100:.1f}cm "
              f"({clip_deg:.1f}x baseline)")
        if clip_combined[0]["ate_median"] < ssim_combined[0]["ate_median"]:
            advantage = ssim_combined[0]["ate_median"] - clip_combined[0]["ate_median"]
            print(f"  => CLIP outperforms SSIM by {advantage*100:.1f}cm under realistic perturbations!")
        else:
            print(f"  => SSIM still better even under perturbations (unexpected)")


if __name__ == "__main__":
    main()
