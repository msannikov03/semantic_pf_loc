"""Visualization utilities for localization results."""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def plot_trajectory_2d(
    estimated: torch.Tensor,
    ground_truth: torch.Tensor,
    title: str = "Trajectory",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot 2D top-down trajectory (XZ plane).

    Args:
        estimated: [T, 4, 4] estimated poses
        ground_truth: [T, 4, 4] GT poses
    """
    est_t = estimated[:, :3, 3].numpy()
    gt_t = ground_truth[:, :3, 3].numpy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(gt_t[:, 0], gt_t[:, 2], "b-", linewidth=2, label="Ground Truth", alpha=0.8)
    ax.plot(est_t[:, 0], est_t[:, 2], "r--", linewidth=1.5, label="Estimated", alpha=0.8)
    ax.plot(gt_t[0, 0], gt_t[0, 2], "go", markersize=10, label="Start")
    ax.plot(gt_t[-1, 0], gt_t[-1, 2], "rs", markersize=10, label="End")

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Z (m)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_over_time(
    trans_errors: torch.Tensor,
    rot_errors: torch.Tensor,
    title: str = "Localization Error",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Plot translation and rotation error over frames."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    frames = np.arange(len(trans_errors))

    ax1.plot(frames, trans_errors.numpy(), "b-", linewidth=1)
    ax1.axhline(y=0.05, color="g", linestyle="--", alpha=0.5, label="5cm threshold")
    ax1.set_xlabel("Frame", fontsize=11)
    ax1.set_ylabel("Translation Error (m)", fontsize=11)
    ax1.set_title("Translation Error", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.plot(frames, rot_errors.numpy(), "r-", linewidth=1)
    ax2.axhline(y=2.0, color="g", linestyle="--", alpha=0.5, label="2° threshold")
    ax2.set_xlabel("Frame", fontsize=11)
    ax2.set_ylabel("Rotation Error (°)", fontsize=11)
    ax2.set_title("Rotation Error", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_convergence_comparison(
    results: dict[str, dict],
    metric: str = "trans",
    title: str = "Convergence Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Plot convergence curves for multiple methods.

    Args:
        results: {method_name: {"trans_errors": Tensor, "rot_errors": Tensor}}
        metric: "trans" or "rot"
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    key = "trans_errors" if metric == "trans" else "rot_errors"
    ylabel = "Translation Error (m)" if metric == "trans" else "Rotation Error (°)"
    threshold = 0.05 if metric == "trans" else 2.0
    thresh_label = "5cm" if metric == "trans" else "2°"

    for i, (name, data) in enumerate(results.items()):
        errors = data[key].numpy() if isinstance(data[key], torch.Tensor) else np.array(data[key])
        ax.plot(np.arange(len(errors)), errors, color=colors[i % len(colors)],
                linewidth=1.5, label=name, alpha=0.85)

    ax.axhline(y=threshold, color="gray", linestyle="--", alpha=0.5, label=f"{thresh_label} threshold")
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_observation_model_comparison(
    results_table: list[dict],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Bar chart comparing observation models across scenes.

    Args:
        results_table: [{"scene": str, "model": str, "ate_median": float, "are_median": float, "success_rate": float}]
    """
    import pandas as pd
    df = pd.DataFrame(results_table)

    scenes = df["scene"].unique()
    models = df["model"].unique()
    x = np.arange(len(scenes))
    width = 0.8 / len(models)
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for i, model in enumerate(models):
        model_data = df[df["model"] == model]
        vals_ate = [model_data[model_data["scene"] == s]["ate_median"].values[0] * 100 for s in scenes]
        vals_sr = [model_data[model_data["scene"] == s]["success_rate"].values[0] * 100 for s in scenes]

        ax1.bar(x + i * width, vals_ate, width, label=model, color=colors[i % len(colors)], alpha=0.85)
        ax2.bar(x + i * width, vals_sr, width, label=model, color=colors[i % len(colors)], alpha=0.85)

    ax1.set_ylabel("ATE Median (cm)", fontsize=11)
    ax1.set_title("Translation Error by Scene", fontsize=12)
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(scenes, fontsize=9, rotation=15)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.set_ylabel("Success Rate (%)", fontsize=11)
    ax2.set_title("Success Rate by Scene", fontsize=12)
    ax2.set_xticks(x + width * (len(models) - 1) / 2)
    ax2.set_xticklabels(scenes, fontsize=9, rotation=15)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 105)

    fig.suptitle("Observation Model Comparison", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ablation_particles(
    results: dict[int, dict],
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Plot ATE and runtime vs particle count.

    Args:
        results: {num_particles: {"ate_median": float, "runtime_ms": float}}
    """
    counts = sorted(results.keys())
    ates = [results[n]["ate_median"] * 100 for n in counts]
    runtimes = [results[n]["runtime_ms"] for n in counts]

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    color1, color2 = "#2196F3", "#FF5722"

    ax1.plot(counts, ates, "o-", color=color1, linewidth=2, markersize=8, label="ATE Median (cm)")
    ax1.set_xlabel("Number of Particles", fontsize=12)
    ax1.set_ylabel("ATE Median (cm)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(counts, runtimes, "s--", color=color2, linewidth=2, markersize=8, label="Runtime (ms)")
    ax2.set_ylabel("Runtime per step (ms)", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")

    ax1.set_title("Particle Count vs Accuracy & Speed", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def generate_latex_table(
    results_table: list[dict],
    caption: str = "Localization Results",
) -> str:
    """Generate a LaTeX table from results.

    Args:
        results_table: [{"scene": str, "model": str, "ate_median": float,
                         "are_median": float, "success_rate": float, "runtime_ms": float}]
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"\textbf{Scene} & \textbf{Method} & \textbf{ATE Med.} & \textbf{ARE Med.} & \textbf{Succ. (\%)} & \textbf{ms/step} \\",
        r"\midrule",
    ]

    prev_scene = None
    for row in results_table:
        scene = row["scene"] if row["scene"] != prev_scene else ""
        prev_scene = row["scene"]
        ate = f"{row['ate_median']*100:.1f} cm"
        are = f"{row['are_median']:.1f}°"
        sr = f"{row['success_rate']*100:.0f}"
        rt = f"{row['runtime_ms']:.0f}"
        lines.append(f"  {scene} & {row['model']} & {ate} & {are} & {sr} & {rt} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)
