"""GSLoc baseline: gradient-only localization without particle filter.

Takes GT pose, adds noise, runs GradientRefiner to optimize.
Tests across multiple noise levels and scenes to compare vs PF+Refine.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import time
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
from semantic_pf_loc.gradient_refiner import GradientRefiner
from semantic_pf_loc.datasets.tum import TUMDataset
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.evaluation.metrics import translation_error, rotation_error
from semantic_pf_loc.utils.pose_utils import scale_intrinsics


# ---------- Configuration ----------

SCENES = [
    {
        "name": "office0",
        "type": "replica",
        "path": "data/replica/office0",
        "native_size": (1200, 680),
    },
    {
        "name": "room0",
        "type": "replica",
        "path": "data/replica/room0",
        "native_size": (1200, 680),
    },
    {
        "name": "fr3_office",
        "type": "tum",
        "path": "data/tum/rgbd_dataset_freiburg3_long_office_household",
        "native_size": (640, 480),
    },
]

# Noise levels: (trans_spread_m, rot_spread_rad)
NOISE_LEVELS = [
    (0.01, 0.01 / 3),
    (0.03, 0.01),       # same as PF init
    (0.05, 0.05 / 3),
    (0.10, 0.10 / 3),
    (0.20, 0.20 / 3),
]

N_FRAMES = 100
RENDER_W = 320
RENDER_H = 240
NUM_ITERATIONS = 50
LR_INIT = 0.01
SUCCESS_TRANS = 0.05   # 5 cm
SUCCESS_ROT = 2.0      # 2 degrees

# Our PF+Refine reference results (from full evaluation, 100 frames)
PF_REFINE_RESULTS = {
    "office0": {"ate_median_cm": 1.4, "are_median_deg": 0.4, "success_rate": 0.74},
    "room0":   {"ate_median_cm": 1.9, "are_median_deg": 0.4, "success_rate": 0.99},
    "fr3_office": {"ate_median_cm": 3.1, "are_median_deg": 0.8, "success_rate": 0.96},
}


def run_gsloc_single_noise(dataset, gmap, scene_cfg, trans_spread, rot_spread):
    """Run GSLoc for one scene at one noise level. Returns dict of metrics."""

    device = "cuda"
    K = dataset.get_intrinsics().float().to(device)
    native_w, native_h = scene_cfg["native_size"]
    K_scaled = scale_intrinsics(K, (native_w, native_h), (RENDER_W, RENDER_H))

    renderer = BatchRenderer(gmap, width=RENDER_W, height=RENDER_H)
    refiner = GradientRefiner(
        renderer,
        num_iterations=NUM_ITERATIONS,
        lr_init=LR_INIT,
        blur_schedule=True,
    )

    n_frames = min(N_FRAMES, len(dataset))
    trans_errors = []
    rot_errors = []
    init_trans_errors = []
    init_rot_errors = []
    runtimes = []

    for i in tqdm(range(n_frames), desc=f"  noise={trans_spread:.2f}m", leave=False):
        sample = dataset[i]
        gt_pose_mat = sample["pose"]  # [4,4] float64
        query_image = sample["image"].float().to(device)  # [H, W, 3]

        # Convert GT to SE3
        gt_se3 = pp.mat2SE3(
            gt_pose_mat.double().to(device).unsqueeze(0), check=False
        ).squeeze(0).float()

        # Add noise: Exp(noise) @ gt_pose
        # randn_se3 takes sigma=(sigma_trans, sigma_rot)
        noise = pp.randn_se3(1, sigma=(trans_spread, rot_spread)).to(device)
        init_pose = noise.Exp() @ gt_se3.unsqueeze(0)  # [1, 7]

        # Measure initial error
        init_mat = init_pose.matrix().cpu().float()     # [1, 4, 4]
        gt_mat = gt_pose_mat.float().unsqueeze(0)       # [1, 4, 4]
        init_te = translation_error(init_mat, gt_mat).item()
        init_re = rotation_error(init_mat, gt_mat).item()
        init_trans_errors.append(init_te)
        init_rot_errors.append(init_re)

        # Refine
        t0 = time.time()
        with torch.no_grad():
            pass  # just for timing fairness
        refined = refiner.refine(init_pose, query_image, K_scaled)  # [1, 7]
        torch.cuda.synchronize()
        runtime_ms = (time.time() - t0) * 1000
        runtimes.append(runtime_ms)

        # Measure refined error
        refined_mat = refined.matrix().detach().cpu().float()  # [1, 4, 4]
        te = translation_error(refined_mat, gt_mat).item()
        re = rotation_error(refined_mat, gt_mat).item()
        trans_errors.append(te)
        rot_errors.append(re)

    del renderer, refiner
    torch.cuda.empty_cache()

    trans_errors_t = torch.tensor(trans_errors)
    rot_errors_t = torch.tensor(rot_errors)
    init_trans_t = torch.tensor(init_trans_errors)
    init_rot_t = torch.tensor(init_rot_errors)

    success = ((trans_errors_t < SUCCESS_TRANS) & (rot_errors_t < SUCCESS_ROT)).float().mean().item()

    return {
        "ate_median": trans_errors_t.median().item(),
        "ate_mean": trans_errors_t.mean().item(),
        "are_median": rot_errors_t.median().item(),
        "are_mean": rot_errors_t.mean().item(),
        "success_rate": success,
        "runtime_mean_ms": np.mean(runtimes),
        "init_ate_median": init_trans_t.median().item(),
        "init_are_median": init_rot_t.median().item(),
        "trans_errors": trans_errors,
        "rot_errors": rot_errors,
        "init_trans_errors": init_trans_errors,
        "init_rot_errors": init_rot_errors,
    }


def plot_noise_vs_ate(all_results, output_dir):
    """Plot noise level vs ATE for each scene, with PF+Refine reference line."""
    fig, axes = plt.subplots(1, len(SCENES), figsize=(5 * len(SCENES), 5), squeeze=False)
    colors_gsloc = "#d62728"
    colors_pf = "#2196F3"

    for j, scene_cfg in enumerate(SCENES):
        ax = axes[0][j]
        name = scene_cfg["name"]

        noise_vals = []
        ate_vals = []
        for (ts, rs), metrics in all_results[name].items():
            noise_vals.append(ts * 100)  # cm
            ate_vals.append(metrics["ate_median"] * 100)  # cm

        ax.plot(noise_vals, ate_vals, "o-", color=colors_gsloc, linewidth=2,
                markersize=7, label="GSLoc (gradient-only)")

        # PF+Refine reference
        if name in PF_REFINE_RESULTS:
            pf_ate = PF_REFINE_RESULTS[name]["ate_median_cm"]
            ax.axhline(y=pf_ate, color=colors_pf, linestyle="--", linewidth=2,
                       label=f"PF+Refine ({pf_ate:.1f} cm)")

        # 5cm success threshold
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


def plot_noise_vs_success(all_results, output_dir):
    """Plot noise level vs success rate for each scene."""
    fig, axes = plt.subplots(1, len(SCENES), figsize=(5 * len(SCENES), 5), squeeze=False)
    colors_gsloc = "#d62728"
    colors_pf = "#2196F3"

    for j, scene_cfg in enumerate(SCENES):
        ax = axes[0][j]
        name = scene_cfg["name"]

        noise_vals = []
        sr_vals = []
        for (ts, rs), metrics in all_results[name].items():
            noise_vals.append(ts * 100)  # cm
            sr_vals.append(metrics["success_rate"] * 100)  # %

        ax.plot(noise_vals, sr_vals, "s-", color=colors_gsloc, linewidth=2,
                markersize=7, label="GSLoc (gradient-only)")

        # PF+Refine reference
        if name in PF_REFINE_RESULTS:
            pf_sr = PF_REFINE_RESULTS[name]["success_rate"] * 100
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


def main():
    output_dir = Path("results/gsloc_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}  # scene -> {(ts, rs): metrics}

    for scene_cfg in SCENES:
        name = scene_cfg["name"]
        print(f"\n{'='*60}")
        print(f"  Scene: {name}")
        print(f"{'='*60}")

        ckpt_path = f"checkpoints/{name}.ckpt"
        if not Path(ckpt_path).exists():
            print(f"  SKIP - checkpoint not found: {ckpt_path}")
            continue

        # Load dataset
        if scene_cfg["type"] == "tum":
            dataset = TUMDataset(scene_cfg["path"], stride=1)
        else:
            dataset = ReplicaDataset(scene_cfg["path"], stride=1)

        # Load map
        gmap = GaussianMap.from_checkpoint(ckpt_path)

        all_results[name] = {}

        for trans_spread, rot_spread in NOISE_LEVELS:
            print(f"\n  Noise: trans={trans_spread:.2f}m, rot={rot_spread:.4f}rad")
            metrics = run_gsloc_single_noise(
                dataset, gmap, scene_cfg, trans_spread, rot_spread
            )
            all_results[name][(trans_spread, rot_spread)] = metrics

            print(f"    Init  ATE={metrics['init_ate_median']*100:.1f}cm  ARE={metrics['init_are_median']:.1f}deg")
            print(f"    Final ATE={metrics['ate_median']*100:.1f}cm  ARE={metrics['are_median']:.1f}deg  SR={metrics['success_rate']*100:.0f}%  ({metrics['runtime_mean_ms']:.0f}ms/frame)")

        del gmap
        torch.cuda.empty_cache()

    # ---------- Summary Table ----------
    print(f"\n\n{'='*80}")
    print(f"  GSLoc BASELINE RESULTS")
    print(f"{'='*80}")

    # Table 1: Default noise level (0.03m / 0.01rad) — direct comparison with PF+Refine
    default_noise = (0.03, 0.01)
    print(f"\n--- Default noise ({default_noise[0]*100:.0f}cm / {default_noise[1]:.2f}rad) vs PF+Refine ---")
    print(f"{'Scene':<14} {'Method':<16} {'ATE Med':>10} {'ARE Med':>10} {'Succ%':>8}")
    print("-" * 60)
    for scene_cfg in SCENES:
        name = scene_cfg["name"]
        if name not in all_results:
            continue
        if default_noise in all_results[name]:
            m = all_results[name][default_noise]
            print(f"{name:<14} {'GSLoc':<16} {m['ate_median']*100:>8.1f}cm {m['are_median']:>9.1f}deg {m['success_rate']*100:>7.0f}%")
        if name in PF_REFINE_RESULTS:
            pf = PF_REFINE_RESULTS[name]
            print(f"{'':<14} {'PF+Refine':<16} {pf['ate_median_cm']:>8.1f}cm {pf['are_median_deg']:>9.1f}deg {pf['success_rate']*100:>7.0f}%")

    # Table 2: All noise levels
    print(f"\n--- All noise levels ---")
    print(f"{'Scene':<14} {'Noise(cm)':>10} {'Init ATE':>10} {'Final ATE':>10} {'ARE':>10} {'Succ%':>8}")
    print("-" * 65)
    for scene_cfg in SCENES:
        name = scene_cfg["name"]
        if name not in all_results:
            continue
        for (ts, rs), m in all_results[name].items():
            print(f"{name:<14} {ts*100:>8.0f}cm {m['init_ate_median']*100:>8.1f}cm {m['ate_median']*100:>8.1f}cm {m['are_median']:>9.1f}deg {m['success_rate']*100:>7.0f}%")

    # ---------- Generate Plots ----------
    print(f"\n--- Generating plots ---")
    plot_noise_vs_ate(all_results, output_dir)
    plot_noise_vs_success(all_results, output_dir)

    # ---------- Save raw results ----------
    # Convert to JSON-serializable format
    serializable = {}
    for scene_name, noise_dict in all_results.items():
        serializable[scene_name] = {}
        for (ts, rs), metrics in noise_dict.items():
            key = f"{ts:.3f}_{rs:.4f}"
            serializable[scene_name][key] = {
                k: v for k, v in metrics.items()
                if k not in ("trans_errors", "rot_errors", "init_trans_errors", "init_rot_errors")
            }
    with open(output_dir / "results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {output_dir / 'results.json'}")

    # Also save full results with tensors
    torch.save(all_results, output_dir / "all_results.pt")
    print(f"  Saved: {output_dir / 'all_results.pt'}")

    print(f"\nDone! Results in {output_dir}/")


if __name__ == "__main__":
    main()
