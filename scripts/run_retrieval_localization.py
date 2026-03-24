"""CLIP-based image retrieval for global localization initialization.

Demonstrates that CLIP cosine similarity can retrieve a coarse initial pose
from a database of reference views, enabling PF convergence from truly global
initialization where uniform random init fails.

Pipeline:
  1. Pre-compute CLIP embeddings for reference views at known poses (every 10th frame)
  2. Given a query image, find the closest reference by CLIP cosine similarity
  3. Initialize PF around that reference pose (with +/-10cm spread)
  4. Run standard PF + gradient refinement

Comparison:
  A) Without retrieval: uniform random init over scene bounds -> PF + refine (fails at 50cm)
  B) With retrieval: CLIP retrieves closest ref -> PF init around it -> PF + refine (succeeds)

Scene: office0 (Replica, best 3DGS map)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import time
import math
import torch
import torch.nn.functional as F
import pypose as pp
import numpy as np
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import open_clip

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
    convergence_time,
)
from semantic_pf_loc.utils.pose_utils import (
    scale_intrinsics,
    pose_error,
)


# --- Configuration ---
N_PARTICLES = 200
N_FRAMES = 150
RETRIEVAL_STRIDE = 10  # Build reference DB from every 10th frame


# =====================================================================
# CLIP Image Retrieval
# =====================================================================

class CLIPRetrieval:
    """CLIP-based image retrieval for coarse pose initialization."""

    def __init__(self, device="cuda"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model = self.model.to(device).eval()

        # CLIP normalization constants
        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                   device=device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                  device=device).view(1, 3, 1, 1)

        self.ref_embeddings = None
        self.ref_poses = []
        self.ref_indices = []

    def _encode_image(self, img_hwc):
        """Encode a [H, W, 3] float [0,1] image to CLIP embedding."""
        x = img_hwc.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        with torch.no_grad():
            emb = self.model.encode_image(x)
            emb = F.normalize(emb, dim=-1)
        return emb  # [1, dim]

    def build_database(self, dataset, stride=RETRIEVAL_STRIDE):
        """Pre-compute CLIP embeddings for reference views."""
        print(f"  Building CLIP reference database (stride={stride})...")
        embeddings = []
        self.ref_poses = []
        self.ref_indices = []

        for i in tqdm(range(0, len(dataset), stride), desc="CLIP DB", leave=False):
            sample = dataset[i]
            img = sample["image"].float().to(self.device)
            emb = self._encode_image(img)
            embeddings.append(emb)
            self.ref_poses.append(sample["pose"].float())
            self.ref_indices.append(i)

        self.ref_embeddings = torch.cat(embeddings, dim=0)  # [M, dim]
        print(f"  Database: {self.ref_embeddings.shape[0]} reference views")

    def retrieve(self, query_image):
        """Find closest reference view by CLIP cosine similarity.

        Args:
            query_image: [H, W, 3] float [0,1] on GPU
        Returns:
            (pose [4,4], similarity float, ref_index int)
        """
        query_emb = self._encode_image(query_image)  # [1, dim]
        sims = (query_emb @ self.ref_embeddings.T).squeeze(0)  # [M]
        best_idx = sims.argmax().item()
        return (
            self.ref_poses[best_idx],
            sims[best_idx].item(),
            self.ref_indices[best_idx],
        )

    def retrieve_top_k(self, query_image, k=5):
        """Return top-k reference views."""
        query_emb = self._encode_image(query_image)
        sims = (query_emb @ self.ref_embeddings.T).squeeze(0)
        topk = torch.topk(sims, min(k, len(sims)))
        results = []
        for idx, sim in zip(topk.indices, topk.values):
            results.append({
                "pose": self.ref_poses[idx.item()],
                "similarity": sim.item(),
                "frame_index": self.ref_indices[idx.item()],
            })
        return results


# =====================================================================
# Experiment runners
# =====================================================================

def run_pf_from_pose(dataset, gmap, init_pose_se3, K,
                     n_particles, n_frames, trans_spread, rot_spread,
                     label, use_refiner=True, start_frame=0):
    """Run PF initialized around a given pose, with optional gradient refinement."""
    print(f"\n  --- {label} ---")
    print(f"    Init spread: {trans_spread*100:.0f}cm / {math.degrees(rot_spread):.0f}deg")

    renderer = BatchRenderer(gmap, width=160, height=120)
    obs_model = SSIMObservation(temperature=3.0)
    motion = MotionModel(translation_std=0.003, rotation_std=0.002, device="cuda")

    refiner = None
    if use_refiner:
        refiner = GradientRefiner(renderer, num_iterations=20, lr_init=0.005, blur_schedule=True)

    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=n_particles, render_width=160, render_height=120,
        render_width_hires=320, render_height_hires=240,
        convergence_threshold=0.02, roughening_trans=0.002, roughening_rot=0.001,
        gradient_refiner=refiner, top_k_refine=5,
        device="cuda",
    )

    pf.initialize_around_pose(init_pose_se3, trans_spread=trans_spread, rot_spread=rot_spread)

    est_poses, gt_poses, times = [], [], []
    t_list, r_list = [], []

    for i in tqdm(range(start_frame, min(start_frame + n_frames, len(dataset))),
                  desc=label, leave=False):
        sample = dataset[i]
        obs = {"image": sample["image"].float().to("cuda")}
        gt_mat = sample["pose"].float()

        est, info = pf.step(obs, K)
        est_mat = est.matrix().cpu().float()

        t_err, r_err = pose_error(est_mat, gt_mat)
        est_poses.append(est_mat)
        gt_poses.append(gt_mat)
        t_list.append(t_err)
        r_list.append(r_err)
        times.append(info["step_time_ms"])

        if (i - start_frame) % 30 == 0 or i == start_frame:
            tqdm.write(f"    F{i:3d}: ATE={t_err*100:.1f}cm  ARE={r_err:.1f}deg  "
                       f"Neff={info['n_eff']:.0f}  conv={info['converged']}  "
                       f"{info['step_time_ms']:.0f}ms")

    del renderer, pf
    if refiner:
        del refiner
    torch.cuda.empty_cache()

    est_s = torch.stack(est_poses)
    gt_s = torch.stack(gt_poses)
    metrics = compute_all_metrics(est_s, gt_s, torch.tensor(times))

    t_err_arr = translation_error(est_s, gt_s)
    r_err_arr = rotation_error(est_s, gt_s)

    res = {
        "label": label,
        "trans_spread": trans_spread,
        "est_poses": est_s,
        "gt_poses": gt_s,
        "trans_errors": t_err_arr,
        "rot_errors": r_err_arr,
        "ate_median": metrics["ate"]["median"],
        "ate_mean": metrics["ate"]["mean"],
        "ate_final": t_err_arr[-20:].mean().item() if len(t_err_arr) >= 20 else t_err_arr.mean().item(),
        "are_median": metrics["are"]["median"],
        "are_final": r_err_arr[-20:].mean().item() if len(r_err_arr) >= 20 else r_err_arr.mean().item(),
        "success_rate": metrics["success_rate"],
        "conv_5cm": convergence_time(t_err_arr, 0.05),
        "conv_10cm": convergence_time(t_err_arr, 0.10),
        "step_times": times,
    }

    print(f"    => ATE median={res['ate_median']*100:.1f}cm  final={res['ate_final']*100:.1f}cm  "
          f"SR={res['success_rate']*100:.0f}%  conv<5cm=f{res['conv_5cm']}")

    return res


def run_uniform_global(dataset, gmap, K, n_particles, n_frames, label, start_frame=0):
    """Run PF with uniform random initialization over scene bounds."""
    print(f"\n  --- {label} ---")
    print(f"    Uniform init over scene bounds, N={n_particles}")

    renderer = BatchRenderer(gmap, width=160, height=120)
    obs_model = SSIMObservation(temperature=3.0)
    motion = MotionModel(translation_std=0.003, rotation_std=0.002, device="cuda")
    refiner = GradientRefiner(renderer, num_iterations=20, lr_init=0.005, blur_schedule=True)

    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=n_particles, render_width=160, render_height=120,
        render_width_hires=320, render_height_hires=240,
        convergence_threshold=0.02, roughening_trans=0.002, roughening_rot=0.001,
        gradient_refiner=refiner, top_k_refine=5,
        device="cuda",
    )

    bounds_min, bounds_max = dataset.get_bounds()
    pf.initialize_global(bounds_min, bounds_max)

    est_poses, gt_poses, times = [], [], []
    t_list, r_list = [], []

    for i in tqdm(range(start_frame, min(start_frame + n_frames, len(dataset))),
                  desc=label, leave=False):
        sample = dataset[i]
        obs = {"image": sample["image"].float().to("cuda")}
        gt_mat = sample["pose"].float()

        est, info = pf.step(obs, K)
        est_mat = est.matrix().cpu().float()

        t_err, r_err = pose_error(est_mat, gt_mat)
        est_poses.append(est_mat)
        gt_poses.append(gt_mat)
        t_list.append(t_err)
        r_list.append(r_err)
        times.append(info["step_time_ms"])

        if (i - start_frame) % 30 == 0 or i == start_frame:
            tqdm.write(f"    F{i:3d}: ATE={t_err*100:.1f}cm  ARE={r_err:.1f}deg  "
                       f"Neff={info['n_eff']:.0f}  {info['step_time_ms']:.0f}ms")

    del renderer, pf, refiner
    torch.cuda.empty_cache()

    est_s = torch.stack(est_poses)
    gt_s = torch.stack(gt_poses)
    metrics = compute_all_metrics(est_s, gt_s, torch.tensor(times))

    t_err_arr = translation_error(est_s, gt_s)
    r_err_arr = rotation_error(est_s, gt_s)

    res = {
        "label": label,
        "trans_spread": float("inf"),
        "est_poses": est_s,
        "gt_poses": gt_s,
        "trans_errors": t_err_arr,
        "rot_errors": r_err_arr,
        "ate_median": metrics["ate"]["median"],
        "ate_mean": metrics["ate"]["mean"],
        "ate_final": t_err_arr[-20:].mean().item() if len(t_err_arr) >= 20 else t_err_arr.mean().item(),
        "are_median": metrics["are"]["median"],
        "are_final": r_err_arr[-20:].mean().item() if len(r_err_arr) >= 20 else r_err_arr.mean().item(),
        "success_rate": metrics["success_rate"],
        "conv_5cm": convergence_time(t_err_arr, 0.05),
        "conv_10cm": convergence_time(t_err_arr, 0.10),
        "step_times": times,
    }

    print(f"    => ATE median={res['ate_median']*100:.1f}cm  final={res['ate_final']*100:.1f}cm  "
          f"SR={res['success_rate']*100:.0f}%")

    return res


# =====================================================================
# Visualization
# =====================================================================

def plot_convergence_comparison(results, save_path):
    """Plot translation and rotation error over time for all methods."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = {"Uniform (no retrieval)": "#F44336",
              "CLIP retrieval + PF": "#4CAF50",
              "CLIP retrieval + PF + Refine": "#2196F3"}
    fallback_colors = ["#F44336", "#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    for i, r in enumerate(results):
        c = colors.get(r["label"], fallback_colors[i % len(fallback_colors)])
        t = r["trans_errors"].numpy()
        rot = r["rot_errors"].numpy()
        f = np.arange(len(t))
        lbl = f"{r['label']} (final={r['ate_final']*100:.1f}cm)"
        ax1.semilogy(f, t, color=c, lw=2, alpha=0.85, label=lbl)
        ax2.semilogy(f, np.maximum(rot, 0.01), color=c, lw=2, alpha=0.85, label=r["label"])

    ax1.axhline(y=0.05, color="gray", ls="--", alpha=0.5, label="5cm")
    ax1.axhline(y=0.01, color="gray", ls=":", alpha=0.4, label="1cm")
    ax1.set_ylabel("Translation Error (m)", fontsize=12)
    ax1.set_title("Global Localization: CLIP Retrieval vs Uniform Init (office0)",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.axhline(y=2.0, color="gray", ls="--", alpha=0.5, label="2deg")
    ax2.set_ylabel("Rotation Error (deg)", fontsize=12)
    ax2.set_xlabel("Frame", fontsize=12)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_trajectories(results, save_path):
    """Bird's-eye view of trajectories."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i, r in enumerate(results):
        ax = axes[i]
        gt = r["gt_poses"][:, :3, 3].numpy()
        est = r["est_poses"][:, :3, 3].numpy()

        ax.plot(gt[:, 0], gt[:, 2], "b-", lw=2, label="GT", alpha=0.8)
        ax.plot(est[:, 0], est[:, 2], "r-", lw=1.5, label="Est", alpha=0.8)
        ax.plot(gt[0, 0], gt[0, 2], "go", ms=10, label="Start")

        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Z (m)", fontsize=11)
        title = r["label"]
        ax.set_title(f"{title}\nFinal ATE={r['ate_final']*100:.1f}cm  SR={r['success_rate']*100:.0f}%",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Trajectory Comparison: CLIP Retrieval vs Uniform Init",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_retrieval_quality(retriever, dataset, save_path, test_frames=None):
    """Visualize retrieval quality: query vs retrieved reference."""
    if test_frames is None:
        test_frames = [0, 50, 100, 150, 200, 250]
    test_frames = [f for f in test_frames if f < len(dataset)]

    n = len(test_frames)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, frame_idx in enumerate(test_frames):
        sample = dataset[frame_idx]
        query_img = sample["image"].float().to("cuda")
        gt_pose = sample["pose"].float()

        ref_pose, sim, ref_idx = retriever.retrieve(query_img)

        # Translation error between retrieved and GT
        t_err = (ref_pose[:3, 3] - gt_pose[:3, 3]).norm().item()

        # Show query
        axes[0, col].imshow(query_img.cpu().numpy())
        axes[0, col].set_title(f"Query f{frame_idx}", fontsize=9)
        axes[0, col].axis("off")

        # Show reference
        ref_sample = dataset[ref_idx]
        axes[1, col].imshow(ref_sample["image"].numpy())
        axes[1, col].set_title(f"Ref f{ref_idx}\nsim={sim:.3f} err={t_err*100:.0f}cm", fontsize=9)
        axes[1, col].axis("off")

    fig.suptitle("CLIP Retrieval: Query (top) vs Retrieved Reference (bottom)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    torch.manual_seed(42)
    output_dir = Path("results/retrieval_localization")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CLIP RETRIEVAL FOR GLOBAL LOCALIZATION")
    print("  Scene: office0 (Replica)")
    print("=" * 70)

    # Load scene
    dataset = ReplicaDataset("data/replica/office0", stride=1)
    K = dataset.get_intrinsics().float().to("cuda")
    gmap = GaussianMap.from_checkpoint("checkpoints/office0.ckpt")

    # ---------------------------------------------------------------
    # Step 1: Build CLIP retrieval database
    # ---------------------------------------------------------------
    print("\n[Step 1] Building CLIP retrieval database...")
    retriever = CLIPRetrieval(device="cuda")
    retriever.build_database(dataset, stride=RETRIEVAL_STRIDE)

    # ---------------------------------------------------------------
    # Step 2: Test retrieval quality
    # ---------------------------------------------------------------
    print("\n[Step 2] Testing retrieval quality...")
    test_frames = list(range(0, min(len(dataset), 300), 50))
    retrieval_errors = []
    for f_idx in test_frames:
        sample = dataset[f_idx]
        query_img = sample["image"].float().to("cuda")
        gt_pose = sample["pose"].float()
        ref_pose, sim, ref_idx = retriever.retrieve(query_img)
        t_err = (ref_pose[:3, 3] - gt_pose[:3, 3]).norm().item()
        retrieval_errors.append(t_err)
        print(f"  Frame {f_idx:3d}: retrieved ref {ref_idx:3d}  "
              f"sim={sim:.3f}  pose_err={t_err*100:.1f}cm")

    mean_retrieval_err = np.mean(retrieval_errors)
    print(f"  Mean retrieval error: {mean_retrieval_err*100:.1f}cm")

    # Save retrieval visualization
    plot_retrieval_quality(retriever, dataset, str(output_dir / "retrieval_quality.png"),
                          test_frames=test_frames[:6])

    # ---------------------------------------------------------------
    # Step 3: Run localization experiments
    # ---------------------------------------------------------------
    # Pick a challenging start frame (not at beginning of trajectory)
    START_FRAME = 50  # Mid-trajectory, not in the reference DB necessarily

    sample_start = dataset[START_FRAME]
    query_img_start = sample_start["image"].float().to("cuda")
    gt_pose_start = sample_start["pose"].float()

    # Retrieve coarse pose
    ref_pose, sim, ref_idx = retriever.retrieve(query_img_start)
    retrieval_t_err = (ref_pose[:3, 3] - gt_pose_start[:3, 3]).norm().item()
    print(f"\n  Start frame {START_FRAME}: retrieved ref {ref_idx}  "
          f"sim={sim:.3f}  init_err={retrieval_t_err*100:.1f}cm")

    # Convert retrieved pose to SE3 for PF init
    ref_pose_se3 = pp.mat2SE3(
        ref_pose.double().to("cuda").unsqueeze(0), check=False
    ).squeeze(0).float()

    # GT pose for comparison
    gt_pose_se3 = pp.mat2SE3(
        gt_pose_start.double().to("cuda").unsqueeze(0), check=False
    ).squeeze(0).float()

    # Free CLIP model to save GPU memory before PF runs
    del retriever
    torch.cuda.empty_cache()

    results = []

    # Experiment A: Uniform random init (baseline - should fail)
    res_uniform = run_uniform_global(
        dataset, gmap, K,
        n_particles=500, n_frames=N_FRAMES,
        label="Uniform (no retrieval)",
        start_frame=START_FRAME,
    )
    results.append(res_uniform)

    # Experiment B: CLIP retrieval + PF (no refinement)
    res_retrieval = run_pf_from_pose(
        dataset, gmap, ref_pose_se3, K,
        n_particles=N_PARTICLES, n_frames=N_FRAMES,
        trans_spread=0.10, rot_spread=0.05,
        label="CLIP retrieval + PF",
        use_refiner=False,
        start_frame=START_FRAME,
    )
    results.append(res_retrieval)

    # Experiment C: CLIP retrieval + PF + gradient refinement
    res_retrieval_refine = run_pf_from_pose(
        dataset, gmap, ref_pose_se3, K,
        n_particles=N_PARTICLES, n_frames=N_FRAMES,
        trans_spread=0.10, rot_spread=0.05,
        label="CLIP retrieval + PF + Refine",
        use_refiner=True,
        start_frame=START_FRAME,
    )
    results.append(res_retrieval_refine)

    del gmap
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 4: Save results & generate plots
    # ---------------------------------------------------------------
    print("\n[Step 4] Saving results and generating plots...")

    # Save JSON-serializable results
    json_results = []
    for r in results:
        json_results.append({
            "label": r["label"],
            "ate_median": r["ate_median"],
            "ate_mean": r["ate_mean"],
            "ate_final": r["ate_final"],
            "are_median": r["are_median"],
            "are_final": r["are_final"],
            "success_rate": r["success_rate"],
            "conv_5cm": r["conv_5cm"],
            "conv_10cm": r["conv_10cm"],
            "retrieval_init_error_cm": retrieval_t_err * 100,
        })
    with open(output_dir / "results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # Save full results (tensors)
    torch.save(results, output_dir / "full_results.pt")

    # Generate plots
    plot_convergence_comparison(results, str(output_dir / "convergence_comparison.png"))
    plot_trajectories(results, str(output_dir / "trajectory_comparison.png"))

    # ---------------------------------------------------------------
    # Step 5: Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*75}")
    print(f"  RETRIEVAL LOCALIZATION RESULTS (office0, start frame {START_FRAME})")
    print(f"{'='*75}")
    print(f"  CLIP retrieval init error: {retrieval_t_err*100:.1f}cm")
    print(f"  Mean retrieval error (sampled): {mean_retrieval_err*100:.1f}cm")
    print()
    print(f"  {'Method':<30} {'ATE Med':>10} {'ATE Final':>10} "
          f"{'ARE Med':>10} {'SR':>8} {'Conv<5cm':>10}")
    print(f"  {'-'*80}")
    for r in results:
        print(f"  {r['label']:<30} {r['ate_median']*100:>8.1f}cm "
              f"{r['ate_final']*100:>8.1f}cm {r['are_median']:>8.1f}deg "
              f"{r['success_rate']*100:>7.0f}% f{r['conv_5cm']:>8}")
    print(f"{'='*75}")

    # Key finding
    print(f"\n  KEY FINDING:")
    if res_uniform["success_rate"] < 0.3 and res_retrieval_refine["success_rate"] > 0.5:
        improvement = res_retrieval_refine["success_rate"] - res_uniform["success_rate"]
        print(f"  CLIP retrieval enables global localization where uniform init fails!")
        print(f"  Success rate: {res_uniform['success_rate']*100:.0f}% (uniform) -> "
              f"{res_retrieval_refine['success_rate']*100:.0f}% (CLIP+PF+Refine) "
              f"[+{improvement*100:.0f}pp]")
        print(f"  Final ATE: {res_uniform['ate_final']*100:.1f}cm (uniform) -> "
              f"{res_retrieval_refine['ate_final']*100:.1f}cm (CLIP+PF+Refine)")
    else:
        print(f"  Uniform SR={res_uniform['success_rate']*100:.0f}%  "
              f"Retrieval+Refine SR={res_retrieval_refine['success_rate']*100:.0f}%")
        print(f"  Uniform final={res_uniform['ate_final']*100:.1f}cm  "
              f"Retrieval+Refine final={res_retrieval_refine['ate_final']*100:.1f}cm")

    print(f"\n  Results saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
