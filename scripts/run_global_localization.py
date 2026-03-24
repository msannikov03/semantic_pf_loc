"""Global localization demo: convergence from wide initialization.

Key findings from experiments:
  - Pure PF at 640x480 with temp=30 converges from 50cm spread to ~15cm in 25 frames
  - But then DRIFTS without gradient refinement (PF alone can't maintain precision)
  - Gradient refinement from the start pushes to wrong local optima

Solution: Two-phase within one PF run:
  Phase 1 (frames 0-30): Pure PF at 640x480, high temp, no gradient refinement
  Phase 2 (frames 30+): Enable gradient refinement on converged PF

Also: switch to standard PF class for Phase 2 (with hires rendering for refine)

Scene: office0 (best 3DGS map, 28 dB PSNR)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pypose as pp
import numpy as np
import math
import time
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_msssim import ssim

from semantic_pf_loc.gaussian_map import GaussianMap
from semantic_pf_loc.batch_renderer import BatchRenderer
from semantic_pf_loc.particle_filter import ParticleFilter
from semantic_pf_loc.motion_model import MotionModel
from semantic_pf_loc.gradient_refiner import GradientRefiner
from semantic_pf_loc.observation.ssim import SSIMObservation
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.evaluation.metrics import (
    translation_error,
    rotation_error,
    convergence_time,
)
from semantic_pf_loc.resampling import systematic_resample, normalize_log_weights, effective_sample_size
from semantic_pf_loc.utils.pose_utils import (
    se3_to_viewmat,
    scale_intrinsics,
    pose_error,
    weighted_se3_mean,
)


def score_particles(particles, obs_img, renderer, K_scaled, temperature, batch_size=80):
    N = particles.shape[0]
    device = particles.device
    log_w = torch.zeros(N, device=device)
    query_nchw = obs_img.permute(2, 0, 1).unsqueeze(0)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        bp = pp.SE3(particles.tensor()[s:e])
        B = e - s
        viewmats = se3_to_viewmat(bp)
        Ks = K_scaled.unsqueeze(0).expand(B, -1, -1)
        rendered, _, _ = renderer.render_batch(viewmats, Ks)
        r_nchw = rendered.permute(0, 3, 1, 2)
        t_nchw = query_nchw.expand(B, -1, -1, -1)
        ssim_vals = ssim(r_nchw, t_nchw, data_range=1.0, size_average=False)
        log_w[s:e] = temperature * torch.log(ssim_vals.clamp(min=1e-8))
    return log_w


def run_global_pf(
    dataset, gmap, gt_pose0_se3, K,
    trans_spread, rot_spread, n_particles, n_frames,
    label, phase1_frames=30,
):
    """Two-phase PF: Phase 1 (pure PF @ 640x480), Phase 2 (PF + gradient refine @ 160/320)."""
    print(f"\n  --- {label} ---")
    print(f"    {trans_spread}m/{math.degrees(rot_spread):.0f}deg  N={n_particles}  "
          f"Phase1={phase1_frames}f@640x480  Phase2={n_frames-phase1_frames}f+GradRefine")

    # Phase 1 renderer (high-res for discrimination)
    renderer_hi = BatchRenderer(gmap, width=640, height=480)
    K_hi = scale_intrinsics(K, (1200, 680), (640, 480))

    # Init particles
    noise = torch.zeros(n_particles, 6, device="cuda")
    noise[:, :3] = torch.randn(n_particles, 3, device="cuda") * trans_spread
    noise[:, 3:] = torch.randn(n_particles, 3, device="cuda") * rot_spread
    base = gt_pose0_se3.unsqueeze(0).expand(n_particles, -1)
    particles = pp.se3(noise).Exp() @ pp.SE3(base)
    weights = torch.ones(n_particles, device="cuda") / n_particles

    init_dists = (particles.tensor()[:, :3].cpu() - gt_pose0_se3.tensor()[:3].cpu().unsqueeze(0)).norm(dim=-1)
    print(f"    Init: min={init_dists.min():.3f}m  med={init_dists.median():.3f}m  max={init_dists.max():.3f}m")

    est_poses, gt_poses = [], []
    t_list, r_list, times = [], [], []
    step = 0

    # ===================== PHASE 1: Pure PF at 640x480 =====================
    print(f"    Phase 1: {phase1_frames} frames, pure PF @ 640x480, temp=30...")
    for i in tqdm(range(min(phase1_frames, len(dataset))), desc=f"{label} P1", leave=False):
        t0 = time.time()
        sample = dataset[i]
        obs_img = sample["image"].float().to("cuda")
        gt_mat = sample["pose"].float()

        obs_resized = torch.nn.functional.interpolate(
            obs_img.permute(2, 0, 1).unsqueeze(0),
            size=(480, 640), mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0)

        # Resample + roughening
        if step > 0:
            indices = systematic_resample(weights, n_particles)
            particles = pp.SE3(particles.tensor()[indices])
            rough = torch.zeros(n_particles, 6, device="cuda")
            rough[:, :3] = torch.randn(n_particles, 3, device="cuda") * 0.005
            rough[:, 3:] = torch.randn(n_particles, 3, device="cuda") * 0.003
            particles = pp.se3(rough).Exp() @ particles

        # Motion noise
        mn = torch.zeros(n_particles, 6, device="cuda")
        mn[:, :3] = torch.randn(n_particles, 3, device="cuda") * 0.003
        mn[:, 3:] = torch.randn(n_particles, 3, device="cuda") * 0.002
        particles = pp.se3(mn).Exp() @ particles

        # Score
        log_w = score_particles(particles, obs_resized, renderer_hi, K_hi, 30.0, batch_size=80)
        weights = normalize_log_weights(log_w).exp()
        n_eff = effective_sample_size(weights)

        # Estimate (best particle)
        est_se3 = particles[weights.argmax()]
        est_mat = est_se3.matrix().cpu().float()

        t_err, r_err = pose_error(est_mat, gt_mat)
        est_poses.append(est_mat)
        gt_poses.append(gt_mat)
        t_list.append(t_err)
        r_list.append(r_err)
        times.append((time.time() - t0) * 1000)
        step += 1

        if i % 10 == 0 or i == 0:
            trans = particles.tensor()[:, :3]
            mean_t = (weights.unsqueeze(-1) * trans).sum(0, keepdim=True)
            var = (weights.unsqueeze(-1) * (trans - mean_t)**2).sum().item()
            tqdm.write(
                f"    P1 F{i:3d}: ATE={t_err*100:.1f}cm  ARE={r_err:.1f}deg  "
                f"Neff={n_eff:.0f}  var={var:.5f}  {times[-1]:.0f}ms"
            )

    phase1_ate = t_list[-1]
    print(f"    Phase 1 end: ATE={phase1_ate*100:.1f}cm  ARE={r_list[-1]:.1f}deg")

    # Free hi-res renderer
    del renderer_hi
    torch.cuda.empty_cache()

    # ===================== PHASE 2: Standard PF + GradRefine =====================
    phase2_frames = n_frames - phase1_frames
    print(f"    Phase 2: {phase2_frames} frames, PF + GradRefine @ 160/320...")

    renderer = BatchRenderer(gmap, width=160, height=120)
    obs_model = SSIMObservation(temperature=3.0)
    motion = MotionModel(translation_std=0.003, rotation_std=0.002, device="cuda")
    refiner = GradientRefiner(renderer, num_iterations=20, lr_init=0.005, blur_schedule=True)

    # Reduce to 200 particles for Phase 2 (take top by weight)
    N_PHASE2 = min(200, n_particles)
    top_idx = torch.topk(weights, N_PHASE2).indices
    top_particles = pp.SE3(particles.tensor()[top_idx])

    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=N_PHASE2,
        render_width=160, render_height=120,
        render_width_hires=320, render_height_hires=240,
        convergence_threshold=0.02,
        roughening_trans=0.002, roughening_rot=0.001,
        gradient_refiner=refiner, top_k_refine=5,
        device="cuda",
    )
    pf.particles = top_particles
    pf.weights = torch.ones(N_PHASE2, device="cuda") / N_PHASE2
    pf._converged = False
    pf._prev_estimate = None
    pf._step_count = 0

    for i in tqdm(range(phase1_frames, phase1_frames + phase2_frames), desc=f"{label} P2", leave=False):
        if i >= len(dataset):
            break
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

        if (i - phase1_frames) % 20 == 0 or i == phase1_frames:
            tqdm.write(
                f"    P2 F{i:3d}: ATE={t_err*100:.2f}cm  ARE={r_err:.2f}deg  "
                f"Neff={info['n_eff']:.1f}  conv={info['converged']}  {info['step_time_ms']:.0f}ms"
            )

    del renderer, pf, refiner
    torch.cuda.empty_cache()

    # Metrics
    est_s = torch.stack(est_poses)
    gt_s = torch.stack(gt_poses)
    t_err = translation_error(est_s, gt_s)
    r_err = rotation_error(est_s, gt_s)
    c5 = convergence_time(t_err, 0.05)
    c10 = convergence_time(t_err, 0.10)
    nf = min(20, len(t_err))
    # Phase 2 success
    p2_t = t_err[phase1_frames:]
    p2_r = r_err[phase1_frames:]
    s5 = ((p2_t < 0.05) & (p2_r < 2.0)).float().mean().item() if len(p2_t) > 0 else 0.0

    res = {
        "label": label, "trans_spread": trans_spread,
        "phase1_frames": phase1_frames,
        "trans_errors": t_err, "rot_errors": r_err,
        "est_poses": est_s, "gt_poses": gt_s,
        "step_times": times,
        "conv_5cm": c5, "conv_10cm": c10,
        "ate_mean": t_err.mean().item(), "ate_median": t_err.median().item(),
        "ate_final": t_err[-nf:].mean().item(),
        "are_mean": r_err.mean().item(), "are_median": r_err.median().item(),
        "are_final": r_err[-nf:].mean().item(),
        "success": s5,
        "phase2_ate_med": p2_t.median().item() if len(p2_t) > 0 else 0,
        "phase2_are_med": p2_r.median().item() if len(p2_r) > 0 else 0,
    }
    print(f"    => Phase2 ATE_med={res['phase2_ate_med']*100:.1f}cm  "
          f"final={res['ate_final']*100:.1f}cm  conv<5cm=f{c5}  "
          f"success={s5*100:.0f}%")

    return res


def plot_convergence(results, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]
    for i, r in enumerate(results):
        t = r["trans_errors"].numpy()
        rot = r["rot_errors"].numpy()
        f = np.arange(len(t))
        c = colors[i % len(colors)]
        lbl = f"{r['label']} (final={r['ate_final']*100:.1f}cm)"
        ax1.semilogy(f, t, color=c, lw=1.5, alpha=0.85, label=lbl)
        ax2.semilogy(f, np.maximum(rot, 0.01), color=c, lw=1.5, alpha=0.85, label=r["label"])
        # Phase boundary
        pf = r.get("phase1_frames", 30)
        ax1.axvline(x=pf, color=c, ls=":", alpha=0.3)
        ax2.axvline(x=pf, color=c, ls=":", alpha=0.3)

    ax1.axhline(y=0.05, color="gray", ls="--", alpha=0.5, label="5cm")
    ax1.axhline(y=0.01, color="gray", ls=":", alpha=0.4, label="1cm")
    ax1.set_ylabel("Translation Error (m)")
    ax1.set_title("Global Localization (Phase 1: PF@640x480, Phase 2: PF+GradRefine)", fontsize=12)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax2.axhline(y=2.0, color="gray", ls="--", alpha=0.5, label="2deg")
    ax2.set_ylabel("Rotation Error (deg)")
    ax2.set_xlabel("Frame")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_trajectories(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4.5))
    if n == 1: axes = [axes]
    for i, r in enumerate(results):
        ax = axes[i]
        gt = r["gt_poses"][:, :3, 3].numpy()
        est = r["est_poses"][:, :3, 3].numpy()
        pf = r.get("phase1_frames", 30)
        ax.plot(gt[:, 0], gt[:, 2], "b-", lw=2, label="GT", alpha=0.8)
        if pf > 0:
            ax.plot(est[:pf, 0], est[:pf, 2], "r--", lw=1, label="P1", alpha=0.5)
        ax.plot(est[pf:, 0], est[pf:, 2], "r-", lw=1.5, label="P2", alpha=0.8)
        ax.plot(gt[0, 0], gt[0, 2], "go", ms=8)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{r['trans_spread']*100:.0f}cm init\nP2 med={r['phase2_ate_med']*100:.1f}cm", fontsize=10)
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Global Localization Trajectories", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    torch.manual_seed(42)
    print("=" * 70)
    print("  GLOBAL LOCALIZATION: Two-Phase Approach")
    print("  Phase 1: Pure PF @ 640x480 (convergence)")
    print("  Phase 2: Standard PF + GradRefine @ 160/320 (tracking)")
    print("=" * 70)

    output_dir = Path("results/global_localization")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ReplicaDataset("data/replica/office0", stride=1)
    K = dataset.get_intrinsics().float().to("cuda")
    gmap = GaussianMap.from_checkpoint("checkpoints/office0.ckpt")
    gt0 = pp.mat2SE3(
        dataset[0]["pose"].double().to("cuda").unsqueeze(0), check=False
    ).squeeze(0).float()

    N_FRAMES = 200
    experiments = [
        {"trans_spread": 0.10, "rot_spread": 0.05, "n_particles": 500, "label": "10cm spread"},
        {"trans_spread": 0.20, "rot_spread": 0.10, "n_particles": 500, "label": "20cm spread"},
        {"trans_spread": 0.30, "rot_spread": 0.15, "n_particles": 500, "label": "30cm spread"},
        {"trans_spread": 0.50, "rot_spread": 0.30, "n_particles": 1000, "label": "50cm spread"},
    ]

    results = []
    for exp in experiments:
        r = run_global_pf(
            dataset, gmap, gt0, K,
            trans_spread=exp["trans_spread"],
            rot_spread=exp["rot_spread"],
            n_particles=exp["n_particles"],
            n_frames=N_FRAMES,
            label=exp["label"],
            phase1_frames=30,
        )
        results.append(r)

    # Summary
    print(f"\n{'=' * 85}")
    print(f"  RESULTS (Two-Phase Global Localization)")
    print(f"{'=' * 85}")
    print(f"  {'Init':<15} {'P1 End':>10} {'P2 Med':>10} {'Final20':>10} "
          f"{'Conv<5cm':>10} {'P2 Succ':>8}")
    print(f"  {'-'*68}")
    for r in results:
        p1_end = r["trans_errors"][r["phase1_frames"]-1].item() if r["phase1_frames"] <= len(r["trans_errors"]) else 0
        print(f"  {r['label']:<15} {p1_end*100:>8.1f}cm "
              f"{r['phase2_ate_med']*100:>8.1f}cm {r['ate_final']*100:>8.1f}cm "
              f"{r['conv_5cm']:>10} {r['success']*100:>7.0f}%")
    print(f"{'=' * 85}")

    for r in results:
        print(f"\n  {r['label']}:")
        print(f"    Full: ATE mean={r['ate_mean']*100:.1f}cm  med={r['ate_median']*100:.1f}cm")
        print(f"    Phase2: ATE med={r['phase2_ate_med']*100:.1f}cm  ARE med={r['phase2_are_med']:.1f}deg")
        print(f"    Final 20: ATE={r['ate_final']*100:.1f}cm  ARE={r['are_final']:.1f}deg")
        print(f"    Conv <5cm: f{r['conv_5cm']}  <10cm: f{r['conv_10cm']}")
        print(f"    Avg step: {np.mean(r['step_times']):.0f}ms")

    plot_convergence(results, str(output_dir / "convergence_global.png"))
    plot_trajectories(results, str(output_dir / "trajectory_global.png"))
    torch.save(results, output_dir / "global_results.pt")
    print(f"\n  Saved to {output_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
