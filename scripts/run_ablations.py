"""Run ablation studies: particle count, render resolution, observation model comparison."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import torch
import pypose as pp
from pathlib import Path

from semantic_pf_loc.gaussian_map import GaussianMap
from semantic_pf_loc.batch_renderer import BatchRenderer
from semantic_pf_loc.particle_filter import ParticleFilter
from semantic_pf_loc.motion_model import MotionModel
from semantic_pf_loc.gradient_refiner import GradientRefiner
from semantic_pf_loc.observation.ssim import SSIMObservation
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.evaluation.metrics import compute_all_metrics
from semantic_pf_loc.utils.visualization import plot_ablation_particles


def run_single(dataset, gmap, n_particles, render_w, render_h, n_frames=100):
    """Run one PF configuration, return metrics."""
    K = dataset.get_intrinsics().float().to("cuda")
    renderer = BatchRenderer(gmap, width=render_w, height=render_h)
    obs = SSIMObservation(temperature=3.0)
    motion = MotionModel(translation_std=0.003, rotation_std=0.002, device="cuda")

    pf = ParticleFilter(
        gmap, renderer, obs, motion,
        num_particles=n_particles, render_width=render_w, render_height=render_h,
        render_width_hires=render_w * 2, render_height_hires=render_h * 2,
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
        obs_dict = {"image": sample["image"].float().to("cuda")}
        est, info = pf.step(obs_dict, K)
        est_poses.append(est.matrix().cpu())
        gt_poses.append(sample["pose"].float())
        times.append(info["step_time_ms"])

    m = compute_all_metrics(torch.stack(est_poses), torch.stack(gt_poses), torch.tensor(times))
    del renderer, pf
    torch.cuda.empty_cache()
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="office0", help="Scene to run ablation on")
    parser.add_argument("--output", default="results/ablations", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ReplicaDataset(f"data/replica/{args.scene}", stride=1)
    gmap = GaussianMap.from_checkpoint(f"checkpoints/{args.scene}.ckpt")

    # Ablation 1: Particle count
    print("=" * 50)
    print("Ablation 1: Particle Count")
    print("=" * 50)
    particle_results = {}
    for n_particles in [50, 100, 200, 400]:
        print(f"  N={n_particles}...", end=" ", flush=True)
        m = run_single(dataset, gmap, n_particles, 160, 120, n_frames=100)
        particle_results[n_particles] = {
            "ate_median": m["ate"]["median"],
            "are_median": m["are"]["median"],
            "success_rate": m["success_rate"],
            "runtime_ms": m.get("runtime_mean_ms", 0),
        }
        print(f"ATE={m['ate']['median']*100:.1f}cm  SR={m['success_rate']*100:.0f}%  {m.get('runtime_mean_ms',0):.0f}ms")

    plot_ablation_particles(
        particle_results,
        save_path=str(output_dir / f"ablation_particles_{args.scene}.png"),
    )

    # Ablation 2: Render resolution
    print("\n" + "=" * 50)
    print("Ablation 2: Render Resolution")
    print("=" * 50)
    for rw, rh in [(64, 48), (80, 60), (120, 90), (160, 120), (240, 180)]:
        print(f"  {rw}x{rh}...", end=" ", flush=True)
        m = run_single(dataset, gmap, 200, rw, rh, n_frames=100)
        print(f"ATE={m['ate']['median']*100:.1f}cm  SR={m['success_rate']*100:.0f}%  {m.get('runtime_mean_ms',0):.0f}ms")

    # Ablation 3: SSIM temperature
    print("\n" + "=" * 50)
    print("Ablation 3: SSIM Temperature")
    print("=" * 50)
    for temp in [1.0, 2.0, 3.0, 5.0, 10.0]:
        K = dataset.get_intrinsics().float().to("cuda")
        renderer = BatchRenderer(gmap, width=160, height=120)
        obs = SSIMObservation(temperature=temp)
        motion = MotionModel(translation_std=0.003, rotation_std=0.002, device="cuda")
        pf = ParticleFilter(gmap, renderer, obs, motion,
            num_particles=200, render_width=160, render_height=120,
            render_width_hires=320, render_height_hires=240,
            convergence_threshold=0.02, roughening_trans=0.002, roughening_rot=0.001,
            device="cuda")

        sample0 = dataset[0]
        gt_pose0 = pp.mat2SE3(sample0["pose"].double().to("cuda").unsqueeze(0), check=False).squeeze(0).float()
        pf.initialize_around_pose(gt_pose0, trans_spread=0.03, rot_spread=0.01)

        est_poses, gt_poses = [], []
        for i in range(min(100, len(dataset))):
            s = dataset[i]
            est, info = pf.step({"image": s["image"].float().to("cuda")}, K)
            est_poses.append(est.matrix().cpu())
            gt_poses.append(s["pose"].float())

        m = compute_all_metrics(torch.stack(est_poses), torch.stack(gt_poses))
        print(f"  temp={temp:<5.1f}  ATE={m['ate']['median']*100:.1f}cm  ARE={m['are']['median']:.1f}°  SR={m['success_rate']*100:.0f}%")
        del renderer, pf; torch.cuda.empty_cache()

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
