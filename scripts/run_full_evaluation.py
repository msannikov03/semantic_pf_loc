"""Run complete evaluation across all scenes, observation models, and configurations.
Produces results tables, figures, and LaTeX output."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pypose as pp
from pathlib import Path
from tqdm import tqdm

from semantic_pf_loc.gaussian_map import GaussianMap
from semantic_pf_loc.batch_renderer import BatchRenderer
from semantic_pf_loc.particle_filter import ParticleFilter
from semantic_pf_loc.motion_model import MotionModel
from semantic_pf_loc.gradient_refiner import GradientRefiner
from semantic_pf_loc.observation.ssim import SSIMObservation
from semantic_pf_loc.observation.clip_image import CLIPImageObservation
from semantic_pf_loc.observation.clip_text import CLIPTextObservation
from semantic_pf_loc.datasets.tum import TUMDataset
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.evaluation.metrics import compute_all_metrics, translation_error, rotation_error
from semantic_pf_loc.utils.visualization import (
    plot_trajectory_2d,
    plot_error_over_time,
    plot_convergence_comparison,
    plot_observation_model_comparison,
    generate_latex_table,
)


SCENES = [
    {"name": "fr1_desk", "type": "tum", "path": "data/tum/rgbd_dataset_freiburg1_desk",
     "native_size": (640, 480), "trans_std": 0.005, "rot_std": 0.003,
     "text_desc": "a wooden desk with a computer monitor and keyboard"},
    {"name": "fr3_office", "type": "tum", "path": "data/tum/rgbd_dataset_freiburg3_long_office_household",
     "native_size": (640, 480), "trans_std": 0.005, "rot_std": 0.003,
     "text_desc": "an office with desks and household items"},
    {"name": "room0", "type": "replica", "path": "data/replica/room0",
     "native_size": (1200, 680), "trans_std": 0.003, "rot_std": 0.002,
     "text_desc": "a living room with a sofa and table"},
    {"name": "room1", "type": "replica", "path": "data/replica/room1",
     "native_size": (1200, 680), "trans_std": 0.003, "rot_std": 0.002,
     "text_desc": "a bedroom with a bed and nightstand"},
    {"name": "office0", "type": "replica", "path": "data/replica/office0",
     "native_size": (1200, 680), "trans_std": 0.003, "rot_std": 0.002,
     "text_desc": "an office with a desk chair and computer"},
]


def run_pf(dataset, gmap, obs_model, motion_params, use_refine=False, n_frames=100, n_particles=200):
    """Run PF on a dataset, return metrics dict + error tensors."""
    K = dataset.get_intrinsics().float().to("cuda")
    renderer = BatchRenderer(gmap, width=160, height=120)
    motion = MotionModel(
        translation_std=motion_params["trans_std"],
        rotation_std=motion_params["rot_std"],
        device="cuda",
    )

    refiner = None
    if use_refine:
        refiner = GradientRefiner(renderer, num_iterations=20, lr_init=0.005, blur_schedule=True)

    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=n_particles, render_width=160, render_height=120,
        render_width_hires=320, render_height_hires=240,
        convergence_threshold=0.02, roughening_trans=0.002, roughening_rot=0.001,
        gradient_refiner=refiner, top_k_refine=5, device="cuda",
    )

    sample0 = dataset[0]
    gt_pose0 = pp.mat2SE3(
        sample0["pose"].double().to("cuda").unsqueeze(0), check=False
    ).squeeze(0).float()
    pf.initialize_around_pose(gt_pose0, trans_spread=0.03, rot_spread=0.01)

    est_poses, gt_poses, times = [], [], []
    for i in range(min(n_frames, len(dataset))):
        sample = dataset[i]
        if obs_model.requires_image:
            obs = {"image": sample["image"].float().to("cuda")}
        else:
            obs = {"text": motion_params.get("text_desc", "")}
        est, info = pf.step(obs, K)
        est_poses.append(est.matrix().cpu())
        gt_poses.append(sample["pose"].float())
        times.append(info["step_time_ms"])

    est_stack = torch.stack(est_poses)
    gt_stack = torch.stack(gt_poses)
    metrics = compute_all_metrics(est_stack, gt_stack, torch.tensor(times))
    metrics["estimated_poses"] = est_stack
    metrics["gt_poses"] = gt_stack

    del renderer, pf
    torch.cuda.empty_cache()
    return metrics


def main():
    output_dir = Path("results/full_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    N_FRAMES = 100
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

        convergence_data = {}

        # SSIM
        print(f"  Running SSIM...")
        obs_ssim = SSIMObservation(temperature=3.0)
        m_ssim = run_pf(dataset, gmap, obs_ssim, scene_cfg, n_frames=N_FRAMES)
        all_results.append({
            "scene": name, "model": "SSIM",
            "ate_median": m_ssim["ate"]["median"],
            "are_median": m_ssim["are"]["median"],
            "success_rate": m_ssim["success_rate"],
            "runtime_ms": m_ssim.get("runtime_mean_ms", 0),
        })
        convergence_data["SSIM"] = {"trans_errors": m_ssim["trans_errors"], "rot_errors": m_ssim["rot_errors"]}
        print(f"    ATE={m_ssim['ate']['median']:.3f}m  ARE={m_ssim['are']['median']:.1f}°  SR={m_ssim['success_rate']*100:.0f}%")

        # SSIM + Gradient Refinement
        print(f"  Running SSIM + GradRefine...")
        obs_ssim2 = SSIMObservation(temperature=3.0)
        m_ssim_ref = run_pf(dataset, gmap, obs_ssim2, scene_cfg, use_refine=True, n_frames=N_FRAMES)
        all_results.append({
            "scene": name, "model": "SSIM+Refine",
            "ate_median": m_ssim_ref["ate"]["median"],
            "are_median": m_ssim_ref["are"]["median"],
            "success_rate": m_ssim_ref["success_rate"],
            "runtime_ms": m_ssim_ref.get("runtime_mean_ms", 0),
        })
        convergence_data["SSIM+Refine"] = {"trans_errors": m_ssim_ref["trans_errors"], "rot_errors": m_ssim_ref["rot_errors"]}
        print(f"    ATE={m_ssim_ref['ate']['median']:.3f}m  ARE={m_ssim_ref['are']['median']:.1f}°  SR={m_ssim_ref['success_rate']*100:.0f}%")

        # CLIP-Image
        print(f"  Running CLIP-Image...")
        obs_clip = CLIPImageObservation(device="cuda", temperature=10.0)
        m_clip = run_pf(dataset, gmap, obs_clip, scene_cfg, n_frames=N_FRAMES)
        all_results.append({
            "scene": name, "model": "CLIP-Image",
            "ate_median": m_clip["ate"]["median"],
            "are_median": m_clip["are"]["median"],
            "success_rate": m_clip["success_rate"],
            "runtime_ms": m_clip.get("runtime_mean_ms", 0),
        })
        convergence_data["CLIP-Image"] = {"trans_errors": m_clip["trans_errors"], "rot_errors": m_clip["rot_errors"]}
        print(f"    ATE={m_clip['ate']['median']:.3f}m  ARE={m_clip['are']['median']:.1f}°  SR={m_clip['success_rate']*100:.0f}%")

        # CLIP-Text
        print(f"  Running CLIP-Text...")
        obs_text = CLIPTextObservation(device="cuda", temperature=10.0)
        m_text = run_pf(dataset, gmap, obs_text, {**scene_cfg, "text_desc": scene_cfg["text_desc"]}, n_frames=N_FRAMES)
        all_results.append({
            "scene": name, "model": "CLIP-Text",
            "ate_median": m_text["ate"]["median"],
            "are_median": m_text["are"]["median"],
            "success_rate": m_text["success_rate"],
            "runtime_ms": m_text.get("runtime_mean_ms", 0),
        })
        convergence_data["CLIP-Text"] = {"trans_errors": m_text["trans_errors"], "rot_errors": m_text["rot_errors"]}
        print(f"    ATE={m_text['ate']['median']:.3f}m  ARE={m_text['are']['median']:.1f}°  SR={m_text['success_rate']*100:.0f}%")

        # Generate per-scene figures
        # Trajectory plot (best method)
        best = m_ssim_ref if m_ssim_ref["ate"]["median"] < m_ssim["ate"]["median"] else m_ssim
        plot_trajectory_2d(
            best["estimated_poses"], best["gt_poses"],
            title=f"Trajectory: {name}",
            save_path=str(figures_dir / f"traj_{name}.png"),
        )

        # Convergence comparison
        plot_convergence_comparison(
            convergence_data, metric="trans",
            title=f"Convergence: {name}",
            save_path=str(figures_dir / f"conv_trans_{name}.png"),
        )
        plot_convergence_comparison(
            convergence_data, metric="rot",
            title=f"Rotation Convergence: {name}",
            save_path=str(figures_dir / f"conv_rot_{name}.png"),
        )

        # Error over time (SSIM+Refine)
        plot_error_over_time(
            m_ssim_ref["trans_errors"], m_ssim_ref["rot_errors"],
            title=f"Error: {name} (SSIM+Refine)",
            save_path=str(figures_dir / f"errors_{name}.png"),
        )

        del gmap
        torch.cuda.empty_cache()

    # Generate cross-scene comparison figure
    plot_observation_model_comparison(
        all_results,
        save_path=str(figures_dir / "model_comparison.png"),
    )

    # Generate LaTeX table
    latex = generate_latex_table(all_results, caption="Localization results across scenes and methods")
    with open(output_dir / "results_table.tex", "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {output_dir / 'results_table.tex'}")

    # Save all results
    torch.save(all_results, output_dir / "all_results.pt")

    # Print final summary
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scene':<12} {'Model':<14} {'ATE Med':>10} {'ARE Med':>10} {'Succ%':>8} {'ms':>6}")
    print("-" * 62)
    for r in all_results:
        print(f"{r['scene']:<12} {r['model']:<14} {r['ate_median']*100:>8.1f}cm {r['are_median']:>9.1f}° {r['success_rate']*100:>7.0f}% {r['runtime_ms']:>5.0f}")

    print(f"\nFigures saved to {figures_dir}/")
    print(f"Results saved to {output_dir / 'all_results.pt'}")


if __name__ == "__main__":
    main()
