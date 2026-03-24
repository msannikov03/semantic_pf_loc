"""DEFINITIVE final evaluation for the Semantic PF Localization project.

Produces the numbers that go in the report:
- 3 good scenes (office0, room0, fr3_office) x 4 observation models x 3 trials
- 2 weak scenes (fr1_desk, room1) x SSIM+Refine only x 3 trials
- Median across trials for each config
- Figures: trajectories, convergence curves, bar chart, LaTeX table
- Full JSON dump of all results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pypose as pp
import json
import time
import gc
from pathlib import Path
from datetime import datetime

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
from semantic_pf_loc.evaluation.metrics import (
    compute_all_metrics, translation_error, rotation_error,
)
from semantic_pf_loc.utils.pose_utils import scale_intrinsics, se3_to_viewmat
from semantic_pf_loc.utils.visualization import (
    plot_trajectory_2d, plot_convergence_comparison,
    plot_error_over_time, plot_observation_model_comparison,
    generate_latex_table,
)


# ============================================================================
# Scene configurations
# ============================================================================

GOOD_SCENES = [
    {
        "name": "office0", "type": "replica",
        "path": "data/replica/office0",
        "ckpt": "checkpoints_depth/office0.ckpt",
        "native_w": 1200, "native_h": 680,
        "trans_std": 0.003, "rot_std": 0.002,
        "text_desc": "an office with a desk chair and computer",
    },
    {
        "name": "room0", "type": "replica",
        "path": "data/replica/room0",
        "ckpt": "checkpoints_depth/room0.ckpt",
        "native_w": 1200, "native_h": 680,
        "trans_std": 0.003, "rot_std": 0.002,
        "text_desc": "a living room with a sofa and table",
    },
    {
        "name": "fr3_office", "type": "tum",
        "path": "data/tum/rgbd_dataset_freiburg3_long_office_household",
        "ckpt": "checkpoints_depth/fr3_office.ckpt",
        "native_w": 640, "native_h": 480,
        "trans_std": 0.005, "rot_std": 0.003,
        "text_desc": "an office with desks and household items",
    },
]

WEAK_SCENES = [
    {
        "name": "room1", "type": "replica",
        "path": "data/replica/room1",
        "ckpt": "checkpoints/room1.ckpt",  # old checkpoint (depth didn't help)
        "native_w": 1200, "native_h": 680,
        "trans_std": 0.003, "rot_std": 0.002,
        "text_desc": "a bedroom with a bed and nightstand",
    },
    {
        "name": "fr1_desk", "type": "tum",
        "path": "data/tum/rgbd_dataset_freiburg1_desk",
        "ckpt": "checkpoints/fr1_desk.ckpt",  # old checkpoint
        "native_w": 640, "native_h": 480,
        "trans_std": 0.005, "rot_std": 0.003,
        "text_desc": "a wooden desk with a computer monitor and keyboard",
    },
]

# Observation model factories
OBS_MODELS = [
    ("SSIM",       lambda: SSIMObservation(temperature=3.0),                    False),
    ("SSIM+Refine", lambda: SSIMObservation(temperature=3.0),                   True),
    ("CLIP-Image", lambda: CLIPImageObservation(device="cuda", temperature=10.0), False),
    ("CLIP-Text",  lambda: CLIPTextObservation(device="cuda", temperature=10.0),  False),
]

N_TRIALS = 3
N_FRAMES = 100
N_PARTICLES = 200
DEVICE = "cuda"


# ============================================================================
# Core evaluation function
# ============================================================================

def run_pf_with_tuned_refine(dataset, gmap, obs_model, scene_cfg,
                              use_refine=False, n_frames=N_FRAMES,
                              n_particles=N_PARTICLES, seed=None):
    """Run a single PF trial with optional post-hoc single-pose refinement.

    The tuned refiner settings: 320x240, 100 iterations, lr=0.01, blur schedule.
    Refinement is applied as post-hoc on the PF weighted mean (not inline top-k).
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    K = dataset.get_intrinsics().float().to(DEVICE)
    native_size = (scene_cfg["native_w"], scene_cfg["native_h"])

    # Low-res renderer for PF
    renderer = BatchRenderer(gmap, width=160, height=120)
    motion = MotionModel(
        translation_std=scene_cfg["trans_std"],
        rotation_std=scene_cfg["rot_std"],
        device=DEVICE,
    )

    # PF WITHOUT any built-in refiner (we do post-hoc refinement)
    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=n_particles,
        render_width=160, render_height=120,
        render_width_hires=320, render_height_hires=240,
        convergence_threshold=0.02,
        roughening_trans=0.002, roughening_rot=0.001,
        gradient_refiner=None,
        device=DEVICE,
    )

    # Setup tuned refiner if needed
    hires_renderer = None
    refiner = None
    K_hires = None
    if use_refine:
        hires_renderer = BatchRenderer(gmap, width=320, height=240)
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
    for i in range(min(n_frames, len(dataset))):
        sample = dataset[i]
        if obs_model.requires_image:
            obs = {"image": sample["image"].float().to(DEVICE)}
        else:
            obs = {"text": scene_cfg.get("text_desc", "")}

        est, info = pf.step(obs, K)

        # Post-hoc single-pose refinement on converged frames
        if use_refine and refiner is not None and info["converged"]:
            est_for_refine = pp.SE3(est.tensor().unsqueeze(0))
            with torch.enable_grad():
                refined = refiner.refine(
                    est_for_refine, sample["image"].float().to(DEVICE), K_hires
                )
            est = refined.squeeze(0)

        est_poses.append(est.matrix().cpu())
        gt_poses.append(sample["pose"].float())
        times.append(info["step_time_ms"])

    est_stack = torch.stack(est_poses)
    gt_stack = torch.stack(gt_poses)
    metrics = compute_all_metrics(est_stack, gt_stack, torch.tensor(times))
    metrics["estimated_poses"] = est_stack
    metrics["gt_poses"] = gt_stack

    # Cleanup
    del renderer, pf
    if hires_renderer is not None:
        del hires_renderer
    if refiner is not None:
        del refiner
    torch.cuda.empty_cache()
    gc.collect()

    return metrics


# ============================================================================
# Main evaluation loop
# ============================================================================

def main():
    torch.set_grad_enabled(False)

    output_dir = Path("results/final_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    start_time = datetime.now()
    print("=" * 70)
    print("  DEFINITIVE FINAL EVALUATION")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Trials per config: {N_TRIALS}")
    print(f"  Frames per trial: {N_FRAMES}")
    print(f"  Particles: {N_PARTICLES}")
    print("=" * 70)

    # Storage for all results
    all_results = []       # flat list for LaTeX/bar chart
    detailed_results = {}  # nested: scene -> model -> {trials, median_metrics}
    convergence_data = {}  # scene -> model -> {trans_errors, rot_errors} from median trial

    # ================================================================
    # Part 1: Good scenes x all 4 models x 3 trials
    # ================================================================
    for scene_cfg in GOOD_SCENES:
        scene_name = scene_cfg["name"]
        print(f"\n{'='*70}")
        print(f"  SCENE: {scene_name} (checkpoint: {scene_cfg['ckpt']})")
        print(f"{'='*70}")

        # Verify checkpoint exists
        if not Path(scene_cfg["ckpt"]).exists():
            print(f"  SKIP - checkpoint not found: {scene_cfg['ckpt']}")
            continue

        # Load dataset
        if scene_cfg["type"] == "tum":
            dataset = TUMDataset(scene_cfg["path"], stride=1)
        else:
            dataset = ReplicaDataset(scene_cfg["path"], stride=1)

        # Load 3DGS map
        gmap = GaussianMap.from_checkpoint(scene_cfg["ckpt"])
        print(f"  Loaded: {gmap.num_gaussians} Gaussians, dataset: {len(dataset)} frames")

        scene_convergence = {}

        for model_name, create_obs, use_refine in OBS_MODELS:
            print(f"\n  --- {model_name} ---")

            trial_ates = []
            trial_ares = []
            trial_srs = []
            trial_runtimes = []
            trial_metrics = []

            for trial in range(N_TRIALS):
                seed = 42 + trial * 1000
                obs = create_obs()
                t0 = time.time()
                m = run_pf_with_tuned_refine(
                    dataset, gmap, obs, scene_cfg,
                    use_refine=use_refine, seed=seed,
                )
                elapsed = time.time() - t0

                ate_med = m["ate"]["median"]
                are_med = m["are"]["median"]
                sr = m["success_rate"]
                rt = m.get("runtime_mean_ms", 0)

                trial_ates.append(ate_med)
                trial_ares.append(are_med)
                trial_srs.append(sr)
                trial_runtimes.append(rt)
                trial_metrics.append(m)

                print(f"    Trial {trial+1}/{N_TRIALS}: "
                      f"ATE={ate_med*100:.2f}cm  ARE={are_med:.2f}deg  "
                      f"SR={sr*100:.0f}%  ({elapsed:.1f}s)")

                # Cleanup obs model (especially CLIP models to free VRAM)
                del obs
                torch.cuda.empty_cache()
                gc.collect()

            # Take median across trials (index 1 of sorted list of 3)
            sorted_ate_idx = sorted(range(N_TRIALS), key=lambda i: trial_ates[i])
            median_idx = sorted_ate_idx[1]  # median of 3

            final_ate = trial_ates[median_idx]
            final_are = trial_ares[median_idx]
            final_sr = trial_srs[median_idx]
            final_rt = trial_runtimes[median_idx]

            print(f"    >> MEDIAN: ATE={final_ate*100:.2f}cm  ARE={final_are:.2f}deg  "
                  f"SR={final_sr*100:.0f}%")

            # Store for table
            all_results.append({
                "scene": scene_name,
                "model": model_name,
                "ate_median": final_ate,
                "are_median": final_are,
                "success_rate": final_sr,
                "runtime_ms": final_rt,
            })

            # Store convergence data from median trial
            median_m = trial_metrics[median_idx]
            scene_convergence[model_name] = {
                "trans_errors": median_m["trans_errors"],
                "rot_errors": median_m["rot_errors"],
            }

            # Store detailed results
            if scene_name not in detailed_results:
                detailed_results[scene_name] = {}
            detailed_results[scene_name][model_name] = {
                "trials": [
                    {
                        "ate_median": trial_ates[i],
                        "are_median": trial_ares[i],
                        "success_rate": trial_srs[i],
                        "runtime_ms": trial_runtimes[i],
                    }
                    for i in range(N_TRIALS)
                ],
                "median_ate": final_ate,
                "median_are": final_are,
                "median_sr": final_sr,
                "median_rt": final_rt,
                "estimated_poses": median_m["estimated_poses"],
                "gt_poses": median_m["gt_poses"],
            }

        # ---- Generate per-scene figures ----
        convergence_data[scene_name] = scene_convergence

        # Find best method for trajectory plot
        best_model = min(
            detailed_results[scene_name].keys(),
            key=lambda m: detailed_results[scene_name][m]["median_ate"],
        )
        best_data = detailed_results[scene_name][best_model]

        plot_trajectory_2d(
            best_data["estimated_poses"], best_data["gt_poses"],
            title=f"Trajectory: {scene_name} ({best_model})",
            save_path=str(figures_dir / f"traj_{scene_name}.png"),
        )
        import matplotlib.pyplot as plt
        plt.close("all")

        # Convergence comparison (all 4 models)
        plot_convergence_comparison(
            scene_convergence, metric="trans",
            title=f"Translation Convergence: {scene_name}",
            save_path=str(figures_dir / f"conv_trans_{scene_name}.png"),
        )
        plt.close("all")
        plot_convergence_comparison(
            scene_convergence, metric="rot",
            title=f"Rotation Convergence: {scene_name}",
            save_path=str(figures_dir / f"conv_rot_{scene_name}.png"),
        )
        plt.close("all")

        # Error over time for best method
        best_conv = scene_convergence[best_model]
        plot_error_over_time(
            best_conv["trans_errors"], best_conv["rot_errors"],
            title=f"Error: {scene_name} ({best_model})",
            save_path=str(figures_dir / f"errors_{scene_name}.png"),
        )
        plt.close("all")

        # Cleanup scene
        del gmap, dataset
        torch.cuda.empty_cache()
        gc.collect()

    # ================================================================
    # Part 2: Weak scenes x SSIM+Refine only x 3 trials
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  WEAK SCENES (SSIM+Refine only)")
    print(f"{'='*70}")

    for scene_cfg in WEAK_SCENES:
        scene_name = scene_cfg["name"]
        print(f"\n  --- {scene_name} ---")

        if not Path(scene_cfg["ckpt"]).exists():
            print(f"  SKIP - checkpoint not found: {scene_cfg['ckpt']}")
            continue

        if scene_cfg["type"] == "tum":
            dataset = TUMDataset(scene_cfg["path"], stride=1)
        else:
            dataset = ReplicaDataset(scene_cfg["path"], stride=1)

        gmap = GaussianMap.from_checkpoint(scene_cfg["ckpt"])
        print(f"  Loaded: {gmap.num_gaussians} Gaussians, dataset: {len(dataset)} frames")

        model_name = "SSIM+Refine"
        trial_ates, trial_ares, trial_srs, trial_runtimes = [], [], [], []
        trial_metrics = []

        for trial in range(N_TRIALS):
            seed = 42 + trial * 1000
            obs = SSIMObservation(temperature=3.0)
            t0 = time.time()
            m = run_pf_with_tuned_refine(
                dataset, gmap, obs, scene_cfg,
                use_refine=True, seed=seed,
            )
            elapsed = time.time() - t0

            trial_ates.append(m["ate"]["median"])
            trial_ares.append(m["are"]["median"])
            trial_srs.append(m["success_rate"])
            trial_runtimes.append(m.get("runtime_mean_ms", 0))
            trial_metrics.append(m)

            print(f"    Trial {trial+1}/{N_TRIALS}: "
                  f"ATE={m['ate']['median']*100:.2f}cm  ARE={m['are']['median']:.2f}deg  "
                  f"SR={m['success_rate']*100:.0f}%  ({elapsed:.1f}s)")

            del obs
            torch.cuda.empty_cache()
            gc.collect()

        sorted_ate_idx = sorted(range(N_TRIALS), key=lambda i: trial_ates[i])
        median_idx = sorted_ate_idx[1]

        final_ate = trial_ates[median_idx]
        final_are = trial_ares[median_idx]
        final_sr = trial_srs[median_idx]
        final_rt = trial_runtimes[median_idx]

        print(f"    >> MEDIAN: ATE={final_ate*100:.2f}cm  ARE={final_are:.2f}deg  "
              f"SR={final_sr*100:.0f}%")

        all_results.append({
            "scene": scene_name,
            "model": model_name,
            "ate_median": final_ate,
            "are_median": final_are,
            "success_rate": final_sr,
            "runtime_ms": final_rt,
        })

        if scene_name not in detailed_results:
            detailed_results[scene_name] = {}
        detailed_results[scene_name][model_name] = {
            "trials": [
                {
                    "ate_median": trial_ates[i],
                    "are_median": trial_ares[i],
                    "success_rate": trial_srs[i],
                    "runtime_ms": trial_runtimes[i],
                }
                for i in range(N_TRIALS)
            ],
            "median_ate": final_ate,
            "median_are": final_are,
            "median_sr": final_sr,
            "median_rt": final_rt,
        }

        del gmap, dataset
        torch.cuda.empty_cache()
        gc.collect()

    # ================================================================
    # Generate cross-scene figures and tables
    # ================================================================
    import matplotlib.pyplot as plt

    # Bar chart: only good scenes (all 4 models)
    good_scene_results = [r for r in all_results if r["scene"] in
                          [s["name"] for s in GOOD_SCENES]]
    plot_observation_model_comparison(
        good_scene_results,
        save_path=str(figures_dir / "model_comparison.png"),
    )
    plt.close("all")

    # LaTeX table — all results (good scenes first with all models, then weak scenes)
    ordered_results = []
    for s in GOOD_SCENES:
        for r in all_results:
            if r["scene"] == s["name"]:
                ordered_results.append(r)
    for s in WEAK_SCENES:
        for r in all_results:
            if r["scene"] == s["name"]:
                ordered_results.append(r)

    latex = generate_latex_table(
        ordered_results,
        caption="Localization results: 3 good scenes (all methods) + 2 weak scenes (best method only). "
                "ATE/ARE are medians across frames; each config run 3 times, median trial reported. "
                "200 particles, 100 frames, local initialization.",
    )
    with open(output_dir / "results_table.tex", "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {output_dir / 'results_table.tex'}")

    # ================================================================
    # Save all results as JSON (human-readable) and torch (for reuse)
    # ================================================================

    # JSON-serializable version
    json_results = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_trials": N_TRIALS,
            "n_frames": N_FRAMES,
            "n_particles": N_PARTICLES,
            "refiner_settings": "320x240, 100 iters, lr=0.01, blur_schedule",
            "init": "local (trans_spread=0.03, rot_spread=0.01)",
        },
        "summary_table": ordered_results,
        "detailed": {},
    }
    for scene_name, scene_data in detailed_results.items():
        json_results["detailed"][scene_name] = {}
        for model_name, model_data in scene_data.items():
            json_results["detailed"][scene_name][model_name] = {
                "trials": model_data["trials"],
                "median_ate": model_data["median_ate"],
                "median_are": model_data["median_are"],
                "median_sr": model_data["median_sr"],
                "median_rt": model_data["median_rt"],
            }

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to {output_dir / 'final_results.json'}")

    # Torch save (includes tensors for later analysis)
    torch.save({
        "all_results": all_results,
        "detailed_results": {
            scene: {
                model: {k: v for k, v in mdata.items()
                        if k not in ("estimated_poses", "gt_poses")}
                for model, mdata in sdata.items()
            }
            for scene, sdata in detailed_results.items()
        },
        "convergence_data": convergence_data,
    }, output_dir / "all_results.pt")
    print(f"Torch results saved to {output_dir / 'all_results.pt'}")

    # ================================================================
    # Print final summary table
    # ================================================================
    elapsed_total = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*78}")
    print(f"  FINAL RESULTS (median of {N_TRIALS} trials per config)")
    print(f"{'='*78}")
    print(f"{'Scene':<14} {'Model':<14} {'ATE Med':>10} {'ARE Med':>10} {'Succ%':>8} {'ms/step':>8}")
    print("-" * 66)

    prev_scene = None
    for r in ordered_results:
        scene_label = r["scene"] if r["scene"] != prev_scene else ""
        prev_scene = r["scene"]
        ate_str = f"{r['ate_median']*100:.1f}cm"
        are_str = f"{r['are_median']:.1f}deg"
        sr_str = f"{r['success_rate']*100:.0f}%"
        rt_str = f"{r['runtime_ms']:.0f}"
        print(f"{scene_label:<14} {r['model']:<14} {ate_str:>10} {are_str:>10} {sr_str:>8} {rt_str:>8}")

    print(f"\nTotal time: {elapsed_total/60:.1f} minutes")
    print(f"Figures: {figures_dir}/")
    print(f"LaTeX:   {output_dir / 'results_table.tex'}")
    print(f"JSON:    {output_dir / 'final_results.json'}")
    print(f"Torch:   {output_dir / 'all_results.pt'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
