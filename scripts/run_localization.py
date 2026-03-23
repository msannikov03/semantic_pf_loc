"""Run particle filter localization on a scene."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import torch
import pypose as pp
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

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
from semantic_pf_loc.evaluation.metrics import compute_all_metrics


def build_observation_model(cfg, device):
    if cfg.observation.type == "ssim":
        return SSIMObservation()
    elif cfg.observation.type == "clip_image":
        return CLIPImageObservation(
            model_name=cfg.observation.clip_model,
            pretrained=cfg.observation.clip_pretrained,
            device=device,
            temperature=cfg.observation.temperature,
        )
    elif cfg.observation.type == "clip_text":
        return CLIPTextObservation(
            model_name=cfg.observation.clip_model,
            pretrained=cfg.observation.clip_pretrained,
            device=device,
            temperature=cfg.observation.temperature,
        )
    else:
        raise ValueError(f"Unknown observation type: {cfg.observation.type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_config", help="Scene config YAML")
    parser.add_argument("--localize_config", default=None, help="Localization config YAML")
    parser.add_argument("--checkpoint", required=True, help="3DGS checkpoint path")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--eval_stride", type=int, default=5, help="Eval frame stride")
    args = parser.parse_args()

    # Merge configs
    cfg = OmegaConf.load("configs/default.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(args.scene_config))
    if args.localize_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.localize_config))

    device = "cuda"

    # Load dataset
    if cfg.scene.type == "tum":
        dataset = TUMDataset(cfg.scene.data_dir, stride=args.eval_stride)
    else:
        dataset = ReplicaDataset(cfg.scene.data_dir, stride=args.eval_stride)

    # Load 3DGS map
    gmap = GaussianMap.from_checkpoint(args.checkpoint, device=device)

    # Build components
    renderer = BatchRenderer(
        gmap,
        width=cfg.particle_filter.render_width,
        height=cfg.particle_filter.render_height,
        near_plane=cfg.train_gs.near_plane,
        far_plane=cfg.train_gs.far_plane,
    )
    obs_model = build_observation_model(cfg, device)
    motion = MotionModel(
        translation_std=cfg.motion_model.translation_std,
        rotation_std=cfg.motion_model.rotation_std,
        device=device,
    )

    refiner = None
    if cfg.gradient_refine.enabled:
        refiner = GradientRefiner(
            renderer=renderer,
            num_iterations=cfg.gradient_refine.num_iterations,
            lr_init=cfg.gradient_refine.lr_init,
            lr_final=cfg.gradient_refine.lr_final,
            loss_type=cfg.gradient_refine.loss_type,
            ssim_weight=cfg.gradient_refine.ssim_weight,
            blur_schedule=cfg.gradient_refine.blur_schedule,
        )

    pf = ParticleFilter(
        gaussian_map=gmap,
        renderer=renderer,
        observation_model=obs_model,
        motion_model=motion,
        num_particles=cfg.particle_filter.num_particles,
        resample_threshold=cfg.particle_filter.resample_threshold,
        gradient_refiner=refiner,
        top_k_refine=cfg.gradient_refine.top_k,
        render_width=cfg.particle_filter.render_width,
        render_height=cfg.particle_filter.render_height,
        render_width_hires=cfg.particle_filter.render_width_hires,
        render_height_hires=cfg.particle_filter.render_height_hires,
        convergence_threshold=cfg.particle_filter.convergence_threshold,
        device=device,
    )

    # Initialize
    bounds_min, bounds_max = dataset.get_bounds()
    pf.initialize_global(bounds_min, bounds_max)

    K = dataset.get_intrinsics().float().to(device)

    # Run localization
    estimated_poses = []
    gt_poses = []
    step_times = []

    pbar = tqdm(range(len(dataset)), desc=f"Localizing {cfg.scene.name} ({obs_model.name})")
    for i in pbar:
        sample = dataset[i]
        gt_pose = sample["pose"].float()  # [4, 4]

        if obs_model.requires_image:
            observation = {"image": sample["image"].float().to(device)}
        else:
            observation = {"text": cfg.observation.text_query}

        estimated, info = pf.step(observation, K)

        est_matrix = estimated.matrix().cpu()
        estimated_poses.append(est_matrix)
        gt_poses.append(gt_pose)
        step_times.append(info["step_time_ms"])

        if i % 10 == 0:
            from semantic_pf_loc.utils.pose_utils import pose_error
            t_err, r_err = pose_error(est_matrix, gt_pose)
            pbar.set_postfix(
                t_err=f"{t_err:.3f}m",
                r_err=f"{r_err:.1f}°",
                n_eff=f"{info['n_eff']:.0f}",
                conv=info["converged"],
            )

    # Compute metrics
    estimated_stack = torch.stack(estimated_poses)  # [T, 4, 4]
    gt_stack = torch.stack(gt_poses)  # [T, 4, 4]
    times = torch.tensor(step_times)

    metrics = compute_all_metrics(
        estimated_stack,
        gt_stack,
        per_step_times_ms=times,
        trans_threshold=cfg.evaluation.success_trans_threshold,
        rot_threshold=cfg.evaluation.success_rot_threshold,
        convergence_threshold=cfg.evaluation.convergence_threshold,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Scene: {cfg.scene.name} | Observation: {obs_model.name}")
    print(f"{'='*60}")
    print(f"ATE RMSE:         {metrics['ate']['rmse']:.4f} m")
    print(f"ATE Median:       {metrics['ate']['median']:.4f} m")
    print(f"ARE Mean:         {metrics['are']['mean']:.2f}°")
    print(f"Success Rate:     {metrics['success_rate']*100:.1f}%")
    print(f"Convergence:      frame {metrics['convergence_frame']}")
    print(f"Runtime:          {metrics['runtime_mean_ms']:.1f} ± {metrics['runtime_std_ms']:.1f} ms")

    # Save results
    output_dir = Path(args.output) / cfg.scene.name / obs_model.name
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "estimated_poses": estimated_stack,
            "gt_poses": gt_stack,
            "metrics": {k: v for k, v in metrics.items()
                       if not isinstance(v, torch.Tensor)},
            "step_times_ms": times,
            "config": OmegaConf.to_container(cfg),
        },
        output_dir / "results.pt",
    )
    print(f"Results saved to {output_dir / 'results.pt'}")


if __name__ == "__main__":
    main()
