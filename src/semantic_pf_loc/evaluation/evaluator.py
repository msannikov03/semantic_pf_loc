"""Full pipeline evaluator."""

import torch
import pypose as pp
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from .metrics import compute_all_metrics
from ..datasets.base import BaseLocalizationDataset
from ..particle_filter import ParticleFilter
from ..utils.pose_utils import pose_error


class Evaluator:
    """Runs particle filter localization and computes metrics."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def evaluate_sequence(
        self,
        dataset: BaseLocalizationDataset,
        particle_filter: ParticleFilter,
        observation_config: dict | None = None,
    ) -> dict:
        """Run PF over all frames, return metrics.

        Args:
            dataset: localization dataset
            particle_filter: initialized PF
            observation_config: {"type": "ssim"/"clip_image"/"clip_text", "text_query": ...}
        Returns:
            dict with metrics
        """
        bounds_min, bounds_max = dataset.get_bounds()
        particle_filter.initialize_global(bounds_min, bounds_max)

        K = dataset.get_intrinsics().float().to(self.device)

        estimated_poses = []
        gt_poses = []
        step_times = []

        for i in tqdm(range(len(dataset)), desc="Evaluating", leave=False):
            sample = dataset[i]
            gt_pose = sample["pose"].float()

            if particle_filter.obs_model.requires_image:
                obs = {"image": sample["image"].float().to(self.device)}
            else:
                text = observation_config.get("text_query", "") if observation_config else ""
                obs = {"text": text}

            estimated, info = particle_filter.step(obs, K)

            estimated_poses.append(estimated.matrix().cpu())
            gt_poses.append(gt_pose)
            step_times.append(info["step_time_ms"])

        estimated_stack = torch.stack(estimated_poses)
        gt_stack = torch.stack(gt_poses)
        times = torch.tensor(step_times)

        return compute_all_metrics(estimated_stack, gt_stack, times)

    def evaluate_multiple_runs(
        self,
        dataset: BaseLocalizationDataset,
        particle_filter: ParticleFilter,
        num_runs: int = 3,
        observation_config: dict | None = None,
    ) -> dict:
        """Average metrics over multiple runs."""
        all_metrics = []
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            metrics = self.evaluate_sequence(dataset, particle_filter, observation_config)
            all_metrics.append(metrics)

        # Average scalar metrics
        avg = {}
        for key in ["ate", "are"]:
            avg[key] = {}
            for subkey in all_metrics[0][key]:
                vals = [m[key][subkey] for m in all_metrics]
                avg[key][subkey] = sum(vals) / len(vals)

        avg["success_rate"] = sum(m["success_rate"] for m in all_metrics) / num_runs
        avg["convergence_frame"] = sum(m["convergence_frame"] for m in all_metrics) / num_runs

        if "runtime_mean_ms" in all_metrics[0]:
            avg["runtime_mean_ms"] = sum(m["runtime_mean_ms"] for m in all_metrics) / num_runs

        return avg
