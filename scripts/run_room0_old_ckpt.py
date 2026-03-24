"""Re-run room0 with OLD checkpoint to compare against depth-supervised."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pypose as pp
import gc
import time
import json

from semantic_pf_loc.gaussian_map import GaussianMap
from semantic_pf_loc.batch_renderer import BatchRenderer
from semantic_pf_loc.particle_filter import ParticleFilter
from semantic_pf_loc.motion_model import MotionModel
from semantic_pf_loc.gradient_refiner import GradientRefiner
from semantic_pf_loc.observation.ssim import SSIMObservation
from semantic_pf_loc.observation.clip_image import CLIPImageObservation
from semantic_pf_loc.observation.clip_text import CLIPTextObservation
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.evaluation.metrics import compute_all_metrics
from semantic_pf_loc.utils.pose_utils import scale_intrinsics


SCENE_CFG = {
    "name": "room0", "type": "replica",
    "path": "data/replica/room0",
    "native_w": 1200, "native_h": 680,
    "trans_std": 0.003, "rot_std": 0.002,
    "text_desc": "a living room with a sofa and table",
}

N_TRIALS = 3
N_FRAMES = 100
N_PARTICLES = 200


def run_trial(dataset, gmap, obs_model, scene_cfg, use_refine, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    K = dataset.get_intrinsics().float().to("cuda")
    native_size = (scene_cfg["native_w"], scene_cfg["native_h"])

    renderer = BatchRenderer(gmap, width=160, height=120)
    motion = MotionModel(
        translation_std=scene_cfg["trans_std"],
        rotation_std=scene_cfg["rot_std"],
        device="cuda",
    )

    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=N_PARTICLES,
        render_width=160, render_height=120,
        render_width_hires=320, render_height_hires=240,
        convergence_threshold=0.02,
        roughening_trans=0.002, roughening_rot=0.001,
        device="cuda",
    )

    hires_renderer, refiner, K_hires = None, None, None
    if use_refine:
        hires_renderer = BatchRenderer(gmap, width=320, height=240)
        K_hires = scale_intrinsics(K, native_size, (320, 240))
        refiner = GradientRefiner(
            hires_renderer, num_iterations=100, lr_init=0.01, blur_schedule=True
        )

    sample0 = dataset[0]
    gt_pose0 = pp.mat2SE3(
        sample0["pose"].double().to("cuda").unsqueeze(0), check=False
    ).squeeze(0).float()
    pf.initialize_around_pose(gt_pose0, trans_spread=0.03, rot_spread=0.01)

    est_poses, gt_poses, times = [], [], []
    for i in range(min(N_FRAMES, len(dataset))):
        sample = dataset[i]
        if obs_model.requires_image:
            obs = {"image": sample["image"].float().to("cuda")}
        else:
            obs = {"text": scene_cfg.get("text_desc", "")}

        est, info = pf.step(obs, K)

        if use_refine and refiner and info["converged"]:
            with torch.enable_grad():
                refined = refiner.refine(
                    pp.SE3(est.tensor().unsqueeze(0)),
                    sample["image"].float().to("cuda"),
                    K_hires,
                )
            est = refined.squeeze(0)

        est_poses.append(est.matrix().cpu())
        gt_poses.append(sample["pose"].float())
        times.append(info["step_time_ms"])

    m = compute_all_metrics(torch.stack(est_poses), torch.stack(gt_poses), torch.tensor(times))

    del renderer, pf
    if hires_renderer:
        del hires_renderer
    torch.cuda.empty_cache()
    gc.collect()
    return m


def main():
    torch.set_grad_enabled(False)

    print("=" * 70)
    print("  room0 with OLD checkpoint (checkpoints/room0.ckpt)")
    print("=" * 70)

    gmap = GaussianMap.from_checkpoint("checkpoints/room0.ckpt")
    dataset = ReplicaDataset("data/replica/room0", stride=1)
    print(f"Loaded: {gmap.num_gaussians} Gaussians, {len(dataset)} frames")

    models = [
        ("SSIM", lambda: SSIMObservation(temperature=3.0), False),
        ("SSIM+Refine", lambda: SSIMObservation(temperature=3.0), True),
        ("CLIP-Image", lambda: CLIPImageObservation(device="cuda", temperature=10.0), False),
        ("CLIP-Text", lambda: CLIPTextObservation(device="cuda", temperature=10.0), False),
    ]

    all_results = {}
    for model_name, create_obs, use_refine in models:
        print(f"\n--- {model_name} ---")
        trial_ates, trial_ares, trial_srs = [], [], []

        for trial in range(N_TRIALS):
            obs = create_obs()
            t0 = time.time()
            m = run_trial(dataset, gmap, obs, SCENE_CFG, use_refine, 42 + trial * 1000)
            elapsed = time.time() - t0

            ate = m["ate"]["median"]
            are = m["are"]["median"]
            sr = m["success_rate"]
            trial_ates.append(ate)
            trial_ares.append(are)
            trial_srs.append(sr)

            print(f"  Trial {trial+1}/{N_TRIALS}: "
                  f"ATE={ate*100:.2f}cm ARE={are:.2f}deg SR={sr*100:.0f}% ({elapsed:.1f}s)")

            del obs
            torch.cuda.empty_cache()
            gc.collect()

        idx = sorted(range(N_TRIALS), key=lambda i: trial_ates[i])[1]
        print(f"  >> MEDIAN: ATE={trial_ates[idx]*100:.2f}cm "
              f"ARE={trial_ares[idx]:.2f}deg SR={trial_srs[idx]*100:.0f}%")

        all_results[model_name] = {
            "trials": [
                {"ate": trial_ates[i], "are": trial_ares[i], "sr": trial_srs[i]}
                for i in range(N_TRIALS)
            ],
            "median_ate": trial_ates[idx],
            "median_are": trial_ares[idx],
            "median_sr": trial_srs[idx],
        }

    # Save
    with open("results/final_evaluation/room0_old_ckpt.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/final_evaluation/room0_old_ckpt.json")


if __name__ == "__main__":
    main()
