"""Generate side-by-side video: query image | 3DGS render from estimated pose.

For each frame:
  1. Get query image from dataset
  2. Run PF + refinement to get estimated pose
  3. Render 3DGS from estimated pose at display resolution
  4. Create side-by-side image with frame number and ATE overlay
  5. Save frames as PNGs
  6. Combine into MP4 and GIF via ffmpeg
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pypose as pp
import numpy as np
import time
import gc
from pathlib import Path

from semantic_pf_loc.gaussian_map import GaussianMap
from semantic_pf_loc.batch_renderer import BatchRenderer
from semantic_pf_loc.particle_filter import ParticleFilter
from semantic_pf_loc.motion_model import MotionModel
from semantic_pf_loc.gradient_refiner import GradientRefiner
from semantic_pf_loc.observation.ssim import SSIMObservation
from semantic_pf_loc.datasets.tum import TUMDataset
from semantic_pf_loc.evaluation.metrics import translation_error, rotation_error
from semantic_pf_loc.utils.pose_utils import scale_intrinsics, se3_to_viewmat


# Scene config for fr3_office
SCENE_CFG = {
    "name": "fr3_office",
    "type": "tum",
    "path": "data/tum/rgbd_dataset_freiburg3_long_office_household",
    "ckpt": "checkpoints_depth/fr3_office.ckpt",
    "native_w": 640,
    "native_h": 480,
    "trans_std": 0.005,
    "rot_std": 0.003,
}

DEVICE = "cuda"
N_PARTICLES = 200
N_FRAMES = 100
DISPLAY_W = 640
DISPLAY_H = 480
SEED = 42


def create_side_by_side(query_img, render_img, frame_idx, ate_error, are_error):
    """Create a side-by-side image with overlay text.

    Args:
        query_img: [H, W, 3] float tensor [0, 1]
        render_img: [H, W, 3] float tensor [0, 1]
        frame_idx: int
        ate_error: float (meters)
        are_error: float (degrees)

    Returns:
        numpy array [H, 2*W+gap, 3] uint8
    """
    import cv2

    h, w = DISPLAY_H, DISPLAY_W
    gap = 10  # pixel gap between panels

    # Resize images to display resolution
    def resize_tensor(img, target_h, target_w):
        img_perm = img.permute(2, 0, 1).unsqueeze(0)
        img_resized = torch.nn.functional.interpolate(
            img_perm, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return img_resized.squeeze(0).permute(1, 2, 0)

    query_disp = resize_tensor(query_img, h, w)
    render_disp = resize_tensor(render_img, h, w)

    # Convert to numpy uint8
    query_np = (query_disp.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    render_np = (render_disp.cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

    # Create canvas
    canvas = np.zeros((h, 2 * w + gap, 3), dtype=np.uint8)
    canvas[:, :w] = query_np
    canvas[:, w + gap:] = render_np

    # Convert RGB to BGR for OpenCV text rendering
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # Add text overlays
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color_white = (255, 255, 255)
    color_bg = (0, 0, 0)

    # Frame number (top-left)
    text_frame = f"Frame {frame_idx:03d}"
    (tw, th), _ = cv2.getTextSize(text_frame, font, font_scale, thickness)
    cv2.rectangle(canvas_bgr, (5, 5), (15 + tw, 15 + th), color_bg, -1)
    cv2.putText(canvas_bgr, text_frame, (10, 10 + th), font, font_scale, color_white, thickness)

    # ATE error (top-right of left panel)
    ate_cm = ate_error * 100
    if ate_cm < 5.0:
        color_ate = (0, 255, 0)  # green = good (BGR)
    elif ate_cm < 20.0:
        color_ate = (0, 255, 255)  # yellow (BGR)
    else:
        color_ate = (0, 0, 255)  # red = bad (BGR)

    text_ate = f"ATE: {ate_cm:.1f}cm"
    (tw, th), _ = cv2.getTextSize(text_ate, font, font_scale, thickness)
    x_pos = w - tw - 15
    cv2.rectangle(canvas_bgr, (x_pos - 5, 5), (x_pos + tw + 5, 15 + th), color_bg, -1)
    cv2.putText(canvas_bgr, text_ate, (x_pos, 10 + th), font, font_scale, color_ate, thickness)

    # ARE error (below ATE)
    text_are = f"ARE: {are_error:.1f}deg"
    (tw2, th2), _ = cv2.getTextSize(text_are, font, font_scale * 0.8, thickness)
    cv2.rectangle(canvas_bgr, (x_pos - 5, 20 + th), (x_pos + tw2 + 5, 30 + th + th2), color_bg, -1)
    cv2.putText(canvas_bgr, text_are, (x_pos, 25 + th + th2), font, font_scale * 0.8, color_white, thickness)

    # Labels
    text_query = "Query Image"
    (tw, th), _ = cv2.getTextSize(text_query, font, font_scale, thickness)
    cx = w // 2 - tw // 2
    cv2.rectangle(canvas_bgr, (cx - 5, h - th - 15), (cx + tw + 5, h - 5), color_bg, -1)
    cv2.putText(canvas_bgr, text_query, (cx, h - 10), font, font_scale, color_white, thickness)

    text_render = "3DGS Render (Est. Pose)"
    (tw, th), _ = cv2.getTextSize(text_render, font, font_scale, thickness)
    cx = w + gap + w // 2 - tw // 2
    cv2.rectangle(canvas_bgr, (cx - 5, h - th - 15), (cx + tw + 5, h - 5), color_bg, -1)
    cv2.putText(canvas_bgr, text_render, (cx, h - 10), font, font_scale, color_white, thickness)

    # Convert back to RGB
    canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    return canvas_rgb


def main():
    torch.set_grad_enabled(False)

    output_dir = Path("results/video")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  GENERATE RESULT VIDEO: fr3_office (SSIM + Refine)")
    print("=" * 70)

    # Set seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Load dataset and 3DGS map
    dataset = TUMDataset(SCENE_CFG["path"], stride=1)
    gmap = GaussianMap.from_checkpoint(SCENE_CFG["ckpt"])
    print(f"Loaded: {gmap.num_gaussians} Gaussians, {len(dataset)} frames")

    K = dataset.get_intrinsics().float().to(DEVICE)
    native_size = (SCENE_CFG["native_w"], SCENE_CFG["native_h"])

    # PF renderer (low-res)
    renderer = BatchRenderer(gmap, width=160, height=120)
    motion = MotionModel(
        translation_std=SCENE_CFG["trans_std"],
        rotation_std=SCENE_CFG["rot_std"],
        device=DEVICE,
    )
    obs_model = SSIMObservation(temperature=3.0)

    pf = ParticleFilter(
        gmap, renderer, obs_model, motion,
        num_particles=N_PARTICLES,
        render_width=160, render_height=120,
        render_width_hires=320, render_height_hires=240,
        convergence_threshold=0.02,
        roughening_trans=0.002, roughening_rot=0.001,
        gradient_refiner=None,
        device=DEVICE,
    )

    # Display renderer (higher res for nice renders in video)
    display_renderer = BatchRenderer(gmap, width=DISPLAY_W, height=DISPLAY_H)
    K_display = scale_intrinsics(K, native_size, (DISPLAY_W, DISPLAY_H))

    # Refiner (post-hoc)
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

    n_frames = min(N_FRAMES, len(dataset))
    est_poses = []
    gt_poses = []

    print(f"\nProcessing {n_frames} frames...")
    t0 = time.time()

    for i in range(n_frames):
        sample = dataset[i]
        query_image = sample["image"].float().to(DEVICE)  # [H, W, 3]
        gt_pose_mat = sample["pose"].float()

        obs = {"image": query_image}
        est, info = pf.step(obs, K)

        # Post-hoc refinement
        if info["converged"]:
            est_for_refine = pp.SE3(est.tensor().unsqueeze(0))
            with torch.enable_grad():
                refined = refiner.refine(
                    est_for_refine, query_image, K_hires
                )
            est = refined.squeeze(0)

        est_mat = est.matrix().cpu()
        est_poses.append(est_mat)
        gt_poses.append(gt_pose_mat)

        # Compute per-frame errors
        t_err = translation_error(est_mat.unsqueeze(0), gt_pose_mat.unsqueeze(0))[0].item()
        r_err = rotation_error(est_mat.unsqueeze(0), gt_pose_mat.unsqueeze(0))[0].item()

        # Render from estimated pose for display
        with torch.no_grad():
            viewmat = se3_to_viewmat(est)  # [4, 4]
            render_img = display_renderer.render_single(
                viewmat, K_display.to(DEVICE)
            )  # [H, W, 3]

        # Create side-by-side frame
        frame_img = create_side_by_side(query_image, render_img, i, t_err, r_err)

        # Save frame
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(frame_img)
        pil_img.save(frames_dir / f"frame_{i:03d}.png")

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  Frame {i:3d}/{n_frames}: ATE={t_err*100:.1f}cm  ARE={r_err:.1f}deg  "
                  f"conv={info['converged']}  [{elapsed:.1f}s elapsed]")

    total_time = time.time() - t0
    print(f"\nAll {n_frames} frames processed in {total_time:.1f}s")

    # Compute overall ATE
    est_stack = torch.stack(est_poses)
    gt_stack = torch.stack(gt_poses)
    t_errors = translation_error(est_stack, gt_stack)
    r_errors = rotation_error(est_stack, gt_stack)
    print(f"Overall ATE median: {t_errors.median().item()*100:.2f}cm")
    print(f"Overall ARE median: {r_errors.median().item():.2f}deg")

    # Generate MP4 with ffmpeg
    print("\nGenerating MP4...")
    mp4_path = output_dir / "fr3_office_result.mp4"
    ffmpeg_cmd = (
        f"ffmpeg -y -framerate 10 "
        f"-i {frames_dir}/frame_%03d.png "
        f'-vf "scale=1280:-2" '
        f"-c:v libx264 -pix_fmt yuv420p "
        f"{mp4_path}"
    )
    ret = os.system(ffmpeg_cmd)
    if ret == 0:
        print(f"  MP4 saved: {mp4_path}")
    else:
        print(f"  WARNING: ffmpeg MP4 failed (exit {ret})")

    # Generate GIF
    print("Generating GIF...")
    gif_path = output_dir / "fr3_office_result.gif"
    # Use palette-based approach for better quality GIF
    palette_cmd = (
        f"ffmpeg -y -framerate 10 "
        f"-i {frames_dir}/frame_%03d.png "
        f'-vf "scale=640:-2,fps=10,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" '
        f"{gif_path}"
    )
    ret = os.system(palette_cmd)
    if ret == 0:
        print(f"  GIF saved: {gif_path}")
    else:
        print(f"  WARNING: ffmpeg GIF failed (exit {ret})")

    # Cleanup
    del renderer, pf, display_renderer, hires_renderer, refiner, gmap, dataset
    torch.cuda.empty_cache()
    gc.collect()

    print("\nDone!")


if __name__ == "__main__":
    main()
