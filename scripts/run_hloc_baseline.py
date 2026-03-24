"""HLoc-style baseline: SIFT feature matching + depth-backed PnP.

Classical visual localization pipeline:
1. Build reference database: every Nth frame, extract SIFT features,
   back-project to 3D using GT depth + intrinsics + known pose.
2. For each query frame: extract SIFT, match against all reference frames
   (FLANN + Lowe ratio test), back-project matched reference keypoints to 3D,
   solve PnP-RANSAC to recover absolute camera pose.

This is the standard "structure-based" localization baseline used in
the HLoc / Visual Localization Benchmark literature.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from semantic_pf_loc.datasets.tum import TUMDataset
from semantic_pf_loc.datasets.replica import ReplicaDataset
from semantic_pf_loc.evaluation.metrics import translation_error, rotation_error


# ===== Configuration =====

SCENES = [
    {
        "name": "office0",
        "type": "replica",
        "path": "data/replica/office0",
    },
    {
        "name": "room0",
        "type": "replica",
        "path": "data/replica/room0",
    },
    {
        "name": "fr3_office",
        "type": "tum",
        "path": "data/tum/rgbd_dataset_freiburg3_long_office_household",
    },
]

REF_STRIDE = 5          # every 5th frame as reference
N_QUERY_FRAMES = 100    # number of query frames to evaluate
EXCLUSION_RADIUS = 2    # skip reference frames within this many indices of query
SIFT_NFEATURES = 3000   # number of SIFT features per image
LOWE_RATIO = 0.75       # Lowe's ratio test threshold
PNP_REPROJ_ERR = 5.0    # PnP RANSAC reprojection threshold (pixels)
PNP_ITERATIONS = 10000  # PnP RANSAC iterations
MIN_INLIERS = 12        # minimum inliers for a valid PnP solution
MAX_DEPTH = 8.0         # ignore depth beyond this (meters)
MIN_DEPTH = 0.1         # ignore depth below this (meters)
TOP_K_REFS = 5          # match against top-K reference images (by descriptor count)
SUCCESS_TRANS = 0.05    # 5 cm
SUCCESS_ROT = 2.0       # 2 degrees

# Our PF+Refine reference results (from full evaluation, 100 frames)
PF_REFINE_RESULTS = {
    "office0":    {"ate_median_cm": 1.4, "are_median_deg": 0.4, "success_rate": 0.74},
    "room0":      {"ate_median_cm": 1.9, "are_median_deg": 0.4, "success_rate": 0.99},
    "fr3_office": {"ate_median_cm": 3.1, "are_median_deg": 0.8, "success_rate": 0.96},
}

# GSLoc reference (from gsloc_baseline, 3cm noise level)
GSLOC_RESULTS = {
    "office0":    {"ate_median_cm": 1.07, "are_median_deg": 0.22, "success_rate": 1.00},
    "room0":      {"ate_median_cm": 1.22, "are_median_deg": 0.25, "success_rate": 0.99},
    "fr3_office": {"ate_median_cm": 1.27, "are_median_deg": 0.77, "success_rate": 0.95},
}


class HLocBaseline:
    """SIFT + Depth-backed PnP localization baseline."""

    def __init__(self, dataset, ref_stride=REF_STRIDE):
        """Build reference database from dataset.

        For each reference frame:
        - Extract SIFT keypoints + descriptors
        - Store depth map, pose, intrinsics for 3D back-projection
        """
        self.sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
        # FLANN with KD-tree for SIFT (L2 norm)
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=100)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        K_tensor = dataset.get_intrinsics()  # [3,3] float64
        self.K = K_tensor.numpy().astype(np.float64)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        self.ref_indices = []       # dataset index of each reference
        self.ref_poses = []         # [4,4] c2w numpy arrays
        self.ref_keypoints = []     # list of cv2.KeyPoint lists
        self.ref_descriptors = []   # list of numpy descriptor arrays
        self.ref_depths = []        # list of [H,W] numpy depth maps

        print(f"  Building reference database (stride={ref_stride})...")
        n_total = len(dataset)
        ref_candidates = list(range(0, n_total, ref_stride))
        for i in tqdm(ref_candidates, desc="  Extracting features", leave=False):
            sample = dataset[i]
            img_tensor = sample["image"]  # [H, W, 3] float in [0,1]
            depth_tensor = sample["depth"]
            pose_tensor = sample["pose"]

            if depth_tensor is None:
                continue

            img_uint8 = (img_tensor.numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            kp, desc = self.sift.detectAndCompute(gray, None)

            if desc is None or len(kp) < 20:
                continue

            self.ref_indices.append(i)
            self.ref_poses.append(pose_tensor.numpy())  # [4,4]
            self.ref_keypoints.append(kp)
            self.ref_descriptors.append(desc.astype(np.float32))
            self.ref_depths.append(depth_tensor.numpy())  # [H,W]

        print(f"  Reference database: {len(self.ref_indices)} frames, "
              f"{sum(len(kp) for kp in self.ref_keypoints)} total keypoints")

    def _backproject_keypoints(self, keypoints, depth_map, c2w):
        """Back-project 2D keypoints to 3D world coordinates using depth.

        Returns:
            pts_3d: [N, 3] world-frame 3D points (only valid depth)
            valid_mask: [M] boolean mask
        """
        pts_3d = []
        valid = []
        for kp in keypoints:
            u, v = kp.pt
            ui, vi = int(round(u)), int(round(v))
            # Bounds check
            if vi < 0 or vi >= depth_map.shape[0] or ui < 0 or ui >= depth_map.shape[1]:
                valid.append(False)
                pts_3d.append([0, 0, 0])
                continue
            d = depth_map[vi, ui]
            if d < MIN_DEPTH or d > MAX_DEPTH:
                valid.append(False)
                pts_3d.append([0, 0, 0])
                continue
            # Back-project to camera frame
            x_cam = (u - self.cx) * d / self.fx
            y_cam = (v - self.cy) * d / self.fy
            z_cam = d
            # Transform to world frame
            pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
            pt_world = c2w @ pt_cam
            pts_3d.append(pt_world[:3])
            valid.append(True)

        return np.array(pts_3d), np.array(valid)

    def localize(self, query_image_tensor, query_idx=None):
        """Localize a query image against the reference database.

        Args:
            query_image_tensor: [H, W, 3] float32 in [0, 1]
            query_idx: dataset index (to exclude nearby refs)

        Returns:
            pose_c2w: [4, 4] estimated camera-to-world, or None if failed
            info: dict with match counts, inliers, etc.
        """
        img_uint8 = (query_image_tensor.numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        kp_q, desc_q = self.sift.detectAndCompute(gray, None)

        info = {"n_query_kp": len(kp_q) if kp_q else 0}

        if desc_q is None or len(kp_q) < 10:
            info["failure_reason"] = "too_few_query_features"
            return None, info

        desc_q = desc_q.astype(np.float32)

        # Try matching against all reference frames, collect 3D-2D correspondences
        all_pts_3d = []
        all_pts_2d = []
        best_n_matches = 0
        best_ref_idx = -1

        for ref_i in range(len(self.ref_indices)):
            # Skip references too close to query (temporal proximity)
            if query_idx is not None:
                if abs(self.ref_indices[ref_i] - query_idx) <= EXCLUSION_RADIUS:
                    continue

            ref_desc = self.ref_descriptors[ref_i]
            if len(ref_desc) < 2:
                continue

            # FLANN knn match
            try:
                matches = self.flann.knnMatch(desc_q, ref_desc, k=2)
            except cv2.error:
                continue

            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < LOWE_RATIO * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 8:
                continue

            if len(good_matches) > best_n_matches:
                best_n_matches = len(good_matches)
                best_ref_idx = ref_i

            # Back-project reference keypoints to 3D
            ref_kps = self.ref_keypoints[ref_i]
            ref_depth = self.ref_depths[ref_i]
            ref_c2w = self.ref_poses[ref_i]
            pts_3d_all, valid_mask = self._backproject_keypoints(
                ref_kps, ref_depth, ref_c2w
            )

            for m in good_matches:
                ref_kp_idx = m.trainIdx
                if valid_mask[ref_kp_idx]:
                    all_pts_3d.append(pts_3d_all[ref_kp_idx])
                    all_pts_2d.append(kp_q[m.queryIdx].pt)

        info["n_3d2d_correspondences"] = len(all_pts_3d)
        info["best_n_matches"] = best_n_matches
        info["best_ref_idx"] = best_ref_idx

        if len(all_pts_3d) < MIN_INLIERS:
            info["failure_reason"] = "too_few_3d2d_correspondences"
            return None, info

        pts_3d = np.array(all_pts_3d, dtype=np.float64)
        pts_2d = np.array(all_pts_2d, dtype=np.float64)

        # Solve PnP with RANSAC
        dist_coeffs = np.zeros(4)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.K, dist_coeffs,
            iterationsCount=PNP_ITERATIONS,
            reprojectionError=PNP_REPROJ_ERR,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success or inliers is None:
            info["failure_reason"] = "pnp_failed"
            return None, info

        n_inliers = len(inliers)
        info["n_inliers"] = n_inliers

        if n_inliers < MIN_INLIERS:
            info["failure_reason"] = "too_few_inliers"
            return None, info

        # Refine with inliers only using iterative PnP
        pts_3d_inliers = pts_3d[inliers.ravel()]
        pts_2d_inliers = pts_2d[inliers.ravel()]
        success_ref, rvec_ref, tvec_ref = cv2.solvePnP(
            pts_3d_inliers, pts_2d_inliers, self.K, dist_coeffs,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success_ref:
            rvec, tvec = rvec_ref, tvec_ref

        # Convert to 4x4 camera-to-world
        R, _ = cv2.Rodrigues(rvec)
        # PnP gives world-to-camera: p_cam = R @ p_world + t
        # So c2w = inv([R t; 0 1])
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = tvec.ravel()
        c2w = np.linalg.inv(w2c)

        return c2w, info


def run_hloc_scene(scene_cfg):
    """Run HLoc baseline on one scene. Returns metrics dict."""
    name = scene_cfg["name"]

    # Load dataset
    if scene_cfg["type"] == "tum":
        dataset = TUMDataset(scene_cfg["path"], stride=1)
    else:
        dataset = ReplicaDataset(scene_cfg["path"], stride=1)

    # Build reference database
    t0 = time.time()
    hloc = HLocBaseline(dataset, ref_stride=REF_STRIDE)
    build_time = time.time() - t0
    print(f"  Database built in {build_time:.1f}s")

    # Choose query frames: first N_QUERY_FRAMES (same as PF evaluation)
    n_query = min(N_QUERY_FRAMES, len(dataset))
    est_poses = []
    gt_poses = []
    runtimes = []
    n_success = 0
    n_fail = 0
    all_infos = []

    for i in tqdm(range(n_query), desc=f"  Localizing {name}"):
        sample = dataset[i]
        gt_pose = sample["pose"].float()  # [4,4]

        t0 = time.time()
        est_c2w, info = hloc.localize(sample["image"], query_idx=i)
        runtime_ms = (time.time() - t0) * 1000
        runtimes.append(runtime_ms)
        all_infos.append(info)

        if est_c2w is not None:
            est_poses.append(torch.from_numpy(est_c2w).float())
            gt_poses.append(gt_pose)
            n_success += 1
        else:
            # Failed — use identity as placeholder (will be counted separately)
            est_poses.append(gt_pose.clone())  # won't affect error if we track failures
            gt_poses.append(gt_pose)
            n_fail += 1

    # Compute errors
    est_stack = torch.stack(est_poses)  # [N, 4, 4]
    gt_stack = torch.stack(gt_poses)    # [N, 4, 4]

    # For failed frames, set large errors (they don't have real estimates)
    trans_errors = translation_error(est_stack, gt_stack)
    rot_errors = rotation_error(est_stack, gt_stack)

    # Mark failed frames with large errors
    fail_mask = []
    for info in all_infos:
        fail_mask.append("failure_reason" in info)
    fail_mask = torch.tensor(fail_mask, dtype=torch.bool)

    # Set failed frames to large errors (they truly failed)
    trans_errors_corrected = trans_errors.clone()
    rot_errors_corrected = rot_errors.clone()
    trans_errors_corrected[fail_mask] = 10.0   # 10 meters = obvious failure
    rot_errors_corrected[fail_mask] = 180.0    # 180 degrees = obvious failure

    # Metrics over ALL frames (including failures as large errors)
    ate_median = trans_errors_corrected.median().item()
    ate_mean = trans_errors_corrected.mean().item()
    are_median = rot_errors_corrected.median().item()
    are_mean = rot_errors_corrected.mean().item()
    sr = ((trans_errors_corrected < SUCCESS_TRANS) &
          (rot_errors_corrected < SUCCESS_ROT)).float().mean().item()

    # Metrics over SUCCESSFUL frames only
    success_mask = ~fail_mask
    if success_mask.any():
        ate_median_succ = trans_errors[success_mask].median().item()
        are_median_succ = rot_errors[success_mask].median().item()
        sr_succ = ((trans_errors[success_mask] < SUCCESS_TRANS) &
                   (rot_errors[success_mask] < SUCCESS_ROT)).float().mean().item()
    else:
        ate_median_succ = float("inf")
        are_median_succ = float("inf")
        sr_succ = 0.0

    # Typical match info
    avg_correspondences = np.mean([info.get("n_3d2d_correspondences", 0) for info in all_infos])
    avg_inliers = np.mean([info.get("n_inliers", 0) for info in all_infos if "n_inliers" in info])

    metrics = {
        "ate_median": ate_median,
        "ate_mean": ate_mean,
        "are_median": are_median,
        "are_mean": are_mean,
        "success_rate": sr,
        "localization_rate": n_success / n_query,
        "n_success": n_success,
        "n_fail": n_fail,
        "n_query": n_query,
        "ate_median_successful": ate_median_succ,
        "are_median_successful": are_median_succ,
        "success_rate_successful": sr_succ,
        "runtime_mean_ms": np.mean(runtimes),
        "build_time_s": build_time,
        "avg_correspondences": avg_correspondences,
        "avg_inliers": avg_inliers,
        "trans_errors": trans_errors_corrected.tolist(),
        "rot_errors": rot_errors_corrected.tolist(),
    }

    return metrics


def plot_comparison(all_results, output_dir):
    """Bar chart comparing HLoc vs PF+Refine vs GSLoc across scenes."""
    scenes = [s["name"] for s in SCENES if s["name"] in all_results]
    n_scenes = len(scenes)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- ATE ---
    ax = axes[0]
    x = np.arange(n_scenes)
    w = 0.25
    hloc_ate = [all_results[s]["ate_median"] * 100 for s in scenes]
    pf_ate = [PF_REFINE_RESULTS[s]["ate_median_cm"] for s in scenes]
    gsloc_ate = [GSLOC_RESULTS[s]["ate_median_cm"] for s in scenes]

    ax.bar(x - w, hloc_ate, w, label="HLoc (SIFT+PnP)", color="#e74c3c", alpha=0.85)
    ax.bar(x, pf_ate, w, label="PF+Refine (Ours)", color="#2196F3", alpha=0.85)
    ax.bar(x + w, gsloc_ate, w, label="GSLoc (Grad-only)", color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=11)
    ax.set_ylabel("ATE Median (cm)", fontsize=12)
    ax.set_title("Translation Error", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=5.0, color="gray", linestyle=":", alpha=0.5, label="5cm threshold")

    # --- ARE ---
    ax = axes[1]
    hloc_are = [all_results[s]["are_median"] for s in scenes]
    pf_are = [PF_REFINE_RESULTS[s]["are_median_deg"] for s in scenes]
    gsloc_are = [GSLOC_RESULTS[s]["are_median_deg"] for s in scenes]

    ax.bar(x - w, hloc_are, w, label="HLoc (SIFT+PnP)", color="#e74c3c", alpha=0.85)
    ax.bar(x, pf_are, w, label="PF+Refine (Ours)", color="#2196F3", alpha=0.85)
    ax.bar(x + w, gsloc_are, w, label="GSLoc (Grad-only)", color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=11)
    ax.set_ylabel("ARE Median (deg)", fontsize=12)
    ax.set_title("Rotation Error", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Success Rate ---
    ax = axes[2]
    hloc_sr = [all_results[s]["success_rate"] * 100 for s in scenes]
    pf_sr = [PF_REFINE_RESULTS[s]["success_rate"] * 100 for s in scenes]
    gsloc_sr = [GSLOC_RESULTS[s]["success_rate"] * 100 for s in scenes]

    ax.bar(x - w, hloc_sr, w, label="HLoc (SIFT+PnP)", color="#e74c3c", alpha=0.85)
    ax.bar(x, pf_sr, w, label="PF+Refine (Ours)", color="#2196F3", alpha=0.85)
    ax.bar(x + w, gsloc_sr, w, label="GSLoc (Grad-only)", color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Success Rate (5cm / 2deg)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    fig.suptitle("HLoc Baseline vs 3DGS-based Methods", fontsize=14, y=1.02)
    fig.tight_layout()
    save_path = output_dir / "hloc_comparison.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_per_frame_errors(all_results, output_dir):
    """Per-frame error plots for each scene."""
    for scene_cfg in SCENES:
        name = scene_cfg["name"]
        if name not in all_results:
            continue
        m = all_results[name]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        frames = np.arange(len(m["trans_errors"]))
        te = np.array(m["trans_errors"]) * 100  # cm

        # Cap display at 50cm for readability
        te_display = np.minimum(te, 50)
        ax1.plot(frames, te_display, ".-", color="#e74c3c", markersize=3, linewidth=0.8)
        ax1.axhline(y=5.0, color="green", linestyle="--", alpha=0.6, label="5cm threshold")
        ax1.set_ylabel("Translation Error (cm)")
        ax1.set_title(f"HLoc Per-Frame Errors: {name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(te_display) * 1.1)

        re = np.array(m["rot_errors"])
        re_display = np.minimum(re, 30)
        ax2.plot(frames, re_display, ".-", color="#9b59b6", markersize=3, linewidth=0.8)
        ax2.axhline(y=2.0, color="green", linestyle="--", alpha=0.6, label="2deg threshold")
        ax2.set_ylabel("Rotation Error (deg)")
        ax2.set_xlabel("Frame")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(re_display) * 1.1)

        fig.tight_layout()
        save_path = output_dir / f"hloc_errors_{name}.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")


def main():
    output_dir = Path("results/hloc_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for scene_cfg in SCENES:
        name = scene_cfg["name"]
        print(f"\n{'='*60}")
        print(f"  Scene: {name}")
        print(f"{'='*60}")

        metrics = run_hloc_scene(scene_cfg)
        all_results[name] = metrics

        print(f"\n  Results:")
        print(f"    Localization rate: {metrics['n_success']}/{metrics['n_query']} "
              f"({metrics['localization_rate']*100:.0f}%)")
        print(f"    ATE median (all):  {metrics['ate_median']*100:.1f} cm")
        print(f"    ARE median (all):  {metrics['are_median']:.1f} deg")
        print(f"    Success rate:      {metrics['success_rate']*100:.0f}%")
        print(f"    ATE median (success only): {metrics['ate_median_successful']*100:.1f} cm")
        print(f"    ARE median (success only): {metrics['are_median_successful']:.1f} deg")
        print(f"    Avg correspondences: {metrics['avg_correspondences']:.0f}")
        print(f"    Avg inliers:         {metrics['avg_inliers']:.0f}")
        print(f"    Runtime: {metrics['runtime_mean_ms']:.0f} ms/frame")

    # ===== Summary Table =====
    print(f"\n\n{'='*90}")
    print(f"  HLOC BASELINE vs PF+REFINE vs GSLOC")
    print(f"{'='*90}")
    print(f"{'Scene':<14} {'Method':<18} {'ATE Med':>10} {'ARE Med':>10} {'Succ%':>8} {'ms':>8}")
    print("-" * 72)
    for scene_cfg in SCENES:
        name = scene_cfg["name"]
        if name not in all_results:
            continue
        m = all_results[name]
        print(f"{name:<14} {'HLoc (SIFT+PnP)':<18} "
              f"{m['ate_median']*100:>8.1f}cm {m['are_median']:>9.1f}deg "
              f"{m['success_rate']*100:>7.0f}% {m['runtime_mean_ms']:>7.0f}")
        if name in PF_REFINE_RESULTS:
            pf = PF_REFINE_RESULTS[name]
            print(f"{'':<14} {'PF+Refine (Ours)':<18} "
                  f"{pf['ate_median_cm']:>8.1f}cm {pf['are_median_deg']:>9.1f}deg "
                  f"{pf['success_rate']*100:>7.0f}%")
        if name in GSLOC_RESULTS:
            gs = GSLOC_RESULTS[name]
            print(f"{'':<14} {'GSLoc (Grad-only)':<18} "
                  f"{gs['ate_median_cm']:>8.1f}cm {gs['are_median_deg']:>9.1f}deg "
                  f"{gs['success_rate']*100:>7.0f}%")
        print()

    # ===== Generate Plots =====
    print(f"\n--- Generating plots ---")
    plot_comparison(all_results, output_dir)
    plot_per_frame_errors(all_results, output_dir)

    # ===== Save Results =====
    # Make JSON-serializable version
    serializable = {}
    for scene_name, metrics in all_results.items():
        serializable[scene_name] = {
            k: v for k, v in metrics.items()
        }
    with open(output_dir / "results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved: {output_dir / 'results.json'}")

    print(f"\nDone! All results in {output_dir}/")


if __name__ == "__main__":
    main()
