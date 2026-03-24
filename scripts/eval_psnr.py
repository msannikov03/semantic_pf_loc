"""Evaluate PSNR for all checkpoints in a directory."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf
from gsplat import rasterization
from semantic_pf_loc.datasets.tum import TUMDataset
from semantic_pf_loc.datasets.replica import ReplicaDataset

CONFIGS = {
    "office0": "configs/train_gs/replica_office0.yaml",
    "room0": "configs/train_gs/replica_room0.yaml",
    "room1": "configs/train_gs/replica_room1.yaml",
    "fr3_office": "configs/train_gs/tum_fr3_office.yaml",
    "fr1_desk": "configs/train_gs/tum_fr1_desk.yaml",
}

def eval_psnr(ckpt_path, config_path, num_frames=50):
    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(config_path),
    )
    tc = cfg.train_gs
    device = "cuda"

    if cfg.scene.type == "tum":
        dataset = TUMDataset(cfg.scene.data_dir)
    else:
        dataset = ReplicaDataset(cfg.scene.data_dir)

    K = dataset.get_intrinsics().float().to(device)
    W, H = dataset.image_size
    W_t = W // tc.data_factor
    H_t = H // tc.data_factor

    K_t = K.clone()
    K_t[0, 0] /= tc.data_factor
    K_t[1, 1] /= tc.data_factor
    K_t[0, 2] /= tc.data_factor
    K_t[1, 2] /= tc.data_factor

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    means = ckpt["means"].to(device)
    quats = ckpt["quats"].to(device)
    scales = ckpt["scales"].to(device)
    opacities = ckpt["opacities"].to(device)
    sh_coeffs = ckpt["sh_coeffs"].to(device)
    sh_degree = ckpt["sh_degree"]

    eval_indices = dataset.get_eval_indices(stride=max(1, len(dataset) // num_frames))
    psnr_values = []

    with torch.no_grad():
        for ei in eval_indices[:num_frames]:
            sample = dataset[ei]
            gt = sample["image"].float().to(device)
            pose = sample["pose"].float().to(device)

            if tc.data_factor > 1:
                gt = F.interpolate(
                    gt.permute(2, 0, 1).unsqueeze(0),
                    size=(H_t, W_t), mode="bilinear", align_corners=False,
                ).squeeze(0).permute(1, 2, 0)

            viewmat = torch.linalg.inv(pose).unsqueeze(0)
            renders, _, _ = rasterization(
                means=means, quats=quats,
                scales=torch.exp(scales),
                opacities=torch.sigmoid(opacities),
                colors=sh_coeffs,
                viewmats=viewmat, Ks=K_t.unsqueeze(0),
                width=W_t, height=H_t,
                sh_degree=sh_degree, packed=False,
                near_plane=tc.near_plane, far_plane=tc.far_plane,
            )
            rendered = renders[0].clamp(0, 1)
            mse = F.mse_loss(rendered, gt)
            psnr = -10.0 * torch.log10(mse + 1e-8)
            psnr_values.append(psnr.item())

    return sum(psnr_values) / len(psnr_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", help="Checkpoint directory")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    print(f"\n=== PSNR Evaluation: {ckpt_dir} ===")
    for scene, config in CONFIGS.items():
        ckpt_path = ckpt_dir / f"{scene}.ckpt"
        if not ckpt_path.exists():
            print(f"{scene}: MISSING")
            continue
        psnr = eval_psnr(str(ckpt_path), config)
        print(f"{scene}: {psnr:.2f} dB")
