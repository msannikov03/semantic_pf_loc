"""Configuration dataclasses."""

from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class SceneConfig:
    name: str = MISSING
    data_dir: str = MISSING
    type: str = "tum"


@dataclass
class TrainGSConfig:
    max_steps: int = 30000
    sh_degree: int = 3
    init_num_gaussians: int = 50000
    ssim_lambda: float = 0.2
    densify_start: int = 500
    densify_stop: int = 15000
    densify_every: int = 100
    lr_means: float = 1.6e-4
    lr_scales: float = 5.0e-3
    lr_opacities: float = 5.0e-2
    lr_sh: float = 2.5e-3
    lr_quats: float = 1.0e-3
    data_factor: int = 2
    batch_size: int = 1
    near_plane: float = 0.01
    far_plane: float = 100.0


@dataclass
class ParticleFilterConfig:
    num_particles: int = 400
    init_mode: str = "global"
    resample_threshold: float = 0.5
    render_width: int = 80
    render_height: int = 60
    render_width_hires: int = 320
    render_height_hires: int = 240
    convergence_threshold: float = 0.1


@dataclass
class MotionModelConfig:
    translation_std: float = 0.02
    rotation_std: float = 0.01


@dataclass
class ObservationConfig:
    type: str = "ssim"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    temperature: float = 10.0
    text_query: str = ""


@dataclass
class GradientRefineConfig:
    enabled: bool = False
    top_k: int = 10
    num_iterations: int = 50
    lr_init: float = 1.0e-2
    lr_final: float = 1.0e-5
    loss_type: str = "l1+ssim"
    ssim_weight: float = 0.2
    blur_schedule: bool = True


@dataclass
class EvaluationConfig:
    success_trans_threshold: float = 0.05
    success_rot_threshold: float = 2.0
    convergence_threshold: float = 0.1
    num_runs: int = 3


@dataclass
class Config:
    scene: SceneConfig = field(default_factory=SceneConfig)
    train_gs: TrainGSConfig = field(default_factory=TrainGSConfig)
    particle_filter: ParticleFilterConfig = field(default_factory=ParticleFilterConfig)
    motion_model: MotionModelConfig = field(default_factory=MotionModelConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    gradient_refine: GradientRefineConfig = field(default_factory=GradientRefineConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
