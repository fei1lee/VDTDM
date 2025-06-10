from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from hydra.conf import RunDir
import platform


@dataclass
class CustomHydraRunDir(RunDir):
    dir: str = './outputs/${run.name}/${now:%Y-%m-%d--%H-%M-%S}'


@dataclass
class RunConfig:
    name: str = 'debug'
    job: str = 'train'
    mixed_precision: str = 'fp16'  # 'no'
    cpu: bool = False
    seed: int = 42
    val_before_training: bool = False
    vis_before_training: bool = False
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    max_steps: int = 300_000
    checkpoint_freq: int = 1000
    val_freq: int = 5_000
    vis_freq: int = 5_000
    log_step_freq: int = 20
    print_step_freq: int = 100

    # Inference config
    num_inference_steps: int = 1000
    diffusion_scheduler: Optional[str] = 'ddpm'
    num_samples: int = 1
    num_sample_batches: Optional[int] = 2
    sample_from_ema: bool = False
    sample_save_evolutions: bool = True  # temporarily set by default
    epoch: int = 10000
    # Training config
    freeze_feature_model: bool = True


@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = 'pmb'


@dataclass
class PointCloudMultiViewModelConfig:
    # Feature extraction arguments
    image_size: int = '${dataset.image_size}'
    image_feature_model: str = 'vit_base_patch16_224_mae'  # or 'vit_base_patch16_224_mae' or 'identity' or 'vit_small_patch16_224_msn'
    w: float = 7.5
    # TODO
    # # New for the rebuttal
    # use_naive_projection: bool = False
    # use_feature_blur: bool = False

    # Point cloud data arguments. Note these are here because the processing happens
    # inside the model, rather than inside the dataset.
    scale_factor: float = "${dataset.scale_factor}"


@dataclass
class PointCloudDiffusionModelConfig(PointCloudMultiViewModelConfig):
    # Diffusion arguments
    beta_start: float = 0.00085  # 1e-5
    beta_end: float = 0.012  # 8e-3
    beta_schedule: str = 'linear'  # 'custom'
    n_views: int = 1
    n_points: int = 8192

@dataclass
class PointMambaConfig(PointCloudDiffusionModelConfig):
    dim: int = 768
    depth: int = 4
    mlp_dim: int = 3072
    drop_rate: float = 0.1


@dataclass
class DatasetConfig:
    type: str


@dataclass
class PointCloudDatasetConfig(DatasetConfig):
    eval_split: str = 'val'
    max_points: int = 16_384
    # image_size: int = 137
    image_size: int = 224
    # image_size: int = 256
    restrict_model_ids: Optional[List] = None  # for only running on a subset of data points


# @dataclass
# class CO3DConfig(PointCloudDatasetConfig):
#     type: str = 'co3dv2'
#     # root: str = os.getenv('CO3DV2')
#     root: str = '/media/lee/software/datasets/co3dv2'
#     # root: str = 'E:/datasets/co3d'
#     # root: str = '/mnt/e/datasets/co3d'
#     category: str = 'hydrant'
#     subset_name: str = 'fewview_dev'
#     mask_images: bool = '${model.use_mask}'


@dataclass
class ShapeNetR2N2Config(PointCloudDatasetConfig):
    n_views: int = 1
    # ubuntu: str = '/media/u401/project/phil'
    ubuntu: str = '/mnt/d'
    win: str = '/mnt/d'
    dataset_path: str = (ubuntu if platform.system() == 'Linux' else win) + '/datasets/ShapeNetPart.v1'
    model_path: str = dataset_path + '/%s/%s/model.obj'
    rendering_path: str = dataset_path + '/%s/%s/rendering'
    camera_path: str = dataset_path + '/%s/%s/rendering/rendering_metadata.txt'
    dataset: str = dataset_path + '/cat1000.json'
    cat_file: str = dataset_path + '/category.txt'
    pcd_path: str = dataset_path + '/%s/%s/model_8192.ply'
    num_rendering: int = 24
    category_id: Optional[List] = None
    scale_factor: float = 7.0


@dataclass
class AugmentationConfig:
    pass


@dataclass
class DataloaderConfig:
    batch_size: int = 8  # 2 for debug
    num_workers: int = 4  # 0 for debug


@dataclass
class LossConfig:
    diffusion_weight: float = 1.0
    rgb_weight: float = 1.0
    consistency_weight: float = 1.0


@dataclass
class CheckpointConfig:
    resume: Optional[str] = None
    resume_training: bool = True
    resume_training_optimizer: bool = True
    resume_training_scheduler: bool = True
    resume_training_state: bool = True


@dataclass
class ExponentialMovingAverageConfig:
    use_ema: bool = False
    # # From Diffusers EMA (should probably switch)
    # ema_inv_gamma: float = 1.0
    # ema_power: float = 0.75
    # ema_max_decay: float = 0.9999
    decay: float = 0.999
    update_every: int = 20


@dataclass
class OptimizerConfig:
    type: str
    name: str
    lr: float = 3e-4
    weight_decay: float = 0.0
    scale_learning_rate_with_batch_size: bool = False
    gradient_accumulation_steps: int = 1
    clip_grad_norm: Optional[float] = 50.0  # 5.0
    kwargs: Dict = field(default_factory=lambda: dict())


@dataclass
class AdadeltaOptimizerConfig(OptimizerConfig):
    type: str = 'torch'
    name: str = 'Adadelta'
    kwargs: Dict = field(default_factory=lambda: dict(
        weight_decay=1e-6,
    ))


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    type: str = 'torch'
    name: str = 'AdamW'
    weight_decay: float = 1e-6
    kwargs: Dict = field(default_factory=lambda: dict(betas=(0.95, 0.999)))


@dataclass
class SchedulerConfig:
    type: str
    kwargs: Dict = field(default_factory=lambda: dict())


@dataclass
class LinearSchedulerConfig(SchedulerConfig):
    type: str = 'transformers'
    kwargs: Dict = field(default_factory=lambda: dict(
        name='linear',
        num_warmup_steps=0,
        num_training_steps="${run.max_steps}",
    ))


@dataclass
class CosineSchedulerConfig(SchedulerConfig):
    type: str = 'transformers'
    kwargs: Dict = field(default_factory=lambda: dict(
        name='cosine',
        num_warmup_steps=2000,  # 0
        num_training_steps="${run.max_steps}",
    ))


@dataclass
class MultiViewConfig:
    run: RunConfig
    logging: LoggingConfig
    dataset: PointCloudDatasetConfig
    augmentations: AugmentationConfig
    dataloader: DataloaderConfig
    loss: LossConfig
    model: PointCloudMultiViewModelConfig
    ema: ExponentialMovingAverageConfig
    checkpoint: CheckpointConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    defaults: List[Any] = field(default_factory=lambda: [
        'custom_hydra_run_dir',
        {'run': 'default'},
        {'logging': 'default'},
        {'model': 'point_mamba'},
        {'dataset': 'shapenet_r2n2'},
        {'augmentations': 'default'},
        {'dataloader': 'default'},
        {'ema': 'default'},
        {'loss': 'default'},
        {'checkpoint': 'default'},
        {'optimizer': 'adam'},
        {'scheduler': 'cosine'},
    ])


cs = ConfigStore.instance()
cs.store(name='custom_hydra_run_dir', node=CustomHydraRunDir, package="hydra.run")
cs.store(group='run', name='default', node=RunConfig)
cs.store(group='logging', name='default', node=LoggingConfig)
cs.store(group='model', name='point_mamba', node=PointMambaConfig)
# cs.store(group='dataset', name='co3d', node=CO3DConfig)
# TODO
cs.store(group='dataset', name='shapenet_r2n2', node=ShapeNetR2N2Config)
# cs.store(group='dataset', name='shapenet_nmr', node=ShapeNetNMRConfig)
cs.store(group='augmentations', name='default', node=AugmentationConfig)
cs.store(group='dataloader', name='default', node=DataloaderConfig)
cs.store(group='loss', name='default', node=LossConfig)
cs.store(group='ema', name='default', node=ExponentialMovingAverageConfig)
cs.store(group='checkpoint', name='default', node=CheckpointConfig)
cs.store(group='optimizer', name='adadelta', node=AdadeltaOptimizerConfig)
cs.store(group='optimizer', name='adam', node=AdamOptimizerConfig)
cs.store(group='scheduler', name='linear', node=LinearSchedulerConfig)
cs.store(group='scheduler', name='cosine', node=CosineSchedulerConfig)
cs.store(name='config', node=MultiViewConfig)
