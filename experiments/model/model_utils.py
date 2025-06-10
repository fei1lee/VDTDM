import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.datasets import BlenderCamera
from pytorch3d.structures import Pointclouds


def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def compute_distance_transform(mask: torch.Tensor):
    image_size = mask.shape[-1]
    distance_transform = torch.stack([
        torch.from_numpy(cv2.distanceTransform(
            (1 - m), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3
        ) / (image_size / 2))
        for m in mask.squeeze(1).detach().cpu().numpy().astype(np.uint8)
    ]).unsqueeze(1).clip(0, 1).to(mask.device)
    return distance_transform


def default(x, d):
    return d if x is None else x


def get_num_points(x: Pointclouds, /):
    return x.points_padded().shape[1]


def get_custom_betas(beta_start: float, beta_end: float, warmup_frac: float = 0.3, num_train_timesteps: int = 1000):
    """Custom beta schedule"""
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    warmup_frac = 0.3
    warmup_time = int(num_train_timesteps * warmup_frac)
    warmup_steps = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    warmup_time = min(warmup_time, num_train_timesteps)
    betas[:warmup_time] = warmup_steps[:warmup_time]
    return betas


def tensor_to_point_cloud(x, scale_factor=7.0, unscale: bool = False):
    points = x[:, :, :3] / (scale_factor if unscale else 1)
    assert x.shape[2] == 3
    return Pointclouds(points=points)


def build_cameras(data, device):
    Rs = torch.transpose(data['Rs'], 0, 1)
    Ts = torch.transpose(data['Ts'], 0, 1)
    K = torch.transpose(data['K'], 0, 1)
    V = data['images'].shape[1]
    cameras = []
    for i in range(V):
        cameras.append(BlenderCamera(Rs[i], Ts[i], K[i]).to(device))
    return cameras
