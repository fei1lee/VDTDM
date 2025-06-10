from pytorch3d.datasets.r2n2.utils import compute_extrinsic_matrix, BlenderCamera
from torch.utils.data import Dataset
from typing import Dict

import json
from collections import OrderedDict

from config.structured import MultiViewConfig
from utils.data_io import category_model_id_pair, get_rendering_file, get_point_cloud_file, read_point_set, \
    get_camera_info
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


class ShapeNetPointRec(Dataset):

    def __init__(self,
                 dataset_portion=None,
                 cfg: MultiViewConfig = None
                 ) -> None:
        super().__init__()
        self.config = cfg.dataset
        self.data = category_model_id_pair(dataset_portion, self.config)
        self.to_tensor = transforms.ToTensor()
        self.torch_to_tensor = torch.tensor
        self.resize = transforms.Resize((self.config.image_size, self.config.image_size))
        self.cat = {}
        with open(self.config.cat_file, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.classes = {v: k for k, v in self.cat.items()}

        cats = json.load(open(cfg.dataset.dataset))
        self.cats = OrderedDict(sorted(cats.items(), key=lambda x: x[0]))

        # set up constants
        self.img_h = self.config.image_size
        self.img_w = self.config.image_size
        self.curr_n_views = self.config.n_views

        self.gray_transform = transforms.Compose([
            transforms.Grayscale()
        ])
        threshold = 0
        self.binary_transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: torch.where(x > threshold / 255, torch.tensor(255), torch.tensor(0)).byte()
        ])

    def __getitem__(self, index) -> Dict:
        category, model_id = self.data[index]
        cat = int(self.classes[category])
        batch_img = torch.zeros(
            (self.curr_n_views, 3, self.img_h, self.img_w), dtype=torch.float32)
        image_ids = np.random.choice(self.config.num_rendering, self.curr_n_views)
        # load multi view images
        for view_id, image_id in enumerate(image_ids):
            img, mask = self.load_img(category, model_id, image_id)
            batch_img[view_id, :, :, :] = img
        Rs, Ts, K, RT = self.load_camera_info(category, model_id, image_ids)
        return {
            "images": batch_img,
            "mask": mask,
            "label": self.load_label(category, model_id),
            "cat": self.torch_to_tensor(cat),
            "cat_label": category,
            "cat_name": self.cats[category]['name'],
            "model_id": model_id,
            "Rs": Rs,
            "Ts": Ts,
            "K": K,
            "RT": RT
        }

    def __len__(self):
        return len(self.data)

    def get_mask(self, image):

        gray_image = self.gray_transform(image)

        return self.binary_transform(gray_image)

    def load_label(self, category, model_id):
        pcf_fn = get_point_cloud_file(category, model_id, self.config)
        return self.torch_to_tensor(read_point_set(pcf_fn), dtype=torch.float32)

    def load_img(self, category, model_id, image_id):
        image_fn = get_rendering_file(category, model_id, image_id, self.config)
        im = Image.open(image_fn).convert('RGB')
        im = self.resize(im)
        return self.to_tensor(im), self.get_mask(im)

    def load_camera_info(self, category, model_id, image_ids):
        MAX_CAMERA_DISTANCE = 1.75
        Rs = []
        Ts = []
        metadata_lines = get_camera_info(category, model_id, image_ids, self.config)
        for i in range(len(metadata_lines)):
            azim, elev, yaw, dist_ratio, fov = metadata_lines[i]
            dist = dist_ratio * MAX_CAMERA_DISTANCE
            # Extrinsic matrix before transformation to PyTorch3D world space.
            RT = compute_extrinsic_matrix(azim, elev, dist)
            R, T, RT = self._compute_camera_calibration(RT)
            Rs.append(R)
            Ts.append(T)
        K = torch.tensor(
            [
                [2.1875, 0.0, 0.0, 0.0],
                [0.0, 2.1875, 0.0, 0.0],
                [0.0, 0.0, -1.002002, -0.2002002],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        # return BlenderCamera(torch.stack(Rs), torch.stack(Ts), K.expand(len(image_ids), 4, 4))
        return torch.stack(Rs), torch.stack(Ts), K.expand(len(image_ids), 4, 4), RT

    def _compute_camera_calibration(self, RT):
        """
        Helper function for calculating rotation and translation matrices from ShapeNet
        to camera transformation and ShapeNet to PyTorch3D transformation.

        Args:
            RT: Extrinsic matrix that performs ShapeNet world view to camera view
                transformation.

        Returns:
            R: Rotation matrix of shape (3, 3).
            T: Translation matrix of shape (3).
        """
        # Transform the mesh vertices from shapenet world to pytorch3d world.
        shapenet_to_pytorch3d = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        RT = torch.transpose(RT, 0, 1).mm(shapenet_to_pytorch3d)  # (4, 4)
        # Extract rotation and translation matrices from RT.
        R = RT[:3, :3]
        T = RT[3, :3]
        return R, T, RT
