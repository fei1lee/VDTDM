import os
import json
from collections import OrderedDict
import numpy as np

import open3d

from config.structured import PointCloudDatasetConfig


def id_to_name(id, category_list):
    for k, v in category_list.items():
        if v[0] <= id < v[1]:
            return k, id - v[0]


def category_model_id_pair(dataset_portion=None, cfg: PointCloudDatasetConfig = None):
    '''
    Load category, model names from a shapenet dataset.
    '''

    if dataset_portion is None:
        dataset_portion = [0, 0.8]

    def model_names(model_path):
        """ Return model names"""
        model_names = [name for name in os.listdir(model_path)
                       if os.path.isdir(os.path.join(model_path, name))]
        return sorted(model_names)

    category_name_pair = []  # full path of the objs files

    cats = json.load(open(cfg.dataset))
    cats = OrderedDict(sorted(cats.items(), key=lambda x: x[0]))

    for k, cat in cats.items():  # load by categories
        if cfg.category_id is not None and cat["id"] not in cfg.category_id:
            continue
        model_path = os.path.join(cfg.dataset_path, cat['id'])
        # category = cat['name']
        models = model_names(model_path)
        num_models = len(models)

        portioned_models = models[int(num_models * dataset_portion[0]):int(num_models *
                                                                           dataset_portion[1])]

        category_name_pair.extend([(cat['id'], model_id) for model_id in portioned_models])

    print('lib/data_io.py: model paths from %s' % (cfg.dataset))

    return category_name_pair


def get_point_cloud_file(category, model_id, cfg: PointCloudDatasetConfig = None):
    return cfg.pcd_path % (category, model_id)


def get_voxel_file(category, model_id, cfg: PointCloudDatasetConfig = None):
    return cfg.voxel_path % (category, model_id)


def get_rendering_file(category, model_id, rendering_id, cfg: PointCloudDatasetConfig = None):
    return os.path.join(cfg.rendering_path % (category, model_id), '%02d.png' % rendering_id)


def get_camera_info(category, model_id, rendering_ids, cfg: PointCloudDatasetConfig = None):
    meta_file = os.path.join(cfg.camera_path % (category, model_id))
    cameras = []
    with open(meta_file, 'r') as f:
        for i, line in enumerate(f):
            ls = line.strip().split()
            cameras.append(list(map(float, ls)))
    res = []
    for idx in rendering_ids:
        res.append(cameras[idx])
    return res


def read_point_set(fp):
    pcd = open3d.io.read_point_cloud(fp)
    return np.array(pcd.points)
