import torch
import torch.nn as nn

from .pvcnn_utils import create_mlp_components, create_pointnet2_sa_components, create_pointnet2_fp_modules, \
    get_timestep_embedding

__all__ = ['PointNet2SSG', 'PointNet2MSG']


class PointNet2(nn.Module):
    def __init__(self, num_classes, num_shapes, sa_blocks, fp_blocks, with_one_hot_shape_id=True,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1, embed_dim=64):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.in_channels = extra_feature_channels + 3
        self.num_shapes = num_shapes
        self.with_one_hot_shape_id = with_one_hot_shape_id

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=sa_blocks, extra_feature_channels=extra_feature_channels, width_multiplier=width_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        # use one hot vector in the last fp module
        sa_in_channels[0] += num_shapes if with_one_hot_shape_id else 0
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, 0.5, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)
        # Time embedding function
        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, inputs, t):
        t_emb = get_timestep_embedding(self.embed_dim, t, inputs.device).float()
        t_emb = self.embedf(t_emb)[:, :, None].expand(-1, -1, inputs.shape[-1])
        # inputs : [B, in_channels + S, N]
        features = inputs[:, :self.in_channels, :]
        if self.with_one_hot_shape_id:
            assert inputs.size(1) == self.in_channels + self.num_shapes
            features_with_one_hot_vectors = inputs
        else:
            features_with_one_hot_vectors = features

        coords, features = features[:, :3, :].contiguous(), features[:, 3:, :].contiguous()
        coords_list, in_features_list = [], []
        for sa_module in self.sa_layers:
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords, t_emb = sa_module((features, coords, t_emb))
        in_features_list[0] = features_with_one_hot_vectors.contiguous()

        for fp_idx, fp_module, t_emb in enumerate(self.fp_layers):
            features, coords = fp_module(
                (coords_list[-1 - fp_idx], coords, torch.cat([features, t_emb], dim=1), in_features_list[-1 - fp_idx]))

        return self.classifier(features)


class PointNet2SSG(PointNet2):
    sa_blocks = [
        (None, (512, 0.2, 64, (64, 64, 128))),
        (None, (128, 0.4, 64, (128, 128, 256))),
        (None, (None, None, None, (256, 512, 1024))),
    ]
    fp_blocks = [((256, 256), None), ((256, 128), None), ((128, 128, 128), None)]

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1):
        super().__init__(
            num_classes=num_classes, num_shapes=num_shapes, sa_blocks=self.sa_blocks, fp_blocks=self.fp_blocks,
            with_one_hot_shape_id=False, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )


class PointNet2MSG(PointNet2):
    sa_blocks = [
        (None, (512, [0.1, 0.2, 0.4], [32, 64, 128], [(32, 32, 64), (64, 64, 128), (64, 96, 128)])),
        (None, (128, [0.4, 0.8], [64, 128], [(128, 128, 256), (128, 196, 256)])),
        (None, (None, None, None, (256, 512, 1024))),
    ]
    fp_blocks = [((256, 256), None), ((256, 128), None), ((128, 128, 128), None)]

    def __init__(self, num_classes, num_shapes, extra_feature_channels=3, width_multiplier=1,
                 voxel_resolution_multiplier=1, embed_dim=64):
        super().__init__(
            num_classes=num_classes, num_shapes=num_shapes, sa_blocks=self.sa_blocks, fp_blocks=self.fp_blocks,
            with_one_hot_shape_id=True, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            embed_dim=embed_dim
        )
