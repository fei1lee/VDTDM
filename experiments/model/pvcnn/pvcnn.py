import torch
import torch.nn as nn
from .pvcnn_utils import create_mlp_components, create_pointnet2_sa_components, create_pointnet2_fp_modules

from experiments.model.point_u_transformer.pvcnn.pvcnn_utils import create_mlp_components, \
    create_pointnet2_sa_components, create_pointnet2_fp_modules


class PVCNN2Base(nn.Module):
    def __init__(
            self,
            num_classes: int,
            use_att: bool = True,
            dropout: float = 0.1,
            extra_feature_channels: int = 3,
            width_multiplier: int = 1,
            voxel_resolution_multiplier: int = 1,
    ):
        super().__init__()
        assert extra_feature_channels >= 0
        self.dropout = dropout
        self.width_multiplier = width_multiplier

        self.in_channels = extra_feature_channels + 3

        # Create PointNet-2 model
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks,
            extra_feature_channels=extra_feature_channels,
            with_se=True,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        # Only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        # Create MLP layers
        self.channels_fp_features = channels_fp_features
        layers, _ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[128, dropout, num_classes],  # was 0.5
            classifier=True,
            dim=2,
            width_multiplier=width_multiplier
        )
        self.classifier = nn.Sequential(*layers)


class PVCNN2SA(nn.Module):
    sa_blocks = [
        ((32, 2, 32), (512, 0.1, 32, (32, 128, 384))),
        # ((64, 3, 16), (512, 0.2, 32, (128, 384))),
    ]

    def __init__(self, use_att=True, dropout=0.1, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.dropout = dropout
        self.width_multiplier = width_multiplier

        self.in_channels = extra_feature_channels + 3

        # Create PointNet-2 model
        sa_layers, self.sa_in_channels, self.channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks,
            extra_feature_channels=extra_feature_channels,
            with_se=True,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

    def forward(self, inputs):
        inputs = inputs.transpose(-1, -2)
        # Separate input coordinates and features
        coords = inputs[:, :3, :].contiguous()  # (B, 3, N)
        features = inputs  # (B, 3 + S, N)

        # Downscaling layers
        coords_list = []
        in_features_list = []
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords = sa_blocks((features, coords))
        # Replace the input features
        in_features_list[0] = inputs[:, 3:, :].contiguous()
        return in_features_list, coords_list, features, coords


class PVCNN2FP(nn.Module):
    fp_blocks = [
        # ((256, 128), (128, 2, 16)),
        ((128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, sa_in_channels, channels_sa_features, use_att=True, dropout=0.1,
                 extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1, ):
        super().__init__()
        assert extra_feature_channels >= 0
        self.dropout = dropout
        self.width_multiplier = width_multiplier

        # Only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        # Create MLP layers
        self.channels_fp_features = channels_fp_features
        layers, _ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[128, dropout, num_classes],  # was 0.5
            classifier=True,
            dim=2,
            width_multiplier=width_multiplier
        )
        self.classifier = nn.Sequential(*layers)

    def forward(self, in_features_list, coords_list, features, coords):
        # Upscaling layers
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords = fp_blocks(
                (  # this is a tuple because of nn.Sequential
                    coords_list[-1 - fp_idx],  # reverse coords list from above
                    coords,  # original point coordinates
                    features,  # keep concatenating upsampled features
                    in_features_list[-1 - fp_idx],  # reverse features list from above
                )
            )
        # Output MLP layers
        output = self.classifier(features)
        return output.transpose(1, 2)


if __name__ == '__main__':
    model = PVCNN2(
        num_classes=3,
        extra_feature_channels=0,
        dropout=0.1, width_multiplier=1,
        voxel_resolution_multiplier=1
    )
    model.cuda()
    print(f'Parameters (total): {sum(p.numel() for p in model.parameters()):_d}')
    print(f'Parameters (train): {sum(p.numel() for p in model.parameters() if p.requires_grad):_d}')
    point = torch.randn((16, 8192, 3)).cuda()
    in_features_list, coords_list, features, coords = model.down(point)
    print(1)
