import torch
from pytorch3d.renderer import CamerasBase, PointsRasterizationSettings, PointsRasterizer
from pytorch3d.structures import Pointclouds

from .pvcnn.pvcnn_utils import create_mlp_components, create_pointnet2_sa_components, create_pointnet2_fp_modules
from torch import nn, Tensor
import numpy as np

from flash_attn.modules.mha import MHA


def get_timestep_embedding(embed_dim, timesteps, device):
    """
    Timestep embedding function. Not that this should work just as well for
    continuous values as for discrete values.
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embed_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == torch.Size([timesteps.shape[0], embed_dim])
    return emb


class VIPCFBlock(nn.Module):

    def __init__(self, dim=768, mlp_dim=3072, drop_rate=0.0, num_center=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mha = MHA(embed_dim=dim, num_heads=8, use_flash_attn=True)
        self.norm2 = nn.LayerNorm(dim)
        self.visible_attention = MHA(embed_dim=dim, num_heads=8, )
        self.unvisible_attention = MHA(embed_dim=dim, num_heads=8, )
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop_rate)
        )
        self.w = nn.Parameter(torch.zeros(num_center + 1))

    def forward(self, x, c, visible_center):
        B, N, D = x.shape
        x = x + self.mha(self.norm1(x))
        x = self.norm2(x)
        visible_group = x[:, 1:, :] * visible_center.unsqueeze(2).expand(B, N - 1, D)
        visible_mask = visible_group.sum(dim=-1) == 0
        unvisible_mask = visible_mask == False
        _mask = torch.full((B, 1), True, device=x.device)
        mask_ = torch.full(c.shape[:2], True, device=x.device)
        visible_mask = torch.cat((_mask, visible_mask, mask_), dim=1)
        unvisible_mask = torch.cat((_mask, unvisible_mask, mask_), dim=1)
        visible_attn = self.visible_attention(torch.cat((x, c), dim=1), key_padding_mask=visible_mask)
        unvisible_attn = self.unvisible_attention(torch.cat((x, c), dim=1), key_padding_mask=unvisible_mask)
        visible_attn[~visible_mask] = 0
        unvisible_attn[~unvisible_mask] = 0
        w = self.w.view(1, self.w.shape[0], 1).repeat(B, 1, D)
        final_attn = visible_attn[:, :x.shape[1], :] + w * unvisible_attn[:, :x.shape[1], :]
        x = x + final_attn
        x = x + self.mlp(self.norm3(x))
        return x


class VIPCFEncoder(nn.Module):

    def __init__(self, dim=768, mlp_dim=3072, drop_rate=0.0, depth=4):
        super().__init__()
        self.depth = depth
        blocks = [VIPCFBlock(dim, mlp_dim, drop_rate) for _ in range(depth)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, condition, visible_center):
        for index, block in enumerate(self.blocks):
            x = block(x, condition, visible_center)
        return x


class VT3D(nn.Module):
    def __init__(self, dim=768, depth=12, mlp_dim=3072, drop_rate=0.1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.drop_rate = drop_rate
        self.pointnet_sa = PVCNN2SA(
            extra_feature_channels=0,
            dropout=0.1, width_multiplier=1,
            voxel_resolution_multiplier=1,
            dim=dim
        )
        self.pointnet_fp = PVCNN2FP(
            num_classes=3,
            sa_in_channels=self.pointnet_sa.sa_in_channels,
            channels_sa_features=self.pointnet_sa.channels_sa_features,
            extra_feature_channels=0,
            dropout=0.1, width_multiplier=1,
            voxel_resolution_multiplier=1
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.dim))
        self.encoder = VIPCFEncoder(dim=self.dim, depth=int(self.depth), mlp_dim=self.mlp_dim,
                                    drop_rate=self.drop_rate)

    def forward(self, pts, y, time_step, camera, scale):
        in_features_list, coords_list, group_input_tokens, center = self.pointnet_sa(pts)
        # divide the point cloud in the same form. This is important
        group_input_tokens = group_input_tokens.transpose(-1, -2).contiguous()
        visible_center = self.render_pointcloud_batch_pytorch3d(camera[0], pointclouds=center / scale)
        # time_step_encode
        time_embedding = get_timestep_embedding(self.dim, time_step, time_step.device)
        time_token = time_embedding.unsqueeze(1)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos_point = self.pos_embed(center.transpose(-1, -2).contiguous())
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos_point), dim=1)
        # condition
        condition = torch.cat((time_token, y), dim=1)
        # transformer
        x = self.encoder(x + pos, condition, visible_center)
        return self.pointnet_fp(in_features_list, coords_list, x[:, 1:, :].transpose(-1, -2).contiguous(), center)

    @torch.autocast('cuda', dtype=torch.float32)
    def render_pointcloud_batch_pytorch3d(
            self,
            cameras: CamerasBase,
            pointclouds: Tensor,
            image_size: int = 224,
            radius: float = 0.075,
            points_per_pixel: int = 1,
    ):
        pointclouds = pointclouds.transpose(-1, -2).contiguous()
        B, N, _ = pointclouds.shape
        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=radius,
            points_per_pixel=points_per_pixel,
        )
        pc = Pointclouds(pointclouds)
        # Rasterizer
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(pc)
        idx = fragments.idx
        visible_index = torch.full((B, N), 0, device=pointclouds.device)
        for i, item in enumerate(idx):
            visible_index[i, torch.unique(item[item > -1]) - i * N] = 1
        return visible_index


class PVCNN2SA(nn.Module):

    def __init__(self, use_att=True, dropout=0.1, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1, dim=384):
        super().__init__()
        assert extra_feature_channels >= 0
        self.sa_blocks = [
            ((32, 2, 32), (512, 0.1, 32, (128, dim)))
        ]
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
        inputs = inputs.transpose(-1, -2).contiguous()
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
        ((128,), (32, 2, 32))
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
        return output.transpose(1, 2).contiguous()
