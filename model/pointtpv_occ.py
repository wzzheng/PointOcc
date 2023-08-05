import torch
from mmengine.model import BaseModule
from mmengine.registry import build_from_cfg
from mmdet3d.registry import MODELS


@MODELS.register_module()
class PointTPV_Occ(BaseModule):
    
    def __init__(self,
                 lidar_tokenizer=None,
                 lidar_backbone=None,
                 lidar_neck=None,
                 tpv_aggregator=None,
                 **kwargs,
                 ):

        super().__init__()

        if lidar_tokenizer:
            self.lidar_tokenizer = build_from_cfg(lidar_tokenizer, MODELS)
        if lidar_backbone:
            self.lidar_backbone = build_from_cfg(lidar_backbone, MODELS)
        if lidar_neck:
            self.lidar_neck = build_from_cfg(lidar_neck, MODELS)
        if tpv_aggregator:
            self.tpv_aggregator = build_from_cfg(tpv_aggregator, MODELS)

        self.fp16_enabled = False

    def extract_lidar_feat(self, points, grid_ind):
        """Extract features of points."""
        x_3view = self.lidar_tokenizer(points, grid_ind)
        tpv_list = []
        x_tpv = self.lidar_backbone(x_3view)
        for x in x_tpv:
            x = self.lidar_neck(x)
            if not isinstance(x, torch.Tensor):
                x = x[0]
            tpv_list.append(x)
        return tpv_list

    def forward(self,
                points=None,
                grid_ind=None,
                grid_ind_vox=None,
                grid_ind_vox_coarse=None,
                voxel_label=None,
                return_loss=True,
        ):
        """Forward training function.
        """
        x_lidar_tpv = self.extract_lidar_feat(points=points, grid_ind=grid_ind)
        outs = self.tpv_aggregator(x_lidar_tpv, voxels=grid_ind_vox, voxels_coarse=grid_ind_vox_coarse, voxel_label=voxel_label, return_loss=return_loss)
        return outs