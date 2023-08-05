
import torch, torch.nn as nn, torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from copy import deepcopy
from utils.lovasz_losses import lovasz_softmax
from utils.sem_geo_loss import geo_scal_loss, sem_scal_loss
import numpy as np


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret


@MODELS.register_module()
class TPVAggregator_Seg(BaseModule):
    def __init__(
        self, tpv_h, tpv_w, tpv_z, nbr_classes=20, 
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, use_checkpoint=False
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        self.classifier = nn.Linear(out_dims, nbr_classes)
        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint
    
    def forward(self, tpv_list, points=None):
        """
        x y z -> w h z
        tpv_list[0]: bs, c, w, h
        tpv_list[1]: bs, c, h, z
        tpv_list[2]: bs, c, z, w
        """
        tpv_xy, tpv_yz, tpv_zx = tpv_list[0], tpv_list[1], tpv_list[2]
        tpv_hw = tpv_xy.permute(0, 1, 3, 2)
        tpv_wz = tpv_zx.permute(0, 1, 3, 2)
        tpv_zh = tpv_yz.permute(0, 1, 3, 2)
        bs, c, _, _ = tpv_hw.shape

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw, 
                size=(int(self.tpv_h*self.scale_h), int(self.tpv_w*self.scale_w)),
                mode='bilinear'
            )
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh, 
                size=(int(self.tpv_z*self.scale_z), int(self.tpv_h*self.scale_h)),
                mode='bilinear'
            )
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz, 
                size=(int(self.tpv_w*self.scale_w), int(self.tpv_z*self.scale_z)),
                mode='bilinear'
            )
        
        if points is not None:
            # points: bs, n, 3
            _, n, _ = points.shape
            points = points.reshape(bs, 1, n, 3)
            voxels = torch.unique(torch.floor(points), dim=2)
            voxels_ind = deepcopy(voxels).type(torch.long).squeeze()
        
            voxels += 0.5
            points[..., 0] = points[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
            points[..., 1] = points[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
            points[..., 2] = points[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1
            sample_loc = points[:, :, :, [0, 1]]
            tpv_hw_pts = F.grid_sample(tpv_hw, sample_loc, padding_mode="border").squeeze(2) # bs, c, n
            sample_loc = points[:, :, :, [1, 2]]
            tpv_zh_pts = F.grid_sample(tpv_zh, sample_loc, padding_mode="border").squeeze(2)
            sample_loc = points[:, :, :, [2, 0]]
            tpv_wz_pts = F.grid_sample(tpv_wz, sample_loc, padding_mode="border").squeeze(2)

            voxels[..., 0] = voxels[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
            voxels[..., 1] = voxels[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
            voxels[..., 2] = voxels[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1

            sample_loc_vox = voxels[:, :, :, [0, 1]]
            tpv_hw_vox = F.grid_sample(tpv_hw, sample_loc_vox, padding_mode="border").squeeze(2) # bs, c, n
            sample_loc_vox = voxels[:, :, :, [1, 2]]
            tpv_zh_vox = F.grid_sample(tpv_zh, sample_loc_vox, padding_mode="border").squeeze(2)
            sample_loc_vox = voxels[:, :, :, [2, 0]]
            tpv_wz_vox = F.grid_sample(tpv_wz, sample_loc_vox, padding_mode="border").squeeze(2)
            fused_vox = tpv_hw_vox + tpv_zh_vox + tpv_wz_vox
            
            # tpv_hw_vox = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, int(self.scale_z*self.tpv_z))
            # tpv_zh_vox = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, int(self.scale_w*self.tpv_w), -1, -1)
            # tpv_wz_vox = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, int(self.scale_h*self.tpv_h), -1)
            # fused_vox = (tpv_hw_vox + tpv_zh_vox + tpv_wz_vox).flatten(2)
            
            fused_pts = tpv_hw_pts + tpv_zh_pts + tpv_wz_pts
            fused = torch.cat([fused_vox, fused_pts], dim=-1) # bs, c, whz+n
            fused = fused.permute(0, 2, 1)
            if self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 2, 1)
            # logits_vox = logits[:, :, :(-n)].reshape(bs, self.classes, int(self.scale_w*self.tpv_w), int(self.scale_h*self.tpv_h), int(self.scale_z*self.tpv_z))
            feats_vox = logits[:, :, :(-n)].permute(0,2,1).squeeze()
            output_shape = [int(self.tpv_w*self.scale_w), int(self.tpv_h*self.scale_h), int(self.tpv_z*self.scale_z), feats_vox.shape[-1]]
            logits_vox = scatter_nd(indices=voxels_ind, updates=feats_vox, shape=output_shape).unsqueeze(0).permute(0,4,1,2,3)
            logits_pts = logits[:, :, (-n):].reshape(bs, self.classes, n, 1, 1)
            return logits_vox, logits_pts
            
        else:
            tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, int(self.scale_z*self.tpv_z))
            tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, int(self.scale_w*self.tpv_w), -1, -1)
            tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, int(self.scale_h*self.tpv_h), -1)
        
            fused = tpv_hw + tpv_zh + tpv_wz
            fused = fused.permute(0, 2, 3, 4, 1)
            if self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 4, 1, 2, 3)
        
            return logits
        

@MODELS.register_module()
class TPVAggregator_Occ(BaseModule):
    def __init__(
        self, tpv_h, tpv_w, tpv_z, grid_size_occ, coarse_ratio, loss_weight=[1,1,1,1],
        nbr_classes=20, in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, use_checkpoint=False
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.loss_weight = loss_weight
        self.grid_size_occ = np.asarray(grid_size_occ).astype(np.int32)
        self.coarse_ratio = coarse_ratio
        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        self.classifier = nn.Linear(out_dims, nbr_classes)
        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint
        
        self.ce_loss_func = nn.CrossEntropyLoss(ignore_index=255)
        self.lovasz_loss_func = lovasz_softmax
    
    def forward(self, tpv_list, voxels=None, voxels_coarse=None, voxel_label=None, return_loss=True):
        """
        x y z -> w h z
        tpv_list[0]: bs, c, w, h
        tpv_list[1]: bs, c, h, z
        tpv_list[2]: bs, c, z, w
        """
        tpv_xy, tpv_yz, tpv_zx = tpv_list[0], tpv_list[1], tpv_list[2]
        tpv_hw = tpv_xy.permute(0, 1, 3, 2)
        tpv_wz = tpv_zx.permute(0, 1, 3, 2)
        tpv_zh = tpv_yz.permute(0, 1, 3, 2)
        bs, c, _, _ = tpv_hw.shape

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw, 
                size=(int(self.tpv_h*self.scale_h), int(self.tpv_w*self.scale_w)),
                mode='bilinear'
            )
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh, 
                size=(int(self.tpv_z*self.scale_z), int(self.tpv_h*self.scale_h)),
                mode='bilinear'
            )
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz, 
                size=(int(self.tpv_w*self.scale_w), int(self.tpv_z*self.scale_z)),
                mode='bilinear'
            )
            
        # voxel_coarse: bs, (vox_w*vox_h*vox_z)/coarse_ratio**3, 3
        _, n, _ = voxels_coarse.shape
        voxels_coarse = voxels_coarse.reshape(bs, 1, n, 3)
        voxels_coarse[..., 0] = voxels_coarse[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
        voxels_coarse[..., 1] = voxels_coarse[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
        voxels_coarse[..., 2] = voxels_coarse[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1

        sample_loc_vox = voxels_coarse[:, :, :, [0, 1]]
        tpv_hw_vox = F.grid_sample(tpv_hw, sample_loc_vox, padding_mode="border").squeeze(2) # bs, c, n
        sample_loc_vox = voxels_coarse[:, :, :, [1, 2]]
        tpv_zh_vox = F.grid_sample(tpv_zh, sample_loc_vox, padding_mode="border").squeeze(2)
        sample_loc_vox = voxels_coarse[:, :, :, [2, 0]]
        tpv_wz_vox = F.grid_sample(tpv_wz, sample_loc_vox, padding_mode="border").squeeze(2)
        fused = tpv_hw_vox + tpv_zh_vox + tpv_wz_vox
        
        fused = fused.permute(0, 2, 1)   # bs, whz, c
        if self.use_checkpoint:
            fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
            logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
        else:
            fused = self.decoder(fused)
            logits = self.classifier(fused)
        W, H, D = int(self.grid_size_occ[0]/self.coarse_ratio), int(self.grid_size_occ[1]/self.coarse_ratio), int(self.grid_size_occ[2]/self.coarse_ratio)
        logits = logits.permute(0, 2, 1)
        B, C, N = logits.shape
        logits = logits.reshape(B, C, W, H ,D)
        
        if return_loss:
            # resize gt                       
            ratio = voxel_label.shape[2] // H
            if ratio != 1:
                voxel_label_coarse = voxel_label.reshape(B, W, ratio, H, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, W, H, D, ratio**3)
                empty_mask = voxel_label_coarse.sum(-1) == 0
                voxel_label_coarse = voxel_label_coarse.to(torch.int64)
                occ_space = voxel_label_coarse[~empty_mask]
                occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
                voxel_label_coarse[~empty_mask] = occ_space
                voxel_label_coarse = torch.mode(voxel_label_coarse, dim=-1)[0]
                voxel_label_coarse[voxel_label_coarse<0] = 255
                voxel_label_coarse = voxel_label_coarse.long()
                with torch.cuda.amp.autocast(enabled=False):
                    logits = logits.float()
                    loss = self.loss_weight[0]*self.ce_loss_func(logits, voxel_label_coarse) + self.loss_weight[1]*self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label_coarse, ignore=255) + \
                        self.loss_weight[2]*sem_scal_loss(logits, voxel_label_coarse, ignore_index=255) + self.loss_weight[3]*geo_scal_loss(logits, voxel_label_coarse, ignore_index=255, non_empty_idx=0)
            else:
                with torch.cuda.amp.autocast(enabled=False):
                    logits = logits.float()
                    loss = self.loss_weight[0]*self.ce_loss_func(logits, voxel_label) + self.loss_weight[1]*self.lovasz_loss_func(torch.softmax(logits, dim=1), voxel_label, ignore=255) + \
                        self.loss_weight[2]*sem_scal_loss(logits, voxel_label, ignore_index=255) + self.loss_weight[3]*geo_scal_loss(logits, voxel_label, ignore_index=255, non_empty_idx=0)
            return loss
        else:
            B_, W_, H_, D_ = voxel_label.shape
            pred = F.interpolate(logits, size=[W_, H_, D_], mode='trilinear', align_corners=False).contiguous()
            return pred