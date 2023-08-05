import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import numpy as np
from spconv.pytorch import SparseConvTensor, SparseMaxPool3d
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class CylinderEncoder_Seg(BaseModule):

    def __init__(self, grid_size, grid_size_rv=None, in_channels=10, out_channels=256, 
                 fea_compre=16, base_channels=32, split=[4,4,4], track_running_stats=True):
        super(CylinderEncoder_Seg, self).__init__()
        
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        self.split = split
        if grid_size_rv is not None:
            self.grid_size_rv = [grid_size[0], grid_size_rv, grid_size[2]]
        
        # point-wise mlp
        self.point_mlp = nn.Sequential(
            nn.BatchNorm1d(in_channels, track_running_stats=track_running_stats),

            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64, track_running_stats=track_running_stats),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128, track_running_stats=track_running_stats),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256, track_running_stats=track_running_stats),
            nn.ReLU(),

            nn.Linear(256, out_channels)
        )

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(out_channels, fea_compre),
                nn.ReLU())
        
        # sparse max pooling
        
        self.pool_xy = SparseMaxPool3d(kernel_size=[1,1,int(self.grid_size[2]/split[2])],
                                              stride=[1,1,int(self.grid_size[2]/split[2])], padding=0)
        self.pool_yz = SparseMaxPool3d(kernel_size=[int(self.grid_size[0]/split[0]),1,1],
                                              stride=[int(self.grid_size[0]/split[0]),1,1], padding=0)
        self.pool_zx = SparseMaxPool3d(kernel_size=[1,int(self.grid_size[1]/split[1]),1],
                                              stride=[1,int(self.grid_size[1]/split[1]),1], padding=0)
        
        in_channels = [int(base_channels * s) for s in split]
        out_channels = [int(base_channels) for s in split]
        self.mlp_xy = nn.Sequential(nn.Linear(in_channels[2], out_channels[2]), nn.ReLU(), nn.Linear(out_channels[2], out_channels[2]))
        self.mlp_yz = nn.Sequential(nn.Linear(in_channels[0], out_channels[0]), nn.ReLU(), nn.Linear(out_channels[0], out_channels[0]))
        self.mlp_zx = nn.Sequential(nn.Linear(in_channels[1], out_channels[1]), nn.ReLU(), nn.Linear(out_channels[1], out_channels[1]))

    def forward(self, points, grid_ind):
        device = points[0].get_device()

        cat_pt_ind, cat_pt_fea = [], []
        for i_batch, res in enumerate(grid_ind):
            cat_pt_ind.append(F.pad(grid_ind[i_batch], (1, 0), 'constant', value=i_batch))

        # cat_pt_fea = torch.cat(points, dim=0)
        cat_pt_fea = points.squeeze()
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=device)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # process feature
        processed_cat_pt_fea = self.point_mlp(cat_pt_fea)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data
        
        # sparse conv & max pooling
        coors = unq.int()
        batch_size = coors[-1][0] + 1
        ret = SparseConvTensor(processed_pooled_data, coors, np.array(self.grid_size), batch_size)
        # ret = self.spconv(ret)
        tpv_xy = self.mlp_xy(self.pool_xy(ret).dense().permute(0,2,3,4,1).flatten(start_dim=3)).permute(0,3,1,2)
        tpv_yz = self.mlp_yz(self.pool_yz(ret).dense().permute(0,3,4,2,1).flatten(start_dim=3)).permute(0,3,1,2)
        tpv_zx = self.mlp_zx(self.pool_zx(ret).dense().permute(0,4,2,3,1).flatten(start_dim=3)).permute(0,3,1,2)

        return [tpv_xy, tpv_yz, tpv_zx]


@MODELS.register_module()
class CylinderEncoder_Occ(BaseModule):

    def __init__(self, grid_size, grid_size_rv=None, in_channels=10, out_channels=256, 
                 fea_compre=16, base_channels=32, split=[4,4,4], track_running_stats=True):
        super(CylinderEncoder_Occ, self).__init__()
        
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        self.split = split
        if grid_size_rv is not None:
            self.grid_size_rv = [grid_size[0], grid_size_rv, grid_size[2]]
        
        # point-wise mlp
        self.point_mlp = nn.Sequential(
            nn.BatchNorm1d(in_channels, track_running_stats=track_running_stats),

            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64, track_running_stats=track_running_stats),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128, track_running_stats=track_running_stats),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256, track_running_stats=track_running_stats),
            nn.ReLU(),

            nn.Linear(256, out_channels)
        )

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(out_channels, fea_compre),
                nn.ReLU())
        
        # sparse max pooling
        
        self.pool_xy = SparseMaxPool3d(kernel_size=[1,1,int(self.grid_size[2]/split[2])],
                                              stride=[1,1,int(self.grid_size[2]/split[2])], padding=0)
        self.pool_yz = SparseMaxPool3d(kernel_size=[int(self.grid_size[0]/split[0]),1,1],
                                              stride=[int(self.grid_size[0]/split[0]),1,1], padding=0)
        self.pool_zx = SparseMaxPool3d(kernel_size=[1,int(self.grid_size[1]/split[1]),1],
                                              stride=[1,int(self.grid_size[1]/split[1]),1], padding=0)
        
        in_channels = [int(base_channels * s) for s in split]
        out_channels = [int(base_channels) for s in split]
        self.mlp_xy = nn.Sequential(nn.Linear(in_channels[2], out_channels[2]), nn.ReLU(), nn.Linear(out_channels[2], out_channels[2]))
        self.mlp_yz = nn.Sequential(nn.Linear(in_channels[0], out_channels[0]), nn.ReLU(), nn.Linear(out_channels[0], out_channels[0]))
        self.mlp_zx = nn.Sequential(nn.Linear(in_channels[1], out_channels[1]), nn.ReLU(), nn.Linear(out_channels[1], out_channels[1]))

    def forward(self, points, grid_ind):
        device = points[0].get_device()

        cat_pt_ind, cat_pt_fea = [], []
        for i_batch, res in enumerate(grid_ind):
            cat_pt_ind.append(F.pad(grid_ind[i_batch], (1, 0), 'constant', value=i_batch))

        # cat_pt_fea = torch.cat(points, dim=0)
        cat_pt_fea = points.squeeze()
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=device)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # process feature
        processed_cat_pt_fea = self.point_mlp(cat_pt_fea)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data
        
        # sparse conv & max pooling
        coors = unq.int()
        batch_size = coors[-1][0] + 1
        ret = SparseConvTensor(processed_pooled_data, coors, np.array(self.grid_size), batch_size)
        # ret = self.spconv(ret)
        tpv_xy = self.mlp_xy(self.pool_xy(ret).dense().permute(0,2,3,4,1).flatten(start_dim=3)).permute(0,3,1,2)
        tpv_yz = self.mlp_yz(self.pool_yz(ret).dense().permute(0,3,4,2,1).flatten(start_dim=3)).permute(0,3,1,2)
        tpv_zx = self.mlp_zx(self.pool_zx(ret).dense().permute(0,4,2,3,1).flatten(start_dim=3)).permute(0,3,1,2)

        return [tpv_xy, tpv_yz, tpv_zx]