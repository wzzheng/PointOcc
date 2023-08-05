
import numpy as np
import torch
import numba as nb
from torch.utils import data


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)


class DatasetWrapper_Point_NuScenes(data.Dataset):
    def __init__(self, in_dataset, grid_size, fill_label=0,
                 fixed_volume_space=False, max_volume_space=[51.2, 51.2, 3], 
                 min_volume_space=[-51.2, -51.2, -5],
                 rotate_aug=False, flip_aug=False, scale_aug=False):
        'Initialization'
        self.point_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.fill_label = fill_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug

    def __len__(self):
        return len(self.point_dataset)

    def __getitem__(self, index):
        data = self.point_dataset[index]
        points, labels = data
        
        # random points augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            points[:, :2] = np.dot(points[:, :2], j)

        # random points augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                points[:, 0] = -points[:, 0]
            elif flip_type == 2:
                points[:, 1] = -points[:, 1]
            elif flip_type == 3:
                points[:, :2] = -points[:, :2]
        # random points augmentation by scale x & y
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            points[:, 0] = noise_scale * points[:, 0]
            points[:, 1] = noise_scale * points[:, 1]

        assert self.fixed_volume_space
        max_bound = np.asarray(self.max_volume_space)  # 51.2 51.2 3
        min_bound = np.asarray(self.min_volume_space)  # -51.2 -51.2 -5
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size                 # 200, 200, 16
        # TODO: intervals should not minus one.
        intervals = crop_range / (cur_grid_size)   

        if (intervals == 0).any(): 
            print("Zero interval!")
        # TODO: grid_ind_float should actually be returned.
        grid_ind_float = (np.clip(points[:, :3], min_bound, max_bound - 1e-3) - min_bound) / intervals
        # grid_ind_float = (np.clip(xyz, min_bound, max_bound) - min_bound) / intervals
        grid_ind = np.floor(grid_ind_float).astype(np.int32)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.fill_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (points, processed_label)

        data_tuple += (grid_ind, labels)

        return data_tuple


def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


class Seg_DatasetWrapper_Point_NuScenes(data.Dataset):
    def __init__(self, in_dataset, grid_size, grid_size_vox=None, fill_label=0, fixed_volume_space=False,
                 max_volume_space=[50, np.pi, 3], min_volume_space=[0, -np.pi, -5],
                 rotate_aug=False, flip_aug=False, scale_aug=False, transform_aug=False, 
                 trans_std=[0.1, 0.1, 0.1]):
        'Initialization'
        self.point_dataset = in_dataset
        self.grid_size = np.asarray(grid_size).astype(np.int32)
        self.grid_size_vox = np.asarray(grid_size_vox).astype(np.int32)
        self.fill_label = fill_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.transform_aug = transform_aug
        self.trans_std = trans_std

    def __len__(self):
        return len(self.point_dataset)

    def __getitem__(self, index):
        data = self.point_dataset[index]
        points, labels = data
        xyz, feat = points[:, :3], points[:, 3:]

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        # random points augmentation by scale x & y
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        
        # random points augmentation by translate xyz
        if self.transform_aug:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate
        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        assert self.fixed_volume_space
        max_bound = np.asarray(self.max_volume_space)
        min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        intervals = crop_range / (self.grid_size)
        intervals_vox = crop_range / (self.grid_size_vox)

        if (intervals == 0).any(): 
            print("Zero interval!")
        xyz_pol_grid = np.clip(xyz_pol, min_bound, max_bound - 1e-3)
        grid_ind = (np.floor((xyz_pol_grid - min_bound) / intervals)).astype(np.int32)
        grid_ind_vox = (np.floor((xyz_pol_grid - min_bound) / intervals_vox)).astype(np.int32)
        grid_ind_vox_float = ((xyz_pol_grid - min_bound) / intervals_vox).astype(np.float32)

        # process labels
        processed_label = np.ones(self.grid_size_vox, dtype=np.uint8) * self.fill_label
        label_voxel_pair = np.concatenate([grid_ind_vox, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind_vox[:, 0], grid_ind_vox[:, 1], grid_ind_vox[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (processed_label, )

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_feat = np.concatenate((return_xyz, xyz_pol, xyz[:, :2], feat), axis=1)

        data_tuple += (grid_ind, labels, return_feat, grid_ind_vox_float)
        
        return data_tuple


class Occ_DatasetWrapper_Point_NuScenes(data.Dataset):
    def __init__(self, in_dataset, grid_size, grid_size_vox=None, grid_size_occ=None, coarse_ratio=4, pc_range=None, fill_label=0,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 3], min_volume_space=[0, -np.pi, -5],
                 rotate_aug=False, flip_aug=False, scale_aug=False, transform_aug=False, 
                 trans_std=[0.1, 0.1, 0.1]):
        'Initialization'
        self.point_dataset = in_dataset
        self.grid_size = np.asarray(grid_size).astype(np.int32)
        self.grid_size_vox = np.asarray(grid_size_vox).astype(np.int32)
        self.grid_size_occ = np.asarray(grid_size_occ).astype(np.int32)
        self.grid_size_occ_coarse = (np.asarray(grid_size_occ) / coarse_ratio).astype(np.int32)
        self.pc_range = np.asarray(pc_range).astype(np.float32)
        self.voxel_size = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size_occ
        self.voxel_size_coarse = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size_occ_coarse
        self.fill_label = fill_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.transform_aug = transform_aug
        self.trans_std = trans_std

    def __len__(self):
        return len(self.point_dataset)

    def __getitem__(self, index):
        data = self.point_dataset[index]
        pcd, points = data
        xyz, feat = points[:, :3], points[:, 3:]
        occ_label = pcd[..., -1:]
        occ_label[occ_label==0] = 255
        occ_xyz_grid = pcd[..., [2,1,0]]  # x y z
        xyz_pol = cart2polar(xyz)

        assert self.fixed_volume_space
        max_bound = np.asarray(self.max_volume_space)
        min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        intervals = crop_range / (self.grid_size)
        intervals_vox = crop_range / (self.grid_size_vox)
        if (intervals == 0).any(): 
            print("Zero interval!")
        xyz_pol_grid = np.clip(xyz_pol, min_bound, max_bound - 1e-3)
        grid_ind = (np.floor((xyz_pol_grid - min_bound) / intervals)).astype(np.int32)
        # get voxel_position_grid_coarse
        dim_array = np.ones(len(self.grid_size_occ_coarse) + 1, np.int32)
        dim_array[0] = -1
        voxel_position_coarse = ((np.indices(self.grid_size_occ_coarse) + 0.5) * self.voxel_size_coarse.reshape(dim_array) + self.pc_range[:3].reshape(dim_array)).reshape(3, -1).transpose(1,0)
        voxel_position_grid_coarse = (np.clip(cart2polar(voxel_position_coarse), min_bound, max_bound - 1e-3) - min_bound) / intervals_vox
        # process labels
        label_voxel_pair = np.concatenate([occ_xyz_grid, occ_label], axis=-1)
        label_voxel_pair = label_voxel_pair[np.lexsort((occ_xyz_grid[:, 0], occ_xyz_grid[:, 1], occ_xyz_grid[:, 2])), :].astype(np.int32)
        processed_label = np.ones(self.grid_size_occ, dtype=np.uint8) * self.fill_label
        processed_label = nb_process_label(processed_label, label_voxel_pair)
        data_tuple = (voxel_position_grid_coarse, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_feat = np.concatenate((return_xyz, xyz_pol, xyz[:, :2], feat), axis=1)

        data_tuple += (grid_ind, return_feat)
        
        return data_tuple


@nb.jit('u1[:,:,:](u1[:,:,:],i4[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def custom_collate_fn(data):
    points = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int32)
    # because we use a batch size of 1, so we can stack these tensor together.
    grid_ind_stack = np.stack([d[2] for d in data]).astype(np.float)
    point_label = np.stack([d[3] for d in data]).astype(np.int32)
    return torch.from_numpy(points), \
        torch.from_numpy(label2stack), \
        torch.from_numpy(grid_ind_stack), \
        torch.from_numpy(point_label)


def seg_custom_collate_fn(data):
    voxel_label = np.stack([d[0] for d in data]).astype(np.int32)
    grid_ind_stack = np.stack([d[1] for d in data]).astype(np.float32)
    point_label = np.stack([d[2] for d in data]).astype(np.int32)
    point_feat = np.stack([d[3] for d in data]).astype(np.float32)
    grid_ind_vox_stack = np.stack([d[4] for d in data]).astype(np.float32)
    
    return torch.from_numpy(point_feat), \
        torch.from_numpy(voxel_label), \
        torch.from_numpy(grid_ind_stack), \
        torch.from_numpy(point_label), \
        torch.from_numpy(grid_ind_vox_stack)


def occ_custom_collate_fn(data):
    voxel_position_coarse = np.stack([d[0] for d in data]).astype(np.float32)
    voxel_label = np.stack([d[1] for d in data]).astype(np.int32)
    grid_ind_stack = np.stack([d[2] for d in data]).astype(np.float32)
    point_feat = np.stack([d[3] for d in data]).astype(np.float32)
    
    return torch.from_numpy(voxel_position_coarse), \
        torch.from_numpy(point_feat), \
        torch.from_numpy(voxel_label), \
        torch.from_numpy(grid_ind_stack), \
