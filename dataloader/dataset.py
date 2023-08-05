import os
import numpy as np
from torch.utils import data
import yaml
import pickle
from mmcv.image.io import imread

class Seg_Point_NuScenes(data.Dataset):
    def __init__(self, data_path, imageset='train', label_mapping="nuscenes.yaml", nusc=None):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.nusc_infos = data['infos']
        self.data_path = data_path
        self.nusc = nusc

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]

        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join('data', self.nusc.get('lidarseg', lidar_sd_token)['filename'])
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        
        lidar_path = info['lidar_path']        
        points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])

        data_tuple = (points, points_label.astype(np.uint8))
        return data_tuple
    

class Occ_Point_NuScenes(data.Dataset):
    def __init__(self, data_path, occ_path, imageset='train', label_mapping="nuscenes.yaml", sweeps_num=-1, nusc=None):
        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.nusc_infos = data['infos']
        self.data_path = data_path
        self.occ_path = occ_path
        self.nusc = nusc
        self.sweeps_num = sweeps_num

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path']        
        points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])
        if self.sweeps_num > 0:
            sweep_points_list = [points]
            ts = info['timestamp']
            if len(info['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(info['sweeps']))
            else:
                choices = np.random.choice(len(info['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = info['sweeps'][idx]
                points_sweep = np.fromfile(sweep['data_path'], dtype=np.float32, count=-1).reshape([-1, 5])
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                sweep_points_list.append(points_sweep)
            points = np.concatenate(sweep_points_list, axis=0)
        
        rel_path = 'scene_{0}/occupancy/{1}.npy'.format(info['scene_token'], info['lidar_token'])
        #  [z y x cls]
        pcd = np.load(os.path.join(self.occ_path, rel_path))
        
        data_tuple = (pcd, points)
        return data_tuple


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name
