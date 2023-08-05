import torch
from dataloader.dataset import Seg_Point_NuScenes, Occ_Point_NuScenes
from dataloader.dataset_wrapper import seg_custom_collate_fn, Seg_DatasetWrapper_Point_NuScenes, \
occ_custom_collate_fn, Occ_DatasetWrapper_Point_NuScenes
from nuscenes import NuScenes


def build_seg(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          grid_size=[200, 200, 16],
          version='v1.0-trainval',
          dist=False,
    ):
    data_path = train_dataloader_config["data_path"]
    train_imageset = train_dataloader_config["imageset"]
    val_imageset = val_dataloader_config["imageset"]
    label_mapping = dataset_config["label_mapping"]

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    train_dataset = Seg_Point_NuScenes(data_path, imageset=train_imageset,
                                     label_mapping=label_mapping, nusc=nusc)
    val_dataset = Seg_Point_NuScenes(data_path, imageset=val_imageset,
                                   label_mapping=label_mapping, nusc=nusc)
    if dataset_config['dataset_type'] == 'Seg_DatasetWrapper_Point_NuScenes':
        train_dataset = Seg_DatasetWrapper_Point_NuScenes(
            train_dataset,
            grid_size=grid_size,
            grid_size_vox=dataset_config['grid_size_vox'],
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            fill_label=dataset_config["fill_label"],
            rotate_aug=dataset_config['rotate_aug'],
            flip_aug=dataset_config['flip_aug'],
            scale_aug=dataset_config['scale_aug'],
            transform_aug=dataset_config['transform_aug'],
            trans_std=dataset_config['trans_std'],
        )

        val_dataset = Seg_DatasetWrapper_Point_NuScenes(
            val_dataset,
            grid_size=grid_size,
            grid_size_vox=dataset_config['grid_size_vox'],
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            fill_label=dataset_config["fill_label"],
        )
        
        collate_fn = seg_custom_collate_fn
    else:
        raise Exception('invalid dataset_type')

    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        sampler = None
        val_sampler = None

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=collate_fn,
                                                       shuffle=False if dist else train_dataloader_config["shuffle"],
                                                       sampler=sampler,
                                                       num_workers=train_dataloader_config["num_workers"])
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn,
                                                     shuffle=False if dist else val_dataloader_config["shuffle"],
                                                     sampler=val_sampler,
                                                     num_workers=val_dataloader_config["num_workers"])

    return train_dataset_loader, val_dataset_loader


def build_occ(dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            grid_size=[200, 200, 16],
            version='v1.0-trainval',
            dist=False,
    ):
    data_path = train_dataloader_config["data_path"]
    occ_path = dataset_config["occ_path"]
    train_imageset = train_dataloader_config["imageset"]
    val_imageset = val_dataloader_config["imageset"]
    label_mapping = dataset_config["label_mapping"]
    sweeps_num = dataset_config['sweeps_num']

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    train_dataset = Occ_Point_NuScenes(data_path, occ_path=occ_path, imageset=train_imageset,
                                     label_mapping=label_mapping, nusc=nusc, sweeps_num=sweeps_num)
    val_dataset = Occ_Point_NuScenes(data_path, occ_path=occ_path, imageset=val_imageset,
                                   label_mapping=label_mapping, nusc=nusc, sweeps_num=sweeps_num)

    if dataset_config['dataset_type'] == 'Occ_DatasetWrapper_Point_NuScenes':
        train_dataset = Occ_DatasetWrapper_Point_NuScenes(
            train_dataset,
            grid_size=grid_size,
            grid_size_vox=dataset_config['grid_size_vox'],
            grid_size_occ=dataset_config['grid_size_occ'],
            coarse_ratio=dataset_config['coarse_ratio'],
            pc_range=dataset_config['pc_range'],
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            fill_label=dataset_config["fill_label"],
            rotate_aug=dataset_config['rotate_aug'],
            flip_aug=dataset_config['flip_aug'],
            scale_aug=dataset_config['scale_aug'],
            transform_aug=dataset_config['transform_aug'],
            trans_std=dataset_config['trans_std'],
        )

        val_dataset = Occ_DatasetWrapper_Point_NuScenes(
            val_dataset,
            grid_size=grid_size,
            grid_size_vox=dataset_config['grid_size_vox'],
            grid_size_occ=dataset_config['grid_size_occ'],
            coarse_ratio=dataset_config['coarse_ratio'],
            pc_range=dataset_config['pc_range'],
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            fill_label=dataset_config["fill_label"],
        )
        
        collate_fn = occ_custom_collate_fn
    else:
        raise Exception('invalid dataset_type')

    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        sampler = None
        val_sampler = None

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=collate_fn,
                                                       shuffle=False if dist else train_dataloader_config["shuffle"],
                                                       sampler=sampler,
                                                       num_workers=train_dataloader_config["num_workers"])
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn,
                                                     shuffle=False if dist else val_dataloader_config["shuffle"],
                                                     sampler=val_sampler,
                                                     num_workers=val_dataloader_config["num_workers"])

    return train_dataset_loader, val_dataset_loader