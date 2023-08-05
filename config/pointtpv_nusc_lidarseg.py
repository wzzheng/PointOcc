_base_ = [
    './_base_/optimizer.py',
    './_base_/schedule.py',
]

load_from = None

lovasz_input = 'points'
ce_input = 'voxel'

optimizer_wrapper = dict(
    optimizer = dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
)

cumulative_iters = 1
find_unused_parameters = False
unique_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
track_running_stats = False

_dim_ = 192

tpv_w_ = 240
tpv_h_ = 180
tpv_z_ = 16
scale_w = 2
scale_h = 2
scale_z = 2

grid_size = [480, 360, 32]
nbr_class = 17

dataset_params = dict(
    version = "v1.0-trainval",
    dataset_type = 'Seg_DatasetWrapper_Point_NuScenes',
    grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
    fill_label = 0,
    ignore_label = 0,
    fixed_volume_space = True,
    label_mapping = "./config/label_mapping/nuscenes.yaml",
    max_volume_space = [50, 3.1415926, 3],
    min_volume_space = [0, -3.1415926, -5],
    rotate_aug = True,
    flip_aug = True,
    scale_aug = True,
    transform_aug = True,
    trans_std=[0.1, 0.1, 0.1],
)

train_data_loader = dict(
    data_path = "data/nuscenes/",
    imageset = "./data/nuscenes/nuscenes_infos_train.pkl",
    batch_size = 1,
    shuffle = True,
    num_workers = 1,
)

val_data_loader = dict(
    data_path = "data/nuscenes/",
    imageset = "./data/nuscenes/nuscenes_infos_val.pkl",
    batch_size = 1,
    shuffle = False,
    num_workers = 1,
)


model = dict(
    type='PointTPV_Seg',
    tpv_aggregator=dict(
        type='TPVAggregator_Seg',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        nbr_classes=nbr_class,
        in_dims=_dim_,
        hidden_dims=2*_dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z
    ),
    lidar_tokenizer=dict(
        type='CylinderEncoder_Seg',
        grid_size=grid_size,
        in_channels=10,
        out_channels=256,
        fea_compre=None,
        base_channels=256,
        split=[16,16,16],
        track_running_stats=track_running_stats,
    ),
    lidar_backbone=dict(
        type='Swin',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        in_channels=256,
        patch_size=4,
        strides=[1,2,2,2],
        frozen_stages=-1,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1,2,3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
          type='Pretrained',
          checkpoint='pretrain/swin_tiny_patch4_window7_224.pth'),
    ),
    lidar_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=_dim_,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(
          type='BN2d',
          requires_grad=True,
          track_running_stats=track_running_stats),
        act_cfg=dict(
          type='ReLU',
          inplace=True),
        upsample_cfg=dict(
          mode='bilinear',
          align_corners=False),
    ),
)
