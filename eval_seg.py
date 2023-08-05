
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from utils.metric_util import MeanIoU
from utils.load_save_util import revise_ckpt
from dataloader.dataset import get_nuScenes_label_name
from builder import loss_builder

from mmengine import Config
from mmengine.logging.logger import MMLogger

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader

    grid_size = cfg.grid_size

    # init DDP
    if args.launcher == 'none':
        distributed = False
        rank = 0
        cfg.gpu_ids = [0]         # debug
    else:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20506")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank
        )
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if dist.get_rank() != 0:
            import builtins
            builtins.print = pass_print


    logger = MMLogger(name='eval_log', log_file=args.log_file, log_level='INFO')

    # build model
    from builder import model_builder
    
    my_model = model_builder.build(cfg.model)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    logger.info(f'Model:\n{my_model}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model.cuda()
    print('done ddp model')

    # generate datasets
    SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label]

    from builder import data_builder
    train_dataset_loader, val_dataset_loader = \
        data_builder.build_seg(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=distributed,
        )

    CalMeanIou_vox = MeanIoU(unique_label, ignore_label, unique_label_str, 'vox')
    CalMeanIou_pts = MeanIoU(unique_label, ignore_label, unique_label_str, 'pts')
    
    # resume and load
    assert osp.isfile(args.ckpt_path)
    print('ckpt path:', args.ckpt_path)
    
    map_location = 'cpu'
    ckpt = torch.load(args.ckpt_path, map_location=map_location)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    print(my_model.load_state_dict(revise_ckpt(ckpt), strict=False))
    print(f'successfully loaded ckpt')
    
    print_freq = cfg.print_freq
    
    # eval
    my_model.eval()
    CalMeanIou_pts.reset()
    CalMeanIou_vox.reset()

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            (points, val_vox_label, val_grid, val_pt_labs, val_grid_vox) = data
            val_grid_int = torch.floor(val_grid_vox).to(torch.long).cuda()
            val_grid_vox = val_grid_vox.to(torch.float32).cuda()
            points = points.cuda()
            val_grid_float = val_grid.to(torch.float32).cuda()
            vox_label = val_vox_label.to(torch.long).cuda()
            val_pt_labs = val_pt_labs.to(torch.long).cuda()
            predict_labels_vox, predict_labels_pts = my_model(points=points, grid_ind=val_grid_float, grid_ind_vox=val_grid_vox)
            
            predict_labels_pts = predict_labels_pts.squeeze(-1).squeeze(-1)
            predict_labels_pts = torch.argmax(predict_labels_pts, dim=1) # bs, n
            predict_labels_pts = predict_labels_pts.detach().cpu()
            val_pt_labs = val_pt_labs.squeeze(-1).cpu()
            
            predict_labels_vox = torch.argmax(predict_labels_vox, dim=1)
            predict_labels_vox = predict_labels_vox.detach().cpu()
            for count in range(len(val_grid_int)):
                CalMeanIou_pts._after_step(predict_labels_pts[count], val_pt_labs[count])
                CalMeanIou_vox._after_step(
                    predict_labels_vox[
                    count, 
                    val_grid_int[count][:, 0], 
                    val_grid_int[count][:, 1], 
                    val_grid_int[count][:, 2]].flatten(),
                    val_pt_labs[count])
            
            if i_iter_val % print_freq == 0 and dist.get_rank() == 0:
                logger.info('[EVAL] Iter %5d: Loss: None'%(i_iter_val))
        
        val_miou_pts = CalMeanIou_pts._after_epoch()
        val_miou_vox = CalMeanIou_vox._after_epoch()
        logger.info('val miou pts is %.3f' % (val_miou_pts))
        logger.info('val miou vox is %.3f' % (val_miou_vox))
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch')
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--log-file', type=str, default=None)

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)
    
    if args.launcher == 'none':
        main(0, args)
    else:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
