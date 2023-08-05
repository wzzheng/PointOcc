
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from utils.metric_util import MeanIoU
from utils.load_save_util import revise_ckpt, revise_ckpt_2
from dataloader.dataset import get_nuScenes_label_name
from builder import loss_builder

from mmengine import Config
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader

    max_num_epochs = cfg.max_epochs
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

    # configure logger
    if local_rank == 0 and rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger(name='train_log', log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

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
        data_builder.build_occ(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=distributed,
        )


    # get optimizer, loss, scheduler
    optimizer = build_optim_wrapper(my_model, cfg.optimizer_wrapper)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader)*max_num_epochs,
        lr_min=1e-6,
        warmup_t=500,
        warmup_lr_init=1e-5,
        t_in_epochs=False
    )
    
    CalMeanIou_sem = MeanIoU(unique_label, ignore_label, unique_label_str, 'semantic')
    CalMeanIou_geo = MeanIoU([1], ignore_label=255, label_str=['occupancy'], name='geometry')
    
    # resume and load
    epoch = 0
    best_val_miou_pts, best_val_miou_vox = 0, 0
    global_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    print('resume from: ', cfg.resume_from)
    print('work dir: ', args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_miou_pts' in ckpt:
            best_val_miou_pts = ckpt['best_val_miou_pts']
        if 'best_val_miou_vox' in ckpt:
            best_val_miou_vox = ckpt['best_val_miou_vox']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))
        

    # training
    print_freq = cfg.print_freq
    cumulative_iters = cfg.get('cumulative_iters', 1)

    while epoch < max_num_epochs:
        my_model.train()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        # for cumulative_iters > 1
        if cumulative_iters > 1:
            total_iters = len(train_dataset_loader)
            divisible_iters = total_iters // cumulative_iters * cumulative_iters
            remainder_iters = total_iters - divisible_iters
            logger.info(f'cumulative_iters: {cumulative_iters}, total_iters: {total_iters}, \
                        divisible_iters: {divisible_iters}, remainder_iters: {remainder_iters}')
        loss_list = []
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, data in enumerate(train_dataset_loader):
            (voxel_position_coarse, points, train_vox_label, train_grid) = data
            points = points.cuda()
            train_grid = train_grid.to(torch.float32).cuda()
            train_grid_vox_coarse = voxel_position_coarse.to(torch.float32).cuda()
            voxel_label = train_vox_label.type(torch.LongTensor).cuda()
            # forward + backward + optimize
            data_time_e = time.time()
            with torch.cuda.amp.autocast():
                loss = my_model(points=points, grid_ind=train_grid, grid_ind_vox=None, 
                                grid_ind_vox_coarse=train_grid_vox_coarse, voxel_label=voxel_label, return_loss=True)
            if cumulative_iters > 1:
                loss_factor = cumulative_iters if i_iter < divisible_iters else remainder_iters
                loss_list.append(loss.item())
                loss = loss / loss_factor
                # loss.backward()
                scaler.scale(loss).backward()
                if (i_iter+1) % cumulative_iters == 0 or i_iter + 1 == len(train_dataset_loader):
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # loss.backward()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                loss_list.append(loss.item())
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and dist.get_rank() == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), lr: %.7f, time: %.3f (%.3f)'%(
                    epoch+1, i_iter, len(train_dataset_loader), 
                    loss_list[-1], np.mean(loss_list), lr,
                    time_e - time_s, data_time_e - data_time_s
                ))
                loss_list = []
            data_time_s = time.time()
            time_s = time.time()
        
        # save checkpoint
        if dist.get_rank() == 0:
            dict_to_save = {
                'state_dict': my_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
                'best_val_miou_pts': best_val_miou_pts,
                'best_val_miou_vox': best_val_miou_vox
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            symlink(save_file_name, dst_file)

        epoch += 1
        
        # eval
        my_model.eval()
        val_loss_list = []
        CalMeanIou_sem.reset()
        CalMeanIou_geo.reset()

        with torch.no_grad():
            for i_iter_val, data in enumerate(val_dataset_loader):
                (voxel_position_coarse, points, val_vox_label, val_grid) = data
                points = points.cuda()
                val_grid = val_grid.to(torch.float32).cuda()
                val_grid_vox_coarse = voxel_position_coarse.to(torch.float32).cuda()
                voxel_label = val_vox_label.type(torch.LongTensor).cpu()
                
                predict_labels_vox = my_model(points=points, grid_ind=val_grid, grid_ind_vox=None, 
                                                grid_ind_vox_coarse=val_grid_vox_coarse, voxel_label=voxel_label, return_loss=False)
                
                predict_labels_vox = torch.argmax(predict_labels_vox, dim=1).detach().cpu()
                CalMeanIou_sem._after_step(predict_labels_vox.flatten(), voxel_label.flatten())
                occ_gt_mask = (voxel_label != 0) & (voxel_label != 255)
                voxel_label[occ_gt_mask] = 1
                occ_pred_mask = (predict_labels_vox != 0)
                predict_labels_vox[occ_pred_mask] = 1
                CalMeanIou_geo._after_step(predict_labels_vox.flatten(), voxel_label.flatten())
                
                if i_iter_val % print_freq == 0 and dist.get_rank() == 0:
                    logger.info('[EVAL] Iter %5d: Loss: None'%(i_iter_val))
            
        val_miou_sem = CalMeanIou_sem._after_epoch()
        val_miou_geo = CalMeanIou_geo._after_epoch()
        logger.info('Current val miou is %.3f' % (val_miou_sem))
        logger.info('Current val iou is %.3f' % (val_miou_geo))
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./work_dir/tpv_lidarseg')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch')
    parser.add_argument('--resume-from', type=str, default='')

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)
    
    if args.launcher == 'none':
        main(0, args)
    else:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
