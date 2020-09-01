import os
import tqdm
import glob
import datetime
from functools import partial
from pathlib import Path
import torch
from torch import nn
from torch import optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter
from dataset.id_dataset import IdTrainingDataset
from appearance_model.tracking_net import TrackingNet
from config import cfg
from utils.fastai_optim import OptimWrapper
from utils.learning_schedules_fastai import CosineWarmupLR
from utils import common_utils

train_root = os.path.join('data', 'score_thres_0.3')
train_feature_pkl = os.path.join(train_root, 'car_features.pkl')
train_tid_pkl = os.path.join(train_root, 'car_tid_0.5.pkl')

val_root = os.path.join('data', 'score_thres_0.3_val')
val_feature_pkl = os.path.join(val_root, 'car_features.pkl')
val_tid_pkl = os.path.join(val_root, 'car_tid_0.5.pkl')

output_dir = Path('./output/train').resolve()
ckpt_dir = output_dir / 'ckpt'
output_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

output_root = "output"
result_sha = "mot_data"
part = 'val'
result_dir = os.path.join(output_root, result_sha, part)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
gt_path = "/media/kemo/Kemo/Kitti/tracking/training"


def main():
    total_epochs = cfg.OPTIMIZATION.NUM_EPOCHS
    batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    validate_interval = 1
    max_ckpt_save_num = 100

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    train_set = IdTrainingDataset(train_feature_pkl, train_tid_pkl)
    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True,
                              shuffle=True, collate_fn=train_set.collate_fn, drop_last=True)

    test_set = IdTrainingDataset(val_feature_pkl, val_tid_pkl)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=True,
                             shuffle=False, collate_fn=test_set.collate_fn)

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    model = TrackingNet(model_cfg=cfg, cls_in_channels=27648, aff_in_channels=27648 * 2)
    model.cuda()

    logger.info(model)

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    start_epoch = it = 0
    last_epoch = -1

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader),
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # validation set

    accumulated_iter = it
    min_test_loss = 100000
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=True) as train_bar:
        total_it_each_epoch = len(train_loader)
        dataloader_iter = iter(train_loader)
        for cur_epoch in train_bar:
            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=cfg.OPTIMIZATION,
                tbar=train_bar, tb_log=tb_log,
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            # validate
            if cur_epoch % validate_interval == 0:
                model.eval()
                test_loss = 0
                total_loss_dict = {}
                with tqdm.trange(0, len(test_loader), desc='val', dynamic_ncols=True, leave=True) as test_bar:
                    test_loader_iter = iter(test_loader)
                    for it in test_bar:
                        batch_dict = next(test_loader_iter)
                        load_data_to_gpu(batch_dict)
                        with torch.no_grad():
                            loss, tb_dict = model(batch_dict)
                        test_bar.set_postfix({
                            'loss': loss.item()
                        })
                        test_bar.refresh()
                        test_loss += loss.item()
                        for key, val in tb_dict.items():
                            if key not in total_loss_dict:
                                total_loss_dict[key] = val
                            else:
                                total_loss_dict[key] += val
                test_loss = test_loss / len(test_loader)
                tb_log.add_scalar('test/loss', test_loss, cur_epoch)
                for key, val in total_loss_dict.items():
                    tb_log.add_scalar('test/' + key, val / len(test_loader), cur_epoch)
                trained_epoch = cur_epoch + 1
                if test_loss < min_test_loss:
                    min_test_loss = test_loss
                    # save trained model
                    ckpt_list = glob.glob(str(ckpt_dir / 'checkpoint_epoch_*.pth'))
                    ckpt_list.sort(key=os.path.getmtime)

                    if ckpt_list.__len__() >= max_ckpt_save_num:
                        for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                            os.remove(ckpt_list[cur_file_idx])

                    ckpt_name = ckpt_dir / ('checkpoint_epoch_%d' % trained_epoch)
                    save_checkpoint(
                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                    )


def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError
    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]

    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

    if optim_cfg.LR_WARMUP:
        lr_warmup_scheduler = CosineWarmupLR(
            optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
            eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
        )

    return lr_scheduler, lr_warmup_scheduler


def train_one_epoch(model, optimizer, train_loader, lr_scheduler, accumulated_iter, optim_cfg,
                    tbar, total_it_each_epoch, dataloader_iter, tb_log=None):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        model.train()
        optimizer.zero_grad()

        load_data_to_gpu(batch)
        loss, tb_dict = model(batch)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        lr_scheduler.step()

        accumulated_iter += 1
        disp_dict = {'loss': loss.item(), 'lr': cur_lr, 'total_it': accumulated_iter}

        # log to console and tensorboard
        tbar.set_postfix(disp_dict)
        tbar.refresh()

        if tb_log is not None:
            tb_log.add_scalar('train/loss', loss.item(), accumulated_iter)
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
            for key, val in tb_dict.items():
                tb_log.add_scalar('train/' + key, val, accumulated_iter)

    return accumulated_iter


def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            batch_dict[key] = val.float().cuda()


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}


def write_kitti_format(results, frame_id, out_file):
    frame_id = frame_id.lstrip('0')
    if len(frame_id) == 0:
        frame_id = '0'
    for tid, info, score in results:
        name = info['name']
        truncated = info['truncated']
        occluded = info['occluded']
        x, y, z = info['location']
        dx, dy, dz = info['dimensions']
        ry = info['rotation_y']
        alpha = info['alpha']
        x1, y1, x2, y2 = info['bbox']
        out_file.write(f"{frame_id} {tid} {name} {truncated} {occluded} {alpha} "
                       f"{x1} {y1} {x2} {y2} "
                       f"{dy} {dx} {dz} {x} {y} {z} "
                       f"{ry} "
                       f"{score}\n")


if __name__ == '__main__':
    main()
