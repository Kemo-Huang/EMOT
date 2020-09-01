import os
import tqdm
from pathlib import Path
import torch
import time
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset.id_dataset import IdTrainingDataset
from appearance_model.tracking_net import TrackingNet
from config import cfg

data_root = os.path.join('data', 'score_thres_0.3_val')
feature_pkl = os.path.join(data_root, 'features.pkl')
tid_txt = os.path.join(data_root, 'rcnn_tid_0.5.txt')

output_dir = Path('output').resolve()
output_dir.mkdir(parents=True, exist_ok=True)
ckpt = 'checkpoint_epoch_163.pth'


def main():
    batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    test_set = IdTrainingDataset(feature_pkl, tid_txt)
    test_loader = DataLoader(test_set, batch_size=batch_size, pin_memory=True,
                             shuffle=False, collate_fn=test_set.collate_fn)

    model = TrackingNet(model_cfg=cfg, cls_in_channels=27648, aff_in_channels=27648 * 2)

    model_state_dict = torch.load(ckpt)['model_state']
    model.load_state_dict(model_state_dict)
    model.train()
    model.cuda()

    total_time = 0
    tbar = tqdm.tqdm(total=len(test_loader), desc='eval', dynamic_ncols=True, leave=True)
    for it, batch_dict in enumerate(test_loader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            start_time = time.time()
            loss, tb_dict = model(batch_dict)
            total_time += time.time() - start_time
        tbar.set_postfix({
            'loss': loss.item()
        })
        tbar.update()
        tb_log.add_scalar('test/loss', loss.item(), it)
        for key, val in tb_dict.items():
            tb_log.add_scalar('test/' + key, val, it)
    tbar.close()
    print(total_time)


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            batch_dict[key] = val.float().cuda()


if __name__ == '__main__':
    main()
