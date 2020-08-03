import os.path
import time
import torch
from torch.utils.data import DataLoader
from utils.sustech_util import write_results


def sustech_mot():
    total_frames = 0
    total_time = 0
    sequence = val_dataset
    seq_loader = DataLoader(
        sequence,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True)
    runtime_seq = 0
    num_frames = 0
    # switch to evaluate mode
    tracking_module.reset()
    with open(os.path.join(result_dir, f'{sequence.name}.txt'), 'w+') as out:
        with torch.no_grad():
            for (det_images, det_points, det_points_split, det_info) in seq_loader:
                det_images = det_images.cuda().squeeze(0)
                det_points = det_points.cuda().squeeze(0)
                for k, v in det_info.items():
                    det_info[k] = v[0]
                # compute output
                end = time.time()
                results = tracking_module.update(
                    det_images, det_points, det_points_split, det_info)
                runtime_seq += (time.time() - end)
                num_frames += 1
                if num_frames % 100 == 0:
                    print(f'Test Frame: [{num_frames}/{len(seq_loader)}]. Time: {runtime_seq}')

                write_results(results, det_info, out)

    print(f"FPS: {num_frames / runtime_seq}")
    total_frames += num_frames
    total_time += runtime_seq
    print(f"total FPS: {total_frames / total_time}")


if __name__ == '__main__':
    result_root = "results"
    result_sha = "data"
    part = 'val'
    # part = 'train'
    result_dir = os.path.join(result_root, result_sha, part)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    ckpt = "pretrained_models/pp_pv_40e_dualadd_subabs_C.pth"

    from dataset.sustech.dataset import TrackingDataset
    from tracking.sustech_tracker import Tracker

    val_root = "/home/kemo/Dataset/sustechscapes-mini-dataset"
    val_dataset = TrackingDataset(root_dir=val_root)

    tracking_module = Tracker(ckpt, t_miss=4, t_hit=0, w_app=0, w_iou=1, w_loc=0)
    sustech_mot()
    # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
