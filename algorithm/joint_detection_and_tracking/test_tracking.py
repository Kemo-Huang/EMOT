import os
import tqdm
from dataset.tracking_dataset import get_sequence_data
import torch
import time
from datetime import datetime
from appearance_model.tracking_net import TrackingNet
from tracking import kitti_tracker, hungarian_tracker
from config import cfg
from evaluate import evaluate
import logging

data_root = os.path.join('data', 'score_thres_0.3_val')
feature_pkl = os.path.join(data_root, 'features.pkl')
result_pkl = os.path.join(data_root, 'result.pkl')
link_pkl = os.path.join(data_root, 'link_val.pkl')

ckpt = os.path.join('output', 'train', 'ckpt', 'checkpoint_epoch_42.pth')
assert os.path.exists(ckpt)

output_root = "output"
result_sha = "mot_data"
part = 'val'
result_dir = os.path.join(output_root, result_sha, part)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
gt_path = "/media/kemo/Kemo/Kitti/tracking/training"

log_file = os.path.join(output_root, result_sha, f'{datetime.now().strftime("%Y-%m-%d-%S-%M-%H")}.log')


def create_logger(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level=logging.INFO)
    logger.addHandler(ch)

    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    fh.setLevel(level=logging.INFO)
    logger.addHandler(fh)

    return logger


def main():
    logger = create_logger(log_file)
    logger.info('load data...')
    start_time = time.time()
    sequence_data = get_sequence_data(feature_pkl, result_pkl, link_pkl)
    logger.info('done. used time: %f seconds' % (time.time() - start_time))

    model = TrackingNet(model_cfg=cfg, cls_in_channels=27648, aff_in_channels=27648 * 2)
    model_state_dict = torch.load(ckpt)['model_state']
    model.load_state_dict(model_state_dict)
    model.eval()
    model.cuda()

    car_tracker = kitti_tracker.Tracker(model, t_miss=4, t_hit=1, w_app=0, w_iou=1, w_loc=0)

    total_time = 0
    total_frames = 0
    for seq, frames in sequence_data.items():
        out_file = open(os.path.join(result_dir, f'{seq}.txt'), 'w+')
        tbar = tqdm.tqdm(total=len(frames), desc=seq, dynamic_ncols=True, leave=True)
        frame_ids = list(frames.keys())
        frame_ids.sort()
        with torch.no_grad():
            for frame_id in frame_ids:
                all_detections = frames[frame_id]

                # ignore other classes
                car_detections = {'frame_id': frame_id}
                mask = all_detections['name'] == 'Car'
                for k, v in all_detections.items():
                    if k != 'frame_id':
                        car_detections[k] = v[mask]

                start_time = time.time()

                car_results = car_tracker.update(car_detections)

                frame_time = time.time() - start_time
                total_time += frame_time
                total_frames += 1
                tbar.set_postfix({'time': frame_time})
                tbar.update()

                write_kitti_format(car_results, frame_id, out_file)
        out_file.close()
        tbar.close()
    logger.info(
        f'total frames: {total_frames}, total time: {total_time}, frames per second: {total_frames / total_time}')

    evaluate(result_sha=result_sha, result_root=output_root, part=part, gt_path=gt_path, logger=logger)


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
    # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=gt_path)
