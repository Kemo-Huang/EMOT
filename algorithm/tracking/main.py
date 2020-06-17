import os.path
import time
import torch
from torch.utils.data import DataLoader
from tracking.tracker import Tracker


def EMOT(t_miss=2, t_hit=2, w_app=0.25, w_iou=0.35, w_loc=0.4):
    tracking_module = Tracker(ckpt, t_miss=t_miss, t_hit=t_hit, w_app=w_app, w_iou=w_iou, w_loc=w_loc)
    total_frames = 0
    total_time = 0
    for i, sequence in enumerate(val_dataset):
        print('Test: [{}/{}]\tSequence ID: KITTI-{}'.format(
            i, len(val_dataset), sequence.name))
        seq_loader = DataLoader(
            sequence,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)
        if len(seq_loader) == 0:
            tracking_module.reset()
            print('Empty Sequence ID: KITTI-{}, skip'.format(
                sequence.name))
        else:
            runtime_seq = 0
            num_frames = 0
            # switch to evaluate mode
            tracking_module.reset()
            with open(os.path.join(result_dir, f'{sequence.name}.txt'), 'w+') as out:
                with torch.no_grad():
                    for (det_boxes, det_images, det_points, det_points_split, detections) in seq_loader:
                        det_boxes = det_boxes.squeeze(0).numpy()
                        for box in det_boxes:
                            # input: x, y, z, l, h, w, yaw
                            # to: x, y, z, yaw, l, w, h
                            box[:] = box[[0, 1, 2, 6, 3, 5, 4]]
                        det_images = det_images.cuda().squeeze(0)
                        det_points = det_points.cuda().squeeze(0)
                        for k, v in detections.items():
                            detections[k] = v[0]
                        # compute output
                        end = time.time()
                        results = tracking_module.update(
                            det_boxes, det_images, det_points, det_points_split, detections)
                        runtime_seq += (time.time() - end)
                        num_frames += 1
                        if num_frames % 100 == 0:
                            print(f'Test Frame: [{num_frames}/{len(seq_loader)}]. Time: {runtime_seq}')

                        # write kitti format
                        for (tid, box, info, score) in results:
                            frame_id = detections['frame_idx'].lstrip('0')
                            if len(frame_id) == 0:
                                frame_id = '0'
                            x, y, z, yaw, l, w, h = box.flatten()
                            x1, y1, x2, y2 = info['bbox']
                            out.write(f"{frame_id} {tid} Car 0 0 {info['alpha']} "
                                      f"{x1} {y1} {x2} {y2} {h} {w} {l} {x} {y} {z} {yaw} {score}\n")
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

    dataset_name = 'kitti'

    ckpt = "pretrained_models/pp_pv_40e_dualadd_subabs_C.pth"

    if dataset_name == 'kitti':
        val_root = "/home/kemo/Kitti/tracking/training"
        from dataset.kitti.dataset import TrackingDataset
        from dataset.kitti.evaluate import evaluate

        val_dataset = TrackingDataset(root_dir=val_root)

        # EMOT(t_miss=2, t_hit=1, w_app=1, w_iou=0, w_loc=0)
        # print("EMOT(t_miss=2, t_hit=1, w_app=1, w_iou=0, w_loc=0)")
        # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
        #
        # EMOT(t_miss=2, t_hit=1, w_app=0, w_iou=1, w_loc=0)
        # print("EMOT(t_miss=2, t_hit=1, w_app=0, w_iou=1, w_loc=0)")
        # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
        #
        # EMOT(t_miss=2, t_hit=1, w_app=0, w_iou=0, w_loc=1)
        # print("EMOT(t_miss=2, t_hit=1, w_app=0, w_iou=0, w_loc=1)")
        # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
        #
        # EMOT(t_miss=2, t_hit=1, w_app=0.5, w_iou=0.5, w_loc=0)
        # print("EMOT(t_miss=2, t_hit=1, w_app=0.5, w_iou=0.5, w_loc=0)")
        # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
        #
        # EMOT(t_miss=2, t_hit=1, w_app=0.5, w_iou=0, w_loc=0.5)
        # print("EMOT(t_miss=2, t_hit=1, w_app=0.5, w_iou=0, w_loc=0.5)")
        # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
        #
        # EMOT(t_miss=2, t_hit=1, w_app=0, w_iou=0.5, w_loc=0.5)
        # print("EMOT(t_miss=2, t_hit=1, w_app=0, w_iou=0.5, w_loc=0.5)")
        # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
        #
        # EMOT(t_miss=2, t_hit=1, w_app=0.3, w_iou=0.35, w_loc=0.35)
        # print("EMOT(t_miss=2, t_hit=1, w_app=0.3, w_iou=0.35, w_loc=0.35)")
        # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
        #
        # EMOT(t_miss=2, t_hit=0, w_app=0, w_iou=1, w_loc=0)
        # print("EMOT(t_miss=2, t_hit=0, w_app=0, w_iou=1, w_loc=0)")
        # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
        #
        # EMOT(t_miss=2, t_hit=2, w_app=0, w_iou=1, w_loc=0)
        # print("EMOT(t_miss=2, t_hit=2, w_app=0, w_iou=1, w_loc=0)")
        # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
        #
        # EMOT(t_miss=3, t_hit=1, w_app=0, w_iou=1, w_loc=0)
        # print("EMOT(t_miss=3, t_hit=1, w_app=0, w_iou=1, w_loc=0)")
        # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)

        EMOT(t_miss=4, t_hit=1, w_app=0, w_iou=1, w_loc=0)
        print("EMOT(t_miss=4, t_hit=1, w_app=0, w_iou=1, w_loc=0)")
        evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)

        # EMOT(t_miss=5, t_hit=1, w_app=0, w_iou=1, w_loc=0)
        # print("EMOT(t_miss=5, t_hit=1, w_app=0, w_iou=1, w_loc=0)")
        # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
