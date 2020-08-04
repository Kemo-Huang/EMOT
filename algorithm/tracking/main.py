import os.path
import time
import torch
from torch.utils.data import DataLoader
from utils.kitti_util import write_kitti_format


def kitti_mot():
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
                    for (det_images, det_points, det_points_split, det_boxes, frame_id) in seq_loader:
                        det_boxes['boxes_2d'] = det_boxes['boxes_2d'][0]
                        boxes_3d = det_boxes['boxes_3d'][0]
                        for box in boxes_3d:
                            # input: x, y, z, l, h, w, yaw
                            # to: x, y, z, yaw, l, w, h
                            box[:] = box[[0, 1, 2, 6, 3, 5, 4]]
                        det_boxes['boxes_3d'] = boxes_3d
                        frame_id = frame_id[0]
                        det_images = det_images.cuda().squeeze(0)
                        det_points = det_points.cuda().squeeze(0)
                        # compute output
                        end = time.time()
                        results = tracking_module.update(
                            det_images, det_points, det_points_split, det_boxes, frame_id)
                        runtime_seq += (time.time() - end)
                        num_frames += 1
                        if num_frames % 100 == 0:
                            print(f'Test Frame: [{num_frames}/{len(seq_loader)}]. Time: {runtime_seq}')

                        write_kitti_format(results, frame_id, out)

            print(f"FPS: {num_frames / runtime_seq}")
            total_frames += num_frames
            total_time += runtime_seq
    print(f"total FPS: {total_frames / total_time}")


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
            for (det_boxes, det_images, det_points, det_points_split, detections) in seq_loader:
                det_boxes = det_boxes.squeeze(0).numpy()
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

                write_kitti_format(results, detections, out)

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
        from tracking.kitti_tracker import Tracker
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

        tracking_module = Tracker(ckpt, t_miss=4, t_hit=1, w_app=0, w_iou=1, w_loc=0)
        kitti_mot()
        print("EMOT(t_miss=4, t_hit=1, w_app=0, w_iou=1, w_loc=0)")
        evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)

        # EMOT(t_miss=5, t_hit=1, w_app=0, w_iou=1, w_loc=0)
        # print("EMOT(t_miss=5, t_hit=1, w_app=0, w_iou=1, w_loc=0)")
        # evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
    else:
        from dataset.sustech.dataset import TrackingDataset
        from tracking.sustech_tracker import Tracker
        from dataset.sustech.evaluate import evaluate

        val_root = "/home/kemo/Kitti/tracking/training"
        val_dataset = TrackingDataset(root_dir=val_root)

        tracking_module = Tracker(ckpt, t_miss=4, t_hit=1, w_app=0, w_iou=1, w_loc=0)
        sustech_mot()
        evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
