from __future__ import print_function

import numpy as np
import os.path
import time
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from tracking.tracker import Tracker
from utils.data_util import generate_seq_dets
from utils.point_util import read_and_prep_points
from kitti_evaluate import evaluate

TRAIN_SEQ_ID = ['0003', '0001', '0013', '0009', '0004',
                '0020', '0006', '0015', '0008', '0012']
VALID_SEQ_ID = ['0005', '0007', '0017', '0011', '0002',
                '0014', '0000', '0010', '0016', '0019', '0018']
TEST_SEQ_ID = [f'{i:04d}' for i in range(29)]
# Valid sequence 0017 has no cars in detection,
# so it should not be included if val with GT detection
# VALID_SEQ_ID = ['0005', '0007', '0011', '0002', '0014', \
#                 '0000', '0010', '0016', '0019', '0018']
TRAINVAL_SEQ_ID = [f'{i:04d}' for i in range(21)]


class TrackingDataset(object):
    def __init__(self, root_dir, link_file, det_file, fix_iou=0.2, fix_count=2,
                 transform=None, num_point_features=4, modality='Car'):
        self.root_dir = root_dir
        self.modality = modality
        self.num_point_features = num_point_features
        self.test = False

        if "trainval" in link_file:
            self.seq_ids = TRAINVAL_SEQ_ID
        elif "train" in link_file:
            self.seq_ids = TRAIN_SEQ_ID
        elif "val" in link_file:
            self.seq_ids = VALID_SEQ_ID
        elif 'test' in link_file:
            self.test = True
            self.seq_ids = TEST_SEQ_ID

        # {sequence: {frame: {} } } }
        self.sequence_det = generate_seq_dets(root_dir, link_file, det_file, self.seq_ids,
                                              iou_threshold=fix_iou, fix_threshold=fix_count,
                                              allow_empty=True, test=self.test)

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        self.get_pointcloud = partial(read_and_prep_points, root_path=root_dir,
                                      num_point_features=num_point_features)

        self.meta = self._generate_meta_seq()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        return self.meta[idx]

    def _generate_meta_seq(self):
        meta = []
        for seq_id in self.seq_ids:
            if seq_id == '0007':
                els = list(self.sequence_det[seq_id].items())
                seq_length = int(els[-1][0])
            else:
                seq_length = len(self.sequence_det[seq_id])
            det_frames = []
            for i in range(0, seq_length):
                frame_id = f'{i:06d}'
                # Get first frame, skip the empty frame
                if frame_id in self.sequence_det[seq_id] and \
                        len(self.sequence_det[seq_id][frame_id]['detection']['name']) > 0:
                    det_frames.append(self.sequence_det[seq_id][frame_id])
                else:
                    continue

            meta.append(SequenceDataset(name=seq_id, modality=self.modality,
                                        root_dir=self.root_dir, det_frames=det_frames, transform=self.transform,
                                        get_pointcloud=self.get_pointcloud))
        return meta


class SequenceDataset(Dataset):

    def __init__(self, name, modality, root_dir, det_frames,
                 transform, get_pointcloud):
        self.root_dir = root_dir
        self.metas = det_frames
        self.idx = 0
        self.seq_len = len(det_frames)
        self.name = name
        self.modality = modality
        self.get_pointcloud = get_pointcloud

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

    def __getitem__(self, idx):
        return self._generate_boxes_img_points(idx)

    def __len__(self):
        return len(self.metas)

    def _generate_boxes_img_points(self, idx):
        frame = self.metas[idx]
        det_imgs = []
        ###
        path = f"{self.root_dir}/image_02/{frame['image_path']}"
        img = Image.open(path)
        det_num = frame['detection']['bbox'].shape[0]
        frame['frame_info']['img_shape'] = np.array([img.size[1], img.size[0]])  # w, h -> h, w
        point_cloud = self.get_pointcloud(info=frame['frame_info'], point_path=frame['point_path'],
                                          dets=frame['detection'])
        for i in range(det_num):
            x1 = np.floor(frame['detection']['bbox'][i][0])
            y1 = np.floor(frame['detection']['bbox'][i][1])
            x2 = np.ceil(frame['detection']['bbox'][i][2])
            y2 = np.ceil(frame['detection']['bbox'][i][3])
            det_imgs.append(
                self.transform(img.crop((x1, y1, x2, y2)).resize((224, 224), Image.BILINEAR)).unsqueeze(0))

        if 'image_idx' in frame['detection'].keys():
            frame['detection'].pop('image_idx')

        det_imgs = torch.cat(det_imgs, dim=0)
        det_points = torch.Tensor(point_cloud['points'])
        det_boxes = point_cloud['boxes']
        det_points_split = point_cloud['det_lens']
        ###
        return det_boxes, det_imgs, det_points, det_points_split, frame['detection']


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
    result_dir = os.path.join(result_root, result_sha, part)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    val_root = "/home/kemo/Kitti/tracking/training"
    val_link = "./data/val.txt"
    val_det = "./data/pp_val_dets.pkl"
    ckpt = "pretrained_models/pp_pv_40e_dualadd_subabs_C.pth"

    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = TrackingDataset(
        root_dir=val_root,
        link_file=val_link,
        det_file=val_det,
        transform=valid_transform,
        fix_iou=1,
        fix_count=0)

    EMOT(t_miss=4, t_hit=1, w_app=0, w_iou=1, w_loc=0)
    evaluate(result_sha=result_sha, result_root=result_root, part=part, gt_path=val_root)
