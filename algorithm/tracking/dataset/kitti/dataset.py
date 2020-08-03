import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.kitti_util import generate_seq_dets
from utils.point_util import read_and_prep_points
from .data.config import *


class TrackingDataset(object):
    def __init__(self, root_dir, link_file=val_link, det_file=val_det, fix_iou=1, fix_count=0, modality='Car'):
        self.root_dir = root_dir
        self.modality = modality
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
        self.sequence_det = generate_seq_dets(root_dir, link_file, det_file,
                                              iou_threshold=fix_iou, fix_threshold=fix_count,
                                              allow_empty=True)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
            for i in range(seq_length):
                frame_id = f'{i:06d}'
                # Get first frame, skip the empty frame
                if frame_id in self.sequence_det[seq_id] and \
                        len(self.sequence_det[seq_id][frame_id]['detection']['name']) > 0:
                    det_frames.append(self.sequence_det[seq_id][frame_id])
                else:
                    continue

            meta.append(SequenceDataset(name=seq_id, modality=self.modality,
                                        root_dir=self.root_dir, det_frames=det_frames, transform=self.transform,
                                        ))
        return meta


class SequenceDataset(Dataset):

    def __init__(self, name, modality, root_dir, det_frames, transform):
        self.root_dir = root_dir
        self.metas = det_frames
        self.idx = 0
        self.seq_len = len(det_frames)
        self.name = name
        self.modality = modality

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

    def __getitem__(self, idx):
        frame = self.metas[idx]
        det_imgs = []
        ###
        path = f"{self.root_dir}/image_02/{frame['image_path']}"
        img = Image.open(path)
        det_num = frame['detection']['bbox'].shape[0]
        frame['frame_info']['img_shape'] = np.array([img.size[1], img.size[0]])  # w, h -> h, w
        point_cloud = read_and_prep_points(root_path=self.root_dir,
                                           info=frame['frame_info'],
                                           point_path=frame['point_path'],
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

    def __len__(self):
        return len(self.metas)
