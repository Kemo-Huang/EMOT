import numpy as np
from functools import partial
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.sustech_util import generate_seq_gts
from utils.point_util import read_and_prep_points


class TrackingDataset(object):
    def __init__(self, root_dir, modality='Car'):
        self.root_dir = root_dir
        self.modality = modality

        # {sequence: {frame: {} } } }
        self.sequence_det = None

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.get_pointcloud = partial(read_and_prep_points, root_path=root_dir)

        self.meta = self._generate_meta_seq()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        return self.meta[idx]

    def _generate_meta_seq(self):
        pass


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
