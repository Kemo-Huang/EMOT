import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.sustech_util import read_gt_seq
from utils.point_util import crop_points_psr
import open3d


class TrackingDataset(Dataset):
    def __init__(self, root_dir, modality='Car', name='mini'):
        self.modality = modality
        self.name = name
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.frames = read_gt_seq(root_dir, modality)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        """
        :return:
            det_boxes - 3D boxes for 3D Kalman filtering and 3D DIOU association
            det_imgs - 2D cropped images
            det_points - 3D cropped point cloud
        """
        frame = self.frames[idx]
        boxes_3d = frame['boxes_3d']
        boxes_2d = frame['boxes_2d']

        img = Image.open(frame['image_path'])
        pcd = open3d.io.read_point_cloud(frame['point_path'])
        points = np.array(pcd.points)
        # open3d.visualization.draw_geometries([pcd])

        det_points = []
        det_points_split = []

        for box_3d in boxes_3d:
            cropped_points = crop_points_psr(points, box_3d)
            if cropped_points.shape[0] > 0:
                det_points.append(torch.Tensor(cropped_points))
                det_points_split.append(len(cropped_points))
            else:
                det_points.append(torch.zeros((1, 3)))
                det_points_split.append(1)
        det_imgs = torch.cat(
            [self.transform(img.crop(box2d).resize((224, 224), Image.BILINEAR)).unsqueeze(0) for box2d in boxes_2d])
        det_points = torch.cat(det_points)
        img.close()

        return det_imgs, det_points, det_points_split, frame
