import pickle
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict


class IdTrainingDataset(Dataset):
    def __init__(self, feature_pkl: str, tid_pkl: str):
        """
        The dataset used for training mot
        Args:
            feature_pkl (str): file: [tensor (N, 27648), ...]
            tid_pkl(str):
        """
        super(IdTrainingDataset, self).__init__()
        with open(feature_pkl, 'rb') as f:
            self.all_features = pickle.load(f)
        with open(tid_pkl, 'rb') as f:
            self.all_tids = pickle.load(f)

        assert len(self.all_tids) == len(self.all_features), f'{len(self.all_tids)}, {len(self.all_features)}'
        ignore_indices = []
        for i in range(len(self.all_tids)):
            if len(self.all_tids[i]) == 0:
                assert self.all_features[i] is None or self.all_features[i].shape[0] == 0, str(self.all_features[i])
                ignore_indices.append(i)
        self.all_features = [self.all_features[idx] for idx in range(len(self.all_tids)) if idx not in ignore_indices]
        self.all_tids = [self.all_tids[idx] for idx in range(len(self.all_tids)) if idx not in ignore_indices]

    def __len__(self):
        return len(self.all_tids) - 1

    def __getitem__(self, idx):
        prev_feature = self.all_features[idx]
        next_feature = self.all_features[idx + 1]
        n_prev = prev_feature.shape[0]
        n_next = next_feature.shape[0]
        prev_tids = self.all_tids[idx]
        next_tids = self.all_tids[idx + 1]
        assert n_prev == len(prev_tids), f'{n_prev} != {len(prev_tids)}'
        assert n_next == len(next_tids), f'{n_next} != {len(next_tids)}'

        # input features
        cls_feature = torch.cat([prev_feature, next_feature])  # (N + M, D)
        aff_feature = self.batch_cat(prev_feature, next_feature)  # (N * M, 2D)

        # generate ground-truths
        prev_false = prev_tids == -1
        cls_gt_prev = ~prev_false
        next_false = next_tids == -1
        cls_gt_next = ~next_false
        cls_gt = np.concatenate([cls_gt_prev, cls_gt_next])
        cls_gt = torch.from_numpy(cls_gt)

        link_gt = torch.zeros((n_prev, n_next), dtype=torch.bool)
        for i in range(n_prev):
            for j in range(n_next):
                if prev_tids[i] != -1 and prev_tids[i] == next_tids[j]:
                    link_gt[i, j] = True

        start_gt = torch.logical_not(torch.sum(link_gt, dim=0, keepdim=False))
        end_gt = torch.logical_not(torch.sum(link_gt, dim=1, keepdim=False))
        start_gt[next_tids == -1] = False
        end_gt[prev_tids == -1] = False

        batch_dict = {
            'num_prev': n_prev,
            'num_next': n_next,
            'cls_feature': cls_feature,
            'aff_feature': aff_feature,
            'cls_gt': cls_gt,
            'link_gt': link_gt,
            'start_gt': start_gt,
            'end_gt': end_gt
        }
        return batch_dict

    @staticmethod
    def batch_minus_abs(feat1: Tensor, feat2: Tensor):
        """
          Absolute subtraction operation
          Args:
              feat1 (Tensor): (N, D, 1)
              feat2 (Tensor): (M, D, 1)

          Returns:
              cor_feat (Tensor): (N * M, D)
          """
        N = feat1.shape[0]
        M = feat2.shape[0]
        feat1_mat = feat1.unsqueeze(1).repeat(1, M, 1)
        feat2_mat = feat2.unsqueeze(0).repeat(N, 1, 1)
        cor_feat = (feat1_mat - feat2_mat).abs()  # (N, M, D)
        cor_feat = cor_feat.view(N * M, -1)
        return cor_feat

    @staticmethod
    def batch_cat(feat1: Tensor, feat2: Tensor):
        """
          Concatenate operation
          Args:
              feat1 (Tensor): (N, D, 1)
              feat2 (Tensor): (M, D, 1)

          Returns:
              cor_feat (Tensor): (N * M, 2 * D)
          """
        N = feat1.shape[0]
        M = feat2.shape[0]
        feat1_mat = feat1.unsqueeze(1).repeat(1, M, 1)
        feat2_mat = feat2.unsqueeze(0).repeat(N, 1, 1)
        cor_feat = torch.cat([feat1_mat, feat2_mat], dim=2)
        cor_feat = cor_feat.view(N * M, -1)
        return cor_feat

    @staticmethod
    def collate_fn(batch_list):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if 'num' in key:
                    ret[key] = np.array(val, dtype=np.int32)
                elif 'gt' in key:
                    gt = [torch.flatten(x) for x in val]
                    ret[key] = torch.cat(gt)  # N
                elif 'feature' in key:
                    ret[key] = torch.cat(val, dim=0)  # (N, D)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
