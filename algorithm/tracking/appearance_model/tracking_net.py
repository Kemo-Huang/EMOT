from .appear_net import AppearanceNet
from .fusion_net import FusionModule
from .point_net import PointNet
from .gcn import AffinityModule
from .new_end import NewEndModule
import torch
from torch import nn


class TrackingNet(nn.Module):

    def __init__(self,
                 appear_len=512,
                 appear_skippool=True,
                 appear_fpn=False,
                 appear_arch='vgg',
                 point_len=512,
                 test_mode=2,
                 dropblock=0,
                 neg_threshold=0.2,
                 affinity_op='minus_abs',
                 use_dropout=False):
        super(TrackingNet, self).__init__()
        self.neg_threshold = neg_threshold
        point_in_channels = 3

        if point_len == 0:
            in_channels = appear_len
        else:
            in_channels = point_len

        self.fusion_module = FusionModule(appear_len, point_len, out_channels=point_len)

        if test_mode == 1:
            print('No image appearance used')
            self.appearance = None
        else:
            self.appearance = AppearanceNet(
                appear_arch,
                appear_len,
                skippool=appear_skippool,
                fpn=appear_fpn,
                dropblock=dropblock)

        # build point net
        if test_mode == 0:
            print("No point cloud used")
            self.point_net = None
        else:
            self.point_net = PointNet(
                point_in_channels,
                out_channels=point_len,
                use_dropout=use_dropout)

        # build negative rejection module
        self.w_det = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, in_channels // 2, 1, 1),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 2, 1, 1, 1),
        )
        new_end = NewEndModule(in_channels)
        self.w_link = AffinityModule(in_channels, new_end=new_end, affinity_op=affinity_op)

    def feature(self, image_dets, point_dets, det_lens):
        feats = []

        appear = self.appearance(image_dets)
        feats.append(appear)

        point_dets = point_dets.transpose(-1, -2).unsqueeze(0)
        points = self.point_net(point_dets, det_lens)
        feats.append(points)

        feats = torch.cat(feats, dim=-1).t().unsqueeze(0)  # LxD->1xDxL
        feats = self.fusion_module(feats)
        return feats

    def determine_det(self, feats):
        det_scores = self.w_det(feats).squeeze(1)  # Bx1xL -> BxL

        # add mask
        det_scores = det_scores.sigmoid()
        mask = det_scores.lt(self.neg_threshold)
        det_scores -= mask.float()
        return det_scores

    def forward(self, image_dets, point_dets, det_lens):
        """
        :param image_dets: N * 3 * 224 * 224
        :param point_dets: N * 3
        :param det_lens: N
        :return: scores: 3 * N
                feats: 3 * 512 * N
        """
        feats = self.feature(image_dets, point_dets, det_lens)
        # confidence estimator
        det_scores = self.determine_det(feats)
        return det_scores[2], feats

    def scoring(self, predictions, detections):
        link_mats, new_scores, end_scores = self.w_link(predictions, detections)
        link_score_prev = nn.functional.softmax(link_mats, dim=-1)
        link_score_next = nn.functional.softmax(link_mats, dim=-2)
        link_scores = (link_score_prev + link_score_next) / 2
        link_scores = link_scores[2].squeeze(0)
        new_scores = nn.functional.pad(new_scores[2], [link_scores.size(0), 0], "constant", 0)
        end_scores = nn.functional.pad(end_scores[2], [0, link_scores.size(1)], "constant", 0)
        return link_scores, new_scores, end_scores
