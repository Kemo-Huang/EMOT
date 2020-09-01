import torch
from torch import nn
import torch.nn.functional as F
from .utils import make_fc_layers, _make_fc_layers


class TrackingNet(nn.Module):
    def __init__(self, model_cfg, cls_in_channels, aff_in_channels):
        super(TrackingNet, self).__init__()
        self.cls_fc = make_fc_layers(cls_in_channels, 1, model_cfg.CLS_FC)
        self.aff_fc = make_fc_layers(aff_in_channels, 1, model_cfg.AFF_FC)
        self.fc1_list, last_c = _make_fc_layers(aff_in_channels, model_cfg.SHARED_FC1)
        self.shared_fc1 = nn.Sequential(*self.fc1_list)
        self.start_fc2 = make_fc_layers(last_c, 1, model_cfg.SHARED_FC2)
        self.end_fc2 = make_fc_layers(last_c, 1, model_cfg.SHARED_FC2)
        self.bce_loss = nn.BCELoss()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_cls_scores(self, cls_feat):
        return torch.sigmoid(self.cls_fc(cls_feat))

    def get_aff_start_end_scores(self, prev_feat, next_feat):
        n_prev = prev_feat.shape[0]
        n_next = next_feat.shape[0]
        feat1_mat = prev_feat.unsqueeze(1).repeat(1, n_next, 1)
        feat2_mat = next_feat.unsqueeze(0).repeat(n_prev, 1, 1)
        aff_feat = torch.cat([feat1_mat, feat2_mat], dim=2).view(n_prev * n_next, -1)

        link_matrix = self.aff_fc(aff_feat).view(n_prev, n_next)
        link_matrix_prev = F.softmax(link_matrix, dim=1)
        link_matrix_next = F.softmax(link_matrix, dim=0)
        link_matrix = (link_matrix_prev + link_matrix_next) / 2  # (N, M)
        link_scores = torch.sigmoid(link_matrix)

        shared_feat = self.shared_fc1(aff_feat).view(n_prev, n_next, -1)
        start_feat = shared_feat.mean(dim=0, keepdim=False)  # (M, D)
        end_feat = shared_feat.mean(dim=1, keepdim=False)  # (N, D)
        start_scores = torch.sigmoid(self.start_fc2(start_feat))
        end_scores = torch.sigmoid(self.start_fc2(end_feat))

        start_scores_pad = torch.zeros((n_prev + n_next, 1), dtype=torch.float32)
        end_scores_pad = torch.zeros((n_prev + n_next, 1), dtype=torch.float32)
        start_scores_pad[n_prev:] = start_scores
        end_scores_pad[:n_prev] = end_scores

        return link_scores, start_scores_pad, end_scores_pad

    def get_cls_loss(self, batch_dict):
        cls_feat = batch_dict['cls_feature']  # (N1 + M1 + ..., D)
        cls_gt = batch_dict['cls_gt']
        cls_scores = torch.flatten(self.cls_fc(cls_feat))
        cls_loss = self.bce_loss(torch.sigmoid(cls_scores), cls_gt)
        return cls_loss

    def get_link_loss(self, batch_dict):
        batch_size = batch_dict['batch_size']
        batch_n_prev = batch_dict['num_prev']  # (B, N)
        batch_n_next = batch_dict['num_next']  # (B, M)
        aff_feat = batch_dict['aff_feature']  # (N1 * M1 + ..., D)

        link_gt = batch_dict['link_gt']
        batch_link_matrix = self.aff_fc(aff_feat)
        link_slice = 0
        link_scores = []
        for idx in range(batch_size):
            n_prev = batch_n_prev[idx]
            n_next = batch_n_next[idx]
            link_matrix = batch_link_matrix[link_slice:link_slice + n_prev * n_next].view(n_prev, n_next)
            link_slice += n_prev * n_next
            link_matrix_prev = F.softmax(link_matrix, dim=1)
            link_matrix_next = F.softmax(link_matrix, dim=0)
            link_matrix = (link_matrix_prev + link_matrix_next) / 2  # (N, M)
            link_scores.append(torch.flatten(link_matrix))  # (N * M，)

        link_scores = torch.cat(link_scores)
        link_loss = self.bce_loss(torch.sigmoid(link_scores), link_gt)
        return link_loss

    def get_start_end_loss(self, batch_dict):
        batch_size = batch_dict['batch_size']
        batch_n_prev = batch_dict['num_prev']  # (B, N)
        batch_n_next = batch_dict['num_next']  # (B, M)
        aff_feat = batch_dict['aff_feature']  # (N1 * M1 + ..., D)

        batch_shared_feat = self.shared_fc1(aff_feat)
        link_slice = 0
        batch_start_feat = []
        batch_end_feat = []
        for idx in range(batch_size):
            n_prev = batch_n_prev[idx]
            n_next = batch_n_next[idx]
            shared_feat = batch_shared_feat[link_slice:link_slice + n_prev * n_next, :].view(n_prev, n_next, -1)
            start_feat = shared_feat.mean(dim=0, keepdim=False)  # (M, D)
            end_feat = shared_feat.mean(dim=1, keepdim=False)  # (N, D)
            batch_start_feat.append(start_feat)
            batch_end_feat.append(end_feat)
        batch_start_feat = torch.cat(batch_start_feat, dim=0)
        batch_end_feat = torch.cat(batch_end_feat, dim=0)

        start_scores = torch.flatten(self.start_fc2(batch_start_feat))
        end_scores = torch.flatten(self.end_fc2(batch_end_feat))

        start_gt = batch_dict['start_gt']
        end_gt = batch_dict['end_gt']

        start_loss = self.bce_loss(torch.sigmoid(start_scores), start_gt)
        end_loss = self.bce_loss(torch.sigmoid(end_scores), end_gt)
        return start_loss, end_loss

    def forward(self, batch_dict):
        # cls_loss = self.get_cls_loss(batch_dict)
        link_loss = self.get_link_loss(batch_dict)
        start_loss, end_loss = self.get_start_end_loss(batch_dict)

        tb_dict = {
            # 'cls_loss': cls_loss.item(),
            'link_loss': link_loss.item(),
            'start_loss': start_loss.item(),
            'end_loss': end_loss.item()
        }

        loss = 2 * link_loss + start_loss + end_loss
        # loss = cls_loss + link_loss + start_loss + end_loss

        return loss, tb_dict
