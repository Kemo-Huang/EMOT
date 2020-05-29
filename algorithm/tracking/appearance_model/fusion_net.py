import torch
import torch.nn as nn


# Common fusion module
class FusionModule(nn.Module):

    def __init__(self, appear_len, point_len, out_channels):
        super(FusionModule, self).__init__()
        self.appear_len = appear_len
        self.point_len = point_len
        self.gate_p = nn.Sequential(
            nn.Conv1d(point_len, point_len, 1, 1),
            nn.Sigmoid(),
        )
        self.gate_i = nn.Sequential(
            nn.Conv1d(appear_len, appear_len, 1, 1),
            nn.Sigmoid(),
        )
        self.input_p = nn.Sequential(
            nn.Conv1d(point_len, out_channels, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
        )
        self.input_i = nn.Sequential(
            nn.Conv1d(appear_len, out_channels, 1, 1),
            nn.GroupNorm(out_channels, out_channels),
        )

    def forward(self, objs):
        """
            objs : 1xDxN
        """
        feats = objs.view(2, -1, objs.size(-1))  # 1x2DxL -> 2xDxL
        gate_p = self.gate_p(feats[:1])  # 2xDxL
        gate_i = self.gate_i(feats[1:])  # 2xDxL
        obj_fused = gate_p.mul(self.input_p(feats[:1])) + gate_i.mul(
            self.input_i(feats[1:]))

        obj_feats = torch.cat([feats, obj_fused.div(gate_p + gate_i)], dim=0)
        return obj_feats
