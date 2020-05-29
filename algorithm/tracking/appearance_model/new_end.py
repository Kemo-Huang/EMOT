import torch.nn as nn


class NewEndModule(nn.Module):

    def __init__(self, in_channels):
        super(NewEndModule, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1),
            nn.GroupNorm(1, in_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, min(in_channels, 512), 1, 1),
            nn.GroupNorm(1, min(in_channels, 512)), nn.ReLU(inplace=True),
            nn.Conv1d(min(in_channels, 512), in_channels // 4, 1, 1),
            nn.GroupNorm(1, in_channels // 4), nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, 1, 1, 1), nn.Sigmoid())

    def forward(self, x):
        """
        x: BxCxNxM
        w_new: BxM
        w_end: BxN
        """
        x = self.conv0(x)
        new_vec = x.mean(dim=-2, keepdim=False)  # 1xCxM
        end_vec = x.mean(dim=-1, keepdim=False)  # 1xCxN
        w_new = self.conv1(new_vec).squeeze(1)  # BxCxM->Bx1xM->BxM
        w_end = self.conv1(end_vec).squeeze(1)  # BxCxN->Bx1xN->BxN
        return w_new, w_end
