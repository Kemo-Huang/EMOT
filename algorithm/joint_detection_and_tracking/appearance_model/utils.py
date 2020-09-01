from torch import nn


def _make_fc_layers(input_channels, conv_list):
    fc_layers = []
    pre_channel = input_channels
    for k in range(0, conv_list.__len__()):
        fc_layers.extend([
            nn.Linear(pre_channel, conv_list[k], bias=False),
            nn.BatchNorm1d(conv_list[k]),
            nn.ReLU(inplace=True)
        ])
        pre_channel = conv_list[k]
        if k == 0:
            fc_layers.append(nn.Dropout(0.2))
    return fc_layers, pre_channel


def make_fc_layers(input_channels, output_channels, fc_list):
    fc_layers, pre_channel = _make_fc_layers(input_channels, fc_list)
    fc_layers.append(nn.Linear(pre_channel, output_channels, bias=True))
    fc_layers = nn.Sequential(*fc_layers)
    return fc_layers
