import torch
import torch.nn as nn
import torch.nn.functional as F


class DropoutSampler(torch.nn.Module):
    def __init__(self, num_features, num_outputs, dropout_rate = 0.5):
        super(DropoutSampler, self).__init__()
        self.linear = nn.Linear(num_features, num_features)
        self.linear2 = nn.Linear(num_features, num_features)
        self.predict = nn.Linear(num_features, num_outputs)
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = F.relu(self.linear(x))
        if self.dropout_rate > 0:
            x = F.dropout(x, self.dropout_rate)
        x = F.relu(self.linear(x))
        # x = F.dropout(x, self.dropout_rate)
        return self.predict(x)


class EncoderMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, pt_dim=3, uses_pt=True):
        super(EncoderMLP, self).__init__()
        self.uses_pt = uses_pt
        self.output = out_dim
        d5 = int(in_dim)
        d6 = int(2 * self.output)
        d7 = self.output
        self.encode_position = nn.Sequential(
                nn.Linear(pt_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.ReLU(),
                )
        d5 = 2 * in_dim if self.uses_pt else in_dim
        self.fc_block = nn.Sequential(
            nn.Linear(int(d5), d6),
            nn.LayerNorm(int(d6)),
            nn.ReLU(),
            nn.Linear(int(d6), d6),
            nn.LayerNorm(int(d6)),
            nn.ReLU(),
            nn.Linear(d6, d7))

    def forward(self, x, pt=None):
        if self.uses_pt:
            if pt is None: raise RuntimeError('did not provide pt')
            y = self.encode_position(pt)
            x = torch.cat([x, y], dim=-1)
        return self.fc_block(x)


class MeanEncoder(torch.nn.Module):
    def __init__(self, input_channels=3, use_xyz=True, output=512, scale=0.04, factor=1):
        super(MeanEncoder, self).__init__()
        self.uses_rgb = False
        self.dim = 3

    def forward(self, xyz, f=None):

        # Fix shape
        if f is not None:
            if len(f.shape) < 3:
                f = f.transpose(0,1).contiguous()
                f = f[None]
            else:
                f = f.transpose(1,2).contiguous()
        if len(xyz.shape) == 3:
            center = torch.mean(xyz, dim=1)
        elif len(xyz.shape) == 2:
            center = torch.mean(xyz, dim=0)
        else:
            raise RuntimeError('not sure what to do with points of shape ' + str(xyz.shape))
        assert(xyz.shape[-1]) == 3
        assert(center.shape[-1]) == 3
        return center, center