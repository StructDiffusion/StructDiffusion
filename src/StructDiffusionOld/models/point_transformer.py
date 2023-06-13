import torch
import torch.nn as nn
from StructDiffusion.utils.pointnet import farthest_point_sample, index_points, square_distance

# adapted from https://github.com/qq456cvb/Point-Transformers


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 
    
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    new_xyz = index_points(xyz, fps_idx) 
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    

class StackedAttention(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        
        x = torch.cat((x1, x2), dim=1)

        return x


class PointTransformerEncoderSmall(nn.Module):

    def __init__(self, output_dim=256, input_dim=6, mean_center=True):
        super(PointTransformerEncoderSmall, self).__init__()

        self.mean_center = mean_center

        # map the second dim of the input from input_dim to 64
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=64)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=64)
        self.pt_last = StackedAttention(channels=64)

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(192, 256, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(256),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(256, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(256, 256)

    def forward(self, xyz, f=None):
        # xyz: B, N, 3
        # f: B, N, D
        center = torch.mean(xyz, dim=1)
        if self.mean_center:
            xyz = xyz - center.view(-1, 1, 3).repeat(1, xyz.shape[1], 1)
        if f is None:
            x = self.pct(xyz)
        else:
            x = self.pct(torch.cat([xyz, f], dim=2))  # B, output_dim

        return center, x

    def pct(self, x):

        # x: B, N, D
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=128, nsample=32, xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)  # B, nsamples, D
        new_xyz, new_feature = sample_and_group(npoint=32, nsample=16, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)  # B, D, nsamples

        x = self.pt_last(feature_1)  # B, D * 2, nsamples
        x = torch.cat([x, feature_1], dim=1)  # B, D * 3, nsamples
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)

        return x


class SampleAndGroup(nn.Module):

    def __init__(self, output_dim=64, input_dim=6, mean_center=True, npoints=(128, 32), nsamples=(32, 16)):
        super(SampleAndGroup, self).__init__()

        self.mean_center = mean_center
        self.npoints = npoints
        self.nsamples = nsamples

        # map the second dim of the input from input_dim to 64
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.gather_local_0 = Local_op(in_channels=output_dim * 2, out_channels=output_dim)
        self.gather_local_1 = Local_op(in_channels=output_dim * 2, out_channels=output_dim)
        self.relu = nn.ReLU()

    def forward(self, xyz, f):
        # xyz: B, N, 3
        # f: B, N, D
        center = torch.mean(xyz, dim=1)
        if self.mean_center:
            xyz = xyz - center.view(-1, 1, 3).repeat(1, xyz.shape[1], 1)
        x = self.sg(torch.cat([xyz, f], dim=2))  # B, nsamples, output_dim

        return center, x

    def sg(self, x):

        # x: B, N, D
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=self.npoints[0], nsample=self.nsamples[0], xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)  # B, nsamples, D
        new_xyz, new_feature = sample_and_group(npoint=self.npoints[1], nsample=self.nsamples[1], xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)  # B, D, nsamples
        x = feature_1.permute(0, 2, 1)  # B, nsamples, D

        return x