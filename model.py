import torch
import torch.nn as nn
import torch.nn.functional as F


class SCAtt(nn.Module):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAtt, self).__init__()
        sequential = []
        for i in range(1, len(mid_dims) - 1):
            sequential.append(nn.Linear(mid_dims[i - 1], mid_dims[i]))
            sequential.append(nn.ReLU())
            if mid_dropout > 0:
                sequential.append(nn.Dropout(mid_dropout))
        self.attention_basic = nn.Sequential(
            *sequential) if len(sequential) > 0 else None

        self.attention_last = nn.Linear(mid_dims[-2], 1)
        self.attention_last2 = nn.Linear(mid_dims[-2], mid_dims[-1])

    def forward(self, att_map, query, value):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)
        att_map_pool = att_map.mean(-2)
        alpha_spatial = self.attention_last(att_map)
        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = torch.sigmoid(alpha_channel)
        alpha_spatial = alpha_spatial.squeeze(-1)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)
        if len(alpha_spatial.shape) == 4:
            value = torch.matmul(alpha_spatial, value)
        else:
            value = torch.matmul(alpha_spatial.unsqueeze(-2),
                                 value).squeeze(-2)

        attn = query * value * alpha_channel
        return attn


class BiLinear(nn.Module):
    def __init__(self, embed_dim, att_heads, att_mid_dim, att_mid_drop):
        super(BiLinear, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        output_dim = embed_dim

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = nn.CELU(1.3)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_q = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = nn.CELU(1.3)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_k = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = nn.CELU(1.3)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v1 = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = nn.CELU(1.3)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v2 = nn.Sequential(*sequential)

        self.attn_net = SCAtt(att_mid_dim, att_mid_drop)

    def forward(self, query, key, value):
        batch_size = query.size()[0]
        q1 = self.in_proj_q(query)
        q2 = self.in_proj_v1(query)

        q1 = q1.view(batch_size, self.num_heads, self.head_dim)
        q2 = q2.view(batch_size, self.num_heads, self.head_dim)

        key = key.view(-1, key.size()[-1])
        value = value.view(-1, value.size()[-1])
        k = self.in_proj_k(key)
        v = self.in_proj_v2(value)
        k = k.view(batch_size, -1, self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads,
                   self.head_dim).transpose(1, 2)

        attn_map = q1.unsqueeze(-2) * k
        attn = self.attn_net(attn_map, q2, v)
        attn = attn.view(batch_size, self.num_heads * self.head_dim)
        return attn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.avg_pool(x).view(n, c)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups,
                 pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=False),
            norm_layer(planes),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=3,
                      stride=1,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=False),
            norm_layer(planes),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(inplanes,
                      planes,
                      kernel_size=3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=False),
            norm_layer(planes),
        )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x),
                                              identity.size()[2:])))
        out = torch.mul(self.k3(x), out)
        out = self.k4(out)

        return out


class attentionBlock(nn.Module):
    def __init__(self, in_channel):
        super(attentionBlock, self).__init__()
        self.branchChannel = in_channel // 2
        self.attention1 = SELayer(self.branchChannel)
        self.sc = SCConv(self.branchChannel,
                         self.branchChannel,
                         stride=1,
                         padding=1,
                         dilation=1,
                         groups=1,
                         pooling_r=4,
                         norm_layer=nn.BatchNorm2d)

    def forward(self, x):
        out = torch.cat((self.attention1(x[:, self.branchChannel:, :, :]),
                         self.sc(x[:, :self.branchChannel, :, :])), 1)
        return x + out


class _NonLocalBlockND(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=3,
                 sub_sample=False,
                 bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, y, z, return_nl_map=False):
        batch_size = x.size(0)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_y = self.phi(y).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_y)
        f_div_C = F.softmax(f, dim=-1)

        g_z = self.g(z).view(batch_size, self.inter_channels, -1)
        g_z = g_z.permute(0, 2, 1)
        o = torch.matmul(f_div_C, g_z)
        o = o.permute(0, 2, 1).contiguous()
        o = o.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(o)
        z = W_y + z
        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True):
        super(NONLocalBlock2D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=2,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )


class Net1(nn.Module):
    def __init__(self, num_classes=20):
        super(Net1, self).__init__()
        """
        Branch 1 lidar
        """
        self.lidar_step1 = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=64, kernel_size=3,
                      padding=0), nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True))
        self.lidar_step2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=0), nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True))
        self.lidar_step3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      padding=0), nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True), attentionBlock(256))
        """
        branch 2 hsi
        """
        self.hsi_step1 = nn.Sequential(
            nn.Conv3d(in_channels=1,
                      out_channels=8,
                      kernel_size=(9, 3, 3),
                      padding=0), nn.BatchNorm3d(num_features=8),
            nn.ReLU(inplace=True))

        self.hsi_step2 = nn.Sequential(
            nn.Conv3d(in_channels=8,
                      out_channels=16,
                      kernel_size=(7, 3, 3),
                      padding=0), nn.BatchNorm3d(num_features=16),
            nn.ReLU(inplace=True))
        self.hsi_step3 = nn.Sequential(
            nn.Conv3d(in_channels=16,
                      out_channels=32,
                      kernel_size=(5, 3, 3),
                      padding=0), nn.BatchNorm3d(num_features=32),
            nn.ReLU(inplace=True))
        self.hsi_step4 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=3,
                      padding=1), nn.BatchNorm2d(num_features=256), nn.ReLU(),
            attentionBlock(256))
        """
        Fusion tools
        """
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=256,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True))
        self.non_fusion = NONLocalBlock2D(256)
        self.non_fusion1 = NONLocalBlock2D(256)
        self.non_fusion2 = NONLocalBlock2D(256)
        self.projector1 = nn.Linear(in_features=256 * 3, out_features=256 * 3)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.projector2 = nn.Linear(in_features=256 * 3,
                                    out_features=num_classes)
        self.bilinear1 = BiLinear(embed_dim=256,
                                  att_heads=8,
                                  att_mid_dim=[32, 16, 32],
                                  att_mid_drop=0.1)
        self.bilinear2 = BiLinear(embed_dim=256,
                                  att_heads=8,
                                  att_mid_dim=[32, 16, 32],
                                  att_mid_drop=0.1)
        self.norm = nn.LayerNorm(256)

    def forward(self, hsi, lidar):
        """
        branch 1 lidar
        """
        x = self.lidar_step1(lidar)
        x = self.lidar_step2(x)
        x = self.lidar_step3(x)
        """
        branch 2 hsi
        """
        y = self.hsi_step1(hsi)
        y = self.hsi_step2(y)
        y = self.hsi_step3(y)
        y = y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
        y = self.hsi_step4(y)
        f1 = torch.cat((x, y), dim=1)
        f1 = self.fusion(f1)

        f3 = F.avg_pool2d(self.non_fusion(x, y, f1),
                          kernel_size=y.shape[2]).reshape(-1, 256)
        f1 = F.avg_pool2d(self.non_fusion1(x, x, y),
                          kernel_size=y.shape[2]).reshape(-1, 256)
        f2 = F.avg_pool2d(self.non_fusion2(y, y, x),
                          kernel_size=y.shape[2]).reshape(-1, 256)
        f11 = torch.stack((f2, f1, f3), 1)
        f21 = torch.stack((f1, f2, f3), 1)

        output = torch.cat(
            (self.norm(f3), self.norm(self.bilinear1(
                f1, f11, f11)), self.norm(self.bilinear2(f2, f21, f21))), 1)

        output = self.dropout(output)
        output = self.relu(self.projector1(output))
        output = self.projector2(output)
        return output


class Net2(nn.Module):
    def __init__(self, num_classes=20):
        super(Net2, self).__init__()
        """
        Branch 1 lidar
        """
        self.lidar_step1 = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=64, kernel_size=3,
                      padding=0), nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True))
        self.lidar_step2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=0), nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True))
        self.lidar_step3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      padding=0), nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True), attentionBlock(256))

        self.hsi_step1 = nn.Sequential(
            nn.Conv3d(in_channels=1,
                      out_channels=8,
                      kernel_size=(9, 3, 3),
                      padding=0), nn.BatchNorm3d(num_features=8),
            nn.ReLU(inplace=True))

        self.hsi_step2 = nn.Sequential(
            nn.Conv3d(in_channels=8,
                      out_channels=16,
                      kernel_size=(7, 3, 3),
                      padding=0), nn.BatchNorm3d(num_features=16),
            nn.ReLU(inplace=True))
        self.hsi_step3 = nn.Sequential(
            nn.Conv3d(in_channels=16,
                      out_channels=32,
                      kernel_size=(5, 3, 3),
                      padding=0), nn.BatchNorm3d(num_features=32),
            nn.ReLU(inplace=True))
        self.hsi_step4 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=3,
                      padding=1), nn.BatchNorm2d(num_features=256), nn.ReLU(),
            attentionBlock(256))
        """
        Fusion tools
        """
        self.fusion_step = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=256,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True))
        self.nonlocalfusion = NONLocalBlock2D(256)
        self.nonlocalfusion1 = NONLocalBlock2D(256)
        self.nonlocalfusion2 = NONLocalBlock2D(256)

        self.fc = nn.Linear(in_features=256 * 3, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU(inplace=True)

        self.norm = nn.LayerNorm(256)

    def forward(self, hsi, lidar):
        """
        branch 1 lidar
        """
        x = self.lidar_step1(lidar)
        x = self.lidar_step2(x)
        x = self.lidar_step3(x)
        """
        branch 2 hsi
        """
        y = self.hsi_step1(hsi)
        y = self.hsi_step2(y)
        y = self.hsi_step3(y)
        y = y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
        y = self.hsi_step4(y)

        f1 = torch.cat((x, y), dim=1)
        f1 = self.fusion_step(f1)

        f3 = F.avg_pool2d(self.nonlocalfusion(x, y, f1),
                          kernel_size=5).reshape(-1, 256)
        f1 = F.avg_pool2d(self.nonlocalfusion1(x, x, y),
                          kernel_size=5).reshape(-1, 256)
        f2 = F.avg_pool2d(self.nonlocalfusion2(y, y, x),
                          kernel_size=5).reshape(-1, 256)

        output = torch.cat((f3, f1, f2), 1)
        output = self.dropout(output)
        output = self.fc(output)

        return output
