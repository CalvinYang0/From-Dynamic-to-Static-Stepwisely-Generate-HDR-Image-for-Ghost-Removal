
import torch.nn as nn

from .modules import *
import torch.nn.functional as f
from torchvision.transforms.functional import rgb_to_grayscale
class invertedBlock(nn.Module):
    def __init__(self, in_channel,ratio=2):
        super(invertedBlock, self).__init__()
        internal_channel = in_channel * ratio
        self.relu = nn.GELU()

        self.conv1 = nn.Conv2d(internal_channel, internal_channel, 7, 1, 3, groups=in_channel,bias=False)

        self.convFFN = ConvFFN(in_channels=in_channel, out_channels=in_channel)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.pw1 = nn.Conv2d(in_channels=in_channel, out_channels=internal_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channel, out_channels=in_channel, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)


    def hifi(self,x):

        x1=self.pw1(x)
        x1=self.relu(x1)
        x1=self.conv1(x1)
        x1=self.relu(x1)
        x1=self.pw2(x1)
        x1=self.relu(x1)
        # x2 = self.conv2(x)
        x3 = x1+x

        x3 = x3.permute(0, 2, 3, 1).contiguous()
        x3 = self.layer_norm(x3)
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x4 = self.convFFN(x3)

        return x4

    def forward(self, x):
        return self.hifi(x)+x
class ConvFFN(nn.Module):

    def __init__(self, in_channels, out_channels, expend_ratio=4):
        super().__init__()

        internal_channels = in_channels * expend_ratio
        self.pw1 = nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.pw2 = nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1,
                             padding=0, groups=1,bias=False)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        x1 = self.pw1(x)
        x2 = self.nonlinear(x1)
        x3 = self.pw2(x2)
        x4 = self.nonlinear(x3)
        return x4 + x

class mixblock(nn.Module):
    def __init__(self, n_feats):
        super(mixblock, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.conv2=nn.Sequential(nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU(),nn.Conv2d(n_feats,n_feats,3,1,1,bias=False),nn.GELU())
        self.alpha=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.ones(1))
    def forward(self,x):
        return self.alpha*self.conv1(x)+self.beta*self.conv2(x)
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class Downupblock(nn.Module):
    def __init__(self, n_feats):
        super(Downupblock, self).__init__()
        self.encoder = mixblock(n_feats)
        self.decoder_high = mixblock(n_feats)

        self.decoder_low = nn.Sequential(mixblock(n_feats), mixblock(n_feats), mixblock(n_feats))
        self.alise = nn.Conv2d(n_feats,n_feats,1,1,0,bias=False)
        self.alise2 = nn.Conv2d(n_feats*2,n_feats,3,1,1,bias=False)
        self.down = nn.MaxPool2d(kernel_size=2)
        self.att = CALayer(n_feats)
        self.raw_alpha=nn.Parameter(torch.ones(1))

        self.raw_alpha.data.fill_(0)
        self.ega=EdgeAttentionModule(n_feats,n_feats)

    def forward(self, x,raw):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)
        if raw is not None:
            high=high+self.ega(raw*high,high)*self.raw_alpha
        x2=self.decoder_low(x2)
        x3 = x2

        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x

class Updownblock(nn.Module):
    def __init__(self, n_feats):
        super(Updownblock, self).__init__()
        self.encoder = mixblock(n_feats)
        self.decoder_high = mixblock(n_feats)  # nn.Sequential(one_module(n_feats),
        #                     one_module(n_feats),
        #                     one_module(n_feats))
        self.decoder_low = nn.Sequential(mixblock(n_feats), mixblock(n_feats), mixblock(n_feats))

        self.alise = nn.Conv2d(n_feats,n_feats,1,1,0,bias=False)  # one_module(n_feats)
        self.alise2 = nn.Conv2d(n_feats*2,n_feats,3,1,1,bias=False)  # one_module(n_feats)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)
        self.raw_alpha=nn.Parameter(torch.ones(1))
        # fill 0
        self.raw_alpha.data.fill_(0)
        self.ega=EdgeAttentionModule(n_feats,n_feats)

    def forward(self, x,raw):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)
        if raw is not None:
            high=high+self.ega(raw*high,high)*self.raw_alpha
        x2=self.decoder_low(x2)
        x3 = x2
        # x3 = self.decoder_low(x2)
        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x

class basic_block(nn.Module):

    def __init__(self, in_channel, out_channel, depth,ratio=1):
        super(basic_block, self).__init__()
        self.rep1 = nn.Sequential(*[invertedBlock(in_channel=in_channel,ratio=ratio) for i in range(depth)])
        self.relu=nn.GELU()
        self.updown=Updownblock(in_channel)
        self.downup=Downupblock(in_channel)
    def forward(self, x,raw=None):


        x1 = self.rep1(x)


        x1=self.updown(x1,raw)
        x1=self.downup(x1,raw)
        return x1+x

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class EdgeAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeAttentionModule, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = 1.0 / (out_channels ** 0.5)

    def forward(self, edge_feature, feature_map):
        query = self.query_conv(edge_feature)
        key = self.key_conv(edge_feature)
        value = self.value_conv(edge_feature)


        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores * self.scale


        attention_weights = f.softmax(attention_scores, dim=-1)


        attended_values = torch.matmul(attention_weights, value)

        output_feature_map = (feature_map + attended_values)

        return output_feature_map


class pgn(nn.Module):
    def __init__(self, n_Feature=64, depth_list=[6,6,6], in_dim=6, out_dim=3):
        super(pgn, self).__init__()
        feature_list = [n_Feature , n_Feature , n_Feature]
        ratio_list = [1,1,1]
        self.conv1_1 = nn.Conv2d(in_dim, n_Feature, kernel_size=3, padding=1, bias=False)
        self.conv1_2 = nn.Conv2d(in_dim, n_Feature, kernel_size=3, padding=1, bias=False)
        self.conv1_3 = nn.Conv2d(in_dim, n_Feature, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_Feature * 3, n_Feature, kernel_size=3, padding=1, bias=False)
        self.convlist = nn.ModuleList(
            [basic_block(feature_list[i], feature_list[i], depth_list[i],ratio_list[i]) for i in
             range(len(depth_list))])
        self.GFF_1x1 = nn.Conv2d(n_Feature * 3, n_Feature, kernel_size=1, padding=0, bias=False)
        self.GFF_3x3 = nn.Conv2d(n_Feature, n_Feature, kernel_size=3, padding=1, bias=False)

        self.conv_up = nn.Conv2d(n_Feature, n_Feature, kernel_size=3, padding=1, bias=False)

        self.conv3 = nn.Conv2d(n_Feature, out_dim, kernel_size=3, padding=1, bias=False)
        self.relu = nn.GELU()

        self.cfc = nn.Parameter(torch.Tensor(n_Feature*3, 2))
        self.cfc.data.fill_(0)
        self.c=nn.Conv2d(n_Feature*3,3,1,1,0)
        self.c2=nn.Conv2d(3,1,1,1,0)
    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t
    def forward(self,x1,x2,x3,edge1,edge2,edge3):
        F1_ = self.relu(self.conv1_1(x1))
        F2_0 = self.conv1_2(x2)
        F2_ = self.relu(F2_0)
        F3_ = self.relu(self.conv1_3(x3))
        F_ = torch.cat((F1_, F2_, F3_), 1)
        edge1=rgb_to_grayscale(edge1)
        edge2=rgb_to_grayscale(edge2)
        edge3=rgb_to_grayscale(edge3)
        edge1=make_laplace(edge1,1)
        edge2=make_laplace(edge2,1)
        edge3=make_laplace(edge3,1)
        edge=torch.cat([edge1,edge2,edge3],1)
        raw=self._style_pooling(F_)
        raw=self._style_integration(raw)
        raw=self.c(raw)
        raw=raw*edge
        raw=self.c2(raw)
        raw=self.relu(raw)
        F_0 = self.conv2(F_)
        F_1 = self.convlist[0](F_0,raw)
        F_2 = self.convlist[1](F_1,raw)
        F_3 = self.convlist[2](F_2,raw)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.relu(self.GFF_1x1(FF))
        FGF = self.relu(self.GFF_3x3(FdLF))
        FDF = FGF + F2_
        us = self.relu(self.conv_up(FDF))
        output = self.conv3(us)
        output = nn.functional.sigmoid(output)
        return output
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]
        z = torch.sum(z, dim=2)[:, :, None, None]
        return torch.sigmoid(z)