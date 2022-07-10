import torch.nn as nn
import math
from .basicblock import BasicBlock2D
from .bottleneck import Bottleneck2D
# from utils.other import transform_input
import ipdb
import torch
# from utils.meter import *

from MaskFormer.demo.demo import get_maskformer


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

K_1st_CONV = 3


class ORN_ResNet(nn.Module):
    def __init__(self,
                layers,
                two_heads=False,
                pooling='avg',
                 ):
        self.two_heads = True
        self.inplanes = 64
        self.input_dim = 5  # from 5D to 4D tensor if 2D conv
        super(ResNet, self).__init__()
        self.num_final_fm = num_final_fm
        self.time = None
        self.relu = nn.ReLU(inplace=True)
        self.resnet = get_maskformer.backbone()
        self.avgpool, self.avgpool_space, self.avgpool_time = None, None, None
        self.out_dim = 5
        self.pooling = pooling

        if self.two_heads:
            list_strides_2nd_head = [2]
            self.object_head = self._make_layer(Bottleneck2D, 512, 3, 2, 1)


        # Pooling method
        if self.pooling == 'rnn':
            cnn_features_size = 2048
            hidden_state_size = 512
            self.rnn = nn.GRU(input_size=cnn_features_size,
                              hidden_size=hidden_state_size,
                              num_layers=1,
                              batch_first=True)
        # Init of the weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_features_map(self, x, time=None , out_dim=None):
        if out_dim is None:
            out_dim = self.out_dim

        if self.time is None:
            B, C, T, W, H = x.size()
            self.time = T

        time = self.time

        # 5D -> 4D if 2D conv at the beginning
        x = transform_input(x, self.input_dim, T=time)
        f = self.backbone(x)

        return transform_input(f['res4'], out_dim, T=time), transform_input(f['res5'], out_dim, T=time)

    def get_two_heads_feature_maps(self, x, T=None, out_dim=None, heads_type='object+context'):
        x = x['clip']  # (B, C, T, W, H)

        # Get the before last feature map
        res4, res5 = self.get_features_map(x, T)

        # Object head
        if 'object' in heads_type:
            fm_objects = layer(fm_objects)
            fm_objects = transform_input(fm_objects, out_dim, T=T)
        else:
            fm_objects = None

        # Activity head
        if 'context' in heads_type:

            fm_context = layer(fm_context)
            fm_context = transform_input(fm_context, out_dim, T=T)
        else:
            fm_context = None

        return fm_context, fm_objects

    def _make_layer(self, block=Bottleneck2D, planes, blocks, stride=2, dilation=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            if block is BasicBlock2D or block is Bottleneck2D:
                conv, batchnorm = nn.Conv2d, nn.BatchNorm2d
            else:
                conv, batchnorm = nn.Conv3d, nn.BatchNorm3d

            downsample = nn.Sequential(
                conv(self.inplanes, planes * block.expansion,
                     kernel_size=1, stride=stride, bias=False, dilation=dilation),
                batchnorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, dilation, nb_temporal_conv=self.nb_temporal_conv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nb_temporal_conv=self.nb_temporal_conv))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x['clip']

        x = self.get_features_map(x, num=self.num_final_fm)

        # Global average pooling
        if self.pooling == 'avg':
            self.avgpool = nn.AvgPool3d((x.size(2), x.size(-1), x.size(-1))) if self.avgpool is None else self.avgpool
            x = self.avgpool(x)
        elif self.pooling == 'rnn':
            self.avgpool_space = nn.AvgPool3d(
                (1, x.size(-1), x.size(-1))) if self.avgpool_space is None else self.avgpool_space
            x = self.avgpool_space(x)
            x = x.view(x.size(0), x.size(1), x.size(2))  # (B,D,T)
            x = x.transpose(1, 2)  # (B,T,D)
            ipdb.set_trace()
            x, _ = self.rnn(x)  # (B,T,D/2)
            x = torch.mean(x, 1)  # (B,D/2)

        # Final classif
        x = x.view(x.size(0), -1)
        # x = self.fc_classifier(x)

        return x
