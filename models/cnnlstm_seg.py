import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from MaskFormer.demo.demo import get_maskformer
from retrieval_head import Head
import math

class CNNLSTM(nn.Module):
    def __init__(self, num_cam):
        super(CNNLSTM, self).__init__()

        # self.backbone = get_maskformer().backbone
        self.num_cam = num_cam
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.avgpool = nn.Sequential()
        self.backbone.fc = nn.Sequential()

        self.backbone_conv = nn.Sequential(
                  nn.Conv2d(2048, 512, kernel_size=3, stride=2, padding=1,),
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True)
                  )
        self.conv1 = nn.Sequential(
                  nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1,),
                  nn.BatchNorm2d(1024),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=1,),
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True)
                    )
        self.seg = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        self.r18 = ResNet(Bottleneck, [3, 4, 6, 3])

        self.fc = nn.Linear(2048*self.num_cam, 1024)

        self.en_lstm = nn.LSTM(input_size=1024, hidden_size=256, num_layers=1, batch_first=True)
        self.relu = nn.ReLU(inplace=True)

        self.head = Head(256, 4, 48)

    def forward(self, fronts, lefts, rights, tops):
        hidden = None
        
        if not tops:
            batch_size = fronts[0].shape[0]
            seq_len = len(fronts)
            w, h = fronts[0].shape[2], fronts[0].shape[3]
        else:
            batch_size = tops[0].shape[0]
            seq_len = len(tops)
            w, h = tops[0].shape[2], tops[0].shape[3]

        for t in range(seq_len):
            x = []
            if not tops:
                x.append(fronts[t])
                x.append(lefts[t])
                x.append(rights[t])
            else:

                x.append(tops[t])
            x = torch.stack(x, dim=0).view(batch_size*self.num_cam, 3, w, h)
            x = normalize_imagenet(x)

            # print(self.seg(x)['out'].shape)
            y = self.backbone(x) + self.r18(self.seg(x)['out'])

            y = y.view(batch_size, 1, 2048*self.num_cam)
            y = self.fc(y)
            y = self.relu(y)

            out, hidden = self.en_lstm(y, hidden)

        ego, actor = self.head(out[:, -1, :])
        return ego, actor

    def forward(self, inputs, tops=False, front_only=True):
        hidden = None
        seq_len = len(inputs)//self.num_cam
        batch_size = inputs[0].shape[0]

        w, h = inputs[0].shape[2], inputs[0].shape[3]

        for t in range(seq_len):
            x = inputs[t]

            if isinstance(x, list):
                x = torch.stack(x, dim=0)
            x.view(batch_size*self.num_cam, 3, w, h)
            x = normalize_imagenet(x)
            x = self.backbone_features(x) + self.r18(self.seg(x)['out'])
            x = self.conv1(x)
            x = self.avgpool(x)
            x = x.view(batch_size, 1, 512*self.num_cam)
            x = self.fc(x)
            x = self.relu(x)

            out, hidden = self.en_lstm(x, hidden)

        ego, actor = self.head(out[:, -1, :])
        return ego, actor


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, conv=None, norm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_groups=None, weight_std=False, beta=False):
        self.inplanes = 64
        self.norm = lambda planes, momentum=0.05: nn.BatchNorm2d(planes, momentum=momentum) if num_groups is None else nn.GroupNorm(num_groups, planes)
        self.conv = Conv2d if weight_std else nn.Conv2d

        super(ResNet, self).__init__()
        if not beta:
            self.conv1 = self.conv(21, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Sequential(
                self.conv(21, 64, 3, stride=2, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False))
        self.bn1 = self.norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilation=2)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        for m in self.modules():
            if isinstance(m, self.conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=max(1, dilation/2), bias=False),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=max(1, dilation/2), conv=self.conv, norm=self.norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, conv=self.conv, norm=self.norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        # size = (x.shape[2], x.shape[3])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x
