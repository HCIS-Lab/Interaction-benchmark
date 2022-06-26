import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from MaskFormer.demo.demo import get_maskformer
from retrieval_head import Head, Road_Head



class CNNPOOL(nn.Module):
    def __init__(self, num_cam, num_ego_class, num_actor_class, road):
        super(CNNPOOL, self).__init__()
        self.num_cam = num_cam
        self.backbone = get_maskformer().backbone

        self.conv1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1,),
                nn.BatchNorm2d(1024) 
                )

        self.conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(1024*self.num_cam, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256)  
                )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = Head(256, num_ego_class, num_actor_class)

        if road:
            self.road_enc = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1,),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=False),
                nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1,),
                nn.BatchNorm2d(256) 
                )
            self.road_fc = Road_Head(256, 20)
            self.fusion = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,),
                nn.BatchNorm2d(256) 
                )
        else:
            self.road_enc = None
    def train_forward(self, x, tops=False, front_only=True):
        seq_len = len(x)//self.num_cam
        batch_size = x[0].shape[0]

        h, w = x[0].shape[2], x[0].shape[3]

        if isinstance(x, list):
            x = torch.stack(x, dim=0)
        x = torch.permute(x, (1, 0, 2, 3, 4))
        
        x = x.contiguous().view(seq_len*batch_size, 3, h, w)
        x = normalize_imagenet(x)
        x = self.backbone(x)['res5']
        out_h, out_w = x.shape[2], x.shape[3]
        x = x.view(batch_size, seq_len, 2048, out_h, out_w)
        x = torch.mean(x, dim=1)

        if self.road_enc:
            x_road_feature = self.road_enc(x)
            x_road = self.avgpool(x_road_feature)
            x_road = torch.flatten(x_road, 1)
            x_road = self.road_fc(x_road)

        x = self.conv1(x)
        x = self.conv2(x)

        if self.road_enc:
            # print(x.shape)
            # print(x_road_feature.shape)
            x = torch.cat((x, x_road_feature), dim=1)
            x = self.fusion(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        ego, actor = self.head(x)

        if self.road_enc:
            return ego, actor, x_road
        else:
            return ego, actor

    def forward(self, fronts, lefts, rights, tops=False):
        hidden = None
        if not tops:
            batch_size = fronts[0].shape[0]
            seq_len = len(fronts)
            h, w = fronts[0].shape[2], fronts[0].shape[3]
        else:
            batch_size = tops[0].shape[0]
            seq_len = len(tops)
            h, w = tops[0].shape[2], tops[0].shape[3]

        for t in range(seq_len):
            x = []
            if not tops:
                x.append(fronts[t])
                x.append(lefts[t])
                x.append(rights[t])
            else:
                x.append(tops[t])

            x = torch.stack(x, dim=0).view(batch_size*self.num_cam, 3, h, w)
            x = self.backbone(x)['res5']
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)

            out, hidden = self.en_lstm(x.view(batch_size, 1, 256), hidden)

        ego, actor = self.head(out[:, -1, :])
        return ego, actor

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

        
