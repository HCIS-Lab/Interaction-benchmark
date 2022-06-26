import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from MaskFormer.demo.demo import get_maskformer
from retrieval_head import Head, Road_Head



class FPNLSTM(nn.Module):
    def __init__(self, num_cam, num_ego_class, num_actor_class, road):
        super(FPNLSTM, self).__init__()
        self.num_cam = num_cam
        self.road = road
        self.backbone = get_maskformer()

        self.conv1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 320, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(320),
                nn.ReLU(inplace=True),
                nn.Conv2d(320, 384, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 512, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1536, kernel_size=5, stride=1, padding='same'),
                nn.BatchNorm2d(1536),
                nn.ReLU(inplace=True),
                nn.Conv2d(1536*self.num_cam, 2048, kernel_size=5, stride=1, padding='same'),
                nn.BatchNorm2d(2048)
                    )

        self.en_lstm = nn.LSTM(input_size=1024, hidden_size=512, num_layers=1, batch_first=True)
        self.en_road_lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=1, batch_first=True)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.head = Head(512, num_ego_class, num_actor_class)

        if self.road:
            self.road_enc = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(256, 320, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(320),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(320, 384, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(384),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(384, 512, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(1024, 1536, kernel_size=5, stride=1, padding='same'),
                    nn.BatchNorm2d(1536),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(1536*self.num_cam, 2048, kernel_size=5, stride=1, padding='same'),
                    nn.BatchNorm2d(2048)
                        )
            self.road_fc = Road_Head(512, 20)
            self.fusion = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(4096, 2048, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(1024)
                )

    def train_forward(self, x, tops=False, front_only=True):
        hidden, hidden_road = None, None
        seq_len = len(x)//self.num_cam
        batch_size = x[0].shape[0]

        w, h = x[0].shape[2], x[0].shape[3]

        if isinstance(x, list):
            x = torch.stack(x, dim=0)

        x = x.view(batch_size*self.num_cam*seq_len, 3, w, h)
        # x = normalize_imagenet(x)
        x = (x - self.backbone.pixel_mean) / self.backbone.pixel_std
        x = self.backbone.get_fpn_features(x, no_dict=True)[-1]
        # print(x.shape)
        x_feature = self.conv1(x)
        
        if self.road:
            x_road_feature = self.road_enc(x)

            x_feature = torch.cat((x_feature, x_road_feature), dim=1)
            x_feature = self.fusion(x_feature)

            x_road_feature = self.pool(x_road_feature)
            x_road_feature = torch.flatten(x_road_feature, 1)
            x_road_feature = x_road_feature.view(batch_size*self.num_cam, seq_len, 2048)
           
        x_feature = self.pool(x_feature)
        x_feature = torch.flatten(x_feature, 1) 
        x_feature = x_feature.view(batch_size*self.num_cam, seq_len, 1024)

        # print(x_feature.shape)

        for t in range(seq_len):
            x_t = x_feature[:, t, :].view(batch_size, 1, 1024)
            out, hidden = self.en_lstm(x_t, hidden)

            x_road_t = x_road_feature[:, t, :].view(batch_size, 1, 2048)
            out_road, hidden_road = self.en_road_lstm(x_road_t, hidden_road)
        # for t in range(seq_len):
        #     x = inputs[t]

        #     if isinstance(x, list):
        #         x = torch.stack(x, dim=0)
        #     x = x.view(batch_size*self.num_cam, 3, w, h)
        #     x = normalize_imagenet(x)
        #     x = self.backbone(x)['res5']
        #     x = self.conv1(x)
        #     x = self.conv2(x)
        #     x = self.pool(x)
        #     x = torch.flatten(x, 1)

            # out, hidden = self.en_lstm(x.view(batch_size, 1, 256), hidden)
        ego = self.head(out[:, -1, :])
        if self.road:
            out_road = self.road_fc(out_road[:, -1, :])
            return ego, out_road

        else:
            return ego, actor

    def forward(self, fronts, lefts, rights, tops=False):
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

        
