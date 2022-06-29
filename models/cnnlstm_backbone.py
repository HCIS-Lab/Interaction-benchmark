import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from MaskFormer.demo.demo import get_maskformer
from retrieval_head import Head, Road_Head


class CNNLSTM_maskformer(nn.Module):
    def __init__(self, num_cam, num_ego_class, num_actor_class, road):
        super(CNNLSTM_maskformer, self).__init__()
        self.num_cam = num_cam
        self.road = road
        self.backbone = get_maskformer()

        self.conv1 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(2048*self.num_cam, 1024*self.num_cam, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(1024*self.num_cam),
                nn.ReLU(inplace=False),
                nn.Conv2d(1024*self.num_cam, 1024, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=False),
                nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(512)
                    )

        self.conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 400, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(400),
                nn.ReLU(inplace=True),
                nn.Conv2d(400, 256, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
                    )

        self.en_lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.head = Head(256, num_ego_class, num_actor_class)

        if self.road:
            self.road_enc = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(512, 400, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(400),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(400, 256, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(256)
                        )

            self.road_fc = Road_Head(256)
            self.fusion = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(256),
                )


    def train_forward(self, x, tops=False, front_only=True):

        hidden, hidden_road = None, None
        seq_len = len(x)//self.num_cam
        batch_size = x[0].shape[0]

        h, w = x[0].shape[2], x[0].shape[3]

        out_road = []
        for i in range(seq_len):
            x_i = x[i*self.num_cam : i*self.num_cam + self.num_cam]
            if isinstance(x_i, list):
                x_i = torch.stack(x_i, dim=0)
                x_i = torch.permute(x_i, (1,0,2,3,4))
                x_i = torch.reshape(x_i, (batch_size*self.num_cam, 3, h, w))
                # x_i = x_i.view(batch_size, self.num_cam, 3, h, w)
            with torch.no_grad():
                x_i= (x_i - self.backbone.pixel_mean) / self.backbone.pixel_std
                x_i = self.backbone.backbone(x_i)['res5']
                down_h, down_w = x_i.shape[-2], x_i.shape[-1]
                # x_i = torch.permute(x_i, (1,0,2,3,4))
                x_i = x_i.view(batch_size, -1, down_h, down_w)
            x_i = self.conv1(x_i)
            x_feature_i = self.conv2(x_i)
            

            if self.road:
                x_road_i = self.road_enc(x_i)
                x_feature_i = torch.cat((x_feature_i, x_road_i), dim=1)
                x_feature_i = self.fusion(x_feature_i)

            x_feature_i = self.pool(x_feature_i)
            x_feature_i = x_feature_i.view(batch_size, 1, 256)


            if self.road:
                x_road_i = self.pool(x_road_i)
                x_road_i = x_road_i.view(batch_size*self.num_cam, 256)
                out_road.append(x_road_i)
                            # out_road, hidden_road = self.en_road_lstm(x_road_i, hidden_road)
            out, hidden = self.en_lstm(x_feature_i, hidden)


        ego, actor = self.head(out[:, -1, :])
        if self.road:
            out_road = torch.stack(out_road, 0)
            out_road = out_road.view(batch_size*seq_len, 256)
            out_road = self.road_fc(out_road)
            # out_road = self.road_fc(out_road[:, -1, :])
            return ego, actor, out_road
        else:
            return ego, actor



        # x = x.view(batch_size*self.num_cam*seq_len, 3, w, h)
        # # x = normalize_imagenet(x)
        # with torch.no_grad():
        #     x = (x - self.backbone.pixel_mean) / self.backbone.pixel_std
        #     x = self.backbone.backbone(x)['res5']
        # x_feature = self.conv1(x)
        
        # if self.road:
            
        #     x_road_feature = self.road_enc(x)

        #     x_feature = torch.cat((x_feature, x_road_feature), dim=1)
        #     x_feature = self.fusion(x_feature)

        #     x_road_feature = self.pool(x_road_feature)
        #     x_road_feature = torch.flatten(x_road_feature, 1)
        #     x_road_feature = x_road_feature.view(batch_size*self.num_cam, seq_len, 256)
           
        # x_feature = self.pool(x_feature)
        # # print(x_feature.shape)
        # # x_feature = torch.flatten(x_feature, 2) 
        # x_feature = x_feature.view(batch_size*self.num_cam, seq_len, 256)


        # out, _ = self.en_lstm(x_feature, hidden_road)
        # # print(lat[0].shape)

        # if self.road:
        #     out_road, _ = self.en_road_lstm(x_road_feature, hidden_road)



        # # for t in range(seq_len):
        # #     x = inputs[t]

        # #     if isinstance(x, list):
        # #         x = torch.stack(x, dim=0)
        # #     x = x.view(batch_size*self.num_cam, 3, w, h)
        # #     x = normalize_imagenet(x)
        # #     x = self.backbone(x)['res5']
        # #     x = self.conv1(x)
        # #     x = self.pool(x)
        # #     x = torch.flatten(x, 1)

        # #     out, hidden = self.en_lstm(x.view(batch_size, 1, 256), hidden)


        # ego, actor = self.head(out[:, -1, :])
        # if self.road:
        #     out_road = self.road_fc(out_road[:, -1, :])
        #     return ego, actor, out_road

        # else:
        #     return ego, actor

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

        
