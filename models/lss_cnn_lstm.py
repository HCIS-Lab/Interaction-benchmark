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
        # self.backbone = get_maskformer()
        self.conv1 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(256, 128, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False),
                nn.Conv2d(128, 128, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False),
                nn.Conv2d(128, 64, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(64)
                    )

        self.conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(64 , 40, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True),
                nn.Conv2d(40, 32, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
                    )

        self.en_lstm = nn.LSTM(input_size=32, hidden_size=16, num_layers=1, batch_first=True)

        # self.en_lstm = ConvLSTM(256, 256, 3, 1, True, True, True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.head = Head(16, num_ego_class, num_actor_class)

        if self.road:
            self.road_enc = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(512, 256, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(256, 128, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=False)
                        )

            self.road_fc = Road_Head(128)
            # self.fusion = nn.Sequential(
            #     nn.ReLU(inplace=False),
            #     nn.Conv2d(512, 256, kernel_size=1, stride=1, padding='same'),
            #     nn.BatchNorm2d(256),
            #     nn.ReLU(inplace=False),
            #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding='same'),
            #     nn.BatchNorm2d(256),
            #     )
            self.fusion = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(256, 128),
                nn.ReLU(inplace=False),
                nn.Linear(128, 128)
                )


    def train_forward(self, x):

        hidden, hidden_road = None, None
        seq_len = len(x)//self.num_cam

        batch_size = x[0].shape[0]
        h, w = x[0].shape[2], x[0].shape[3]

        out_road = []
        for i in range(seq_len):
            x_i = x[i*self.num_cam : i*self.num_cam + self.num_cam]
            if isinstance(x_i, list):
                x_i = torch.stack(x_i, dim=0) #[v, b, 2048, h, w]
                x_i = torch.permute(x_i, (1,0,2,3,4)) #[b, v, 2048, h, w]
                x_i = torch.reshape(x_i, (batch_size, 2048*self.num_cam, h, w)) #[b, 2048*3, h, w]

            x_i = self.conv1(x_i)
            x_feature_i = self.conv2(x_i)
            

            x_feature_i = self.pool(x_feature_i)
            x_feature_i = x_feature_i.view(batch_size, 1, 256)

            if self.road:
                x_road_i = self.road_enc(x_i)
                x_road_i = self.pool(x_road_i)
                x_road_i = x_road_i.view(batch_size, 128)
                out_road.append(x_road_i)
            out, hidden = self.en_lstm(x_feature_i, hidden)

        out = out[:, -1, :]
        if self.road:
            out_road = torch.stack(out_road, 0) #[len, batch, 256]
            out_road = torch.permute(out_road, (1, 0, 2)) #[batch, len, 256]
            out_road_feature = torch.mean(out_road, dim=1) #[batch, 256]
            out = torch.cat((out, out_road_feature), dim=1)
            out = self.fusion(out)

        ego, actor = self.head(out)

        if self.road:
            out_road = torch.reshape(out_road, (batch_size*seq_len, -1))
            out_road = self.road_fc(out_road)
            return ego, actor, out_road
        else:
            return ego, actor


    def forward(self, bev_f):
        batch_size = bev_f[0].shape[0]
        seq_len = len(bev_f)
        w, h = bev_f[0].shape[1], bev_f[0].shape[2]

        for t in range(seq_len):
            x = bev_f[t]
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            print(x.shape)
            out, hidden = self.en_lstm(x.view(batch_size, 1, 256), hidden)

        ego, actor = self.head(out[:, -1, :])
        return ego, actor


    
