import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from MaskFormer.demo.demo import get_maskformer
from retrieval_head import Head



class CNNLSTM(nn.Module):
    def __init__(self, num_cam):
        super(CNNLSTM, self).__init__()
        self.num_cam = num_cam
        self.backbone = get_maskformer().backbone

        self.conv1 = nn.Sequential(
                  nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1,),
                  nn.BatchNorm2d(1024),
                  nn.ReLU(inplace=True)
                    )

        self.conv2 = nn.Sequential(
                  nn.Conv2d(1024*self.num_cam, 256, kernel_size=1, stride=1, padding=1),
                  nn.BatchNorm2d(256),
                  nn.ReLU(inplace=True)
                    )

        self.en_lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.head = Head(256, 4, 25)

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
        
