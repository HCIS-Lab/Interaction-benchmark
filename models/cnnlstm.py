import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from MaskFormer.demo.demo import get_maskformer

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, pretrained_backbone=True):
        super(CNNLSTM, self).__init__()
        self.num_classes = 38

        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1,)
        self.bn1 = nn.BatchNorm2d(512)

        self.en_lstm = nn.LSTM(input_size=900, hidden_size=512, num_layers=1, batch_first=True)
        
        self.fc1 = nn.Linear(64, xxx)
        self.bn2 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(64, self.num_classes)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x_3d_front, x_3d_right, x_3d_left):
        hidden = None
        for t in range(x_3d_front.size(1)):
            x = self.conv1(x)
            x = self.bn(x)
            x = self.relu(x)

            x_left = self.conv1(x_left)
            x_left = self.bn(x_left)
            x_left = self.relu(x_left)

            x_right = self.conv1(x_right)
            x_right = self.bn(x_right)
            x_right = self.relu(x_right)

            x = torch.cat((x, x_left, x_right), dim=1)
            _, hidden = self.en_lstm(x.unsqueeze(1), hidden)

        out = self.fc1(hidden)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc2(hidden)
        return out
        
