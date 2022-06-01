import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from MaskFormer.demo.demo import get_maskformer
from retrieval_head import Head


class CNNLSTM(nn.Module):
    def __init__(self, num_cam, num_ego_class, num_actor_class):
        super(CNNLSTM, self).__init__()

        self.num_cam = num_cam
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.avgpool = nn.Sequential()
        self.backbone.fc = nn.Sequential()
        self.conv1 = nn.Sequential(
                  nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1,),
                  nn.BatchNorm2d(1024),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=1,),
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True)
                    )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.num_cam, 256)

        self.en_lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.relu = nn.ReLU(inplace=True)

        self.head = Head(128, num_ego_class, num_actor_class)

    def backbone_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def train_forward(self, inputs, tops=False, front_only=True):
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
            x = self.backbone_features(x)
            x = self.conv1(x)
            x = self.avgpool(x)
            x = x.view(batch_size, 1, 512*self.num_cam)
            x = self.fc(x)
            x = self.relu(x)

            out, hidden = self.en_lstm(x, hidden)

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
            x = self.backbone_features(x)
            x = self.conv1(x)
            x = self.avgpool(x)
            x = x.view(batch_size, 1, 512*self.num_cam)
            x = self.fc(x)
            x = self.relu(x)

            out, hidden = self.en_lstm(x, hidden)

        _, actor = self.head(out[:, -1, :])
        return actor

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
