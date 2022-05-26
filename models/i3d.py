import torch
# Choose the `slow_r50` model 

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

        # self.backbone = get_maskformer().backbone
        self.num_cam = num_cam
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

        self.fc = nn.Linear(2048*self.num_cam, 1024)

        self.en_lstm = nn.LSTM(input_size=1024, hidden_size=256, num_layers=1, batch_first=True)
        self.relu = nn.ReLU(inplace=True)

        self.head = Head(256, num_ego_class, num_actor_class)

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
            x = self.backbone(x)

            x = x.view(batch_size, 1, 2048*self.num_cam)
            x = self.fc(x)
            x = self.relu(x)

            out, hidden = self.en_lstm(x, hidden)

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
