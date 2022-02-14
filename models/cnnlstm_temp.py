import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

import backbone.get_maskrcnn_backbone

class CNNLSTM_temp(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM_temp, self).__init__()
        self.trg_vocab_size = 38
        self.resnet = get_maskrcnn_backbone()

        self.resnet.fc = nn.Sequential(nn.Linear(2048,300))
        self.en_lstm = nn.LSTM(input_size=900, hidden_size=512, num_layers=1, batch_first=True)
        self.de_lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, batch_first=True)
        self.emb = nn.Linear(1, 512)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, self.trg_vocab_size)
    def forward(self, x_3d_front, x_3d_right, x_3d_left, trg):
        hidden = None
        for t in range(x_3d_front.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d_front[:, t, :, :, :]) 
                x_left = self.resnet(x_3d_left[:, t, :, :, :])  
                x_right = self.resnet(x_3d_right[:, t, :, :, :])  
                x = torch.cat((x, x_left, x_right), dim=1)
            _, hidden = self.en_lstm(x.unsqueeze(1), hidden)

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        outputs = torch.zeros(batch_size, trg_len, self.trg_vocab_size).to(self.device)
        input = trg[:, 0]
        
        for i in range(1, trg_len):
            input = self.emb(input)
            out, hidden = self.de_lstm(input, hidden)
            out = self.fc1(out)
            out = F.relu(out)
            out = self.fc2(out)
            outputs[t] = out
            top1 = out.argmax(1) 
            input = top1
        return outputs
