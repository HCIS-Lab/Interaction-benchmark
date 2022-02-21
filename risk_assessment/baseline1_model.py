import enum
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import numpy as np
import fnmatch

__all__ = [
    'Baseline_Jinkyu',
]

class SADataset(Dataset):
    def __init__(self, root,training=True):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        # root : /mnt/sdb/Dataset/SA/
        if training:
            root = os.path.join(root,"training")
        else:
            root = os.path.join(root,"testing")
        self.datas_path = []
        files_list = fnmatch.filter(os.listdir(root), "batch_baseline1*")
        for file in files_list:
            self.datas_path.append(os.path.join(root,file))
        

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform)
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        path = self.datas_path[index]
        data = np.load(path)
        features = data['data'][0]
        labels = data['label']
        file_name = data['file_name']
        return features,labels,file_name

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.datas_path)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class custom_loss(nn.Module):
    def __init__(self, time):
        # --------------------------------------------
        # Initialization
        # --------------------------------------------
        super(custom_loss, self).__init__()
        self.cel =  nn.CrossEntropyLoss(reduction = 'none')
        self.time = time

    def forward(self, outputs, targets):
        # targets: b (True or false)
        # outputs: txbx2
        targets = targets.long() # convert to 0 or 1
        # outputs = outputs.permute(1,0,2) #txbx2
        loss = torch.tensor(0.0).to('cuda')
        for i,pred in enumerate(outputs):
            #bx2
            temp_loss = self.cel(pred,targets) # b
            exp_loss = torch.multiply(torch.exp(torch.tensor(-(self.time-i-1)/20.0)),temp_loss)
            # if postive => exp loss
            exp_loss = torch.multiply(exp_loss,targets)
            loss = torch.add(loss,torch.mean(torch.add(temp_loss,exp_loss)))
        return loss



class Baseline_Jinkyu(nn.Module):
    def __init__(self, inputs, time_steps=100):
        super(Baseline_Jinkyu, self).__init__()

        self.hidden_size = 512
        self.D = 256
        self.L = 240
        self.H = 512
        if inputs in ['camera', 'sensor', 'both']:
            self.with_camera = 'sensor' not in inputs
            self.with_sensor = 'camera' not in inputs
        else:
            raise(RuntimeError(
                'Unknown inputs of {}, '
                'supported inputs consist of "camera", "sensor", "both"',format(inputs)))
        self.time_steps = time_steps
        # self.pretrained = pretrained

        # self.backbone = nn.Sequential(
        #     nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2 ),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(48, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        # )

        #InceptionResNetV2(num_classes=1001)

        self.project_w = nn.Linear(self.D, self.D)
        self.w = nn.Linear(self.H, self.D)
        self.w_attn = nn.Linear(self.D, 1)

        # self.camera_features = nn.Sequential(
        #     nn.Conv2d(1536, 20, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     Flatten(),
        # )

        if self.with_camera and self.with_sensor:
            raise(RuntimeError('Sensor Data is not Input'))
        elif self.with_camera:
            self.fusion_size = self.D*self.L
        elif self.with_sensor:
            raise(RuntimeError('Sensor Data is not Input'))
        else:
            raise(RuntimeError('Inputs of camera and sensor cannot be both empty'))

        self.drop = nn.Dropout(p=0.1)
        self.lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)


        self.collision_pred = nn.Sequential(
            nn.Linear(self.hidden_size, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2),
        )
        
        # self.vel_regressor= nn.Sequential(
        #     nn.Linear(self.hidden_size, 100),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(100, 50),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(50, 10),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(10, 1),
        # )

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('BasicConv2d') != -1:
                pass
            elif classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.001)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.001)
                m.bias.data.fill_(0.001)


    def step(self, camera_input, hx, cx):

        hx, cx = self.lstm(self.drop(camera_input), (hx, cx))

        return self.collision_pred(self.drop(hx)), hx, cx

    def project_feature(self, feature):
        feature_flat = feature.view(-1, self.D)
        feature_proj = self.project_w(feature_flat)
        feature_proj = feature_proj.view(-1, self.L, self.D)

        return feature_proj

    def attntion_layer(self, features, features_proj, h):
        ####################################
        # features: b,1,L,D
        # features_proj: b,1,L,D
        ####################################
        #print(self.w(h).shape)
        h_attn = torch.tanh(features_proj+torch.unsqueeze(self.w(h),1)) # b,L,D
        out_attn = self.w_attn(h_attn.view(-1, self.D)).view(-1, self.L) #b,L
        alpha = nn.functional.softmax(out_attn,dim=1) #b,L
        alpha_logp = nn.functional.log_softmax(out_attn,dim=1)
        context = (features*torch.unsqueeze(alpha,2)).view(-1,self.L*self.D)# b,D

        return context, alpha, alpha_logp

    def forward(self, camera_inputs, device='cuda'):

        batch_size = camera_inputs.shape[0]
        t = camera_inputs.shape[1]
        # c = camera_inputs.shape[2]
        # w = camera_inputs.shape[3]
        # h = camera_inputs.shape[4]

        # initialize LSTM
        hx = torch.zeros((batch_size, self.hidden_size)).to(device)
        cx = torch.zeros((batch_size, self.hidden_size)).to(device)
        collision_stack = []
        attention_stack = []
        # logit_vel_stack = []


        # camera_inputs = self.backbone(camera_inputs.view(-1,c,w,h)) 
        # camera_inputs = camera_inputs.permute(0,3,1,2).contiguous()

        # camera_inputs :(bs,t,256,24,10)
        camera_inputs = camera_inputs.view(-1,self.D,self.L)
        camera_inputs = camera_inputs.contiguous() #(bxt, self.L, self.D)

        features_proj = self.project_feature(camera_inputs) #(bxt, L, D)


        for l in range(0, self.time_steps):
            features_curr =  camera_inputs.view(batch_size,t,self.L, self.D)[:,l]
            features_proj_curr = features_proj.view(batch_size,t,self.L, self.D)[:,l]
            context, alpha, alpha_logp = self.attntion_layer(features_curr,features_proj_curr,hx)
            pred, hx, cx = self.step(context, hx, cx)
            collision_stack.append(pred)
            attention_stack.append(alpha)
            # logit_vel_stack.append(vel)
        collision_stack = torch.stack(collision_stack)
        # logit_vel_stack = torch.stack(logit_vel_stack).view(-1) #t x batch
        attention_stack = torch.stack(attention_stack)
        return collision_stack, attention_stack#, logit_vel_stack