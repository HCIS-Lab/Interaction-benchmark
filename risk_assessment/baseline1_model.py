import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import numpy as np
import fnmatch
import cv2

__all__ = [
    'Baseline_Jinkyu',
]

class SADataset(Dataset):
    def __init__(self, root,backbone=True,training=True):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        # root : /mnt/sdb/Dataset/SA/
#        if training:
#            root = os.path.join(root,"training")
#        else:
#            root = os.path.join(root,"testing")
        
        self.datas_path = []
        # self.datas_path = []
        self.backbone = backbone
        if self.backbone:
            # positive or negative
            self.labels = []
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4222, 0.4162, 0.4185),(0.2900, 0.2839, 0.2778)),
            ])
            folders = os.listdir(root)
            for pos_neg in folders:
                if pos_neg == 'negative':
                    label = False
                else:
                    label = True
                for file in os.listdir(os.path.join(root,pos_neg)):
                    self.datas_path.append(os.path.join(root,pos_neg,file))
                    self.labels.append(label)

        else:
            files_list = os.listdir(root)
            paths = ['training','testing']
            # diff_files_list = fnmatch.filter(os.listdir(root), "batch_baseline1*")
            # files_list = list(set(files_list) - set(diff_files_list))
            for path in paths:
                files_list = os.listdir(os.path.join(root,path))
                for file in files_list:
                    self.datas_path.append(os.path.join(root, path,file))
        

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform)
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        path = self.datas_path[index]
        if self.backbone:
            features = []
            cap = cv2.VideoCapture(path)
            ret, img = cap.read()
            while(ret):
                img = cv2.resize(img,(640,320),interpolation=cv2.INTER_AREA)
                img = self.transform(img)
                features.append(img)
                ret, img = cap.read()
            labels = self.labels[index]
            labels = torch.tensor(labels).to('cuda')
            file_name = path
            features = torch.stack(features).to('cuda')
        else:
            data = np.load(path)
            features = data['detectron'][0]
            labels = data['label']
            file_name = os.path.join(path,str(data['file_name']))
        return features,labels,file_name

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.datas_path)


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
        # targets = targets.long() # convert to 0 or 1
        # outputs = outputs.permute(1,0,2) #txbx2
        loss = torch.tensor(0.0).to('cuda')
        for i,pred in enumerate(outputs):
            #bx2
            temp_loss = self.cel(pred,targets) # b
            exp_loss = torch.multiply(torch.exp(torch.tensor(-max(0,self.time-i-1) / 20.0)), temp_loss)
            # if postive => exp loss
            exp_loss = torch.multiply(exp_loss,targets)
            loss = torch.add(loss,torch.mean(torch.add(temp_loss,exp_loss)))
        return loss



class Baseline_Jinkyu(nn.Module):
    def __init__(self, time_steps=100, backbone = True):
        super(Baseline_Jinkyu, self).__init__()

        self.hidden_size = 256
        if not backbone:
            self.D = 256 # channel
        self.D2 = 64 # reduce channel
        self.height = 10
        self.width = 20
        self.L = self.height*self.width
        self.fusion_size = self.D2*self.L
        self.time_steps = time_steps
        self.relu = nn.ReLU(inplace=True)
        self.backbone = None
        if not backbone:
            self.conv = nn.Sequential(
                nn.Conv2d(self.D,self.D2, kernel_size=1),
                nn.BatchNorm2d(self.D2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.3),
            )
        else:
            self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2 ),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.init_h = nn.Sequential(
            nn.Linear(self.D2,self.hidden_size),
            nn.Tanh()
        )
        self.init_c = nn.Sequential(
            nn.Linear(self.D2,self.hidden_size),
            nn.Tanh()
        )
        #InceptionResNetV2(num_classes=1001)

        # self.bn1 = nn.BatchNorm1d(self.D2)

        self.project_w = nn.Sequential(
            nn.BatchNorm1d(self.D2),
            nn.Linear(self.D2, self.D2),
            # nn.BatchNorm1d(self.D2),
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.w_out = nn.Linear(self.hidden_size, self.D2)
        self.w_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_ctx2out = nn.Linear(self.fusion_size, self.hidden_size)
        self.w = nn.Linear(self.hidden_size, self.D2)
        # self.w = nn.Sequential(
        #     nn.Linear(self.H, self.D2),
        #     nn.BatchNorm1d(self.D2),
        #     nn.ReLU(inplace=True),
        # )
        self.w_attn = nn.Linear(self.D2, 1)
        self.drop = nn.Dropout(p=0.5)
        self.lstm = nn.LSTMCell(self.fusion_size, self.hidden_size)


        self.collision_pred = nn.Linear(self.hidden_size, 2),
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            # elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            #     nn.init.normal_(m.weight,mean=1.0,std=0.001)
            #     m.bias.data.fill_(0.001)
        # for m in self.modules():
        #     classname = m.__class__.__name__
        #     if classname.find('BasicConv2d') != -1:
        #         pass
        #     elif classname.find('Conv') != -1:
        #         m.weight.data.normal_(0.0, 0.001)
        #     elif classname.find('BatchNorm') != -1:
        #         # m.weight.data.normal_(1.0, 0.001)
        #         # m.bias.data.fill_(0.001)
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()


    def normalization(self, camera_inputs):
        camera_inputs = camera_inputs.view(-1,self.D2)
        camera_inputs = self.bn1(camera_inputs)
        return camera_inputs

    def decode_lstm(self, h, context):
        h = self.drop(h)
        h_logits = self.w_h(h)
        h_logits += self.w_ctx2out(context)
        h_logits = self.drop(h_logits)
        out_logits = self.w_out(h_logits)
        return out_logits

    def project_feature(self, feature):
        feature_flat = feature.view(-1, self.D2)
        feature_proj = self.project_w(feature_flat)
        feature_proj = feature_proj.view(-1, self.L, self.D2)

        return feature_proj

    def attntion_layer(self, features, features_proj, h):
        ####################################
        # features: b,1,L,D
        # features_proj: b,1,L,D
        ####################################
        #print(self.w(h).shape)
        h_attn = self.relu(features_proj+torch.unsqueeze(self.w(h),1)) # b,L,D
        out_attn = self.drop(self.w_attn(h_attn.view(-1, self.D2)).view(-1, self.L)) #b,L
        alpha = nn.functional.softmax(out_attn,dim=1) #b,L
        # alpha_logp = nn.functional.log_softmax(out_attn,dim=1)
        context = (features*torch.unsqueeze(alpha,2)).view(-1,self.L*self.D2)# b,D

        return context, alpha #, alpha_logp

    def forward(self, camera_inputs, device='cuda'):
        
        batch_size, t, self.D, h, w = camera_inputs.shape
        camera_inputs = camera_inputs.view(-1,self.D,h,w)
        if self.backbone:
            camera_inputs = self.backbone(camera_inputs)
        else:
            camera_inputs = self.conv(camera_inputs)
        # initialize LSTM
        x0 = camera_inputs.view(batch_size,t,self.L,self.D2)[:,0].sum(dim=1)/self.L
        hx = self.init_h(x0)
        cx = self.init_c(x0)
        collision_stack = []
        attention_stack = []
        # attention_log_stack = []

        # camera_inputs = self.normalization(camera_inputs)

        features_proj = self.project_feature(camera_inputs) #(bxt, L, D)
        
        camera_inputs = camera_inputs.view(batch_size,t,self.L, self.D2)
        features_proj = features_proj.view(batch_size,t,self.L, self.D2)
        for l in range(self.time_steps):
            features_curr =  camera_inputs[:,l]
            features_proj_curr = features_proj[:,l]
            context, alpha = self.attntion_layer(features_curr,features_proj_curr,hx)
            hx, cx = self.lstm(context, (hx, cx))
            pred = self.decode_lstm(hx, context)
            collision_stack.append(pred)
            attention_stack.append(alpha)
            # attention_log_stack.append(alpha_logp)
        collision_stack = torch.stack(collision_stack)
        attention_stack = torch.stack(attention_stack)
        # attention_log_stack = torch.stack(attention_log_stack)
        return collision_stack, attention_stack #, attention_log_stack
