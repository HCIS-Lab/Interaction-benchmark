import enum
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import numpy as np
import fnmatch

__all__ = [
    'SA',
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
        files_list = os.listdir(root)
        diff_files_list = fnmatch.filter(os.listdir(root), "batch_baseline1*")
        files_list = list(set(files_list)-set(diff_files_list))
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
        features = data['data_flat'][0]
        labels = data['label']
        file_name = data['file_name']
        bboxs = data['bboxes']
        return features,labels,file_name,bboxs

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



class Baseline_SA(nn.Module):
    def __init__(self,with_frame ,time_steps=100,features_size=1024,frame_features_size=1024,hidden_layer_size=256,lstm_size=512):
        super(Baseline_SA, self).__init__()
        self.time_steps = time_steps
        self.features_size = features_size
        self.with_frame = with_frame
        if self.with_frame:
            self.frame_features_size = frame_features_size
        self.hidden_layer_size = hidden_layer_size
        self.lstm_layer_size = lstm_size
        if self.with_frame:
            self.frame_layer = nn.Sequential(
                nn.Linear(self.frame_features_size, self.hidden_layer_size),
                nn.ReLU(inplace=True)
            )
        self.object_layer = nn.Sequential(
            nn.Linear(self.features_size, self.hidden_layer_size),
            nn.ReLU(inplace=True)
        )
        self.fusion_size = 2*self.hidden_layer_size if self.with_frame else self.hidden_layer_size
        self.drop = nn.Dropout(p=0.5)
        self.lstm = nn.LSTMCell(self.fusion_size, self.lstm_layer_size)
        self.output_layer = nn.Sequential(
            nn.Linear(self.lstm_layer_size, 2),
            nn.ReLU(inplace=True),

        )
        # self.att_w = torch.normal(mean = 0,std = 0.01,size=(self.hidden_layer_size,1),requires_grad = True).to('cuda')
        # self.att_wa = torch.normal(mean = 0,std = 0.01,size=(self.lstm_layer_size,self.hidden_layer_size),requires_grad = True).to('cuda')
        # self.att_ua = torch.normal(mean = 0,std = 0.01,size=(self.hidden_layer_size,self.hidden_layer_size),requires_grad = True).to('cuda')
        # self.att_ba = torch.zeros(self.hidden_layer_size,requires_grad = True).to('cuda')
        self.att_w = nn.Parameter(torch.normal(mean = 0,std = 0.01,size=(self.hidden_layer_size,1)),requires_grad = True)#.to('cuda')
        self.att_wa = nn.Parameter(torch.normal(mean = 0,std = 0.01,size=(self.lstm_layer_size,self.hidden_layer_size)),requires_grad = True)#.to('cuda')
        self.att_ua = nn.Parameter(torch.normal(mean = 0,std = 0.01,size=(self.hidden_layer_size,self.hidden_layer_size)),requires_grad = True)#.to('cuda')
        self.att_ba = nn.Parameter(torch.zeros(self.hidden_layer_size),requires_grad = True)#.to('cuda')

    def step(self, fusion, hx, cx):

        hx, cx = self.lstm(self.drop(fusion), (hx, cx))

        return self.output_layer(hx), hx, cx

    def attention_layer(self, object, h_prev):
        brcst_w = torch.tile(torch.unsqueeze(self.att_w, 0), (20,1,1)) # n x h x 1
        image_part = torch.matmul(object, torch.tile(torch.unsqueeze(self.att_ua, 0), (20,1,1))) + self.att_ba # n x b x h
        e = torch.tanh(torch.matmul(h_prev,self.att_wa)+image_part) # n x b x h
        return brcst_w,e

    def forward(self, input_features):#, input_frame):
        # features: b,t,20,C
        batch_size = input_features.size()[0]
        hx = torch.zeros((batch_size, self.lstm_layer_size)).to('cuda')
        cx = torch.zeros((batch_size, self.lstm_layer_size)).to('cuda')
        out = []
        zeros_object =  torch.sum(input_features.permute(1,2,0,3),3).eq(0).float().contiguous()# frame x n x b
        for i in range(self.time_steps):
            # img = input_frame[:,i].view(-1,self.frame_features_size)
            object = input_features[:,i].permute(1,0,2).contiguous() # nxbxc
            object = object.view(-1,self.features_size).contiguous() # (nxb)xc
            # img = self.frame_layer(img)
            object = self.object_layer(object) #(nxb)xh
            object = object.view(20,batch_size,self.hidden_layer_size)
            # object = torch.matmul(object,torch.unsqueeze(zeros_object[i],2))
            object = object*torch.unsqueeze(zeros_object[i],2)
            brcst_w,e = self.attention_layer(object,hx)
            alphas = torch.mul(nn.functional.softmax(torch.sum(torch.matmul(e,brcst_w),2),0),zeros_object[i])
            attention_list = torch.mul(torch.unsqueeze(alphas,2),object)
            attention = torch.sum(attention_list,0) # b x h
            # concat frame & object
            # fusion = torch.cat((img,attention),1)
            pred,hx,cx = self.step(attention,hx,cx)
            out.append(pred)
            if i == 0:
                soft_pred = nn.functional.softmax(pred,dim=1)
                all_alphas = torch.unsqueeze(alphas,0)
            else:
                temp_soft_pred = nn.functional.softmax(pred,dim=1)
                soft_pred = torch.cat([soft_pred,temp_soft_pred],1)
                temp_alphas = torch.unsqueeze(alphas,0)
                all_alphas = torch.cat([all_alphas, temp_alphas],0)
        out_stack = torch.stack(out)
        soft_pred = soft_pred.view(batch_size,self.time_steps,-1)
        all_alphas = all_alphas.permute(2,0,1)
        return soft_pred, all_alphas, out_stack