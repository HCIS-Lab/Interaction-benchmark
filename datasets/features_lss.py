import os
import json
from PIL import Image
from argparse import ArgumentParser
import threading
import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import json 
sys.path.append('../models')
from lss import LSS
from data_lss import compile_data

device = torch.device('cuda:1')
###LSS
# from pyquaternion import Quaternion
COLOR = np.uint8([
        (0, 0, 0),
        (66, 62, 64),
        (116, 191, 101),
        (255, 255, 255),
        (136, 138, 133),
        (0, 0, 142),
        (220, 20, 60),
        (0, 0, 1)
        ])


def save_feature(features, path_list):
    for i in range(features.shape[0]):
        f = features[i,:,:,:]
        f = f.cpu().detach().numpy()
        np.save(path_list[i], f)

scale=4
root='/data/carla_dataset/data_collection'
save_root='/data/carla_feature/data_collection'

        
scale = float(scale)

type_list = ['interactive']

grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [4.0, 45.0, 1.0]
}

data_aug_conf = {
    'resize_lim': (0.193, 0.225),
    'final_dim': (128, 352),
    'rotate_lim': (-5.4, 5.4),
    'H': 720, 'W': 1280,
    'rand_flip': True,
    'bot_pct_lim': (0.0, 0.22),
    'cams': ['left', 'front', 'right'],
    'Ncams': 3
}

model = LSS(grid_conf, data_aug_conf, outC=6).to(device)
pretrained_dict = torch.load('../models/model139000.pt')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(pretrained_dict)


model.eval()


for t, type in enumerate(type_list):
    basic_scenarios = [os.path.join(root, type, s) for s in os.listdir(os.path.join(root, type))]
    save = os.path.join(save_root, type)
    if not os.path.isdir(save):
        os.mkdir(save)

    # iterate scenarios
    print('searching data')
    for s in tqdm(basic_scenarios, file=sys.stdout):
        # a basic scenario
        scenario_id = s.split('/')[-1]
        if not os.path.isdir(os.path.join(save, scenario_id)):
            os.mkdir(os.path.join(save, scenario_id))
        save_scen = os.path.join(save, scenario_id, 'variant_scenario')
        if not os.path.isdir(save_scen):
            os.mkdir(save_scen)

        # if road_class != 5:
        variants_path = os.path.join(s, 'variant_scenario')
        variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path) if os.path.isdir(os.path.join(variants_path, v))]
        
        for v in variants:
            v_id = v.split('/')[-1]
            save_v = os.path.join(save_scen, v_id)

            if not os.path.isdir(save_v):
                os.mkdir(save_v)
            # a data sample
            fronts = []

            lefts = []
            rights = []
            tops = []
            if not os.path.isdir(save_v + "/r5_"+str(scale)):
                os.mkdir(save_v + "/r5_"+str(scale))

            if os.path.isdir(v+"/rgb/front/"):
                fronts = [v+"/rgb/front/"+ img for img in os.listdir(v+"/rgb/front/") if os.path.isfile(v+"/rgb/front/"+ img)]
                if not os.path.isdir(save_v + "/r5_"+str(scale)+"/front/"):
                    os.mkdir(save_v + "/r5_"+str(scale)+"/front/")
                n_fronts = [save_v + "/r5_"+str(scale)+"/front/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/front/")]
    
                if not os.path.isdir(save_v + "/r5_"+str(scale)+"/bev/"):
                    os.mkdir(save_v + "/r5_"+str(scale)+"/bev/")
                n_bevs = [save_v +"/r5_"+str(scale)+"/bev/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/front/")]
            # ---------------------
            if os.path.isdir(v+"/rgb/right/"):
                rights = [v+"/rgb/right/"+ img for img in os.listdir(v+"/rgb/right/") if os.path.isfile(v+"/rgb/right/"+ img)]
                if not os.path.isdir(save_v + "/r5_"+str(scale)+"/right/"):
                    os.mkdir(save_v + "/r5_"+str(scale)+"/right/")
                n_rights = [save_v +"/r5_"+str(scale)+"/right/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/right/")]
            # -----------------------
            if os.path.isdir(v+"/rgb/left/"):
                lefts = [v+"/rgb/left/"+ img for img in os.listdir(v+"/rgb/left/") if os.path.isfile(v+"/rgb/left/"+ img)]
                if not os.path.isdir(save_v + "/r5_"+str(scale)+"/left/"):
                    os.mkdir(save_v + "/r5_"+str(scale)+"/left/")
                n_lefts = [save_v +"/r5_"+str(scale)+"/left/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/left/")]
            

            

            fronts = sorted(fronts)
            lefts = sorted(lefts)
            rights = sorted(rights)

            n_bevs = sorted(n_bevs)


            front_tensor = []
            for i in range(len(fronts)):
                try:
                    front_img = Image.open(fronts[i]).convert('RGB')
                    front_tensor.append(front_img)
                except:
                    print(fronts[i])
            left_tensor = []
            for i in range(len(lefts)):
                try:
                    left_img = Image.open(lefts[i]).convert('RGB')
                    left_tensor.append(left_img)
                except:
                    print(lefts[i])
            right_tensor = []
            for i in range(len(rights)):
                try:
                    right_img = Image.open(rights[i]).convert('RGB')
                    right_tensor.append(right_img)
                except:
                    print(rights[i])
            

            length = len(front_tensor)
            bev_tensor_ls = []
            try:
                for i in range(length):
                    bev_tensor_ls.append([left_tensor[i], front_tensor[i], right_tensor[i]])
            except:
                print("empty stack")
                continue
            
            with torch.no_grad():

                bev_f = [None for _ in range(length)]

                for i in range(length): 
                    bev_seg = model(bev_tensor_ls[i])

                    bev_f[i] = model.features(bev_tensor_ls[i])
                    segs = bev_seg.clone().sigmoid().cuda()
                    mi, _ = torch.max(segs, dim=1)
                    mi = mi.view((1, 1, 200, 200))
                    drivable_area = torch.zeros_like(mi)
                    drivable_area[mi > 0.5] = 1
                    non_drive = torch.ones((1, 1, 200, 200)).cuda() - drivable_area
                    new_class = torch.ones((1, 1, 200, 200)).cuda()
                    new_class[:, :, :100, :] = 0

                    pred_cat = torch.cat((non_drive, torch.unsqueeze(segs[:,3,:,:],1), torch.unsqueeze(segs[:,1,:,:],1),
                                        torch.unsqueeze(segs[:,4,:,:],1), torch.unsqueeze(segs[:,5,:,:],1),
                                        torch.unsqueeze(segs[:,0,:,:],1),
                                        torch.unsqueeze(segs[:,2,:,:],1), new_class), 1) # shape(_, 8, 200, 200)
                    
                    new_pred_cat = np.zeros((1, 8, 200, 200))
                    pred_cat_c = np.asarray(pred_cat.clone().cpu())[:, :, 0:100, 0:200]
                    new_pred_cat[:, :, 100:200, :] = pred_cat_c
                    segs = torch.from_numpy(new_pred_cat).cuda()
                    
                    result = Image.fromarray(COLOR[segs[0].argmax(0).detach().cpu().numpy()])
                    result.save('./og_results/eval_{}.png'.format(str(i).zfill(4)))  


                bev_features = torch.cat([bev_f[0]], dim=0)

                for i in range(1, length):
                    bev_features = torch.cat([bev_features, bev_f[i]], dim=0)
                
                save_feature(bev_features, n_bevs)
