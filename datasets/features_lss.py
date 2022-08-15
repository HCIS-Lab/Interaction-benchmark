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
from lss_cnn_lstm import CNNLSTM_maskformer

device = torch.device('cuda:1')
###LSS
# from pyquaternion import Quaternion

def save_feature(features, path_list):
   
    for i in range(features.shape[0]):
        f = features[i,:,:,:]
        f = f.numpy()

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
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
    'Ncams': 3
}

model = LSS(grid_conf, data_aug_conf, outC=1, scale=scale).to(device)
pretrained_dict = torch.load('../models/model525000.pt')
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretr)
model.load_state_dict(pretrained_dict)

model.eval()

model2 = CNNLSTM_maskformer(num_cam=3, num_ego_class=1, num_actor_class=1, road=False).to(device)
model2.train()

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
                if not os.path.isdir(save_v + "/r5_"+str(scale)+"/bev_seg/"):
                    os.mkdir(save_v + "/r5_"+str(scale)+"/bev_seg/")
                n_bevs = [save_v +"/r5_"+str(scale)+"/bev/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/front/")]
                n_bevs_seg = [save_v +"/r5_"+str(scale)+"/bev_seg/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/front/")]
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

            front_tensor = []
            for i in range(len(fronts)):
                try:
                    front_tensor.append(Image.open(fronts[i]))
                except:
                    print(fronts[i])
            left_tensor = []
            for i in range(len(lefts)):
                try:
                    left_tensor.append(Image.open(lefts[i]))
                except:
                    print(lefts[i])
            right_tensor = []
            for i in range(len(rights)):
                try:
                    right_tensor.append(Image.open(rights[i]))
                except:
                    print(rights[i])
            

            l = len(front_tensor) // 12
            front_tensor_ls = []
            left_tensor_ls = []
            right_tensor_ls = []
            try:
                for i in range(12):
                    front_tensor_ls.append(front_tensor[i*l:(i+1)*l])
                    left_tensor_ls.append(left_tensor[i*l:(i+1)*l])
                    right_tensor_ls.append(right_tensor[i*l:(i+1)*l])
            except:
                print("empty stack")
                continue

            with torch.no_grad():
                bev_seg = [None for _ in range(12)]
                bev_f = [None for _ in range(12)]
                bev_tensor_ls = [None for _ in range(12)]

                for i in range(12):      
                    bev_tensor_ls[i] = [front_tensor_ls[i], left_tensor_ls[i], right_tensor_ls[i]]
                    bev_seg[i] = model(bev_tensor_ls[i]).cpu()
                    bev_f[i] = model.features(bev_tensor_ls[i]).cpu()

                bev_segmentation = torch.cat([bev_seg[0]], dim=0)
                bev_features = torch.cat([bev_f[0]], dim=0)

                for i in range(1, 12):
                    bev_segmentation = torch.cat([bev_segmentation, bev_seg[i]], dim=0)
                    bev_features = torch.cat([bev_features, bev_f[i]], dim=0)

            save_feature(bev_segmentation, n_bevs_seg)
            save_feature(bev_features, n_bevs)
            
            # bev_features = bev_features.to(device)
            # ego, actor = model2(bev_features)