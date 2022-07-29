import os
import json
from PIL import Image
from argparse import ArgumentParser

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import json 
import random
from tool import get_rot
sys.path.append('../models')

from lss import LSS
from MaskFormer.demo.demo import get_maskformer

###LSS
# from pyquaternion import Quaternion
print(torch.cuda.current_device())
def scale_and_crop_image(image, scale=4.0, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(image.width // scale), int(image.height // scale))

    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    # start_x = height//2 - crop//2
    # start_y = width//2 - crop//2
    # cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
    # cropped_image = np.transpose(cropped_image, (2,0,1))
    cropped_image = np.transpose(image, (2,0,1))

    return cropped_image

def scale(image, scale=4.0):

    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    
    return image

torch.cuda.empty_cache()


# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='r5')
# args = parser.parse_args()
# print(args)

# if args.model == 'r5':
#     model = get_maskformer().cuda()
 
scale=2
root='/data/carla_dataset/data_collection'
save_root='/data/carla_feature/data_collection'

        
scale = float(scale)

type_list = ['interactive']

#model = get_maskformer().cuda()
#print(model.pixel_mean)
#print(model.pixel_std)

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
    'Ncams': 2, 
}

model = LSS(grid_conf, data_aug_conf, outC=1).to('cuda')
model.load_state_dict(torch.load('../models/model525000.pt'))

model.eval()
for t, type in enumerate(type_list):
    basic_scenarios = [os.path.join(root, type, s) for s in os.listdir(os.path.join(root, type))]

    # iterate scenarios
    print('searching data')
    for s in tqdm(basic_scenarios, file=sys.stdout):
        # a basic scenario
        scenario_id = s.split('/')[-1]

        # if road_class != 5:
        variants_path = os.path.join(s, 'variant_scenario')
        variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path) if os.path.isdir(os.path.join(variants_path, v))]
        
        for v in variants:
            
            # a data sample
            fronts = []

            lefts = []
            rights = []
            tops = []
            f_path = v+"/rgb_f/" + '_'+str(int(scale)) 
            if not os.path.isdir(f_path):
                os.makedirs(f_path)

            if os.path.isdir(v+"/rgb/front/"):
                fronts = [v+"/rgb/front/"+ img for img in os.listdir(v+"/rgb/front/") if os.path.isfile(v+"/rgb/front/"+ img)]
                if not os.path.isdir(f_path + "/front/"):
                    os.mkdir(f_path + "/front/")
                n_fronts = [f_path + "/front/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/front/")]
            # ---------------------
            if os.path.isdir(v+"/rgb/right/"):
                rights = [v+"/rgb/right/"+ img for img in os.listdir(v+"/rgb/right/") if os.path.isfile(v+"/rgb/right/"+ img)]
                if not os.path.isdir(f_path + "/right/"):
                    os.mkdir(f_path + "/right/")
                n_rights = [f_path +"/right/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/right/")]
            # -----------------------
            if os.path.isdir(v+"/rgb/left/"):
                lefts = [v+"/rgb/left/"+ img for img in os.listdir(v+"/rgb/left/") if os.path.isfile(v+"/rgb/left/"+ img)]
                if not os.path.isdir(f_path + "/left/"):
                    os.mkdir(f_path + "/left/")
                n_lefts = [f_path +"/left/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/left/")]

            front_tensor = []
            for i in range(len(fronts)):
                try:
                    front_tensor.append(torch.from_numpy(np.array(
                        Image.open(fronts[i]).convert('RGB'))).float())
                    a = torch.from_numpy(np.array(
                        Image.open(fronts[i]).convert('RGB'))).float()
                   # print(a.shape)
                except:
                    print(fronts[i])

            left_tensor = []
            for i in range(len(lefts)):
                left_tensor.append(torch.from_numpy(np.array(
                    Image.open(lefts[i]).convert('RGB'))).float())
            right_tensor = []
            for i in range(len(rights)):
                right_tensor.append(torch.from_numpy(np.array(
                    Image.open(rights[i]).convert('RGB'))).float())
            # try:
            #     front_tensor = torch.stack(front_tensor)
            #     left_tensor = torch.stack(left_tensor)
            #     right_tensor = torch.stack(right_tensor)
            # except:
            #     print('empty stack')
            #     continue

            l = len(front_tensor)//4
            front_tensor_ls = []
            left_tensor_ls = []
            right_tensor_ls = []
            try:
                front_tensor_ls.append(torch.stack(front_tensor[:l]))
                front_tensor_ls.append(torch.stack(front_tensor[l:2*l]))
                front_tensor_ls.append(torch.stack(front_tensor[2*l:3*l]))
                front_tensor_ls.append(torch.stack(front_tensor[3*l:]))

                left_tensor_ls.append(torch.stack(left_tensor[:l]))
                left_tensor_ls.append(torch.stack(left_tensor[l:2*l]))
                left_tensor_ls.append(torch.stack(left_tensor[2*l:3*l]))
                left_tensor_ls.append(torch.stack(left_tensor[3*l:]))

                right_tensor_ls.append(torch.stack(right_tensor[:l]))
                right_tensor_ls.append(torch.stack(right_tensor[l:2*l]))
                right_tensor_ls.append(torch.stack(right_tensor[2*l:3*l]))
                right_tensor_ls.append(torch.stack(right_tensor[3*l:]))
            except:
                print('empty stack')
                continue


            with torch.no_grad():
                bev_tensor_ls = [None for _ in range(4)]
                for i in range(4):
                    front_tensor_ls[i] = front_tensor_ls[i].to('cuda', dtype=torch.float32)
                    left_tensor_ls[i] = left_tensor_ls[i].to('cuda', dtype=torch.float32)
                    right_tensor_ls[i] = right_tensor_ls[i].to('cuda', dtype=torch.float32)
                    bev_tensor_ls[i] = torch.stack([front_tensor_ls[i], left_tensor_ls[i], right_tensor_ls[i]])
                    bev_tensor_ls[i] = model.features(bev_tensor_ls[i]).cpu()
                bev_features = torch.cat(bev_tensor_ls[0], bev_tensor_ls[1], bev_tensor_ls[2], bev_tensor_ls[3], dim=0)

            # front_tensor = front_tensor.to('cuda', dtype=torch.float32)
            # with torch.no_grad():   
            #     front_tensor = (front_tensor - model.pixel_mean) / model.pixel_std
            #     features = model.backbone(front_tensor)['res5']

            # for i in range(features.shape[0]):
            #     f = features[i,:,:,:]
            #     f = f.cpu().numpy()
            #     np.save(n_fronts[i], f)

            # # ---------------------------------------------
            # left_tensor = left_tensor.to('cuda', dtype=torch.float32)
            # with torch.no_grad():   
            #     left_tensor = (left_tensor - model.pixel_mean) / model.pixel_std
            #     features = model.backbone(left_tensor)['res5']

            # for i in range(features.shape[0]):
            #     f = features[i,:,:,:]
            #     f = f.cpu().numpy()
            #     np.save(n_lefts[i], f)

            # # ---------------------------------------------
            # right_tensor = right_tensor.to('cuda', dtype=torch.float32)
            # with torh.no_grad():
            #     right_tensor = (right_tensor - model.pixel_mean) / model.pixel_std
            #     features = model.backbone(right_tensor)['res5']
            # for i in range(features.shape[0]):
            #     f = features[i,:,:,:]
            #     f = f.cpu().numpy()
            #     np.save(n_rights[i], f)
            


            # front_tensor = []
            # left_tensor = []
            # right_tensor = []

