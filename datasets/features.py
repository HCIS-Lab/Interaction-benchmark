import os
import json
from PIL import Image

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import json 
import random
from tool import get_rot
sys.path.append('/home/hankung/Desktop/Interaction_benchmark/models')

from MaskFormer.demo.demo import get_maskformer

###LSS
# from pyquaternion import Quaternion

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

 
scale=4
root='/media/hankung/ssd/carla_13/CARLA_0.9.13/PythonAPI/examples/data_collection'
        
scale = float(scale)

type_list = ['non-interactive']


model = get_maskformer().backbone.cuda()

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
        variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path)]
        
        for v in variants:
            
            # a data sample
            fronts = []

            lefts = []
            rights = []
            tops = []

            if not os.path.isdir(v+"/rgb_f/"):
                    os.mkdir(v+"/rgb_f/")

            if os.path.isdir(v+"/rgb/front/"):
                fronts = [v+"/rgb/front/"+ img for img in os.listdir(v+"/rgb/front/") if os.path.isfile(v+"/rgb/front/"+ img)]
                if not os.path.isdir(v+"/rgb_f/front/"):
                    os.mkdir(v+"/rgb_f/front/")
                n_fronts = [v+"/rgb_f/front/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/front/")]
            # ---------------------
            if os.path.isdir(v+"/rgb/right/"):
                rights = [v+"/rgb/right/"+ img for img in os.listdir(v+"/rgb/right/") if os.path.isfile(v+"/rgb/right/"+ img)]
                if not os.path.isdir(v+"/rgb_f/right/"):
                    os.mkdir(v+"/rgb_f/right/")
                n_rights = [v+"/rgb_f/right/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/right/")]
            # -----------------------
            if os.path.isdir(v+"/rgb/left/"):
                lefts = [v+"/rgb/left/"+ img for img in os.listdir(v+"/rgb/left/") if os.path.isfile(v+"/rgb/left/"+ img)]
                if not os.path.isdir(v+"/rgb_f/left/"):
                    os.mkdir(v+"/rgb_f/left/")
                n_lefts = [v+"/rgb_f/left/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/left/")]

            front_tensor = []
            for i in range(len(fronts)):
                front_tensor.append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(fronts[i]).convert('RGB'), scale))).float())
            left_tensor = []
            for i in range(len(lefts)):
                left_tensor.append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(lefts[i]).convert('RGB'), scale))).float())
            right_tensor = []
            for i in range(len(rights)):
                right_tensor.append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(rights[i]).convert('RGB'), scale))).float())

            front_tensor = torch.stack(front_tensor)
            left_tensor = torch.stack(left_tensor)
            right_tensor = torch.stack(right_tensor)

            front_tensor = front_tensor.cuda()
            with torch.no_grad():   
                features = model(front_tensor)['res5']

            for i in range(features.shape[0]):
                f = features[i,:,:,:]
                f = f.cpu().numpy()
                np.save(n_fronts[i], f)

            # ---------------------------------------------
            left_tensor = left_tensor.cuda()
            with torch.no_grad():   
                features = model(left_tensor)['res5']

            for i in range(features.shape[0]):
                f = features[i,:,:,:]
                f = f.cpu().numpy()
                np.save(n_lefts[i], f)

            # ---------------------------------------------
            right_tensor = right_tensor.cuda()
            with torch.no_grad():   
                features = model(right_tensor)['res5']

            for i in range(features.shape[0]):
                f = features[i,:,:,:]
                f = f.cpu().numpy()
                np.save(n_rights[i], f)


            front_tensor = []
            left_tensor = []
            right_tensor = []




