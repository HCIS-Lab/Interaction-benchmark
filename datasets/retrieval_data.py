import os
import json
from PIL import Image

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys


is_interactive = {'True': 0, 'False': 1}
ego_action = {'foward': 0, 'left_turn': 1, 'right_turn': 2, 'slide_left': 3, 'slide_right': 4,
 'u-turn': 5, 'backward': 6, 'crossing': 7, 'None': 8}
actor_type = {'car': 0, 'truck': 1, 'bike': 2, 'motor': 3, 'pedestrian': 4, 'None': 5}
actor_action = {'foward': 0, 'left_turn': 1, 'right_turn': 2, 'slide_left': 3, 'slide_right': 4,
 'u-turn': 5, 'stop': 6, 'backward': 7, 'crossing': 8, 'None': 9}
regulation = {'None': 0, 'parking': 1, 'jay-walker': 2, 'running traffic light': 3, 
'driving on a sidewalk': 4,  'stop sign': 5}
# weather = {'ClearNoon':, 'CloudyNoon', 'WetNoon', 'WetCloudyNoon', 'MidRainyNoon', 'HardRainNoon', 'SoftRainNoon',
#     'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'MidRainSunset', 'HardRainSunset', 'SoftRainSunset',
#     'ClearNight', 'CloudyNight', 'WetNight', 'WetCloudyNight', 'MidRainyNight', 'HardRainNight', 'SoftRainNight'}


class Retrieval_Data(Dataset):

    def __init__(self, root, config):
        
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.ignore_sides = config.ignore_sides
        self.ignore_rear = config.ignore_rear

        self.input_resolution = config.input_resolution
        self.scale = config.scale

        self.front = []
        self.left = []
        self.right = []

        scenarios = []
        files = os.listdir(root)
        for f in files:
            fullpath = join(root, f)
            if os.path.isdir(fullpath):
                scenarios.append(fullpath)

        # iterate scenarios (town0x)
        for sub_root in tqdm(scenarios, file=sys.stdout):
            preload_file = os.path.join(sub_root, str(self.seq_len) + '.npy')

            # dump to npy if no preload
            if not os.path.exists(preload_file):
                with open(os.path.join(sub_root, 'scenario_description.json')) as f:
                scenario_description = json.load(f)

                preload_front = []
                preload_left = []
                preload_right = []
                # jsonfile
                # a basis scenario
                variants_path = os..path.join(sub_root, 'variants_scenarios')
                variants = os.listdir(variants_scenarios)
                
                # iterate basis scenario data
                routes = [folder for folder in variants if not os.path.isfile(os.path.join(variants_path, folder))]
                for route in routes:

                    route_dir = os.path.join(variants_path, route)
                    # with open('%s/dynamic_description.json' % (stored_path), 'w') as file:
                    #     dynamic_description = json.load(file)

                    # first frame of sequence not used
                    fronts = []
                    lefts = []
                    rights = []

                    # a data sample
                    for i in range(self.seq_len):
                        # images
                        filename = f"{str(self.seq_len+1+i).zfill(4)}.png"
                        fronts.append(route_dir+"/Camera RGB/front/"+filename)
                        lefts.append(route_dir+"/Camera RGB/left/"+filename)
                        rights.append(route_dir+"/Camera RGB/right/"+filename)

                    preload_front.append(fronts)
                    preload_left.append(lefts)
                    preload_right.append(rights)


                # dump to npy
                preload_dict = {}
                preload_dict['front'] = preload_front
                preload_dict['left'] = preload_left
                preload_dict['right'] = preload_right
                preload_dict['rear'] = preload_rear
                preload_dict['is_interactive'] = is_interactive[scenario_description['interaction']]
                preload_dict['ego_action'] = ego_action[scenario_description['my_action']]
                preload_dict['actor_type'] = actor_type[scenario_description['interaction_actor_type']]
                preload_dict['actor_action'] = actor_action[scenario_description['interaction_action_type']]
                preload_dict['regulation'] = regulation[scenario_description['violation']]

                # preload_dict['topology'] = scenario_description[]
                # preload_dict['weather'] = 
                np.save(preload_file, preload_dict)

            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)
            self.front += preload_dict.item()['front']
            self.left += preload_dict.item()['left']
            self.right += preload_dict.item()['right']
            self.rear += preload_dict.item()['rear']
            self.is_interactive += preload_dict.item()['is_interactive']
            self.ego_action += preload_dict.item()['ego_action']
            self.actor_type += preload_dict.item()['actor_type']
            self.regulation += preload_dict.item()['regulation']
            # self.topology += preload_dict.item()['topology']
            # self.weather += preload_dict.item()['weather']
            print("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.front)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['lefts'] = []
        data['rights'] = []
        data['rears'] = []

        seq_fronts = self.front[index]
        seq_lefts = self.left[index]
        seq_rights = self.right[index]
        seq_rears = self.rear[index]


        for i in range(self.seq_len):
            data['fronts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(Image.open(seq_fronts[i]), scale=self.scale, crop=self.input_resolution))))
            if not self.ignore_sides:
                data['lefts'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_lefts[i]), scale=self.scale, crop=self.input_resolution))))
                data['rights'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_rights[i]), scale=self.scale, crop=self.input_resolution))))
            if not self.ignore_rear:
                data['rears'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_rears[i]), scale=self.scale, crop=self.input_resolution))))
        return data

def scale_and_crop_image(image, scale=1, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height//2 - crop//2
    start_y = width//2 - crop//2
    cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image

