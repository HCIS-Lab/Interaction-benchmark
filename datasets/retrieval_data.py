import os
import json
from PIL import Image

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import json 



class Retrieval_Data(Dataset):

    def __init__(self, 
                seq_len=20, 
                step=6,
                root='/media/hankung/ssd/carla_13/CARLA_0.9.13/PythonAPI/examples/data_collection'):
        
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        self.front = []
        self.left = []
        self.right = []

        with open('retrieval_interactive_gt.json') as f:
            gt_interactive = json.load(f)
        # with open('retrieval_non-interactive_gt.json') as f:
        #     gt_non_interactive = json.load(f)
        # with open('retrieval_collision_gt.json') as f:
        #     gt_collision = json.load(f)
        # with open('retrieval_obstacle_gt.json') as f:
        #     gt_obstacle = json.load(f)

        type = 'interactive'
        basic_scenarios = [os.path.join(root, type, s) for s in os.listdir(os.path.join(root, type))]

        # iterate scenarios
        for s in tqdm(basic_scenarios, file=sys.stdout):
            # a basic scenario
            scenario_id = s.split('/'[-1])
            road_class, gt_ego, gt_actor = get_class(gt_interactive, scenario_id)

            variants_path = os.path.join(s, 'variants_scenarios')
            variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path)]
            
            for v in variants:
                # first frame of sequence not used
                fronts = []
                lefts = []
                rights = []

                # a data sample
                for i in range(0, self.seq_len*step, step):
                    # images
                    filename = f"{str(self.seq_len+1+i).zfill(4)}.png"
                    fronts.append(v+"/rgb/front/"+filename)
                    lefts.append(v+"/rgb/left/"+filename)
                    rights.append(v+"/rgb/right/"+filename)

                self.id += s.split('/')[-1]
                self.variants += v.split('/')[-1]
                self.front += fronts
                self.left += lefts
                self.right += rights
                self.road_class += road_class
                self.gt_ego += gt_ego
                self.gt_actor += gt_actor
            # print("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.front)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['lefts'] = []
        data['rights'] = []
        data['road'] = self.road_class[index]
        data['ego'] = self.gt_ego[index]
        data['actor'] = self.gt_actor[index]
        data['id'] = self.id[index]

        seq_fronts = self.front[index]
        seq_lefts = self.left[index]
        seq_rights = self.right[index]


        for i in range(self.seq_len):
            data['fronts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(Image.open(seq_fronts[i])))))
            if not self.ignore_sides:
                data['lefts'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_lefts[i])))))
                data['rights'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_rights[i])))))

        return data

def scale_and_crop_image(image, scale=0.5, crop=256):
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

def get_class(gt_dict, id):        
    road_type = {'i-': 0, 't1': 1, "t2": 2, "t3": 3, 's-': 4, 'r-': 5}

    ego_4way = {'z1-z1': 0, 'z1-z2': 1, 'z1-z3':2, 'z1-z4':3}
    4way_label = {'z1-z1': 0, 'z1-z2': 1, 'z1-z3':2, 'z1-z4':3,
                    'z2-z1': 4, 'z2-z2':5, 'z2-z3': 6, 'z2-z4': 7,
                    'z3-z1': 8, 'z3-z2': 9, 'z3-z3': 10, 'z3-z4': 11,
                    'z4-z1': 12, 'z4-z2': 13, 'z4-z3': 14, 'z4-z4': 15,
                    'c1-c2': 16, 'c1-c4': 17,
                    'c2-c1': 18, 'c2-c3': 19,
                    'c3-c2': 20, 'c3-c4': 21,
                    'c4-c1': 22, 'c4-c3': 23,
                    '': 24}

    ego_3way_1 = {'t1-t1': 0, 't1-t2': 1, 't1-t4': 2}
    3way_1_label = {'t1-t1': 0, 't1-t2': 1, 't1-t4': 2, 
                    't2-t1': 3, 't2-t2': 4, 't2-t4': 5,
                    't4-t1': 6, 't4-t2': 7, 't4-t4': 8,
                    'c1-cf': 9, 'c1-c4': 10,
                    'cf-c1': 11, 'cf-c4': 12,
                    'c4-c1': 13, 'c4-cf': 14
                    '': 15}

    ego_3way_2 = {'t1-t1': 0, 't1-t2': 1, 't1-t3': 2}
    3way_2_label = {'t1-t1': 0, 't1-t2': 1, 't1-t3': 2,
                    't2-t1': 3, 't2-t2': 4, 't2-t3': 5,
                    't3-t1': 6, 't3-t2': 7, 't3-t3': 8,
                    'c1-c2': 9, 'c1-cl': 10,
                    'c2-c1': 11, 'c2-cl': 12,
                    'cl-c1': 13, 'cl-c2': 14,
                    '': 15}

    ego_3way_3 = {'t1-t1': 0, 't1-t3': 1, 't1-t4': 2}
    3way_3_label = {'t1-t1': 0, 't1-t3': 1, 't1-t4': 2,
                    't3-t1': 3, 't3-t3': 4, 't3-t4': 5,
                    't4-t1': 6, 't4-t3': 7, 't4-t4': 8,
                    'c3-c4': 9, 'c3-cr': 10,
                    'c4-c3': 11, 'c4-cr': 12,
                    'cr-c3': 13, 'cr-c4': 14,
                    '': 15}

    ego_straight = {'s-s': 0, 's-sl': 1, 's-sr': 2}
    straight_label = {'s-s': 0, 's-sl': 1, 's-sr': 2,
                        'sl-s': 3,
                        'sr-s': 4,
                        'jl-s': 5, 'jl-sl': 6, 'jl-jr': 7
                        'jr-s': 8, 'jr-sr': 9, 'jr-jl': 10,
                        '': 11}

    road_class = class_text.split('_')[1][:2]
    road_class = road_type[road_class]

    class_text = gt_dict[id]
    ego_class, actor_class = class_text.split(',')[0], class_text.split(',')[1]

    if road_class == 0:
        ego_class = ego_4way[ego_class]
        actor_class = 4way_label[actor_class]

    elif road_class == 1:
        ego_class = ego_3way_1[ego_class]
        actor_class = 3way_3_label[actor_class]

    elif road_class == 2:
        ego_class = ego_3way_2[ego_class]
        actor_class = 3way_2_label[actor_class]

    elif road_class == 3:
        ego_class = ego_3way_3[ego_class]
        actor_class = 3way_3_label[actor_class]

    elif road_class == 4:
        ego_class = ego_straight[ego_class]
        actor_class = straight_label[actor_class]

    return road_class, ego_class, actor_class



