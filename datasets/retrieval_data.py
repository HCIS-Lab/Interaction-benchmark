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


class Retrieval_Data(Dataset):

    def __init__(self, 
                seq_len=8, 
                step=9,
                training=True,
                is_top=False,
                front_only=True,
                viz=False,
                root='/media/hankung/ssd/carla_13/CARLA_0.9.13/PythonAPI/examples/data_collection'):
        
        self.is_top = is_top
        self.front_only = front_only
        self.viz = viz
        self.id = []
        self.variants = []
        self.front = []
        self.left = []
        self.right = []
        self.top = []
        self.road_class = []
        self.gt_ego = []
        self.gt_actor = []

        self.seq_len = seq_len
        type_list = ['interactive_t','non-interactive_t']

        for t, type in enumerate(type_list):
            basic_scenarios = [os.path.join(root, type, s) for s in os.listdir(os.path.join(root, type))]

            # iterate scenarios
            print('searching data')
            for s in tqdm(basic_scenarios, file=sys.stdout):
                # a basic scenario
                scenario_id = s.split('/')[-1]
                if training and scenario_id.split('_')[0] != '10' or not training and scenario_id.split('_')[0] == '10':

                    # if road_class != 5:
                    variants_path = os.path.join(s, 'variant_scenario')
                    variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path)]
                    
                    for v in variants:

                        # get retrieval label
                        v_id = v.split('/')[-1]
                        if os.path.isfile(v+'/retrieve_gt.txt'):
                            with open(v+'/retrieve_gt.txt') as f:
                                gt = []

                                for line in f:
                                    line = line.replace('\n', '')
                                    if line != '\n':
                                        gt.append(line)
                                gt = list(set(gt))
                                if 'None' in gt:
                                    continue

                        elif os.path.isfile(v+'/retrive_gt.txt'):
                            with open(v+'/retrive_gt.txt') as f:
                                gt = []

                                for line in f:
                                    line = line.replace('\n', '')
                                    if line != '\n':
                                        gt.append(line)
                                gt = list(set(gt))
                                if 'None' in gt:
                                    continue

                        else:
                            continue

                        try:
                            road_class, gt_ego, gt_actor = get_multi_class(gt, scenario_id, v_id)
                        except:
                            continue

                        # first frame of sequence not used
                        fronts = []
                        lefts = []
                        rights = []
                        tops = []
                        # a data sample

                        # start = 60
                        
                        # for i in range(start, start + seq_len*step, step):
                        #     # images
                        #     filename = f"{str(i).zfill(8)}.png"
                        #     if self.is_top:
                        #         tops.append(v+"/rgb/top/"+filename)
                        #     else:
                        #         if os.path.isfile(v+"/rgb/front/"+filename):
                        #             fronts.append(v+"/rgb/front/"+filename)
                        #         if not self.front_only:
                        #             if os.path.isfile(v+"/rgb/left/"+filename):
                        #                 lefts.append(v+"/rgb/left/"+filename)
                        #             if os.path.isfile(v+"/rgb/right/"+filename):
                        #                 rights.append(v+"/rgb/right/"+filename)

                        # if len(fronts) != seq_len and self.front_only:
                        #     continue
                        # if (len(rights) != seq_len or len(lefts) != seq_len) and not self.front_only:
                        #     continue

                        fronts = [v+"/rgb/front/"+ img for img in os.listdir(v+"/rgb/front/") if isfile(v+"/rgb/front/"+ img)]
                        fronts = fronts.sort()
                        # rights = [v+"/rgb/right/"+ img for img in os.listdir(v+"/rgb/right/") if isfile(v+"/rgb/right/"+ img)]
                        # rights = rights.sort()
                        # lefts = [v+"/rgb/left/"+ img for img in os.listdir(v+"/rgb/left/") if isfile(v+"/rgb/left/"+ img)]
                        # lefts = lefts.sort()

                        self.id.append(s.split('/')[-1])
                        self.variants.append(v.split('/')[-1])
                        if self.is_top:
                            self.top.append(tops)
                        else:    
                            self.front.append(fronts)
                            if not self.front_only:
                                self.left.append(lefts)
                                self.right.append(rights)

                        self.road_class.append(road_class)
                        self.gt_ego.append(gt_ego)
                        self.gt_actor.append(gt_actor)
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
        data['tops'] = []
        data['road'] = self.road_class[index]
        data['ego'] = self.gt_ego[index]
        data['actor'] = self.gt_actor[index]
        data['id'] = self.id[index]
        if self.viz:
            data['img_front'] = []

        if self.is_top:
            seq_tops = self.top[index]
        else:
            seq_fronts = self.front[index]
            if not self.front_only:
                seq_lefts = self.left[index]
                seq_rights = self.right[index]

        # for i in range(self.seq_len):
        #     if self.is_top:
        #         data['tops'].append(torch.from_numpy(np.array(
        #             scale_and_crop_image(Image.open(seq_tops[i]).convert('RGB')))))
        #     else:
        #         data['fronts'].append(torch.from_numpy(np.array(
        #             scale_and_crop_image(Image.open(seq_fronts[i]).convert('RGB')))))
        #         if self.viz:
        #             data['img_front'].append(np.array((Image.open(seq_fronts[i]).convert('RGB'))))
        #         if not self.front_only:
        #             data['lefts'].append(torch.from_numpy(np.array(
        #                 scale_and_crop_image(Image.open(seq_lefts[i]).convert('RGB')))))
        #             data['rights'].append(torch.from_numpy(np.array(
        #                 scale_and_crop_image(Image.open(seq_rights[i]).convert('RGB')))))
        total_frame_num = len(seq_fronts)
        step = total_frame_num // (self.seq_len-1)
        num_smaple = total_frame_num % step
        start_idx = random.randint(0, num_smaple-1)

        iter = 0
        while(len(data['fronts']) == self.seq_len and iter < num_smaple):
            iter += 1
            data['fronts'] = []
            for i in range(start_idx, total_frame_num, step):
                if os.path.isfile(seq_fronts[i]):
                    data['fronts'].append(torch.from_numpy(np.array(
                            scale_and_crop_image(Image.open(seq_fronts[i]).convert('RGB')))))
                else:
                    (start_idx += 1) % num_smaple
                    break

        if len(data[fronts]) != self.seq_len:
            print('failure data name:')
            print(data['id'])
        return data

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


def get_multi_class(gt_list, s_id, v_id):   
    road_type = {'i-': 0, 't1': 1, "t2": 2, "t3": 3, 's-': 4, 'r-': 5}

    ego_table = {'e:z1-z1': 0, 'e:z1-z2': 1, 'e:z1-z3':2, 'e:z1-z4': 3,
                    'e:s-s': 4, 'e:s-sl': 5, 'e:s-sr': 6,
                    'e:ri-r1': 7, 'e:r1-r2': 8, 'e:r1-ro':9}

    actor_table = {'c:1-1': 0, 'c:1-2': 1, 'c:1-3':2, 'c:1-4':3,
                    'c:2-1': 4, 'c:2-2':5, 'c:2-3': 6, 'c:2-4': 7,
                    'c:3-1': 8, 'c:3-2': 9, 'c:3-3': 10, 'c:3-4': 11,
                    'c:4-1': 12, 'c:4-2': 13, 'c:4-3': 14, 'c:4-4': 15,

                    'b:1-1': 16, 'b:1-2': 17, 'b:1-3': 18, 'b:1-4': 19,
                    'b:2-1': 20, 'b:2-2':21, 'b:2-3': 22, 'b:2-4': 23,
                    'b:3-1': 24, 'b:3-2': 25, 'b:3-3': 26, 'b:3-4': 27,
                    'b:4-1': 28, 'b:4-2': 29, 'b:4-3': 30, 'b:4-4': 31,

                    'c:s-s': 32, 'c:s-sl': 33, 'c:s-sr': 34,
                    'c:sl-s': 35, 'c:sl-sl': 36,
                    'c:sr-s': 37, 'c:sr-sr': 38,
                    'c:jl-jr': 39, 'c:jr-jl': 40,

                    'b:s-s': 41, 'b:s-sl': 42, 'b:s-sr': 43,
                    'b:sl-s': 44, 'b:sl-sl': 45,
                    'b:sr-s': 46, 'b:sr-sr': 47,
                    'b:jl-jr': 48, 'b:jr-jl': 49,

                    'p:c1-c2': 50, 'p:c1-c4': 51, 'p:c1-cr': 52,  'p:c1-cf': 53,
                    'p:c2-c1': 54, 'p:c2-c3': 55, 'p:c2-cl': 56,
                    'p:c3-c2': 57, 'p:c3-c4': 58, 'p:c3-cr': 59,
                    'p:c4-c1': 60, 'p:c4-c3': 61, 'p:c4-cr': 62, 'p:c4-cf': 63,
                    'p:cf-c1': 64, 'p:cf-c4': 65,
                    'p:cl-c1': 66, 'p:cl-c2': 67,
                    'p:cr-c3': 68, 'p:cr-c4': 69,

                    'c:ri-r1': 70, 'c:rl-r1': 71, 'c:r1-r2': 72, 'c:r1-ro': 73, 'c:ri-r2': 74,
                    'b:ri-r1': 75, 'b:rl-r1': 76, 'b:r1-r2': 77, 'b:r1-ro': 78, 'b:ri-r2': 79}

    road_class = s_id.split('_')[1][:2]
    road_class = road_type[road_class]
    actor_class = [0]*80
    for gt in gt_list:
        gt = gt.lower()
        if gt[0] == 'e':
            ego_class = ego_table[gt]
        else:
            actor_class[actor_table[gt]] = 1
    if ego_class == None:
        return

    ego_label = torch.tensor(ego_class)
    actor_label = torch.FloatTensor(actor_class)
    # print(ego_label)
    # ego_ind = ego_table[ego_class]
    # actor_ind = actor_table[actor_class]
    # ego_label[ego_ind] = torch.FloatTensor([1.0])
    # actor_label[actor_ind] = torch.FloatTensor([1.0])
    return road_class, ego_label, actor_label



