import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import json

class CarlaDataset(Dataset):
    def __init__(self, root, key, baseline,gt_box=True):
        """
            key: collision, non_interactive...
            baseline: int, 1,2 or 3.
        """
        self.gt_box = gt_box
        self.baseline = baseline
        self.datas_path = []
        self.labels = []
        label = None
        
        curr_path = root
        path = [curr_path]
        for scenario_type in os.listdir(curr_path):
            if scenario_type in key:
                label = True if scenario_type == 'collision' else False
                curr_path = path[-1]
                curr_path = os.path.join(curr_path,scenario_type)
                path.append(curr_path)
                for scenario_id in os.listdir(curr_path):
                    print(scenario_id)
                    curr_path = path[-1]
                    curr_path = os.path.join(curr_path,scenario_id,'variant_scenario')
                    path.append(curr_path)
                    for variant_name in os.listdir(curr_path):
                        print('\t', variant_name)
                        curr_path = path[-1]
                        curr_path = os.path.join(curr_path,variant_name)
                        self.datas_path.append(curr_path)
                        self.labels.append(label)
                    path.pop()
                path.pop()
    
    def __getitem__(self, index):
        # list frame_n, 81
        # nn.linear(xxx,81)
        """
             baseline1 read full_frame(p5 or res5), collision label
             baseline2 read full_frame, roi, bbox, collision label
             baseline3 read full_frame, roi, bbox, collision label, risky object label
        """
        path = self.datas_path(index)
        frame_features = torch.load(os.path.join(path,'features','frame.pt'))
        collision_label = self.labels[index]
        if self.baseline >1:
            bbox = []
            if self.gt_box:
                roi = torch.load(os.path.join(path,'features','roi_gt.pt'))
            else:
                roi = torch.load(os.path.join(path,'features','roi.pt'))
            if self.gt_box:
                bbox_path = os.path.join(path, 'bbox/front')
                bbox_path = sorted(os.listdir(bbox_path))
                f = open(path, 'r')
                s = f.read().split(' ')
                f.close()
                start = int(s[0])
                end = int(s[1])
                bbox_path_list = bbox_path[start:end]
                for bbox_path in bbox_path_list:
                    json_file = open(os.path.join(path,bbox_path))
                    data = json.load(json_file)
                    json_file.close()
                    bbox.append(data)
            # TODO else:
        if self.baseline==1:
            return frame_features, collision_label
        elif self.baseline==2:
            return frame_features,roi,bbox,collision_label
        elif self.baseline==3:
            risky_object = int(s[2])+1
            return frame_features,roi,bbox,collision_label,risky_object
        
    def __len__(self):
       return len(self.datas_path)
