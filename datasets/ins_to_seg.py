from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import sys
import torch
from torchvision import transforms
root='/data/scenario_retrieval/carla-1/PythonAPI/examples/data_collection/'
type_list=['interactive']
def main():
    basic_scenarios = [os.path.join(root,type_list[0], s) for s in os.listdir(os.path.join(root, type_list[0])) if '6_' in s[:2] or '7_' in s[:2] or '10_' in s[:3]]
            
    for s in tqdm(basic_scenarios, file=sys.stdout): 
        variants_path = os.path.join(s, 'variant_scenario')
        variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path) if os.path.isdir(os.path.join(variants_path, v))]
        for v in variants:  
            if os.path.isdir(os.path.join(v, 'instance_segmentation/lbc_ins')):
                lbc_ins_path = os.path.join(v, 'instance_segmentation/lbc_ins')
                lbc_seg_path = os.path.join(v, 'instance_segmentation/ins_to_sem')
                if not os.path.isdir(lbc_seg_path):
                    os.mkdir(lbc_seg_path)
                    
                for lbc_ins_file in os.listdir(lbc_ins_path):
                    lbc_ins = Image.open(os.path.join(lbc_ins_path, lbc_ins_file)).convert('RGB').resize((200, 200))
                    lbc_ins = transforms.ToTensor()(lbc_ins) * 255
                    lbc_seg = np.zeros((10, 200, 200))
                    for i in range(200):
                        for j in range(200):
                            if lbc_ins[0][i, j] == 1: #buildings
                                lbc_seg[0][i, j] = 1
                            elif lbc_ins[0][i, j] == 2: #fences
                                lbc_seg[1][i, j] = 1
                            elif lbc_ins[0][i, j] == 4: #Pedestrains 
                                lbc_seg[2][i, j] = 1
                            elif lbc_ins[0][i, j] == 5: #Poles
                                lbc_seg[3][i, j] = 1
                            elif lbc_ins[0][i, j] == 6: #Roadlines
                                lbc_seg[4][i, j] = 1
                            elif lbc_ins[0][i, j] == 7: #Roads
                                lbc_seg[5][i, j] = 1
                            elif lbc_ins[0][i, j] == 8: #Sidewalks
                                lbc_seg[6][i, j] = 1
                            elif lbc_ins[0][i, j] == 10: #Vehicles
                                lbc_seg[7][i, j] = 1
                            elif lbc_ins[0][i, j] == 11: #Walls
                                lbc_seg[8][i, j] = 1
                            else: #Others
                                lbc_seg[9][i, j] = 1
                    np.save(os.path.join(lbc_seg_path, lbc_ins_file[:9]) + '.npy', lbc_seg)
if __name__ == '__main__':
    main()    