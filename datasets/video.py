import cv2
import os
from tqdm import tqdm
import numpy as np

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


root = '/data/scenario_retrieval/results/'

basic_scenarios = [os.path.join(root, s) for s in os.listdir(root)]
for s in tqdm(basic_scenarios):
    variant_path = os.path.join(s, 'variant_scenario')
    variants = [os.path.join(variant_path, v) for v in os.listdir(variant_path)]
    for imgs in variants:
        video = cv2.VideoWriter(imgs + '/video.mp4', fourcc, 24, (256, 128))
        for img_path in os.listdir(imgs):
            image = cv2.imread(os.path.join(imgs, img_path))
            video.write(image)
        
        cv2.destroyAllWindows()
        video.release()