import numpy as np 
from PIL import Image
import os

path = '/data/carla_feature/data_collection/interactive/10_i-1_1_p_c_l_1_j/variant_scenario/CloudyNoon_high_/r5_4.0/bev_seg/'

if not os.path.isdir('./bev_img/'):
    os.mkdir('./bev_img/')

files = os.listdir(path)
files = sorted(files)
for f in files:
    img = np.load(path + f) * 255
    img = img.astype(np.uint8)
    img = img.squeeze(0)
    img = Image.fromarray(img)
    img.save('./bev_img/'+f[:9]+'png')
    