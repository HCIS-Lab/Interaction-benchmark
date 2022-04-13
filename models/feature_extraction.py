import torch
from MaskFormer.demo.demo import get_maskformer
from PIL import Image
from torchvision import transforms

import argparse

import os 
import numpy as np
import time

def get_models():
    backbone = get_maskformer().backbone
    return backbone

def save_features(backbone, view, rgb_img_path, rgb_features_path, batch_size=64):
    img_view_path = os.path.join(rgb_img_path, view)
    feature_view_path = os.path.join(rgb_features_path, view)

    if not os.path.exists(feature_view_path):
        os.makedirs(feature_view_path)
    images_name = []
    for f in os.listdir(img_view_path):
        f_path = os.path.join(img_view_path, f)
        # print(f_path)
        if os.path.isfile(f_path):
            images_name.append(f)

    length = len(images_name)
    images = []
    for i, img_name in enumerate(images_name):

        img = Image.open(os.path.join(img_view_path, img_name)).convert('RGB')
        img = np.asarray(img)
        # print(img.shape)
        img = np.transpose(img, (2,0,1))
        img = torch.as_tensor(img.astype("float32"))
        images.append(img)
        if len(images) == batch_size or i == length-1:
            images = torch.stack(images).cuda()
            features = backbone(images)
            features = features['res5']
            features = features.cpu().detach().numpy()
            for j in range(images.shape[0]):
                np.save(os.path.join(feature_view_path, images_name[i//batch_size + j]), features[j])
            images = []
    # for i, image in enumerate(images):



if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description='features extraction')
    argparser.add_argument(
        '-root',
        type=str,
        default='/media/hankung/ssd/carla_13/CARLA_0.9.13/PythonAPI/examples/data_collection',
        help='name of the scenario')
    argparser.add_argument(
        '-type',
        type=str,
        default='obstacle',
        help='type of the scenario')

    args = argparser.parse_args()

    backbone = get_models()
    backbone = backbone.cuda()
    backbone.eval()

    path_to_data_collection = os.path.join(args.root, args.type)
    primary_scenario_path = [os.path.join(path_to_data_collection, d) for d in os.listdir(path_to_data_collection) if os.path.isdir(os.path.join(path_to_data_collection, d))]
    with torch.no_grad():
        for i, scenario_path in enumerate(primary_scenario_path):
            for variant_name in os.listdir(os.path.join(scenario_path, 'variant_scenario')):
                variant_scenario_path = os.path.join(scenario_path, 'variant_scenario', variant_name)
                rgb_img_path = os.path.join(variant_scenario_path, 'rgb')
                rgb_features_path = os.path.join(variant_scenario_path, 'rgb_features')
                if not os.path.exists(rgb_features_path):
                    os.makedirs(rgb_features_path)
                view = ['front', 'left', 'right']
                print('saving %s' %variant_name)
                for v in view:
                    print(v)
                    save_features(backbone, v, rgb_img_path, rgb_features_path, 64)
