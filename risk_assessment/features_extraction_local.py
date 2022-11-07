import os
import os.path as osp
import json
import csv

from MaskFormer import mask_former
from detectron2.structures import Boxes
import torch
from maskrcnn import get_maskrcnn
import cv2
from MaskFormer.demo.demo import get_maskformer
import numpy as np
import time
from carla_feature_util import *
import zipfile
import argparse

def get_models(device):
    maskformer = get_maskformer()
    detectron = get_maskrcnn()
    return maskformer.cuda(device).eval(), detectron.cuda(device).eval()

def test(root, obj_n_max, batch_n,pred_box=False,time=100):
    variant_path = get_paths(root)
    for variant_name in variant_path:
        print('\t',variant_name)
        # Read rgb
        img_archive = zipfile.ZipFile(os.path.join(variant_name,'rgb','front.zip'), 'r')
        zip_file_list = img_archive.namelist()
        img_file_list = sorted(zip_file_list)[1:] # the first element is a folder
        for img_zip in img_file_list:
            imgdata = img_archive.read(img_zip)
            imgdata = cv2.imdecode(np.frombuffer(imgdata, np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow('My Image', imgdata)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def get_features(root_nas, root_local, batch_n, zip, device, s_type, pred_box=False):
    ## Model(maskformer backbone & detectron2)
    models = get_models(device)

    # nas_path, local_path = get_paths(root,local,[s_type])
    # create_tracklet2(variant_path)
    # each variant scenario
    for s_id in os.listdir(os.path.join(root_local,s_type)):
        for variant in os.listdir(os.path.join(root_local,s_type,s_id,'variant_scenario')):
            try:
                local = os.path.join(root_local,s_type,s_id,'variant_scenario',variant)
                nas = os.path.join(root_nas,s_type,s_id,'variant_scenario',variant)
                # print(local.split('/')[-4:])
                bbox_path = osp.join(nas, 'bbox/front')
                bbox_list = sorted(os.listdir(bbox_path))
                first_frame = int(bbox_list[0].split('.')[0])
                if zip:
                    img_archive = zipfile.ZipFile(os.path.join(nas,'rgb','front.zip'), 'r')
                    zip_file_list = img_archive.namelist()
                    img_file_list = sorted(zip_file_list)[1:]
                else:
                    rgb_path = osp.join(nas, 'rgb/front')
                    img_file_list = sorted(os.listdir(rgb_path))
                index = 0
                assert len(img_file_list)==len(bbox_list)
                if not os.path.isdir(os.path.join(local,'features')):
                    print(local.split('/')[-4:])
                    os.mkdir(os.path.join(local,'features'))
                    os.mkdir(os.path.join(local,'features','rgb'))
                    os.mkdir(os.path.join(local,'features','rgb','front'))
                else:
                    continue
                while index<len(bbox_list):
                    img_in = []
                    box_in = []
                    curr_frame = first_frame+ index
                    #read each frame's bbox & rgb data
                    for _ in range(batch_n):
                        if index == len(bbox_list):
                            break
                        json_file = open(osp.join(bbox_path, bbox_list[index]))
                        data = json.load(json_file)
                        json_file.close()
                        if zip:
                            img = img_archive.read(img_file_list[index])
                            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                            img = cv2.resize(img, (640, 360))
                        else:
                            img = cv2.imread(osp.join(rgb_path, img_file_list[index]))
                        img_in.append(img)
                        box_in.append(data)
                        index += 1
                    full_frames,rois_gt = run_model(models,img_in,batch_n,box_in,20,device) # full_frame, roi_gt
                    for i in range(len(img_in)):
                        path = os.path.join(local,'features','rgb','front')
                        frame_num = curr_frame + i 
                        print(frame_num,end='\r')
                        os.mkdir('%s/%.8d' % (path,frame_num))
                        torch.save(full_frames[i].detach().clone().cpu(),'%s/%.8d/frame.pt'%(path,frame_num))
                        torch.save(rois_gt[i],'%s/%.8d/object.pt'%(path,frame_num))
            except:
                print(local.split('/')[-4:])
                print('Assertion error, passing')
            
            print('')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--device',
        default=0,
        type=int,
        help='cuda device')
    argparser.add_argument(
        '--type',
        default='collision',
        help='which scenario type')
    argparser.add_argument(
        '--batch',
        default=20,
        type=int,
        help='batch number') 
    args = argparser.parse_args()
    # root = "/media/tony/Carla/carla/PythonAPI/examples/data_collection/collision"
    root_nas = "/mnt/Final_Dataset/dataset/"
    local = "/data/carla_dataset/data_collection/"
    get_features(root_nas,local,args.batch,True,args.device,args.type)
    # get_features_zip(root_nas,80,10,args.device)
