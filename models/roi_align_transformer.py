from turtle import back
from torchvision.ops.roi_align import roi_align
import torch
from maskrcnn import get_maskrcnn
import cv2
from MaskFormer.demo.demo import get_maskformer
import os 
import numpy as np
import time
VIDEO_PATH = ['../../Anticipating-Accidents/dataset/videos/training','../../Anticipating-Accidents/dataset/videos/testing']

def get_models():
    backbone = get_maskformer().backbone
    model = get_maskrcnn()
    return backbone,model
def test():
    backbone,model = get_models()
    backbone = backbone.cuda()
    backbone.eval()
    model = model.cuda()
    model.eval()
    inputs = []
    img = cv2.imread('test_img.jpg')
    height, width = img.shape[:2]
    print(height,width)
    frame = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    # print(frame.shape)
    inputs.append({"image": frame, "height": height, "width": width})
    images = model.preprocess_image(inputs)
    # print(images)
    features = backbone(images.tensor)
    # features = backbone(inputs)
    features_2 = model.backbone(images.tensor)  # set of cnn features
    proposals, _ = model.proposal_generator(images, features_2, None)  # RPN
    features_ = [features_2[f] for f in model.roi_heads.in_features]
    # batch_baseline1[:,frame_i*10:(frame_i+1)*10] = features_[1].cpu().numpy()
    # continue
    # ROI ALIGN
    box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
    # Flatten roi align features
    box_features_flat = model.roi_heads.box_head(box_features)  # features of all 1k candidates

    predictions = model.roi_heads.box_predictor(box_features_flat)
    pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
    pred_instances = model.roi_heads.forward_with_given_boxes(features_2, pred_instances)
    # output boxes, masks, scores, etc
    pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
    # print(pred_inds[0].size())
    instances = pred_instances[0]["instances"]
    roi_input = []
    roi_input.append(instances.pred_boxes.tensor)
    roi_input.append(instances.pred_boxes.tensor)
    print(features['res2'].size())
    print(features['res3'].size())
    print(features['res4'].size())
    print(features['res5'].size())
    print(roi_input[0])
    roi = roi_align(features['res5'],roi_input,2)
    # print(roi.size())
    return

if __name__ == '__main__':
    backbone,model = get_models()
    backbone = backbone.cuda()
    backbone.eval()
    model = model.cuda()
    model.eval()
    img_per_batch = 5
    with torch.no_grad():
        for t_n,path in enumerate(VIDEO_PATH):
            batch_count = 1
            for n_p in os.listdir(path):
                print(n_p)
                label = None
                if n_p == 'positive':
                    label = True
                else:
                    label = False
                now_path = os.path.join(path,n_p)
                for file in os.listdir(now_path):
                    print("Batch number:",batch_count,"\n\tFile name:",file)
                    batch_labels = np.zeros((1),dtype=bool)
                    batch_file_name = np.zeros((1),dtype=str)
                    batch_data_flat = np.zeros((1,100,20,8192),dtype=np.float32)
                    batch_scores = np.zeros((1,100,20),dtype=np.float32)
                    batch_bbox = np.zeros((1,100,20,4),dtype=np.float32)
                    batch_classes = np.ones((1,100,20),dtype=int)
                    batch_risky = np.zeros((1,100,20),dtype=int)
                    batch_classes = np.negative(batch_classes)
                    cap = cv2.VideoCapture(now_path+'/'+file) 
                    for frame_i in range(100//img_per_batch):
                        print("\t\tFrame num:",frame_i*img_per_batch,end='\r')
                        inputs = []
                        for _ in range(img_per_batch):
                            _, frame2 = cap.read()
                            ######
                            # frame = cv2.resize(frame,(160,80),interpolation=cv2.INTER_AREA)
                            ######
                            height, width = frame2.shape[:2]
                            frame = torch.as_tensor(frame2.astype("float32").transpose(2, 0, 1))
                            inputs.append({"image": frame, "height": height, "width": width})
                        images = model.preprocess_image(inputs)
                        features_maskformer = backbone(images.tensor)
                        # roi align
                        features = model.backbone(images.tensor)  # set of cnn features
                        proposals, _ = model.proposal_generator(images, features, None)  # RPN
                        features_ = [features[f] for f in model.roi_heads.in_features]
                        # ROI ALIGN
                        box_features_flat = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
                        # Flatten roi align features
                        box_features_flat = model.roi_heads.box_head(box_features_flat)  # features of all 1k candidates
                        predictions = model.roi_heads.box_predictor(box_features_flat)
                        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
                        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)
                        # output boxes, masks, scores, etc
                        pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
                        roi_input = []
                        for i in range(img_per_batch):
                            ins = pred_instances[i]["instances"]
                            temp_size = ins.pred_boxes.tensor.size()[0]
                            if temp_size<20:
                                temp_tensor = ins.pred_boxes.tensor
                                zeros_tensor = torch.zeros(20-temp_size,4).cuda()
                                roi_input.append(torch.cat((temp_tensor,zeros_tensor),0))
                            else:
                                roi_input.append(ins.pred_boxes.tensor[:20])
                        roi = roi_align(features_maskformer['res5'],roi_input,2)
                        roi = roi.view(img_per_batch,20,-1)
                        batch_data_flat[:,frame_i*img_per_batch:(frame_i+1)*img_per_batch] = roi.cpu().numpy()
                        for i in range(img_per_batch):
                            size = pred_inds[i].size(dim=0)
                            if size > 20:
                                batch_bbox[:,frame_i*img_per_batch+i] = pred_instances[i]['instances'].pred_boxes.tensor[:20].cpu().numpy()
                                batch_scores[:,frame_i*img_per_batch+i] = pred_instances[i]['instances'].scores[:20].cpu().numpy()
                                batch_classes[:,frame_i*img_per_batch+i] = pred_instances[i]['instances'].pred_classes[:20].cpu().numpy()
                            else:
                                batch_bbox[:,frame_i*img_per_batch+i,:size] = pred_instances[i]['instances'].pred_boxes.tensor.cpu().numpy()
                                batch_scores[:,frame_i*img_per_batch+i,:size] = pred_instances[i]['instances'].scores.cpu().numpy()
                                batch_classes[:,frame_i*img_per_batch+i,:size] = pred_instances[i]['instances'].pred_classes.cpu().numpy()
                            # for j,bbox in enumerate(pred_instances[i]['instances'].pred_boxes):
                            #     if IOU(bbox,gt_bbox)>=0.6:
                            #         batch_risky[:,frame_i*img_per_batch+i,j]=1

                    batch_labels = label
                    batch_file_name = file
                    # training
                    if t_n==0:
                        np.savez('/mnt/sdb/Dataset/SA_Maskformer/training/batch_%04d' % batch_count, file_name=batch_file_name, label=batch_labels, data_flat=batch_data_flat, bboxes=batch_bbox, classes=batch_classes, scores=batch_scores)
                    else:
                        np.savez('/mnt/sdb/Dataset/SA_Maskformer/testing/batch_%04d' % batch_count, file_name=batch_file_name, label=batch_labels, data_flat=batch_data_flat, bboxes=batch_bbox, classes=batch_classes, scores=batch_scores)
                    print("")
                    # if t_n==0:
                    # 	np.savez('/mnt/sdb/Dataset/SA/training/batch_baseline1_%04d' % batch_count, file_name=batch_file_name, label=batch_labels, data=batch_baseline1)
                    # else:
                    # 	np.savez('/mnt/sdb/Dataset/SA/testing/batch_baseline1_%04d' % batch_count, file_name=batch_file_name, label=batch_labels, data=batch_baseline1)
                    batch_count += 1
