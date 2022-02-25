from maskrcnn import get_maskrcnn
import torch
import cv2
from PIL import Image
import os
import numpy as np

def roi_align(img_path,img_per_batch):
    # get model from detectron2
    model = get_maskrcnn()
	model = model.cuda()
	model.eval()
    inputs = []
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    # Input should be: List[Dict[str, torch.Tensor]]
    inputs.append({"image": img, "height": height, "width": width})
    images = model.preprocess_image(inputs)
    features = model.backbone(images.tensor)  # set of cnn features
    proposals, _ = model.proposal_generator(images, features, None)  # RPN
    # features pyramid (p2,p3,p4,p5)
    features_ = [features[f] for f in model.roi_heads.in_features]
    # ROI ALIGN (rectangle)
    box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
    # Flatten roi align features, shape: Batch x 1000 x 1024
    box_features_flat = model.roi_heads.box_head(box_features)  # features of all 1k candidates

    predictions = model.roi_heads.box_predictor(box_features_flat)
    # pred_inds : Prediction result index(box_features_flat)
    pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
    pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)
    # output boxes, masks, scores, etc
    pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
    for i in range(img_per_batch):
        # number of objects
        size = pred_inds[i].size(dim=0)
        current_features_flat = box_features_flat[i*1000:(i+1)*1000]
		current_features_flat = current_features_flat[pred_inds[i]]
        boxes = pred_instances[i]['instances'].pred_boxes
        classes = pred_instances[i]['instances'].pred_classes
