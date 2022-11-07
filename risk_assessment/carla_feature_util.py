import os
import os.path as osp
import json
import csv
from detectron2.structures import Boxes
import torch
import numpy as np

def get_paths(root,s_types):
    scenario_path = []
    variant_path =[]
    for s_type in s_types:
        for s_id in os.listdir(osp.join(root, s_type)):
            scenario_path.append(osp.join(root, s_type, s_id))
    
    for s_path in scenario_path:
        variant_id = os.listdir(osp.join(s_path, 'variant_scenario'))
        for v_id in variant_id:
            variant_path.append(osp.join(s_path, 'variant_scenario',v_id))
    return variant_path

def order_match(variant_path, n_obj, order_by_freq=False):
    match = {}  # dict[actor_id]: feature order
    match_inverse = {}
    count = {}
    order = 0
    with open(osp.join(variant_path, 'tracklet.csv'), newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            actor_id = int(row[1]) # & 0xffff

            if actor_id not in match:
                match[actor_id] = order
                match_inverse[order] = actor_id
                order += 1
        
            if order == n_obj:
                break
        
        # ## order by freq
        if order_by_freq:
            for row in rows:
                actor_id = int(row[1])
                if actor_id not in count:
                    count[actor_id] = 1
                else:
                    count[actor_id] += 1
            print(count)
            sorted_key = sorted(count, key=count.get, reverse=True)
            for actor_id in sorted_key:
                match[actor_id] = order
                order += 1

    print(match)
    print(order)
    with open('%s/tracker.json' % (variant_path), 'w') as f:
        json.dump(match, f)
    with open('%s/tracker_inverse.json' % (variant_path), 'w') as f:
        json.dump(match_inverse, f)
    return match

def create_tracklet2(variant_path):
    
    for v_path in variant_path:
        print(v_path)
        bbox_path = osp.join(v_path, "bbox/front")
        json_list = sorted(os.listdir(bbox_path))
        
        #tracking_results = []
        with open(osp.join(v_path, 'tracklet.csv'), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for j in json_list:
                bbox_list = []
                id_list = []
                frame = int(osp.splitext(j)[0])

                json_file = open(osp.join(bbox_path, j))
                data = json.load(json_file)
                json_file.close()
                # data_list
                
                for dict in data:
                    bbox_list.append(dict['box'])
                    id_list.append(dict['actor_id'])
                for bbox, id in zip(bbox_list, id_list):
                    # bbox: left_bot, right_top
                    h = bbox[2]-bbox[0]
                    w = bbox[3]-bbox[1]

                    row = [frame, id, bbox[0], bbox[1], h, w]
                    writer.writerow(row)
                #     #tracking_results.append([frame, id, bbox[0][0], bbox[0][1], h, w])

def run_model(models, inputs_raw, batch_n,data_list,obj_n_max,device,tracking=True,pred_box=False):
    """
        inputs: List[img]
        return: frame_features, roi_gt, roi_pred, bbox_pred,
    """
    maskformer, model = models
    # device = maskformer.device.index
    inputs = []
    for frame in inputs_raw:
        height, width = frame.shape[:2]
        frame = torch.as_tensor(frame.astype("float32").transpose(2, 0, 1)).cuda(device)
        inputs.append({"image": frame, "height": height, "width": width})
    with torch.no_grad():
        # p2~p5
        fpn_features = maskformer.get_fpn_features(inputs) # res5, [mask(p2),p2,p3,p4,p5]
        features_maskformer = fpn_features[4]
        ###new
        roi_input_list = []
        roi_input_size = []
        # data_list: batch of json
        for datas in data_list:
            temp_list = []
            roi_input_size.append(len(datas))
            for data in datas:
                temp_list.append(data['box'])
            temp_list = np.round(np.array(temp_list)*0.5)
            temp_list = torch.from_numpy(temp_list).cuda(device).view(-1,4)
            temp_list = Boxes(temp_list)
            roi_input_list.append(temp_list)
        roi_gt = model.roi_heads.box_pooler(fpn_features[1:],roi_input_list)
        counter = 0
        out_roi = []
        for size in roi_input_size:
            out_roi.append(roi_gt[counter:counter+size].detach().clone().cpu())
            counter += size
        if pred_box:
            # TODO pred: class_id, score
            # roi align
            images = model.preprocess_image(inputs)
            features = model.backbone(images.tensor)  # set of cnn features
            proposals, _ = model.proposal_generator(images, features, None)  # RPN
            # Dict to List
            features_ = [features[f] for f in model.roi_heads.in_features] 
            # ROI ALIGN
            box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
            # Flatten roi align features
            box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
            predictions = model.roi_heads.box_predictor(box_features)
            pred_instances, _ = model.roi_heads.box_predictor.inference(predictions, proposals)
            pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)
            # output boxes, masks, scores, etc
            pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
            roi_input = []
            for i in range(batch_n):
                ins = pred_instances[i]["instances"]
                size = pred_instances[i]['instances'].scores.size(dim=0)
                if size>obj_n_max:
                    roi_input.append(ins.pred_boxes[:obj_n_max])
                else:
                    temp = Boxes(torch.zeros(obj_n_max-size,4).cuda(device))
                    pred_boxes = Boxes.cat([ins.pred_boxes,temp])
                    roi_input.append(pred_boxes)
            roi_pred = model.roi_heads.box_pooler(fpn_features[1:],roi_input).view(batch_n,20,-1)
        if pred_box:
            return features_maskformer.cpu(),roi_gt.cpu(), Boxes.cat(roi_input).tensor.view(batch_n,obj_n_max,-1).cpu()
        else:
            return features_maskformer,out_roi