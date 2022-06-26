import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np

from backbone import *
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
from roi_align.roi_align import CropAndResize # crop_and_resize module
from MaskFormer.demo.demo import get_maskformer
from maskrcnn import get_maskrcnn
from retrieval_head import Head
from detectron2.structures import Boxes


class GCN_Module(nn.Module):
    def __init__(self, num_box, seq_len):
        super(GCN_Module, self).__init__()
                
        # number of feature relation
        self.num_feature_relation = 512
        
        # number of graph
        self.num_graph = 12

        # number of boxes
        self.num_box = num_box

        self.seq_len = seq_len
        
        # num_feature_gcn
        self.num_feature_gcn = 512

        self.NFG_ONE = self.num_feature_gcn
        
        self.fc_rn_theta_list = torch.nn.ModuleList([nn.Linear(self.num_feature_gcn, self.num_feature_relation) for i in range(self.num_graph)])
        self.fc_rn_phi_list = torch.nn.ModuleList([nn.Linear(self.num_feature_gcn, self.num_feature_relation) for i in range(self.num_graph)])
        
        
        self.fc_gcn_list = torch.nn.ModuleList([nn.Linear(self.num_feature_gcn, self.NFG_ONE, bias=False) for i in range(self.num_graph)])
        
        # if cfg.dataset_name == 'volleyball':
        self.nl_gcn_list = torch.nn.ModuleList([nn.LayerNorm([self.seq_len*self.num_box, self.NFG_ONE]) for i in range(self.num_graph)])
        # else:
        #     self.nl_gcn_list = torch.nn.ModuleList([nn.LayerNorm([NFG_ONE]) for i in range(num_graph)])
        
            

        
    def forward(self, graph_box_feature, box_in_flat, crop_size):
        """
        graph_boxes_features  [B*T, N, NFG (num_feature_gcn)]
        """
        
        # GCN graph modeling
        # Prepare boxes similarity relation
        batch_size, seq_len_x_num_box = graph_box_feature.shape[0], graph_box_feature.shape[1]
        # print(graph_box_feature.shape)
        
        OH, OW = crop_size
        pos_threshold = 0.2
        
        # Prepare position mask
        graph_box_positions = box_in_flat  #B*T*N, 4
        graph_box_positions[:,0] = (graph_box_positions[:,0] + graph_box_positions[:,2]) / 2 
        graph_box_positions[:,1] = (graph_box_positions[:,1] + graph_box_positions[:,3]) / 2 
        graph_box_positions = graph_box_positions[:,:2].reshape(batch_size, seq_len_x_num_box, 2)  #B*T, N, 2
        
        graph_box_distances = calc_pairwise_distance_3d(graph_box_positions, graph_box_positions)  #B, N, N
        
        position_mask = (graph_box_distances > (pos_threshold*OW))
        
        
        relation_graph = None
        graph_box_feature_list = []
        for i in range(self.num_graph):
            graph_box_feature_theta = self.fc_rn_theta_list[i](graph_box_feature)  #B,N,NFR
            graph_box_feature_phi = self.fc_rn_phi_list[i](graph_box_feature)  #B,N,NFR

#             graph_boxes_features_theta=self.nl_rn_theta_list[i](graph_boxes_features_theta)
#             graph_boxes_features_phi=self.nl_rn_phi_list[i](graph_boxes_features_phi)

            similarity_relation_graph = torch.matmul(graph_box_feature_theta, graph_box_feature_phi.transpose(1,2))  #B,N,N

            similarity_relation_graph = similarity_relation_graph/np.sqrt(self.num_feature_relation)

            similarity_relation_graph = similarity_relation_graph.reshape(-1,1)  #B*N*N, 1
            
        
        
            # Build relation graph
            relation_graph = similarity_relation_graph

            relation_graph = relation_graph.reshape(batch_size, seq_len_x_num_box, seq_len_x_num_box)

            relation_graph[position_mask] =- float('inf')

            relation_graph = torch.softmax(relation_graph, dim=2)       
        
            # Graph convolution
            one_graph_box_feature = self.fc_gcn_list[i](torch.matmul(relation_graph, graph_box_feature))  #B, N, NFG_ONE
            one_graph_box_feature = self.nl_gcn_list[i](one_graph_box_feature)
            one_graph_box_feature = F.relu(one_graph_box_feature)
            
            graph_box_feature_list.append(one_graph_box_feature)
        
        graph_box_feature = torch.sum(torch.stack(graph_box_feature_list), dim=0) #B, N, NFG
        
        return graph_box_feature,relation_graph

class ARG(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, num_cam, num_ego_class, num_actor_class, seq_len):
        super(ARG, self).__init__()
        
        self.num_box = 10
        num_emb = 256
        
        self.seq_len = seq_len

        self.num_feature_box = 512
        self.num_feature_relation, self.num_feature_gcn = 512, 512
        self.num_graph = 8
        self.gcn_layers = 1

        self.num_cam = num_cam
        self.num_ego_class = num_ego_class
        self.num_actor_class = num_actor_class
        
        self.seg_model = get_maskformer()
        # self.backbone = self.seg_model.backbone
        self.det = get_maskrcnn()
        
        crop_size = 7
        self.roi_align = RoIAlign(crop_size, crop_size)
        
        self.fc_emb_1 = nn.Linear(crop_size*crop_size*num_emb, self.num_feature_box)
        self.nl_emb_1 = nn.LayerNorm([self.num_feature_box])
        
        
        self.gcn_list = torch.nn.ModuleList([GCN_Module(self.num_box, self.seq_len) for i in range(self.gcn_layers) ])    
        
        
        # self.dropout_global=nn.Dropout(p=self.cfg.train_dropout_prob)
    
        # self.fc_actions = nn.Linear(num_features_gcn, self.cfg.num_actions)
        # self.fc_activities = nn.Linear(selfnum_features_gcn, self.cfg.num_activities)
        self.head = Head(self.num_feature_gcn, num_ego_class, num_actor_class)
        self.relu = nn.ReLU(inplace=True)
        # self.norm1 = nn.BatchNorm2d(self.num_feature_gcn)
        # self.norm2 = nn.BatchNorm2d(self.num_feature_gcn)
        # self.norm3 = nn.BatchNorm2d(self.num_feature_gcn)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
                    
    def loadmodel(self, filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ', filepath)
        
                
    def train_forward(self, images_in):
        
        self.seq_len = len(images_in)//self.num_cam
        batch_size = images_in[0].shape[0]
        H, W = images_in[0].shape[2], images_in[0].shape[3]

        # images_in = images_in.view(batch_size*self.num_cam*self.seq_len, 3, H, W)


        with torch.no_grad():
            # images_in = F.interpolate(images_in, size=(OW,OH), mode='bilinear', align_corners=True)
            image_list = [{'image': image} for image in images_in]
            image_rcnn = self.det.preprocess_image(image_list)
            features = self.det.backbone(image_rcnn.tensor)  # set of cnn features
            proposals, _ = self.det.proposal_generator(image_rcnn, features, None)  # RPN
            features_ = [features[f] for f in self.det.roi_heads.in_features]
            box_features = self.det.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
            box_features = self.det.roi_heads.box_head(box_features)  # features of all 1k candidates
            predictions = self.det.roi_heads.box_predictor(box_features)
            pred_instances, _ = self.det.roi_heads.box_predictor.inference(predictions, proposals)
            pred_instances = self.det.roi_heads.forward_with_given_boxes(features, pred_instances)
            # output boxes, masks, scores, etc
            feature_size = [(H, W)]* len(pred_instances)
            pred_instances = self.det._postprocess(pred_instances, image_list, feature_size)  # scale box to orig size
            boxes_in = []

            for box in pred_instances:
                size = box['instances'].scores.size(dim=0)
                ins = box["instances"]
                if size < self.num_box:
                    temp = Boxes(torch.zeros(self.num_box-size, 4).cuda())
                    pred_boxes = Boxes.cat([ins.pred_boxes, temp])
                    boxes_in.append(pred_boxes)
                else:
                    boxes_in.append(ins.pred_boxes[:self.num_box])

            # boxes_in = torch.stack(boxes_in, dim=0)
        num_feature_relation, num_feature_gcn = self.num_feature_relation, self.num_feature_gcn
        num_graph = self.num_graph  
        
        # if not self.training:
        #     batch_size = batch_size*3
        #     self.seq_len = self.seq_len//3
        #     images_in.reshape((batch_size, self.seq_len) + images_in.shape[2:])
        #     boxes_in.reshape((batch_size, self.seq_len) + boxes_in.shape[2:])
        
        # Reshape the input data
        images_in_flat = torch.reshape(images_in, (batch_size*self.seq_len, 3, H, W))  #B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in, (batch_size*self.seq_len*self.num_box, 4))  #B*T*N, 4

        boxes_idx = [i * torch.ones(self.num_box, dtype=torch.int) for i in range(batch_size*self.seq_len)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx, (batch_size*self.seq_len*self.num_box,))  #B*T*N,
        
        # Use backbone to extract features of images_in
        # Pre-precess first
        # images_in_flat = prep_images(images_in_flat)
        images_in_flat = normalize_imagenet(images_in_flat)
        # print(images_in_flat.shape)
        outputs = self.seg_model.get_fpn_features(images_in_flat, no_dict=True)
            
        
        # Build  features
        
        # assert outputs[0].shape[2:4] == torch.Size([OH,OW])
        features_multiscale = []
        outputs = outputs[1:]
        OH, OW = outputs[0].shape[2], outputs[0].shape[3]

        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features = F.interpolate(features, size=(OH,OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale = torch.cat(features_multiscale, dim=1)  #B*T, D, OH, OW
        # print(features_multiscale.shape)
        
        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*N, D, K, K,
        boxes_features = self.det.roi_heads.box_pooler(outputs, boxes_in)
        boxes_features = boxes_features.reshape(batch_size, self.seq_len, self.num_box, -1)  #B,T,N, D*K*K
        
        
        # Embedding 
        boxes_features = self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features = self.nl_emb_1(boxes_features)
        boxes_features = F.relu(boxes_features)
                
        # GCN       
        graph_boxes_features = boxes_features.reshape(batch_size, self.seq_len*self.num_box, self.num_feature_gcn)
        
#         visual_info=[]
        for i in range(len(self.gcn_list)):
            graph_boxes_features, relation_graph = self.gcn_list[i](graph_boxes_features, boxes_in, (OH, OW))
#             visual_info.append(relation_graph.reshape(B,T,N,N))
        
        
       
        # fuse graph_boxes_features with boxes_features
        graph_boxes_features = graph_boxes_features.reshape(batch_size, self.seq_len, self.num_box, self.num_feature_gcn)  
        boxes_features = boxes_features.reshape(batch_size, self.seq_len, self.num_box, self.num_feature_box)
        
#         boxes_states= torch.cat( [graph_boxes_features,boxes_features],dim=3)  #B, T, N, NFG+NFB
        boxes_states = graph_boxes_features + boxes_features
    
        # boxes_states = self.dropout_global(boxes_states)

        
        # Predict actions
        #B*T*N, NFS
        # boxes_states = self.relu(boxes_states)
        # boxes_states = self.norm1(boxes_states)
        boxes_states = boxes_states.reshape(batch_size, self.seq_len, self.num_box, self.num_feature_gcn)  #B, T, N*NFS
        boxes_states = torch.mean(boxes_states, dim=1) #B, N*NFS
        boxes_states = self.relu(boxes_states)
        # boxes_states = self.norm2(boxes_states)
        boxes_states = torch.sum(boxes_states, dim=1)
        boxes_states = self.relu(boxes_states)
        # boxes_states = self.norm3(boxes_states)
        # actions_scores = self.fc_actions(boxes_states_flat)  #B*T*N, actn_num
        
        # # Predict activities
        # boxes_states_pooled, _ = torch.max(boxes_states, dim=2)  
        # boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, self.num_feature_gcn)  
        
        ego, actor = self.head(boxes_states)
        # activities_scores = self.fc_activities(boxes_states_pooled_flat)  #B*T, acty_num
        
        # # Temporal fusion
        # actions_scores = actions_scores.reshape(batch_size, self.seq_len, self.num_box, -1)
        # actions_scores = torch.mean(actions_scores, dim=1).reshape(batch_size*self.num_box, -1)
        # activities_scores = activities_scores.reshape(batch_size, self.seq_len, -1)
        # activities_scores = torch.mean(activities_scores, dim=1).reshape(batch_size, -1)
        # ego, actor = ego.reshape(batch_size, self.seq_len, -1), actor.reshape(batch_size, self.seq_len, -1)
        # ego, actor = torch.mean(ego, dim=1).reshape(batch_size, -1), torch.mean(actor, dim=1).reshape(batch_size, -1)
        return ego, actor
        # if not self.training:
        #     batch_size = batch_data//3
        #     actions_scores = torch.mean(actions_scores.reshape(batch_size, 3, self.num_box, -1),dim=1).reshape(batch_size*self.num_box, -1)
        #     activities_scores = torch.mean(activities_scores.reshape(batch_size, 3, -1),dim=1).reshape(batch_size, -1)
       
       
        # return actions_scores, activities_scores
       
def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    batch_size = X.shape[0]
    
    rx = X.pow(2).sum(dim=2).reshape((batch_size, -1, 1))
    ry = Y.pow(2).sum(dim=2).reshape((batch_size, -1, 1))
    
    dist = rx-2.0*X.matmul(Y.transpose(1,2)) + ry.transpose(1,2)
    
    return torch.sqrt(dist)

def prep_images(images):
    """
    preprocess images
    Args:
        images: pytorch tensor
    """
    images = images.div(255.0)
    
    images = torch.sub(images,0.5)
    images = torch.mul(images,2.0)
    
    return images

def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x

        