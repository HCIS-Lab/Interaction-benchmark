import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np

from backbone import *
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
from roi_align.roi_align import CropAndResize # crop_and_resize module
from MaskFormer.mask_former import mask_former_model


class GCN_Module(nn.Module):
    def __init__(self, cfg):
        super(GCN_Module, self).__init__()
                
        # number of feature relation
        num_feature_relation = 256
        
        # number of graph
        num_graph=16

        # number of boxes
        N=12

        T=10
        
        # num_feature_gcn
        num_features_gcn=1024

        NFG_ONE=NFG
        
        self.fc_rn_theta_list=torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])
        self.fc_rn_phi_list=torch.nn.ModuleList([nn.Linear(NFG, NFR) for i in range(NG)])
        
        
        self.fc_gcn_list=torch.nn.ModuleList([nn.Linear(NFG, NFG_ONE, bias=False) for i in range(NG)])
        
        if cfg.dataset_name=='volleyball':
            self.nl_gcn_list=torch.nn.ModuleList([nn.LayerNorm([T*N,NFG_ONE]) for i in range(NG)])
        else:
            self.nl_gcn_list=torch.nn.ModuleList([nn.LayerNorm([NFG_ONE]) for i in range(NG)])
        
            

        
    def forward(self, graph_boxes_features, boxes_in_flat):
        """
        graph_boxes_features  [B*T, N, NFG (num_feature_gcn)]
        """
        
        # GCN graph modeling
        # Prepare boxes similarity relation
        B, N, num_feature_gcn = graph_boxes_features.shape
        num_feature_relation = 256
        num_graph = 16
        NFG_ONE = num_feature_gcn
        
        OH, OW = self.cfg.out_size
        pos_threshold = 0.2
        
        # Prepare position mask
        graph_boxes_positions = boxes_in_flat  #B*T*N, 4
        graph_boxes_positions[:,0] = (graph_boxes_positions[:,0] + graph_boxes_positions[:,2]) / 2 
        graph_boxes_positions[:,1] = (graph_boxes_positions[:,1] + graph_boxes_positions[:,3]) / 2 
        graph_boxes_positions = graph_boxes_positions[:,:2].reshape(B,N,2)  #B*T, N, 2
        
        graph_boxes_distances = calc_pairwise_distance_3d(graph_boxes_positions,graph_boxes_positions)  #B, N, N
        
        position_mask = (graph_boxes_distances > (pos_threshold*OW))
        
        
        relation_graph = None
        graph_boxes_features_list = []
        for i in range(NG):
            graph_boxes_features_theta = self.fc_rn_theta_list[i](graph_boxes_features)  #B,N,NFR
            graph_boxes_features_phi = self.fc_rn_phi_list[i](graph_boxes_features)  #B,N,NFR

#             graph_boxes_features_theta=self.nl_rn_theta_list[i](graph_boxes_features_theta)
#             graph_boxes_features_phi=self.nl_rn_phi_list[i](graph_boxes_features_phi)

            similarity_relation_graph = torch.matmul(graph_boxes_features_theta,graph_boxes_features_phi.transpose(1,2))  #B,N,N

            similarity_relation_graph = similarity_relation_graph/np.sqrt(NFR)

            similarity_relation_graph = similarity_relation_graph.reshape(-1,1)  #B*N*N, 1
            
        
        
            # Build relation graph
            relation_graph = similarity_relation_graph

            relation_graph = relation_graph.reshape(B,N,N)

            relation_graph[position_mask] =- float('inf')

            relation_graph = torch.softmax(relation_graph,dim=2)       
        
            # Graph convolution
            one_graph_boxes_features = self.fc_gcn_list[i]( torch.matmul(relation_graph,graph_boxes_features) )  #B, N, NFG_ONE
            one_graph_boxes_features = self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features = F.relu(one_graph_boxes_features)
            
            graph_boxes_features_list.append(one_graph_boxes_features)
        
        graph_boxes_features = torch.sum(torch.stack(graph_boxes_features_list),dim=0) #B, N, NFG
        
        return graph_boxes_features,relation_graph

class ARG(nn.Module):
    """
    main module of GCN for the volleyball dataset
    """
    def __init__(self, cfg):
        super(ARG, self).__init__()
        
        T, N = 10, self.cfg.num_boxes
        num_emb = 1024
        crop_size = self.cfg.crop_size[0]
        num_features_boxes = 1024
        num_features_relation, num_features_gcn= 256, 1024
        num_graph = 16
        self.gcn_layers = 1
        
        self.backbone = mask_former_model.

        
        self.roi_align = RoIAlign(*self.cfg.crop_size)
        
        self.fc_emb_1 = nn.Linear(crop_size*crop_size*num_emb, num_features_boxes)
        self.nl_emb_1 = nn.LayerNorm([num_features_boxes])
        
        
        self.gcn_list = torch.nn.ModuleList([GCN_Module(self.cfg) for i in range(self.gcn_layers) ])    
        
        
        # self.dropout_global=nn.Dropout(p=self.cfg.train_dropout_prob)
    
        self.fc_actions = nn.Linear(num_features_gcn, self.cfg.num_actions)
        self.fc_activities = nn.Linear(num_features_gcn, self.cfg.num_activities)
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
                    
    def loadmodel(self,filepath):
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        self.fc_emb_1.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from: ',filepath)
        
                
    def forward(self, batch_data):
        images_in, boxes_in = batch_data
        
        # read config parameters
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFR, NFG=self.cfg.num_features_relation, self.cfg.num_features_gcn
        NG = self.cfg.num_graph
        
        D = 1024
        K = self.cfg.crop_size[0]
        
        
        if not self.training:
            B = B*3
            T = T//3
            images_in.reshape((B,T)+images_in.shape[2:])
            boxes_in.reshape((B,T)+boxes_in.shape[2:])
        
        
        # Reshape the input data
        images_in_flat = torch.reshape(images_in,(B*T,3,H,W))  #B*T, 3, H, W
        boxes_in_flat = torch.reshape(boxes_in,(B*T*N,4))  #B*T*N, 4

        boxes_idx = [i * torch.ones(N, dtype=torch.int)   for i in range(B*T) ]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # B*T, N
        boxes_idx_flat = torch.reshape(boxes_idx,(B*T*N,))  #B*T*N,
        
        # Use backbone to extract features of images_in
        # Pre-precess first
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)
            
        
        # Build  features
        assert outputs[0].shape[2:4] == torch.Size([OH,OW])
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4]!=torch.Size([OH,OW]):
                features=F.interpolate(features, size=(OH,OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        
        features_multiscale=torch.cat(features_multiscale, dim=1)  #B*T, D, OH, OW
        
        
        # RoI Align
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features=self.roi_align(features_multiscale,
                                            boxes_in_flat,
                                            boxes_idx_flat)  #B*T*N, D, K, K,
        
        boxes_features=boxes_features.reshape(B, T, N, -1)  #B,T,N, D*K*K
        
        
        # Embedding 
        boxes_features=self.fc_emb_1(boxes_features)  # B,T,N, NFB
        boxes_features=self.nl_emb_1(boxes_features)
        boxes_features=F.relu(boxes_features)
        
        
        
        # GCN       
        graph_boxes_features=boxes_features.reshape(B, T*N, NFG)
        
#         visual_info=[]
        for i in range(len(self.gcn_list)):
            graph_boxes_features, relation_graph = self.gcn_list[i](graph_boxes_features, boxes_in_flat)
#             visual_info.append(relation_graph.reshape(B,T,N,N))
        
        
       
        # fuse graph_boxes_features with boxes_features
        graph_boxes_features = graph_boxes_features.reshape(B, T, N, NFG)  
        boxes_features = boxes_features.reshape(B, T, N, NFB)
        
#         boxes_states= torch.cat( [graph_boxes_features,boxes_features],dim=3)  #B, T, N, NFG+NFB
        boxes_states = graph_boxes_features + boxes_features
    
        boxes_states = self.dropout_global(boxes_states)

        NFS = NFG
        
        # Predict actions
        boxes_states_flat = boxes_states.reshape(-1,NFS)  #B*T*N, NFS
        actions_scores = self.fc_actions(boxes_states_flat)  #B*T*N, actn_num
        
        # Predict activities
        boxes_states_pooled, _ = torch.max(boxes_states, dim=2)  
        boxes_states_pooled_flat = boxes_states_pooled.reshape(-1, NFS)  
        
        activities_scores = self.fc_activities(boxes_states_pooled_flat)  #B*T, acty_num
        
        # Temporal fusion
        actions_scores = actions_scores.reshape(B, T, N,-1)
        actions_scores = torch.mean(actions_scores, dim=1).reshape(B*N, -1)
        activities_scores = activities_scores.reshape(B, T, -1)
        activities_scores = torch.mean(activities_scores, dim=1).reshape(B,-1)
        
        if not self.training:
            B = B//3
            actions_scores = torch.mean(actions_scores.reshape(B, 3, N, -1),dim=1).reshape(B*N, -1)
            activities_scores = torch.mean(activities_scores.reshape(B, 3, -1),dim=1).reshape(B, -1)
       
       
        return actions_scores, activities_scores
       
def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B = X.shape[0]
    
    rx = X.pow(2).sum(dim=2).reshape((B, -1, 1))
    ry = Y.pow(2).sum(dim=2).reshape((B, -1, 1))
    
    dist = rx-2.0*X.matmul(Y.transpose(1,2)) + ry.transpose(1,2)
    
    return torch.sqrt(dist)
        