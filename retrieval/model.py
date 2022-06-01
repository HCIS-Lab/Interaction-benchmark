import torch
from torch import nn


import sys
sys.path.append('/home/hankung/Desktop/Interaction_benchmark/models')

from models import *




def generate_model(model_name, num_cam, num_ego_class, num_actor_class):
	if model_name == 'cnnlstm_image':
		model = cnnlstm_image.CNNLSTM(num_cam, num_ego_class, num_actor_class)
	elif model_name == 'cnnlstm_backbone':
		model = cnnlstm_backbone.CNNLSTM(num_cam, num_ego_class, num_actor_class)
	elif model_name == 'slowfast':
		model = cnnlstm_backbone.CNNLSTM(num_cam, num_ego_class, num_actor_class)
	elif model_name == 'i3d':
		model = i3d.I3D(num_classes=opt.n_classes)
	elif model_name == 'orn':
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)
	elif model_name == 'arg':
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)
	elif model_name == 'topology':
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)

	return model.to(device)