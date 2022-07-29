import torch
from torch import nn
import numpy as np

import sys
sys.path.append('/home/hankung/Desktop/Interaction_benchmark/models')

# from models import *
import cnnlstm_image
import slowfast
# import arg
import cnnlstm_backbone
import cat_cnnlstm
import cnnpool
import fpnlstm
import convlstm




def generate_model(model_name, num_cam, num_ego_class, num_actor_class, seq_len, road):
	if model_name == 'cnnlstm_imagenet':
		model = cnnlstm_image.CNNLSTM(num_cam, num_ego_class, num_actor_class)
	elif model_name == 'cnnlstm_maskformer':
		model = cnnlstm_backbone.CNNLSTM_maskformer(num_cam, num_ego_class, num_actor_class, road)
	elif model_name == 'cat_cnnlstm':
		model = cat_cnnlstm.Cat_CNNLSTM_maskformer(num_cam, num_ego_class, num_actor_class, road)
	elif model_name == 'convlstm':
		model = convlstm.ConvLstm(num_cam, num_ego_class, num_actor_class, road)
	elif model_name == 'fpnlstm':
		model = fpnlstm.FPNLSTM(num_cam, num_ego_class, num_actor_class, road)
	elif model_name == 'cnnpool':
		model = cnnpool.CNNPOOL(num_cam, num_ego_class, num_actor_class, road)
	elif model_name == 'slowfast':
		model = slowfast.resnet50(num_ego_class, num_actor_class)
	elif model_name == 'i3d':
		model = i3d.I3D(num_classes=opt.n_classes)
	# elif model_name == 'arg':
	# 	model = arg.ARG(num_cam, num_ego_class, num_actor_class, seq_len)
	elif model_name == 'orn':
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)
	elif model_name == 'arg':
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)
	elif model_name == 'topology':
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)

	for param in model.parameters():
	    param.requires_grad = True
	try:

		for param in model.backbone.parameters():
		    param.requires_grad = False
		print('resnet no grad')
	except:
		print('no detection model')
	try:
		for param in model.seg_model.parameters():
		    param.requires_grad = False
		print('seg_model no grad')
	except:
		print('no seg_model')
	try:
		for param in model.det.parameters():
		    param.requires_grad = False
		print('det no grad')
	except:
		print('no detection model')

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print ('Total trainable parameters: ', params)

	return model