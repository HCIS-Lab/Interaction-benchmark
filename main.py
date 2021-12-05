import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
import os
from os import walk
import random
import math
import numpy as np
import threading
from multiprocessing.pool import ThreadPool

import copy

from PIL import Image, ImageOps, ImageFilter

from train import train_epoch
from torch.utils.data import DataLoader, TensorDataset
from validation import val_epoch
from opts import parse_opts
from model import generate_model
from torch.optim import lr_scheduler
# from dataset import get_training_set, get_validation_set
from mean import get_mean, get_std
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

# from spatial_transforms import (
# 	Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
# 	MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
# from temporal_transforms import LoopPadding, TemporalRandomCrop
# from target_transforms import ClassLabel, VideoID
# from target_transforms import Compose as TargetCompose

def frame_crop(files, n_frames):
	n_files = len(files)
	step = int(max(1, math.floor(n_files+1)/n_frames))
	new_video = []
	for j in range(0, n_files, step):
		new_video.append(files[j])
		if len(new_video) == n_frames:
			break
		
	# for j in range(1, n_frames, step):
	# 	sample_j = copy.deepcopy(sample)
	# 	sample_j['frame_indices'] = list(
	# 		range(j, min(n_frames + 1, j + sample_duration)))
	# 	dataset.append(sample_j)
	return new_video

def open_img(root_path, files):
	dataset = []

	for i, file in enumerate(files):
		img = Image.open(os.path.join(root_path, file)).convert('RGB')
		# img = np.array(img).astype(np.float32).transpose((2, 0, 1))
		img = FixedResize(img)
		img = np.array(img).astype(np.float32).transpose((2, 0, 1))
		# img = torch.from_numpy(img).float()
		dataset.append(img)
	return dataset

def FixedResize(img, size=224):
    """change the short edge length to size"""
    # print(img.size)
    w, h = img.size
    # if w > h:
    #     oh = size
    #     ow = int(1.0 * w * oh / h)
    # else:
    #     ow = size
    #     oh = int(1.0 * h * ow / w)
    img = img.resize((size,size), Image.BILINEAR)
    return img

def get_gt(gt):
	gt = np.array(gt).astype(np.float32)
	gt = torch.from_numpy(gt).float()
	return gt

def carla_dataset(root_path, n_frames=8, train_split=0.8):
	video_names = np.load('scenario_id.npy')
	annotations = np.load('gt.npy')
	# num_scenario = video_names.shape[0]
	dataset_front = []
	dataset_right = []
	dataset_left = []
	dataset_gt = []  
	video_names = tqdm(video_names)
	for i, name in enumerate(video_names):
		scenario_name = os.path.join(root_path, str(name))

		front_files = next(walk(os.path.join(scenario_name, 'front')), (None, None, []))[2]
		right_files = next(walk(os.path.join(scenario_name, 'right')), (None, None, []))[2]
		left_files = next(walk(os.path.join(scenario_name, 'left')), (None, None, []))[2]
		if len(front_files) < n_frames or len(right_files) < n_frames or len(left_files) < n_frames:
			print(name)
			continue
		# front_files = [f for f in os.path.listdir(os.path.join(scenario_name, 'front')) if isfile(os.path.join(scenario_name, 'front', f))]
		# right_files = [f for f in os.path.listdir(os.path.join(scenario_name, 'right')) if isfile(os.path.join(scenario_name, 'right', f))]
		# left_files = [f for f in os.path.listdir(os.path.join(scenario_name, 'left')) if isfile(os.path.join(scenario_name, 'left', f))]
		front_files = frame_crop(front_files, n_frames)
		right_files = frame_crop(right_files, n_frames)
		left_files = frame_crop(left_files, n_frames)
		front_files.sort()
		right_files.sort()
		left_files.sort()
		
		pool1 = ThreadPool(processes=10)
		pool2 = ThreadPool(processes=10)
		pool3 = ThreadPool(processes=10)
		p1_data = pool1.apply_async(open_img, (os.path.join(scenario_name, 'front'), front_files))
		p2_data = pool1.apply_async(open_img, (os.path.join(scenario_name, 'right'), right_files))
		p3_data = pool1.apply_async(open_img, (os.path.join(scenario_name, 'left'), left_files))


		dataset_front.append(p1_data.get())
		dataset_right.append(p2_data.get())
		dataset_left.append(p3_data.get())
		dataset_gt.append(annotations[i])

	dataset_front = np.stack(dataset_front, 0)
	dataset_right = np.stack(dataset_right, 0)
	dataset_left = np.stack(dataset_left, 0)
	dataset_gt = np.stack(dataset_gt, 0)
	dataset_front = torch.from_numpy(dataset_front).float()
	dataset_right = torch.from_numpy(dataset_right).float()
	dataset_left = torch.from_numpy(dataset_left).float()
	dataset_gt = torch.from_numpy(dataset_gt).long()
	num_train_samples = int(dataset_front.shape[0]*train_split)

	train_data = [dataset_front[:num_train_samples], dataset_right[:num_train_samples],
					dataset_left[:num_train_samples], dataset_gt[:num_train_samples]]
	eval_data = [dataset_front[num_train_samples:], dataset_right[num_train_samples:],
					dataset_left[num_train_samples:], dataset_gt[num_train_samples:]]
	return train_data, eval_data

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[3]), reverse=True)
    front, right, left, gt = zip(*data)

    pad_label = []
    # lens = []
    max_len = len(gt[0])
    for i in range(len(label)):
        temp_label = label[i]
        temp_label += [37] * (max_len - len(label[i]))
        pad_label.append(temp_label)
        # lens.append(len(label[i]))
    return front, right, left, pad_label 

def resume_model(opt, model, optimizer):
	""" Resume model 
	"""
	checkpoint = torch.load(opt.resume_path)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print("Model Restored from Epoch {}".format(checkpoint['epoch']))
	start_epoch = checkpoint['epoch'] + 1
	return start_epoch


# def get_loaders(opt):
# 	""" Make dataloaders for train and validation sets
# 	"""
# 	# train loader
# 	opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
# 	if opt.no_mean_norm and not opt.std_norm:
# 		norm_method = Normalize([0, 0, 0], [1, 1, 1])
# 	elif not opt.std_norm:
# 		norm_method = Normalize(opt.mean, [1, 1, 1])
# 	else:
# 		norm_method = Normalize(opt.mean, opt.std)
# 	spatial_transform = Compose([
# 		# crop_method,
# 		Scale((opt.sample_size, opt.sample_size)),
# 		# RandomHorizontalFlip(),
# 		ToTensor(opt.norm_value), norm_method
# 	])
# 	temporal_transform = TemporalRandomCrop(16)
# 	target_transform = ClassLabel()
# 	training_data = get_training_set(opt, spatial_transform,
# 									 temporal_transform, target_transform)
# 	train_loader = torch.utils.data.DataLoader(
# 		training_data,
# 		batch_size=opt.batch_size,
# 		shuffle=True,
# 		num_workers=opt.num_workers,
# 		pin_memory=True)

# 	# validation loader
# 	spatial_transform = Compose([
# 		Scale((opt.sample_size, opt.sample_size)),
# 		# CenterCrop(opt.sample_size),
# 		ToTensor(opt.norm_value), norm_method
# 	])
# 	target_transform = ClassLabel()
# 	temporal_transform = LoopPadding(16)
# 	validation_data = get_validation_set(
# 		opt, spatial_transform, temporal_transform, target_transform)
# 	val_loader = torch.utils.data.DataLoader(
# 		validation_data,
# 		batch_size=opt.batch_size,
# 		shuffle=False,
# 		num_workers=opt.num_workers,
# 		pin_memory=True)
# 	return train_loader, val_loader


def main_worker():
	opt = parse_opts()
	print(opt)

	seed = 1
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	print('setting devie')
	# CUDA for PyTorch
	device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")
	print('setting writer')
	# tensorboard
	summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')

	print('loading model')
	# defining model
	model =  generate_model(opt, device)
	# get data loaders
	print('data loading')
	root_path = '/home/hankung/Desktop/carla_911/CARLA_0.9.11/PythonAPI/examples/_out/Camera RGB'
	train_dataset, eval_dataset = carla_dataset(root_path)
	train_dataset = TensorDataset(train_dataset[0], train_dataset[1], train_dataset[2], train_dataset[3])
	train_loader = DataLoader(
	    train_dataset,
	    batch_size=opt.batch_size,
	    shuffle=True,
		num_workers=opt.num_workers,
		pin_memory=True,
		drop_last=True,
		collate_fn=collate_fn
	)
	eval_dataset = TensorDataset(eval_dataset[0], eval_dataset[1], eval_dataset[2], eval_dataset[3])
	eval_loader = DataLoader(
	    eval_dataset,
	    batch_size=1,
	    shuffle=False,
		num_workers=opt.num_workers,
		pin_memory=True,
		collate_fn=collate_fn
	)

	# train_loader, val_loader = get_loaders(opt)
	print('DataLoader done')
	# optimizer
	crnn_params = list(model.parameters())
	optimizer = torch.optim.Adam(crnn_params, lr=opt.lr_rate, weight_decay=opt.weight_decay)

	# scheduler = lr_scheduler.ReduceLROnPlateau(
	# 	optimizer, 'min', patience=opt.lr_patience)
	criterion = nn.CrossEntropyLoss(ignore_index=37)

	# resume model
	if opt.resume_path:
		start_epoch = resume_model(opt, model, optimizer)
	else:
		start_epoch = 1
	# start training
	for epoch in range(start_epoch, opt.n_epochs + 1):
		train_loss = train_epoch(
			model, train_loader, criterion, optimizer, epoch, opt.log_interval, device)
		is_print=False
		# if epoch == opt.n_epochs:
		# 	is_print = True
		eval_loss, val_acc = val_epoch(
			model, eval_loader, criterion, device, is_print)

		# saving weights to checkpoint
		if (epoch) % opt.save_interval == 0:
			# scheduler.step(val_loss)
			# write summary
			summary_writer.add_scalar(
				'losses/train_loss', train_loss, global_step=epoch)
			# summary_writer.add_scalar(
			# 	'losses/val_loss', val_loss, global_step=epoch)
			# summary_writer.add_scalar(
			# 	'acc/train_acc', train_acc * 100, global_step=epoch)
			# summary_writer.add_scalar(
			# 	'acc/val_acc', val_acc * 100, global_step=epoch)

			state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
			torch.save(state, os.path.join('snapshots', f'{opt.model}-Epoch-{epoch}.pth'))
			print("Epoch {} model saved!\n".format(epoch))

if __name__ == "__main__":
	main_worker()


