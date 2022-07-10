import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn

torch.backends.cudnn.benchmark = True

import sys
sys.path.append('/home/hankung/Desktop/Interaction_benchmark/datasets')
sys.path.append('/home/hankung/Desktop/Interaction_benchmark/config')
sys.path.append('/home/hankung/Desktop/Interaction_benchmark/models')

# from .configs.config import GlobalConfig
import feature_data


from sklearn.metrics import average_precision_score, precision_score, f1_score, recall_score, accuracy_score, hamming_loss
# from torchmetrics import F1Score

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from PIL import Image

# import cnnlstm_backbone
from model import generate_model
from torchvision import models

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='cnnlstm_imagenet', help='Unique experiment identifier.')

parser.add_argument('--front_only', help="", action="store_true")
parser.add_argument("--top", help="", action="store_true")

parser.add_argument('--seg', help="", action="store_true")
parser.add_argument('--road', help="", action="store_true")
parser.add_argument('--lss', help="", action="store_true")

# parser.add_argument('--seg', type=bool, default=False, help='')

parser.add_argument('--seq_len', type=int, default=16, help='')
parser.add_argument('--scale', type=float, default=4, help='')
parser.add_argument('--bce', type=float, default=1, help='')
parser.add_argument('--weight', type=float, default=15, help='')



parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--test', help="", action="store_true")
parser.add_argument('--viz', help="", action="store_true")


args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)
print(args)
writer = SummaryWriter(log_dir=args.logdir)


class Engine(object):
	"""Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""

	def __init__(self,  cur_epoch=0, cur_iter=0, top=False, front_only=False, bce_weight=1, road_topo=False):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.top = top
		self.front_only = front_only
		self.road_topo = road_topo
		self.train_loss = []
		self.val_loss = []
		self.bestval = 1e10
		self.bce_weight = bce_weight

	def train(self):
		loss_epoch = 0.
		ce_loss_epoch = 0.
		bce_loss_epoch = 0.
		road_loss_epoch = 0.
		num_batches = 0

		model.train()
		# Train loop
		for data in tqdm(dataloader_train):
			
			# efficiently zero gradients
			for p in model.parameters():
				p.grad = None

			# create batch and move to GPU
			fronts_in = data['fronts']
			lefts_in = data['lefts']
			rights_in = data['rights']
			if self.top:
				tops_in = data['tops']

			fronts = []
			lefts = []
			rights = []
			tops = []

			for i in range(seq_len):
				if self.top:
					tops.append(tops_in[i].to(args.device, dtype=torch.float32))
				else:
					fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
					if not self.front_only:
						lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
						rights.append(rights_in[i].to(args.device, dtype=torch.float32))

			# labels
			road = torch.FloatTensor(data['road_para']).to(args.device)
			batch_size = road.shape[0]
			road = road.view(batch_size*seq_len, 11)
			ego = data['ego'].to(args.device)
			actor = torch.FloatTensor(data['actor']).to(args.device)

			if self.road_topo:
				pred_ego, pred_actor, pred_road_para = model.train_forward(fronts+lefts+rights+tops)
			else:
				pred_ego, pred_actor = model.train_forward(fronts+lefts+rights+tops)

			pos_weight = torch.ones([num_actor_class])*args.weight
			ce = nn.CrossEntropyLoss(reduction='mean').cuda()
			bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight).cuda()
			road_bce = nn.BCEWithLogitsLoss(reduction='mean').cuda()

			# print(pred_actor[0].data)
			ce_loss = ce(pred_ego, ego)
			bce_loss = bce(pred_actor, actor)

			if self.road_topo:
				road_loss = (road_bce(pred_road_para, road))
			else:
				road_loss = 0.

			loss = ce_loss + self.bce_weight * bce_loss + road_loss
			loss.backward()

			loss_epoch += float(loss.item())
			ce_loss_epoch += float(ce_loss.item())
			bce_loss_epoch += float(bce_loss.item())
			if self.road_topo:
				road_loss_epoch += float(road_loss.item())

			num_batches += 1

			optimizer.step()
			optimizer.zero_grad()
			writer.add_scalar('train_loss', loss.item(), self.cur_iter)
			self.cur_iter += 1
		
		
		loss_epoch = loss_epoch / num_batches
		ce_loss_epoch = ce_loss_epoch / num_batches
		bce_loss_epoch = bce_loss_epoch / num_batches
		road_loss_epoch = road_loss_epoch / num_batches
		print('total loss')
		print(loss_epoch)
		print('ce loss:')
		print(ce_loss_epoch)
		print('bce loss:')
		print(bce_loss_epoch)

		if self.road_topo:
			print('road loss:')
			print(road_loss_epoch)

		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1

	def validate(self, cam=False):
		model.eval()
		with torch.no_grad():	
			num_batches = 0
			loss = 0.

			total_ego = 0
			total_actor = 0

			correct_ego = 0
			correct_actor = 0
			mean_f1 = 0
			label_actor_list = []
			pred_actor_list = []
			label_road_para_list = []
			pred_road_para_list = []

			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_val), 0):
				
				id = data['id']
				v = data['variants']
				# create batch and move to GPU
				fronts_in = data['fronts']
				lefts_in = data['lefts']
				rights_in = data['rights']
				if self.top:
					tops_in = data['tops']

				fronts = []
				lefts = []
				rights = []
				tops = []

				for i in range(seq_len):
					if self.top:
						tops.append(tops_in[i].to(args.device, dtype=torch.float32))
					else:
						fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
						if not self.front_only:
							lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
							rights.append(rights_in[i].to(args.device, dtype=torch.float32))

				road = torch.FloatTensor(data['road_para']).to(args.device)
				batch_size = road.shape[0]
				road = road.view(batch_size*seq_len, 11)
				ego = data['ego'].to(args.device)
				actor = torch.FloatTensor(data['actor']).to(args.device)

				if self.road_topo:
					pred_ego, pred_actor, pred_road_para = model.train_forward(fronts+lefts+rights+tops)
				else:
					pred_ego, pred_actor = model.train_forward(fronts+lefts+rights+tops)


				ce = nn.CrossEntropyLoss(reduction='mean').cuda()
				pos_weight = torch.ones([num_actor_class])*args.weight
				bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight).cuda()
				road_bce = nn.BCEWithLogitsLoss(reduction='mean').cuda()

				ego_loss = ce(pred_ego, ego)
				actor_loss = bce(pred_actor, actor)
				if self.road_topo:
					road_loss = (road_bce(pred_road_para, road))
				else:
					road_loss = 0.
				loss = ego_loss + self.bce_weight * actor_loss + road_loss

				num_batches += 1

				_, pred_ego = torch.max(pred_ego.data, 1)
				pred_actor = torch.sigmoid(pred_actor)
				pred_actor = pred_actor > 0.5
				pred_actor = pred_actor.float()

				if self.road_topo:
					pred_road_para = torch.sigmoid(pred_road_para)
					pred_road_para = pred_road_para > 0.5
					pred_road_para = pred_road_para.float()


				label_actor_list.append(actor.detach().cpu().numpy())
				pred_actor_list.append(pred_actor.detach().cpu().numpy())
				if self.road_topo:
					label_road_para_list.append(road.detach().cpu().numpy())
					pred_road_para_list.append(pred_road_para.detach().cpu().numpy())


				# mean_f1 += f1_score(
				# 	actor.detach().cpu().numpy(), 
				# 	pred_actor.detach().cpu().numpy(),
				# 	average='macro',
				# 	zero_division=0)

				total_ego += ego.size(0)

				correct_ego += (pred_ego == ego).sum().item()

			pred_actor_list = np.squeeze(np.stack(pred_actor_list, axis=0), axis=1)
			label_actor_list = np.squeeze(np.stack(label_actor_list, axis=0), axis=1)
			# print(np.stack(pred_road_para_list, axis=0).shape)
			# print(np.stack(label_road_para_list, axis=0).shape)
			if self.road_topo:
				pred_road_para_list = np.reshape(np.stack(pred_road_para_list, axis=0), (batch_size*seq_len, -1))
				label_road_para_list = np.reshape(np.stack(label_road_para_list, axis=0), (batch_size*seq_len, -1))


			mean_f1 = f1_score(
					pred_actor_list.astype('int64'),
					label_actor_list.astype('int64'), 
					average='samples',
					zero_division=0)
			# map = average_precision_score(
			# 		pred_actor_list.astype('int64'),
			# 		label_actor_list.astype('int64'), 
			# 		average='samples'
			# 		)
			precision = precision_score(
				pred_actor_list,
				label_actor_list,
				average='micro',
				zero_division=0
				)
			recall = recall_score(
					pred_actor_list.astype('int64'),
					label_actor_list.astype('int64'), 
					average='samples',
					zero_division=0
					)
			###############################
			if self.road_topo:
				r_mean_f1 = f1_score(
						pred_road_para_list.astype('int64'),
						label_road_para_list.astype('int64'), 
						average='samples',
						zero_division=0)

				r_precision = precision_score(
					pred_road_para_list,
					label_road_para_list,
					average='micro',
					zero_division=0
					)
				r_recall = recall_score(
						pred_road_para_list.astype('int64'),
						label_road_para_list.astype('int64'), 
						average='samples',
						zero_division=0
						)

			acc = accuracy_score(label_actor_list, pred_actor_list)
			hamming = hamming_loss(label_actor_list, pred_actor_list)
			print('----------------------Ego--------------------------------')
			print(f'Accuracy of the ego: {100 * correct_ego // total_ego} %')
			print('----------------------actor--------------------------------')
			print(f'Accuracy of the actor: {acc}')
			print(f'hamming of the actor: {hamming}')


			print(f'precision of the actor: {precision}')
			print(f'recall of the actor: {recall}')
			print(f'f1 of the actor: {mean_f1}')

			mean_f1 = f1_score(
					pred_actor_list.astype('int64'),
					label_actor_list.astype('int64'), 
					average=None,
					zero_division=0)
			np.save(os.path.join(args.logdir, str(args.road)+'actor_f1.npy'), mean_f1)

			print('----------------------Road--------------------------------')

			if self.road_topo:
				acc = accuracy_score(label_road_para_list, pred_road_para_list)
				hamming = hamming_loss(label_road_para_list, pred_road_para_list)
				print(f'Accuracy of the road: {acc}')
				print(f'hamming of the road: {hamming}')
				print(f'precision of the road: {r_precision}')
				print(f'recall of the road: {r_recall}')
				print(f'f1 of the road: {r_mean_f1}')

				r_mean_f1 = f1_score(
						pred_road_para_list.astype('int64'),
						label_road_para_list.astype('int64'), 
						average=None,
						zero_division=0)
				np.save(os.path.join(args.logdir, str(args.road) + 'road_f1.npy'), r_mean_f1)
					
			loss = loss / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Loss: {loss:3.3f}')

			writer.add_scalar('val_loss', loss, self.cur_epoch)
			
			self.val_loss.append(loss.data)

	def vizualize(self, cam=False):
		model.eval()
		model.en_lstm.train()
		# model.train()
		num_batches = 0
		loss = 0.

		total_ego = 0
		total_actor = 0

		correct_ego = 0
		correct_actor = 0
		mean_f1 = 0
		label_actor_list = []
		pred_actor_list = []

		# Validation loop
		for batch_num, data in enumerate(tqdm(dataloader_val), 0):
			
			# create batch and move to GPU
			fronts_in = data['fronts']
			lefts_in = data['lefts']
			rights_in = data['rights']
			if self.top:
				tops_in = data['tops']

			fronts = []
			lefts = []
			rights = []
			tops = []

			for i in range(seq_len):
				if self.top:
					tops.append(tops_in[i].to(args.device, dtype=torch.float32))
				else:
					fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
					if not self.front_only:
						lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
						rights.append(rights_in[i].to(args.device, dtype=torch.float32))

			road = data['road'].to(args.device)
			ego = data['ego'].to(args.device)
			label_actor = torch.FloatTensor(data['actor']).to(args.device)

			if cam:
				# viz

				cam.batch_size = 1
				fronts = list(fronts[seq_len//2])
				input_tensor = torch.cat(fronts, dim=0)
				w, h = input_tensor.shape[1], input_tensor.shape[2]
				input_tensor = input_tensor.view(1, 3, w, h)

				grayscale_cam = cam(input_tensor=input_tensor)

				# In this example grayscale_cam has only one image in the batch:
				grayscale_cam = grayscale_cam[0, :]

				img = np.float32(np.array(data['img_front'][seq_len//2])).reshape(360, 640, 3)
				img = show_cam_on_image(img, grayscale_cam, use_rgb=True)
				img = Image.fromarray(img, mode="RGB")
				img.save(str(batch_num)+".jpeg")
	
				# model.eval()
			num_batches += 1

			total_ego += ego.size(0)



	def save(self):

		save_best = False
		if self.val_loss[-1] <= self.bestval:
			self.bestval = self.val_loss[-1]
			self.bestval_epoch = self.cur_epoch
			save_best = True
		
		# Create a dictionary of all data to save

		# log_table = {
		# 	'epoch': self.cur_epoch,
		# 	'iter': self.cur_iter,
		# 	'bestval': float(self.bestval.data),
		# 	'bestval_epoch': self.bestval_epoch,
		# 	'train_loss': self.train_loss,
		# 	'val_loss': self.val_loss,
		# }

		# Save ckpt for every epoch
		torch.save(model.state_dict(), os.path.join(args.logdir, 'model_%d.pth'%self.cur_epoch))

		# Save the recent model/optimizer states
		torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
		torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))

		# Log other data corresponding to the recent model
		# with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
		# 	f.write(json.dumps(log_table))

		tqdm.write('====== Saved recent model ======>')
		
		if save_best:
			torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
			tqdm.write('====== Overwrote best model ======>')

# Config
# config = GlobalConfig()
torch.cuda.empty_cache() 
seq_len = args.seq_len
is_top = args.top
front_only = args.front_only
# num_ego_class = 10
# num_actor_class = 84
num_ego_class = 4
num_actor_class = 36
if is_top or front_only:
	num_cam = 1
else:
	num_cam = 3


# Data

train_set = feature_data.Feature_Data(seq_len=seq_len, is_top=is_top, front_only=front_only, scale=args.scale, seg=args.seg, lss=args.lss, num_cam=num_cam)
val_set = feature_data.Feature_Data(seq_len=seq_len, training=False, is_top=is_top, front_only=front_only, scale=args.scale, viz=args.viz, seg=args.seg, lss=args.lss, num_cam=num_cam)
# print(val_set)
dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# Model
model = generate_model(args.id, num_cam, num_ego_class, num_actor_class, args.seq_len, args.road).cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
trainer = Engine(top=is_top, bce_weight=args.bce, road_topo=args.road, front_only=args.front_only)

if args.viz:
	# cam = GradCAM(model=model, 
	# 	           target_layers=[model.slow_res5[-1]],
	# 	           use_cuda=True)
	cam = GradCAM(model=model, 
		           target_layers=[model.conv1],
		           use_cuda=True)

	cam.batch_size = 1
else:
	cam = False


# Create logdir
if not os.path.isdir(args.logdir):
	os.makedirs(args.logdir)
	print('Created dir:', args.logdir)
elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
	print('Loading checkpoint from ' + args.logdir)
	with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
		log_table = json.load(f)

	# Load variables
	trainer.cur_epoch = log_table['epoch']
	if 'iter' in log_table: trainer.cur_iter = log_table['iter']
	trainer.bestval = log_table['bestval']
	trainer.train_loss = log_table['train_loss']
	trainer.val_loss = log_table['val_loss']

	# Load checkpoint
	model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
	optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'recent_optim.pth')))

# Log args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
	json.dump(args.__dict__, f, indent=2)
if not args.test:
	for epoch in range(trainer.cur_epoch, args.epochs): 
		trainer.train()
		if epoch % args.val_every == 0: 
				trainer.validate(None)
				trainer.save()
		if args.viz and epoch % 20 == 0:
				trainer.vizualize(cam)
else:
	trainer.validate(cam=cam)