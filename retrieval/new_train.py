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
import retrieval_data
import cnnlstm_image
# import cnnlstm_seg

# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image, \
#                                          deprocess_image, \
#                                          preprocess_image

# import cnnlstm_backbone

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='cnnlstmimagenet', help='Unique experiment identifier.')

parser.add_argument('--front_only', type=bool, default=True, help='')
parser.add_argument('--top', type=bool, default=False, help='')
parser.add_argument('--seg', type=bool, default=False, help='')


parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--test', type=bool, default=False, help='eval only')
parser.add_argument('--viz', type=bool, default=False, help='grad cam')


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

	def __init__(self,  cur_epoch=0, cur_iter=0, side=False, top=False, front_only=True):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.top = top
		self.front_only = front_only
		self.train_loss = []
		self.val_loss = []
		self.bestval = 1e10

	def train(self):
		loss_epoch = 0.
		num_batches = 0
		model.train()
		# Train loop
		for data in tqdm(dataloader_train):
			
			# efficiently zero gradients
			# for p in model.parameters():
			# 	p.grad = None
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
			road = data['road'].to(args.device)
			ego = data['ego'].to(args.device)
			actor = torch.FloatTensor(data['actor']).to(args.device)

			pred_ego, pred_actor = model(fronts+lefts+rights+tops)


			pred_actor = F.sigmoid(pred_actor)
			ce = nn.CrossEntropyLoss().cuda()
			bce = nn.BCELoss(weight=None, size_average=True).cuda()
			loss = (ce(pred_ego, ego) + bce(pred_actor, actor))/2
			loss.backward()
			loss_epoch += float(loss.item())

			num_batches += 1
			optimizer.step()
			optimizer.zero_grad()
			writer.add_scalar('train_loss', loss.item(), self.cur_iter)
			self.cur_iter += 1
		
		
		loss_epoch = loss_epoch / num_batches
		print(loss_epoch)
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
			if cam:
				grayscale_cam = cam(input_tensor=input_tensor,
	                        target_category=None,
	                        aug_smooth=True,
	                        eigen_smooth=True)

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
				actor = torch.FloatTensor(data['actor']).to(args.device)

				pred_ego, pred_actor = model(fronts+lefts+rights+tops)

				pred_actor = F.sigmoid(pred_actor)
				ce = nn.CrossEntropyLoss().cuda()
				bce = nn.BCELoss(weight=None, size_average=True).cuda()
				loss = (ce(pred_ego, ego) + bce(pred_actor, actor))/2

				num_batches += 1

				_, pred_ego = torch.max(pred_ego.data, 1)
				pred_actor = pred_actor > 0.5

				total_ego += ego.size(0)
				total_actor += actor.size(0) * actor.size(1)
				correct_ego += (pred_ego == ego).sum().item()
				correct_actor += (pred_actor == actor).sum().item()
			print(f'Accuracy of the ego: {100 * correct_ego // total_ego} %')
			print(f'Accuracy of the actor: {100 * correct_actor // total_actor} %')


					
			loss = loss / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Loss: {loss:3.3f}')

			writer.add_scalar('val_loss', loss, self.cur_epoch)
			
			self.val_loss.append(loss.data)

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
seq_len = 8
step = 5
is_top = args.top
front_only = args.front_only
num_ego_class = 10
num_actor_class = 80

if is_top or front_only:
	num_cam = 1
else:
	num_cam = 3


# Data
train_set = retrieval_data.Retrieval_Data(seq_len=seq_len, step=step, is_top=is_top, front_only=front_only)
val_set = retrieval_data.Retrieval_Data(seq_len=seq_len, step=step, training=False, is_top=is_top, front_only=front_only)
# print(val_set)
dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
# dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# Model

model = cnnlstm_image.CNNLSTM(num_cam, num_ego_class, num_actor_class).cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
trainer = Engine(top=is_top)

for param in model.backbone.parameters():
    param.requires_grad = False
if args.seg:
	for param in model.seg.parameters():
	    param.requires_grad = False

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('Total trainable parameters: ', params)


if args.viz:
	cam = GradCAM(model=model, 
		           target_layer=model.conv1,
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
		# if epoch % args.val_every == 0: 
		# 	trainer.validate(cam)
		# 	trainer.save()
else:
	trainer.validate(cam=cam)