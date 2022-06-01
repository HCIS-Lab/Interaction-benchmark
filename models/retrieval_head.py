import torch
import torch.nn as nn


class Head(nn.Module):
	def __init__(self, in_channel, num_ego_classes, num_actor_classes):
		super(Head, self).__init__()
		self.fc_ego = nn.Linear(in_channel, num_ego_classes)
		self.fc_actor = nn.Linear(in_channel, num_actor_classes)


	def forward(self, x):
		y_ego = self.fc_ego(x)
		y_actor = self.fc_actor(x)
		return y_ego, y_actor