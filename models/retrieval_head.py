import torch
import torch.nn as nn

class Head(nn.Module):
	def __init__(self, in_channel, num_ego_classes, num_actor_classes):
		super(Head, self).__init__()
		self.fc_ego = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(in_channel, 4)
                )


		self.fc_actor = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Linear(in_channel, 36)
                )

	def forward(self, x):
		y_ego = self.fc_ego(x)
		y_actor = self.fc_actor(x)
		return y_ego, y_actor

class Road_Head(nn.Module):
	def __init__(self, in_channel, num_road_para=11):
		super(Road_Head, self).__init__()
		self.fc_road = nn.Sequential(
	                nn.ReLU(inplace=False),
	                nn.Linear(in_channel, num_road_para)
	                )

	def forward(self, x):
		road_para = self.fc_road(x)
		return road_para