from torch import nn
import torch
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

class Cilrs(nn.Module):
    """ Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self):
        super().__init__()
        # Simple CNN
        self.conv1 = Conv2d(in_channels=8, out_channels=32, kernel_size=(5, 5)) # Output width: 204
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # Output width: 102
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5)) # # Output width: 106
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # Output width: 53
        self.fc1 = Linear(in_features=141376, out_features=512)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=512, out_features=255)

        # controller
        self.num_branch = 6
        fc_branch_list = []
        for i in range(self.num_branch):
            fc_branch_list.append(nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 3)
        ))

        self.branches = nn.ModuleList(fc_branch_list)

    def forward(self, segs, cmds):
        # Image encoding
        x = self.conv1(segs)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        x = torch.cat((x, cmds), 1)

        control_pred = 0.
        for i, branch in enumerate(self.branches):
            # Choose control for branch of only active command
            control_pred += branch(x) * (i == (cmds).expand(segs.size(0), 3))

        return control_pred