import torch.nn as nn
import torch

import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from MaskFormer.demo.demo import get_maskformer
from retrieval_head import Head, Road_Head

# https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        # if hidden_state is not None:
        #     raise NotImplementedError()
        # else:
        if hidden_state == None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param





class ConvLstm(nn.Module):
    def __init__(self, num_cam, num_ego_class, num_actor_class, road):
        super(ConvLstm, self).__init__()
        self.num_cam = num_cam
        self.road = road
        self.backbone = get_maskformer()

        self.conv1 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(2048*self.num_cam, 1024*self.num_cam, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(1024*self.num_cam),
                nn.ReLU(inplace=False),
                nn.Conv2d(1024*self.num_cam, 1024, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=False),
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(512)
                    )

        self.conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 400, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(400),
                nn.ReLU(inplace=True),
                nn.Conv2d(400, 256, kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
                    )

        self.en_lstm = ConvLSTM(256, 128, (3,3), 1, True, True)
        # self.en_lstm = ConvLSTM(256, 256, 3, 1, True, True, True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.head = Head(128, num_ego_class, num_actor_class)

        if self.road:
            self.road_enc = nn.Sequential(
                    nn.ReLU(inplace=False),
                    nn.Conv2d(512, 400, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(400),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(400, 256, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(256)
                        )

            self.road_fc = Road_Head(256)
            self.fusion = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding='same'),
                nn.BatchNorm2d(256),
                )


    def train_forward(self, x):

        hidden, hidden_road = None, None
        seq_len = len(x)//self.num_cam

        batch_size = x[0].shape[0]
        h, w = x[0].shape[2], x[0].shape[3]

        out_road = []
        for i in range(seq_len):
            x_i = x[i*self.num_cam : i*self.num_cam + self.num_cam]
            if isinstance(x_i, list):
                x_i = torch.stack(x_i, dim=0)
                x_i = torch.permute(x_i, (1,0,2,3,4))
                x_i = torch.reshape(x_i, (batch_size*self.num_cam, 3, h, w))
                # x_i = x_i.view(batch_size, self.num_cam, 3, h, w)
            with torch.no_grad():
                x_i = (x_i - self.backbone.pixel_mean) / self.backbone.pixel_std
                x_i = self.backbone.backbone(x_i)['res5']
                down_h, down_w = x_i.shape[-2], x_i.shape[-1]
                # x_i = torch.permute(x_i, (1,0,2,3,4))
                x_i = torch.reshape(x_i, (batch_size, -1, down_h, down_w))
            x_i = self.conv1(x_i)
            x_feature_i = self.conv2(x_i)
            

            if self.road:
                x_road_i = self.road_enc(x_i)
                x_feature_i = torch.cat((x_feature_i, x_road_i), dim=1)
                x_feature_i = self.fusion(x_feature_i)


            x_feature_i = torch.unsqueeze(x_feature_i, 1)

            if self.road:
                x_road_i = self.pool(x_road_i)
                x_road_i = x_road_i.view(batch_size, 256)
                out_road.append(x_road_i)

            _, hidden = self.en_lstm(x_feature_i, hidden)
        hidden = hidden[0][0]
        hidden = self.pool(hidden)
        hidden = hidden.view(batch_size, -1)
        ego, actor = self.head(hidden)
        if self.road:
            out_road = torch.stack(out_road, 0)
            out_road = out_road.view(batch_size*seq_len, 256)
            out_road = self.road_fc(out_road)
            return ego, actor, out_road
        else:
            return ego, actor


    def forward(self, fronts, lefts, rights, tops=False):
        hidden = None
        if not tops:
            batch_size = fronts[0].shape[0]
            seq_len = len(fronts)
            w, h = fronts[0].shape[2], fronts[0].shape[3]
        else:
            batch_size = tops[0].shape[0]
            seq_len = len(tops)
            w, h = tops[0].shape[2], tops[0].shape[3]

        for t in range(seq_len):
            x = []
            if not tops:
                x.append(fronts[t])
                x.append(lefts[t])
                x.append(rights[t])
            else:
                x.append(tops[t])

            x = torch.stack(x, dim=0).view(batch_size*self.num_cam, 3, w, h)
            x = self.backbone(x)['res5']
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)

            out, hidden = self.en_lstm(x.view(batch_size, 1, 256), hidden)

        ego, actor = self.head(out[:, -1, :])
        return ego, actor

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

        
