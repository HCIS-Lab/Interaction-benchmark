from orn.backbone.imagenet_pretraining import load_pretrained_2D_weights
from orn.backbone.resnet.basicblock import BasicBlock2D, BasicBlock3D, BasicBlock2_1D
from orn.backbone.resnet.bottleneck import Bottleneck2D, Bottleneck3D, Bottleneck2_1D
from orn.backbone.resnet.resnet import ResNet
import ipdb

__all__ = [
    'resnet_two_heads',
]


def resnet_two_heads(fe):
    """Constructs a ResNet-18 model with 2 heads
    """
    # Params
    depth, blocks, pooling, object_head = 50, \
                                          '3D_3D_3D_3D', \
                                          'rnn', \
                                          '2D'
    list_block, list_layers = get_cnn_features(depth=depth,
                                               str_blocks=blocks)
    blocks_object_head, _ = get_cnn_features(depth=depth,
                                             str_blocks=object_head)

    model = ORN_ResNet(
                list_layers,
                two_heads=True,
                pooling=pooling)
    return model


def get_cnn_features(str_blocks='2D_2D_2D_2D', depth=18):
    # List of blocks
    list_block = []

    # layers
    if depth == 18:
        list_layers = [2, 2, 2, 2]
        nature_of_block = 'basic'
    elif depth == 34:
        list_layers = [3, 4, 6, 3]
        nature_of_block = 'basic'
    elif depth == 50:
        list_layers = [3, 4, 6, 3]
        nature_of_block = 'bottleneck'
    else:
        raise NameError

    # blocks
    if nature_of_block == 'basic':
        block_2D, block_3D, block_2_1D = BasicBlock2D, BasicBlock3D, BasicBlock2_1D
    elif nature_of_block == 'bottleneck':
        block_2D, block_3D, block_2_1D = Bottleneck2D, Bottleneck3D, Bottleneck2_1D
    else:
        raise NameError

    # From string to blocks
    list_block_id = str_blocks.split('_')

    # Catch from the options if exists
    for i, str_block in enumerate(list_block_id):
        # Block kind
        if str_block == '2D':
            list_block.append(block_2D)
        elif str_block == '2.5D':
            list_block.append(block_2_1D)
        elif str_block == '3D':
            list_block.append(block_3D)
        else:
            ipdb.set_trace()
            raise NameError

    return list_block, list_layers
