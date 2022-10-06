import numpy as np


CONVERTER = np.uint8([
    0,    # unlabeled
    1,    # building
    2,    # fence
    0,    # other
    3,    # ped
    4,    # pole
    5,    # road line
    6,    # road
    7,    # sidewalk
    0,    # vegetation
    8,    # car
    9,    # wall
    0,    # traffic sign
    0,    # sky
    0,    # ground
    0,    # bridge
    0,    # railtrack
    0,    # guardrail
    0,    # trafficlight
    0,    # static
    0,    # dynamic
    0,    # water
    0,    # terrain
    0, 
    0, 
    0
])


COLOR = np.uint8([
        (0, 0, 0),
        (66, 62, 64),
        (116, 191, 101),
        ( 255, 255, 255),
        (136, 138, 133),
        (0, 0, 142),
        (220, 20, 60),
        (0,0,1)
        ])

# # 1
# segmentation_type = ['car', 'cross_walk', 
#                         'ped', 'road', 'road_line', 'side_walk']
# # 2
# semantic_to_index = {
#                     'non-drivable_area': 0, # 0
#                     'road': 1,     # 1
#                     'cross_walk': 2,        # 2
#                     'road_line': 3,  # 3
#                     'side_walk': 4,         # 4
#                     'car': 5,        # 5
#                     'ped': 6                # 6
#                     }


COLOR2 = np.uint8([
        (0, 0, 142),
        (116, 191, 101),
        (220, 20, 60),
        (66, 62, 64),
        (255, 255, 255),
        (136, 138, 133 ),
        (0,0,0 ),
        (0,0,1)
        ])