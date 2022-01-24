from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

import torch

def get_maskrcnn_backbone():
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
	model = build_model(cfg)
	return model.backbone

# test
tensor = torch.ones([1, 3, 224, 224], dtype=torch.float32).cuda()
model = get_maskrcnn_backbone().cuda()
model.eval()
features = model(tensor)
# proposals, _ = model.proposal_generator(images, features)
# instances, _ = model.roi_heads(images, features, proposals)