from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.structures.image_list import ImageList
import torch
from torchvision import transforms
import cv2
from PIL import Image

def get_maskrcnn():
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
	path = cfg.MODEL.WEIGHTS
	model = build_model(cfg)
	DetectionCheckpointer(model).load(path)
	return model


# test

# loader = transforms.Compose([
#     transforms.ToTensor()])

# # im = cv2.imread("1.jpg")
# im = Image.open('1.jpg').convert('RGB')
# # im = Image.fromarray(im)
# img = loader(im).unsqueeze(0)
# im = img.to('cuda', torch.float)
# print(im.shape)
# im = ImageList.from_tensors([im.cuda()])
tensor = torch.ones([1, 3, 224, 224], dtype=torch.float32).cuda()
model = get_maskrcnn().cuda()
# print(model.backbone.__dict__)

model.eval()

# ResNet features
features = model.backbone.bottom_up(tensor)
for k, v in features.items():
	print(k)

print('---------------------------------')
# FPN features
features = model.backbone(tensor)
for k, v in features.items():
	print(k)

proposals, _ = model.proposal_generator(im, features)
instances, _ = model.roi_heads(im, features, proposals)
