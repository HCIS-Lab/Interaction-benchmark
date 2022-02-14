from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.structures.image_list import ImageList
import torch
from torchvision import transforms
import cv2
from PIL import Image

def get_maskrcnn_feature_extractor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    path = cfg.MODEL.WEIGHTS
    model = build_model(cfg)
    DetectionCheckpointer(model).load(path)
    return model.backbone.bottom_up


# test

# loader = transforms.Compose([
#     transforms.ToTensor()])

# # # im = cv2.imread("1.jpg")
# original_image = cv2.imread("./input.jpg")

# with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258

#     original_image = original_image[:, :, ::-1]
#     # height, width = original_image.shape[:2]
#     # image = self.aug.get_transform(original_image).apply_image(original_image)
#     image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    # inputs = {"image": image, "height": height, "width": width}
    # predictions = self.model([inputs])[0]
    # return predictions


# im = Image.open('1.jpg').convert('RGB')
# # im = Image.fromarray(im)
# img = loader(im).unsqueeze(0)
# im = img.to('cuda', torch.float)
# print(im.shape)
# im = ImageList.from_tensors([im.cuda()])
# tensor = torch.ones([1, 3, 224, 224], dtype=torch.float32).cuda()
# model = get_maskrcnn_feature_extractor().cuda()
# model_dict = model.state_dict()
# print(model_dict.keys())

# model.eval()

# # ResNet features
# features = model(tensor)
# print(features['res5'].shape)
