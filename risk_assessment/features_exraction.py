from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.structures.image_list import ImageList
from detectron2.structures import Boxes
import torch
from torchvision import transforms
import cv2
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
import os
import numpy as np

VIDEO_PATH = ['../../Anticipating-Accidents/dataset/videos/training','../../Anticipating-Accidents/dataset/videos/testing']

def get_maskrcnn():
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
	path = cfg.MODEL.WEIGHTS
	model = build_model(cfg)
	DetectionCheckpointer(model).load(path)
	class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
	return model,class_names

def main():
	model,class_names = get_maskrcnn()
	model = model.cuda()
	model.eval()
	img_per_batch = 10
	union_object_num = 100
	torch.cuda.empty_cache()
	# # List[Dict[str, torch.Tensor]]
	# inputs = [{"image": image, "height": height, "width": width}]

	# file_name, data, pair_data
	with torch.no_grad():
		for t_n,path in enumerate(VIDEO_PATH):
			batch_count = 1
			for n_p in os.listdir(path):
				print(n_p)
				label = None
				if n_p == 'positive':
					label = True
				else:
					label = False
				now_path = os.path.join(path,n_p)
				for file in os.listdir(now_path):
					print("Batch number:",batch_count,"\nFile name:",file)

					batch_labels = np.zeros((1),dtype=np.bool)
					batch_file_name = np.zeros((1),dtype=str)
					batch_data_flat = np.zeros((1,100,20,1024),dtype=np.float32)
					batch_udata = np.zeros((1,100,union_object_num,1024),dtype=np.float32)
					batch_bbox = np.zeros((1,100,20,4),dtype=np.float32)
					batch_classes = np.ones((1,100,20),dtype=int)
					batch_classes = np.negative(batch_classes)

					# batch_baseline1 = np.zeros((1,100,256,12,20),dtype=np.float32)

					cap = cv2.VideoCapture(now_path+'/'+file) 
					for frame_i in range(100//img_per_batch):
						inputs = []
						for _ in range(img_per_batch):
							_, frame = cap.read()
							######
							# frame = cv2.resize(frame,(160,80),interpolation=cv2.INTER_AREA)
							######
							height, width = frame.shape[:2]
							frame = torch.as_tensor(frame.astype("float32").transpose(2, 0, 1))
							inputs.append({"image": frame, "height": height, "width": width})
						images = model.preprocess_image(inputs)
						features = model.backbone(images.tensor)  # set of cnn features
						proposals, _ = model.proposal_generator(images, features, None)  # RPN
						features_ = [features[f] for f in model.roi_heads.in_features]
						# batch_baseline1[:,frame_i*10:(frame_i+1)*10] = features_[1].cpu().numpy()
						# continue
						# ROI ALIGN
						box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
						# Flatten roi align features
						box_features_flat = model.roi_heads.box_head(box_features)  # features of all 1k candidates

						predictions = model.roi_heads.box_predictor(box_features_flat)
						pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
						pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)
						# output boxes, masks, scores, etc
						pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size

						u_bboxes = []
						for instance in pred_instances:
							temp_ubox = []
							temp_count = 0
							for i,c in enumerate(instance['instances'].pred_boxes):
								for d in instance['instances'].pred_boxes[i+1:]:
									temp_count+=1
									a = c.cpu().numpy()
									b = d.cpu().numpy()
									u_box = [min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
									if temp_count>union_object_num:
										break
									temp_ubox.append(u_box)
								if temp_count>union_object_num:
									break
							if temp_count<union_object_num:
								temp_ubox += [[0,0,0,0]]*(union_object_num-temp_count)
							u_bboxes.append(Boxes(torch.tensor(temp_ubox)).to('cuda'))
						box_features_union = model.roi_heads.box_pooler(features_,u_bboxes)
						box_features_union = model.roi_heads.box_head(box_features_union)
						box_features_union = box_features_union.view(10,union_object_num,1024)
						# features of the proposed boxes
						# feats = box_features[pred_inds]
						for i in range(img_per_batch):
							size = pred_inds[i].size(dim=0)
							current_features_flat = box_features_flat[i*1000:(i+1)*1000]
							current_features_flat = current_features_flat[pred_inds[i]]
							if size > 20:
								batch_data_flat[:,frame_i*10+i] = current_features_flat[:20].cpu().numpy()
								batch_bbox[:,frame_i*10+i] = pred_instances[i]['instances'].pred_boxes.tensor[:20].cpu().numpy()
								batch_classes[:,frame_i*10+i] = pred_instances[i]['instances'].pred_classes[:20].cpu().numpy()
							else:
								batch_data_flat[:,frame_i*10+i,:size] = current_features_flat.cpu().numpy()
								batch_bbox[:,frame_i*10+i,:size] = pred_instances[i]['instances'].pred_boxes.tensor.cpu().numpy()
								batch_classes[:,frame_i*10+i,:size] = pred_instances[i]['instances'].pred_classes.cpu().numpy()

						batch_udata[:,frame_i*10:(frame_i+1)*10] = box_features_union.cpu().numpy()
					batch_labels = label
					batch_file_name = file

					# training
					if t_n==0:
						np.savez('/mnt/sdb/Dataset/SA/training/batch_%04d' % batch_count, file_name=batch_file_name, label=batch_labels, data_flat=batch_data_flat, data_union=batch_udata, bboxes=batch_bbox, classes=batch_classes)
					else:
						np.savez('/mnt/sdb/Dataset/SA/testing/batch_%04d' % batch_count, file_name=batch_file_name, label=batch_labels, data_flat=batch_data_flat, data_union=batch_udata, bboxes=batch_bbox, classes=batch_classes)
					
					# if t_n==0:
					# 	np.savez('/mnt/sdb/Dataset/SA/training/batch_baseline1_%04d' % batch_count, file_name=batch_file_name, label=batch_labels, data=batch_baseline1)
					# else:
					# 	np.savez('/mnt/sdb/Dataset/SA/testing/batch_baseline1_%04d' % batch_count, file_name=batch_file_name, label=batch_labels, data=batch_baseline1)
					batch_count += 1

	
		# draw_img = cv2.imread("test.jpg")
		# draw_bbox(draw_img, pred_instances[0]['instances'],class_names)
	return

def main2():
	cfg = get_cfg()
	# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
	class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
	predictor = DefaultPredictor(cfg)
	im = cv2.imread("2.jpg")
	outputs = predictor(im)
	# print(class_names)
	draw_bbox(im,outputs["instances"].pred_boxes,outputs["instances"].pred_classes,class_names)
	return

def draw_bbox(draw_img,instances,class_names):
	font = cv2.FONT_HERSHEY_SIMPLEX
	for i,bbox in enumerate(instances.pred_boxes):
		cv2.rectangle(draw_img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,0,0),1)
		cv2.putText(draw_img,class_names[instances.pred_classes[i]],(int(bbox[0]),int(bbox[1])), font, 1.0,(0,0,255),1,cv2.LINE_AA)
	cv2.imshow('My Image', draw_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()