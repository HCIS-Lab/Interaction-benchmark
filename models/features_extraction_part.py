                    cap = cv2.VideoCapture(now_path+'/'+file) 
                    for frame_i in range(100//img_per_batch):
                        print("\t\tFrame num:",frame_i*img_per_batch,end='\r')
                        inputs = []
                        for _ in range(img_per_batch):
                            _, frame = cap.read()
                            ######
                            frame = cv2.resize(frame,(640,320),interpolation=cv2.INTER_NEAREST)
                            ######
                            height, width = frame.shape[:2]
                            frame = torch.as_tensor(frame.astype("float32").transpose(2, 0, 1))
                            inputs.append({"image": frame, "height": height, "width": width})
                        images = model.preprocess_image(inputs)
                        fpn_features = maskformer.get_fpn_features(inputs)
                        features_maskformer = fpn_features[3]
                        # roi align
                        features = model.backbone(images.tensor)  # set of cnn features
                        proposals, _ = model.proposal_generator(images, features, None)  # RPN
                        features_ = [features[f] for f in model.roi_heads.in_features]
                        # ROI ALIGN
                        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
                        proposals_list = []
                        for x in proposals:
                            proposals_list.append(x.proposal_boxes.tensor.size(dim=0))
                        # Flatten roi align features
                        box_features_flat = model.roi_heads.box_head(box_features)  # features of all 1k candidates
                        predictions = model.roi_heads.box_predictor(box_features_flat)
                        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
                        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)
                        # output boxes, masks, scores, etc
                        pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
                        roi_input = []
                        for i in range(img_per_batch):
                            ins = pred_instances[i]["instances"]
                            size = pred_instances[i]['instances'].scores.size(dim=0)
                            if size>20:
                                roi_input.append(ins.pred_boxes[:20])
                            else:
                                roi_input.append(ins.pred_boxes)
                        roi = model.roi_heads.box_pooler(fpn_features[1:],roi_input)
                        roi = roi.view(-1,256*7*7).contiguous()
                        # batch_data_flat[:,frame_i*img_per_batch:(frame_i+1)*img_per_batch] = roi.cpu().numpy()
                        features_maskformer = features_maskformer.unsqueeze(0).view(1,img_per_batch,-1)
                        batch_data[:,frame_i*img_per_batch:(frame_i+1)*img_per_batch] = features_maskformer.cpu().numpy()
                        box_features = box_features.view(-1,256*7*7)
