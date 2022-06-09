def get_features(root_path, object_per_frame, batch_n):

    """
    Args:
        object_per_frame : Maximum number of objects in a frame.
        batch_n : Batch size.
    
    Path description:
    root_path/
        {Scenario_type}/
            {Scenario_ID}/
                variant_scenario/
                    {variant_name}/
                        features/
                            frame_{ID}/
                                p5_features N x 256 x (H//32) x (W//32)
                                roi N x object_per_frame x 256 x 7 x 7
                        rgb/
                            front.zip
    """
    maskformer,model = get_models()
    maskformer = maskformer.cuda()
    maskformer.eval()
    model = model.cuda()
    model.eval()
    scenario_key = ['collision','interactive','non-interactive']
    curr_path = root_path
    path = [curr_path]
    count = 0
    for scenario_type in os.listdir(curr_path):
        if scenario_type in scenario_key:
            curr_path = path[-1]
            curr_path = os.path.join(curr_path,scenario_type)
            path.append(curr_path)
            for scenario_id in os.listdir(curr_path):
                if count == 8:
                    return
                count+=1
                print(scenario_id)
                curr_path = path[-1]
                curr_path = os.path.join(curr_path,scenario_id,'variant_scenario')
                path.append(curr_path)
                for variant_name in os.listdir(curr_path):
                    curr_path = path[-1]
                    curr_path = os.path.join(curr_path,variant_name)
                    if not os.path.isdir(os.path.join(curr_path,'features')):
                        os.mkdir(os.path.join(curr_path,'features'))
                    # Read rgb
                    img_archive = zipfile.ZipFile(os.path.join(curr_path,'rgb','front.zip'), 'r')
                    zip_file_list = img_archive.namelist()
                    img_file_list = sorted(zip_file_list)[1:] # the first element is a folder
                    # Read bbox
                    bbox_archive = zipfile.ZipFile(os.path.join(curr_path,'bbox','front.zip'), 'r')
                    zip_file_list = bbox_archive.namelist()
                    bbox_file_list = sorted(zip_file_list)[2:]
                    # Read collision history
                    json_file = open(os.path.join(curr_path,'collision_history.json'),'r')
                    history = json.loads(json_file.read())
                    json_file.close()
                    collision_frame = history[0]['frame']
                    # Align frame number
                    index = -1
                    while img_file_list[index][-7:-4]!=bbox_file_list[-1][-8:-5]:
                        index -= 1
                    if index!=-1:
                        img_file_list = img_file_list[:index+1]
                    # If video length < 100
                    if len(bbox_file_list)<100 or len(img_file_list)<100:
                        continue
                    print("\t",variant_name)
                    img_file_list = img_file_list[-100:]
                    bbox_file_list = bbox_file_list[-100:]
                    # Init data
                    features_tensor = None
                    roi_tensor = None
                    bbox_tensor = None
                    i = 0
                    while i < 100:
                        inputs = []
                        for _ in range(batch_n):
                            img_file = img_file_list[i]
                            bbox_file = bbox_file_list[i]
                            imgdata = img_archive.read(img_file)
                            imgdata = cv2.imdecode(np.frombuffer(imgdata, np.uint8), cv2.IMREAD_COLOR)
                            bbox = bbox_archive.read(bbox_file)
                            bbox = bbox.decode("UTF-8")
                            bbox = ast.literal_eval(bbox)
                            frame = cv2.resize(imgdata,(640,320),interpolation=cv2.INTER_NEAREST)
                            height, width = frame.shape[:2]
                            frame = torch.as_tensor(frame.astype("float32").transpose(2, 0, 1))
                            inputs.append({"image": frame, "height": height, "width": width})
                            i += 1
                        featrues_batch, roi_batch, bbox_batch = run_model([maskformer,model],inputs,batch_n)
                        if roi_tensor is None:
                            features_tensor = featrues_batch
                            roi_tensor = roi_batch
                            bbox_tensor = bbox_batch
                        else:
                            features_tensor = torch.cat((features_tensor,featrues_batch))
                            roi_tensor = torch.cat((roi_tensor,roi_batch))
                            bbox_tensor = torch.cat((bbox_tensor,bbox_batch))
                    torch.save(features_tensor,os.path.join(curr_path,'features','features.pt'))
                    torch.save(roi_tensor,os.path.join(curr_path,'features','roi.pt'))
                    torch.save(bbox_tensor,os.path.join(curr_path,'features','bbox.pt'))
                path.pop()
            path.pop()
            return

def run_model(models, inputs, batch_n):
    maskformer, model = models
    with torch.no_grad():
        images = model.preprocess_image(inputs)
        fpn_features = maskformer.get_fpn_features(inputs)
        features_maskformer = fpn_features[4]
        # roi align
        features = model.backbone(images.tensor)  # set of cnn features
        proposals, _ = model.proposal_generator(images, features, None)  # RPN
        features_ = [features[f] for f in model.roi_heads.in_features]
        # ROI ALIGN
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        # Flatten roi align features
        box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
        predictions = model.roi_heads.box_predictor(box_features)
        pred_instances, _ = model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)
        # output boxes, masks, scores, etc
        pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
        roi_input = []
        for i in range(batch_n):
            ins = pred_instances[i]["instances"]
            size = pred_instances[i]['instances'].scores.size(dim=0)
            if size>20:
                roi_input.append(ins.pred_boxes[:20])
            else:
                temp = Boxes(torch.zeros(20-size,4).cuda())
                pred_boxes = Boxes.cat([ins.pred_boxes,temp])
                roi_input.append(pred_boxes)
        roi = model.roi_heads.box_pooler(fpn_features[1:],roi_input).view(batch_n,20,-1)
        return features_maskformer.cpu(),roi.cpu(), Boxes.cat(roi_input).tensor.view(batch_n,20,-1).cpu()

# class CARLA_IMG_FEATURE():
if __name__ == '__main__':
    get_features('/run/user/1002/gvfs/smb-share:server=hcis_nas.local,share=carla/dataset',20,5)
    exit()
    maskformer,model = get_models()
    maskformer = maskformer.cuda()
    maskformer.eval()
    model = model.cuda()
    model.eval()
    img_per_batch = 5
    with torch.no_grad():
        # postive or negative
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
                    print("Batch number:",batch_count,"\n\tFile name:",file)
                    batch_labels = np.zeros((1),dtype=bool)
                    batch_file_name = np.zeros((1),dtype=str)
                    batch_data = np.zeros((1,100,256*20*40),dtype=np.float32)
                    batch_data_flat = np.zeros((1,100,20,256*7*7),dtype=np.float32)
                    batch_data_detectron = np.zeros((1,100,20,256*7*7),dtype=np.float32)
                    batch_scores = np.zeros((1,100,20),dtype=np.float32)
                    batch_bbox = np.zeros((1,100,20,4),dtype=np.float32)
                    batch_classes = np.ones((1,100,20),dtype=int)
                    batch_risky = np.zeros((1,100,20),dtype=int)
                    batch_classes = np.negative(batch_classes)
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
                        roi_index = 0
                        proposals_index = 0
                        for i in range(img_per_batch):
                            inds = pred_inds[i]
                            size = pred_instances[i]['instances'].scores.size(dim=0)
                            if size > 20:
                                batch_bbox[:,frame_i*img_per_batch+i] = pred_instances[i]['instances'].pred_boxes[:20].tensor.cpu().numpy()
                                batch_scores[:,frame_i*img_per_batch+i] = pred_instances[i]['instances'].scores[:20].cpu().numpy()
                                batch_classes[:,frame_i*img_per_batch+i] = pred_instances[i]['instances'].pred_classes[:20].cpu().numpy()
                                batch_data_detectron[:,frame_i*img_per_batch+i] = box_features[proposals_index:proposals_index+proposals_list[i]][inds][:20].cpu().numpy()
                                batch_data_flat[:,frame_i*img_per_batch+i] = roi[roi_index:roi_index+20].cpu().numpy()
                                roi_index += 20
                            else:
                                batch_bbox[:,frame_i*img_per_batch+i,:size] = pred_instances[i]['instances'].pred_boxes.tensor.cpu().numpy()
                                batch_scores[:,frame_i*img_per_batch+i,:size] = pred_instances[i]['instances'].scores.cpu().numpy()
                                batch_classes[:,frame_i*img_per_batch+i,:size] = pred_instances[i]['instances'].pred_classes.cpu().numpy()
                                batch_data_detectron[:,frame_i*img_per_batch+i,:size] = box_features[proposals_index:proposals_index+proposals_list[i]][inds].cpu().numpy()
                                batch_data_flat[:,frame_i*img_per_batch+i,:size] = roi[roi_index:roi_index+size].cpu().numpy()
                                roi_index += size
                            proposals_index += proposals_list[i]
                            # for j,bbox in enumerate(pred_instances[i]['instances'].pred_boxes):
                            #     if IOU(bbox,gt_bbox)>=0.6:
                            #         batch_risky[:,frame_i*img_per_batch+i,j]=1
                    batch_labels = label
                    batch_file_name = file
                    # training
                    if t_n==0:
                        # saved as dict
                        np.savez('/mnt/sdb/Dataset/SA/training/batch_%04d' % batch_count, file_name=batch_file_name, label=batch_labels,data=batch_data ,data_flat=batch_data_flat, data_flat_detectron = batch_data_detectron, bboxes=batch_bbox, classes=batch_classes, scores=batch_scores)
                    else:
                        np.savez('/mnt/sdb/Dataset/SA/testing/batch_%04d' % batch_count, file_name=batch_file_name, label=batch_labels,data=batch_data ,data_flat=batch_data_flat, data_flat_detectron = batch_data_detectron, bboxes=batch_bbox, classes=batch_classes, scores=batch_scores)
                    print("")
                    # if t_n==0:
                    # 	np.savez('/mnt/sdb/Dataset/SA/training/batch_baseline1_%04d' % batch_count, file_name=batch_file_name, label=batch_labels, data=batch_baseline1)
                    # else:
                    # 	np.savez('/mnt/sdb/Dataset/SA/testing/batch_baseline1_%04d' % batch_count, file_name=batch_file_name, label=batch_labels, data=batch_baseline1)
                    batch_count += 1
                    cap.release()
