import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy

from utils import load_value_file


############################################


def frame_crop(files, n_frames):
    n_files = len(files)
    step = int(max(1, math.floor(n_files+1)/n_frames))
    new_video = []
    for j in range(0, n_files, step):
        new_video.append(files[j])
        if len(new_video) == n_frames:
            break
        
    # for j in range(1, n_frames, step):
    #   sample_j = copy.deepcopy(sample)
    #   sample_j['frame_indices'] = list(
    #       range(j, min(n_frames + 1, j + sample_duration)))
    #   dataset.append(sample_j)
    return new_video

def open_img(root_path, files):
    dataset = []

    for i, file in enumerate(files):
        img = Image.open(os.path.join(root_path, file)).convert('RGB')
        # img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = FixedResize(img)
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        # img = torch.from_numpy(img).float()
        dataset.append(img)
    return dataset

def FixedResize(img, size=224):
    """change the short edge length to size"""
    # print(img.size)
    w, h = img.size
    # if w > h:
    #     oh = size
    #     ow = int(1.0 * w * oh / h)
    # else:
    #     ow = size
    #     oh = int(1.0 * h * ow / w)
    img = img.resize((size,size), Image.BILINEAR)
    return img

def get_gt(gt):
    gt = np.array(gt).astype(np.float32)
    gt = torch.from_numpy(gt).float()
    return gt

def carla_dataset(root_path, n_frames=8, train_split=0.8):
    video_names = np.load('scenario_id.npy')
    annotations = np.load('gt.npy')
    # num_scenario = video_names.shape[0]
    dataset_front = []
    dataset_right = []
    dataset_left = []
    dataset_gt = []  
    video_names = tqdm(video_names)
    for i, name in enumerate(video_names):
        scenario_name = os.path.join(root_path, str(name))

        front_files = next(walk(os.path.join(scenario_name, 'front')), (None, None, []))[2]
        right_files = next(walk(os.path.join(scenario_name, 'right')), (None, None, []))[2]
        left_files = next(walk(os.path.join(scenario_name, 'left')), (None, None, []))[2]
        if len(front_files) < n_frames or len(right_files) < n_frames or len(left_files) < n_frames:
            print(name)
            continue
        # front_files = [f for f in os.path.listdir(os.path.join(scenario_name, 'front')) if isfile(os.path.join(scenario_name, 'front', f))]
        # right_files = [f for f in os.path.listdir(os.path.join(scenario_name, 'right')) if isfile(os.path.join(scenario_name, 'right', f))]
        # left_files = [f for f in os.path.listdir(os.path.join(scenario_name, 'left')) if isfile(os.path.join(scenario_name, 'left', f))]
        front_files = frame_crop(front_files, n_frames)
        right_files = frame_crop(right_files, n_frames)
        left_files = frame_crop(left_files, n_frames)
        front_files.sort()
        right_files.sort()
        left_files.sort()
        
        pool1 = ThreadPool(processes=10)
        pool2 = ThreadPool(processes=10)
        pool3 = ThreadPool(processes=10)
        p1_data = pool1.apply_async(open_img, (os.path.join(scenario_name, 'front'), front_files))
        p2_data = pool1.apply_async(open_img, (os.path.join(scenario_name, 'right'), right_files))
        p3_data = pool1.apply_async(open_img, (os.path.join(scenario_name, 'left'), left_files))


        dataset_front.append(p1_data.get())
        dataset_right.append(p2_data.get())
        dataset_left.append(p3_data.get())
        dataset_gt.append(annotations[i])

    dataset_front = np.stack(dataset_front, 0)
    dataset_right = np.stack(dataset_right, 0)
    dataset_left = np.stack(dataset_left, 0)
    dataset_gt = np.stack(dataset_gt, 0)
    dataset_front = torch.from_numpy(dataset_front).float()
    dataset_right = torch.from_numpy(dataset_right).float()
    dataset_left = torch.from_numpy(dataset_left).float()
    dataset_gt = torch.from_numpy(dataset_gt).long()
    num_train_samples = int(dataset_front.shape[0]*train_split)

    train_data = [dataset_front[:num_train_samples], dataset_right[:num_train_samples],
                    dataset_left[:num_train_samples], dataset_gt[:num_train_samples]]
    eval_data = [dataset_front[num_train_samples:], dataset_right[num_train_samples:],
                    dataset_left[num_train_samples:], dataset_gt[num_train_samples:]]
    return train_data, eval_data

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[3]), reverse=True)
    front, right, left, gt = zip(*data)

    pad_label = []
    # lens = []
    max_len = len(gt[0])
    for i in range(len(label)):
        temp_label = label[i]
        temp_label += [37] * (max_len - len(label[i]))
        pad_label.append(temp_label)
        # lens.append(len(label[i]))
    return front, right, left, pad_label 





############################################


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    f = open('gt.txt', 'r')
    lines = f.readlines()
    f.close
    return lines
    # with open(data_file_path, 'r') as data_file:
    #     return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data.items():
        # this_subset = value['subset']

        # if this_subset == subset:
            video_names.append(key)
            annotations.append(value)

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class UCF101(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=20,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return clip, target

    def __len__(self):
        return len(self.data)

