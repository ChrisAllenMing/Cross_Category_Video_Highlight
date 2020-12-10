import os,sys
sys.path.append('./')
from mypath import Path

import torch
import cv2
import pdb
import math
import json
import shutil
import random
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ActivityNet(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]->[segments]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ActivityNet'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            category (str): Determines which kind of video to use. Defaults to 'sport'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Defaults is False.
    """

    def __init__(self, dataset='ActivityNet', split='train', category = 'sport', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        self.clip_len = clip_len
        self.split = split
        self.category = category
        folder = os.path.join(self.output_dir, split, category)

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        # Define all the categories in the dataset
        if dataset == 'ActivityNet':
            self.category_list = ['eat_drink', 'personal_care', 'household', 'sport', 'social']
            if not self.category in self.category_list:
                raise RuntimeError('The {} is not in the category list of {} dataset.'.format(self.category, dataset))

        if not self.check_integrity():
            raise RuntimeError('The processed dataset has not been found.' +
                               ' You need to process it first.')

        # Obtain all the filename of segments and the corresponding label
        # Going through each class folder one at a time
        self.fnames = list()
        self.video_names = list()
        self.segment2index = dict()
        self.labels = list()
        for video_id, video_name in enumerate(sorted(os.listdir(folder))):
            try:
                label_file = open(os.path.join(folder, video_name, 'label.json'), 'r')
            except:
                continue
            video_label = json.load(label_file)['label']
            tmp_video_listing = os.listdir(os.path.join(folder, video_name))
            tmp_video_listing.remove('label.json')

            for segment_id, segment_name in enumerate(sorted(tmp_video_listing)):
                segment_path = os.path.join(folder, video_name, segment_name)
                self.fnames.append(segment_path)
                self.video_names.append(os.path.join(folder, video_name))
                self.segment2index[segment_path] = video_id
                self.labels.append(video_label[segment_id])

        assert len(self.fnames) == len(self.labels)
        print('Number of {} segments: {:d}'.format(split, len(self.fnames)))

        self.labels = np.array(self.labels, dtype=int)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        if self.split == 'train':
            # Load the highlighted and non-highlighted video segments
            pos_buffer, neg_buffer, pos_label, neg_label = self.load_segment_pair(index)

            pos_buffer = self.crop(pos_buffer, self.clip_len, self.crop_size)
            pos_buffer = self.normalize(pos_buffer)
            pos_buffer = self.to_tensor(pos_buffer)

            neg_buffer = self.crop(neg_buffer, self.clip_len, self.crop_size)
            neg_buffer = self.normalize(neg_buffer)
            neg_buffer = self.to_tensor(neg_buffer)

            pos_label = np.array(pos_label)
            neg_label = np.array(neg_label)

            return torch.from_numpy(pos_buffer), torch.from_numpy(neg_buffer), torch.from_numpy(pos_label), \
                   torch.from_numpy(neg_label)
        else:
            # Load the segments for evaluation
            buffer = self.load_frames(self.fnames[index])
            buffer = self.crop(buffer, self.clip_len, self.crop_size)
            buffer = self.normalize(buffer)
            buffer = self.to_tensor(buffer)

            label = np.array(self.labels[index])
            video_id = np.array(self.segment2index[self.fnames[index]])

            return torch.from_numpy(buffer), torch.from_numpy(label), torch.from_numpy(video_id)

    def check_integrity(self):
        if not os.path.exists(self.output_dir):
            return False
        else:
            return True

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_segment_pair(self, index):
        fname = self.fnames[index]
        video_name = self.video_names[index]
        label = self.labels[index]

        if label == 1:
            valid_indices = np.where(np.logical_and(np.array(self.video_names) == video_name, self.labels != 1))[0]
            valid_indices = valid_indices.tolist()
            if len(valid_indices) == 0:
                valid_indices = np.where(np.array(self.video_names) == video_name)[0]
                valid_indices = valid_indices.tolist()
                # valid_indices.remove(index)
            index_ = random.sample(valid_indices, 1)[0]

            fname_ = self.fnames[index_]
            neg_label = self.labels[index_]
            pos_label = label

            pos_frames = sorted([os.path.join(fname, img) for img in os.listdir(fname)])
            pos_buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
            frame_idx = 0
            for i, frame_name in enumerate(pos_frames):
                if i == self.clip_len:
                    break
                frame = np.array(cv2.imread(frame_name)).astype(np.float64)
                pos_buffer[frame_idx] = frame
                frame_idx += 1

            neg_frames = sorted([os.path.join(fname_, img) for img in os.listdir(fname_)])
            neg_buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
            frame_idx = 0
            for i, frame_name in enumerate(neg_frames):
                if i == self.clip_len:
                    break
                frame = np.array(cv2.imread(frame_name)).astype(np.float64)
                neg_buffer[frame_idx] = frame
                frame_idx += 1
        else:
            valid_indices = np.where(np.logical_and(np.array(self.video_names) == video_name, self.labels == 1))[0]
            valid_indices = valid_indices.tolist()
            if len(valid_indices) == 0:
                valid_indices = np.where(np.array(self.video_names) == video_name)[0]
                valid_indices = valid_indices.tolist()
                # valid_indices.remove(index)
            index_ = random.sample(valid_indices, 1)[0]

            fname_ = self.fnames[index_]
            pos_label = self.labels[index_]
            neg_label = label

            pos_frames = sorted([os.path.join(fname_, img) for img in os.listdir(fname_)])
            pos_buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
            frame_idx = 0
            for i, frame_name in enumerate(pos_frames):
                if i == self.clip_len:
                    break
                frame = np.array(cv2.imread(frame_name)).astype(np.float64)
                pos_buffer[frame_idx] = frame
                frame_idx += 1

            neg_frames = sorted([os.path.join(fname, img) for img in os.listdir(fname)])
            neg_buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
            frame_idx = 0
            for i, frame_name in enumerate(neg_frames):
                if i == self.clip_len:
                    break
                frame = np.array(cv2.imread(frame_name)).astype(np.float64)
                neg_buffer[frame_idx] = frame
                frame_idx += 1

        return pos_buffer, neg_buffer, pos_label, neg_label

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        frame_idx = 0
        for i, frame_name in enumerate(frames):
            if i == self.clip_len:
                break
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[frame_idx] = frame
            frame_idx += 1

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        try:
            time_index = np.random.randint(buffer.shape[0] - clip_len)
        except:
            time_index = 0

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = ActivityNet(dataset='ActivityNet', split='train', category='sport', clip_len=16, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=1)

    for i, sample in enumerate(train_loader):
        pos_buffer = sample[0]
        neg_buffer = sample[1]
        pos_label = sample[2]
        neg_label = sample[3]

        print(pos_buffer.shape)
        print(neg_buffer.shape)
        print(pos_label)
        print(neg_label)

        if i == 1:
            break