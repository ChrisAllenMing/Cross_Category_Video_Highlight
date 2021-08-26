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

class ActivityNet_Set(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]->[segments]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ActivityNet'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            category (str): Determines which kind of video to use. Defaults to 'sport'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            set_size (int): Determines how many clips in the set for learning. Defaults to 20.
            preprocess (bool): Determines whether to preprocess dataset. Defaults is False.
    """

    def __init__(self, dataset='ActivityNet', split='train', category = 'sport', clip_len=16, set_size=20,
                 preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        self.clip_len = clip_len
        self.set_size = set_size
        self.half_size = int(set_size // 2)
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
        # Load a set of video segments
        if self.split == 'train':
            buffer, buffer_label = self.load_segment_set(index)
            buffer = self.crop(buffer, self.clip_len, self.crop_size)
            buffer = self.normalize(buffer)
            buffer = self.to_tensor(buffer)

            return torch.from_numpy(buffer), torch.from_numpy(buffer_label)
        else:
            buffer, buffer_label, buffer_index = self.load_segment_set(index)
            buffer = self.crop(buffer, self.clip_len, self.crop_size)
            buffer = self.normalize(buffer)
            buffer = self.to_tensor(buffer)
            video_id = np.array(self.segment2index[self.fnames[index]])

            return torch.from_numpy(buffer), torch.from_numpy(buffer_label), torch.tensor(buffer_index), \
                   torch.tensor(video_id)

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
        for i, segment in enumerate(buffer):
            for j, frame in enumerate(buffer[i]):
                frame -= np.array([[[90.0, 98.0, 102.0]]])
                buffer[i, j] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((0, 4, 1, 2, 3))

    # Load a set of video segments for training
    def load_segment_set(self, index):
        video_name = self.video_names[index]
        same_video_indices = np.where(np.array(self.video_names) == video_name)[0]

        if self.split == 'train':
            context_size = random.randint(1, self.half_size)
        else:
            context_size = self.half_size
        start_idx = index - context_size if (index - context_size) >= same_video_indices[0] else same_video_indices[0]
        end_idx = index + context_size if (index + context_size) <= same_video_indices[-1] else same_video_indices[-1]
        buffer_len = end_idx - start_idx + 1

        buffer = np.empty((buffer_len, self.clip_len, self.resize_height, self.resize_width, 3))
        buffer_label = np.empty(buffer_len)
        for segment_id in range(start_idx, end_idx + 1):
            fname = self.fnames[segment_id]
            frames = sorted([os.path.join(fname, img) for img in os.listdir(fname)])
            frame_idx = 0
            for i, frame_name in enumerate(frames):
                if i == self.clip_len:
                    break
                frame = np.array(cv2.imread(frame_name)).astype(np.float64)
                buffer[segment_id - start_idx, frame_idx] = frame
                frame_idx += 1

            label = self.labels[segment_id]
            buffer_label[segment_id - start_idx] = label

        if self.split == 'train':
            return buffer, buffer_label
        else:
            return buffer, buffer_label, index - start_idx

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        frame_interval = int(math.floor(100. / float(self.clip_len)))
        frame_idx = 0
        for i, frame_name in enumerate(frames):
            if not i % frame_interval == 0:
                continue
            if i == (frame_interval * self.clip_len):
                break
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[frame_idx] = frame
            frame_idx += 1

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        try:
            time_index = np.random.randint(buffer.shape[1] - clip_len)
        except:
            time_index = 0

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[2] - crop_size)
        width_index = np.random.randint(buffer.shape[3] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[:, time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = ActivityNet_Set(dataset='ActivityNet', split='train', category='sport', clip_len=16, set_size=20,
                                 preprocess=False)
    test_data = ActivityNet_Set(dataset='ActivityNet', split='validation', category='sport', clip_len=16, set_size=20,
                                preprocess=False)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    for i, train_sample in enumerate(train_loader):
        train_buffer = train_sample[0]
        train_label = train_sample[1]

        print('Training input shape: ', train_buffer.shape)
        print('Training label shape: ', train_label.shape)

        if i == 1:
            break

    for j, test_sample in enumerate(test_loader):
        test_buffer = test_sample[0]
        test_label = test_sample[1]
        test_index = test_sample[2]
        test_video_id = test_sample[3]

        print('Test input shape: ', test_buffer.shape)
        print('Test label shape: ', test_label.shape)
        print('Test index: ', test_index[0].item())
        print('Test video id: ', test_video_id[0].item())

        if j == 1:
            break