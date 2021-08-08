import os,sys
sys.path.append('./')
from mypath import Path

import torch
import cv2
import math
import json
import shutil
import random
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class YouTube_Highlights_Set(Dataset):
    r"""YouTube Highlights Dataset for set-based learning. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]->[segments].

        Args:
            dataset (str): Name of dataset. Defaults to 'YouTube_Highlights'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            category (str): Determines which video category to use. Defaults to 'surfing'
            clip_len (int): Determines how many frames are there in each clip. Defaults to 100.
            set_size (int): Determines how many clips in the set for learning. Defaults to 20.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='YouTube_Highlights', split='train', category = 'surfing', clip_len=100, set_size=20,
                 preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        self.clip_len = clip_len
        self.set_size = set_size
        self.half_size = int(set_size // 2)
        self.split = split
        self.category = category
        folder = os.path.join(self.output_dir, split, category)

        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        # Define all the categories in the dataset
        if dataset == 'YouTube_Highlights':
            self.category_list = ['dog', 'gymnastics', 'parkour', 'skating', 'skiing', 'surfing']
            if not self.category in self.category_list:
                raise RuntimeError('The {} is not in the category list of {} dataset.'.format(self.category, dataset))

        if not self.check_integrity():
            raise RuntimeError('Neither the preprocessed dataset nor the raw one has been found.' +
                               ' You need to download it from official website.')

        if (not self.check_preprocess()) or preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filename of segments and the corresponding label
        self.fnames = list()
        self.video_names = list()
        self.segment2index = dict()
        self.labels = list()
        for video_id, video_name in enumerate(sorted(os.listdir(folder))):
            try:
                label_file = open(os.path.join(folder, video_name, 'match_label.json'), 'r')
            except:
                continue
            video_label = json.load(label_file)[1]
            tmp_video_listing = os.listdir(os.path.join(folder, video_name))
            tmp_video_listing.remove('match_label.json')

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
            if not os.path.exists(self.root_dir):
                return False
            else:
                return True
        else:
            return True

    def check_preprocess(self):
        # Check the existence of paths and image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for tmp_class in os.listdir(os.path.join(self.output_dir, 'train')):
            try:
                tmp_class_listing = os.listdir(os.path.join(self.output_dir, 'train', tmp_class))
            except:
                continue

            for tmp_video in tmp_class_listing:
                try:
                    tmp_video_listing = os.listdir(os.path.join(self.output_dir, 'train', tmp_class, tmp_video))
                except:
                    continue

                for tmp_segment in tmp_video_listing:
                    tmp_segment_path = os.path.join(self.output_dir, 'train', tmp_class, tmp_video, tmp_segment)
                    try:
                        tmp_first_frame = sorted(os.listdir(tmp_segment_path))[0]
                    except:
                        continue
                    frame_name = os.path.join(tmp_segment_path, tmp_first_frame)
                    first_frame = cv2.imread(frame_name)

                    if first_frame.shape[0] != self.resize_height or first_frame.shape[1] != self.resize_width:
                        return False

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        if not os.path.exists(os.path.join(self.output_dir, 'train')):
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split the train/val/test splits
        for tmp_class in self.category_list:
            class_path = os.path.join(self.root_dir, tmp_class)
            video_label_pairs = list()
            for tmp_video in os.listdir(class_path):
                video_path = os.path.join(class_path, tmp_video)
                try:
                    video_files = os.listdir(video_path)
                except:
                    continue

                video_label = [tmp_video, None, None]
                for tmp_file in video_files:
                    if tmp_file[-4:] == '.mp4':
                        video_label[1] = tmp_file
                    elif tmp_file == 'match_label.json':
                        video_label[2] = tmp_file

                if video_label[1] is not None and video_label[2] is not None:
                    video_label_pairs.append(video_label)

            train, test = train_test_split(video_label_pairs, test_size=0.3, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train')
            test_dir = os.path.join(self.output_dir, 'test')

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video_label in train:
                self.process_video(video_label, tmp_class, train_dir)
                src_label_path = os.path.join(class_path, video_label[0], video_label[2])
                tgt_label_path = os.path.join(train_dir, tmp_class, video_label[0], video_label[2])
                shutil.copy(src_label_path, tgt_label_path)
            for video_label in test:
                self.process_video(video_label, tmp_class, test_dir)
                src_label_path = os.path.join(class_path, video_label[0], video_label[2])
                tgt_label_path = os.path.join(test_dir, tmp_class, video_label[0], video_label[2])
                shutil.copy(src_label_path, tgt_label_path)

        print('Preprocessing finished.')

    def process_video(self, video_label, category, save_dir):
        video_name = video_label[0]
        if not os.path.exists(os.path.join(save_dir, category)):
            os.mkdir(os.path.join(save_dir, category))
        if not os.path.exists(os.path.join(save_dir, category, video_name)):
            os.mkdir(os.path.join(save_dir, category, video_name))

        label_file = os.path.join(self.root_dir, category, video_name, video_label[2])
        label = json.load(open(label_file, 'r'))
        segment_interval = label[0]
        for segment_id in range(len(label[1])):
            segment_path = os.path.join(save_dir, category, video_name, 'segment_' + str(segment_id).zfill(3))
            if not os.path.exists(segment_path):
                os.mkdir(segment_path)

        capture = cv2.VideoCapture(os.path.join(self.root_dir, category, video_name, video_label[1]))
        frame_cnt = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        segment_id = 0
        cnt = 0
        retaining = True

        # Process the original video to segments
        while cnt < frame_cnt and retaining:
            retaining, frame = capture.read()
            if frame is None:
                continue

            if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))

            if segment_id <= (len(label[1]) - 1):
                if cnt >= segment_interval[segment_id][0] and cnt <= (segment_interval[segment_id][1] - 1):
                    segment_path = os.path.join(save_dir, category, video_name, 'segment_' + str(segment_id).zfill(3))
                    segment_len = len(os.listdir(segment_path))
                    frame_name = os.path.join(segment_path, str(segment_len).zfill(5) + '.jpg')
                    cv2.imwrite(filename=frame_name, img=frame)
                    print("Process frame: ", frame_name)
                else:
                    segment_id += 1
                    if segment_id <= (len(label[1]) - 1):
                        if cnt >= segment_interval[segment_id][0] and cnt <= (segment_interval[segment_id][1] - 1):
                            segment_path = os.path.join(save_dir, category, video_name, 'segment_' + str(segment_id).zfill(3))
                            segment_len = len(os.listdir(segment_path))
                            frame_name = os.path.join(segment_path, str(segment_len).zfill(5) + '.jpg')
                            cv2.imwrite(filename=frame_name, img=frame)
                            print("Process frame: ", frame_name)

            if segment_id < (len(label[1]) - 1):
                if cnt >= segment_interval[segment_id+1][0] and cnt <= (segment_interval[segment_id+1][1] - 1):
                    segment_path = os.path.join(save_dir, category, video_name,
                                                'segment_' + str(segment_id + 1).zfill(3))
                    segment_len = len(os.listdir(segment_path))
                    frame_name = os.path.join(segment_path, str(segment_len).zfill(5) + '.jpg')
                    cv2.imwrite(filename=frame_name, img=frame)
                    print("Process frame: ", frame_name)

            cnt += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

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
            frame_interval = int(math.floor(100. / float(self.clip_len)))
            frame_idx = 0
            for i, frame_name in enumerate(frames):
                if not i % frame_interval == 0:
                    continue
                if i == (frame_interval * self.clip_len):
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

        # Crop the buffer
        buffer = buffer[:, time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = YouTube_Highlights_Set(dataset='YouTube_Highlights', split='train', clip_len=16, set_size=20,
                                        preprocess=False)
    test_data = YouTube_Highlights_Set(dataset='YouTube_Highlights', split='test', clip_len=16, set_size=20,
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