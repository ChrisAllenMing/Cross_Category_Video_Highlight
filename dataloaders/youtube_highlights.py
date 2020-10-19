import os,sys
sys.path.append('./')
from mypath import Path

import torch
import cv2
import pdb
import json
import shutil
import random
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class YouTube_Highlights(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]->[segments]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'YouTube_Highlights'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how manu frames are there in each clip. Defaults to 100.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='YouTube_Highlights', split='train', category = 'surfing', clip_len=100, preprocess=False):
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
        # Going through each class folder one at a time
        self.fnames = list()
        self.video_names = list()
        self.segment2index = dict()
        self.labels = list()
        for video_id, video_name in enumerate(sorted(os.listdir(folder))):
            label_file = open(os.path.join(folder, video_name, 'match_label.json'), 'r')
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
        # video_name = video_label[1].split('.')[0]
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
                else:
                    segment_id += 1
                    if segment_id <= (len(label[1]) - 1):
                        if cnt >= segment_interval[segment_id][0] and cnt <= (segment_interval[segment_id][1] - 1):
                            segment_path = os.path.join(save_dir, category, video_name, 'segment_' + str(segment_id).zfill(3))
                            segment_len = len(os.listdir(segment_path))
                            frame_name = os.path.join(segment_path, str(segment_len).zfill(5) + '.jpg')
                            cv2.imwrite(filename=frame_name, img=frame)

            if segment_id < (len(label[1]) - 1):
                if cnt >= segment_interval[segment_id+1][0] and cnt <= (segment_interval[segment_id+1][1] - 1):
                    segment_path = os.path.join(save_dir, category, video_name,
                                                'segment_' + str(segment_id + 1).zfill(3))
                    segment_len = len(os.listdir(segment_path))
                    frame_name = os.path.join(segment_path, str(segment_len).zfill(5) + '.jpg')
                    cv2.imwrite(filename=frame_name, img=frame)

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
            index_ = random.sample(valid_indices.tolist(), 1)[0]

            fname_ = self.fnames[index_]
            neg_label = self.labels[index_]
            pos_label = label

            pos_frames = sorted([os.path.join(fname, img) for img in os.listdir(fname)])
            pos_buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
            for i, frame_name in enumerate(pos_frames):
                if i >= self.clip_len:
                    break
                frame = np.array(cv2.imread(frame_name)).astype(np.float64)
                pos_buffer[i] = frame

            neg_frames = sorted([os.path.join(fname_, img) for img in os.listdir(fname_)])
            neg_buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
            for i, frame_name in enumerate(neg_frames):
                if i >= self.clip_len:
                    break
                frame = np.array(cv2.imread(frame_name)).astype(np.float64)
                neg_buffer[i] = frame
        else:
            valid_indices = np.where(np.logical_and(np.array(self.video_names) == video_name, self.labels == 1))[0]
            index_ = random.sample(valid_indices.tolist(), 1)[0]

            fname_ = self.fnames[index_]
            pos_label = self.labels[index_]
            neg_label = label

            pos_frames = sorted([os.path.join(fname_, img) for img in os.listdir(fname_)])
            pos_buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
            for i, frame_name in enumerate(pos_frames):
                if i >= self.clip_len:
                    break
                frame = np.array(cv2.imread(frame_name)).astype(np.float64)
                pos_buffer[i] = frame

            neg_frames = sorted([os.path.join(fname, img) for img in os.listdir(fname)])
            neg_buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
            for i, frame_name in enumerate(neg_frames):
                if i >= self.clip_len:
                    break
                frame = np.array(cv2.imread(frame_name)).astype(np.float64)
                neg_buffer[i] = frame

        return pos_buffer, neg_buffer, pos_label, neg_label

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            if i >= self.clip_len:
                break
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

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
    train_data = YouTube_Highlights(dataset='YouTube_Highlights', split='train', clip_len=100, preprocess=False)
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