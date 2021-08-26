import os, sys
import json
import shutil
import cv2
import copy
import pdb

anno_file = open('./Annotations/activity_net.v1-3.min.json', 'r')
anno = json.load(anno_file)
src_data_dir = '~/data/ActivityNet/'
tgt_data_dir = '~/data/ActivityNet_processed/'

# analyze the label structure
label_tree = anno['taxonomy']
label_dict_file = './Annotations/label_dict.json'

if os.path.exists(label_dict_file):
    label_dict_f = open(label_dict_file, 'r')
    label_dict_ = json.load(label_dict_f)
    label_dict = label_dict_['label_dict']
    label_space = label_dict_['label_space']
else:
    label_dict = dict()
    org_label_space = ["Eating and drinking Activities", "Sports, Exercise, and Recreation", \
                       "Socializing, Relaxing, and Leisure", "Personal Care", "Household Activities"]
    label_space = ["eat_drink", "sport", "social", "personal_care", "household"]
    label_space_dict = dict(zip(org_label_space, label_space))

    for tmp_node in label_tree:
        if tmp_node['nodeName'] == 'Root' or tmp_node['nodeName'] in org_label_space:
            continue

        curr_node = copy.deepcopy(tmp_node)
        while not curr_node['parentName'] in org_label_space:
            for tmp_node_ in label_tree:
                if tmp_node_['nodeName'] == curr_node['parentName']:
                    curr_node = tmp_node_
                    break

        if not tmp_node['nodeName'] in label_dict:
            label_dict[tmp_node['nodeName']] = label_space_dict[curr_node['parentName']]

    label_dict_ = {'label_dict': label_dict, 'label_space': label_space}
    label_dict_json = json.dumps(label_dict_)
    label_dict_f = open(label_dict_file, 'w')
    label_dict_f.write(label_dict_json)
    label_dict_f.close()

print(label_space)
print('Label dictionary loaded.')

# analyze the training and validation videos for different categories
database = anno['database']
video_listing = os.listdir(src_data_dir)
zero_list = [0, 0, 0, 0, 0]
train_cnt = dict(zip(label_space, zero_list))
valid_cnt = dict(zip(label_space, zero_list))

for video_name in video_listing:
    vid = video_name[2:].split('.')[0]
    if vid in database:
        tmp_subset = database[vid]['subset']
        tmp_annos = database[vid]['annotations']

        if tmp_subset == 'training':
            for tmp_anno in tmp_annos:
                tmp_label = tmp_anno['label']
                train_cnt[label_dict[tmp_label]] += 1
        elif tmp_subset == 'validation':
            for tmp_anno in tmp_annos:
                tmp_label = tmp_anno['label']
                valid_cnt[label_dict[tmp_label]] += 1

print('Training instances: ', train_cnt)
print('Validation instances: ', valid_cnt)


# process each video to highlight and non-highlight segments
def compute_iou(segment1, segment2, epsilon=1e-5):
    # min_start_time = min(segment1[0], segment2[0])
    # max_end_time = max(segment1[1], segment2[1])
    # union = max_end_time - min_start_time
    union = segment1[1] - segment1[0]

    max_start_time = max(segment1[0], segment2[0])
    min_end_time = min(segment1[1], segment2[1])
    intersection = max(0, min_end_time - max_start_time)

    iou = intersection / (union + epsilon)

    return iou


if not os.path.exists(tgt_data_dir):
    os.mkdir(tgt_data_dir)

tgt_train_dir = os.path.join(tgt_data_dir, 'train')
tgt_valid_dir = os.path.join(tgt_data_dir, 'validation')

if not os.path.exists(tgt_train_dir):
    os.mkdir(tgt_train_dir)
    for tmp_category in label_space:
        tmp_category_dir = os.path.join(tgt_train_dir, tmp_category)
        if not os.path.exists(tmp_category_dir):
            os.mkdir(tmp_category_dir)

if not os.path.exists(tgt_valid_dir):
    os.mkdir(tgt_valid_dir)
    for tmp_category in label_space:
        tmp_category_dir = os.path.join(tgt_valid_dir, tmp_category)
        if not os.path.exists(tmp_category_dir):
            os.mkdir(tmp_category_dir)

resize_height = 128
resize_width = 171
segment_len = 16
tgt_fps = 5
iou_thr = 0.6

for video_name in video_listing:
    if video_name.split('.')[-1] != 'mp4':
        continue
    vid = video_name[2:].split('.')[0]

    if vid in database:
        tmp_subset = database[vid]['subset']
        if tmp_subset == 'training':
            tmp_subset_dir = tgt_train_dir
        elif tmp_subset == 'validation':
            tmp_subset_dir = tgt_valid_dir

        tmp_annos = database[vid]['annotations']
        tmp_category_list = list()

        for tmp_anno in tmp_annos:
            tmp_label = tmp_anno['label']
            if not label_dict[tmp_label] in tmp_category_list:
                tmp_category_list.append(label_dict[tmp_label])

        for tmp_category in tmp_category_list:
            tmp_video_dir = os.path.join(tmp_subset_dir, tmp_category, vid)
            if not os.path.exists(tmp_video_dir):
                os.mkdir(tmp_video_dir)

        tmp_init_dict = [{'time_segment': [], 'label': []} for tmp_category in tmp_category_list]
        tmp_category_dict = dict(zip(tmp_category_list, tmp_init_dict))

        # traverse the video, get the segments and labels for associated video category
        tmp_video_file = os.path.join(src_data_dir, video_name)
        capture = cv2.VideoCapture(tmp_video_file)

        fps = capture.get(cv2.CAP_PROP_FPS)
        sample_interval = max(1, int(fps / tgt_fps))
        frame_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cnt = 0
        frame_cnt = 0
        segment_id = 0
        retaining = True
        start_time = 0
        end_time = 0
        frame_buffer = list()

        while cnt < frame_num and retaining:
            retaining, frame = capture.read()
            if cnt % sample_interval != 0 or frame is None:
                cnt += 1
                continue

            if (frame_height != resize_height) or (frame_width != resize_width):
                frame = cv2.resize(frame, (resize_width, resize_height))
            frame_buffer.append(frame)
            frame_cnt += 1

            if frame_cnt == segment_len:
                # compare with all annotated segments
                end_time = float(cnt) / fps
                tmp_segment = [start_time, end_time]

                for tmp_category in tmp_category_list:
                    tmp_category_dict[tmp_category]['time_segment'].append(tmp_segment)
                    tmp_category_dict[tmp_category]['label'].append(-1)

                for tmp_anno in tmp_annos:
                    tmp_label = tmp_anno['label']
                    tmp_gt_segment = tmp_anno['segment']
                    iou = compute_iou(tmp_segment, tmp_gt_segment)
                    print('IoU: ', iou)
                    tmp_category_dict[label_dict[tmp_label]]['label'][-1] = iou

                # store the segment
                for tmp_category in tmp_category_list:
                    tmp_segment_dir = os.path.join(tmp_subset_dir, tmp_category, vid, 'segment_' + str(segment_id).zfill(4))
                    if not os.path.exists(tmp_segment_dir):
                        os.mkdir(tmp_segment_dir)

                    for frame_id, tmp_frame in enumerate(frame_buffer):
                        tmp_frame_name = os.path.join(tmp_segment_dir, str(frame_id).zfill(3) + '.jpg')
                        cv2.imwrite(filename=tmp_frame_name, img=tmp_frame)
                        print("Save frame: ", tmp_frame_name)

                frame_buffer = list()
                frame_cnt = 0
                segment_id += 1
                start_time = float(cnt + 1) / fps

            cnt += 1

        for tmp_category in tmp_category_list:
            tmp_label_dict = tmp_category_dict[tmp_category]
            tmp_label_dict_json = json.dumps(tmp_label_dict)

            tmp_video_dir = os.path.join(tmp_subset_dir, tmp_category, vid)
            tmp_label_file = open(os.path.join(tmp_video_dir, 'label.json'), 'w')
            tmp_label_file.write(tmp_label_dict_json)
            tmp_label_file.close()
