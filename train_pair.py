import os, sys
sys.path.append('./')
sys.path.append('./dataloaders')
sys.path.append('./network')

import timeit
from datetime import datetime
import socket
import glob
from tqdm import tqdm
import argparse

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import average_precision_score
import numpy as np

# from dataloaders.dataset import VideoDataset
from dataloaders.youtube_highlights import YouTube_Highlights
from network import C3D_model, R2Plus1D_model, R3D_model
from network import score_net

###################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'YouTube_Highlights', help = 'the dataset name')
parser.add_argument('--category', type = str, default = 'surfing', help = 'the category of videos to train on')
parser.add_argument('--model', type = str, default = 'C3D', help = 'the name of the model')
parser.add_argument('--epochs', type = int, default = 100, help = 'the number of training epochs')
parser.add_argument('--resume_epoch', type = int, default = 0, help = 'the epoch that the model store from')
parser.add_argument('--lr', type = float, default = 0.001, help = 'the learning rate')
parser.add_argument('--dropout_ratio', type = float, default = 0.5, help = 'the dropout ratio')
parser.add_argument('--batch_size', type = int, default = 10, help = 'the batch size')
parser.add_argument('--clip_len', type = int, default = 100, help = 'the length of each video clip')
parser.add_argument('--use_test', action = 'store_true', default = False,
                    help = 'whether to evaluate the model every epoch')
parser.add_argument('--test_interval', type = int, default = 20, help = 'the interval between tests')
parser.add_argument('--snapshot', type = int, default = 20, help = 'the interval between save models')
parser.add_argument('--gpu_id', type = str, default = None, help = 'the gpu device id')

opt = parser.parse_args()
print (opt)

###################################

# Use the specified GPU if available else revert to CPU
if opt.gpu_id is not None and torch.cuda.is_available():
    device = torch.device("cuda:" + opt.gpu_id)
else:
    device = torch.device("cpu")
print ("Device being used:", device)

# Define running configs
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if opt.resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
save_name = opt.model + '_' + opt.dataset

###################################

def train_model(epoch, train_loader, encoder, score_model, optimizer, scheduler, writer):
    # Train the score model to assign higher scores to highlight segments
    encoder.train()
    score_model.train()
    scheduler.step()

    start_time = timeit.default_timer()
    running_loss = 0.0

    for pos_inputs, neg_inputs, pos_labels, neg_labels in tqdm(train_loader):
        pos_inputs = pos_inputs.to(device)
        neg_inputs = neg_inputs.to(device)
        pos_labels = pos_labels.to(device)
        neg_labels = neg_labels.to(device)

        pos_emb = encoder(pos_inputs)
        pos_score = score_model(pos_emb).squeeze(-1)
        neg_emb = encoder(neg_inputs)
        neg_score = score_model(neg_emb).squeeze(-1)

        margin = (pos_labels - neg_labels) * 0.5
        score_diff = margin - pos_score + neg_score
        contrastive_loss = torch.mean(
            torch.max(torch.stack([score_diff, torch.zeros_like(score_diff)], dim=0), dim=0)[0])
        running_loss += contrastive_loss.item() * pos_inputs.shape[0]

        optimizer.zero_grad()
        contrastive_loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)

    print("[Train] Epoch: {}/{} Loss: {}".format(epoch + 1, opt.epochs, epoch_loss))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

    if epoch % opt.snapshot == (opt.snapshot - 1):
        torch.save({
            'epoch': epoch + 1,
            'encoder': encoder.state_dict(),
            'score_model': score_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth'))
        print("Save model at {}\n".format(
            os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth')))

    return

###################################

def test_model(epoch, test_loader, encoder, score_model, writer):
    # Evaluate the highlight score for each test segment
    encoder.eval()
    score_model.eval()

    start_time = timeit.default_timer()
    video_scores = dict()
    video_labels = dict()

    for inputs, labels, video_ids in tqdm(test_loader):
        inputs = inputs.to(device)

        emb = encoder(inputs)
        score = score_model(emb).squeeze(-1)

        for batch_idx in range(video_ids.shape[0]):
            tmp_score = score[batch_idx].item()
            tmp_label = labels[batch_idx].item()
            tmp_video_id = video_ids[batch_idx].item()

            if tmp_video_id in video_scores:
                video_scores[tmp_video_id].append(tmp_score)
                video_labels[tmp_video_id].append(tmp_label)
            else:
                video_scores[tmp_video_id] = list()
                video_scores[tmp_video_id].append(tmp_score)
                video_labels[tmp_video_id] = list()
                video_labels[tmp_video_id].append(tmp_label)

    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

    # Compute AP within each video and report their mean
    aps = list()
    for video_id in video_scores.keys():
        tmp_scores = np.array(video_scores[video_id], dtype=np.float32)
        tmp_scores = tmp_scores / tmp_scores.sum()

        tmp_labels = np.array(video_labels[video_id], dtype=np.float32)
        tmp_labels = np.float32(tmp_labels > 0)

        ap = average_precision_score(tmp_labels, tmp_scores)
        aps.append(ap)

    map = np.array(aps).mean()
    writer.add_scalar('data/test_map_epoch', map, epoch)
    print("[Test] Epoch: {}/{} mAP: {}".format(epoch + 1, opt.epochs, map))

    return

###################################

if __name__ == "__main__":
    if opt.model == 'C3D':
        encoder = C3D_model.C3D(pretrained=True, feature_extraction=True)
        score_model = score_net.ScoreFCN(emb_dim = 4096)
        train_params = [{'params':C3D_model.get_1x_lr_params(encoder), 'lr': 0},
                        {'params':score_model.parameters(), 'lr': opt.lr}]
    else:
        print('We only consider to use C3D model for feature extraction')
        raise NotImplementedError

    optimizer = optim.SGD(train_params, lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    if opt.resume_epoch == 0:
        print("Training from scratch...")
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', save_name + '_epoch-' + str(opt.resume_epoch - 1) + '.pth'),
                                map_location=lambda storage, loc: storage)
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', save_name + '_epoch-' + str(opt.resume_epoch - 1) + '.pth')))
        encoder.load_state_dict(checkpoint['encoder'])
        score_model.load_state_dict(checkpoint['score_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    encoder.to(device)
    score_model.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Start training on {} dataset...'.format(opt.dataset))
    train_dataset = YouTube_Highlights(dataset=opt.dataset, split='train', category=opt.category, clip_len=opt.clip_len)
    test_dataset = YouTube_Highlights(dataset=opt.dataset, split='test', category = opt.category, clip_len=opt.clip_len)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=1)
    best_test = 0

    for epoch in range(opt.resume_epoch, opt.epochs):
        train_model(epoch, train_loader, encoder, score_model, optimizer, scheduler, writer)

        if opt.use_test or epoch % opt.test_interval == (opt.test_interval - 1):
            test_result = test_model(epoch, test_loader, encoder, score_model, writer)
            if test_result > best_test:
                best_test = test_result

    print('Best test performance: ', best_test)

    writer.close()