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
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import numpy as np
import math

from dataloaders.youtube_highlights_set import YouTube_Highlights_Set
from dataloaders.activity_net_set import ActivityNet_Set
from network import C3D_model
from network import transformer
from network import score_net
from da_metrics import *

###################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'YouTube_Highlights', help = 'the dataset name')
parser.add_argument('--src_category', type = str, default = 'surfing', help = 'the category of videos to train on')
parser.add_argument('--tgt_category', type = str, default = 'skiing', help = 'the category of videos to test on')
parser.add_argument('--model', type = str, default = 'C3D', help = 'the name of the model')
parser.add_argument('--epochs', type = int, default = 50, help = 'the number of training epochs')
parser.add_argument('--resume_epoch', type = int, default = 0, help = 'the epoch that the model store from')
parser.add_argument('--lr', type = float, default = 0.001, help = 'the learning rate')
parser.add_argument('--dropout_ratio', type = float, default = 0.5, help = 'the dropout ratio')
parser.add_argument('--batch_size', type = int, default = 1, help = 'the batch size')
parser.add_argument('--set_size', type = int, default = 20, help = 'the maximum size of a set')
parser.add_argument('--clip_len', type = int, default = 16, help = 'the length of each video clip')
parser.add_argument('--use_transformer', action = 'store_true', default=False, help = 'whether to use transformer')
parser.add_argument('--depth', type=int, default=5, help = 'the depth of transformer')
parser.add_argument('--heads', type=int, default=8, help = 'the number of attention heads in transformer')
parser.add_argument('--mlp_dim', type=int, default=8192, help = 'the dimension of the internal feature in MLP')
parser.add_argument('--da_weight', type=float, default=1.0, help = 'the weight of domain adaptation loss')
parser.add_argument('--da_metric', type=str, default='mmd', choices=['mmd', 'coral'],
                    help='metric for domain adaptation')
parser.add_argument('--snapshot', type = int, default = 5, help = 'the interval between save models')
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

def train_model(epoch, train_src_loader, train_tgt_loader, encoder, transformer, score_model,
                optimizer, scheduler, writer, epsilon=1e-5):
    # Train the transformer and score model to predict the score distribution of a set of video segments
    encoder.train()
    transformer.train()
    score_model.train()
    scheduler.step()

    tgt_data_iter = iter(train_tgt_loader)
    tgt_num_iter = len(train_tgt_loader)
    start_time = timeit.default_timer()
    running_kl_loss = 0.0
    running_da_loss = 0.0
    iter_cnt = 0

    for inputs, labels in tqdm(train_src_loader):
        # The first dimension is assumed to be 1
        inputs = inputs.float().squeeze(0).to(device)
        labels = labels.float().squeeze(0).to(device)

        # sample the mini-batch of target domain
        if iter_cnt != 0 and iter_cnt % tgt_num_iter == 0:
            tgt_data_iter = iter(train_tgt_loader)
        iter_cnt += 1

        tgt_inputs, _ = next(tgt_data_iter)
        tgt_inputs = tgt_inputs.float().squeeze(0).to(device)

        # Skip the batch if all segments are non-highlight
        if not torch.any(labels > 0):
            continue

        # Get the scores for score category video segments
        emb = encoder(inputs)
        tgt_emb = encoder(tgt_inputs)
        if opt.use_transformer:
            emb_ = emb.unsqueeze(0)
            transformed_emb_ = transformer(emb_)
            transformed_emb = transformed_emb_.squeeze(0)
            score = score_model(transformed_emb).squeeze(-1)

            tgt_emb_ = tgt_emb.unsqueeze(0)
            tgt_transformed_emb_ = transformer(tgt_emb_)
            tgt_transformed_emb = tgt_transformed_emb_.squeeze(0)
        else:
            score = score_model(emb).squeeze(-1)
        score = torch.sigmoid(score) * 2. - 1.

        # Define the supervised loss
        label_distribution = F.softmax(labels, dim=0)
        kl_loss = F.kl_div(F.log_softmax(score, dim=0), label_distribution)

        # Define the domain adaptation loss
        if opt.da_metric == 'mmd':
            da_loss_func = mmd_rbf_noaccelerate
        elif opt.da_metric == 'coral':
            da_loss_func = coral_distance
        else:
            raise ValueError('Only mmd and coral are available')

        if opt.da_metric in ['mmd', 'coral']:
            min_set_size = min(emb.shape[0], tgt_emb.shape[0])
            if opt.use_transformer:
                da_loss = da_loss_func(transformed_emb[:min_set_size, :], tgt_transformed_emb[:min_set_size, :])
            else:
                da_loss = da_loss_func(emb[:min_set_size, :], tgt_emb[:min_set_size, :])
        else:
            da_loss = 0

        loss = kl_loss + opt.da_weight * da_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print('Skip the NaN or INF loss')
            continue
        running_kl_loss += kl_loss.item()
        running_da_loss += da_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_kl_loss = running_kl_loss / len(train_src_loader)
    epoch_da_loss = running_da_loss / len(train_src_loader)
    writer.add_scalar('data/train_kl_loss_epoch', epoch_kl_loss, epoch)
    writer.add_scalar('data/train_da_loss_epoch', epoch_da_loss, epoch)

    print("[Train] Epoch: {}/{} KL Loss: {} DA Loss: {}".format(epoch + 1, opt.epochs, epoch_kl_loss, epoch_da_loss))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

    if epoch % opt.snapshot == (opt.snapshot - 1):
        torch.save({
            'epoch': epoch + 1,
            'encoder': encoder.state_dict(),
            'transformer': transformer.state_dict(),
            'score_model': score_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth'))
        print("Save model at {}\n".format(
            os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth')))

    return

###################################

def test_model(epoch, test_loader, encoder, transformer, score_model, writer, epsilon=1e-5):
    # Evaluate the highlight score for each test segment
    encoder.eval()
    transformer.eval()
    score_model.eval()

    start_time = timeit.default_timer()
    video_scores = dict()
    video_labels = dict()

    for inputs, labels, index, video_id in tqdm(test_loader):
        inputs = inputs.float().squeeze(0).to(device)
        labels = labels.float().squeeze(0)
        index = index[0].item()
        video_id = video_id[0].item()

        emb = encoder(inputs)
        if opt.use_transformer:
            emb_ = emb.unsqueeze(0)
            transformed_emb_ = transformer(emb_)
            transformed_emb = transformed_emb_.squeeze(0)
            score = score_model(transformed_emb).squeeze(-1)
        else:
            score = score_model(emb).squeeze(-1)
        score = torch.sigmoid(score) * 2. - 1.

        if torch.any(torch.isnan(score)) or torch.any(torch.isinf(score)):
            print('Skip invalid samples')
            continue

        tmp_score = score[index].item()
        tmp_label = labels[index].item()

        if video_id in video_scores:
            video_scores[video_id].append(tmp_score)
            video_labels[video_id].append(tmp_label)
        else:
            video_scores[video_id] = list()
            video_scores[video_id].append(tmp_score)
            video_labels[video_id] = list()
            video_labels[video_id].append(tmp_label)

    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

    # Compute AP within each video and report their mean
    aps = list()
    for video_id in video_scores.keys():
        tmp_scores = np.array(video_scores[video_id], dtype=np.float64)
        large_scores = np.float64(tmp_scores > 200)
        tmp_scores = tmp_scores * (1 - large_scores) + 200. * large_scores
        tmp_scores = np.exp(tmp_scores) / (np.exp(tmp_scores).sum() + epsilon)

        tmp_labels = np.array(video_labels[video_id], dtype=np.float64)
        if opt.dataset == 'YouTube_Highlights':
            tmp_labels = np.float64(tmp_labels > 0)

        # Exclude the samples with invalid labels
        try:
            ap = average_precision_score(tmp_labels, tmp_scores)
        except:
            print('Skip the invalid sample')
            continue
        if math.isnan(ap) or math.isinf(ap):
            print('Skip invalid test video')
            continue
        aps.append(ap)

    map = np.array(aps).mean()
    writer.add_scalar('data/test_map_epoch', map, epoch)
    print("[Test] Epoch: {}/{} mAP: {}".format(epoch + 1, opt.epochs, map))

    return

###################################

if __name__ == "__main__":
    # Define the model
    if opt.model == 'C3D':
        encoder = C3D_model.C3D(pretrained=True, feature_extraction=True)
        transformer = transformer.Transformer(dim=4096, depth=opt.depth, heads=opt.heads, mlp_dim=opt.mlp_dim,
                                              dropout=opt.dropout_ratio)
        score_model = score_net.ScoreFCN(emb_dim=4096)
        if opt.da_metric in ['mmd', 'coral']:
            train_params = [{'params':C3D_model.get_1x_lr_params(encoder), 'lr': opt.lr * 0.1},
                            {'params':transformer.parameters(), 'lr': opt.lr},
                            {'params':score_model.parameters(), 'lr': opt.lr}]
        else:
            raise ValueError('Only mmd and coral are available')
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
        transformer.load_state_dict(checkpoint['transformer'])
        score_model.load_state_dict(checkpoint['score_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    encoder.to(device)
    transformer.to(device)
    score_model.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Define the data
    print('Start training on {} dataset...'.format(opt.dataset))
    if opt.dataset == 'YouTube_Highlights':
        train_src_dataset = YouTube_Highlights_Set(dataset=opt.dataset, split='train', category=opt.src_category,
                                                   clip_len=opt.clip_len, set_size=opt.set_size)
        train_tgt_dataset = YouTube_Highlights_Set(dataset=opt.dataset, split='train', category=opt.tgt_category,
                                                   clip_len=opt.clip_len, set_size=opt.set_size)
        test_dataset = YouTube_Highlights_Set(dataset=opt.dataset, split='test', category=opt.tgt_category,
                                              clip_len=opt.clip_len, set_size=opt.set_size)
    elif opt.dataset == 'ActivityNet':
        train_src_dataset = ActivityNet_Set(dataset=opt.dataset, split='train', category=opt.src_category,
                                            clip_len=opt.clip_len, set_size=opt.set_size)
        train_tgt_dataset = ActivityNet_Set(dataset=opt.dataset, split='train', category=opt.tgt_category,
                                            clip_len=opt.clip_len, set_size=opt.set_size)
        test_dataset = ActivityNet_Set(dataset=opt.dataset, split='validation', category=opt.tgt_category,
                                       clip_len=opt.clip_len, set_size=opt.set_size)
    else:
        raise ValueError('Dataset {} is not available.'.format(opt.dataset))

    train_src_loader = DataLoader(train_src_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    train_tgt_loader = DataLoader(train_tgt_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1)

    # Training
    for epoch in range(opt.resume_epoch, opt.epochs):
        train_model(epoch, train_src_loader, train_tgt_loader, encoder, transformer, score_model,
                    optimizer, scheduler, writer)

    # Evaluation
    test_model(epoch, test_loader, encoder, transformer, score_model, writer)

    writer.close()