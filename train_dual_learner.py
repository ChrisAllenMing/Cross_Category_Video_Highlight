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
import random

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import numpy as np

from dataloaders.youtube_highlights_set import YouTube_Highlights_Set
from dataloaders.activity_net_set import ActivityNet_Set
from network import C3D_model
from network import transformer
from network import score_net

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
parser.add_argument('--min_ratio', type=float, default=0.3, help = 'the minimal ratio of target segments in a mix set')
parser.add_argument('--distill_weight', type=float, default=1.0, help = 'the weight of distillation loss')
parser.add_argument('--snapshot', type = int, default = 5, help = 'the interval between save models')
parser.add_argument('--gpu_id', type = str, default = None, help = 'the gpu device id')

opt = parser.parse_args()
opt.max_ratio = 1 - opt.min_ratio
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

def train_model(epoch, train_src_loader, train_tgt_loader, encoder, transformer, fine_score_model,
                coarse_score_model, optimizer, scheduler, writer, epsilon=1e-5):
    # Train the transformer and scoring model to predict the score distribution of a set of video segments
    transformer.train()
    fine_score_model.train()
    coarse_score_model.train()
    scheduler.step()

    tgt_data_iter = iter(train_tgt_loader)
    tgt_num_iter = len(train_tgt_loader)
    start_time = timeit.default_timer()
    running_sup_loss = 0.0
    running_distill_loss = 0.0
    iter_cnt = 0

    for src_inputs, src_labels in tqdm(train_src_loader):
        # The first dimension is assumed to be 1
        src_inputs = src_inputs.float().squeeze(0).to(device)
        src_labels = src_labels.float().squeeze(0).to(device)

        # Sample the mini-batch of target domain
        if iter_cnt != 0 and iter_cnt % tgt_num_iter == 0:
            tgt_data_iter = iter(train_tgt_loader)
        iter_cnt += 1

        tgt_inputs, _ = next(tgt_data_iter)
        tgt_inputs = tgt_inputs.float().squeeze(0).to(device)

        # Skip the batch if all segments are non-highlight
        if not torch.any(src_labels > 0):
            continue

        # Get the segment embeddings
        src_emb = encoder(src_inputs)
        tgt_emb = encoder(tgt_inputs)

        # Construct the mix embedding batch
        mix_set_size = min(src_emb.shape[0], tgt_emb.shape[0])
        tgt_ratio = opt.min_ratio + (opt.max_ratio - opt.min_ratio) * random.random()
        tgt_sample_size = max(1, int(mix_set_size * tgt_ratio))
        tgt_sample_indices = random.sample(range(tgt_emb.shape[0]), tgt_sample_size)
        tgt_component = tgt_emb[tgt_sample_indices, ...]

        src_sample_size = mix_set_size - tgt_sample_size
        src_sample_indices = random.sample(range(src_emb.shape[0]), src_sample_size)
        src_component = src_emb[src_sample_indices, ...]

        mix_emb = torch.cat([src_component, tgt_component], dim = 0)
        mix_labels = torch.ones(mix_set_size, dtype=src_labels.dtype, device=src_labels.device)
        mix_labels[:src_sample_size] = -1.

        # Get the scores for different segments in three kinds of sets
        if opt.use_transformer:
            src_emb_ = src_emb.unsqueeze(0)
            src_transformed_emb_ = transformer(src_emb_)
            src_transformed_emb = src_transformed_emb_.squeeze(0)
            src_score = fine_score_model(src_transformed_emb).squeeze(-1)

            tgt_emb_ = tgt_emb.unsqueeze(0)
            tgt_transformed_emb_ = transformer(tgt_emb_)
            tgt_transformed_emb = tgt_transformed_emb_.squeeze(0)
            fine_tgt_score = fine_score_model(tgt_transformed_emb).squeeze(-1)
            coarse_tgt_score = coarse_score_model(tgt_transformed_emb).squeeze(-1)

            mix_emb_ = mix_emb.unsqueeze(0)
            mix_transformed_emb_ = transformer(mix_emb_)
            mix_transformed_emb = mix_transformed_emb_.squeeze(0)
            mix_score = coarse_score_model(mix_transformed_emb).squeeze(-1)
        else:
            src_score = fine_score_model(src_emb).squeeze(-1)
            fine_tgt_score = fine_score_model(tgt_emb).squeeze(-1)
            coarse_tgt_score = coarse_score_model(tgt_emb).squeeze(-1)
            mix_score = coarse_score_model(mix_emb).squeeze(-1)

        src_score = torch.sigmoid(src_score) * 2. - 1.
        fine_tgt_score = torch.sigmoid(fine_tgt_score) * 2. - 1.
        coarse_tgt_score = torch.sigmoid(coarse_tgt_score) * 2. - 1.
        mix_score = torch.sigmoid(mix_score) * 2. - 1.

        # Define the supervised loss
        src_label_distribution = F.softmax(src_labels, dim=0)
        mix_label_distribution = F.softmax(mix_labels, dim=0)
        src_sup_loss = F.kl_div(F.log_softmax(src_score, dim=0), src_label_distribution)
        mix_sup_loss = F.kl_div(F.log_softmax(mix_score, dim=0), mix_label_distribution)
        sup_loss = src_sup_loss + mix_sup_loss

        # Define the distillation loss
        tgt_score_avg = (fine_tgt_score + coarse_tgt_score) / 2.
        tgt_score_distribution = F.softmax(tgt_score_avg, dim=0)
        fine_distill_loss = F.kl_div(F.log_softmax(fine_tgt_score, dim=0), tgt_score_distribution)
        coarse_distill_loss = F.kl_div(F.log_softmax(coarse_tgt_score, dim=0), tgt_score_distribution)
        distill_loss = (fine_distill_loss + coarse_distill_loss) / 2.

        loss = sup_loss + opt.distill_weight * distill_loss

        if torch.isnan(loss) or torch.isinf(loss):
            print('Skip the NaN or INF loss')
            continue
        running_sup_loss += sup_loss.item()
        running_distill_loss += distill_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_sup_loss = running_sup_loss / len(train_src_loader)
    epoch_distill_loss = running_distill_loss / len(train_src_loader)
    writer.add_scalar('data/train_sup_loss_epoch', epoch_sup_loss, epoch)
    writer.add_scalar('data/train_distill_loss_epoch', epoch_distill_loss, epoch)

    print("[Train] Epoch: {}/{} Supervised Loss: {} Distillation Loss: {}".format(epoch + 1, opt.epochs,
                                                                                  epoch_sup_loss, epoch_distill_loss))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

    if epoch % opt.snapshot == (opt.snapshot - 1):
        torch.save({
            'epoch': epoch + 1,
            'encoder': encoder.state_dict(),
            'transformer': transformer.state_dict(),
            'fine_score_model': fine_score_model.state_dict(),
            'coarse_score_model': coarse_score_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth'))
        print("Save model at {}\n".format(
            os.path.join(save_dir, 'models', save_name + '_epoch-' + str(epoch) + '.pth')))

    return

###################################

def test_model(epoch, test_loader, encoder, transformer, fine_score_model, coarse_score_model, writer, epsilon=1e-5):
    # Evaluate the highlight score for each test segment
    transformer.eval()
    fine_score_model.eval()
    coarse_score_model.eval()

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
            fine_score = fine_score_model(transformed_emb).squeeze(-1)
            coarse_score = coarse_score_model(transformed_emb).squeeze(-1)
        else:
            fine_score = fine_score_model(emb).squeeze(-1)
            coarse_score = coarse_score_model(emb).squeeze(-1)

        fine_score = torch.sigmoid(fine_score) * 2. - 1.
        coarse_score = torch.sigmoid(coarse_score) * 2. - 1.
        score = (fine_score + coarse_score) / 2.

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
        tmp_labels = np.float64(tmp_labels > 0)

        ap = average_precision_score(tmp_labels, tmp_scores)
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
        fine_score_model = score_net.ScoreFCN(emb_dim=4096)
        coarse_score_model = score_net.ScoreFCN(emb_dim=4096)
        train_params = [{'params':transformer.parameters(), 'lr': opt.lr},
                        {'params':fine_score_model.parameters(), 'lr': opt.lr},
                        {'params':coarse_score_model.parameters(), 'lr': opt.lr}]
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
        fine_score_model.load_state_dict(checkpoint['fine_score_model'])
        coarse_score_model.load_state_dict(checkpoint['coarse_score_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    encoder.to(device)
    transformer.to(device)
    fine_score_model.to(device)
    coarse_score_model.to(device)

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
    else:
        raise ValueError('Only YouTube Highlights is included in this demo project.')

    train_src_loader = DataLoader(train_src_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    train_tgt_loader = DataLoader(train_tgt_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1)

    # Training 
    for epoch in range(opt.resume_epoch, opt.epochs):
        train_model(epoch, train_src_loader, train_tgt_loader, encoder, transformer, fine_score_model,
                    coarse_score_model, optimizer, scheduler, writer)

    # Evaluation
    test_model(epoch, test_loader, encoder, transformer, fine_score_model, coarse_score_model, writer)

    writer.close()