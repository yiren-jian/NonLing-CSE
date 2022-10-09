### By Chongyang Gao and Yiren Jian
### Adapted from https://github.com/YuanGongND/ssast

import argparse
import os
import ast
import pickle
import sys
import time
import torch
import json
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import numpy as np

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default=None, help="training data json")

parser.add_argument("--dataset", type=str, help="the dataset used for training")
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--num_mel_bins", type=int, default=128, help="number of input mel bins")

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")

parser.add_argument("-top_classes", type=int, default=50, help="how many classes is used. The classes are chosen with the descending order of the number of data samples.")

args = parser.parse_args()

# dataset spectrogram mean and std, used to normalize the input
norm_stats = {'librispeech':[-4.2677393, 4.5689974], 'howto100m':[-4.2677393, 4.5689974], 'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
target_length = {'librispeech': 1024, 'howto100m':1024, 'audioset':1024, 'esc50':512, 'speechcommands':128}
# if add noise for data augmentation, only use for speech commands
noise = {'librispeech': False, 'howto100m': False, 'audioset': False, 'esc50': False, 'speechcommands':True}


if __name__ == '__main__':

    input_json = '/home/yiren/ssast/src/prep_data/librispeech/librispeech_tr100_cut.json'

    args.num_mel_bins = 128
    args.target_length = 1024
    args.freqm = 48
    args.timem = 192
    args.mixup = 0.5
    args.top_classes = 50

    #### skip data preprocessing for mean and std

    # args.dataset_mean=mean_number
    # args.dataset_std=std_number

    args.dataset = 'librispeech'
    args.data_train = input_json
    args.noise=False
    # raise NotImplementedError
    audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset,
                  'mode':'train', 'noise':args.noise, 'skip_norm': True}


    # top k data
    with open(args.data_train, 'r') as fp:
            data_all = json.load(fp)
    top_dk = [data_all['top_k'][i][0] for i in range(args.top_classes)]
    print(top_dk)
    top_data = []
    top_label = {'index':[] ,'mid': [] ,'display_name': []}
    for i in range(len(data_all['data'])):
        if data_all['data'][i]['labels'] in top_dk:
            top_data.append(data_all['data'][i])
    index = 0
    for i in range(len(data_all['csv']['display_name'])):
        if data_all['csv']['display_name'][i] in top_dk:
            top_label['index'].append(index)
            top_label['mid'].append(data_all['csv']['mid'][i])
            top_label['display_name'].append(data_all['csv']['display_name'][i])
            index += 1


    args.batch_size = 20
    args.num_workers = 1
    audio_dataset = dataloader.AudioDataset(top_data, label_csv=top_label, audio_conf=audio_conf)
    print(len(audio_dataset))
    train_loader = torch.utils.data.DataLoader(
            audio_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    for i, (audio_input, labels) in enumerate(train_loader):
        bsz = audio_input.size(0)
        print(audio_input.shape)   #### [20, 1024, 128]   raw data
        print(labels.shape)   ##   [20, 251]
