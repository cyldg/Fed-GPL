import argparse
import collections
import json
import os
import random
import sys
import uuid
import wandb
import numpy as np
import PIL
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm
from domainbed import algorithms, datasets, hparams_registry
from domainbed.datasets import DG_Dataset
from domainbed.lib import misc
from domainbed.dataloader import my_dataloader
from domainbed.lib.fast_data_loader import FastDataLoader, InfiniteDataLoader
from domainbed.lib.torchmisc import dataloader
import pandas as pd
import datetime
import time
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from domainbed.vpt_structure import build_promptmodel

from utils import public_utils
import copy
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from iopath.common.file_io import PathManager as PathManagerBase
from iopath.common.file_io import HTTPURLHandler
PathManager = PathManagerBase()
output_folder = os.path.join("ISIC_VPL",datetime.now().strftime("%m%d%Y_%H%M%S"),"ISIC")
PathManager.mkdirs(output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--prompt',type=int,default =10)
    parser.add_argument('--iterations',type=int,default=50)
    parser.add_argument('--epoches',type=int,default=2)
    parser.add_argument('--data_len',type=int,default=400)
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')


    args = parser.parse_args()

    start_step = 0
    algorithm_dict = None

    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logout = f"ISIC_fed_vptdeep_{args.prompt}_{args.data_len}"
    logout = os.path.join(output_folder,logout)
    if torch.cuda.is_available():
        device = "cuda:1"
    else:
        device = "cpu"


    domain = ['cleantrain','dark_cornertrain','gel_bubbletrain','rulertrain','hairtrain']
    train_dataset  = []
    min_data_len = args.data_len

    for i in range(len(domain)):

        temp = DG_Dataset(root_dir=args.data_dir+'Skin/',domain=domain[i],split='train', color_jitter=True)

        print('origin traing dataset size:',len(temp))
        random_sample_indices = random.sample(list(range(len(temp))), min_data_len)
        temp = torch.utils.data.Subset(temp, random_sample_indices)
        train_dataset.append(temp)
        print('training dataset size:',len(train_dataset[i]))
    num_workers_eval = 8 
    batch_size_eval = 32
    domain_test = ['cleantest','dark_cornertest','gel_bubbletest','rulertest','hairtest']
    test_dataset = []
    for i in range(len(domain)):
        test_dataset.append(DG_Dataset(root_dir=args.data_dir+'Skin/',domain=domain_test[i],split='val', color_jitter=True))
    print('test dataset size:',len(test_dataset[i]))

    train_loader = []
    test_loader = []
    for i in range (len(domain)):
        train_loader.append(torch.utils.data.DataLoader(train_dataset[i], batch_size=batch_size_eval, shuffle=True, pin_memory=True))
        test_loader.append(torch.utils.data.DataLoader(test_dataset[i], batch_size=batch_size_eval, shuffle=False, pin_memory=True))
        print(len(train_loader[i]))


    client_num = 5
    server_model = build_promptmodel(num_classes=2, edge_size=224, patch_size=16,
                          Prompt_Token_num=args.prompt, VPT_type="Deep").to(device)
    client_models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]
    best_acc = 0.
    best_aoc = 0.
    loss_fun = nn.CrossEntropyLoss()
    lr = 5e-6

    for rounds in range(args.iterations):
        optimizers = [optim.SGD(params=client_models[idx].parameters(), lr=lr) for idx in range(client_num)]
        for i, model in enumerate(client_models):
            for  iters in range(args.epoches):
                train_loss, train_acc,train_aoc,accuracy,precision,recall,f1,roc_auc,average_precision,specificity = public_utils.ISIC_vpt_train(model, train_loader[i], optimizers[i], loss_fun, device)
                print(f'{domain[i]} | Train iters: {iters} | Train Loss: {train_loss:.{4}f} | Train Acc: {train_acc:.{4}f}| train aoc:{train_aoc:.{4}f}')
        public_utils.communication_fedavg(server_model, client_models)  

        test_acc_mean = []
        test_aoc_mean = []
        for client_idx, datasite in enumerate(domain):
            lo , test_acc ,test_aoc,accuracy,precision,recall,f1,roc_auc,average_precision,specificity= public_utils.test_ISIC_vpt(client_models[client_idx], test_loader[client_idx], loss_fun, device)
            test_acc_mean.append(test_acc)
            test_aoc_mean.append(test_aoc)

            print(f'Test Epoch:{rounds} | {domain[client_idx]} | Test Acc: {np.mean(test_acc):.{4}f}| test aoc:{test_aoc:.{4}f}')
            if logout:
                with open(f'{logout}.txt', 'a+') as f_out:
                    f_out.write(f'Test Epoch:{rounds} | {domain[client_idx]} | Test Acc: {np.mean(test_acc):.{4}f}| test aoc:{test_aoc:.{4}f} \n')   
        for client_idx, datasite in enumerate(domain):
            print("savemodel")
            dict = client_models[client_idx].obtain_prompt()
            print(dict.keys())
            torch.save(dict, os.path.join(output_folder,f'paramfor{client_idx}.pth'))
        if best_acc < np.mean(test_acc_mean):
            best_acc = np.mean(test_acc_mean)
        if best_aoc < np.mean(test_aoc_mean):
            best_aoc = np.mean(test_aoc_mean)
        print(f'Test Epoch:{rounds} | Gloabl | Test Acc Mean: {np.mean(test_acc_mean):.{4}f} | Best Acc: {best_acc:.{4}f} | Best Aoc: {best_aoc:.{4}f}')
        
        if logout:
            with open(f'{logout}.txt', 'a+') as f_out:
                f_out.write(f'Test Epoch:{rounds} | Test Acc Mean: {np.mean(test_acc_mean):.{4}f} | Best Acc: {best_acc:.{4}f} | Best Aoc: {best_aoc:.{4}f}\n')

