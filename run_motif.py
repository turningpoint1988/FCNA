#!/usr/bin/python

import os
import sys
import time
import argparse
import math
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# custom functions defined by user
from FCNmotif import FCN, FCNA, FCNAR, FCNGRU, FCNAGRU, FCNAT
from datasets import EPIDataSetTrain, EPIDataSetTest
from trainer import Trainer
from loss import OhemNegLoss
from utils import Dict


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="EPIAttention Network for EPI interaction identification")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of a specified data.")

    parser.add_argument("-g", dest="gpu", type=str, default='1',
                        help="choose gpu device. eg. '0,1,2' ")
    parser.add_argument("-s", dest="seed", type=int, default=5,
                        help="Random seed to have reproducible results.")
    # Arguments for Adam or SGD optimization
    parser.add_argument("-b", dest="batch_size", type=int, default=1,
                        help="Number of sequences sent to the network in one step.")
    parser.add_argument("-lr", dest="learning_rate", type=float, default=0.01,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("-t", dest="threshold", type=float, default=0.5,
                        help="Threshold value for positive")
    parser.add_argument("-e", dest="max_epoch", type=int, default=30,
                        help="Number of training steps.")
    parser.add_argument("-w", dest="weight_decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")

    parser.add_argument("-r", dest="restore", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
        torch.manual_seed(args.seed)
    motifLen_dict = Dict(os.getcwd() + '/motifLen.txt')
    motifLen = motifLen_dict[args.name]
    Data = np.load(osp.join(args.data_dir, '%s_data.npz' % args.name))
    seqs, denselabel = Data['data'], Data['denselabel']
    ##
    cv_num = 5
    interval = int(len(seqs) / cv_num)
    index = range(len(seqs))
    f = open(osp.join(args.checkpoint, 'record.txt'), 'w')
    f.write('CV\tIOU\n')
    iou_mean = 0.
    for cv in range(cv_num):
        index_test = index[cv*interval:(cv+1)*interval]
        index_train = list(set(index) - set(index_test))
        # build training data generator
        data_tr = seqs[index_train]
        denselabel_tr = denselabel[index_train]
        train_data = EPIDataSetTrain(data_tr, denselabel_tr)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
        # build test data generator
        data_te = seqs[index_test]
        denselabel_te = denselabel[index_test]
        test_data = EPIDataSetTest(data_te, denselabel_te)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
        # we implement many trials for different weight initialization
        iou_best = 0
        trial_best = 0
        for trial in range(5):
            model = FCNAT(motiflen=motifLen)
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
            criterion = OhemNegLoss(device)
            start_epoch = 0
            if args.restore:
                print("Resume it from {}.".format(args.restore_from))
                checkpoint = torch.load(args.restore)
                state_dict = checkpoint["model_state_dict"]
                model.load_state_dict(state_dict)

            # if there exists multiple GPUs, using DataParallel
            if len(args.gpu.split(',')) > 1 and (torch.cuda.device_count() > 1):
                model = nn.DataParallel(model, device_ids=[int(id_) for id_ in args.gpu.split(',')])

            executor = Trainer(model=model,
                               optimizer=optimizer,
                               criterion=criterion,
                               device=device,
                               checkpoint=args.checkpoint,
                               start_epoch=start_epoch,
                               max_epoch=args.max_epoch,
                               train_loader=train_loader,
                               test_loader=test_loader,
                               thres=args.threshold)

            iou, state_dict = executor.train()
            if iou_best < iou:
                iou_best = iou
                trial_best = trial
                checkpoint_file = osp.join(args.checkpoint, 'model_best%d.pth' % cv)
                torch.save({
                    'model_state_dict': state_dict
                }, checkpoint_file)
        iou_mean += iou_best
        f.write("{}\t{}\t{:.3f}\n".format(cv, trial_best, iou_best))
        f.flush()
    f.write("mean\t{:.3f}\n".format(iou_mean/cv_num))
    f.close()


if __name__ == "__main__":
    main()

