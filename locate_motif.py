#!/usr/bin/python

import os
import sys
import time
import argparse
import math
import numpy as np
import os.path as osp

import torch
from torch.utils.data import DataLoader

# custom functions defined by user
from FCNmotif import FCN, FCNA, FCNGRU, FCNAGRU
from datasets import EPIDataSetTrain, EPIDataSetTest
from trainer import Trainer
from loss import OhemNegLoss
from utils import Dict


WINDOW = 50


def Rencode(seq_onehot):
    seq = ''
    for i in range(WINDOW):
        character = seq_onehot[:, i]
        if np.sum(character) == 0:
            seq += 'N'
            continue
        index = np.argmax(character)
        if index == 0:
            seq += 'A'
        elif index == 1:
            seq += 'C'
        elif index == 2:
            seq += 'G'
        elif index == 3:
            seq += 'T'
            
    return seq


def extract(denselabel_p, data, denselabel_t):
    position = 0
    maxcount = 0
    data = data[0].data.cpu().numpy()
    num = 1
    for i in range(len(denselabel_p) - WINDOW + 1):
        subseq = denselabel_p[i:(i+WINDOW)]
        count = np.sum(subseq)
        if maxcount < count:
            maxcount = count
            position = i
            num = 1
        elif maxcount == count:
            position += i
            num += 1
    start = int(np.ceil(position / num))
    seq_loc = data[:, start:(start+WINDOW)]
    seq = Rencode(seq_loc)
    denselabel = denselabel_t[start:(start+WINDOW)]
    return seq, denselabel


def locate(device, model, state_dict, train_loader, test_loader, outdir, thres, cv):
    # loading model parameters
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    f1 = open(osp.join(outdir, 'train_fold{}.fasta'.format(cv)), 'w')
    f2 = open(osp.join(outdir, 'train_label{}.txt'.format(cv)), 'w')
    total_count = 0
    count = 0
    for i_batch, sample_batch in enumerate(train_loader):
        X_data = sample_batch["data"].float().to(device)
        denselabel = sample_batch["denselabel"].float().to(device)
        with torch.no_grad():
            denselabel_p = model(X_data)
        denselabel_p = (denselabel_p.view(-1).data.cpu().numpy() > thres)
        denselabel_t = denselabel.view(-1).data.cpu().numpy()
        if np.sum(denselabel_p) < motifLen // 5:
            continue
        else:
            seq_tr, dense_tr = extract(denselabel_p, X_data, denselabel_t)
            total_count += 1
            if np.sum(dense_tr) > 0:
                count += 1
            f1.write('>seq{}\n'.format(i_batch))
            f1.write('{}\n'.format(seq_tr))
            f2.write('{}\n'.format(' '.join([str(int(e)) for e in dense_tr])))
    acc_tr = count / total_count
    f1.close()
    f2.close()
    #
    f1 = open(osp.join(outdir, 'test_fold{}.fasta'.format(cv)), 'w')
    f2 = open(osp.join(outdir, 'test_label{}.txt'.format(cv)), 'w')
    total_count = 0
    count = 0
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        denselabel = sample_batch["denselabel"].float().to(device)
        with torch.no_grad():
            denselabel_p = model(X_data)
        denselabel_p = (denselabel_p.view(-1).data.cpu().numpy() > thres)
        denselabel_t = denselabel.view(-1).data.cpu().numpy()
        if np.sum(denselabel_p) < motifLen // 5:
            continue
        else:
            seq_te, dense_te = extract(denselabel_p, X_data, denselabel_t)
            total_count += 1
            if np.sum(dense_te) > 0:
                count += 1
            f1.write('>seq{}\n'.format(i_batch))
            f1.write('{}\n'.format(seq_te))
            f2.write('{}\n'.format(' '.join([str(int(e)) for e in dense_te])))
    acc_te = count / total_count
    f1.close()
    f2.close()
    return acc_tr, acc_te


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FCN for motif location")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of a specified data.")

    parser.add_argument("-t", dest="thres", type=float, default=0.5,
                        help="threshold value.")
    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")
    parser.add_argument("-o", dest="outdir", type=str, default='./Refine/',
                        help="Where to save experimental results.")

    return parser.parse_args()


args = get_args()
motifLen_dict = Dict(os.getcwd() + '/motifLen.txt')
motifLen = motifLen_dict[args.name]


def main():
    """Create the model and start the training."""
    if torch.cuda.is_available():
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
    Data = np.load(osp.join(args.data_dir, '%s_data.npz' % args.name))
    seqs, denselabel = Data['data'], Data['denselabel']
    ##
    cv_num = 5
    interval = int(len(seqs) / cv_num)
    index = range(len(seqs))
    # choose the 1-fold cross-validation
    for cv in range(1):
        index_test = index[cv * interval:(cv + 1) * interval]
        index_train = list(set(index) - set(index_test))
        # build training data generator
        data_tr = seqs[index_train]
        denselabel_tr = denselabel[index_train]
        train_data = EPIDataSetTrain(data_tr, denselabel_tr)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=1)
        # build test data generator
        data_te = seqs[index_test]
        denselabel_te = denselabel[index_test]
        test_data = EPIDataSetTest(data_te, denselabel_te)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
        # Load weights
        checkpoint_file = osp.join(args.checkpoint, 'model_best%d.pth' % cv)
        chk = torch.load(checkpoint_file)
        state_dict = chk['model_state_dict']
        model = FCNA(motiflen=motifLen)
        acc_tr, acc_te = locate(device, model, state_dict, train_loader, test_loader, args.outdir, args.thres, cv)
        print("acc_tr: {:.3f}\tacc_te: {:.3f}\n".format(acc_tr, acc_te))


if __name__ == "__main__":
    main()

