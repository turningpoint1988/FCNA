# coding:utf-8
import os.path as osp
import os
import sys
import argparse
import itertools
import numpy as np
from Bio import SeqIO


def one_hot(seq):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    for c in seq:
        temp.append(seq_dict.get(c, [0, 0, 0, 0]))
    return temp


def get_data(train_fasta, train_label, test_fasta, test_label):
    seqs_all = []
    dense_all = []
    seqs_tr = open(train_fasta).readlines()[1::2]
    labels_tr = open(train_label).readlines()
    seqs_te = open(test_fasta).readlines()[1::2]
    labels_te = open(test_label).readlines()
    seqs = seqs_tr + seqs_te
    labels_dense = labels_tr + labels_te
    assert len(seqs) == len(labels_dense), "the length is not consistent."
    for i in range(len(seqs)):
        seq = seqs[i].strip()
        dense = labels_dense[i].strip().split()
        dense = [int(e) for e in dense]
        seqs_all.append(one_hot(seq))
        dense_all.append(dense)

    seqs_all = np.array(seqs_all, dtype=np.float32)
    seqs_all = seqs_all.transpose((0, 2, 1))
    dense_all = np.array(dense_all, dtype=np.int32)

    return seqs_all, dense_all


def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    parser.add_argument("-d", dest="dir", type=str, default='./Refine/TF')
    parser.add_argument("-n", dest="name", type=str, default='')

    return parser.parse_args()


def main():
    params = get_args()
    name = params.name
    data_dir = params.dir
    out_dir = osp.join(params.dir, 'data/')
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    print('Experiment on %s dataset' % name)
    train_fasta = data_dir + '/train_fold0.fasta'
    train_label = data_dir + '/train_label0.txt'
    test_fasta = data_dir + '/test_fold0.fasta'
    test_label = data_dir + '/test_label0.txt'
    seqs, labels_dense = get_data(train_fasta, train_label, test_fasta, test_label)

    np.savez(out_dir+'%s_data.npz' % name, data=seqs, denselabel=labels_dense)


if __name__ == '__main__':  main()
