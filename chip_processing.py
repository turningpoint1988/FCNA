# coding:utf-8
import os.path as osp
import os
import sys
import argparse
import itertools
import numpy as np
from Bio import SeqIO


SEQ_LEN = 501
INDEX = ['chr' + str(i + 1) for i in range(23)]
INDEX[22] = 'chrX'
CHROM = {}


def one_hot(sequence_dict, chrom, start, end):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    seq = str(sequence_dict[chrom].seq[start:end])
    for c in seq:
        temp.append(seq_dict.get(c, [0, 0, 0, 0]))
    return temp


def denselabel(data, pfmfile):
    """data: N*4*L, pfm: k*4"""
    pfm = []
    with open(pfmfile, 'r') as f:
        for line in f:
            line_split = line.strip().split()
            pfm.append([float(i) for i in line_split])
    pfm = np.asarray(pfm)
    pfm = pfm.transpose((1, 0))
    N, _, L = data.shape
    _, k = pfm.shape
    denselabels = []
    for i in range(N):
        data_row = data[i]
        records = np.zeros(L-k+1)
        for j in range(L-k+1):
            records[j] = np.sum(data_row[:, j:(j+k)] * pfm)
        best_index = np.argmax(records)
        denselabel_row = np.zeros(L)
        denselabel_row[best_index:(best_index+k)] = 1.
        denselabels.append(denselabel_row)

    return np.asarray(denselabels)  # N*L


def pos_location(chr, start, end, resize_len):
    original_len = end - start
    if original_len < resize_len:
        start_update = start - np.ceil((resize_len - original_len) / 2)
    elif original_len > resize_len:
        start_update = start + np.ceil((original_len - resize_len) / 2)
    else:
        start_update = start

    end_update = start_update + resize_len
    if end_update > CHROM[chr]:
        end_update = CHROM[chr]
        start_update = end_update - resize_len
    return int(start_update), int(end_update)


def get_data(seqs_bed, sequence_dict, pfmfile):
    seqs = []
    lines = open(seqs_bed).readlines()
    index = list(range(len(lines)))
    # np.random.shuffle(index)
    for i in index:
        line_split = lines[i].strip().split()
        chr = line_split[0]
        if chr not in INDEX:
            continue
        start, end = int(line_split[1]), int(line_split[2])
        start_p, end_p = pos_location(chr, start, end, SEQ_LEN)
        seqs.append(one_hot(sequence_dict, chr, start_p, end_p))       

    seqs = np.array(seqs, dtype=np.float32)
    seqs = seqs.transpose((0, 2, 1))

    labels_dense = denselabel(seqs, pfmfile)

    return seqs, labels_dense


def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    parser.add_argument("-d", dest="dir", type=str, default='')
    parser.add_argument("-n", dest="name", type=str, default='')

    return parser.parse_args()


def main():
    params = get_args()
    name = params.name
    data_dir = params.dir
    out_dir = osp.join(params.dir, 'data/')
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open('/home/**/FCNMotif/hg19/hg19.fa'), 'fasta'))
    print('Experiment on %s dataset' % name)
    seqs_bed = data_dir + '/all_sort_merge.bed'
    pfmfile = data_dir + '/%s.txt' % name
    seqs, labels_dense = get_data(seqs_bed, sequence_dict, pfmfile)

    np.savez(out_dir+'%s_data.npz' % name, data=seqs, denselabel=labels_dense)


if __name__ == '__main__':  main()
