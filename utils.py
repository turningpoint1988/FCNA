#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import copy
import math
import warnings


import numpy as np


# -----------------------------------------------------------------------------
#  compute the number of paramters
# -----------------------------------------------------------------------------


def get_n_params(parameters):
    pp = 0
    for p in parameters:
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
# -----------------------------------------------------------------------------
# Dict
# -----------------------------------------------------------------------------


def Dict(infile):
    dict={}
    lines = open(infile).readlines()
    for line in lines:
        line_split = line.strip().split()
        dict[line_split[0]] = int(line_split[1])
    return dict

# -----------------------------------------------------------------------------
# IOU Evaluation
# -----------------------------------------------------------------------------


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    for ind_class in range(n_class):
        print('===>' + 'label {}'.format(ind_class) + ':\t' + str(round(iu[ind_class] * 100, 2)))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return mean_iu, iu[0], iu[1]


