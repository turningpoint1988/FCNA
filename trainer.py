import os
import math
import datetime
import numpy as np
import os.path as osp
from copy import deepcopy
from utils import label_accuracy_score

import torch


class Trainer(object):
    """build a trainer"""
    def __init__(self, model, optimizer, criterion, device, checkpoint, start_epoch, max_epoch,
                 train_loader, test_loader, thres):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.checkpoint = checkpoint
        self.thres = thres
        self.epoch = 0
        self.iou_best = 0
        self.state_best = None

    def train(self):
        """training the model"""
        self.model.to(self.device)
        self.criterion.to(self.device)
        for epoch in range(self.start_epoch, self.max_epoch):
            # set training mode during the training process
            self.model.train()
            self.epoch = epoch
            # self.LR_policy.step() # for cosine learning strategy
            for i_batch, sample_batch in enumerate(self.train_loader):
                X_data = sample_batch["data"].float().to(self.device)
                denselabel = sample_batch["denselabel"].float().to(self.device)
                self.optimizer.zero_grad()
                denselabel_p = self.model(X_data)
                loss = self.criterion(denselabel_p, denselabel)
                if np.isnan(loss.item()):
                    raise ValueError('loss is nan while training')
                loss.backward()
                self.optimizer.step()
                print("epoch/i_batch: {}/{}---loss: {:.4f}---lr: {:.5f}".format(self.epoch, i_batch,
                                                    loss.item(), self.optimizer.param_groups[0]['lr']))
            # validation and save the model with higher accuracy
            self.test()

        return self.iou_best, self.state_best

    def test(self):
        """validate the performance of the trained model."""
        self.model.eval()
        seg_p_all = []
        seg_t_all = []
        for i_batch, sample_batch in enumerate(self.test_loader):
            X_data = sample_batch["data"].float().to(self.device)
            denselabel = sample_batch["denselabel"].float().to(self.device)
            with torch.no_grad():
                denselabel_p = self.model(X_data)
            seg_p_all.append(denselabel_p.view(-1).data.cpu().numpy() > self.thres)
            seg_t_all.append(denselabel.view(-1).data.cpu().numpy())

        iou, _, _ = label_accuracy_score(seg_t_all, seg_p_all, n_class=2)
        if self.iou_best < iou:
            self.iou_best = iou
            self.state_best = deepcopy(self.model.state_dict())
        print("iou: {:.3f}\n".format(iou))

