#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:42:41 2020

@author: calmac
"""
from torch.nn import init
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_perform_stats(preds, labels, n_classes):
    accuracy = accuracy_score(labels, preds) 
    precisions = precision_score(labels, preds, average=None,
                                 labels=range(n_classes), zero_division=0.)
    recalls = recall_score(labels, preds, average=None, labels=range(n_classes),
                           zero_division=0.)
    f1 = f1_score(labels, preds, average=None, labels=range(n_classes),
                           zero_division=0.)
    f1_micro = f1_score(labels, preds, average='micro', labels=range(n_classes),
                           zero_division=0.)
    perform_stats = {'accuracy': accuracy, 'precision': precisions,
                     'recall': recalls, 'f1': f1, 'f1_weighted':f1_micro}
    return perform_stats  

def print_progress(epoch=None, n_epoch=None, n_iter=None, iters_one_batch=None,
                   mean_loss=None, cur_lr=None, metric_collects=None,
                   prefix=None):
    """
    Print the training progress.
    :epoch: epoch number
    :n_epoch: total number of epochs
    :n_iter: current iteration number
    :mean_loss: mean loss of current batch
    :iters_one_batch: number of iterations per batch
    :cur_lr: current learning rate
    :metric_collects: dictionary returned by function calc_multi_cls_measures
    :returns: None
    """
    accuracy = metric_collects['accuracy']
    precisions = metric_collects['precisions']
    recalls = metric_collects['recalls']

    log_str = ''
    if epoch is not None:
        log_str += 'Ep: {0}/{1}|'.format(epoch, n_epoch)

    if n_iter is not None:
        log_str += 'It: {0}/{1}|'.format(n_iter, iters_one_batch)

    if mean_loss is not None:
        log_str += 'Loss: {0:.4f}|'.format(mean_loss)

    log_str += 'Acc: {:.4f}|'.format(accuracy)
    templ = 'Pr: ' + ', '.join(['{:.4f}'] * 2) + '|'
    log_str += templ.format(*(precisions[1:].tolist()))
    templ = 'Re: ' + ', '.join(['{:.4f}'] * 2) + '|'
    log_str += templ.format(*(recalls[1:].tolist()))

    if cur_lr is not None:
        log_str += 'lr: {0}'.format(cur_lr)
    log_str = log_str if prefix is None else prefix + log_str
    print(log_str)
