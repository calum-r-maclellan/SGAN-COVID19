#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:42:41 2020

@author: calmac
"""

# Standard packages for reading/processing data 
import numpy as np
from numpy.random import default_rng
import os
import cv2
from PIL import Image
import skimage
from skimage.io import imread
from imgaug import augmenters as iaa

# PyTorch libraries
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms

# my stuff
import dataset_helpers
import settings 
args = settings.parse_arguments()

##########################################################################
#                           X-ray data reader for TRAINING  
##########################################################################
class COVIDx_sgan(data.Dataset):
    def __init__(
            self,
            image_dir,
            dataset,
            labelled=True,
            transforms=True
        ):
        
        """ Initialise some things """
        self.imgpath = image_dir
        self.dataset = dataset
        self.labelled = labelled
        self.transforms = transforms      

    def __len__(self):
     
        if self.labelled:
          X, _ = self.dataset # get length from image array
          return len(X)
        else: 
          return len(self.dataset)
    
    def _transformImage(self,img):
        transform = transforms.Compose([
                  dataset_helpers.Xray_resize(),                            
                  transforms.ToTensor(), 
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]) 
        return transform(img)

    def __getitem__(self, idx):
        
        # Run this section if we are using lbl_dataloader
        if self.labelled: 
          # Get batch of labelled real images (s) from dataloader
          [self.X_lbl, self.labels] = self.dataset
          file_s = self.X_lbl[idx] # return files with label
          img_path = os.path.join(self.imgpath, file_s)    
          X_lbl = imread(img_path)
          X_lbl = dataset_helpers._config_images(X_lbl) 
          if self.transforms:  
            X_lbl = self._transformImage(X_lbl)
          labels = self.labels[idx]       
          return X_lbl, labels, file_s
          
        # otherwise, run if using unlbl_dataloader  
        else: 
          # Get batch of unlabelled real images (us) from dataloader
          self.X_unlbl = self.dataset
          file_us = self.X_unlbl[idx]           # return files without label
          img_path = os.path.join(self.imgpath, file_us)    
          X_unlbl = imread(img_path)
          X_unlbl = dataset_helpers._config_images(X_unlbl)           
          if self.transforms:  
            X_unlbl = self._transformImage(X_unlbl)
          return X_unlbl

##########################################################################
#                       X-ray data reader for TEST data. 
#
# Need different dataloaders since the training one loads both labelled and unlabelled
# data, whereas the testing one just loads images + labels for classification.
#
##########################################################################
class COVIDx_test(data.Dataset):
    def __init__(
            self,
            image_dir,
            txt_file,
            transforms=True
        ): 
        """ Initialise some things """
        self.imgpath = image_dir
        self.csv_file = dataset_helpers._process_txt_file(txt_file)   # read .txt file containing image info.
        self.labels, self.ids = dataset_helpers._convert_labels(self.csv_file)  # array of one hot encoded labels: processed here for faster indexing with dataloader

    def __len__(self):
        return len(self.csv_file)
    
    def _transformImage(self,img):
        transform = transforms.Compose([
                  dataset_helpers.Xray_resize(),                            
                  transforms.ToTensor(), 
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]) 
        return transform(img)

    def __getitem__(self, idx):
        samples = self.csv_file[idx].split()                # get full line of data at row idx 
        filenames = samples[1]                               # get filename from 2nd column
        img_path = os.path.join(self.imgpath, filenames)     # get image with that filename
        image = imread(img_path)
        imgs = dataset_helpers._config_images(image)
        imgs = self._transformImage(imgs)
        labels = self.labels[idx]
        ids = self.ids[idx]
        return imgs, labels, ids, filenames # return filenames for GradCAM

