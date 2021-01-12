#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Set of functions necessary to load and format data correctly for SGAN.

Created on Fri Aug 28 11:45:37 2020

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

# Settings
import settings
args = settings.parse_arguments()

""" 
Updates:
14.08:
  - added other_idx to find normal and pneum images not included in balanced dataset
  - this will allow us to enlarge the unlabelled dataset for training SGAN for 
    performance comparisons with ResNets etc.
"""

##########################################################################
#     Functions for balancing the dataset relative to COVID sample size
########################################################################## 
class balanceToCovid(data.Dataset):
    def __init__(self, csv_path):
        self.csvfile = _process_txt_file(csv_path) 
        self.labels, self.ids = _convert_labels(self.csvfile)
        self.n_covids = count_covids(self.labels)
        normal_files, normal_ids = [], []
        pneum_files, pneum_ids = [], []
        covid_files, covid_ids = [], []
        y = self.labels
        for j in range(len(y)):
          if y[j]==0: # if the label at row j is 'normal'
            normal_files.append(self.csvfile[j].split()[1])  # append normal_files with filename at that row 
            normal_ids.append(self.ids[j])                   # append with ids
          elif y[j]==1: # if the label at row j is 'pneum'
            pneum_files.append(self.csvfile[j].split()[1])
            pneum_ids.append(self.ids[j])                   # append with ids
          elif y[j]==2: # if the label at row j is 'covid'
            covid_files.append(self.csvfile[j].split()[1]) 
            covid_ids.append(self.ids[j])                   # append with ids

        # Store files as separate keys in a dict()
        self.all_files = {'normal': normal_files, 'pneum': pneum_files, 'covid': covid_files}
        self.all_ids = {'normal': normal_ids, 'pneum': pneum_ids, 'covid': covid_ids}
        self.n_files = [len(self.all_files[x]) for x in ['normal', 'pneum', 'covid']]
        
    def __call__(self):  
        # Create lists for storing randomly selected 265 files of non-covid classes, 
        # and add all lists together at the end to form our balanced dataset.
        # Return ids for assigning SGAN data.
        x_n = []
        x_p = []
        id_n = []
        id_p = []
        y_bal = []
        Xn_other, Xp_other = [], []    # images not included in balanced data
        rng = default_rng()
        for i in range(args.n_classes):
          rand_idx = rng.choice(range(self.n_files[i]), self.n_covids, replace=False)  # choose random, unique (replace=False) instances
          # now use rand_idx to get indices not included in balanced dataset
          other_idx = np.setdiff1d(range(self.n_files[i]), rand_idx) # find rows in range NOT picked in rand_idx
          # use other_idx to retreive images not added to balanced dataset
          if i==0:
            [x_n.append(self.all_files['normal'][j]) for j in rand_idx]
            [id_n.append(self.all_ids['normal'][j]) for j in rand_idx]
            [Xn_other.append(self.all_files['normal'][j]) for j in other_idx]
            # print(len(Xn_other))  # should be 7701
          elif i==1:
            [x_p.append(self.all_files['pneum'][j]) for j in rand_idx]
            [id_p.append(self.all_ids['pneum'][j]) for j in rand_idx]
            [Xp_other.append(self.all_files['pneum'][j]) for j in other_idx]
            # print(len(Xp_other)) # should be 5186
          X_bal = x_n + x_p + self.all_files['covid']   # all filenames
          y_bal = np.concatenate(( np.zeros(self.n_covids),np.ones(self.n_covids),np.full(self.n_covids,2) ), axis=None) 
          id_bal = id_n + id_p + self.all_ids['covid']  # all ids
          X_other = Xn_other + Xp_other
          y_other = np.concatenate( (np.zeros(len(Xn_other)), np.ones(len(Xp_other)) ), axis=None)
        return [X_bal, y_bal.astype(int), id_bal], [X_other, y_other.astype(int)]



##########################################################################
#    Function for taking in user specified number of images/class
#    and returning labelled and unlabelled subsets of our balanced
#    COVIDx dataset for training SGAN.
########################################################################## 
# Given a class-balanced dataset, and the number of images per class we want to
# train the model on (user input), return labelled and unlabelled subsets of the balanced_dataset according 
# to patient IDs. (to prevent duplicates).
def get_SGAN_data(balanced_dataset, n_per_class, n_classes):
    X, y, ids = balanced_dataset  # only use ids to select covid images (to avoid duplicates in unlabelled dataset)
    X_lbl, y_lbl = list(), list() # lists for storing labelled images
    X_unlbl = list()              # list for storing unlabelled images
    rng = default_rng()
    for c in range(n_classes):
      # get row indices where class i occurs 
      rows = np.where(y==c)[0]
      # now extract files and ids at those rows
      X_with_class = [X[r] for r in rows]
      ids_with_class = [ids[r] for r in rows]
      if c!=2:   # if we're not on COVID class, patients dont have multiple scans, so
        # choose random row instances for labelled data
        ilx = rng.choice(range(len(X_with_class)), n_per_class, replace=False) 
        # add to labelled list
        [X_lbl.append(X_with_class[j]) for j in ilx]
        [y_lbl.append(c) for j in ilx]
        # now use ilx to get indices not included in labelled dataset
        iux = np.setdiff1d(range(len(X_with_class)), ilx) # find elements in range(265) NOT in ilx
        # add to unlabelled list
        [X_unlbl.append(X_with_class[j]) for j in iux]
      # otherwise, we need to make sure we select images based on patient ID
      # so that we dont have duplicates in labelled and unlabelled datasets.
      else: 
        # randomly select indices
        ilx = rng.choice(range(len(X_with_class)), n_per_class, replace=False)
        # get the covid patient ids at those random indices
        poss_lbl_ids = [ids_with_class[j] for j in ilx] # get ids for testing inclusion validity
        count=0     # initialise image counter
        used_ids=[] # create list for storing patient IDs already included in dataset -> failsafe against duplicates
        # print('Counter:')
        for i in range(n_per_class):             # for 1:n_per_class patients 
          test_id = poss_lbl_ids[i]              # extract an id
          if i>0 and test_id in used_ids:        # v2 check: ensures randomness, rather than checking previous id in an ordered list
            # print('already included patient {}s data. go to next patient'.format(test_id))
            continue                             # -> go to next patient
          n_ids = ids_with_class.count(test_id)        # count how many times this id occurs in ids_with_class          
          if n_ids+count < n_per_class:                # if we are still under the max number of images
            count+=n_ids                               # increment counter 
            used_ids.append(test_id)                   # add id to list of patients used in dataset for checking duplicate IDs
            print(count)
            id_rows = np.where(np.asarray(ids_with_class)==np.asarray(test_id))[0]    # get the rows where that id occurs
            [X_lbl.append(X_with_class[r]) for r in id_rows] # append X_list with images for that patient id
            [y_lbl.append(c) for r in id_rows]               # append y_list with labels for that patient id
            continue
          elif n_ids+count > n_per_class:              # if by adding these images to X_list will exceed our limit 
            # print('adding patient {}s images would exceed {} image limit. go to next patient.'.format(test_id,n_per_class))
            continue                                          # -> go to next patient
          elif (n_ids+count)==n_per_class:             # if, by adding these images we have met our max limit, leave loop
            count+=n_ids
            used_ids.append(test_id)                   # add id to list of patients used in dataset for checking duplicate IDs
            id_rows = np.where(np.asarray(ids_with_class)==np.asarray(test_id))[0]    # get the rows where that id occurs
            [X_lbl.append(X_with_class[r]) for r in id_rows] # append X_list with images for that patient id
            [y_lbl.append(c) for r in id_rows]               # append y_list with labels for that patient id
            # Summarise:
            print('====================')
            print('count={}. \nFinished selecting {} patients for labelled dataset.'.format(count, len(used_ids)))
            print('*****')
            break
        # Now that we have our labelled covid_ids, use these to find patients
        # for unlabelled dataset
        unlbl_ids = np.setdiff1d(ids_with_class, used_ids) # get ids from all covid ids that arent in used_ids
        # use these ids to extract images and add to X_unlbl
        for i in range(len(unlbl_ids)):
          id = unlbl_ids[i]
          id_rows = np.where(np.asarray(ids_with_class)==np.asarray(id))[0]    # get the rows where that id occurs
          [X_unlbl.append(X_with_class[r]) for r in id_rows] # append X_list with images for that patient id
        print('====================')
        print('Finished selecting {} patients for unlabelled dataset.'.format(len(unlbl_ids)) )
        print('*****')
    print('====================')
    print('SGAN dataset summary:')
    print('====================')
    print('From balanced_dataset:')
    print('\tNumber of labelled images: {}, {} per class'.format(len(X_lbl), count) )
    print('\tNumber of unlabelled images: {}, {} per class'.format(len(X_unlbl), len(X_with_class)-count) )
    # print('Number of COVID-19 patients in labelled={} and unlabelled dataset={}.'.format(len(used_ids), len(unlbl_ids)) )
    # print('COVID-19 patients included in labelled dataset: {}'.format(used_ids) )
    # print('COVID-19 patients included in unlabelled dataset: {}'.format(unlbl_ids) )
    # print('-----')
    # print('Duplicates between labelled and unlabelled? {}'.format((np.setdiff1d(unlbl_ids, used_ids))))
    return [X_lbl, y_lbl], X_unlbl # return our labelled and unlabelled datasets


##########################################################################
#                           Helper functions  
##########################################################################    
""" Convert string labels to OHE versions """ 
# Take in csv/txt file, split, and extract 3rd column (which contains the str labels of pathologies)
# Next, encode these strings as integers using pathology_mapping. 
# Send to _one_hot_encode(). 
# Return OHE labels to __init__().
# choose to return either int_labels for in-built loss (here: nn.CrossEntropyLoss), or ohe_labels for custom loss.
def _convert_labels(csvfile):
    int_labels, ids = _get_int_labels(csvfile)
    ohe_labels = _one_hot_encode(int_labels)
    return int_labels, ids # return chosen label format along with patient ids for counting later

""" Take in .txt file, split apart, and return 3rd column as integer labels """
def _get_int_labels(csvfile):
    pathology_mapping = { 'normal': 0, 'pneumonia': 1, 'COVID-19': 2 } # assign integers to pathology labels
    int_labels = np.empty(len(csvfile)).astype(np.int)  # initialise storage array
    ids = [] 
    for i, patient_i in enumerate(csvfile):             # for each patient in our file list, get index i and string of info for that patient
        patient_list = patient_i.split()                # extract ith patient info as a list using split()
        pathology = patient_list[2]                     # get pathology name (in 3rd column)
        int_labels[i] = pathology_mapping[pathology]    # now map that name to its corresponding int_label
        # Count patients: use id since some patients have multiple images
        ids.append(patient_list[0])   
    return int_labels, ids  # return int_labels for all pathologies

""" Return one hot encoded equivalents of integer labels from _get_int_labels(). """
def _one_hot_encode(int_labels):
    labels = torch.from_numpy(int_labels)
    return F.one_hot(labels, args.n_classes)

""" Does the np.stacking offline so its quicker for the dataloader to retrieve images """
def _config_images(image):
    if len(image.shape) < 3:                              # if image is grayscale (H,W)
      image = np.stack((image, image, image), axis=2)     # stack to give new shape: (H, W, 3)
    elif image.shape[2] == 4: 
      imgray = image[:, :, 0]                             # remove colour channel
      image = np.stack((imgray, imgray, imgray), axis=2)  # new shape: (H, W, 3)
    return image

""" For resizing the image without needing to convert to PIL image (as with torchvision.transforms.Resize()) """
class Xray_resize(object):
    def __init__(self):
      self.size = args.GAN_imgresize
    def __call__(self, img):
      return cv2.resize(img, (self.size, self.size))

""" Read .txt file and store as list """
def _process_txt_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
    return files

""" Others """
def countPatients(ids):
    patientCount = 0
    for i in range(0, len(ids)):
        if i==0:
            patientCount += 1
            continue
        if ids[i]==ids[i-1]:
            continue
        else: 
            patientCount += 1
    return patientCount

def count_covids(labels):
    n_covids = 0
    for (i, y) in enumerate(labels):
        if y == 2:
            n_covids += 1 # only count covid images
        else:
            continue
    return n_covids

##########################################################################
#   Notes on the get_SGAN_data() algorithm
########################################################################## 
# Algorithm for selecting labelled/unlabelled dataset:
        # given a balanced dataset, return labelled data with N samples per class (user input) with no duplicates. 
        # for each class (normal,pneum,covid):
        #   find the rows in balanced_dataset that have that class label 
        #   extract all images and ids at those rows (265 per class)
        #   # Extracting N samples stage
        #   if class isnt covid:
        #     no duplicates/multiple scans, 
        #     so select N random indices between 0 and 265
        #     extract all these images/labels and add to our labelled dataset
        #     loop until all N norm and pneum data added to list (len=2N)
        #   else we are on covid class:
        #     have duplicates/multiple scans -> need to select N images based on patient IDs
        #     set N as limit, and use IDs to extract all images of each patient,
        #     randomly select row indices, 
        #     extract patient IDs at those rows -> pick one at a time (test_id),
        #     v1.<order IDs to allow for duplicate checks> ===> BAD!!!: never select images at end of list ==> not truly random
        #     v2.<check to make sure we havent already added this patients data using np.setdiff1d() >,
        #       if true:
        #         already included this patients data -> go to next patient id,
        #       else:
        #         using id, count how many images that patient has
        #        <check if including this patient's images will exceed N>
        #       if not: 
        #         increment counter,
        #         add id to used_ids list for duplicate checks,
        #         add images and labels to labelled dataset,
        #       if true:
        #         too many images -> try next patient,   
        #       iterate through patients until N is reached.          
        #       when counter==N -> break
        #       len(labelled dataset) now == 3N
        # finished. 
"""
Python pseudocode:
Algorithm: partition data into labelled and unlabelled subsets for training SGAN.
--------
Inputs: class-balanced dataset, num labelled images per class (n_per_class), num classes
Return: labelled dataset {X_label, y}, unlabelled dataset {X_unlabel}
--------
Initialise: 
lists for storing labelled dataset: {X_label, y}
list for storing unlabelled dataset: {X_unlabel}
--------
for c = 1 -> num_classes:
    # scan across all rows of dataset
    rows =                              # get row indices where class c occurs
    X_all, ids_all = balanced_data[rows] # extract filenames and patient ids at those rows
    if c is not COVID-19 class: 
      # if we're not on COVID class, no multiple scans per patient -> no need to consider patient IDs
      ### Get normal and pneumonia data 
      ## Labelled dataset
      # choose random row instances from files
      # get the files and labels of the images at those rows 
      # append image list and label list 
      ## Unlabelled dataset
      # find rows in balanced dataset that werent used in labelled dataset 
      # get files of the images at those rows
      X_# append image list
    else:  
      # otherwise, we need to make sure we select images based on patient ID
      # so that we dont have duplicates in labelled and unlabelled datasets.
      ### Get COVID data 
      ## Get labelled data
      # choose n_per_class random indices 
      for i = 1: 1 n_per_class:
          id = ...  # extract an ID
          if i>0 and (id is in used_ids):    # if current patient has already been included
            continue                           # move to next patient
          n_ids = ids_all.count(id)          # count how many images this patient has
          if (n_ids+count) < n_per_class:    # if adding these images keeps us below n_per_class
            counter+=1                         # increment counter
            used_ids.append()                  # store this patients ID in used_ids
            X_lbl.append(), y_lbl.append()     # append X_lbl and y_lbl
          elif (n_ids+count) > n_per_class:  # if adding these images to X_lbl will exceed our limit 
            continue                           # move to next patient
          elif (n_ids+count)==n_per_class:   # if by adding these images we have met our max limit
            break # leave loop


      ## Get unlabelled data


 # randomly select indices
        ilx = rng.choice(range(len(X_with_class)), n_per_class, replace=False)
        # get the covid patient ids at those random indices
        poss_lbl_ids = [ids_with_class[j] for j in ilx] # get ids for testing inclusion validity
        count=0     # initialise image counter
        used_ids=[] # create list for storing patient IDs already included in dataset -> failsafe against duplicates
        # print('Counter:')
        for i in range(n_per_class):             # for 1:n_per_class patients 
          test_id = poss_lbl_ids[i]              # extract an id
          if i>0 and test_id in used_ids:        # v2 check: ensures randomness, rather than checking previous id in an ordered list
            # print('already included patient {}s data. go to next patient'.format(test_id))
            continue                             # -> go to next patient
          n_ids = ids_with_class.count(test_id)        # count how many times this id occurs in ids_with_class          
          if n_ids+count < n_per_class:                # if we are still under the max number of images
            count+=n_ids                               # increment counter 
            used_ids.append(test_id)                   # add id to list of patients used in dataset for checking duplicate IDs
            print(count)
            id_rows = np.where(np.asarray(ids_with_class)==np.asarray(test_id))[0]    # get the rows where that id occurs
            [X_lbl.append(X_with_class[r]) for r in id_rows] # append X_list with images for that patient id
            [y_lbl.append(c) for r in id_rows]               # append y_list with labels for that patient id
            continue
          elif n_ids+count > n_per_class:              # if by adding these images to X_list will exceed our limit 
            # print('adding patient {}s images would exceed {} image limit. go to next patient.'.format(test_id,n_per_class))
            continue                                          # -> go to next patient
          elif (n_ids+count)==n_per_class:             # if, by adding these images we have met our max limit, leave loop
            count+=n_ids
            used_ids.append(test_id)                   # add id to list of patients used in dataset for checking duplicate IDs
            id_rows = np.where(np.asarray(ids_with_class)==np.asarray(test_id))[0]    # get the rows where that id occurs
            [X_lbl.append(X_with_class[r]) for r in id_rows] # append X_list with images for that patient id
            [y_lbl.append(c) for r in id_rows]               # append y_list with labels for that patient id
            # Summarise:
            print('====================')
            print('count={}. \nFinished selecting {} patients for labelled dataset.'.format(count, len(used_ids)))
            print('*****')
            break
        # Now that we have our labelled covid_ids, use these to find patients
        # for unlabelled dataset
        unlbl_ids = np.setdiff1d(ids_with_class, used_ids) # get ids from all covid ids that arent in used_ids
        # use these ids to extract images and add to X_unlbl
        for i in range(len(unlbl_ids)):
          id = unlbl_ids[i]
          id_rows = np.where(np.asarray(ids_with_class)==np.asarray(id))[0]    # get the rows where that id occurs
          [X_unlbl.append(X_with_class[r]) for r in id_rows] # append X_list with images for that patient id
        print('====================')
        print('Finished selecting {} patients for unlabelled dataset.'.format(len(unlbl_ids)) )
        print('*****')


"""
