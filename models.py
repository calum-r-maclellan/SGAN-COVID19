#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:42:41 2020

@author: calmac
"""
from __future__ import print_function
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchvision import models
import settings
args = settings.parse_arguments()


##########################################################################
#                           Semi-supervised model  
# Useful for:
#        - learning what a 'healthy' CXR/CT lung image looks like.
#        - leveraging limited labelled data available, whilst at the same time 
#          learning from unlabelled data. 
#        - dealing with domain shift/generalisation issues between datasets.

##########################################################################
  
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into conv. transpose for learning+upsampling 

            # in=[bs, 100, 1, 1], out=[bs, 512, 7, 7]
            # nn.ConvTranspose2d(latent_dim, 512, 8, 1, 0, bias=False), # uncommment for image size 128, 128
            nn.ConvTranspose2d(args.latent_dim, 512, 7, 1, 0, bias=False), # uncomment for size 224, 224
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # in: [bs, 512, 7, 7], out: [bs, 256, 14, 14]
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # in: [bs, 256, 14, 14], out: [bs, 128, 28, 28]
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # in: [bs, 128, 28, 28], out: [bs, 64, ..., ...]
            # nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # uncomment for image size = (128,128)
            nn.ConvTranspose2d(128, 64, 4, 4, 0, bias=False), # uncomment for image size = (224,224)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # in: [bs, 64, ..., ...], out: [bs, 3, 224, 224]
            nn.ConvTranspose2d(64, args.in_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise_input):
        return self.main(noise_input)


# Designed with separate output layer approach, where one layer deals with predicting the likelihood
# of the image being real (unsupervised branch), and the other deals with predicting the K class of the 
# images well, as well as whether it is a generated image or not (i.e. total=K+1 classes).
# There is another implementation proposed by Goodfellow that uses a stacked approach, but I've 
# done this first just to get it working. 
class customDiscriminator(nn.Module):
    def __init__(self):
        super(customDiscriminator, self).__init__()
        
        self.nconvs = 5
        self.fm_max = 512
        self.filter_size = [self.fm_max, self.fm_max//2, self.fm_max//4, self.fm_max//8] # [512, 256, 128, 64]
        
        # Use strided convolutions to downsample rather than maxpooling
        # to allow the network to learn its own pooling function.
        self.conv_blocks = nn.Sequential(
            # block 1: ConvLRelu:  in=[bs, 3, 224, 224], out=[bs, 64, 112, 112]
            nn.Conv2d(args.in_channels, self.filter_size[3], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # block 2: ConvBnLRelu in=[bs, 64, 112, 112], out=[bs, 128, 56, 56]
            nn.Conv2d(self.filter_size[3], self.filter_size[2], 3, 2, 1),
            nn.BatchNorm2d(self.filter_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            # block 3: ConvBnLRelu in=[bs, 128, 56, 56], out=[bs, 256, 28, 28]
            nn.Conv2d(self.filter_size[2], self.filter_size[1], 3, 2, 1),
            nn.BatchNorm2d(self.filter_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            # block 4: ConvBnLRelu in=[bs, 256, 28, 28], out=[bs, 512, 14, 14]
            nn.Conv2d(self.filter_size[1], self.filter_size[0], 3, 2, 1),
            nn.BatchNorm2d(self.filter_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            # block 5: ConvBnLRelu in=[bs, 512, 14, 14], out=[bs, 512, 7, 7]
            nn.Conv2d(self.filter_size[0], self.filter_size[0], 3, 2, 1),
            nn.BatchNorm2d(self.filter_size[0]),
            nn.LeakyReLU(0.2, inplace=True)             
        )

        # The height and width of downsampled image after nconvs blocks
        ds_size = args.GAN_imgresize // 2 ** self.nconvs # 224 // 2*5 = 7

        # Fully connected/classifier layer: in=[bs, 8192], out=[bs, 3]  
        self.fc = nn.Linear(self.filter_size[0] * ds_size ** 2, args.n_classes) 

    def custom_activation(self,output):
        # Use trick from Goodfellow et al to prevent creating two branches of supervised network.
        # Take unnormalised activations (pre softmax) from fc output (K classes), and normalise
        # them to values between [0,1] for real/fake prediction.
        Zx = torch.sum(torch.exp(output), dim=1)
        return Zx / (Zx + 1.0)

    def forward(self, img):
        # Output from convolution layers
        out = self.conv_blocks(img)         # conv_blocks output: [bs, 512, 4, 4]
        out = out.view(out.shape[0], -1)    # flattened features: [bs, 8192] (8192=512*4*4) 
        # Classifier output
        out = self.fc(out)                  # fc output: [bs, n_classes]
        supervised_output = out                               # raw class predictions: 
        unsupervised_output = self.custom_activation(out)     # real/fake prob -> BCE loss
        return supervised_output, unsupervised_output

# Load a pretrained, state-of-the-art (sota) classification architecture as the discriminator.
class sotaDiscriminator(nn.Module):
    def __init__(self, in_channels, n_classes, fixed=False):
        super(sotaDiscriminator, self).__init__()

        # Just change backbone for desired model by loading in different models
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        #num_ftrs = model.classifier[6].in_features
        if fixed:
          for param in model.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential()   # initialise backbone as sequential layers
        [self.backbone.add_module(name, child) for name, child in model.named_children() if name!= 'fc'] 
        self.fc = nn.Linear(num_ftrs, n_classes) # create new fc layer with 3 output classes to predict
    
    def custom_activation(self,output):
        # Use trick from Goodfellow et al to prevent creating two output branches.
        # Take unnormalised activations (pre softmax) from fc output (K classes), and normalise
        # them to values between [0,1] for real/fake prediction.
        Zx = torch.sum(torch.exp(output), dim=1)
        return Zx / (Zx + 1.0)

    def forward(self, x):        
        pooled_features = self.backbone(x)   
        pooled_features = pooled_features.view(pooled_features.size(0), -1)                
        output = self.fc(pooled_features)                      
        supervised_output = output
        unsupervised_output = self.custom_activation(output)
        return supervised_output, unsupervised_output
