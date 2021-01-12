#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:42:41 2020

@author: calmac
"""

from __future__ import print_function
import time 
import os
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
import numpy as np

# my stuff
import dataset
import dataset_helpers
import models
import utils
import settings 
args = settings.parse_arguments()

device = torch.device(‘cuda’ if torch.cuda.is_available() else ‘cpu’)

##########################################################################
#                      Main loop for training SGAN
##########################################################################

def main():
    
    """ Get datasets """
    # Assign paths to images and csv file
    img_path, csv_path = train_img_path, train_csv_path
  
    n_per_class = int(input('How many images/class?: \n'))
    print('Retrieving dataset...')
    
    # Balance the dataset to COVID classes
    bal_data = dataset_helpers.balanceToCovid(csv_path)
    balanced_dataset, other_dataset = bal_data()
    [X_other, y_other] = other_dataset
    print(len(X_other))
    
    # Now create labelled and unlabelled dataset according to patient ID
    labelled_dataset, unlabelled_dataset = dataset_helpers.get_SGAN_data(balanced_dataset, n_per_class, n_classes=args.n_classes)
    
    # Add images from other_dataset (unused norm and pneum) into unlabelled dataset to make 
    # it full size.
    full_ul_dataset = unlabelled_dataset + X_other
    print('Total unlabelled dataset size: {}'.format(len(full_ul_dataset)))
    
    # Define separate datasets w/ corresponding loaders for labelled/unlabelled dataset 
    # Assign different training pipelines to iterate over  
    lbl_dataset = dataset.COVIDx_sgan(img_path, labelled_dataset, labelled=True)
    
    # Full unlabelled dataset
    full_unlbl_dataset = dataset.COVIDx_sgan(img_path, full_ul_dataset, labelled=False)
    
    
    """ Assign dataloaders """
    # Now assign different dataloaders to each dataset pipeline
    lbl_dataloader = torch.utils.data.DataLoader(lbl_dataset, batch_size=args.l_batchSize,
                                                 shuffle=True, num_workers=args.num_workers)
    full_unlbl_dataloader = torch.utils.data.DataLoader(full_unlbl_dataset, batch_size=args.ul_batchSize,
                                                   shuffle=True, num_workers=args.num_workers) 
    
    dataloaders = {'label':lbl_dataloader, 'unlabel':full_unlbl_dataloader}
    
    # Start training
    train_SGAN(dataloaders, resume_training=False)

##########################################################################
#                      Training functions
##########################################################################

def train_SGAN(dataloaders, resume_training=False, G_path=None, D_path=None):
    
    """ Functions for writing training results to .txt files """
    def log_string_loss(out_str):  
            LOG_loss.write(out_str+'\n')
            LOG_loss.flush()
    
    def log_string_acc(out_str):  
            LOG_acc.write(out_str+'\n')
            LOG_acc.flush()

    """ Get dataloaders """
    lbl_dataloader = dataloaders['label']
    full_unlbl_dataloader = dataloaders['unlabel']
    
    """ Loss functions """
    unsupervised_loss = torch.nn.BCELoss()
    supervised_loss   = torch.nn.CrossEntropyLoss()
    
    """ Send to GPU """
    cuda = True if torch.cuda.is_available() else False
    
    """ Initialise models and weights """
    if resume_training:
      '''Generator params'''
      # Initialise 
      G_checkpoint = torch.load(G_path)
      generator = models.Generator().cuda()
      optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
      # Flood params with state_dict()
      generator.load_state_dict(G_checkpoint['model_state_dict'])
      optimizer_G.load_state_dict(G_checkpoint['optim_state_dict'])
      g_epochLoss = G_checkpoint['loss']
      generator.train()
      '''Repeat for Discrim'''
      D_checkpoint = torch.load(D_path)
      discriminator = models.sotaDiscriminator().cuda()
      optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
      discriminator.load_state_dict(D_checkpoint['model_state_dict'])
      optimizer_D.load_state_dict(D_checkpoint['optim_state_dict'])
      d_epochLoss = D_checkpoint['loss']
      discriminator.train()
      # also load which epoch training stopped 
      epochs_done = D_checkpoint['epoch']
      num_epochs = 100-epochs_done
    else:
      # load weights into G
      generator = models.Generator().cuda()
      generator.apply(utils.init_weights)
      generator.train()
      optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
      # repeat for D
      discriminator = models.sotaDiscriminator(args.in_channels, args.n_classes).cuda()
      discriminator.train()
      optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    
    """ Start training """
    t_start = time.time()
    print('Start training...')

    for epoch in range(args.num_epochs):
      print()
      print('Epoch {}/{}'.format(epoch+1, args.num_epochs))
      # Initialise lists for storing epoch results
      dLossList, gLossList = [], [] # only want supervised loss list
      acc_list, f1_list = [], []    # perform stats per epoch                    

      """ Start iterating over UNLABELLED data """
      for i, X_unlbl in enumerate(full_unlbl_dataloader):

          # Need this as a failsafe against last batch iteration (i) 
          # having smaller size than batch_size
          bs = X_unlbl.shape[0]

          “”” Get batch of LABELLED DATA separate from the main iterator """
          [X_lbl, labels,_] = next(iter(lbl_dataloader))

          # Configure inputs
          X_lbl, labels, X_unlbl = Variable(X_lbl.type(FloatTensor)), Variable(labels.type(LongTensor)), Variable(X_unlbl.type(FloatTensor))
          X_lbl, labels, X_unlbl = X_lbl.to(device), labels.to(device), X_unlbl.to(device)

          """ Adversarial ground truths """
          real, fake = Variable(torch.ones(bs).type(FloatTensor), requires_grad=False), Variable(torch.zeros(bs).type(FloatTensor), requires_grad=False)
          real, fake = real.to(device), fake.to(device)

          # ============================================================
          #  Update Discriminator: 3 different losses:
          #     Real images:
          #     1. supervised loss: for classifying small dataset as a normal 
          #        classifier would. 
          #     2. (real) unsupervised loss: want D to learn what
          #        a real image is, so supply unlabelled image pred with a real label (1) 
          #     Fake images:
          #     3. (fake) unsupervised loss: also want D to learn when a fake image 
          #        comes from the generator, so give generated image pred a fake label (0)
          # ============================================================

          optimizer_D.zero_grad()
       
          """ Loss for real images """
          # Labelled real images
          pred_sup, _ = discriminator(X_lbl)                # return supervised class output (softmax probs)
          d_sup_loss = supervised_loss(pred_sup, labels)    # compare with labels 
          d_sup_loss.backward()                             # calculate gradients and backprop
          dLossList.append(d_sup_loss.detach().cpu().numpy())

          # Unlabelled real images
          _, pred_real = discriminator(X_unlbl)             # return unsupervised real/fake prob
          d_real_loss = unsupervised_loss(pred_real, real)  # image is actually real, so give real label
          d_real_loss.backward()                            # calculate gradients and backprop

          """ Loss for fake images""" 
          # To the discriminator, these images are fake so we use the 'fake' labels
          # here to teach it that they are fake. (contrasted with generators loss, 
          # which uses 'real' labels since it considers its outputs real!).
          # Sample noise (and labels: next version) as generator input
          z = torch.randn(bs, args.latent_dim, 1, 1, device=device)
          # gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, bs))) # uncomment for including labels as conditional input for G
          gen_imgs = generator(z) # Generate a batch of fake images

          # Train D to detect fakes from G
          _, pred_fake = discriminator(gen_imgs.detach()) # .detach() to prevent gradients flowing through G
          d_fake_loss = unsupervised_loss(pred_fake, fake)
          d_fake_loss.backward() 

          """ Total discriminator loss """
          d_unsup_loss = (d_real_loss + d_fake_loss) / 2 
          optimizer_D.step()

          # ============================================================
          # Update Generator
          #   From the generators perspective, its images are real. 
          #   So when we send gen_imgs through D, and compute the loss,
          #   we use the 'real' labels for the generators update. 
          # ============================================================

          optimizer_G.zero_grad()

          # Sample noise and labels as generator input
          z = torch.randn(bs, args.latent_dim, 1, 1, device=device)
          # gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, bs))) 
          gen_imgs = generator(z) # Generate a batch of images

          # Loss measures generator's ability to fool the discriminator.
          # When compared with 1, a value of 0 from D indicates that 
          # G needs to massively change itself to make its images better,
          # and vice versa.
          _, prob_gen = discriminator(gen_imgs)      # get D guess at how likely this image is generated
          g_loss = unsupervised_loss(prob_gen, real) # eval how far this is away from 1 (real): this forces the generators loss to increase for low prob_gen, making images more real on next iter
          g_loss.backward()
          gLossList.append(g_loss.detach().cpu().numpy())
          optimizer_G.step()

          # ============================================================
          # Performance stats
          #   Calculate discriminator classification accuracy
          # ============================================================
          probs = F.softmax(pred_sup, dim=1).detach().cpu().numpy()
          preds = np.argmax(probs, axis=1)
          lbls = labels.detach().cpu().numpy()  # get true label
          perform_stats = utils.compute_perform_stats(preds, lbls, args.n_classes)
          acc_list.append(perform_stats['accuracy'])
          f1_list.append(perform_stats['f1'])
  
      # Print epoch results    
      d_epochLoss = np.mean(dLossList)
      g_epochLoss = np.mean(gLossList)
      acc_epoch = np.mean(acc_list)
      f1_epoch = np.mean(f1_list)

      print('D loss: {:.4f}, acc: {:.4f}, f1: {:.4f}\nG loss: {:.4f}'.format(
              d_epochLoss, acc_epoch, f1_epoch, g_epochLoss
              )
      )
      # print('D sup_loss: {:.4f}, d_realloss: {:.4f}, d_fakeloss: {:.4f}, avg_unsup_loss: {:.4f} \nG loss: {:.4f}'.format(
      #         d_epochLoss.item(), d_real_loss.item(), d_fake_loss.item(), d_unsup_loss.item(),     g_epochLoss.item()
      #     ) 
      # )

      # Write D loss results to txt file
      log_string_loss( ('%f') % (d_epochLoss) )
      log_string_acc( ('%f') %  (acc_epoch) )

      # Save model and optimiser state_dict() for D and G every epoch to resume training if it disconnects
      if args.save_model() and (epoch+1) % 1 == 0:

       	 # Make file names  
        D_file = ('D_epoch_{}_classLoss_{:.4f}_acc_{:.4f}_f1_{:.4f}.pth.tar’.format(
                    epoch+1, d_epochLoss, acc_epoch, f1_epoch)
        )
        G_file = ('G_epoch_{}_loss_{:.4f}.pth.tar’.format(
                    epoch+1, g_epochLoss)
        )

        # Save D model and optim state dict
        torch.save({
            'epoch': epoch,
            'loss': d_epochLoss,
            'model_state_dict': discriminator.state_dict(),
            'optim_state_dict': optimizer_D.state_dict(),
        }, os.path.join(train_train_model_dir, D_file))
        # repeat for G
        torch.save({
            'epoch': epoch,
            'loss': g_epochLoss,
            'model_state_dict': generator.state_dict(),
            'optim_state_dict': optimizer_G.state_dict(),
        }, os.path.join(train_train_model_dir, G_file))
        
      
    """ end training """
    time_elapsed = time.time() - t_start
    print('Training complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))
    log_string_loss( ('%f') % (time_elapsed//60) )


if __name__ == '__main__':
    
    """ Set up stuff """
    # Assign GPU or CPU (depending on whether we want to train (GPU) or not (CPU))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Data directories
    # path to get to our data: change only this line below to run on any GPU enabled PC 
    pc_root = '/home/Users/calummac/COVID-19'   # lab GPU PC
    #pc_root = 'gdrive/My Drive'                 # Google drive for using Colab notebook ("semi_GAN.ipynb"")
    data_root  = os.path.join(pc_root,'Datasets/ChestXray_Datasets/COVIDx_v3/data')   # root to file location in Gdrive
    train_img_path   = os.path.join(data_root, 'train')                  # image folder
    train_csv_path   = os.path.join(data_root, 'train_COVIDx_v3.txt')    # spreadsheet of image/patient meta datatest_image_dir = 'test'                
    test_img_path    = os.path.join(data_root, 'test')  
    test_csv_path    = os.path.join(data_root, 'test_COVIDx_v3.txt')    
    
    # Create folder to store model paths
    model_path = os.path.join(data_root, 'experiments')  
    os.makedirs(model_path, exist_ok=True)
    
    # Specific model paths 
    semigan_dir = os.path.join(model_path, 'semigan')  # create folder for storing model weights
    os.makedirs(semigan_dir, exist_ok=True)
    
    # Create folder for model trained on the test data (smaller, lower performance)
    train_test_model_dir  = os.path.join(semigan_dir, 'train_on_test')   
    os.makedirs(train_test_model_dir, exist_ok=True)
    
    # Create folder for model trained on the training data (larger)
    #train_train_model_dir = os.path.join(semigan_dir, 'train_on_train')  
    #os.makedirs(train_train_model_dir, exist_ok=True)
    
    # DISCRIMINATOR: Text files for storing loss/acc results for discriminator/classifier
    train_log_root = os.path.join(train_train_model_dir, 'train_log')   # path to folder
    if not os.path.exists(train_log_root): os.mkdir(train_log_root)
    LOG_loss = open(os.path.join(train_log_root, 'D_lossPerEpoch.txt'), 'w')
    LOG_acc = open(os.path.join(train_log_root, 'D_accPerEpoch.txt'), 'w')
    
    ''' Now call the main() function to get to the actual training step '''
    main()
                
