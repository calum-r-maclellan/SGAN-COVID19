# SGAN-COVID19
PyTorch implementation of a Semi-supervised Generative Adversarial Network (SGAN) for performing multi-class classification of COVID-19 from chest X-ray imaging.

## Whats included 
A brief overview of the idea behind this work, the way it was implemented, and corresponding results compared to supervised equivalents. A Jupyter Notebook has been provided (semi_GAN.ipynb) in a tutorial-styled layout, which is set up to run on Google Colab for building the proposed model and reproducing the results (TODO: add model weights for discriminator - Github fussy about the .pth.tar file size!). See instructions in the notebook itself for how to get this working (heads up: if it doesnt load, refresh your browser and it should fix itself). 

## Idea
- supervised learning models painfully inefficient: require significant database of labelled samples to discriminate between classes.
- doesn't represent how we learn: some direct instruction at beginning (labelled data), but majority of learning gained through experience and observation on our own (unlabelled data).
- same applies to radiologist training: direct instruction on a few samples at beginning, with more knowledge/experience accumulated through years of practice. 
- design semi-supervised approach to (loosely) mimic radiologist strategy of detecting COVID-19 from chest X-rays: accumulate extensive knowledge of non-COVID before learning to understand COVID-19 features.
- develop experience of non-COVID patients using large database of unlabelled data, update knowledge with small labelled database of COVID and non-COVID data. 

# Experiments
## Dataset
COVIDx_v3. See https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md

## Training workflow 
![](https://github.com/calum-r-maclellan/SGAN-COVID19/blob/main/pics/sgan_workflow.png)
Schematic of the SGAN training workflow, including the training computations and objective functions.

## Classification results
![](https://github.com/calum-r-maclellan/SGAN-COVID19/blob/main/pics/class_perf.png)

## GradCAM++ results 
![](https://github.com/calum-r-maclellan/SGAN-COVID19/blob/main/pics/gradcam++.png)

# Conclusions and future work
- 
- 
