# SGAN-COVID19
PyTorch implementation of a Semi-supervised Generative Adversarial Network (SGAN) for performing multi-class classification of COVID-19 from chest X-ray imaging.

## Whats included 
A brief overview of the idea behind this work, the way it was implemented, and corresponding results compared to state-of-the-art supervised networks (see my DeepLearning_COVID19 repo). A Jupyter Notebook has been provided (semi_GAN.ipynb) in a tutorial-styled layout, which is set up to run on Google Colab for building the proposed model and reproducing the results (TODO: add model weights for discriminator - Github fussy about the .pth.tar file size!). See instructions in the notebook itself for how to get this working (heads up: if it doesnt load, refresh your browser and it should fix itself). 

## Idea
- supervised learning models painfully inefficient: require significant database of labelled samples in order to learn class-discriminative features.
- doesn't represent how we learn: some direct instruction at beginning (labelled data), but majority of learning gained through experience and observation on our own (unlabelled data).
- same applies to radiologist training: direct instruction on a few samples at beginning, with more knowledge/experience accumulated through years of practice. 
- design semi-supervised approach to (loosely) mimic radiologists strategy for learning to identify COVID-19 from chest X-rays: accumulate extensive knowledge of non-COVID before learning to understand the discriminative COVID-19 features.
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

# Conclusions
- despite seeing significantly less non-COVID labelled data, SGAN demonstrates strong similarity in diagnostic accuracy to the supervised equivalent.
- SGAN presents as a highly promising architecture for performing semi-supervised learning on this task.
- however it needs further investigation due to the potential fitting of confounding variables. 
- this work lays the foundations to build upon this model and devise novel methods to improve the classification performance, and enhance the ability to detect underlying covariates most strongly linked to COVID-19.
