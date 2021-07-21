# Two-Steps Clustering Adversarial Defense
author : Yi-Wei Chiu

Please cite `my paper(working now)` if you use this code or method.

Yi-Wei Chiu *Two-Steps Clustering Adversarial Defense* Unpublished doctoral dissertation. National Yang Ming Chiao Tung University, Taiwan. (2021)

## State-of-the-art performance
The first performance on the CIFAR-10 dataset.
The second performance on the SVHN dataset.

Natural accuracy: No adversarial attack

FGSM, PGD-20 and CW$_\infty$ robustness : Adversarial attack 

![](https://i.imgur.com/h4h92dr.png)
![](https://i.imgur.com/SWesmwX.png)

## t-SNE performance
Watch the t-SNE difference in ADV and TSCAD with natural examples and adversarial examples

![](https://i.imgur.com/2A2Bt3O.png)

## Training 
compile the TSCAD.ipynb

## Requirements
main requirements:
* torch == 1.5.0
* torchvision == 0.6.0
* tqdm == 4.46.0

## Try on your dataset
write your reading data in utils.py
And use the reading function in the TSCAD.ipynb
You can also modify more parameter settings such as iterations per epoch, learning rate,..., etc. from the argparser in TSCAD.ipynb
