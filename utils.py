#!/usr/bin/env python
# coding: utf-8

# In[1]:

import apex.amp as amp
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, imsave
from sklearn.metrics import confusion_matrix, recall_score, classification_report, accuracy_score
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from semi.MCMC_loss import DBI, margin

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

torch.cuda.set_device(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")


# In[2]:
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
#cifar10_mean = (0.0,0.0,0.0)
#cifar10_std  = (1.0,1.0,1.0)


# In[2]:
def read_SVHN():
    batch_size = 256
    transform1 = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                transforms.Normalize(cifar10_mean, cifar10_std)])
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(cifar10_mean, cifar10_std)])

    trainset = torchvision.datasets.SVHN(root='./data', split ='train',
                                            download=True, transform=transform1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)

    testset = torchvision.datasets.SVHN(root='./data', split ='test',
                                           download=True, transform=transform1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)
    return trainset, trainloader, testset, testloader

def read_cifar10():
    batch_size = 128
    transform1 =transforms.Compose([transforms.RandomCrop(32,padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(cifar10_mean, cifar10_std),
                                    ])
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(cifar10_mean, cifar10_std),])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    devset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    devloader = torch.utils.data.DataLoader(devset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainset, trainloader, testset, testloader, devset, devloader



mu = torch.tensor(cifar10_mean).view(3,1,1).float().cuda()
std = torch.tensor(cifar10_std).view(3,1,1).float().cuda()

#epsilon = (args.epsilon / 255.) / std
#alpha = (args.alpha / 255.) / std
#pgd_alpha = (2 / 255.) / std

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    pgd_dist = 0 
    model.eval()
    for i, (X, y) in tqdm(enumerate(test_loader)):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            a = pgd_delta.reshape(pgd_delta.shape[0],-1).cpu()
            pgd_dist += sum(np.linalg.norm(a,ord=2,axis=1))
    return pgd_loss/n, pgd_acc/n, pgd_dist/n

def evaluate_pgd1(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (8 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    pgd_dist = 0 
    model.eval()
    for i, (X, y) in tqdm(enumerate(test_loader)):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            a = pgd_delta.reshape(pgd_delta.shape[0],-1).cpu()
            pgd_dist += sum(np.linalg.norm(a,ord=2,axis=1))
    return pgd_loss/n, pgd_acc/n, pgd_dist/n

def evaluate_pgd2(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (0.003) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    pgd_dist = 0 
    model.eval()
    for i, (X, y) in tqdm(enumerate(test_loader)):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            a = pgd_delta.reshape(pgd_delta.shape[0],-1).cpu()
            pgd_dist += sum(np.linalg.norm(a,ord=2,axis=1))
    return pgd_loss/n, pgd_acc/n, pgd_dist/n
def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in tqdm(enumerate(test_loader)):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n

