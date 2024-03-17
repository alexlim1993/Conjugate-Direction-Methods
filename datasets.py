# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:20:15 2022

@author: uqalim8
"""

import torch, sklearn, ucimlrepo
import sklearn.datasets as skdatasets
import torchvision.datasets as datasets
from constants import cTYPE, cCUDA

TEXT = "{:<20} : {:>20}"

def prepareData(folder_path, one_hot, dataset):
    
    if dataset == "MNISTb":
        print(TEXT.format("Dataset", dataset))
        return MNIST(folder_path, one_hot, 2)
    
    if dataset == "MNIST":
        print(TEXT.format("Dataset", dataset))
        return MNIST(folder_path, one_hot, 10)
    
    if dataset == "CIFAR10b":
        print(TEXT.format("Dataset", dataset))
        return CIFAR10(folder_path, one_hot, 2)
    
    if dataset == "CIFAR10":
        print(TEXT.format("Dataset", dataset))
        return CIFAR10(folder_path, one_hot, 10)
    
    if dataset == "MNISTs":
        print(TEXT.format("Dataset", dataset))
        return MNISTs(folder_path, one_hot, 10)
    
    if dataset == "MNISTsb":
        print(TEXT.format("Dataset", dataset))
        return MNISTs(folder_path, one_hot, 2)
    
    if dataset == "Covtype":
        print(TEXT.format("Dataset", dataset))
        return Covtype(one_hot, 7)
    
    if dataset == "Covtypeb":
        print(TEXT.format("Dataset", dataset))
        return Covtype(one_hot, 2)
    
    if dataset == "WineQuality":
        print(TEXT.format("Dataset", dataset))
        return wineQuality()

def wineQuality():
    # https://archive.ics.uci.edu/dataset/186/wine+quality
    wine_quality = ucimlrepo.fetch_ucirepo(id = 186)
    X = wine_quality.data.features.to_numpy()
    Y = torch.tensor(X[:, -1], dtype = cTYPE, device = cCUDA)
    X = torch.tensor(X[:, :-1], dtype = cTYPE, device = cCUDA)
    return X, Y, None, None

def MNISTs(folder_path, one_hot, classes):
    """
    MNIST small size (8 by 8 pixels)
    """
    
    trainX, trainY = skdatasets.load_digits(return_X_y = True)
    trainX = torch.tensor(trainX, dtype = cTYPE, device = cCUDA)
    trainY = torch.tensor(trainY, dtype = torch.long, device = cCUDA) % classes
    
    if one_hot:
        trainY = torch.nn.functional.one_hot(trainY.long(), classes).to(cTYPE)
        
    return trainX, trainY, None, None
    
def MNIST(folder_path, one_hot, classes):
    
    train_set = datasets.MNIST(folder_path, train = True, download = True)
    test_set = datasets.MNIST(folder_path, train = False, download = True)

    train_set.data = train_set.data.reshape(train_set.data.shape[0], -1).to(cTYPE).to(cCUDA)
    test_set.data = test_set.data.reshape(test_set.data.shape[0], -1).to(cTYPE).to(cCUDA)

    train_set.targets = train_set.targets % classes
    test_set.targets =test_set.targets % classes

    if one_hot:
        train_set.targets = torch.nn.functional.one_hot(train_set.targets.long(), classes)
        test_set.targets = torch.nn.functional.one_hot(test_set.targets.long(), classes)
            
    return train_set.data, train_set.targets.to(cTYPE), \
        test_set.data, test_set.targets.to(cTYPE)
        
        
def CIFAR10(folder_path, one_hot, classes):

    train_set = datasets.CIFAR10(folder_path, train = True, download = True)
    test_set = datasets.CIFAR10(folder_path, train = False, download = True)
    
    X = torch.cat([torch.tensor(train_set.data.reshape(train_set.data.shape[0], -1), 
                                dtype = cTYPE, device = cCUDA),
                   torch.tensor(test_set.data.reshape(test_set.data.shape[0], -1), 
                                dtype = cTYPE, device = cCUDA)], dim = 0) 
    
    Y = torch.cat([torch.tensor(train_set.targets, device = cCUDA), 
                   torch.tensor(test_set.targets, device = cCUDA)], dim = 0) % classes
    
    del train_set, test_set
    if one_hot:
        Y = torch.nn.functional.one_hot(Y.long(), classes)
    
    print(TEXT.format("data size", str(tuple(X.shape))))
    return X, Y.to(cTYPE), None, None

def Covtype(one_hot, classes = 7):
    X, Y = sklearn.datasets.fetch_covtype(return_X_y = True)
    X = torch.tensor(X, dtype = cTYPE)
    Y = (torch.tensor(Y, dtype = torch.long) - 1) % classes
    
    if one_hot:
        Y = torch.nn.functional.one_hot(Y, 7).to(cTYPE)
        
    print(TEXT.format("Data size", str(tuple(X.shape)))) 
    return X[:500], Y[:500], None, None