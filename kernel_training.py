# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:51:51 2023

@author: uqalim8
"""

import torch, sklearn, datasets
from draw import plotGraphs
from constants import cTYPE, cCUDA
import solvers, utils

SEED = 1234
DATASET = "WineQuality"
KERNEL = "RBK"
SAMPLES = None
SOLVER = ["CR"]
CG, CR, MR = None, None, None
LOW_RANK = 1000
MAXIT = 2000
SIGMA = 0 #1e-6
TOL = 1e-7
FOLDER = "./kernelResults"
scale_L = 1e-4

def RBK(X, Y = None, L = 1):
    assert L > 0
    if Y is None:
        X_norm_sq = torch.norm(X, dim = 1) ** 2
        K = X_norm_sq[:, None] + X_norm_sq[None, :] - 2 * torch.inner(X, X)
        return torch.exp(-L * K)
    X_norm_sq = torch.norm(X, dim = 1) ** 2
    Y_norm_sq = torch.norm(Y, dim = 1) ** 2
    K = X_norm_sq[:, None] + Y_norm_sq[None, :] - 2 * torch.inner(X, Y)
    return torch.exp(-L * K)

def LaplaceK(X, L = 10):
    X_norm_sq = torch.norm(X, dim = 1) ** 2
    K = X_norm_sq[:, None] + X_norm_sq[None, :] - 2 * torch.inner(X, X)
    K[K < 1e-14] = 0
    return torch.exp(-L * torch.sqrt(K))

def Linear(X, L = 0.1, M = 0, C = 0):
    return M ** 2 + (L ** 2) * torch.inner(X - C, X - C)

# class Neystrom:
    
#     def __init__(self, X, rank, kernel_func, full_matrix = False):
#         self._X = X
#         self._m = rank
#         self._K = kernel_func
        
#     def _process(self, X, m, kernel_func):
#         Xm = X[:m, :]
#         Km = kernel_func(Xm)
#         Kminv = torch.linalg.inv(Km)
#         K21 = X @ Xm.T                               # check if K21 is n by m
        
def lowRank_eigenDecomp(K, rank, sigma, full_matrix = False):
    eigval, eigvecs = torch.linalg.eigh(K)
    perm = torch.randperm(eigval.shape[0])
    eigval = eigval[perm]
    eigval = eigval[:rank]
    eigvecs = eigvecs[perm]
    eigvecs = eigvecs[:, :rank]
    
    if full_matrix:
        return eigvecs @ torch.diag(eigval) @ eigvecs.T + sigma * torch.eye(eigvecs.shape[0])
    
    return lambda v : eigvecs @ (torch.diag(eigval) @ (eigvecs.T @ v)) + sigma * v

def Avec(A, x):
    if callable(A):
        A(x)
    return torch.mv(A, x)

def prediction(xk, Kp, testY):
    return torch.norm(testY - xk.reshape(1, -1) @ Kp) / testY.shape[0]

if __name__ == "__main__":
    torch.manual_seed(SEED)
    
    if DATASET == "SYN":
        X, Y = sklearn.datasets.make_regression(SAMPLES, 50)
        X = torch.tensor(X, dtype = cTYPE, device = cCUDA)
        Y = torch.tensor(Y, dtype = cTYPE, device = cCUDA)
        
    elif DATASET == "WineQuality":
        X, Y, _, _ = datasets.prepareData(FOLDER, None, DATASET)
        
        # meanX, stdX = torch.std_mean(X, dim = 0)
        # X = (X - meanX) / stdX
        # meanY, stdY = torch.std_mean(Y)
        # Y = (Y - meanY) / stdY
        
        perm = torch.randperm(X.shape[0])
        ceil = int(X.shape[0] * 0.8)
        X, Y = X[perm], Y[perm]
        trainX, trainY = X[:ceil], Y[:ceil]
        print("Training samples:", trainY.shape[0])
        testX, testY = X[ceil:], Y[ceil:]
        print("Test samples:", testY.shape[0])
        
    
    if KERNEL == "RBK":
        K = RBK(trainX, L = scale_L)
        pred = lambda x : prediction(x, RBK(trainX, testX, L = scale_L), testY)
        
    elif KERNEL == "LINEAR":
        K = Linear(trainX)
    elif KERNEL == "LK":
        K = LaplaceK(trainX)

    # if LOW_RANK != trainX.shape[0]:
    #     print("LOW RANK:", LOW_RANK)
    #     K = lowRank_eigenDecomp(K, LOW_RANK, SIGMA)
    
    if "CG" in SOLVER:
        CG = solvers.ConjugateGradient(K, trainY, maxit = MAXIT, tol = TOL)
        CG.solve(pred)
        utils.saveRecords(FOLDER, "CG", CG.stat)
    
    if "CR" in SOLVER:
        CR = solvers.ConjugateResidual(K, trainY, maxit = MAXIT, tol = TOL)
        CR.solve(pred)
        utils.saveRecords(FOLDER, "CR", CR.stat)
        
    if "MR" in SOLVER:
        MR = solvers.MinimalResidual(K, trainY, maxit = MAXIT, tol = TOL)
        MR.solve(pred)
        utils.saveRecords(FOLDER, "MR", MR.stat)

    #plotGraphs(CG, CR, MR, FOLDER)
    