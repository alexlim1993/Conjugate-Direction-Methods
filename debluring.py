# -*- coding: utf-8 -*-
"""
Created on Sat May  7 09:14:08 2022

@author: Yang

Image debluring experiment for paper:
Obtaining Pseudo-inverse Solutions With MINRES
https://arxiv.org/abs/2309.17096
Authors: Yang Liu, Andre Milzarek, Fred Roosta 
"""

import torch
import numpy as np
import math
import os
from scipy.linalg import toeplitz
from scipy.sparse import kron

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Pblurring(object):
    # Construct Gaussian bluring matrix class
    def __init__(self, n, bandw, sigma, dtype=torch.float64, device="cpu"):
        self.dv = device
        self.dt = dtype
        bandw = min(bandw,n)
        tmp = torch.exp(-torch.arange(0,bandw)**2/sigma/sigma/2)
        z = np.zeros(n)
        z[:bandw] = tmp
        npT1 = toeplitz(z)/np.sqrt(2*math.pi)/sigma
        T1 = torch.tensor(npT1, dtype=self.dt, device=self.dv)
        self.Toeplitz = T1
        self.rank = n 
        self.U, self.s, self.V = torch.svd(T1) # for truncated svd methods
    
    def range_(self, index):
        img = lambda X: self.U[:,index] @ X @ self.U[:,index].T
        return img
    
    def kron_(self):
        # Function handle of applying Gaussian bluring matrix to an image X
        T = self.Toeplitz
        img = lambda X: T @ X @ T
        return img
    
    def A_(self): 
        # represent Gaussian debluring matrix as a Kronecker produck of
        # Two Toeplitz matrix
        return kron(self.Toeplitz, self.Toeplitz).todense()
    
    def kron_G_vec(self, C, re=False): 
        # Function handle of using preconditioner C
        if re:
            img = lambda tX: C @ tX @ C.T # S @ tx
        else:
            img = lambda B: C.T @ B @ C # S.T @ b
        return img
    
    def kron_AG_vec(self, C, re=False):
        # Function handle of using range-preseved preconditioner C
        T = C.T @ self.Toeplitz
        if re:
            img = lambda tX: T.T @ tX @ T # S @ tx
        else:
            img = lambda B: T @ B @ T.T # S.T @ b
        return img
    
    def kron_G(self, C):
        # Explicit methods using preconditioner C
        T = self.Toeplitz
        T2 = C.T @ (T @ C)
        img = lambda X: T2 @ X @ T2
        return img
    
    def kron_AG(self, C):
        # Explicit methods using range-preseved preconditioner C
        T = self.Toeplitz
        T2 = C.T @ (T @ (T @ (T @ C)))
        img = lambda X: T2 @ X @ T2
        return img