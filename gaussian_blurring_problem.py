# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:29:58 2023

@author: uqalim8
"""
import torch, utils
from debluring import Pblurring
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from constants import cCUDA, cTYPE
from solvers import (ConjugateGradient,
                     ConjugateResidual,
                     MinimalResidual)

FOLDER_NAME = "gaussian_results_5"
CG, CR, MR = None, None, None
SOLVERS = ["CG"]
TOL = 1e-5
MAXIT = 300
#PRINT = False

def normalize(x):
    x = x - torch.min(x)
    return x/torch.max(x)

def Ax(A, x, d):
    x = x.reshape(d, d)
    x = A(x)
    return x.reshape(-1)

if __name__ == "__main__":
    
    pic = '20220607225846.jpg'
    img = Image.open('test_images/' + pic)
    
    data = np.asarray(img)
    red = torch.tensor(data[:,:,0], dtype = cTYPE, device = cCUDA)
    green = torch.tensor(data[:,:,1], dtype = cTYPE, device = cCUDA)
    blue = torch.tensor(data[:,:,2], dtype = cTYPE, device = cCUDA)
    
    d, d = red.shape
    myA = Pblurring(d, bandw=101, sigma=9, dtype = cTYPE, device = cCUDA) 
    
    A = myA.kron_() 
    A_r = A(red)
    A_g = A(green)
    A_b = A(blue)
    
    # Add noise
    noise_r = torch.rand(d,d, dtype = cTYPE, device = cCUDA) * 1
    noise_g = torch.rand(d,d, dtype = cTYPE, device = cCUDA) * 1
    noise_b = torch.rand(d,d, dtype = cTYPE, device = cCUDA) * 1
    B_r = A_r + noise_r
    B_g = A_g + noise_g
    B_b = A_b + noise_b
    
    G = lambda x : Ax(A, x, d) 
    if "CG" in SOLVERS:
        print(10 * "=" + " Conjugate Gradient (red)" + 10 * "=")
        CGr = ConjugateGradient(G, B_r.reshape(-1), maxit = MAXIT, tol = TOL)
        CGr.solve()
        # utils.saveRecords(FOLDER_NAME, "CGr", CGr.stat)
        # r = torch.norm(CGr.xk - red.reshape(-1)) / torch.norm(red.reshape(-1))
        # del CGr
        
        # print(10 * "=" + " Conjugate Gradient (green)" + 10 * "=")
        # CGg = ConjugateGradient(G, B_g.reshape(-1), maxit = MAXIT, tol = TOL)
        # CGg.solve()
        # utils.saveRecords(FOLDER_NAME, "CGg", CGg.stat)
        # g = torch.norm(CGg.xk - green.reshape(-1)) / torch.norm(green.reshape(-1))
        # del CGg
 
        # print(10 * "=" + " Conjugate Gradient (blue)" + 10 * "=")
        # CGb = ConjugateGradient(G, B_b.reshape(-1), maxit = MAXIT, tol = TOL)
        # CGb.solve()
        # utils.saveRecords(FOLDER_NAME, "CGb", CGb.stat)
        # b = torch.norm(CGb.xk - blue.reshape(-1)) / torch.norm(blue.reshape(-1))
        # del CGb
                
    if "MR" in SOLVERS:
        print(10 * "=" + " Minimal Residual (red)" + 10 * "=")
        MRr = MinimalResidual(G, B_r.reshape(-1), maxit = MAXIT, tol = TOL)
        MRr.solve()
        utils.saveRecords(FOLDER_NAME, "MRr", MRr.stat)
        del MRr
        
        print(10 * "=" + " Minimal Residual (green)" + 10 * "=")
        MRg = MinimalResidual(G, B_g.reshape(-1), maxit = MAXIT, tol = TOL)
        MRg.solve()
        utils.saveRecords(FOLDER_NAME, "MRg", MRg.stat)
        del MRg
        
        print(10 * "=" + " Minimal Residual (blue)" + 10 * "=")
        MRb = MinimalResidual(G, B_b.reshape(-1), maxit = MAXIT, tol = TOL)
        MRb.solve()
        utils.saveRecords(FOLDER_NAME, "MRb", MRb.stat)
        del MRb
        
    if "CR" in SOLVERS:
        print(10 * "=" + " Conjugate Residual (red)" + 10 * "=")
        CRr = ConjugateResidual(G, B_r.reshape(-1), maxit = MAXIT, tol = TOL)
        CRr.solve()
        utils.saveRecords(FOLDER_NAME, "CRr", CRr.stat)
        r = torch.norm(CRr.xk - red.reshape(-1)) / torch.norm(red.reshape(-1))
        del CRr
        
        print(10 * "=" + " Conjugate Residual (green)" + 10 * "=")
        CRg = ConjugateResidual(G, B_g.reshape(-1), maxit = MAXIT, tol = TOL)
        CRg.solve()
        utils.saveRecords(FOLDER_NAME, "CRg", CRg.stat)
        g = torch.norm(CRg.xk - green.reshape(-1)) / torch.norm(green.reshape(-1))
        del CRg
        
        print(10 * "=" + " Conjugate Residual (blue)" + 10 * "=")
        CRb = ConjugateResidual(G, B_b.reshape(-1), maxit = MAXIT, tol = TOL)
        CRb.solve()
        utils.saveRecords(FOLDER_NAME, "CRb", CRb.stat)
        b = torch.norm(CRb.xk - blue.reshape(-1)) / torch.norm(blue.reshape(-1))
        del CRb
        
    print(r + g + b / 3)