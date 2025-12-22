# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:08:06 2023

@author: uqalim8
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib, sys
import torch


from scipy.sparse.linalg import lsqr
import numpy as np

sys.path.append("..")

from solvers import CGLS, LSQR, ConjugateGradient, ConjugateResidual, MinimalResidual
from utils import makeFolder


FONT = {'size':14, 'weight':'bold'} 
matplotlib.rc('font', **FONT)
    
N = 300
RANGE = [50, 299, 300]
TERMINATE = RANGE
TOL = 0
REO = False
VERBOSE = False
FOLDER = "./bigcg"

STAT = "|rk|/|b|" #"|xk-x*|/|x*|"

makeFolder(FOLDER) 

for i in RANGE:
    torch.manual_seed(2025)
    b = torch.randn(N, dtype = torch.float64)

    ratio = float(torch.norm(b[i:]) / torch.norm(b))
    for j in range(0, 1):
        D = torch.rand(N, dtype = torch.float64)
        #D[:int(i * 0.1)] = -D[:int(i * 0.1)]
        file_name = FOLDER + f"/relr_noCG{i}.png"
        # if j:
        #     D = torch.randn(N, dtype = torch.float64)
        #     file_name = FOLDER + f"/CG{i}in.png"
        # else:
        #     D = torch.rand(N, dtype = torch.float64)
        #     file_name = FOLDER + f"/CG{i}.png"
        
        xdagger = torch.zeros_like(b)
        D[i:] = 0
        xdagger[:i] = b[:i] / D[:i]
        H = torch.diag(D)
        
        SOLVER_CG = ConjugateGradient(H, b, maxit = i, tol = TOL, reO = REO, prinT = VERBOSE)
        SOLVER_CG.solve(lambda x : torch.norm(x - xdagger) / torch.norm(xdagger))
        
        SOLVER_CR = ConjugateResidual(H, b, maxit = i, tol = TOL, reO = REO, prinT = VERBOSE)
        SOLVER_CR.solve(lambda x : torch.norm(x - xdagger) / torch.norm(xdagger))
        
        SOLVER_MR = MinimalResidual(H, b, maxit = i, tol = TOL, reO = REO, prinT = VERBOSE)
        SOLVER_MR.solve(lambda x : torch.norm(x - xdagger) / torch.norm(xdagger))
        
        SOLVER_CGLS = CGLS(H, b, maxit = i, tol = TOL, reO = REO, prinT = VERBOSE)
        SOLVER_CGLS.solve(lambda x : torch.norm(x - xdagger) / torch.norm(xdagger))
        
        SOLVER_LSQR = LSQR(H, b, maxit = i, tol = TOL, reO = REO, prinT = VERBOSE)
        SOLVER_LSQR.solve(lambda x : torch.norm(x - xdagger) / torch.norm(xdagger))
        
        #lsqr(np.array(H), np.array(b), show = True, iter_lim = 100)
        
        SOLVER_CG = SOLVER_CG.stat
        SOLVER_CR = SOLVER_CR.stat
        SOLVER_MR = SOLVER_MR.stat
        SOLVER_CGLS = SOLVER_CGLS.stat
        SOLVER_LSQR = SOLVER_LSQR.stat
        
        plt.figure(figsize=(11, 7))
        #plt.semilogy(range(0, i + 1), SOLVER_CG[STAT][:i + 1], alpha = 0.5, color='#0072B2', linestyle = "--", linewidth = 4, label = "CG(alg 1)")
        #plt.semilogy(range(0, i + 1), SOLVER_CG[STAT][:i + 1], alpha = 0.5, color='#0072B2', linestyle = "-", linewidth = 2, label = "CG(alg 3)")
        plt.semilogy(range(0, i + 1), SOLVER_CR[STAT][:i + 1], alpha = 0.5, color='#E69F00', linestyle = "--", linewidth = 4, label = "CR(alg 2)")
        plt.semilogy(range(0, i + 1), SOLVER_CR[STAT][:i + 1], alpha = 0.5, color='#E69F00', linestyle = "-", linewidth = 2, label = "CR(alg 4)")
        plt.semilogy(range(0, i + 1), SOLVER_MR[STAT][:i + 1], alpha = 0.5, color='#009E73', linestyle = ":", linewidth = 2, label = "MINRES")
        plt.semilogy(range(0, i + 1), SOLVER_CGLS[STAT][:i + 1], alpha = 0.5, color='#D55E00', linestyle = "-", linewidth = 2, label = "CGLS")
        plt.semilogy(range(0, i + 1), SOLVER_LSQR[STAT][:i + 1], alpha = 0.5, color='#CC79A7', linestyle = "--", linewidth = 4, label = "LSQR")
        
        print(f"\nnullspace {i}")
        print("CG(alg 1): ", torch.norm(torch.tensor(SOLVER_CG["xk"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        print("CG(alg 3): ", torch.norm(torch.tensor(SOLVER_CG["xk_lifted"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        print("CR(alg 2): ", torch.norm(torch.tensor(SOLVER_CR["xk"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        print("CR(alg 4): ", torch.norm(torch.tensor(SOLVER_CR["xk_lifted"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        print("MINRES: ", torch.norm(torch.tensor(SOLVER_MR["xk_lifted"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        print("CGLS: ", torch.norm(torch.tensor(SOLVER_CGLS["xk"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        print("LSQR: ", torch.norm(torch.tensor(SOLVER_LSQR["xk"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        
        plt.title(fr"dim(Null(A)):{N - i};" + " nullspace ratio:{:.2f}".format(ratio), fontsize = 25)      
        #plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        #plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xticks(size = 20)
        plt.yticks(size = 20)
        plt.xlabel("iteration k", fontsize = 20)
        #plt.ylabel("$\|\mathbf{x}_k - \mathbf{x}^+\| / \|\mathbf{x}^+\|$", fontsize = 20)
        plt.ylabel("$\|\mathbf{r}_k\| / \|\mathbf{b}\|$", fontsize = 20)
        plt.legend()
        plt.savefig(file_name)
        plt.close()