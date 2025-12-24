# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:08:06 2023

@author: uqalim8
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib, sys
import torch, math


from scipy.sparse.linalg import lsqr
import numpy as np

sys.path.append("..")

from solvers import CGLS, LSQR, ConjugateGradient, ConjugateResidual, MinimalResidual
from utils import makeFolder


FONT = {'size':25, 'weight':'bold'} 
matplotlib.rc('font', **FONT)
    
N = 500
RANGE = [10, 499, 500]
TERMINATE = RANGE
TOL = 0
REO = False
VERBOSE = False
FOLDER = "./bigcg"

STAT = "|xk-x*|/|x*|" #"|xk-x*|/|x*|" , "|rk|/|b|" , "|Ark|/|Ab|"

makeFolder(FOLDER) 

for i in RANGE:
    torch.manual_seed(2026)
    b = torch.rand(N, dtype = torch.float64)

    ratio = float(torch.norm(b[i:]) / torch.norm(b))
    for j in range(0, 1):
        D = torch.abs(torch.rand(N, dtype = torch.float64))
        #D[:int(i * 0.1)] = -D[:int(i * 0.1)]
        file_name = FOLDER + f"/{i}.png"
        # if j:
        #     D = torch.randn(N, dtype = torch.float64)
        #     file_name = FOLDER + f"/CG{i}in.png"
        # else:
        #     D = torch.rand(N, dtype = torch.float64)
        #     file_name = FOLDER + f"/CG{i}.png"
        
        xdagger = torch.zeros_like(b)
        D[i:] = 0
        
        random = torch.multinomial(torch.tensor(range(i), dtype = torch.float64), 5)# math.ceil(i * 0.1))
        D[random] = D[random] * -0.1
        
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
        
        #plt.figure(figsize=(11, 7))
        fig, ax = plt.subplots(2, 3, figsize=(30, 15))
        
        ax[1, 0].semilogy(range(0, i + 1), SOLVER_CG["|rk|/|b|"][:i + 1], alpha = 0.5, color='#0072B2', linestyle = "--", linewidth = 5, label = "CG(alg 1)")
        ax[1, 0].semilogy(range(0, i + 1), SOLVER_CG["|rk|/|b|"][:i + 1], alpha = 0.5, color='#E69F00', linestyle = "-", linewidth = 3, label = "CG(alg 3)")
        ax[1, 0].legend()

        ax[0, 0].semilogy(range(0, i + 1), SOLVER_CR["|rk|/|b|"][:i + 1], alpha = 0.5, color='#0072B2', linestyle = "--", linewidth = 5, label = "CR(alg 2)")
        ax[0, 0].semilogy(range(0, i + 1), SOLVER_CR["|rk|/|b|"][:i + 1], alpha = 0.5, color='#E69F00', linestyle = "-", linewidth = 3, label = "CR(alg 4)")
        ax[0, 0].semilogy(range(0, i + 1), SOLVER_MR["|rk|/|b|"][:i + 1], alpha = 0.5, color='#009E73', linestyle = ":", linewidth = 7, label = "MINRES")
        ax[0, 0].semilogy(range(0, i + 1), SOLVER_CGLS["|rk|/|b|"][:i + 1], alpha = 0.5, color='#D55E00', linestyle = "-", linewidth = 3, label = "CGLS")
        ax[0, 0].semilogy(range(0, i + 1), SOLVER_LSQR["|rk|/|b|"][:i + 1], alpha = 0.5, color='#CC79A7', linestyle = "--", linewidth = 5, label = "LSQR")
        ax[0, 0].legend()
        ax[0, 0].set_title("$\|\mathbf{r}_k\|/\|\mathbf{b}\|$")


        ax[1, 1].semilogy(range(0, i + 1), SOLVER_CG["|Ark|/|Ab|"][:i + 1], alpha = 0.5, color='#0072B2', linestyle = "--", linewidth = 5, label = "CG(alg 1)")
        ax[1, 1].semilogy(range(0, i + 1), SOLVER_CG["|Ark|/|Ab|"][:i + 1], alpha = 0.5, color='#E69F00', linestyle = "-", linewidth = 3, label = "CG(alg 3)")
        ax[1, 1].legend()

        ax[0, 1].semilogy(range(0, i + 1), SOLVER_CR["|Ark|/|Ab|"][:i + 1], alpha = 0.5, color='#0072B2', linestyle = "--", linewidth = 5, label = "CR(alg 2)")
        ax[0, 1].semilogy(range(0, i + 1), SOLVER_CR["|Ark|/|Ab|"][:i + 1], alpha = 0.5, color='#E69F00', linestyle = "-", linewidth = 3, label = "CR(alg 4)")
        ax[0, 1].semilogy(range(0, i + 1), SOLVER_MR["|Ark|/|Ab|"][:i + 1], alpha = 0.5, color='#009E73', linestyle = ":", linewidth = 7, label = "MINRES")
        ax[0, 1].semilogy(range(0, i + 1), SOLVER_CGLS["|Ark|/|Ab|"][:i + 1], alpha = 0.5, color='#D55E00', linestyle = "-", linewidth = 3, label = "CGLS")
        ax[0, 1].semilogy(range(0, i + 1), SOLVER_LSQR["|Ark|/|Ab|"][:i + 1], alpha = 0.5, color='#CC79A7', linestyle = "--", linewidth = 5, label = "LSQR")
        ax[0, 1].legend()
        ax[0, 1].set_title("$\|\mathbf{Ar}_k|/\|\mathbf{Ab}\|$")


        ax[1, 2].semilogy(range(0, i + 1), SOLVER_CG["|xk-x*|/|x*|"][:i + 1], alpha = 0.5, color='#0072B2', linestyle = "--", linewidth = 5, label = "CG(alg 1)")
        ax[1, 2].semilogy(range(0, i + 1), SOLVER_CG["|xk-x*|/|x*|"][:i + 1], alpha = 0.5, color='#E69F00', linestyle = "-", linewidth = 3, label = "CG(alg 3)")
        ax[1, 2].legend()

        ax[0, 2].semilogy(range(0, i + 1), SOLVER_CR["|xk-x*|/|x*|"][:i + 1], alpha = 0.5, color='#0072B2', linestyle = "--", linewidth = 5, label = "CR(alg 2)")
        ax[0, 2].semilogy(range(0, i + 1), SOLVER_CR["|xk-x*|/|x*|"][:i + 1], alpha = 0.5, color='#E69F00', linestyle = "-", linewidth = 3, label = "CR(alg 4)")
        ax[0, 2].semilogy(range(0, i + 1), SOLVER_MR["|xk-x*|/|x*|"][:i + 1], alpha = 0.5, color='#009E73', linestyle = ":", linewidth = 7, label = "MINRES")
        ax[0, 2].semilogy(range(0, i + 1), SOLVER_CGLS["|xk-x*|/|x*|"][:i + 1], alpha = 0.5, color='#D55E00', linestyle = "-", linewidth = 3, label = "CGLS")
        ax[0, 2].semilogy(range(0, i + 1), SOLVER_LSQR["|xk-x*|/|x*|"][:i + 1], alpha = 0.5, color='#CC79A7', linestyle = "--", linewidth = 5, label = "LSQR")
        ax[0, 2].legend()
        ax[0, 2].set_title("$\|\mathbf{x}_k-\mathbf{x}^+\|/\|\mathbf{x}^+\|$")

        
        print(f"\nnullspace {i}")
        print("CG(alg 1): ", torch.norm(torch.tensor(SOLVER_CG["xk"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        print("CG(alg 3): ", torch.norm(torch.tensor(SOLVER_CG["xk_lifted"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        print("CR(alg 2): ", torch.norm(torch.tensor(SOLVER_CR["xk"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        print("CR(alg 4): ", torch.norm(torch.tensor(SOLVER_CR["xk_lifted"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        print("MINRES: ", torch.norm(torch.tensor(SOLVER_MR["xk_lifted"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        print("CGLS: ", torch.norm(torch.tensor(SOLVER_CGLS["xk"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        print("LSQR: ", torch.norm(torch.tensor(SOLVER_LSQR["xk"], dtype = torch.float64) - xdagger) / torch.norm(xdagger))
        
        fig.suptitle(fr"dim(Null(A)):{N - i};" + " nullspace ratio:{:.3f}".format(ratio), fontsize = 50)      
        #plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
        #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        #plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #plt.xticks(size = 20)
        #plt.yticks(size = 20)
        #plt.xlabel("iteration k", fontsize = 20)
        #plt.ylabel("$\|\mathbf{Ar}_k\| / \|\mathbf{Ab}\|$", fontsize = 20)
        #plt.ylabel("$\|\mathbf{x}_k - \mathbf{x}^+\| / \|\mathbf{x}^+\|$", fontsize = 20)
        #plt.ylabel("$\|\mathbf{r}_k\| / \|\mathbf{b}\|$", fontsize = 20)
        plt.legend()
        plt.savefig(file_name)
        plt.close()