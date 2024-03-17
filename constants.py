# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 16:18:46 2023

@author: uqalim8
"""

import torch

cCUDA = False
cTYPE = torch.float64
cZERO = 1e-12

if cCUDA:
    cCUDA = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    cCUDA = torch.device("cpu")

class const():
    pass

cMR = const()
cMR.alpha0 = 1
cMR.gradtol = 1e-6
cMR.maxite = 1e5
cMR.restol = 0.0001
cMR.inmaxite = 100
cMR.maxorcs = 1e7
cMR.lineMaxite = 100
cMR.lineBetaB = 1e-4
cMR.lineRho = 0.5
cMR.lineBetaFB = 1e-4

cCR = const()
cCR.alpha0 = 1
cCR.gradtol = 1e-6
cCR.maxite = 1e5
cCR.restol = 0.0001
cCR.inmaxite = 100
cCR.maxorcs = 1e7
cCR.lineMaxite = 100
cCR.lineBetaB = 1e-4
cCR.lineRho = 0.5
cCR.lineBetaFB = 1e-4


cCG = const()
cCG.alpha0 = 1
cCG.gradtol = 1e-6
cCG.maxite = 1e5
cCG.restol = 0.0001
cCG.inmaxite = 100
cCG.maxorcs = 1e7
cCG.lineMaxite = 100
cCG.lineBeta = 1e-4
cCG.lineRho = 0.5

cTR_STEI = const()
cTR_STEI.gradtol = 1e-6
cTR_STEI.maxite = 1e6
cTR_STEI.inmaxite = 1000
cTR_STEI.maxorcs = 1e6
cTR_STEI.restol = 0.001                          
cTR_STEI.deltaMax = 1e10
cTR_STEI.delta0 = 1e5
cTR_STEI.eta = 0.01
cTR_STEI.eta1 = 1/4
cTR_STEI.eta2 = 3/4
cTR_STEI.gamma1 = 1/4
cTR_STEI.gamma2 = 2

cL_BFGS = const()
cL_BFGS.alpha0 = 1
cL_BFGS.gradtol = 1e-9
cL_BFGS.m = 20
cL_BFGS.maxite = 1e6
cL_BFGS.maxorcs = 1e6
cL_BFGS.lineMaxite = 100
