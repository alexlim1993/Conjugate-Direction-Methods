# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:29:42 2023

@author: uqalim8
"""
import torch
from constants import cTYPE, cCUDA, cZERO

def CR(A, b, rtol = 1e-2, maxit = 100):
    
    xk = torch.zeros_like(b, dtype = cTYPE, device = cCUDA)
    pk, rk = b, b
    Ar = Ax(A, rk)
    Ap = Ar
    rAr = torch.dot(rk, Ar)
    pAAp = torch.dot(Ar, Ar)
    norm_Ab = torch.sqrt(pAAp)
    norm_b = torch.norm(b)
    k = 1
    while torch.norm(rk) / norm_b > rtol and torch.norm(Ar) / norm_Ab > cZERO and rAr > cZERO and k < maxit:
        
        alpha = rAr/pAAp
        xk = xk + alpha * pk
        rk = rk - alpha * Ap
        Ar = Ax(A, rk)
        rArp = torch.dot(rk, Ar)
        beta = rArp / rAr
        pk = rk + beta * pk
        Ap = Ar + beta * Ap
        rAr = rArp
        pAAp = torch.dot(Ap, Ap)
        k += 1
        
    if k == maxit:
        dtype = "MAX"
        return xk, k, dtype, torch.norm(b - Ax(A, xk) - rk), torch.abs(torch.dot(b, Ar)) / (norm_b * torch.norm(Ar))
    
    elif rAr < cZERO or torch.norm(Ar) / norm_Ab < cZERO:
        dtype = "NPC"
        if k == 1:
            return rk, k, dtype, 0, 0
        return rk, k, dtype, torch.norm(b - Ax(A, xk) - rk), torch.abs(torch.dot(b, Ar)) / (norm_b * torch.norm(Ar))
    
    dtype = "Sol"
    return xk, k, dtype, torch.norm(b - Ax(A, xk) - rk), torch.abs(torch.dot(b, Ar)) / (norm_b * torch.norm(Ar))
    
def Ax(A, x):
    if callable(A):
        return A(x)
    return torch.mv(A, x)