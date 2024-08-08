# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:29:42 2023

@author: uqalim8
"""
import torch

def CR(A, b, tol, maxit, reOrtho = 0):
    """
    Conjugate Residual method:
        - directly minimizing the residual ||b - Ax|| within the Krylov subspace
        - it works under inconsistent and indefinite system
        - restart may require if encountering rAr near 0 for indefinite systems
    
    A - a square symmetric matrix
    b - a target vector b
    tol - desire tolerence for relative residual and relative normal residual
    maxit - maximum iteration (it should be less than the dimension of the matrix)
    reOrtho - re-orthogonalization required. 
            reOrtho = -1 : it will store all previous Ap vectors
            reOrtho = 0 : no re-orthogonalization
            reOrtho = N : re-orthogonalization up to N number of Ap vectors 
    """
    xk = torch.zeros_like(b, dtype = torch.float64)
    pk, rk = b, b
    Ar = Ax(A, rk)
    Ap = Ar
    rAr = torch.dot(rk, Ar)
    pAAp = torch.dot(Ar, Ar)
    norm_Ab = torch.sqrt(pAAp)
    norm_Ark = norm_Ab
    
    if reOrtho:
        AP = Ap.reshape(-1, 1) / norm_Ab
    
    # statistics
    norm_b = torch.norm(b)
    norm_rk = norm_b
    records_rk, records_Ark = [norm_rk / norm_b], [norm_Ark / norm_Ab]
    k = 0
    while norm_rk / norm_b > tol and norm_Ark / norm_Ab > tol and k < maxit:
        alpha = rAr/pAAp
        xk = xk + alpha * pk
        rk = rk - alpha * Ap
        
        # re-orthogonalization
        if reOrtho:
            rk = rk - AP @ Ax(AP.T, rk)
            
        Ar = Ax(A, rk)
        rArp = torch.dot(rk, Ar)
        beta = rArp / rAr
        pk = rk + beta * pk
        Ap = Ar + beta * Ap
        
        # re-orthogonalization
        if reOrtho:
            #Ap = Ap - AP @ Ax(AP.T, Ap)
            AP = torch.concat([AP, Ap.reshape(-1, 1) / torch.norm(Ap)], dim = 1)
            if AP.shape[-1] > reOrtho and reOrtho != -1:
                AP = AP[:, 1 :]
        
        # updates
        rAr = rArp
        pAAp = torch.dot(Ap, Ap)
        k += 1
        
        # record stats
        norm_rk = torch.norm(rk)
        norm_Ark = torch.norm(Ar)
        records_rk.append(norm_rk / norm_b)
        records_Ark.append(norm_Ark / norm_Ab)
        
    return xk, k, records_rk, records_Ark
    
def Ax(A, x):
    if callable(A):
        return A(x)
    return torch.mv(A, x)

if __name__ == "__main__":
    N = 1000
    A = torch.randn(N, N, dtype = torch.float64)
    A = (A.T @ A)
    b = torch.rand(N, dtype = torch.float64)
    x, k, relr, relAr = CR(A, b, 1e-8, N, 50)
    
    import matplotlib.pyplot as plt
    plt.semilogy(relr, label = "relR")
    plt.semilogy(relAr, label = "relAr")
    plt.legend()
    plt.show()