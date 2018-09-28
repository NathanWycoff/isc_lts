#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/heur_isc.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.28.2018

## Implement the heuristic ISC procedure describred in 
import numpy as np

def cor(a, b):
    """
    Calculate sample correlation between two 1D arrays
    """
    ma = np.mean(a)
    mb = np.mean(b)
    ac = a - ma
    bc = b - mb
    cov = (ac.dot(bc)) / len(a)
    return  cov / (np.std(a) * np.std(b))

def heur_isc(Y):
    """
    :param Y: A Voxel by Time by Subject tensor giving observed voxel values.
    """
    V, T, N = np.shape(Y)
    R = np.empty([V,N])
    for v in range(V):
        Yv = Y[v,:,:]
        for i in range(N):
            n1 = Yv[:,[ii for ii in range(N) if not ii == i]]
            n1m = np.mean(Yv, axis = 1)
            R[v,i] = cor(n1m, Yv[:,i])
    return np.mean(R, axis = 1)
