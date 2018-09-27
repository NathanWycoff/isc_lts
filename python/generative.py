#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/generative.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.26.2018

#TODO:
# 1) Verify that we are starting the AR process the way we want to
# 2) Throw a constant term in

import numpy as np
from scipy.stats import truncnorm

def gen_lts(V, P, T, rho, tau_z, tau_y):
    """
    Generate data from the Latent Time Series Model for Inter-Subject Correlation

    :param: V Number of voxels
    :param: P Number of participants
    :param: T Number of time points
    :param: rho Hyperparam on proportion of important voxels
    :param: tau_z Variance Hyperparam on AR process variance for each voxel
    :param: tau_y Variance Hyperparam on observed data
    """
    # Draw from priors
    phi = np.empty(V)
    isactive = [False for _ in range(V)]
    sigma_z = np.empty(V)
    sigma_y = np.empty(V)
    for v in range(V):
        # Draw from a spike-slab U[-1, 1] with prob rho on phi
        # Draw from a spike-slab truncated normal for variance of latent AR process 
        isactive[v] = np.random.binomial(1, rho)
        if isactive[v]:
            phi[v] = np.random.uniform(low = -1, high = 1, size = 1)
            sigma_z[v] = abs(np.random.normal(loc = 0, scale = tau_z))
        else:
            phi[v] = 0
            sigma_z[v] = 0
        
        # Draw from a truncated normal for variance of observed data
        sigma_y[v] = abs(np.random.normal(loc = 0, scale = tau_y))

    # Draw latent variables
    Z = np.empty([V,T])
    for v in range(V):
        Z[v,0] = np.random.normal(loc = 0, scale = sigma_z[v])
        for t in range(1, T):
            Z[v,t] = np.random.normal(loc = phi[v] * Z[v,t-1], scale = sigma_z[v])

    # Draw observed data
    Y = np.empty([V,T,P])
    for v in range(V):
        for t in range(T):
            for p in range(P):
                Y[v,t,p] = np.random.normal(loc = Z[v,t], scale = sigma_y[v])

    retd = {'isactive' : isactive, 'phi' : phi, 'sigma_z' : sigma_z, \
            'sigma_y' : sigma_y, 'Z' : Z, 'Y' : Y}
    return retd
