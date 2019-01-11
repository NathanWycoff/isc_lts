#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/tests/accel_mle.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.04.2019

## Verify that our manual calculations of logs/inverses/solves are all good.
execfile('python/core/maxlik.py')
execfile('python/core/generative.py')

def printall(phi, sigma2, T):
    A = form_ar_cov(phi, sigma2, T)
    print(np.linalg.slogdet(A)[1])
    print(np.linalg.inv(A))
    print(ar_prec_ldet(phi, sigma2, T))

## ## ## ## ## ## ## ## ## ## ## ## ## 
## Look directly at the linear algebra
phi = 0.2
sigma2 = 10
T = 4
printall(phi, sigma2, T)

phi = 0.9
sigma2 = 0.1
T = 4
printall(phi, sigma2, T)

## ## ## ## ## ## ## ## ## ## ## ## ## 
## Verify that the mle solutions are equivalent.
np.random.seed(123)
V = 1# Number of voxels
N = 50# Number of participants
T = 40# Number of time points
rho = 1.0 # Hyperparam on important voxels
tau_z = 1# variance Hyperparam on AR process variance for each voxel
tau_y = 1# variance Hyperparam on AR process variance for each voxel

gen = gen_lts(V, N, T, rho, tau_z, tau_y)
Y = gen['Y'][0,:,:]

true_sig = gen['sigma_y'][0]
true_phi = gen['phi'][0]
true_sigz = gen['sigma_z'][0]

ret = mle_lts(Y)
ret_fast = mle_lts_fast(Y)

print(ret['theta'])
print(ret_fast['theta'])
print([np.square(true_sig), true_phi, np.square(true_sigz)])
# Oh man we did it boy

