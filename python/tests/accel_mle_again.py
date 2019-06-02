#!/usr/bin/env python3

# -*- coding: utf-8 -*-
#  python/tests/accel_mle_again.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 06.01.2019

## Verify that our manual calculations of logs/inverses/solves are all good.
#exec(open("python/core/maxlik.py").read())
exec(open("python/core/maxlik_again.py").read())
exec(open("python/core/generative.py").read())

## Verify construction of variance and precision for latent time series.
phi = 0.45
sigmaz2 = 1.5
T = 5
VAR = form_ar_cov(phi, sigmaz2, T)
PREC = ar_prec_ldet(phi, sigmaz2, T)[1]

np.round(np.linalg.inv(VAR),2)
PREC

np.max(np.abs(VAR.dot(PREC) - np.eye(T)))

## Verify constant time determinant calculation.
phi = 0.45
sigmaz2 = 1.5
T = 5
det, PREC = ar_prec_ldet(phi, sigmaz2, T)
np.linalg.slogdet(PREC)[1]
det

## Check likelihood for some points
np.random.seed(123)
V = 1# Number of voxels
N = 3# Number of participants
T = 10# Number of time points
rho = 1.0 # Hyperparam on important voxels
tau_z = 1# variance Hyperparam on AR process variance for each voxel
tau_y = 1# variance Hyperparam on AR process variance for each voxel

gen = gen_lts(V, N, T, rho, tau_z, tau_y)
Y = gen['Y'][0,:,:]

true_sig = gen['sigma_y'][0]
true_phi = gen['phi'][0]
true_sigz = gen['sigma_z'][0]
theta = [true_sig, true_phi, true_sigz]

print(llik_lts(theta, Y))
print(llik_lts_fast(theta, Y))

## Check likelihood at scale
np.random.seed(123)
V = 1# Number of voxels
N = 3# Number of participants
T = 10881# Number of time points
# The desired T for the sherlock problem is 108815
rho = 1.0 # Hyperparam on important voxels
tau_z = 1# variance Hyperparam on AR process variance for each voxel
tau_y = 1# variance Hyperparam on AR process variance for each voxel

gen = gen_lts(V, N, T, rho, tau_z, tau_y)
Y = gen['Y'][0,:,:]

true_sig = gen['sigma_y'][0]
true_phi = gen['phi'][0]
true_sigz = gen['sigma_z'][0]
theta = [true_sig, true_phi, true_sigz]

#print(llik_lts(theta, Y))
print(llik_lts_fast(theta, Y))

## Check consistency
np.random.seed(123)
V = 1# Number of voxels
N = 3# Number of participants
T = 10881# Number of time points
# The desired T for the sherlock problem is 108815
rho = 1.0 # Hyperparam on important voxels
tau_z = 1# variance Hyperparam on AR process variance for each voxel
tau_y = 1# variance Hyperparam on AR process variance for each voxel

gen = gen_lts(V, N, T, rho, tau_z, tau_y)
Y = gen['Y'][0,:,:]

true_sig = gen['sigma_y'][0]
true_phi = gen['phi'][0]
true_sigz = gen['sigma_z'][0]
theta = [true_sig, true_phi, true_sigz]

#print(llik_lts(theta, Y))
print(llik_lts_fast(theta, Y))
