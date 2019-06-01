#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/tests/accel_mle.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.04.2019

## Verify that our manual calculations of logs/inverses/solves are all good.
exec(open("python/core/maxlik.py").read())
exec(open("python/core/generative.py").read())

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

## Check log-determinant of constant tridiag.
b = 0.7
a = np.sqrt(7*np.square(b))
T = 30
A = np.diag([a for _ in range(T)])
for t in range(T-1):
    A[t,t+1] = A[t+1,t] = b

print(np.log(np.linalg.det(A)))
print(td_const_ldet(a, b, T))

## Check likelihood for some points
np.random.seed(123)
V = 1# Number of voxels
N = 3# Number of participants
T = 2# Number of time points
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
print(llik_lts_again(theta, Y))



## ## ## ## ## ## ## ## ## ## ## ## ## 
## Verify that the mle solutions are equivalent.
np.random.seed(123)
V = 1# Number of voxels
N = 5# Number of participants
T = 10# Number of time points
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
