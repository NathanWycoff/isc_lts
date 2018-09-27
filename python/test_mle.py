#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/test_mle.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.27.2018

## Test MLE consistency
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
execfile('python/maxlik.py')
execfile('python/generative.py')

### Now draw some data from our model, test consistency
V = 1# Number of voxels
N = 50# Number of participants
T = 40# Number of time points
rho = 0 # Hyperparam on important voxels
tau_z = 1# variance Hyperparam on AR process variance for each voxel
tau_y = 1# variance Hyperparam on AR process variance for each voxel

gen = gen_lts(V, N, T, rho, tau_z, tau_y)
Y = gen['Y'][0,:,:]

true_sig = gen['sigma_y'][0]
true_phi = gen['phi'][0]
true_sigz = gen['sigma_z'][0]

ret = mle_lts(Y)

print(ret['theta'])
print([np.square(true_sig), true_phi, np.square(true_sigz)])
# Oh man we did it boy

# Informally test for convexity
n_inits = 10
inits = [[abs(np.random.normal(1, scale = 10)), np.random.uniform(-1, 1), \
        abs(np.random.normal(1, scale = 10))] for _ in range(n_inits)]

opts = []
for init in inits:
    minvar = 1e-2
    to_opt = lambda theta: -llik_lts(theta, Y)
    bounds = [(minvar, np.inf), (-1, 1), (minvar, np.inf)]
    opt = minimize(to_opt, init, bounds = bounds, method = 'L-BFGS-B')['x']
    opts.append(opt)

