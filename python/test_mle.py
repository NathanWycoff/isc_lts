#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/test_mle.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.27.2018

## Test MLE consistency
import numpy as np
from scipy.optimize import minimize
execfile('python/maxlik.py')
execfile('python/generative.py')

### Draw some random params, not even from our model, just to test convergence
N = 2
T = 3
Y = np.random.normal(size = [T, N])
z = np.random.normal(size = [T, 1])

llik([1.0, 1.0, 1.0], Y)

# To prevent L-BFGS-B from actually trying a zero variance
minvar = 1e-2

to_opt = lambda theta: -llik(theta, Y)
bounds = [(minvar, np.inf), (-1, 1), (minvar, np.inf)]
minimize(to_opt, [1, 1, 1], bounds = bounds, method = 'L-BFGS-B')

# Informally test for convexity
n_inits = 10
inits = [[abs(np.random.normal(1, scale = 10)), np.random.uniform(-1, 1), \
        abs(np.random.normal(1, scale = 10))] for _ in range(n_inits)]

opts = []
for init in inits:
    to_opt = lambda theta: -llik(theta, Y)
    bounds = [(minvar, np.inf), (-1, 1), (minvar, np.inf)]
    opt = minimize(to_opt, init, bounds = bounds, method = 'L-BFGS-B')['x']
    opts.append(opt)
# Hmm possibly not in phi

### Now draw some data from our model, test consistency
V = 1# Number of voxels
P = 5000# Number of participants
T = 400# Number of time points
rho = 1 # Hyperparam on important voxels
tau_z = 1# variance Hyperparam on AR process variance for each voxel
tau_y = 1# variance Hyperparam on AR process variance for each voxel

gen = gen_lts(V, P, T, rho, tau_z, tau_y)
Y = gen['Y'][0,:,:]

# To prevent L-BFGS-B from actually trying a zero variance
minvar = 1e-2

true_sig = gen['sigma_y'][0]
true_phi = gen['phi'][0]
true_sigz = gen['sigma_z'][0]

to_opt = lambda theta: -llik(theta, Y)
bounds = [(minvar, np.inf), (-1, 1), (minvar, np.inf)]
ret = minimize(to_opt, [true_sig, true_phi, true_sigz], bounds = bounds, \
        method = 'L-BFGS-B')
print(ret['x'])
print([np.square(true_sig), true_phi, np.square(true_sigz)])
# Oh man we did it boy
