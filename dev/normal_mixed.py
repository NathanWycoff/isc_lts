#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  dev/normal_mixed.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.19.2018

import numpy as np
from scipy.stats import norm, chi2
from scipy.linalg import block_diag, cholesky, solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt

m = 2000
b = 20
bn = 10#Note, I assume k is even later.
n = b * bn

sigma_y = 1
sigma_b = 0
Z = np.random.normal(size=[m,n])

# Make the covariance matrix
J = np.ones(shape=[bn,bn])
SIGMA = sigma_y * np.identity(n) + sigma_b * block_diag(*[J for _ in range(b)])
R = cholesky(SIGMA)# Seems to be upper triangular, may vary with python version?

X = R.T.dot(Z.T).T

def nllik(sigmas, y):
    sigmay=sigmas[0]
    sigmab=sigmas[1]
    SIGMA = sigmay * np.identity(n) + sigmab * block_diag(*[J for _ in range(b)])
    ll = -0.5 * np.linalg.slogdet(SIGMA)[1] - 0.5 * y.T.dot(solve(SIGMA, y))
    return -ll

ll_alt = []
for i in range(m):
    ll = -minimize(nllik, [1, 1], args = (X[i,:]), method = 'L-BFGS-B', \
            bounds = [(1e-5, np.Inf), (1e-5, np.Inf)])['fun']
    ll_alt.append(ll)
ll_alt = np.array(ll_alt)

ll_null = []
for i in range(m):
    ll = -minimize(nllik, [1, 1], args = (X[i,:]), method = 'L-BFGS-B', \
            bounds = [(1e-5, np.Inf), (1e-8, 1e-8)])['fun']
    ll_null.append(ll)
ll_null = np.array(ll_null)

# Verify the asymptotic p value.
lrt = -2*(ll_null - ll_alt)
crit = chi2.ppf(0.90, 1)
np.mean(lrt > crit)

pvals = [1-chi2.cdf(l, 1) for l in lrt]
plt.hist(pvals)
plt.show()
