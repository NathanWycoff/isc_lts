#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/verify_z_integral.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.27.2018

## Verify that our analytic method of integrating Z out works.
import numpy as np
from scipy.integrate import tplquad

# Draw some random params, not even from our model
N = 4
T = 3
Y = np.random.normal(size = [T, N])
z = np.random.normal(size = [T, 1])
sigma2 = 1.0
Ls = np.random.normal(size = [T, T])
SIGMA = Ls.T.dot(Ls) * 0.01
SIGMAi = np.linalg.inv(SIGMA)

logdet = lambda A: np.linalg.slogdet(A)[1]
sqmag = lambda x: np.sum([np.square(xi) for xi in x])
C1 = -N / 2.0 * logdet(2 * np.pi * sigma2 * np.identity(T)) + \
        -0.5 * logdet(2 * np.pi * SIGMA)

# Compute the integral numerically 
def pdf(z1, z2, z3):
    z = np.reshape([z1, z2, z3], [3, 1])
    ret = C1 + -0.5 * (z.T.dot(SIGMAi).dot(z) + \
        sum([sqmag(Y[:,i] - z.T) for i in range(N)]) / sigma2)
    return np.exp(ret)

qret = tplquad(pdf, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf, \
        lambda x,y: -np.inf, lambda x,y: np.inf)
num = qret[0]

# See if it matches our analytic formula
SIGMAzi = 1/(sigma2 / N) * np.identity(T) + SIGMAi
SIGMAz = np.linalg.inv(SIGMAzi)
muz = np.reshape(SIGMAz.dot(np.sum(Y, axis = 1).T) / sigma2, [T,1])
logI = C1 + \
        +0.5 * logdet(2 * np.pi * SIGMAz) + \
        +0.5 * muz.T.dot(SIGMAzi).dot(muz) + \
        -0.5 * np.matrix.trace(Y.T.dot(Y)) / sigma2
I = np.exp(logI)
