#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/verify_z_density.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.27.2018

## Verify the posterior of Z is what we think it is.
import numpy as np

# Draw some random params, not even from our model
N = 2
T = 3
Y = np.random.normal(size = [T, N])
z = np.random.normal(size = [T, 1])
sigma2 = 1.0
Ls = np.random.normal(size = [T, T])
SIGMA = Ls.T.dot(Ls)
SIGMAi = np.linalg.inv(SIGMA)

logdet = lambda A: np.linalg.slogdet(A)[1]
sqmag = lambda x: np.sum([np.square(xi) for xi in x])
C1 = -N / 2.0 * logdet(2 * np.pi * sigma2 * np.identity(T)) + \
        -0.5 * logdet(2 * np.pi * SIGMA)

# Initial joint dist
logd = C1 + \
        -0.5 * (z.T.dot(SIGMAi).dot(z) + \
        sum([sqmag(Y[:,i] - z.T) for i in range(N)]) / sigma2)

# Reparameterized in order to aide integration
SIGMAzi = 1/(sigma2 / N) * np.identity(T) + SIGMAi
SIGMAz = np.linalg.inv(SIGMAzi)
muz = np.reshape(SIGMAz.dot(np.sum(Y, axis = 1).T) / sigma2, [T,1])
logd2 = C1 + \
        -0.5 * (z-muz).T.dot(SIGMAzi).dot(z-muz) + \
        +0.5 * muz.T.dot(SIGMAzi).dot(muz) + \
        -0.5 * np.matrix.trace(Y.T.dot(Y)) / sigma2

