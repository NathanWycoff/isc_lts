#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  dev/normal.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.18.2018

import numpy as np
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt

m = 1000
n = 100#Note, I assume k is even later.

sigma = 1
X = np.random.normal(size=[m,n])

ll_alt = []
for i in range(m):
    means = [np.mean(X[i,:(n/2)]) for _ in range(n/2)] + \
            [np.mean(X[i,(n/2):]) for _ in range(n/2)]
    ll_alt.append(np.sum(norm.logpdf(X[i,:], loc = means, scale = np.std(X[i,:]))))

ll_null = np.array([np.sum(norm.logpdf(X[i,:], loc = 0, scale = sigma)) \
        for i in range(m)])


lrt = -2*(ll_null - ll_alt)
pvals = [chi2.cdf(l, 3) for l in lrt]
plt.hist(pvals)
plt.show()
