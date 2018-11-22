#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/null_err_cont.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.18.2018

# Confirm asymptotic error control
from scipy.stats import cumfreq
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from scipy.stats import chi2
execfile('python/maxlik.py')
execfile('python/generative.py')
execfile('python/lib.py')
execfile('python/heur_isc.py')

# Generate a bunch of data
V = 1000# Number of voxels
N = 30# Number of participants
T = 30# Number of time points
rho = 0 # Hyperparam on important voxels
tau_z = 0.5# variance Hyperparam on AR process variance for each voxel
tau_y = 2# variance Hyperparam on observed data

gen = gen_lts(V, N, T, rho, tau_z, tau_y)

Y = gen['Y']
Y = np.random.normal(size=[V,T,N])

lrts = lrt_lts(Y)
#TODO: Negatives are not normal
lrts = [0 if x < 0 else x for x in lrts]

heur = heur_isc(Y)

pvals = [lts_pval(lrts).lrt_2_pval(l) for l in lrts]
np.mean(np.array(pvals) < 0.05)

# Plot a histogram of the p values, which should be uniform
plt.hist(pvals)
plt.show()
#plt.savefig('./images/null_hist.pdf')
