#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/compare_isc.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.27.2018

## Compare the LTS model vs the current standard for ISC
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
execfile('python/maxlik.py')
execfile('python/generative.py')
execfile('python/lib.py')

# Generate a bunch of data
V = 20# Number of voxels
N = 10# Number of participants
T = 10# Number of time points
rho = 0.5 # Hyperparam on important voxels
tau_z = 1# variance Hyperparam on AR process variance for each voxel
tau_y = 1# variance Hyperparam on AR process variance for each voxel

gen = gen_lts(V, N, T, rho, tau_z, tau_y)

lrts = lrt_lts(gen['Y'])
slrts = [-x for x in lrts]
slrts = [softmax([-x, 0])[0] for x in lrts]

fpr_lts, tpr_lts, threshs_lts = roc_curve(gen['isactive'], slrts)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_lts, tpr_lts, label='LTS')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
