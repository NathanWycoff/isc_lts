#!/usr/bin/env python
# -*- coding: utf-7 -*-
#  python/compare_isc.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.27.2018

## Compare the LTS model vs the current standard for ISC
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
N = 5# Number of participants
T = 10# Number of time points
rho = 0.2 # Hyperparam on important voxels
tau_z = 0.5# variance Hyperparam on AR process variance for each voxel
tau_y = 2# variance Hyperparam on observed data

gen = gen_lts(V, N, T, rho, tau_z, tau_y)

lrts = lrt_lts(gen['Y'])
#slrts = [x for x in lrts]
# This should get us p-values.
slrts = [chi2.cdf(x, 2) for x in lrts]
#slrts = [softmax([-x, 0])[0] for x in lrts]

heur = heur_isc(gen['Y'])

fpr_lts, tpr_lts, threshs_lts = roc_curve(gen['isactive'], slrts)
fpr_heur, tpr_heur, threshs_heur = roc_curve(gen['isactive'], heur)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_lts, tpr_lts, label='LTS')
plt.plot(fpr_heur, tpr_heur, label='Heuristic')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
#plt.savefig('./images/roc.pdf')
