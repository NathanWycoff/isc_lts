#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/exp/sherlock_app.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 12.18.2018

## Apply both methods to the Sherlock data and compare and store the results.
import numpy as np
execfile('python/core/heur_isc.py')
execfile('python/core/maxlik.py')

# Data processed in python/exp/data_playground.py 
mask = np.load('data/mask.npy')
brains = np.load('data/brains.npy')

# Extract only nonmasked voxels
flatmask = mask.flatten().astype(bool)
Y = brains.reshape([np.prod(brains.shape[:3]), brains.shape[3], brains.shape[4]])
Yf = Y[flatmask,:,:]

# Run the analysis on these voxels
heur = heur_isc(Yf)
lrts = lrt_lts(Yf)

# Backtransform the results into brain shaped arrays
heur_brain = np.full(Y.shape[0], -1.0)
heur_brain[flatmask] = heur
heur_brain = heur_brain.reshape(brains.shape[:3])

# Save the results
np.save('data/heur_brain.npy', heur_brain)

# Save as nib file
an_aff = np.load('data/an_aff.npy') 
heur_brain_nii = nib.AnalyzeImage(heur_brain, nii.affine)
nib.save(heur_brain_nii, 'data/heur_brain.nii')
