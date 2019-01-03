#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/exp/sherlock_anl.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 01.02.2019

## Analyze the results from running both methods on the Sherlock data.
import numpy as np
from nilearn import plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
heur_brain = np.load('data/heur_brain.npy')

def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = heur_brain[26, :, :]
slice_1 = heur_brain[:, 30, :]
slice_2 = heur_brain[:, :, 16]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")  
plt.savefig('coolio.png')

plotting.plot_glass_brain('data/heur_brain.nii', 'coolioo.png', colorbar = True)
