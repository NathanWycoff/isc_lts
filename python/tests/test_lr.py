#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/test_lr.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.27.2018

# Test out our likelihood ratio!
import numpy as np
execfile('python/maxlik.py')
execfile('python/generative.py')

# Generate a bunch of data
V = 100# Number of voxels
N = 50# Number of participants
T = 40# Number of time points
rho = 0.5 # Hyperparam on important voxels
tau_z = 1# variance Hyperparam on AR process variance for each voxel
tau_y = 1# variance Hyperparam on AR process variance for each voxel

gen = gen_lts(V, N, T, rho, tau_z, tau_y)

lrts = lrt_lts(gen['Y'])

on = [bool(x) for x in gen['isactive']]
off = [not x for x in on]
np.mean(lrts[on])
np.mean(lrts[off])
