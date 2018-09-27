#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/gen_pg.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.26.2018

import matplotlib.pyplot as plt
execfile('python/generative.py')

# Mess around with our generative model
V = 10# Number of voxels
P = 5# Number of participants
T = 4# Number of time points
rho = 0.5 # Hyperparam on important voxels
tau_z = 1# variance Hyperparam on AR process variance for each voxel
tau_y = 1# variance Hyperparam on AR process variance for each voxel

gen = gen_lts(V, P, T, rho, tau_z, tau_y)

# Look at latent AR process for both variables
on = [bool(x) for x in gen['isactive']]
off = [not x for x in on]

gen['Z'][on,:]
gen['Z'][off,:]
