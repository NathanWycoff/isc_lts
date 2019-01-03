#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#  python/exp/data_playground.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.22.2018

# Load up cropped and masked fMRI data.
import nibabel as nib
import numpy as np

# Determine which files have our target stimulus
dir_pre = '/home/nate/ds001110-download/sub-'
dir_post = '/func/'

file_pre = 'sub-'
file_post = '_task-SherlockMovie_bold.nii.gz'

brains = []
masks = []
for i in range(1,38):
    if i < 10:
        file_path = file_pre + '0' + str(i) + file_post
        dir_path = dir_pre + '0' + str(i) + dir_post
    else:
        file_path = file_pre + str(i) + file_post
        dir_path = dir_pre + str(i) + dir_post
    try:
        nii = nib.load(dir_path + file_path)
        brain = nii.get_data()
        brains.append(brain)
    except IOError:
        print("Subject %d did not perform the task"%i)

# Save one of the affine transforms at random for later
an_aff = nii.affine

# Make a mask (assuming the data were masked already)
brain_shape = brains[0].shape[:3]
mask = np.ones(shape=brain_shape)
for brain in brains:
    maskb = brain[:,:,:,0] != 0
    maskb = maskb
    mask *= maskb

# Crop data to specs
crop_begin = 18
crop_end = 960
for i in range(len(brains)):
    brains[i] = brains[i][:,:,:,crop_begin:crop_end]

def stdze(x):
    sd = np.std(x)
    mean = np.mean(x)

    if (sd > 0):
        return (x - mean) / sd
    else:
        return np.zeros_like(x)

# Standardize by Voxels TODO: Implement on the fly
#for i in range(len(brains)):
#    brains[i] = brains[i].astype(float)
#    for j in range(brains[i].shape[0]):
#        for k in range(brains[i].shape[1]):
#            for l in range(brains[i].shape[2]):
#                brains[i][j,k,l,:] = stdze(brains[i][j,k,l,:])

brains = np.array(brains)
brains = np.moveaxis(brains, source = 0, destination = 4)

np.save('data/mask.npy', mask)
np.save('data/brains.npy', brains)
np.save('data/an_aff.npy', an_aff)
