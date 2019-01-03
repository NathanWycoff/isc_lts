#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/maxlik.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.27.2018

## Maximize the likelihood with Z integrated out.
from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np
from tqdm import tqdm

logdet = lambda A: np.linalg.slogdet(A)[1]
sqmag = lambda x: np.sum([np.square(xi) for xi in x])

def form_ar_cov(phi, sigmaz2, T):
    """
    Form the AR 1 covariance matrix
    """
    pd = 0
    SIGMAz = np.empty(shape = [T, T])
    for t1 in range(T):
        pd += pow(np.square(phi), t1)
        for t2 in range(t1, T):
            SIGMAz[t1, t2] = SIGMAz[t2, t1] = pow(phi, t2 - t1) * pd
    SIGMAz *= sigmaz2
    
    return SIGMAz

def invdet_ar_cov(phi, sigmaz2, T):
    """
    Form the inverse of the AR 1 covariance matrix as well as its determinant
    """
    pd = 0
    SIGMAzi = np.empty(shape = [T, T])

    SIGMAzi *= 1.0/sigmaz2
    
    return SIGMAzi

def llik_lts(theta, Y):
    """
    Log likelihood with Z integrated out of the params

    :param: theta A vector with parameters [sigma2, phi, sigmaz2]
    :param: Y T by N matrix giving the observed data.
    """
    sigma2, phi, sigmaz2 = theta
    T = Y.shape[0]
    N = Y.shape[1]
    SIGMA = form_ar_cov(phi, sigmaz2, T)
    SIGMAi = np.linalg.inv(SIGMA)

    SIGMAzi = 1/(sigma2 / N) * np.identity(T) + SIGMAi
    SIGMAz = np.linalg.inv(SIGMAzi)
    muz = np.reshape(SIGMAz.dot(np.sum(Y, axis = 1).T) / sigma2, [T,1])

    C1 = -N / 2.0 * logdet(2 * np.pi * sigma2 * np.identity(T)) + \
            -0.5 * logdet(2 * np.pi * SIGMA)
    ll = C1 + \
            +0.5 * logdet(2 * np.pi * SIGMAz) + \
            +0.5 * muz.T.dot(SIGMAzi).dot(muz) + \
            -0.5 * np.matrix.trace(Y.T.dot(Y)) / sigma2
    return(ll[0,0])

def mle_lts(Y, bound = False):
    """
    Maximize the LTS likelihood.

    :param: Y T by N matrix giving the observed data.
    :param: bound If true, we conduct the optimization under the null hypothesis, constraining the latent AR coefficient phi to be zero, and the latent time series variance to be (approximately) zero as well.
    """
    # To prevent L-BFGS-B from actually trying a zero variance
    minvar = 1e-8
    to_opt = lambda theta: -llik_lts(theta, Y)
    if bound:
        bounds = [(minvar, np.inf), (0, 0), (minvar, minvar)]
    else:
        bounds = [(minvar, np.inf), (-1, 1), (minvar, np.inf)]
    ret = minimize(to_opt, [1, 0, 1], bounds = bounds, \
            method = 'L-BFGS-B')
    return {'theta' : ret['x'], 'll' : -ret['fun']}

def lrt_lts_i(Y):
    """
    Likelihood ratio test for determining whether a single voxel is active.

    Returns -2 times the logliklihood.
    """
    ll_null = mle_lts(Y, bound = True)['ll']
    ll_lts = mle_lts(Y)['ll']
    return -2 * (ll_null - ll_lts)

def lrt_lts(Ys, verb = True):
    """
    Likelihood ratio test for determining active voxels.
    """
    lrts = np.empty(Ys.shape[0])
    for i in tqdm(range(Ys.shape[0])):
        lrts[i] = lrt_lts_i(Ys[i,:,:])
    return lrts

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

class lts_pval(object):
    """
    Gets p values from LRT via the empirical distribution from a simulation.
    """
    def __init__(self, null_lrts):
        self.val, self.cumprob = ecdf(null_lrts)

    def lrt_2_pval(self, lrt):
        """
        Give me an LRT, and I will give you its p-value.
        """
        ind = (self.val < lrt)[::-1].argmax()
        pval = 1-self.cumprob[ind]
        return pval
