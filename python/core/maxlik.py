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

#def form_ar_cov(phi, sigmaz2, T):
#    """
#    Form the AR 1 covariance matrix
#    """
#    SIGMAz = np.empty(shape = [T, T])
#    for t1 in range(T):
#        for t2 in range(t1, T):
#            SIGMAz[t1, t2] = SIGMAz[t2, t1] = pow(phi, t2 - t1)
#    SIGMAz *= sigmaz2 / (1.0 - pow(phi, 2))
#    
#    return SIGMAz
#
#

def tridiag_det(A):
    """
    Calculate the determinant of a tridiagonal matrix

    A should be an np array
    """
    if (type(A) is not np.ndarray):
        raise ValueError("A must be a np.ndarray")
    if (A.shape[0] is not A.shape[1]):
        raise ValueError("A must be square.")
    N = A.shape[0]
    f = np.empty(N+2)
    f[0] = 0
    f[1] = 1
    f[2] = A[1,1]
    for n in range(3,N+2):
        f[n] = A[n-2,n-2] * f[n-1] - A[n-2,n-3] * A[n-3,n-2] * f[n-2]

    return(f[-1])


def ar_prec_ldet(phi, sigmaz2, T):
    """
    Efficient Calculation of the log determinant of the AR 1 precision matrix
    """
    if (phi > 1 + np.sqrt(np.finfo(float).eps)):
        raise ValueError("AR process is assumed stationary but we just got passed a phi > 1")
    ld = T*np.log(sigmaz2)
    inv = np.zeros([T,T])
    diag_arg = (1.0 + pow(phi,2)) / sigmaz2
    od_arg = -phi / float(sigmaz2)
    for t1 in range(T):
        if t1 == T-1:
            inv[t1,t1] = 1.0 / sigmaz2
        else:
            inv[t1,t1] = diag_arg
            inv[t1+1,t1] = inv[t1,t1+1] = od_arg

    return [ld, inv]

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

def llik_lts_fast(theta, Y):
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

    ld, SIGMAi = ar_prec_ldet(phi, sigmaz2, T)

    SIGMAz = np.linalg.inv(SIGMAzi)
    muz = np.reshape(np.sum(Y, axis = 1).T / sigma2, [T,1])

    C1 = -N / 2.0 * T * np.log(2 * np.pi * sigma2) + \
            -0.5 * (T * np.log(2 * np.pi) + ld)
    ll = C1 + \
            +0.5 * logdet(2 * np.pi * SIGMAz) + \
            +0.5 * muz.T.dot(SIGMAz).dot(muz) + \
            -0.5 * np.matrix.trace(Y.T.dot(Y)) / sigma2
    return(ll[0,0])

#def llik_lts_fast(theta, Y):
#    """
#    Log likelihood with Z integrated out of the params
#
#    :param: theta A vector with parameters [sigma2, phi, sigmaz2]
#    :param: Y T by N matrix giving the observed data.
#    """
#    sigma2, phi, sigmaz2 = theta
#    T = Y.shape[0]
#    N = Y.shape[1]
#
#    ld, SIGMAi = ar_prec_ldet(phi, sigmaz2, T)
#
#    SIGMAzi = 1/(sigma2 / N) * np.identity(T) + SIGMAi
#    SIGMAz = np.linalg.inv(SIGMAzi)
#    muz = np.reshape(SIGMAz.dot(np.sum(Y, axis = 1).T) / sigma2, [T,1])
#
#    C1 = -N / 2.0 * T * np.log(2 * np.pi * sigma2) + \
#            -0.5 * T * (np.log(2 * np.pi) + ld)
#    ll = C1 + \
#            +0.5 * logdet(2 * np.pi * SIGMAz) + \
#            +0.5 * muz.T.dot(SIGMAzi).dot(muz) + \
#            -0.5 * np.matrix.trace(Y.T.dot(Y)) / sigma2
#    return(ll[0,0])


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

def mle_lts_fast(Y, bound = False):
    """
    Maximize the LTS likelihood.

    :param: Y T by N matrix giving the observed data.
    :param: bound If true, we conduct the optimization under the null hypothesis, constraining the latent AR coefficient phi to be zero, and the latent time series variance to be (approximately) zero as well.
    """
    # To prevent L-BFGS-B from actually trying a zero variance
    minvar = 1e-8
    to_opt = lambda theta: -llik_lts_fast(theta, Y)
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
