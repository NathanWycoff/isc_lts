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

#def form_ar_cov(phi, sigmaz2, T):
#    """
#    Form the AR 1 covariance matrix
#    """
#    pd = 0
#    SIGMAz = np.empty(shape = [T, T])
#    for t1 in range(T):
#        pd += pow(np.square(phi), t1)
#        for t2 in range(t1, T):
#            SIGMAz[t1, t2] = SIGMAz[t2, t1] = pow(phi, t2 - t1) * pd
#    SIGMAz *= sigmaz2
#    
#    return SIGMAz

def form_ar_cov(phi, sigmaz2, T):
    """
    Form the AR 1 covariance matrix
    """
    SIGMAz = np.empty(shape = [T, T])
    for t1 in range(T):
        for t2 in range(t1, T):
            SIGMAz[t1, t2] = SIGMAz[t2, t1] = pow(phi, t2 - t1)
    SIGMAz *= sigmaz2 / (1.0 - pow(phi, 2))
    
    return SIGMAz

def ar_trid_det(A):
    """
    Get the determinant of a tridiagonal matrix with constant off-diagonal terms, and constant diagonals except for the first and last elements, which are equal to each other but not the other diagonal elements in constant time.
    """

    if T < 5:
        return(np.linalg.det(A))
    else:
        gamma = A[0,0]
        alpha = A[1,1]
        beta = A[1,2]

        # Get determinant of "middle matrices"
        d1 = np.exp(td_const_ldet(alpha, beta, T-2))
        print(np.abs(np.linalg.det(A[1:-1,1:-1]) - d1))
        d2 = np.exp(td_const_ldet(alpha, A[1,2], T-3))
        print(np.abs(np.linalg.det(A[2:-1,2:-1]) - d2))
        d3 = np.exp(td_const_ldet(alpha, A[1,2], T-4))
        print(np.abs(np.linalg.det(A[3:-1,3:-1]) - d3))

        # Get determinant of all except last col/row
        dJ1 = gamma * d1 - pow(beta,2)*d2
        print(np.abs(np.linalg.det(A[:-1,:-1]) - dJ1))
        dJ2 = gamma * d2 - pow(beta,2)*d3
        print(np.abs(np.linalg.det(A[:-2,:-2]) - dJ2))

        # Get determinant of the whole enchilada.
        detA = gamma * dJ1 - pow(beta,2)*dJ2
        print(detA - np.linalg.det(A))



#TODO: Delete this prolly
# Get good determinant for tridiag
##NOTE: assumes sigamz2 = 1
#sigmaz2 = 1
#SIGMA = form_ar_cov(phi, sigmaz2, T)
#SIGMAi = np.round(np.linalg.inv(SIGMA),2)
#gamma = SIGMAi[0,0]
#alpha = SIGMAi[1,1]
#
#A = np.empty([T,T])
#A[:,:] = SIGMAi
#A[0,0] = A[T-1,T-1] = A[1,1]
#
##Check 1: should be equal
#np.linalg.slogdet(A)[1]
#ldetA = td_const_ldet(SIGMAi[1,1], SIGMAi[1,0], T)
#ldetA
#
##Check 2: still good after 1,1 diagonal shift
#A1 = np.empty([T,T])
#A1[:] = A
#A1[0,0] = SIGMAi[0,0]
#np.linalg.slogdet(A1)[1]
#ldetA + np.log(np.abs(1 + (gamma - alpha)*sigmaz2/(1-pow(phi,2))))

#TODO: Box this up
## Approach 2 (T assumed >= 5):

def td_const_ldet(a, b, T):
    """
    Calculate the log determinant of a tridiagonal matrix with banded terms, such that every diagonal element is a, and the super and sub diagonal elements are b, and the matrix is TxT.

    A should be an np array
    """
    d = np.sqrt(np.square(a) - 4.0 * np.square(b))
    # Make this split because we are raising something to a potentially large power and
    # need to be ensure that only happens for something < 1.
    return(-np.log(d) + (T+1) * (np.log(0.5) + np.log(a+d)) + np.log(1 - pow((a-d)/(a+d), T+1)))
    #return(-np.log(d) + (pow((a + d)/2,T+1) - pow((a - d) / 2, T+1)))
    # Next line is probably never needed:
    #return(-np.log(d) + (T+1) * (np.log(0.5) + np.log(a-d)) + np.log(pow((a+d) / (a-d), T+1) - 1))



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



#def ar_prec_ldet(phi, sigmaz2, T):
#    """
#    Efficient Calculation of the log determinant of the AR 1 precision matrix
#    """
#    if (phi > 1 + np.sqrt(np.finfo(float).eps)):
#        raise ValueError("AR process is assumed stationary but we just got passed a phi > 1")
#    ld = T*np.log(sigmaz2)
#    inv = np.zeros([T,T])
#    diag_arg = (1.0 - pow(phi,2)) / sigmaz2
#    od_arg = -phi / float(sigmaz2)
#    for t1 in range(T):
#        if t1 == T-1:
#            #TODO: Decide on one of these two.
#            #inv[t1,t1] = 1.0 / sigmaz2
#            inv[t1,t1] = diag_arg
#        else:
#            inv[t1,t1] = diag_arg
#            inv[t1+1,t1] = inv[t1,t1+1] = od_arg
#
#    return [ld, inv]

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

def llik_lts_again(theta, Y):
    """
    Log likelihood with Z integrated out of the params

    :param: theta A vector with parameters [sigma2, phi, sigmaz2]
    :param: Y T by N matrix giving the observed data.
    """
    sigma2, phi, sigmaz2 = theta
    T = Y.shape[0]
    N = Y.shape[1]
    SIGMA = form_ar_cov(phi, sigmaz2, T)
    ld, SIGMAi = ar_prec_ldet(phi, sigmaz2, T)

    SIGMAzi = 1/(sigma2 / N) * np.identity(T) + SIGMAi
    ldSIGMAzi = td_const_ldet(SIGMAzi[0,0], SIGMAzi[0,1], T)
    SIGMAz = np.linalg.inv(SIGMAzi)
    muz = np.reshape(np.sum(Y, axis = 1).T / sigma2, [T,1])

    C1 = -N / 2.0 * T * np.log(2 * np.pi * sigma2) + \
            -0.5 * (T * np.log(2 * np.pi) + ld)
            #+0.5 * logdet(2 * np.pi * SIGMAz) + \
    ll = C1 + \
            +0.5 * (T*np.log(2*np.pi) - ldSIGMAzi) + \
            +0.5 * muz.T.dot(SIGMAz).dot(muz) + \
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
    #SIGMA = form_ar_cov(phi, sigmaz2, T)
    ld, SIGMAi = ar_prec_ldet(phi, sigmaz2, T)

    SIGMAzi = 1.0/(sigma2 / N) * np.identity(T) + SIGMAi

    SIGMAz = np.linalg.inv(SIGMAzi)
    muz = np.reshape(np.sum(Y, axis = 1).T / sigma2, [T,1])

    # NOTE: PROBLEM: SIGMAzi is only ALMOST tridiagonal.
    ldSIGMAzi = td_const_ldet(SIGMAzi[0,0], SIGMAzi[0,1], T)
    #ldSIGMAzi = np.log(1 - np.square(phi)/sigmaz2*(T/sigma2+(1+np.square(phi))/sigmaz2)) + \
    #        - td_const_ldet(T/sigma2 + (1 + np.square(phi))/sigmaz2, -phi/sigmaz2, T)

    C1 = -N / 2.0 * T * np.log(2 * np.pi * sigma2) + \
            -0.5 * (T * np.log(2 * np.pi) + ld)
            #+0.5 * logdet(2 * np.pi * SIGMAz) + \
            #+0.5 * (T*np.log(2*np.pi) - np.log(tridiag_det(SIGMAzi))) + \
    ll = C1 + \
            +0.5 * (T*np.log(2*np.pi) - ldSIGMAzi) + \
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
