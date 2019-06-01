#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  python/core/maxlik_again.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 06.01.2019

## Maximize the likelihood with Z integrated out.
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import solve_banded
import numpy as np
from tqdm import tqdm


logdet = lambda A: np.linalg.slogdet(A)[1]
sqmag = lambda x: np.sum([np.square(xi) for xi in x])

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

#TODO: Avoid ever computing the matrix explicitly.
def ar_prec_ldet(phi, sigmaz2, T):
    """
    Calculate our precision matrix and its log determinant.
    """
    gamma = 1.0/sigmaz2
    alpha = (1+pow(phi,2))/sigmaz2
    beta =  -phi/sigmaz2

    SIGMAi = np.diag(np.repeat(alpha,T)) + np.diag(np.repeat(beta, T-1), 1) + np.diag(np.repeat(beta, T-1), -1)
    SIGMAi[0,0] = SIGMAi[T-1,T-1] = gamma

    det = ar_trid_det(SIGMAi)
    ldet = np.log(np.abs(det))

    return [ldet, SIGMAi]

def td_const_ldet(a, b, T):
    """
    Calculate the log determinant of a tridiagonal matrix with banded terms, such that every diagonal element is a, and the super and sub diagonal elements are b, and the matrix is TxT.

    A should be an np array
    """
    d = np.sqrt(np.square(a) - 4.0 * np.square(b))
    return(-np.log(d) + (T+1) * (np.log(0.5) + np.log(a+d)) + np.log(1 - pow((a-d)/(a+d), T+1)))

#TODO: Computation in logspace perhaps?
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
        d2 = np.exp(td_const_ldet(alpha, A[1,2], T-3))
        d3 = np.exp(td_const_ldet(alpha, A[1,2], T-4))

        # Get determinant of all except last col/row
        dJ1 = gamma * d1 - pow(beta,2)*d2
        dJ2 = gamma * d2 - pow(beta,2)*d3

        # Get determinant of the whole enchilada.
        detA = gamma * dJ1 - pow(beta,2)*dJ2

        return(detA)

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
            -0.5 * logdet(2 * np.pi * SIGMA) + \
            +0.5 * logdet(2 * np.pi * SIGMAz)

    ll = C1 + \
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
    #SIGMA = form_ar_cov(phi, sigmaz2, T)
    ld, SIGMAi = ar_prec_ldet(phi, sigmaz2, T)

    SIGMAzi = 1.0/(sigma2 / N) * np.identity(T) + SIGMAi

    #SIGMAz = np.linalg.inv(SIGMAzi)
    muz = np.reshape(np.sum(Y, axis = 1).T / sigma2, [T,1])

    ldSIGMAzi = np.log(np.abs(ar_trid_det(SIGMAzi)))

    # Get SIGMAzi into banded form.
    banded_form = [[0] + list(np.diag(SIGMAzi,1)), 
            list(np.diag(SIGMAzi)),
            list(np.diag(SIGMAzi,1)) + [0]] 

    C1 = -N / 2.0 * T * np.log(2 * np.pi * sigma2) + \
            -0.5 * (T * np.log(2 * np.pi) - ld) +\
            +0.5 * (T*np.log(2*np.pi) - ldSIGMAzi)

    ll = C1 + \
            +0.5 * muz.T.dot(solve_banded((1,1), banded_form, muz)) + \
            -0.5 * np.matrix.trace(Y.T.dot(Y)) / sigma2
    return(ll[0,0])

