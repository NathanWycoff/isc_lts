#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/maxlik.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.27.2018

## Maximize the likelihood with Z integrated out.

logdet = lambda A: np.linalg.slogdet(A)[1]
sqmag = lambda x: np.sum([np.square(xi) for xi in x])

def form_ar_cov(phi, sigmaz2, T):
    """
    Form the AR covariance matrix
    """
    pd = 0
    SIGMAz = np.empty(shape = [T, T])
    for t1 in range(T):
        pd += pow(np.square(phi), t1)
        for t2 in range(t1, T):
            SIGMAz[t1, t2] = SIGMAz[t2, t1] = pow(phi, t2 - t1) * pd
    SIGMAz *= sigmaz2
    
    return SIGMAz

def llik(theta, Y):
    """
    Likelihood with Z integrated out of the params

    :param: theta A vector with parameters [sigma2, phi, sigmaz2]
    :param: A T by N matrix giving the observed data.
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
