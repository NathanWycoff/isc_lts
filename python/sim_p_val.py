#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/sim_p_val.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.22.2018

# Obtain p values via simulation

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
