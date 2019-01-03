#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  python/lib.py Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 09.27.2018

#A numerically stable softmax function
def softmax(x):
    return np.exp(x - np.max(x) - np.log(sum(np.exp(x - np.max(x)))))
