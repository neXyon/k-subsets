#!/usr/bin/python
# -*- coding: utf8 -*-
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Luis F. Simoes
# Joerg H. Mueller

from itertools import combinations
from scipy.misc import comb
import numpy as np

# --------------------------------------------------------------------
# k-element subsets: reference sequence generators
# --------------------------------------------------------------------

def random_with_repetitions(n, k):
    """
    Random sequence, with repetitions.
    * May generate duplicate subsets;
    * May not generate all possible subsets.
    """
    elems = np.arange(n)
    while True:
        yield list(np.random.choice(elems, k, replace=False))



def random(n, k):
    """
    Random sequence.
    """
    seq = np.random.permutation(list(combinations(np.arange(n), k)))
    for s in seq:
        yield list(s)

##### --------------------------------------------------------------------

def random_with_repetition_expectation(n, k):
    """
    Derives the expected number of tries needed to find a correct
    sequence for a purely random algorithm.

    The probability for one random try to find the correct subset is

    p = # correct subsets / # possible subsets

    for a uniform distribution over the subsets. The expectation of when
    a correct subset will be found then is

    E = sum i * p * (1 - p)^i for i from 0 to infinity,

    which is

    E = 1 / p.
    """
    N = comb(n, k, True)

    t = np.arange(k, n + 1)

    S_t = comb(n, t)

    S = np.sum(S_t)

    E_Tau = N / comb(t, k)

    return 1 / S * np.sum(S_t * E_Tau)

##### --------------------------------------------------------------------

def random_expectation(n, k, t):
    """
    Derives the expected number of tries needed to find a correct
    sequence for a random algorithm that tries each sequence at
    maximum once.

    The probability for the nth random try to find the correct subset
    IF the previous ones went wrong is

    p_n = # correct subsets / (# possible subsets - n + 1).

    for a uniform distribution over the subsets.

    The expectation of when a correct subset will be found then is

    E = sum i * p_i * (prod 1 - p_j for j from 1 to i - 1) for i from 0 to max

    where max is the number of wrong subsets plus one.
    """
    T = 0
    q = 1
    N = comb(n, k, True)
    S = comb(t, k)

    for i in range(1, N + 1):
        p = S/(N - i + 1)
        p_i = p * q
        T += i * p_i
        q *= (1 - p)
    return T
