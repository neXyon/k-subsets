#!/usr/bin/python
# -*- coding: utf8 -*-
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Luis F. Simoes
# Joerg H. Mueller

import numpy as np

##### --------------------------------------------------------------------

def bin_coef(n, k):
    """
    Computes the binomial coefficient "n choose k"
    (number of ways of picking `k` unordered outcomes from `n` possibilities).

    See: http://mathworld.wolfram.com/BinomialCoefficient.html
    """
    if k < 0 or n < k:
        return 0
    if 2 * k > n:
        k = n - k
    b = 1
    if k > 0:
        for i in range(k):
            b = (b * (n - i)) // (i + 1)
    return b

# based on GEN.tar.gz / ksubsetlex.c / BinCoef() -- http://www.math.mtu.edu/~kreher/cages/Src.html

##### --------------------------------------------------------------------

def bit_count_32(i):
    """
    Number of bits set to 1 in the given integer(s) `i`.
    Vectorized implementation of the 'Hamming Weight'/'popcount'/'sideways addition' algorithm.

    Based on:
    * https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    * http://stackoverflow.com/a/109025
    * http://stackoverflow.com/a/407758
    """
    #assert 0 <= i < 0x100000000  # i in [0, 2**32-1]
    #assert i.dtype == np.uint32 or i.dtype == np.int32
    i = np.uint32(i)

    i = i - ((i >> np.uint32(1)) & np.uint32(0x55555555))
    i = (i & np.uint32(0x33333333)) + ((i >> np.uint32(2)) & np.uint32(0x33333333))
    return np.multiply((i + (i >> np.uint32(4)) & np.uint32(0xF0F0F0F)), np.uint32(0x1010101)) >> np.uint32(24)

##### --------------------------------------------------------------------

def binary_to_sequence(seq, n = None):
    """
    Generates the subsets of a binary encoded sequence.
    """
    if n is None:
        x = np.max(seq)
        n = 0

        while x > 0:
            x >>= 1
            n += 1

    lut = [(bit, np.uint32(1 << bit)) for bit in range(n)]

    for subset in seq:
        yield [bit for bit, x in lut if subset & x]

##### --------------------------------------------------------------------

def colex_uint(n, k):
    """
    Compute the lexicographically next bit permutation.
    See: https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation

    Examples
    --------
    >>> list(colex_uint(6,2))
    [3, 5, 6, 9, 10, 12, 17, 18, 20, 24, 33, 34, 36, 40, 48]
    >>> [np.binary_repr(x, width=4) for x in colex_uint(4,2)]
    ['0011', '0101', '0110', '1001', '1010', '1100']
    """
    v = sum(1 << i for i in range(k))
    yield v
    for _ in range(bin_coef(n,k) - 1):
        t = (v | (v - 1)) + 1
        t = t | ((((t & -t) // (v & -v)) >> 1) - 1)
        yield t
        v = t

##### --------------------------------------------------------------------

def make_scenario(n, k):
    """
    Generates a scenario in binary representation.
    It returns all scenes with t set bits (k < t <= n) and the colexicographic
    sequence which equal the scenes with exactly k set bits.
    """

    scn = np.arange(2**n, dtype=np.uint32)
    bitcount = bit_count_32(scn)

    scenes = scn[bitcount > k]
    sequence = scn[bitcount == k]

    return scenes, sequence
