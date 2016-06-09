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
import numpy as np

from .util import bin_coef

# --------------------------------------------------------------------
# Implementation of algorithms from Sec. 2.3 (k-Element subsets) of:
#
#    Kreher, D. L., & Stinson, D. R. (1998). Combinatorial algorithms:
#    generation, enumeration, and search (Vol. 7). CRC press.
#    http://www.math.mtu.edu/~kreher/cages.html
# --------------------------------------------------------------------

##### --------------------------------------------------------------------
##### ---------------------------------- 2.3.1 - Lexicographic ordering

def lex_rank(n, k, T):
    """
    Returns `r`, the rank of the `k`-subset `T`, drawn from a total
    of `n` elements, assuming a lexicographic ordering.

    * T must be in increasing order;
    * assumes subsets take values from {0, 1, ..., n - 1}.

    Implements Algorithm 2.7 from:
    Kreher, D. L., & Stinson, D. R. (1998). Combinatorial algorithms:
    generation, enumeration, and search (Vol. 7). CRC press.
    http://www.math.mtu.edu/~kreher/cages.html
    """
    r = 0
    for i in range(k):
        lo = 1 if i==0 else (T[i - 1] + 2)
        hi = T[i]
        for j in range(lo, hi + 1):
            r += bin_coef(n - j, k - (i + 1))
    return r

# based on GEN.tar.gz / ksubsetlex.c / kSubsetLexRank() -- http://www.math.mtu.edu/~kreher/cages/Src.html
# Changes:
# * now assumes T subsets are composed of values from {0, 1, ..., n - 1}
# * Insertion of 0 value at T's head (`T = [0] + T`) no longer required
# * `if lo <= hi` unnecessary; range won't iterate unless condition holds


def lex_unrank(n, k, r):
    """
    Returns the `r`-th subset of `k` elements, drawn from a total
    of `n` elements, assuming a lexicographic ordering.

    Implements Algorithm 2.8 from:
    Kreher, D. L., & Stinson, D. R. (1998). Combinatorial algorithms:
    generation, enumeration, and search (Vol. 7). CRC press.
    http://www.math.mtu.edu/~kreher/cages.html

    Example
    -------
    >>> lex_unrank(10, 2, 44)
    [8, 9]
    """
#    assert k <= n, "Expected k <= n, got: n = %d, k = %d." % (n,k)
#    assert r < bin_coef(n, k), "Expected r in {0, ..., C(%d,%d)-1=%d},"\
#        " got %d." % (n, k, bin_coef(n,k) - 1, r)

    T = []
    x = 1
    for i in range(1, k + 1):
        y = bin_coef(n - x, k - i)
        while y <= r:
            r = r - y
            x = x + 1
            y = bin_coef(n - x, k - i)
        T.append(x - 1)
        x = x + 1

    return T

# based on GEN.tar.gz / ksubsetlex.c / kSubsetLexUnrank() -- http://www.math.mtu.edu/~kreher/cages/Src.html


def lex_iter(n, k):
    for s in combinations(range(n), k):
        yield list(s)




##### --------------------------------------------------------------------
##### ---------------------------------- 2.3.2 Co-lex ordering

def colex_rank(n, k, T):
    """
    Algorithm 2.9

    returns r, the rank of the ksubset T
    """
    r = 0
    for i in range(k):
        r += bin_coef(T[i], i + 1)
    return r

# based on GEN.tar.gz / ksubsetcolex.c / kSubsetColexRank -- http://www.math.mtu.edu/~kreher/cages/Src.html
# Changes:
# * now expects T subsets to be composed of values from {0, 1, ..., n - 1}
# * changed from 1- into 0-based indexing
# * T is now expected to be in increasing element order


def colex_unrank(n, k, r):
    """
    Algorithm 2.10

    returns T, the ksubset of rank r
    """
    T = [0] * k
    x = n
    for i in range(k):
        while bin_coef(x, k - i) > r:
            x -= 1
        T[k - i - 1] = x
        r -= bin_coef(x, k - i)
    return T

# based on GEN.tar.gz / ksubsetcolex.c / kSubsetColexUnrank -- http://www.math.mtu.edu/~kreher/cages/Src.html
# Changes:
# * now produces T subsets composed of values from {0, 1, ..., n - 1}
# * changed from 1- into 0-based indexing
# * T is now produced in increasing element order


def colex_iter(n, k):
    for r in range(bin_coef(n, k)):
        yield colex_unrank(n, k, r)




##### --------------------------------------------------------------------
##### ---------------------------------- 2.3.3 - Minimal change ordering (a.k.a.: revolving door ordering)

def revdoor_rank(_, k, T):
    """
    Algorithm 2.11

    returns r, the rank of ksubset T
    """
    r = 0
    s = 1
    for i in range(k-1, -1, -1):
        r = r + bin_coef(T[i] + 1, i+1) * s
        s = -s
    if (k % 2) == 1:
        r = r - 1
    return r

# based on GEN.tar.gz / revdoor.c / kSubsetRevDoorRank() -- http://www.math.mtu.edu/~kreher/cages/Src.html
# Changes:
# * now assumes T subsets are composed of values from {0, 1, ..., n - 1}


def revdoor_unrank(n, k, r):
    """
    Algorithm 2.12

    returns T, the ksubset of rank r
    """
    T = [0] * k
    x = n
    for i in range(k-1, -1, -1):
        y = bin_coef(x, i + 1)
        while y > r:
            x = x - 1
            y = bin_coef(x, i + 1)
        T[i] = x
        r = bin_coef(x + 1, i + 1) - r - 1
    return T

# based on GEN.tar.gz / revdoor.c / kSubsetRevDoorUnrank() -- http://www.math.mtu.edu/~kreher/cages/Src.html
# Changes:
# * now produces T subsets composed of values from {0, 1, ..., n - 1}


def revdoor_successor(n, k, T):
    """
    Algorithm 2.13

    replaces T by its successor
    note that the last k-subset is replaced by the zeroth one
    """
    T = list(T) + [n+1]

    j = 0
    while j < k and T[j] == j:
        j = j + 1

    if (k - j) % 2 == 0:
        if j == 0:
            T[0] = T[0] - 1
        else:
            # j > 0
            T[j - 1] = j
            T[j - 2] = j - 1
    else:
        # k - j is odd
        if T[j + 1] != T[j] + 1:
            T[j - 1] = T[j]
            T[j] = T[j] + 1
        else:
            T[j + 1] = T[j]
            T[j] = j

    return T[:-1]

# based on GEN.tar.gz / revdoor.c / kSubsetRevDoorSuccessor() -- http://www.math.mtu.edu/~kreher/cages/Src.html
# Changes:
# * changed from 1- into 0-based indexing over T
# * now assumes T subsets are composed of values from {0, 1, ..., n - 1}


def revdoor_iter(n, k):
    """
    Minimal change ordering (a.k.a.: revolving door ordering).

    Generates a sequence of k-element subsets of S = {0,...,n-1} in which any
    two consecutive subsets differ by at most one element (consecutive subsets
    differ by the deletion of one element and the insertion of another).
    """
    T = range(k)
    yield T

    for rank in range(1, bin_coef(n, k)):
        T = revdoor_successor(n, k, T)
        yield T




##### --------------------------------------------------------------------

class lex(object):

    def __init__(self, n, k):
        self.n = n
        self.k = k

    def __iter__(self):
        return lex_iter(self.n, self.k)

    def rank(self, T):
        return lex_rank(self.n, self.k, T)

    def unrank(self, r):
        return lex_unrank(self.n, self.k, r)



class colex(object):

    def __init__(self, n, k):
        self.n = n
        self.k = k

    def __iter__(self):
        return colex_iter(self.n, self.k)

    def rank(self, T):
        return colex_rank(self.n, self.k, T)

    def unrank(self, r):
        return colex_unrank(self.n, self.k, r)



class revdoor(object):

    def __init__(self, n, k):
        self.n = n
        self.k = k

    def __iter__(self):
        return revdoor_iter(self.n, self.k)

    def rank(self, T):
        return revdoor_rank(self.n, self.k, T)

    def unrank(self, r):
        return revdoor_unrank(self.n, self.k, r)



# --------------------------------------------------------------------
# Implementation of algorithms of:
#
#   Arndt, J. (2010). Matters Computational: ideas, algorithms,
#   source code. Springer Science & Business Media.
#
# Implementations are based on
# http://www.jjj.de/fxt/demo/comb/index.html
# --------------------------------------------------------------------


##### --------------------------------------------------------------------

def emk(n, k):
    """
    Eades-McKay sequence.

    A strong minimal-change order sequence.
    """
    x = np.arange(k + 1, dtype=np.uint32)
    x[k] = n
    s = x.copy()
    a = x[:-1].copy()

    yield x[:-1].copy()

    broken = True
    while broken:
        broken = False

        for j in range(k - 1, -1, -1):
            sj = s[j]
            m = x[j + 1] - sj - 1

            if m != 0:
                u = x[j] - sj

                if (j & 1) == 0:
                    u += 1
                    if u > m:
                        u = 0
                else:
                    u -= 1
                    if u < 0:
                        u = m

                u += sj

                if u != a[j]:
                    x[j] = u
                    s[j + 1] = u + 1
                    yield x[:-1].copy()
                    broken = True
                    break

            a[j] = x[j]

##### --------------------------------------------------------------------

def next_endo(x, m):
    """
    Endo step.
    """
    if x & 1:
        x += 2
        if x > m:
            x = m - (m & 1)
    else:
        x = 1 if x == 0 else x - 2

    return x

def next_enup(x, m):
    """
    Enup step.
    """
    if x & 1:
        x = 0 if x == 1 else x - 2
    else:
        x += 2
        if x > m:
            x = m - (not (m & 1))

    return x

def enup(n, k, first=next_enup, second=next_endo):
    """
    Strong minimal-change order for combinations via enup (endo) steps.
    """
    x = np.arange(k + 1, dtype=np.uint32)
    x[k] = n
    s = x.copy()
    a = x[:-1].copy()

    yield x[:-1].copy()

    broken = True
    while broken:
        broken = False

        for j in range(k - 1, -1, -1):
            sj = s[j]
            m = x[j + 1] - sj - 1

            if m != 0:
                u = x[j] - sj

                if (sj & 1) == 0:
                    u = first(u, m)
                else:
                    u = second(u, m)

                u += sj

                if u != a[j]:
                    x[j] = u
                    s[j + 1] = u + 1
                    yield x[:-1].copy()
                    broken = True
                    break

            a[j] = x[j]

def endo(n, k):
    """
    Strong minimal-change order for combinations (Chase's sequence) via endo steps.
    """
    for x in enup(n, k, first=next_endo, second=next_enup):
        yield x

##### --------------------------------------------------------------------

def pref(n, k):
    """
    Combinations via prefix shifts ("cool-lex" order).
    """
    s = n - k
    t = k
    n = s + t
    b = np.zeros(n, dtype=np.bool)
    b[:k] = 1
    x = 0
    y = 0

    nums = np.arange(n)

    yield nums[b]

    while x < n - 1:
        if x == 0:
            x = 1
            b[t] = True
            b[0] = False
            yield nums[b]
        else:
            b[x] = False
            b[y] = True
            x += 1
            y += 1
            if b[x] == 0:
                b[x] = True
                b[0] = False
                if y > 1:
                    x = 1
                y = 0
            yield nums[b]

##### --------------------------------------------------------------------

def lam(n, k):
    """
    Minimal-change order by Lam and Soicher.

    See:
    Lam, Clement WH, and Leonard H. Soicher.
    "Three new combination algorithms with the minimal change property."
    Communications of the ACM 25.8 (1982): 555-559.
    """
    a = np.arange(k + 2, dtype=np.uint32)
    t = a + 1

    top = 0

    if k % 2 == 0:
        a[k + 1] = n + 1
        a[k] = k
        if k < n:
            top = k
    else:
        a[k] = n
        if k<n:
            top = k - 1


    a[1] = 1
    t[k] = 0

    yield a[1:-1] - 1

    while top != 0:
        if top == 2:
            top = t[2]
            t[2] = 3
            while True:
                a[1] = a[2]
                a[2] += 1
                yield a[1:-1] - 1
                while True:
                    a[1] -= 1
                    yield a[1:-1] - 1
                    if a[1] == 1:
                        break
                if a[2] == a[3] - 1:
                    break
        else:
            if top % 2 == 0:
                a[top - 1] = a[top]
                a[top] += 1
                if a[top] == a[top + 1] - 1:
                    t[top - 1] = t[top]
                    t[top] = top + 1
                top -= 2
            else:
                a[top] -= 1
                if a[top] > top:
                    top -= 1
                    a[top] = top
                else:
                    a[top - 1] = top - 1
                    i = top
                    top = t[top]
                    t[i] = i + 1
            yield a[1:-1] - 1
