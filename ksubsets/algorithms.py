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
from .util import *

def brute_force(n, k):
    """
    Determines the score of all k-subset sequences according to the objective.

    Runs through all possible permutations of k-subset sequences and scores
    them according to the objective.

    Runs for up to n = 5 in a reasonable amount of time. Higher values for
    n are not recommended.
    """

    def _score_sequence(scenes, sequence):
        score = 0

        removed = len(scenes) + len(sequence)

        for i, subset in enumerate(sequence):
            matched = np.bitwise_and(scenes, subset) == subset
            # score it's the (i + 1)th index and the other +1 is for the scene == subset (which is not included in scenes)
            last = removed
            removed = (matched.sum() + 1)

            # if only interested in the number of optimal solutions, the next
            # two lines can skip sequences not having the monotonicity property.
            #if removed > last:
            #    return -1

            score += removed * (i + 1)
            scenes = scenes[~matched]

        return score

    def _perm(n):
        p = np.arange(n + 1) - 1
        p[0] = 0

        yield p[1:]

        while True:
            i = n - 1
            while p[i] > p[i + 1]:
                i -= 1

            if i < 1:
                return

            j = n

            while p[i] > p[j]:
                j -= 1

            p[[i, j]] = p[[j, i]]

            r = n
            s = i + 1

            while r > s:
                p[[r, s]] = p[[s, r]]
                r -= 1
                s += 1

            yield p[1:]

    scenes, sequence = make_scenario(n, k)

    scores = {}

    for permutation in _perm(len(sequence)):
        seq = sequence[permutation]
        score = _score_sequence(scenes, seq)
        if score in scores:
            scores[score] += 1
        else:
            scores[score] = 1

    return scores

##### --------------------------------------------------------------------

def tree_search(n, k):
    """
    Finds one optimal k-subset sequence according to the objective.

    Uses several properties of the objective to prune branches of the tree
    that are guaranteed to contain at best solutions equal to the current
    best.

    Runs for up to n = 6 in a reasonable amount of time. Higher values for
    n are not recommended.
    """

    def _check_group(groups, value):
        # Note: this algorithm does not exclude all possible element permutations!
        #  It's good enough though for the values of n, k that the tree_search can handle.

        new_groups = []
        for group, shift in groups:
            a = (value >> shift) & group
            if a & (a >> 1) != (a >> 1):
                return False

            if a != 0 and a != group:
                lens = bit_count_32([group, a])
                if lens[0] > 1:
                    new_groups.append(((group & (~a)) >> lens[1], shift + lens[1]))
                if lens[1] > 1:
                    new_groups.append((a, shift))
            else:
                new_groups.append((group, shift))

        return new_groups

    scenes, seq = make_scenario(n, k)

    seq_len = len(seq)

    best_seq = [seq.copy()]
    best_score = len(scenes) * seq_len

    removed_scenes = [[] for i in seq]

    stack = []
    group_stack = []
    score_stack = []

    removed = len(scenes) + seq_len
    groups = [((1 << n) - 1, 0)]

    # the first grouping would remove all other elements, so we can just start
    # with [0] instead of range(seq_len)
    for i in [0]:
        stack.append((0, i, False))
        stack.append((0, i, True))
        group_stack.append(groups.copy())
        score_stack.append((0, removed))

    while stack != []:
        start, index, enter = stack.pop()

        seq[start], seq[index] = seq[index], seq[start]

        if enter:
            groups = group_stack.pop()
            score, last_removed = score_stack.pop()

            if start == seq_len - 1:
                score = score + start + 1
                if score < best_score:
                    best_score = score
                    best_seq = [seq.copy()]
                elif score == best_score:
                    best_seq.append(seq.copy())
            else:
                if groups != []:
                    groups = _check_group(groups, seq[start])
                    if groups == False:
                        continue

                matched = np.bitwise_and(scenes, seq[start]) == seq[start]
                # removed scenes are the matched scenes plus the scene that equals the sequence
                removed = matched.sum() + 1

                if removed > last_removed or removed == 1:
                    continue

                # ttd = start + 1
                score += removed * (start + 1)

                remaining_scenes = len(scenes) - removed + 1
                quick_remove = remaining_scenes // (removed - 1)
                last_remove = remaining_scenes % (removed - 1)
                last_remove_start = start + quick_remove

                # an even closer bound, but computationally more expensive (so less performant) would be:
                #if score + (start + seq_len + 2) * (seq_len - start - 1) // 2 + last_remove * (last_remove_start + 1) + quick_remove * (last_remove_start - 4 - start) * (last_remove_start + 2 + start) // 2 > best_score:
                if score + (start + seq_len + 2) * (seq_len - start - 1) // 2 + (start + 1) * (len(scenes) - removed) > best_score:
                    continue

                removed_scenes[start] = scenes[matched]
                scenes = scenes[~matched]

                if len(scenes) == 0:
                    stack.append((seq_len - 1, seq_len - 1, False))
                    stack.append((seq_len - 1, seq_len - 1, True))
                    group_stack.append(groups.copy())
                    score_stack.append((score + np.sum(np.arange(start + 2, seq_len)), 1))
                else:
                    for i in range(start + 1, seq_len):
                        stack.append((start + 1, i, False))
                        stack.append((start + 1, i, True))
                        group_stack.append(groups.copy())
                        score_stack.append((score, removed))

        else: # exit
            scenes = np.array(np.hstack([removed_scenes[start], scenes]), dtype=np.uint32)
            removed_scenes[start] = []

    return best_seq, best_score

##### --------------------------------------------------------------------

def gse(n, k, lex_order = True, keep_order = True):
    """
    Greedy Scene Elimination sequence.

    Iterator over the `k`-element subsets drawn out of `n` elements.
    Produces at each step the subset discovering the highest number
    of scenes considering all undiscovered scenes so far.
    The default base is the lexicographic order and can be changed to
    colexicographic with the parameter `lex_order` set to False.
    Ties are broken with the first element in the remaining sequence
    eliminating the highest number of scenes.
    The `keep_order` parameter ensures that ties resolve in the order
    of the base sequence. If set to False this order is not kept
    similar to an unstable sorting algorithm.
    """
    scenes, seq = make_scenario(n, k)

    if lex_order:
        from itertools import combinations
        seq = [int(np.bitwise_or.reduce(1 << np.array(x))) for x in combinations(range(n), k)]

    seq_len = len(seq)

    # at the beginning every subset removes 2 ** (n - k) scenes (including the subset itself)
    removing = np.ones(seq_len) * (2 ** (n - k) - 1)

    for start in range(seq_len):
        max_i = np.argmax(removing[start:]) + start

        matched = np.bitwise_and(scenes, seq[max_i]) == seq[max_i]

        if max_i != start:
            # the following line preserves the original order
            if keep_order:
                temp = seq[start]
                seq[start] = seq[max_i]
                seq[start + 2:max_i + 1] = seq[start + 1:max_i]
                seq[start + 1] = temp
                temp = removing[start]
                removing[start]  = removing[max_i]
                removing[start + 2:max_i + 1] = removing[start + 1:max_i]
                removing[start + 1] = temp
            else:
                seq[start], seq[max_i] = seq[max_i], seq[start]
                removing[start], removing[max_i] = removing[max_i], removing[start]

        matched_scenes = scenes[matched]
        scenes = scenes[~matched]

        if len(scenes) == 0: # early stopping
            break

        for i in range(start + 1, seq_len):
            new_matched = np.bitwise_and(matched_scenes, seq[i]) == seq[i]
            removing[i] -= np.sum(new_matched)

    return binary_to_sequence(seq, n)

##### --------------------------------------------------------------------

def mis(n, k, keep_order = True):
    """
    Minimally intersecting subsets sequence.

    Iterator over the `k`-element subsets drawn out of `n` elements.
    Produces at each step the subset having the lowest degree of
    overlap with all the subsets produced up to that point (ties
    broken in favour of lowest rank in a lexicographic ordering).
    """
    assert n <= 32, 'Only implemented for n <= 32.'

    # generate all possible subsets, represented as unsigned ints
    sequence = np.array(list(colex_uint(n,k)), dtype=np.uint32)

    # cumulative score by which to rate subsets
    cumul_score = np.zeros(len(sequence), dtype=np.uint64)

    for start in range(len(sequence) - 1):
        # find among the remaining subsets, the index to the one with
        # lowest degree of overlap with all the subsets produced so far
        ix = cumul_score.argmin()
        max_i = ix + start

        subset = sequence[max_i]

        if max_i != start:
            # the following line preserves the original order
            if keep_order:
                temp = sequence[start]
                sequence[start] = sequence[max_i]
                sequence[start + 2:max_i + 1] = sequence[start + 1:max_i]
                sequence[start + 1] = temp
            else:
                sequence[start], sequence[max_i] = sequence[max_i], sequence[start]

        #sequence[ix:-1] = sequence[ix+1:]
        cumul_score[ix:-1] = cumul_score[ix+1:]

        #sequence = sequence[:-1]
        cumul_score = cumul_score[:-1]

        # number of common elements among subset `subs[ix]` and all those in sub
        nr_common = bit_count_32(np.bitwise_and(subset, sequence[start + 1:]))

        # increment the ratings of each subset in `m` by `base` raised
        # to the power of the number of elements in common with the
        # last yielded subset
        cumul_score += (1 << nr_common) - 1

    return binary_to_sequence(sequence, n)

##### --------------------------------------------------------------------

def base_unrank(n, k, base=2, unrank_func=None):
    """
    k-element subsets generation by "Base Unranking"
    """

    from math import log, ceil
    from itertools import product

    # unranking function to use; one of:
    # k_subset_lex_unrank, k_subset_colex_unrank, k_subset_revdoor_unrank
    if unrank_func is None:
        from .ksubsets import revdoor_unrank
        unrank_func = revdoor_unrank

    # get the number of subsets in the sequence
    seq_len = bin_coef(n, k)
    # get the number of digits required in the given numeral system `base`
    # to count up to `seq_len`
    nr_digits = int(ceil(log(seq_len) / log(base)))

    # count from 0 to base**nr_digits - 1, by prioritizing increments to the
    # most significant digits, skipping values greater than `seq_len`, and
    # "unranking" the remaining ones (unranking generates the n-th subset
    # in a given reference sequence, such as lex, colex, or revdoor).
    for s in product(*[ [j*(base**i) for j in range(base)] for i in range(nr_digits) ]):
        rank = sum(s)
        if rank >= seq_len:
            continue
        yield unrank_func(n, k, rank)

##### --------------------------------------------------------------------

# --------------------------------------------------------------------
# Pattern generation algorithm based on:
#
#   Mortari, D., Samaan, M. A., Bruccoleri, C., & Junkins, J. L.
#   (2004). The pyramid star identification technique. Navigation,
#   51(3), 171-183.
# --------------------------------------------------------------------

def pattern_generator(n, k, base=None):
    """
    k-element subsets generation by pattern repetition of some base sequence.

    By default a recursive pattern generator.
    """
    if base is None:
        base = pattern_generator

    if k == 0:
        yield []
        return

    for pattern in base(n - 1, k - 1):
        pattern = [0] + [i+1 for i in pattern]
        yield pattern
        while pattern[-1] != n-1:
            pattern = [i+1 for i in pattern]
            yield pattern
