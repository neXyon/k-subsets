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
from .algorithms import bits_on_count__32

def score_sequence(seq, n=None, k=None):
    """
    Scores a sequence according how quickly it discovers scenes.
    The resulting score is the average number of subsets in the sequence
    necessary to detect a scene.
    """
    seq = list(seq)

    if n is None:
        n = np.max(seq) + 1

    if k is None:
        k = len(seq[0])

    # for each possible set element, get an integer
    # representation that contains a single 1 bit
    elem_bit = np.left_shift(1, np.arange(n, dtype=np.uint32))
    # encode each subset into a single integer
    seq = elem_bit[seq,].reshape(-1,k).sum(axis=1)

    bin_size = 2 ** 28

    scene_count = 0
    ttd__sum = 0

    for start in range(0, 2**n, bin_size):
        scenes = np.arange(start, min([start + bin_size, 2 ** n]), dtype=np.uint32)
        nr_true = bits_on_count__32(scenes).astype(np.uint8)
        # keep only solvable scenes (number of True elements >= k)
        mask = nr_true >= k
        scenes = scenes[mask]
        nr_true = nr_true[mask]

        scene_count += len(scenes)

        # Time to discovery: cumulative sum + amount of added items.
        # Each vector position corresponds to a number of true
        # bits/elements in the matched scenes
        ttd = 0

        seq_iter = seq

        for subset in seq_iter:
            ttd += 1

            # create Boolean mask of scenes matched by the current subset
            matched = bits_on_count__32(np.bitwise_and(subset, scenes)) == k
            # create histogram of scenes matched for each number of true bits
            nr_matched = np.sum(matched)

            ttd__sum += ttd * nr_matched

            # discard matched scenes
            matched = ~matched
            scenes = scenes[matched]
            nr_true = nr_true[matched]

        if len(scenes) > 0:
            raise ValueError('Sequence not complete.')

    # calculate the average score
    return ttd__sum / scene_count

##### --------------------------------------------------------------------

def validate_sequence(sequence, n, k):
    """
    Validates a sequence to check if the subsets have the correct length and no duplicates.
    """
    validated = []

    for subset in sequence:
        if not (k == len(set(subset)) and max(subset) < n and min(subset) >= 0):
            print('Invalid subset detected: {0}.'.format(subset))
        else:
            validated.append(subset)

    return validated
