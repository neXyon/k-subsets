#!/usr/bin/python
# -*- coding: utf8 -*-
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Luis F. Simoes
# Joerg H. Mueller

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_sequence(sequence):
    """
    Plots a sequence as a grid.

    The grid contains as many rows as subsets in the sequence and as many
    columns as the maximum number in the subsets.
    A grid cell is then plotted filled if the corresponding element is part of
    the subset.
    """
    sequence = list(sequence)

    table = np.ones((len(sequence), np.max(sequence) + 1), dtype=np.int)

    for i, ind in enumerate(sequence):
        table[i, ind] = 0

    fig = plt.figure(figsize=np.array(table.shape[::-1]) / 8, frameon=True, linewidth=1)
    fig.subplots_adjust(0.005, 0.002, 0.995, 0.998)
    ax = fig.add_subplot(111, frameon=True)
    plt.imshow(table, interpolation='none')
    ax.grid(True, color=[0, 0, 0], linewidth=1)
    ax.set_xticks(np.arange(table.shape[1] + 1) - 0.5)
    ax.set_yticks(np.arange(table.shape[0] + 1) - 0.5)

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

    return fig
