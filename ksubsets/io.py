#!/usr/bin/python
# -*- coding: utf8 -*-
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Luis F. Simoes
# Joerg H. Mueller

import pandas as pd

def save_sequence(sequence, filename):
    """
    Savea a sequence into a csv file.
    """
    sequence = list(sequence)

    df = pd.DataFrame(sequence)
    df.to_csv(filename, header=False, index=False)

def load_sequence(filename):
    """
    Loads a sequence from a csv file.
    """
    sequence = pd.read_csv(filename, header=None)

    return sequence.values
