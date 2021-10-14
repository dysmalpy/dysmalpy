# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Utility calculations for models, shared between different model types

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging
import copy

# Third party imports
import numpy as np


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')

np.warnings.filterwarnings('ignore')


def get_geom_phi_rad_polar(x, y):
    """
    Calculate polar angle phi from x, y.
    """
    R = np.sqrt(x ** 2 + y ** 2)
    # Assume ring is in midplane
    phi_geom_rad = np.arcsin(y/R)
    sh_x = np.shape(x)
    if len(sh_x) > 0:
        # Array-like
        phi_geom_rad[x < 0] = np.pi - np.arcsin(y[x < 0]/R[x < 0])
        # Handle position = 0 case:
        phi_geom_rad[R == 0] = R[R==0] * 0.
    else:
        # Float
        if x < 0.:
            phi_geom_rad = np.pi - np.arcsin(y/R)
        elif x == 0.:
            # Handle position = 0 case:
            phi_geom_rad = 0.

    return phi_geom_rad

def replace_values_by_refarr(arr, ref_arr, excise_val, subs_val):
    """
    Excise (presumably NaN) values from an array,
    by replacing all entries where ref_arr == excise_value with subs_val, or
       arr[ref_arr==excise_val] = subs_val
    """
    arr_out = copy.deepcopy(arr)

    if len(np.shape(ref_arr)) == 0:
        if ref_arr == excise_val:
            arr_out = subs_val
    else:
        arr_out[ref_arr==excise_val] = subs_val

    return arr_out
