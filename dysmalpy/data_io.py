# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Basic data IO functions

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

## Standard library
import logging

# Third party imports
import os
import dill as _pickle
import copy

__all__ = ['ensure_dir', 'load_pickle', 'dump_pickle']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')




def ensure_dir(dir):
    """ Short function to ensure dir is a directory; if not, make the directory."""
    if not os.path.exists(dir):
        logger.info( "Making path="+dir)
        os.makedirs(dir)
    return None


def load_pickle(filename):
    """ Small wrapper function to load a pickled structure """
    with open(filename, 'rb') as f:
        data = copy.deepcopy(_pickle.load(f))
    return data


def dump_pickle(data, filename=None, overwrite=False):
    """ Small wrapper function to pickle a structure """
    # Check for existing file:
    if (not overwrite) and (filename is not None):
        if os.path.isfile(filename):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
            return None

    with open(filename, 'wb') as f:
        _pickle.dump(data, f )
    return None
