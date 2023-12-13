# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Basic data IO functions

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

## Standard library
import logging

# Third party imports
import os
import copy

# Pickling: 
import dill as pickle_module

# ---------------------------------------------------------
# dill py<=3.7 -> py>=3.8 + higher hack:
# See https://github.com/uqfoundation/dill/pull/406
pickle_module._dill._reverse_typemap['CodeType'] = pickle_module._dill._create_code
# ---------------------------------------------------------

__all__ = ['ensure_dir', 'ensure_path_trailing_slash', 'load_pickle', 'dump_pickle']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

def ensure_dir(dir):
    """ Short function to ensure dir is a directory; if not, make the directory."""
    if not os.path.exists(dir):
        logger.info( "Making path="+dir)
        os.makedirs(dir)
    return None

def ensure_path_trailing_slash(path):
    if (path[-1] != os.sep):
        path += os.sep
    return path


def load_pickle(filename):
    """ Small wrapper function to load a pickled structure """
    with open(filename, 'rb') as f:
        # data = copy.deepcopy(pickle_module.load(f))
        data = copy.deepcopy(_rename_unpickler(f).load())
    return data


def dump_pickle(data, filename=None, overwrite=False):
    """ Small wrapper function to pickle a structure """
    # Check for existing file:
    if (not overwrite) and (filename is not None):
        if os.path.isfile(filename):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
            return None

    with open(filename, 'wb') as f:
        pickle_module.dump(data, f )
    return None

_rename_modules = {'astropy.cosmology.scalar_inv_efuncs': 'astropy.cosmology.flrw.scalar_inv_efuncs'}

# From stack overflow: https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
class _rename_unpickler(pickle_module.Unpickler):

    def find_class(self, module, name):
        if module in _rename_modules.keys():
            renamed_module = _rename_modules[module]
        else:
            renamed_module = module

        try:
            return super(_rename_unpickler, self).find_class(renamed_module, name)
        except ModuleNotFoundError:
            return super(_rename_unpickler, self).find_class(module, name)