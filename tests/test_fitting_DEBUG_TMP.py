# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Testing of DYSMALPY fitting calculations
#    + Primarily using the FITTING_WRAPPER functionality, as a shortcut

import os
import shutil

import math

import numpy as np

from dysmalpy.fitting_wrappers import dysmalpy_fit_single
from dysmalpy.fitting_wrappers import utils_io as fw_utils_io
from dysmalpy import fitting, config


# TESTING DIRECTORY
path = os.path.abspath(__file__)
_dir_tests = os.path.dirname(path) + '/'
_dir_tests_data = _dir_tests+'test_data/'

skip_fits = False   # normal
#skip_fits = True    # DEBUGGING


dir = '/Users/sedona/software_public/dysmalpy/tests/test_data/PYTEST_OUTPUT/GS4_43501_2D_out_mpfit_BRANCH_BROKEN/'


def read_params(param_filename=None):
    param_filename_full=_dir_tests_data+param_filename
    params = fw_utils_io.read_fitting_params(fname=param_filename_full)
    return params

def remake_cube_alt_branch():
    param_filename = 'fitting_2D_mpfit.params'
    params = read_params(param_filename=param_filename)

    fgal = dir+'GS4_43501_galaxy_model.pickle'
    fmpfit = dir+'GS4_43501_mpfit_results.pickle'

    gal, results = fitting.reload_all_fitting(filename_galmodel=fgal,
                        filename_results=fmpfit, fit_method='mpfit')

    config_c_m_data = config.Config_create_model_data(**params)
    config_sim_cube = config.Config_simulate_cube(**params)
    kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}


    kwargs_all = {**kwargs_galmodel, **params}

    gal.create_model_data(**kwargs_all)

    f_cube = dir+'cube_3D_branch_broken.fits'
    overwrite=True
    gal.model_cube.data.write(f_cube, overwrite=overwrite)

    return gal



###
