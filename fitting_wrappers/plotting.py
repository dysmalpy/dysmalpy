# Script to plot kin

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys
from contextlib import contextmanager


import datetime

import numpy as np
import pandas as pd
import astropy.units as u

from dysmalpy import data_classes
from dysmalpy import parameters
from dysmalpy import fitting
from dysmalpy import galaxy
from dysmalpy import models
from dysmalpy import plotting
from dysmalpy import utils as dysmalpy_utils

from dysmalpy import aperture_classes

import scipy.optimize as scp_opt

from astropy.table import Table

import astropy.io.fits as fits

try:
    import utils_io
except:
    from . import utils_io



#
def plot_curve_components_overview(fname_gal=None, fname_results=None, param_filename=None, 
        overwrite = False, 
        overwrite_curve_files=False, 
        outpath=None):
    
    # Reload the galaxy:
    gal = galaxy.load_galaxy_object(filename=fname_gal)
    # gal, results = fitting.reload_all_fitting(filename_galmodel=fname_gal, 
    #                     filename_mcmc_results=fname_results)
    
    params = utils_io.read_fitting_params(fname=param_filename)
    
    if 'aperture_radius' not in params.keys():
        params['aperture_radius'] = -99.
    
    plotting.plot_rotcurve_components(gal=gal, 
                overwrite=overwrite, overwrite_curve_files=overwrite_curve_files, 
                outpath = outpath,
                profile1d_type = params['profile1d_type'], 
                oversample=params['oversample'], oversize=params['oversize'], 
                aperture_radius=params['aperture_radius'])
                
                
    return None
