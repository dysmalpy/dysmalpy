# Script to plot kin

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys
from contextlib import contextmanager

import copy

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
    import dysmalpy_fit_single_1D
    import dysmalpy_fit_single_2D
except:
    from . import utils_io
    from . import dysmalpy_fit_single_1D
    from . import dysmalpy_fit_single_2D



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
    
    if ('moment_calc' in params.keys()):
        moment_calc = params['moment_calc']
    else:
        moment_calc = False
        
    #
    if ('partial_weight' in params.keys()):
        partial_weight = params['partial_weight']
    else:
        # Preserve previous default behavior
        partial_weight = False
    
    
    plotting.plot_rotcurve_components(gal=gal, 
                overwrite=overwrite, overwrite_curve_files=overwrite_curve_files, 
                outpath = outpath,
                profile1d_type = params['profile1d_type'], 
                oversample=params['oversample'], oversize=params['oversize'], 
                aperture_radius=params['aperture_radius'],
                moment=moment_calc,
                partial_weight=partial_weight)
                
                
    return None


#
def plot_results_multid(param_filename=None, data=None, fit_ndim=None,
    remove_shift=False,
    show_1d_apers=False):
    # Read in the parameters from param_filename:
    params = utils_io.read_fitting_params(fname=param_filename)
    
    # Setup some paths:
    outdir = utils_io.ensure_path_trailing_slash(params['outdir'])
    params['outdir'] = outdir
    
    if fit_ndim == 2:
        gal, fit_dict = dysmalpy_fit_single_2D.setup_single_object_2D(params=params, data=data)
    elif fit_ndim == 1:
        gal, fit_dict = dysmalpy_fit_single_1D.setup_single_object_1D(params=params, data=data)
    
    
    # Reload the best-fit:
    if fit_dict['fit_method'] == 'mcmc':
        gal, results = fitting.reload_all_fitting(filename_galmodel=fit_dict['f_model'], 
                    filename_mcmc_results=fit_dict['f_mcmc_results'])
    elif fit_dict['fit_method'] == 'mpfit':
        gal, results = fitting.reload_all_fitting_mpfit(filename_galmodel=fit_dict['f_model'], 
                    filename_results=fit_dict['f_results'])
    
    # Load the other data:
    if fit_ndim == 2:
        if 'fdata_1d_mask' in params.keys():
            fdata_mask = params['fdata_1d_mask']
        else:
            fdata_mask = None
        params1d = copy.deepcopy(params)
        params1d['data_inst_corr'] = params1d['data_inst_corr_1d']
        data1d = utils_io.load_single_object_1D_data(fdata=params['fdata_1d'], fdata_mask=fdata_mask, params=params1d)
        data1d.filename_velocity = params['fdata_1d']
        
        if (params['profile1d_type'] != 'circ_ap_pv') & (params['profile1d_type'] != 'single_pix_pv'):
            data_orig = copy.deepcopy(gal.data)
            gal.data = data1d
            data1d.apertures = utils_io.setup_basic_aperture_types(gal=gal, params=params)
            # Reset:
            gal.data = data_orig
        
        
        gal.data1d = data1d
    elif fit_ndim == 1:
        data2d = utils_io.load_single_object_2D_data(params=params, skip_crop=True)
        gal.data2d = data2d
    
    # Plot:
    plotting.plot_model_multid(gal, theta=results.bestfit_parameters, 
            fitdispersion=fit_dict['fitdispersion'], 
            oversample=fit_dict['oversample'],fileout=fit_dict['f_plot_bestfit_multid'],
            show_1d_apers=show_1d_apers, remove_shift=remove_shift)
            
    
    return None
    