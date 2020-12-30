# Script to fit single object in 2D with Dysmalpy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import platform
from contextlib import contextmanager
import sys
import shutil

import matplotlib
matplotlib.use('agg')

from dysmalpy import galaxy
from dysmalpy import models
from dysmalpy import fitting
from dysmalpy import instrument
from dysmalpy import parameters
from dysmalpy import plotting
from dysmalpy import config

from dysmalpy.instrument import DoubleBeam, Moffat, GaussianBeam

import copy
import numpy as np
import astropy.units as u

try:
    import utils_io
    from plotting import plot_bundle_2D
    from dysmalpy_fit_single import dysmalpy_fit_single
except ImportError:
    from . import utils_io
    from .plotting import plot_bundle_2D
    from .dysmalpy_fit_single import dysmalpy_fit_single

# Backwards compatibility
def dysmalpy_fit_single_2D(param_filename=None, data=None, datadir=None,
             outdir=None, plot_type='pdf', overwrite=None):
    return dysmalpy_fit_single(param_filename=param_filename, data=data, datadir=datadir,
                outdir=outdir, plot_type=plot_type, overwrite=overwrite)


def dysmalpy_reanalyze_single_2D(param_filename=None, data=None, datadir=None, outdir=None, plot_type='pdf'):

    # Read in the parameters from param_filename:
    params = utils_io.read_fitting_params(fname=param_filename)

    # OVERRIDE SETTINGS FROM PARAMS FILE if passed directly -- eg from an example Jupyter NB:
    if datadir is not None:
        params['datadir'] = datadir
    if outdir is not None:
        params['outdir'] = outdir

    # Setup some paths:
    outdir = utils_io.ensure_path_trailing_slash(params['outdir'])
    params['outdir'] = outdir

    fitting.ensure_dir(params['outdir'])

    if 'plot_type' not in params.keys():
        params['plot_type'] = plot_type
    else:
        plot_type = params['plot_type']

    # Check if fitting already done:
    if params['fit_method'] == 'mcmc':
        # Copy paramfile that is OS independent
        shutil.copy(param_filename, outdir)

        # Reload the results, etc
        #######################
        # Reload stuff
        galtmp, fit_dict = utils_io.setup_single_object_2D(params=params, data=data)

        config_c_m_data = config.Config_create_model_data(**fit_dict)
        config_sim_cube = config.Config_simulate_cube(**fit_dict)
        kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

        try:
            gal, results = fitting.reload_all_fitting(filename_galmodel=fit_dict['f_model'],
                                    filename_results=fit_dict['f_mcmc_results'],
                                    fit_method=params['fit_method'])
        except:
            # Something went wrong after sampler was saved
            gal = copy.deepcopy(galtmp)

            # +++++++++++++++++++++++
            # Setup for oversampled_chisq:
            gal = fitting.setup_oversampled_chisq(gal)
            # +++++++++++++++++++++++


            sampler_dict = fitting.load_pickle(fit_dict['f_sampler'])
            results = fitting.MCMCResults(model=gal.model, sampler=sampler_dict,
                                      f_plot_trace_burnin = fit_dict['f_plot_trace_burnin'],
                                      f_plot_trace = fit_dict['f_plot_trace'],
                                      f_sampler = fit_dict['f_sampler'],
                                      f_plot_param_corner = fit_dict['f_plot_param_corner'],
                                      f_plot_bestfit = fit_dict['f_plot_bestfit'],
                                      f_results= fit_dict['f_mcmc_results'],
                                      f_chain_ascii = fit_dict['f_chain_ascii'])
            if fit_dict['oversampled_chisq']:
                results.oversample_factor_chisq = gal.data.oversample_factor_chisq

        # Do all analysis, plotting, saving:

        fit_dict['overwrite'] = overwrite
        fit_dict['plot_type'] = plot_type

        kwargs_all = {**kwargs_galmodel, **fit_dict}

        results.analyze_plot_save_results(gal, **kwargs_all)

        # Reload fitting stuff to get the updated gal object
        gal, results = fitting.reload_all_fitting(filename_galmodel=fit_dict['f_model'],
                                    filename_results=fit_dict['f_mcmc_results'],
                                    fit_method=params['fit_method'])

        # Save results
        utils_io.save_results_ascii_files(fit_results=results, gal=gal, params=params,
                        overwrite=overwrite)

    elif params['fit_method'] == 'mpfit':
        galtmp, fit_dict = utils_io.setup_single_object_2D(params=params, data=data)

        # reload results:
        gal, results = fitting.reload_all_fitting(filename_galmodel=fit_dict['f_model'],
                                    filename_results=fit_dict['f_results'],
                                    fit_method=params['fit_method'])
        # Don't reanalyze anything...
    else:
        raise ValueError(
            '{} not accepted as a fitting method. Please only use "mcmc" or "mpfit"'.format(
                params['fit_method']))


    # Plot multid, if enabled:
    plot_bundle_2D(params=params, param_filename=param_filename, plot_type=plot_type, overwrite=overwrite)

    return None




if __name__ == "__main__":

    param_filename = sys.argv[1]

    try:
        if sys.argv[2].strip().lower() != 'reanalyze':
            datadir = sys.argv[2]
        else:
            datadir = None
    except:
        datadir = None

    try:
        if sys.argv[2].strip().lower() == 'reanalyze':
            reanalyze = True
        else:
            reanalyze = False
    except:
        reanalyze = False

    if reanalyze:
        dysmalpy_reanalyze_single_2D(param_filename=param_filename, datadir=datadir)
    else:
        dysmalpy_fit_single_2D(param_filename=param_filename, datadir=datadir)
