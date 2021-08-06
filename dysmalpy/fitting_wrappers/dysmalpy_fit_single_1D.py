# Script to fit single object in 1D with Dysmalpy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import platform
from contextlib import contextmanager
import sys
import shutil

import matplotlib
# Check if there is a display for plotting, or if there is an SSH/TMUX session.
# If no display, or if SSH/TMUX, use the matplotlib "agg" backend for plotting.
havedisplay = "DISPLAY" in os.environ
if havedisplay:
    exitval = os.system('python -c "import matplotlib.pyplot as plt; plt.figure()"')
    skipconds = (("SSH_CLIENT" in os.environ) | ("TMUX" in os.environ) | ("SSH_CONNECTION" in os.environ) | (os.environ["TERM"].lower().strip()=='screen') | (exitval != 0))
    havedisplay = not skipconds
if not havedisplay:
    matplotlib.use('agg')

from dysmalpy import galaxy
from dysmalpy import models
from dysmalpy import fitting
from dysmalpy import instrument
from dysmalpy import parameters
from dysmalpy import plotting
from dysmalpy import config

import copy
import numpy as np
import astropy.units as u

try:
    import utils_io
    from plotting import plot_bundle_1D
    from dysmalpy_fit_single import dysmalpy_fit_single
except ImportError:
    from . import utils_io
    from .plotting import plot_bundle_1D
    from .dysmalpy_fit_single import dysmalpy_fit_single


# Backwards compatibility
def dysmalpy_fit_single_1D(param_filename=None, data=None, datadir=None,
             outdir=None, plot_type='pdf', overwrite=None):
     return dysmalpy_fit_single(param_filename=param_filename, data=data, datadir=datadir,
                 outdir=outdir, plot_type=plot_type, overwrite=overwrite)


def dysmalpy_reanalyze_single_1D(param_filename=None, data=None,
            datadir=None, outdir=None, plot_type='pdf', overwrite=True):

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
        galtmp, fit_dict = utils_io.setup_single_object_1D(params=params, data=data)


        config_c_m_data = config.Config_create_model_data(**fit_dict)
        config_sim_cube = config.Config_simulate_cube(**fit_dict)
        kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

        gal, results = fitting.reload_all_fitting(filename_galmodel=fit_dict['f_model'],
                                    filename_results=fit_dict['f_mcmc_results'],
                                    fit_method=params['fit_method'])

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
        galtmp, fit_dict = utils_io.setup_single_object_1D(params=params, data=data)

        config_c_m_data = config.Config_create_model_data(**fit_dict)
        config_sim_cube = config.Config_simulate_cube(**fit_dict)
        kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

        # reload results:
        gal, results = fitting.reload_all_fitting(filename_galmodel=fit_dict['f_model'],
                                    filename_results=fit_dict['f_results'],
                                    fit_method=params['fit_method'])
        # Don't reanalyze anything...
    else:
        raise ValueError(
            '{} not accepted as a fitting method. Please only use "mcmc" or "mpfit"'.format(
                params['fit_method']))

    # Make component plot:
    if fit_dict['do_plotting']:
        plot_bundle_1D(params=params, fit_dict=fit_dict, param_filename=param_filename,
                plot_type=plot_type,overwrite=overwrite,**kwargs_galmodel)


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
        dysmalpy_reanalyze_single_1D(param_filename=param_filename, datadir=datadir)
    else:
        dysmalpy_fit_single_1D(param_filename=param_filename, datadir=datadir)
