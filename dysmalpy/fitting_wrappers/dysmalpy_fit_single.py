# Script to fit single object with Dysmalpy: get dimension from paramfile data names

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import platform
from contextlib import contextmanager
import sys
import shutil

try:
    import tkinter_io
except ImportError:
    from . import tkinter_io

def dysmalpy_fit_single(param_filename=None, data=None, datadir=None,
            outdir=None, plot_type='pdf', overwrite=None):
    """
    Fit observed kinematics, based on settings / files specified in the parameter file.

    Input:
        param_filename:     Path to parameters file.

    Optional input:
        data:               Galaxy data (`Data1D`/`Data2D`/`Data3D`/`Data0D` instance)
                            Otherwise, loads data based on data filenames in parameters file.

        datadir:            Path to data directory. If set, overrides datadir set in the parameters file.

        outdir:             Path to output directory. If set, overrides outdir set in the parameters file.

        plot_type:          Filetype ending for fitting plots. (e.g., 'pdf', 'png')

        overwrite:          Option to overwrite any pre-existing fititng files.
                            If set, overrides overwrite set in the parameters file.

    Output:
            Saves fitting results to outdir (specifed in call to `dysmalpy_fit_single` or in parameters file).
    """

    # Only load full imports later to speed up usage from command line.
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

    from dysmalpy import fitting
    from dysmalpy import config

    import copy
    import numpy as np

    from dysmalpy.fitting_wrappers.plotting import plot_bundle_1D, plot_bundle_2D
    from dysmalpy.fitting_wrappers import utils_io

    # Get fitting dimension:
    ndim = utils_io.get_ndim_fit_from_paramfile(param_filename=param_filename)


    # Read in the parameters from param_filename:
    params = utils_io.read_fitting_params(fname=param_filename)

    # Check if 'overwrite' is set in the params file.
    # But the direct input from calling the script overrides any setting in the params file.
    if overwrite is None:
        if 'overwrite' in params.keys():
            overwrite = params['overwrite']
        else:
            overwrite = False

    # OVERRIDE SETTINGS FROM PARAMS FILE if passed directly -- eg from an example Jupyter NB:
    if datadir is not None:
        params['datadir'] = datadir
    if outdir is not None:
        params['outdir'] = outdir

    # Setup some paths:

    # Ensure output directory is specified: if relative file path,
    #   EXPLICITLY prepend paramfile path
    outdir = params['outdir']
    outdir = utils_io.ensure_path_trailing_slash(params['outdir'])
    params['outdir'] = outdir
    outdir, params = utils_io.check_outdir_specified(params, outdir, param_filename=param_filename)


    if 'datadir' in params.keys():
        if params['datadir'] is not None:
            datadir = utils_io.ensure_path_trailing_slash(params['datadir'])
            params['datadir'] = datadir

    ####
    # OVERRIDE SETTINGS FROM PARAMS FILE if passed directly -- eg from an example Jupyter NB:
    if plot_type is not None:
        params['plot_type'] = plot_type
    else:
        if 'plot_type' in params.keys():
            plot_type = params['plot_type']
        else:
            params['plot_type'] = plot_type

    # Check if fitting already done:
    if params['fit_method'] == 'mcmc':
        fit_exists = os.path.isfile(outdir+'{}_mcmc_results.pickle'.format(params['galID']))
    elif params['fit_method'] == 'mpfit':
        fit_exists = os.path.isfile(outdir + '{}_mpfit_results.pickle'.format(params['galID']))
    else:
        raise ValueError(
            '{} not accepted as a fitting method. Please only use "mcmc" or "mpfit"'.format(
                params['fit_method']))

    if fit_exists and not (overwrite):
        print('------------------------------------------------------------------')
        print(' Fitting already complete for: {}'.format(params['galID']))
        print('   make new output folder or remove previous fitting files')
        print('------------------------------------------------------------------')
        print(' ')
    else:
        if 'datadir' in params.keys():
            datadir = params['datadir']

        # Check if you can find filename; if not open datadir interface:
        datadir, params = utils_io.check_datadir_specified(params, datadir, ndim=ndim,
                                                        param_filename=param_filename)

        fitting.ensure_dir(outdir)

        # Copy paramfile that is OS independent
        utils_io.preserve_param_file(param_filename, params=params,
                                     datadir=datadir, outdir=outdir)

        # Cleanup if overwriting:
        if fit_exists:
            if params['fit_method'] == 'mcmc':
                os.remove(outdir+'{}_mcmc_results.pickle'.format(params['galID']))

            elif params['fit_method'] == 'mpfit':
                os.remove(outdir + '{}_mpfit_results.pickle'.format(params['galID']))

        #######################
        # Setup
        if ndim == 1:
            gal, fit_dict = utils_io.setup_single_object_1D(params=params, data=data)
        elif ndim == 2:
            gal, fit_dict = utils_io.setup_single_object_2D(params=params, data=data)
        elif ndim == 3:
            gal, fit_dict = utils_io.setup_single_object_3D(params=params, data=data)
        else:
            raise ValueError("ndim={} not recognized!".format(ndim))

        config_c_m_data = config.Config_create_model_data(**fit_dict)
        config_sim_cube = config.Config_simulate_cube(**fit_dict)
        kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

        fit_dict['overwrite'] = overwrite
        fit_dict['plot_type'] = plot_type

        kwargs_all = {**kwargs_galmodel, **fit_dict}

        # Clean up existing log file:
        if fit_dict['f_log'] is not None:
            if os.path.isfile(fit_dict['f_log']):
                os.remove(fit_dict['f_log'])

        # Fit
        if fit_dict['fit_method'] == 'mcmc':
            results = fitting.fit_mcmc(gal, **kwargs_all)

        elif fit_dict['fit_method'] == 'mpfit':
            results = fitting.fit_mpfit(gal, **kwargs_all)

        # Save results
        utils_io.save_results_ascii_files(fit_results=results, gal=gal, params=params,
                        overwrite=overwrite)

        # Make component plot:
        if fit_dict['do_plotting']:
            if ndim == 1:
                plot_bundle_1D(params=params, fit_dict=fit_dict, param_filename=param_filename,
                        plot_type=plot_type,overwrite=overwrite,**kwargs_galmodel)
            elif ndim == 2:
                plot_bundle_2D(params=params, param_filename=param_filename, plot_type=plot_type,
                            overwrite=overwrite)
            elif ndim == 3:
                pass

    return None


if __name__ == "__main__":
    try:
        param_filename = sys.argv[1]
    except:
        param_filename = tkinter_io.get_paramfile_tkinter()

    dysmalpy_fit_single(param_filename=param_filename)
