# Script to fit single object with Dysmalpy: get dimension from paramfile data names

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys

try:
    import tkinter_io
except ImportError:
    from . import tkinter_io

def dysmalpy_fit_single(param_filename=None, datadir=None, outdir=None,
                        data_loader=None, plot_type='pdf', overwrite=None):
    """
    Fit observed kinematics, based on settings / files specified in the parameter file.

    Input:
        param_filename:     Path to parameters file.

    Optional input:
        data_loader:        Custom function to load data. Takes kwargs: params, datadir

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
        
    from dysmalpy import data_io

    from dysmalpy.fitting_wrappers.plotting import plot_1D_rotcurve_components
    from dysmalpy.fitting_wrappers import utils_io


    # Read in the parameters from param_filename:
    # params = utils_io.read_fitting_params(fname=param_filename)
    params = utils_io.read_fitting_params(fname=param_filename)

    # ---------------------------------
    # Check some value validity:
    if params['fit_method'] not in ['mcmc', 'mpfit', 'nested']:
        raise ValueError(
            '{} not accepted as a fitting method. Please only use "mcmc", "nested", or "mpfit"'.format(
                params['fit_method']))
    # ---------------------------------


    # Check if 'overwrite' is set in the params file.
    # But the direct input from calling the script overrides any setting in the params file.
    if overwrite is not None:
        # Overwrite params setting:
        params['overwrite'] = overwrite

    # OVERRIDE SETTINGS FROM PARAMS FILE if passed directly -- eg from an example Jupyter NB:
    if datadir is not None:
        params['datadir'] = datadir
    if outdir is not None:
        params['outdir'] = outdir

    # Setup some paths:

    # Ensure output directory is specified: if relative file path,
    #   EXPLICITLY prepend paramfile path
    outdir = data_io.ensure_path_trailing_slash(params['outdir'])
    params['outdir'] = outdir
    outdir, params = utils_io.check_outdir_specified(params, outdir, param_filename=param_filename)
    params['outdir'] = outdir


    if params['datadir'] is not None:
        datadir = data_io.ensure_path_trailing_slash(params['datadir'])
        params['datadir'] = datadir

    ####
    # OVERRIDE SETTINGS FROM PARAMS FILE if passed directly -- eg from an example Jupyter NB:
    if plot_type is not None:
        params['plot_type'] = plot_type


    # Check if fitting already done:
    fit_exists = os.path.isfile(params['outdir']+'{}_{}_results.pickle'.format(params['galID'],
                                            params['fit_method']))

    if fit_exists and not (params['overwrite']):
        print('------------------------------------------------------------------')
        print(' Fitting already complete for: {}'.format(params['galID']))
        print('   make new output folder or remove previous fitting files')
        print('------------------------------------------------------------------')
        print(' ') 
        
    else:
        # Get fitting dimension of at least 1 obs:
        ndim = utils_io.get_ndim_fit_from_paramfile(0, param_filename=param_filename)

        # Check if you can find filename; if not open datadir interface:
        datadir, params = utils_io.check_datadir_specified(params, params['datadir'], ndim=ndim,
                                                        param_filename=param_filename)
        params['datadir'] = datadir

        data_io.ensure_dir(params['outdir'])

        # Copy paramfile that is OS independent
        utils_io.preserve_param_file(param_filename, params=params,
                                     datadir=params['datadir'],
                                     outdir=params['outdir'])


        # Cleanup if overwriting:
        if fit_exists:
            os.remove(params['outdir']+'{}_{}_results.pickle'.format(params['galID'],params['fit_method']))

        #######################
        # Setup galaxy, output options
        gal, output_options = utils_io.setup_single_galaxy(params=params, 
                        data_loader=data_loader)

        # Clean up existing log file:
        if output_options.f_log is not None:
            if os.path.isfile(output_options.f_log):
                os.remove(output_options.f_log)


        # Setup fitter:
        fitter = utils_io.setup_fitter(params=params)

        # Fit
        results = fitter.fit(gal, output_options)

        # Make component plot:
        if output_options.do_plotting:
            plot_1D_rotcurve_components(output_options=output_options)


        print('------------------------------------------------------------------')
        print(' Dysmalpy {} fitting complete for: {}'.format(fitter.fit_method, gal.name))
        print('   output folder: {}'.format(params['outdir']))
        print('------------------------------------------------------------------')
        print(' ')

    return None


if __name__ == "__main__":
    try:
        param_filename = sys.argv[1]
    except:
        param_filename = tkinter_io.get_paramfile_tkinter()

    dysmalpy_fit_single(param_filename=param_filename)
