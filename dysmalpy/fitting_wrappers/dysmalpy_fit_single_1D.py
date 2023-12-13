# Script to fit single object in 1D with Dysmalpy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys, shutil
from dysmalpy import data_io

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

try:
    import utils_io
    from plotting import plot_1D_rotcurve_components
    from dysmalpy_fit_single import dysmalpy_fit_single
except ImportError:
    from . import utils_io
    from .plotting import plot_1D_rotcurve_components
    from .dysmalpy_fit_single import dysmalpy_fit_single


# Backwards compatibility
def dysmalpy_fit_single_1D(param_filename=None, datadir=None,
             outdir=None, plot_type='pdf', overwrite=None):
     return dysmalpy_fit_single(param_filename=param_filename, datadir=datadir,
                 outdir=outdir, plot_type=plot_type, overwrite=overwrite)


def dysmalpy_reanalyze_single_1D(param_filename=None,
            datadir=None, outdir=None, plot_type='pdf', overwrite=True):


    # Read in the parameters from param_filename:
    params = utils_io.read_fitting_params(fname=param_filename)

    # OVERRIDE SETTINGS FROM PARAMS FILE if passed directly -- eg from an example Jupyter NB:
    if datadir is not None:
        params['datadir'] = datadir
    if outdir is not None:
        params['outdir'] = outdir

    # Setup some paths:
    outdir = data_io.ensure_path_trailing_slash(params['outdir'])
    params['outdir'] = outdir

    data_io.ensure_dir(params['outdir'])

    if 'plot_type' not in params.keys():
        params['plot_type'] = plot_type

    params['overwrite'] = overwrite

    # Check if fitting already done:
    if params['fit_method'] == 'mcmc':
        # Copy paramfile that is OS independent
        shutil.copy(param_filename, outdir)

        # Reload the results, etc
        #######################
        # Reload stuff
        gal, output_options = utils_io.setup_single_galaxy(params=params)

        gal, results = fitting.reload_all_fitting(filename_galmodel=output_options.f_model,
                                    filename_results=output_options.f_results,
                                    fit_method=params['fit_method'])

        # Do all analysis, plotting, saving:
        results.analyze_plot_save_results(gal, output_options=output_options)


    elif params['fit_method'] == 'mpfit':
        # Reload the results, etc
        #######################
        # Reload stuff
        gal, output_options = utils_io.setup_single_galaxy(params=params)

        # reload results:
        gal, results = fitting.reload_all_fitting(filename_galmodel=output_options.f_model,
                                    filename_results=output_options.f_results,
                                    fit_method=params['fit_method'])
        # Don't reanalyze anything...
    else:
        raise ValueError(
            '{} not accepted as a fitting method. Please only use "mcmc" or "mpfit"'.format(
                params['fit_method']))

    # Make component plot:
    if output_options.do_plotting:
        plot_1D_rotcurve_components(gal=gal, output_options=output_options)

    return None




if __name__ == "__main__":

    param_filename = sys.argv[1]

    # try:
    #     if sys.argv[2].strip().lower() != 'reanalyze':
    #         datadir = sys.argv[2]
    #     else:
    #         datadir = None
    # except:
    #     datadir = None

    # try:
    #     if sys.argv[2].strip().lower() == 'reanalyze':
    #         reanalyze = True
    #     else:
    #         reanalyze = False
    # except:
    #     reanalyze = False

    # if reanalyze:
    #     dysmalpy_reanalyze_single_1D(param_filename=param_filename, datadir=datadir)
    # else:
    #     dysmalpy_fit_single_1D(param_filename=param_filename, datadir=datadir)



    try:
        datadir = sys.argv[2]
    except:
        datadir = None

    dysmalpy_fit_single_1D(param_filename=param_filename, datadir=datadir)
