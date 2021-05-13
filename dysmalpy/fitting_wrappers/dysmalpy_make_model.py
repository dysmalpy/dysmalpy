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

def dysmalpy_make_model(param_filename=None, outdir=None, overwrite=None):

    # Only load full imports later to speed up usage from command line.
    from dysmalpy import config

    from dysmalpy.fitting_wrappers import utils_io

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
    if outdir is not None:
        params['outdir'] = outdir

    # Setup some paths:

    # Ensure output directory is specified: if relative file path,
    #   EXPLICITLY prepend paramfile path
    outdir = params['outdir']
    outdir = utils_io.ensure_path_trailing_slash(params['outdir'])
    params['outdir'] = outdir
    outdir, params = utils_io.check_outdir_specified(params, outdir, param_filename=param_filename)


    filename_extra = ''
    if 'filename_extra' in params.keys():
        if params['filename_extra'] is not None:
            filename_extra =  params['filename_extra']

    # Check if fitting already done:
    model_exists = os.path.isfile(outdir+'{}{}_model_cube.fits'.format(params['galID'], filename_extra))

    if model_exists and not (overwrite):
        print('------------------------------------------------------------------')
        print(' Model already complete for: {}'.format(params['galID']))
        print('   make new output folder or remove previous model files')
        print('------------------------------------------------------------------')
        print(' ')
    else:
        #fitting.ensure_dir(outdir)
        utils_io.ensure_dir(outdir)

        # Copy paramfile that is OS independent
        utils_io.preserve_param_file(param_filename, params=params, outdir=outdir)

        # Cleanup if overwriting:
        if model_exists:
            os.remove(outdir+'{}{}_model_cube.fits'.format(params['galID'], filename_extra))

        ############################################################################################
        # ------------------------------------------------------------
        # Setup galaxy, instrument, model:
        gal = utils_io.setup_gal_model_base(params=params)

        # Override FOV from the cube shape:
        gal.instrument.fov = [params['fov_npix'], params['fov_npix']]

        f_cube = outdir+'{}{}_model_cube.fits'.format(params['galID'], filename_extra)

        ############################################################################################

        config_c_m_data = config.Config_create_model_data(**params)
        config_sim_cube = config.Config_simulate_cube(**params)
        kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

        params['overwrite'] = overwrite

        # Additional settings:
        kwargs_galmodel['from_data'] = False
        kwargs_galmodel['ndim_final'] = 3


        # Make model
        gal.create_model_data(**kwargs_galmodel)

        # Save cube
        gal.model_cube.data.write(f_cube, overwrite=overwrite)


    return None


if __name__ == "__main__":
    try:
        param_filename = sys.argv[1]
    except:
        param_filename = tkinter_io.get_paramfile_tkinter()

    dysmalpy_make_model(param_filename=param_filename)
