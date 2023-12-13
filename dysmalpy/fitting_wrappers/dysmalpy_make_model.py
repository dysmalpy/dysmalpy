# Script to fit single object with Dysmalpy: get dimension from paramfile data names

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys

try:
    import tkinter_io
except ImportError:
    from . import tkinter_io

def dysmalpy_make_model(param_filename=None, outdir=None, overwrite=None):
    """
    Generate a DYSMAL model, based on parameter values specified in the parameter file.

    Input:
        param_filename:     Path to parameters file.

    Optional input:
        outdir:             Path to output directory. If set, overrides outdir set in the parameters file.

        overwrite:          Option to overwrite any pre-existing fititng files.
                            If set, overrides overwrite set in the parameters file.

    Output:
            Saves model files to outdir (specifed in call to `dysmalpy_make_model` or in parameters file).
    """

    # Only load full imports later to speed up usage from command line.
    from dysmalpy import config

    from dysmalpy.fitting_wrappers import utils_io
    from dysmalpy import data_io

    # Read in the parameters from param_filename:
    params = utils_io.read_fitting_params(fname=param_filename)

    # Check if 'overwrite' is set in the params file.
    # But the direct input from calling the script overrides any setting in the params file.
    if overwrite is None:
        if 'overwrite' in params.keys():
            overwrite = params['overwrite']
        else:
            overwrite = False
    params['overwrite'] = overwrite

    # OVERRIDE SETTINGS FROM PARAMS FILE if passed directly -- eg from an example Jupyter NB:
    if outdir is not None:
        params['outdir'] = outdir

    # Setup some paths:

    # Ensure output directory is specified: if relative file path,
    #   EXPLICITLY prepend paramfile path
    outdir = data_io.ensure_path_trailing_slash(params['outdir'])
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
        #data_io.ensure_dir(outdir)
        data_io.ensure_dir(outdir)

        # Copy paramfile that is OS independent
        utils_io.preserve_param_file(param_filename, params=params, outdir=outdir)

        # Cleanup if overwriting:
        if model_exists:
            os.remove(outdir+'{}{}_model_cube.fits'.format(params['galID'], filename_extra))

        ############################################################################################
        # ------------------------------------------------------------
        # Setup galaxy, model:
        gal = utils_io.setup_gal_model_base(params=params)

        # Set up empty observation:
        obs = utils_io.make_empty_observation(0, params=params, ndim=3)

        # Add the observation to the Galaxy
        gal.add_observation(obs)

        f_cube = outdir+'{}{}_model_cube.fits'.format(params['galID'], filename_extra)

        ############################################################################################

        # Make model
        gal.create_model_data()

        # Save cube
        gal.observations[obs.name].model_cube.data.write(f_cube, overwrite=overwrite)

        print('------------------------------------------------------------------')
        print(' Dysmalpy model complete for: {}'.format(params['galID']))
        print('   output folder: {}'.format(outdir))
        print('------------------------------------------------------------------')
        print(' ')



    return None


if __name__ == "__main__":
    try:
        param_filename = sys.argv[1]
    except:
        param_filename = tkinter_io.get_paramfile_tkinter()

    dysmalpy_make_model(param_filename=param_filename)
