# Script to fit single object in 3D with Dysmalpy

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

import copy
import numpy as np
import astropy.units as u

try:
    import utils_io
except:
    from . import utils_io


def user_specific_load_3D_data(param_filename=None):
    # EDIT THIS FILE TO HAVE SPECIFIC LOADING OF DATA!

    params = utils_io.read_fitting_params(fname=param_filename)

    # Recommended to trim cube to around the relevant line only,
    # both for speed of computation and to avoid noisy spectral resolution elements.

    FNAME_CUBE = None
    FNAME_ERR = None
    FNAME_MASK = None
    FNAME_MASK_SKY = None       # sky plane mask -- eg, mask areas away from galaxy.
    FNAME_MASK_SPEC = None      # spectral dim masking -- eg, mask a skyline.
                                #  ** When trimming cube mind that masks need to be appropriately trimmed too.

    # Optional: set RA/Dec of reference pixel in the cube: mind trimming....
    ref_pixel = None
    ra = None
    dec = None

    pixscale=params['pixscale']

    # +++++++++++++++++++++++++++++++++++++++++++
    # Upload the data set to be fit
    cube = fits.getdata(FNAME_CUBE)
    err_cube = fits.getdata(FNAME_ERR)
    mask_sky = fits.getdata(FNAME_MASK_SKY)
    mask_spec = fits.getdata(FNAME_MASK_SPEC)

    spec_type = 'velocity'    # either 'velocity' or 'wavelength'
    spec_arr = None           # Needs to be array of vel / wavelength for the spectral dim of cube.
                              #  1D arr. Length must be length of spectral dim of cube.
    spec_unit = u.km/u.s      # EXAMPLE! set as needed.

    # Auto mask some bad data
    if automask:
        # Add settings here: S/N ??

        pass

    data3d = data_classes.Data3D(cube, pixscale, spec_type, spec_arr,
                            err_cube = err_cube, mask_cube=mask_cube,
                            mask_sky=mask_sky, mask_spec=mask_spec,
                            ra=ra, dec=dec,
                             ref_pixel=ref_pixel, spec_unit=spec_unit)

    return data3d

def default_load_3D_data(param_filename=None):
    params = utils_io.read_fitting_params(fname=param_filename)

    data3d = utils_io.load_single_object_3D_data(params=params)
    return data3d
#
def dysmalpy_fit_single_3D_wrapper(param_filename=None, default_load_data=True, overwrite=False):

    if default_load_data:
        data3d = default_load_3D_data(param_filename=param_filename)
    else:
        data3d = user_specific_load_3D_data(param_filename=param_filename)

    dysmalpy_fit_single_3D(param_filename=param_filename, data=data3d, overwrite=overwrite)

    return None

def dysmalpy_fit_single_3D(param_filename=None, data=None, datadir=None,
        outdir=None, plot_type='pdf', overwrite=False):

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
    outdir = utils_io.ensure_path_trailing_slash(params['outdir'])
    params['outdir'] = outdir

    fitting.ensure_dir(params['outdir'])

    if 'plot_type' not in params.keys():
        params['plot_type'] = plot_type
    else:
        plot_type = params['plot_type']

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
        print("   make new output folder or remove previous fitting files")
        print('------------------------------------------------------------------')
        print(" ")
    else:
        # Copy paramfile that is OS independent
        if platform.system == 'Windows':
            param_filename_nopath = param_filename.split('\\')[-1]
        else:
            param_filename_nopath = param_filename.split('/')[-1]
        galID_strp = "".join(params['galID'].strip().split("_"))
        galID_strp = "".join(galID_strp.split("-"))
        galID_strp = "".join(galID_strp.split(" "))
        paramfile_strp = "".join(param_filename_nopath.strip().split("_"))
        paramfile_strp = "".join(paramfile_strp.split("-"))
        paramfile_strp = "".join(paramfile_strp.split(" "))
        if galID_strp.strip().lower() in paramfile_strp.strip().lower():
            # Already has galID in param filename:
            shutil.copy(param_filename, outdir)
        else:
            # Copy, prepending galID
            shutil.copy(param_filename, outdir+"{}_{}".format(params['galID'], param_filename_nopath))


        #######################
        # Setup
        gal, fit_dict = setup_single_object_3D(params=params, data=data)

        config_c_m_data = config.Config_create_model_data(**fit_dict)
        config_sim_cube = config.Config_simulate_cube(**fit_dict)
        kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

        # Clean up existing log file:
        if os.path.isfile(fit_dict['f_log']):
            os.remove(fit_dict['f_log'])

        # #######
        # # DEBUGGING:
        # gal.create_model_data(oversample=fit_dict['oversample'], oversize=fit_dict['oversize'],
        #                       line_center=gal.model.line_center)
        # gal.model_cube.data.write(fit_dict['f_cube'], overwrite=True)
        #
        # gal.model_data.data.write(fit_dict['f_cube']+'.scaled.fits', overwrite=True)
        # gal.data.data = gal.data.data * gal.data.mask
        # gal.data.data.write(fit_dict['f_cube']+'.data.fits', overwrite=True)
        #
        # raise ValueError
        # #######

        # Fit
        if fit_dict['fit_method'] == 'mcmc':
            results = fitting.fit_mcmc(gal, nWalkers=fit_dict['nWalkers'], nCPUs=fit_dict['nCPUs'],
                                  scale_param_a=fit_dict['scale_param_a'], nBurn=fit_dict['nBurn'],
                                  nSteps=fit_dict['nSteps'], minAF=fit_dict['minAF'],
                                  maxAF=fit_dict['maxAF'],
                                  nEff=fit_dict['nEff'], do_plotting=fit_dict['do_plotting'],
                                  red_chisq=fit_dict['red_chisq'],
                                  oversampled_chisq = fit_dict['oversampled_chisq'],
                                  fitdispersion=fit_dict['fitdispersion'],
                                  fitflux=fit_dict['fitflux'],
                                  blob_name=fit_dict['blob_name'],
                                  linked_posterior_names=fit_dict['linked_posterior_names'],
                                  outdir=fit_dict['outdir'],
                                  f_plot_trace_burnin=fit_dict['f_plot_trace_burnin'],
                                  f_plot_trace=fit_dict['f_plot_trace'],
                                  f_model=fit_dict['f_model'],
                                  f_model_bestfit=fit_dict['f_model_bestfit'],
                                  f_cube=fit_dict['f_cube'],
                                  f_sampler=fit_dict['f_sampler'],
                                  f_burn_sampler=fit_dict['f_burn_sampler'],
                                  f_plot_param_corner=fit_dict['f_plot_param_corner'],
                                  f_plot_bestfit=fit_dict['f_plot_bestfit'],
                                  f_mcmc_results=fit_dict['f_mcmc_results'],
                                  f_chain_ascii=fit_dict['f_chain_ascii'],
                                  f_vel_ascii=fit_dict['f_vel_ascii'],
                                  f_log=fit_dict['f_log'],
                                  overwrite=overwrite,
                                  plot_type=plot_type,
                                  **kwargs_galmodel)

        elif fit_dict['fit_method'] == 'mpfit':
            results = fitting.fit_mpfit(gal, fitdispersion=fit_dict['fitdispersion'],
                                        fitflux=fit_dict['fitflux'],
                                        maxiter=fit_dict['maxiter'],
                                        do_plotting=fit_dict['do_plotting'],
                                        outdir=fit_dict['outdir'],
                                        f_model=fit_dict['f_model'],
                                        f_model_bestfit=fit_dict['f_model_bestfit'],
                                        f_cube=fit_dict['f_cube'],
                                        f_plot_bestfit=fit_dict['f_plot_bestfit'],
                                        f_results=fit_dict['f_results'],
                                        f_vel_ascii=fit_dict['f_vel_ascii'],
                                        f_log=fit_dict['f_log'],
                                        blob_name=fit_dict['blob_name'],
                                        overwrite=overwrite,
                                        plot_type=plot_type,
                                        **kwargs_galmodel)

        # Save results
        utils_io.save_results_ascii_files(fit_results=results, gal=gal, params=params,
                        overwrite=overwrite)

    return None


def setup_single_object_3D(params=None, data=None):

    # ------------------------------------------------------------
    # Load data:
    if data is None:
        data = utils_io.load_single_object_3D_data(params=params)


    # ------------------------------------------------------------
    # Setup galaxy, instrument, model:

    gal = utils_io.setup_gal_model_base(params=params)

    # Override FOV from the cube shape:
    gal.instrument.fov = [data.shape[2], data.shape[1]]

    # ------------------------------------------------------------

    gal.data = data

    # ------------------------------------------------------------
    # Setup fitting dict:
    fit_dict = utils_io.setup_fit_dict(params=params, ndim_data=3)

    return gal, fit_dict



if __name__ == "__main__":

    param_filename = sys.argv[1]

    dysmalpy_fit_single_3D_wrapper(param_filename=param_filename)
