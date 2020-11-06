# Script to fit single object in 2D with Dysmalpy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
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
    import plotting as fw_plotting
except:
    from . import utils_io
    from . import plotting as fw_plotting

def dysmalpy_fit_single_2D(param_filename=None, data=None, datadir=None,
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
        gal, fit_dict = setup_single_object_2D(params=params, data=data)

        config_c_m_data = config.Config_create_model_data(**fit_dict)
        config_sim_cube = config.Config_simulate_cube(**fit_dict)
        kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

        # Clean up existing log file:
        if os.path.isfile(fit_dict['f_log']):
            os.remove(fit_dict['f_log'])

        # Fit
        if fit_dict['fit_method'] == 'mcmc':
            results = fitting.fit_mcmc(gal, nWalkers=fit_dict['nWalkers'],
                                       nCPUs=fit_dict['nCPUs'],
                                       scale_param_a=fit_dict['scale_param_a'],
                                       nBurn=fit_dict['nBurn'],
                                       nSteps=fit_dict['nSteps'],
                                       minAF=fit_dict['minAF'],
                                       maxAF=fit_dict['maxAF'],
                                       nEff=fit_dict['nEff'],
                                       do_plotting=fit_dict['do_plotting'],
                                       red_chisq=fit_dict['red_chisq'],
                                       oversampled_chisq=fit_dict['oversampled_chisq'],
                                       fitdispersion=fit_dict['fitdispersion'],
                                       fitflux=fit_dict['fitflux'],
                                       blob_name=fit_dict['blob_name'],
                                       outdir=fit_dict['outdir'],
                                       save_bestfit_cube=False,
                                       linked_posterior_names=fit_dict['linked_posterior_names'],
                                       model_key_re=fit_dict['model_key_re'],
                                       model_key_halo=fit_dict['model_key_halo'],
                                       f_plot_trace_burnin=fit_dict['f_plot_trace_burnin'],
                                       f_plot_trace=fit_dict['f_plot_trace'],
                                       f_model=fit_dict['f_model'],
                                       f_model_bestfit=fit_dict['f_model_bestfit'],
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


        # Plot multid, if enabled:
        if 'fdata_1d' in params.keys():
            fw_plotting.plot_results_multid(param_filename=param_filename, fit_ndim=2, show_1d_apers=True, remove_shift=True,
                                    plot_type=plot_type)

    return None


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
        galtmp, fit_dict = setup_single_object_2D(params=params, data=data)

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
        results.analyze_plot_save_results(gal,
                      blob_name=fit_dict['blob_name'],
                      linked_posterior_names=fit_dict['linked_posterior_names'],
                      model_key_re=fit_dict['model_key_re'],
                      model_key_halo=fit_dict['model_key_halo'],
                      fitdispersion=fit_dict['fitdispersion'],
                      f_model=fit_dict['f_model'],
                      f_model_bestfit=fit_dict['f_model_bestfit'],
                      f_vel_ascii = fit_dict['f_vel_ascii'],
                      save_data=True,
                      save_bestfit_cube=False,
                      do_plotting = True,
                      **kwargs_galmodel)

        # Reload fitting stuff to get the updated gal object
        gal, results = fitting.reload_all_fitting(filename_galmodel=fit_dict['f_model'],
                                    filename_results=fit_dict['f_mcmc_results'],
                                    fit_method=params['fit_method'])

        # Save results
        utils_io.save_results_ascii_files(fit_results=results, gal=gal, params=params,
                        overwrite=overwrite)

    elif params['fit_method'] == 'mpfit':
        galtmp, fit_dict = setup_single_object_2D(params=params, data=data)

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
    if 'fdata_1d' in params.keys():
        fw_plotting.plot_results_multid(param_filename=param_filename, fit_ndim=2, show_1d_apers=True, remove_shift=True,
                        plot_type=plot_type)

    return None



def setup_single_object_2D(params=None, data=None):
    # ------------------------------------------------------------
    # Setup galaxy, instrument, model:

    gal = utils_io.setup_gal_model_base(params=params)

    # ------------------------------------------------------------
    # Load data:
    if data is None:
        gal.data = utils_io.load_single_object_2D_data(params=params)
    else:
        gal.data = data

    # ------------------------------------------------------------
    # Setup fitting dict:
    fit_dict = utils_io.setup_fit_dict(params=params, ndim_data=2)

    return gal, fit_dict






if __name__ == "__main__":

    param_filename = sys.argv[1]

    try:
        if sys.argv[2].strip().lower() == 'reanalyze':
            reanalyze = True
        else:
            reanalyze = False
    except:
        reanalyze = False

    if reanalyze:
        dysmalpy_reanalyze_single_2D(param_filename=param_filename)
    else:
        dysmalpy_fit_single_2D(param_filename=param_filename)
