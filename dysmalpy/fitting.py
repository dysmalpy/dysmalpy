# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Main classes and functions for fitting DYSMALPY kinematic models
#   to the observed data using MCMC

# Some handling of MCMC / posterior distribution analysis inspired by speclens,
#    with thanks to Matt George:
#    https://github.com/mrgeorge/speclens/blob/master/speclens/fit.py

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

## Standard library
import logging
from multiprocessing import cpu_count, Pool
import abc

# DYSMALPY code
from dysmalpy import plotting
from dysmalpy import galaxy
from dysmalpy.parameters import UniformLinearPrior
from dysmalpy.instrument import DoubleBeam, Moffat, GaussianBeam
from dysmalpy import config

from dysmalpy.utils import fit_uncertainty_ellipse

# Third party imports
import os
import numpy as np
from collections import OrderedDict
import six
import astropy.units as u
import dill as _pickle
import copy
from dysmalpy.extern.mpfit import mpfit
import emcee

from dysmalpy import utils_io as dpy_utils_io

if np.int(emcee.__version__[0]) >= 3:
    import h5py

import time, datetime

from scipy.stats import gaussian_kde
from scipy.optimize import fmin


__all__ = ['fit_mcmc', 'fit_mpfit', 'MCMCResults', 'MPFITResults']


# ACOR SETTINGS
acor_force_min = 49
# Force it to run for at least 50 steps, otherwise acor times might be completely wrong.

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')



def fit(*args, **kwargs):

    wrn_msg = "fitting.fit has been depreciated.\n"
    wrn_msg += "Instead call 'fitting.fit_mcmc' or 'fitting.fit_mpfit'."

    raise ValueError(wrn_msg)

    return None


def fit_mcmc(gal, **kwargs):
    """
    Fit observed kinematics using MCMC and a DYSMALPY model set.

    Input:
            gal:            observed galaxy, including kinematics.
                            also contains instrument the galaxy was observed with (gal.instrument)
                            and the DYSMALPY model set, with the parameters to be fit (gal.model)

            mcmc_options:   dictionary with MCMC fitting options
                            ** potentially expand this in the future, and force this to
                            be an explicit set of parameters -- might be smarter!!!

    Output:
            MCMCResults class instance containing the bestfit parameters, sampler information, etc.
    """
    config_c_m_data = config.Config_create_model_data(**kwargs)
    config_sim_cube = config.Config_simulate_cube(**kwargs)
    kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

    config_fit = config.Config_fit_mcmc(**kwargs)
    kwargs_fit = config_fit.dict

    # --------------------------------
    # Check option validity:
    if kwargs_fit['blob_name'] is not None:
        valid_blobnames = ['fdm', 'mvirial', 'alpha', 'rb']
        if isinstance(kwargs_fit['blob_name'], str):
            # Single blob
            blob_arr = [kwargs_fit['blob_name']]
        else:
            # Array of blobs
            blob_arr = kwargs_fit['blob_name'][:]

        for blobn in blob_arr:
            if blobn.lower().strip() not in valid_blobnames:
                raise ValueError("blob_name={} not recognized as option!".format(blobn))


    # Temporary: testing:
    if kwargs_fit['red_chisq']:
        raise ValueError("red_chisq=True is currently *DISABLED* to test lnlike impact vs lnprior")

    # Check the FOV is large enough to cover the data output:
    dpy_utils_io._check_data_inst_FOV_compatibility(gal)

    # --------------------------------
    # Basic setup:

    # For compatibility with Python 2.7:
    mod_in = copy.deepcopy(gal.model)
    gal.model = mod_in

    #if nCPUs is None:
    if kwargs_fit['cpuFrac'] is not None:
        kwargs_fit['nCPUs'] = np.int(np.floor(cpu_count()*kwargs_fit['cpuFrac']))

    # +++++++++++++++++++++++
    # Setup for oversampled_chisq:
    if kwargs_fit['oversampled_chisq']:
        gal = setup_oversampled_chisq(gal)
    # +++++++++++++++++++++++

    # Output filenames
    if (len(kwargs_fit['outdir']) > 0):
        if (kwargs_fit['outdir'][-1] != '/'): kwargs_fit['outdir'] += '/'
    ensure_dir(kwargs_fit['outdir'])

    # --------------------------------
    # Setup fit_kwargs dict:

    fit_kwargs = {**kwargs_galmodel, **kwargs_fit}

    # --------------------------------
    # Split by emcee version:

    if np.int(emcee.__version__[0]) >= 3:
        mcmcResults = _fit_emcee_3(gal, **fit_kwargs)
    else:
        mcmcResults = _fit_emcee_221(gal, **fit_kwargs)

    return mcmcResults






def _fit_emcee_221(gal, **kwargs ):

    # OLD version
    config_c_m_data = config.Config_create_model_data(**kwargs)
    config_sim_cube = config.Config_simulate_cube(**kwargs)
    kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

    config_fit = config.Config_fit_mcmc(**kwargs)
    kwargs_fit = config_fit.dict

    # Check to make sure previous sampler won't be overwritten: custom if continue_steps:
    if kwargs_fit['continue_steps'] and (kwargs_fit['f_sampler'] is None):
        kwargs_fit['f_sampler'] = kwargs_fit['outdir']+'mcmc_sampler_continue.pickle'
    if (kwargs_fit['f_sampler_tmp'] is None):
        kwargs_fit['f_sampler_tmp'] = kwargs_fit['outdir']+'mcmc_sampler_INPROGRESS.pickle'

    # If the output filenames aren't defined: use default output filenames
    if kwargs_fit['f_plot_trace_burnin'] is None:
        kwargs_fit['f_plot_trace_burnin'] = kwargs_fit['outdir']+'mcmc_burnin_trace.{}'.format(kwargs_fit['plot_type'])
    if kwargs_fit['f_plot_trace'] is None:
        kwargs_fit['f_plot_trace'] = kwargs_fit['outdir']+'mcmc_trace.{}'.format(kwargs_fit['plot_type'])
    if kwargs_fit['save_model'] and (kwargs_fit['f_model'] is None):
        kwargs_fit['f_model'] = kwargs_fit['outdir']+'galaxy_model.pickle'
    if kwargs_fit['save_bestfit_cube'] and (kwargs_fit['f_cube'] is None):
        kwargs_fit['f_cube'] = kwargs_fit['outdir']+'mcmc_bestfit_cube.fits'
    if kwargs_fit['f_sampler'] is None:
        kwargs_fit['f_sampler'] = kwargs_fit['outdir']+'mcmc_sampler.pickle'
    if kwargs_fit['save_burn'] and (kwargs_fit['f_burn_sampler'] is None):
        kwargs_fit['f_burn_sampler'] = kwargs_fit['outdir']+'mcmc_burn_sampler.pickle'
    if kwargs_fit['f_plot_param_corner'] is None:
        kwargs_fit['f_plot_param_corner'] = kwargs_fit['outdir']+'mcmc_param_corner.{}'.format(kwargs_fit['plot_type'])
    if kwargs_fit['f_plot_bestfit'] is None:
        kwargs_fit['f_plot_bestfit'] = kwargs_fit['outdir']+'mcmc_best_fit.{}'.format(kwargs_fit['plot_type'])
    if kwargs_fit['save_results'] & (kwargs_fit['f_results'] is None):
        # LEGACY SUPPORT: WILL BE DEPRECIATED:
        if kwargs_fit['f_mcmc_results'] is not None:
            kwargs_fit['f_results'] = kwargs_fit['f_mcmc_results']
        else:
            kwargs_fit['f_results'] = kwargs_fit['outdir']+'mcmc_results.pickle'
    if kwargs_fit['f_chain_ascii'] is None:
        kwargs_fit['f_chain_ascii'] = kwargs_fit['outdir']+'mcmc_chain_blobs.dat'
    if kwargs_fit['save_vel_ascii'] & (kwargs_fit['f_vel_ascii'] is None):
        kwargs_fit['f_vel_ascii'] = kwargs_fit['outdir']+'galaxy_bestfit_vel_profile.dat'


    if kwargs_fit['save_model_bestfit'] & (kwargs_fit['f_model_bestfit'] is None):
        if gal.data.ndim == 1:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-1dplots.txt'
        elif gal.data.ndim == 2:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-velmaps.fits'
        elif gal.data.ndim == 3:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-cube.fits'
        elif gal.data.ndim == 0:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-0d.txt'

    # ---------------------------------------------------
    # Check for existing files if overwrite=False:
    if (not kwargs_fit['overwrite']):
        fnames = []
        fnames_opt = [ kwargs_fit['f_plot_trace_burnin'], kwargs_fit['f_plot_trace'],
                    kwargs_fit['f_sampler'], kwargs_fit['f_plot_param_corner'],
                    kwargs_fit['f_plot_bestfit'], kwargs_fit['f_results'],
                    kwargs_fit['f_chain_ascii'], kwargs_fit['f_vel_ascii'],
                    kwargs_fit['f_model'], kwargs_fit['f_cube'], kwargs_fit['f_burn_sampler'] ]
        for fname in fnames_opt:
            if fname is not None:
                fnames.append(fname)

        for fname in fnames:
                if os.path.isfile(fname):
                    logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(kwargs_fit['overwrite'], fname))

        # Return early if it won't save the results, sampler:
        if os.path.isfile(kwargs_fit['f_sampler']) or os.path.isfile(kwargs_fit['f_results']):
            msg = "overwrite={}, and one of 'f_sampler' or 'f_results' won't be saved,".format(kwargs_fit['overwrite'])
            msg += " so the fit will not be saved.\n Specify new outfile or delete old files."
            logger.warning(msg)
            return None
    # ---------------------------------------------------

    # Setup file redirect logging:
    if kwargs_fit['f_log'] is not None:
        loggerfile = logging.FileHandler(kwargs_fit['f_log'])
        loggerfile.setLevel(logging.INFO)
        logger.addHandler(loggerfile)

    # ++++++++++++++++++++++++++++++

    # --------------------------------
    # Initialize emcee sampler
    kwargs_dict_mcmc = {'fitdispersion':kwargs_fit['fitdispersion'],
                    'fitflux':kwargs_fit['fitflux'],
                    'blob_name': kwargs_fit['blob_name'],
                    'model_key_re':kwargs_fit['model_key_re'],
                    'model_key_halo': kwargs_fit['model_key_halo'],
                    'red_chisq': kwargs_fit['red_chisq'],
                    'oversampled_chisq': kwargs_fit['oversampled_chisq']}

    kwargs_dict = {**kwargs_dict_mcmc, **kwargs_galmodel}

    # kwargs_dict = {**kwargs_fit, **kwargs_galmodel}

    nBurn_orig = kwargs_fit['nBurn']

    nDim = gal.model.nparams_free

    if (not kwargs_fit['continue_steps']) & ((not kwargs_fit['save_intermediate_sampler_chain']) \
        | (not os.path.isfile(kwargs_fit['f_sampler_tmp']))):
        sampler = emcee.EnsembleSampler(kwargs_fit['nWalkers'], nDim, log_prob,
                    args=[gal], kwargs=kwargs_dict,
                    a = kwargs_fit['scale_param_a'], threads = kwargs_fit['nCPUs'])

        # --------------------------------
        # Initialize walker starting positions
        initial_pos = initialize_walkers(gal.model, nWalkers=kwargs_fit['nWalkers'])
    #
    elif kwargs_fit['continue_steps']:
        kwargs_fit['nBurn'] = 0
        if input_sampler is None:
            try:
                input_sampler = load_pickle(kwargs_fit['f_sampler'])
            except:
                message = "Couldn't find existing sampler in {}.".format(kwargs_fit['f_sampler'])
                message += '\n'
                message += "Must set input_sampler if you will restart the sampler."
                raise ValueError(message)

        sampler = reinitialize_emcee_sampler(input_sampler, gal=gal,
                            kwargs_dict=kwargs_dict,
                            scale_param_a=kwargs_fit['scale_param_a'])

        initial_pos = input_sampler['chain'][:,-1,:]
        if kwargs_fit['blob_name'] is not None:
            blob = input_sampler['blobs']

        # Close things
        input_sampler = None

    elif kwargs_fit['save_intermediate_sampler_chain'] & (os.path.isfile(kwargs_fit['f_sampler_tmp'])):
        input_sampler = load_pickle(kwargs_fit['f_sampler_tmp'])

        sampler = reinitialize_emcee_sampler(input_sampler, gal=gal,
                            kwargs_dict=kwargs_dict,
                            scale_param_a=kwargs_fit['scale_param_a'])
        kwargs_fit['nBurn'] = nBurn_orig - (input_sampler['burn_step_cur'] + 1)

        initial_pos = input_sampler['chain'][:,-1,:]
        if kwargs_fit['blob_name'] is not None:
            blob = input_sampler['blobs']

        # If it saved after burn finished, but hasn't saved any of the normal steps: reset sampler
        if ((kwargs_fit['nBurn'] == 0) & (input_sampler['step_cur'] < 0)):
            blob = None
            sampler.reset()
            if kwargs_fit['blob_name'] is not None:
                 sampler.clear_blobs()

        # Close things
        input_sampler = None


    # --------------------------------
    # Output some fitting info to logger:
    logger.info("*************************************")
    logger.info(" Fitting: {} with MCMC".format(gal.name))
    if gal.data.filename_velocity is not None:
        logger.info("    velocity file: {}".format(gal.data.filename_velocity))
    if gal.data.filename_dispersion is not None:
        logger.info("    dispers. file: {}".format(gal.data.filename_dispersion))

    #logger.info('\n')
    logger.info('\n'+'nCPUs: {}'.format(kwargs_fit['nCPUs']))
    logger.info('nWalkers: {}'.format(kwargs_fit['nWalkers']))
    logger.info('lnlike: red_chisq={}'.format(kwargs_fit['red_chisq']))
    logger.info('lnlike: oversampled_chisq={}'.format(kwargs_fit['oversampled_chisq']))

    #logger.info('\n')
    logger.info('\n'+'blobs: {}'.format(kwargs_fit['blob_name']))


    #logger.info('\n')
    logger.info('\n'+'mvirial_tied: {}'.format(gal.model.components['halo'].mvirial.tied))
    if 'mhalo_relation' in gal.model.components['disk+bulge'].__dict__.keys():
        logger.info('mhalo_relation: {}'.format(gal.model.components['disk+bulge'].mhalo_relation))
    if 'truncate_lmstar_halo' in gal.model.components['disk+bulge'].__dict__.keys():
        logger.info('truncate_lmstar_halo: {}'.format(gal.model.components['disk+bulge'].truncate_lmstar_halo))
    logger.info('nSubpixels: {}'.format(kwargs_galmodel['oversample']))

    ################################################################
    # --------------------------------
    # Run burn-in
    if kwargs_fit['nBurn'] > 0:
        logger.info('\nBurn-in:'+'\n'
                    'Start: {}\n'.format(datetime.datetime.now()))
        start = time.time()

        ####
        pos = initial_pos
        prob = None
        state = None
        blob = None
        for k in six.moves.xrange(nBurn_orig):
            # --------------------------------
            # If recovering intermediate save, only start past existing chain length:
            if kwargs_fit['save_intermediate_sampler_chain']:
                if k < sampler.chain.shape[1]:
                    continue

            logger.info(" k={}, time.time={}, a_frac={}".format( k, datetime.datetime.now(),
                        np.mean(sampler.acceptance_fraction)  ) )
            ###
            pos_cur = pos.copy()    # copy just in case things are set strangely

            # Run one sample step:
            if kwargs_fit['blob_name'] is not None:
                pos, prob, state, blob = sampler.run_mcmc(pos_cur, 1, lnprob0=prob,
                        rstate0=state, blobs0 = blob)
            else:
                pos, prob, state = sampler.run_mcmc(pos_cur, 1, lnprob0=prob, rstate0=state)


            # --------------------------------
            # Save intermediate steps if set:
            if kwargs_fit['save_intermediate_sampler_chain']:
                if ((k+1) % kwargs_fit['nStep_intermediate_save'] == 0):
                    sampler_dict_tmp = make_emcee_sampler_dict(sampler, nBurn=0, emcee_vers=2)
                    sampler_dict_tmp['burn_step_cur'] = k
                    sampler_dict_tmp['step_cur'] = -99
                    if kwargs_fit['f_sampler_tmp'] is not None:
                        # Save stuff to file, for future use:
                        dump_pickle(sampler_dict_tmp, filename=kwargs_fit['f_sampler_tmp'], overwrite=True)
            # --------------------------------

        #####
        end = time.time()
        elapsed = end-start


        try:
            acor_time = sampler.get_autocorr_time(low=5, c=10)
        except:
            acor_time = "Undefined, chain did not converge"

        #######################################################################################
        # Return Burn-in info
        # ****
        endtime = str(datetime.datetime.now())
        nthingsmsg = 'nCPU, nParam, nWalker, nBurn = {}, {}, {}, {}'.format(kwargs_fit['nCPUs'],
            nDim, kwargs_fit['nWalkers'], kwargs_fit['nBurn'])
        scaleparammsg = 'Scale param a= {}'.format(kwargs_fit['scale_param_a'])
        timemsg = 'Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s)'.format( elapsed, np.floor(elapsed/60.),
                (elapsed/60.-np.floor(elapsed/60.))*60. )
        macfracmsg = "Mean acceptance fraction: {:0.3f}".format(np.mean(sampler.acceptance_fraction))
        acortimemsg = "Autocorr est: "+str(acor_time)
        logger.info('\nEnd: '+endtime+'\n'
                    '\n******************\n'
                    ''+nthingsmsg+'\n'
                    ''+scaleparammsg+'\n'
                    ''+timemsg+'\n'
                    ''+macfracmsg+'\n'
                    "Ideal acceptance frac: 0.2 - 0.5\n"
                    ''+acortimemsg+'\n'
                    '******************')

        nBurn_nEff = 2
        try:
            if nBurn < np.max(acor_time) * nBurn_nEff:
                nburntimemsg = 'nBurn is less than {}*acorr time'.format(nBurn_nEff)
                logger.info('\n#################\n'
                            ''+nburntimemsg+'\n'
                            '#################\n')
                # Give warning if the burn-in is less than say 2-3 times the autocorr time
        except:
            logger.info('\n#################\n'
                        "acorr time undefined -> can't check convergence\n"
                        '#################\n')

        # --------------------------------
        # Save burn-in sampler, if desired
        if (kwargs_fit['save_burn']) & (kwargs_fit['f_burn_sampler'] is not None):
            sampler_burn = make_emcee_sampler_dict(sampler, nBurn=0, emcee_vers=2)
            # Save stuff to file, for future use:
            dump_pickle(sampler_burn, filename=kwargs_fit['f_burn_sampler'], overwrite=kwargs_fit['overwrite'])


        # --------------------------------
        # Plot burn-in trace, if output file set
        if (kwargs_fit['do_plotting']) & (kwargs_fit['f_plot_trace_burnin'] is not None):
            sampler_burn = make_emcee_sampler_dict(sampler, nBurn=0, emcee_vers=2)
            mcmcResultsburn = MCMCResults(model=gal.model, sampler=sampler_burn)
            plotting.plot_trace(mcmcResultsburn, fileout=kwargs_fit['f_plot_trace_burnin'],
                        overwrite=kwargs_fit['overwrite'])

        # Reset sampler after burn-in:
        sampler.reset()
        if kwargs_fit['blob_name'] is not None:
             sampler.clear_blobs()

    else:
        # --------------------------------
        # No burn-in: set initial position:
        if nBurn_orig > 0:
            logger.info('\nUsing previously completed burn-in'+'\n')

        pos = np.array(initial_pos)
        prob = None
        state = None

        if (not kwargs_fit['continue_steps']) | (not kwargs_fit['save_intermediate_sampler_chain']):
            blob = None

    #######################################################################################
    # ****
    # --------------------------------
    # Run sampler: Get start time
    logger.info('\nEnsemble sampling:\n'
                'Start: {}\n'.format(datetime.datetime.now()))
    start = time.time()

    if sampler.chain.shape[1] > 0:
        logger.info('\n   Resuming with existing sampler chain'+'\n')

    # --------------------------------
    # Run sampler: output info at each step
    for ii in six.moves.xrange(kwargs_fit['nSteps']):

        # --------------------------------
        # If continuing chain, only start past existing chain length:
        if kwargs_fit['continue_steps'] | kwargs_fit['save_intermediate_sampler_chain']:
            if ii < sampler.chain.shape[1]:
                continue

        pos_cur = pos.copy()    # copy just in case things are set strangely

        # --------------------------------
        # Only do one step at a time:
        if kwargs_fit['blob_name'] is not None:
            pos, prob, state, blob = sampler.run_mcmc(pos_cur, 1, lnprob0=prob,
                    rstate0=state, blobs0 = blob)
        else:
            pos, prob, state = sampler.run_mcmc(pos_cur, 1, lnprob0=prob, rstate0=state)
        # --------------------------------

        # --------------------------------
        # Give output info about this step:
        nowtime = str(datetime.datetime.now())
        stepinfomsg = "ii={}, a_frac={}".format( ii, np.mean(sampler.acceptance_fraction) )
        timemsg = " time.time()={}".format(nowtime)
        logger.info( stepinfomsg+timemsg )

        try:
            acor_time = sampler.get_autocorr_time(low=5, c=10)
            logger.info( "{}: acor_time ={}".format(ii, np.array(acor_time) ) )
        except:
            acor_time = "Undefined, chain did not converge"
            logger.info(" {}: Chain too short for acor to run".format(ii) )

        # --------------------------------
        # Case: test for convergence and truncate early:
        # Criteria checked: whether acceptance fraction within (minAF, maxAF),
        #                   and whether total number of steps > nEff * average autocorrelation time:
        #                   to make sure the paramter space is well explored.
        if ((kwargs_fit['minAF'] is not None) & (kwargs_fit['maxAF'] is not None) & \
                (kwargs_fit['nEff'] is not None) & (acor_time is not None)):
            if ((kwargs_fit['minAF'] < np.mean(sampler.acceptance_fraction) < kwargs_fit['maxAF']) & \
                ( ii > np.max(acor_time) * kwargs_fit['nEff'] )):
                    if ii == acor_force_min:
                        logger.info(" Enforced min step limit: {}.".format(ii+1))
                    if ii >= acor_force_min:
                        logger.info(" Finishing calculations early at step {}.".format(ii+1))
                        break

        # --------------------------------
        # Save intermediate steps if set:
        if kwargs_fit['save_intermediate_sampler_chain']:
            if ((ii+1) % kwargs_fit['nStep_intermediate_save'] == 0):
                sampler_dict_tmp = make_emcee_sampler_dict(sampler, nBurn=0, emcee_vers=2)
                sampler_dict_tmp['burn_step_cur'] = nBurn_orig - 1
                sampler_dict_tmp['step_cur'] = ii
                if kwargs_fit['f_sampler_tmp'] is not None:
                    # Save stuff to file, for future use:
                    dump_pickle(sampler_dict_tmp, filename=kwargs_fit['f_sampler_tmp'], overwrite=True)
        # --------------------------------

    # --------------------------------
    # Check if it failed to converge before the max number of steps, if doing convergence testing
    finishedSteps= ii+1
    if (finishedSteps  == kwargs_fit['nSteps']) & ((kwargs_fit['minAF'] is not None) & \
            (kwargs_fit['maxAF'] is not None) & (kwargs_fit['nEff'] is not None)):
        logger.info(" Caution: no convergence within nSteps={}.".format(kwargs_fit['nSteps']))

    # --------------------------------
    # Finishing info for fitting:
    end = time.time()
    elapsed = end-start
    logger.info("Finished {} steps".format(finishedSteps)+"\n")

    try:
        acor_time = sampler.get_autocorr_time(low=5, c=10)
    except:
        acor_time = "Undefined, chain did not converge"

    #######################################################################################
    # ***********
    # Consider overall acceptance fraction
    endtime = str(datetime.datetime.now())
    nthingsmsg = 'nCPU, nParam, nWalker, nSteps = {}, {}, {}, {}'.format(kwargs_fit['nCPUs'],
        nDim, kwargs_fit['nWalkers'], kwargs_fit['nSteps'])
    scaleparammsg = 'Scale param a= {}'.format(kwargs_fit['scale_param_a'])
    timemsg = 'Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s)'.format(elapsed, np.floor(elapsed/60.),
            (elapsed/60.-np.floor(elapsed/60.))*60. )
    macfracmsg = "Mean acceptance fraction: {:0.3f}".format(np.mean(sampler.acceptance_fraction))
    acortimemsg = "Autocorr est: "+str(acor_time)
    logger.info('\nEnd: '+endtime+'\n'
                '\n******************\n'
                ''+nthingsmsg+'\n'
                ''+scaleparammsg+'\n'
                ''+timemsg+'\n'
                ''+macfracmsg+'\n'
                "Ideal acceptance frac: 0.2 - 0.5\n"
                ''+acortimemsg+'\n'
                '******************')

    # --------------------------------
    # Save sampler, if output file set:
    #   Burn-in is already cut by resetting the sampler at the beginning.
    # Get pickleable format:  # _fit_io.make_emcee_sampler_dict
    sampler_dict = make_emcee_sampler_dict(sampler, nBurn=0, emcee_vers=2)


    if kwargs_fit['f_sampler'] is not None:
        # Save stuff to file, for future use:
        dump_pickle(sampler_dict, filename=kwargs_fit['f_sampler'], overwrite=kwargs_fit['overwrite'])


    # --------------------------------
    # Cleanup intermediate saves:
    if kwargs_fit['save_intermediate_sampler_chain']:
        if kwargs_fit['f_sampler_tmp'] is not None:
            if os.path.isfile(kwargs_fit['f_sampler_tmp']):
                os.remove(kwargs_fit['f_sampler_tmp'])
    # --------------------------------

    if kwargs_fit['nCPUs'] > 1:
        sampler.pool.close()

    ##########################################
    ##########################################
    ##########################################

    # --------------------------------
    # Bundle the results up into a results class:
    mcmcResults = MCMCResults(model=gal.model, sampler=sampler_dict, **kwargs_fit)

    if kwargs_fit['oversampled_chisq']:
        mcmcResults.oversample_factor_chisq = gal.data.oversample_factor_chisq

    # Do all analysis, plotting, saving:
    kwargs_all = {**kwargs_galmodel, **kwargs_fit}
    mcmcResults.analyze_plot_save_results(gal, **kwargs_all)

    # Clean up logger:
    if kwargs_fit['f_log'] is not None:
        logger.removeHandler(loggerfile)

    return mcmcResults

def _fit_emcee_3(gal, **kwargs ):
    config_c_m_data = config.Config_create_model_data(**kwargs)
    config_sim_cube = config.Config_simulate_cube(**kwargs)
    kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

    config_fit = config.Config_fit_mcmc(**kwargs)
    kwargs_fit = config_fit.dict

    # filetype for saving sampler: HDF5
    ftype_sampler = 'h5'

    # If the output filenames aren't defined: use default output filenames
    if kwargs_fit['f_plot_trace_burnin'] is None:
        kwargs_fit['f_plot_trace_burnin'] = kwargs_fit['outdir']+'mcmc_burnin_trace.{}'.format(kwargs_fit['plot_type'])
    if kwargs_fit['f_plot_trace'] is None:
        kwargs_fit['f_plot_trace'] = kwargs_fit['outdir']+'mcmc_trace.{}'.format(kwargs_fit['plot_type'])
    if kwargs_fit['save_model'] and (kwargs_fit['f_model'] is None):
        kwargs_fit['f_model'] = kwargs_fit['outdir']+'galaxy_model.pickle'
    if kwargs_fit['save_bestfit_cube'] and (kwargs_fit['f_cube'] is None):
        kwargs_fit['f_cube'] = kwargs_fit['outdir']+'mcmc_bestfit_cube.fits'
    if kwargs_fit['f_sampler'] is None:
        kwargs_fit['f_sampler'] = kwargs_fit['outdir']+'mcmc_sampler.{}'.format(ftype_sampler)
    if kwargs_fit['f_plot_param_corner'] is None:
        kwargs_fit['f_plot_param_corner'] = kwargs_fit['outdir']+'mcmc_param_corner.{}'.format(kwargs_fit['plot_type'])
    if kwargs_fit['save_results'] & (kwargs_fit['f_results'] is None):
        # LEGACY SUPPORT: WILL BE DEPRECIATED:
        if 'f_mcmc_results' in kwargs_fit.keys():
            if kwargs_fit['f_mcmc_results'] is not None:
                kwargs_fit['f_results'] = kwargs_fit['f_mcmc_results']
        if kwargs_fit['f_results'] is None:
            # Check if still None after checking for legacy 'f_mcmc_results'
            kwargs_fit['f_results'] = kwargs_fit['outdir']+'mcmc_results.pickle'
    if kwargs_fit['f_plot_bestfit'] is None:
        kwargs_fit['f_plot_bestfit'] = kwargs_fit['outdir']+'mcmc_best_fit.{}'.format(kwargs_fit['plot_type'])
    if kwargs_fit['f_plot_param_corner'] is None:
        kwargs_fit['f_plot_param_corner'] = kwargs_fit['outdir']+'mcmc_results.pickle'
    if kwargs_fit['f_chain_ascii'] is None:
        kwargs_fit['f_chain_ascii'] = kwargs_fit['outdir']+'mcmc_chain_blobs.dat'
    if kwargs_fit['save_vel_ascii'] & (kwargs_fit['f_vel_ascii'] is None):
        kwargs_fit['f_vel_ascii'] = kwargs_fit['outdir']+'galaxy_bestfit_vel_profile.dat'

    if kwargs_fit['save_model_bestfit'] & (kwargs_fit['f_model_bestfit'] is None):
        if gal.data.ndim == 1:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-1dplots.txt'
        elif gal.data.ndim == 2:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-velmaps.fits'
        elif gal.data.ndim == 3:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-cube.fits'
        elif gal.data.ndim == 0:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-0d.txt'

    # ---------------------------------------------------
    # Check for existing files if overwrite=False:
    if (not kwargs_fit['overwrite']):
        fnames = []
        fnames_opt = [ kwargs_fit['f_plot_trace_burnin'], kwargs_fit['f_plot_trace'], kwargs_fit['f_plot_param_corner'],
                    kwargs_fit['f_plot_bestfit'], kwargs_fit['f_plot_param_corner'],
                    kwargs_fit['f_chain_ascii'], kwargs_fit['f_vel_ascii'],
                    kwargs_fit['f_model'], kwargs_fit['f_cube'] ]
        for fname in fnames_opt:
            if fname is not None:
                fnames.append(fname)

        for fname in fnames:
            if os.path.isfile(fname):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(kwargs_fit['overwrite'], fname))

        # Return early if it won't save the results, sampler:
        if os.path.isfile(kwargs_fit['f_results']):
            msg = "overwrite={}, and 'f_results' won't be saved,".format(kwargs_fit['overwrite'])
            msg += " so the fit will not be saved.\n Specify new outfile or delete old files."
            logger.warning(msg)
            return None

        # Check length of sampler:
        if os.path.isfile(kwargs_fit['f_sampler']):
            backend = emcee.backends.HDFBackend(kwargs_fit['f_sampler'], name='mcmc')

            try:
                if backend.get_chain().shape[0] >= kwargs_fit['nSteps']:
                    if os.path.isfile(kwargs_fit['f_results']):
                        msg = "overwrite={}, and 'f_sampler' already contains {} steps,".format(kwargs_fit['overwrite'], backend.get_chain().shape[0])
                        msg += " so the fit will not be saved.\n Specify new outfile or delete old files."
                        logger.warning(msg)
                        return None
                else:
                    pass
            except:
                pass
    else:
        # Overwrite: remove old file versions
        if os.path.isfile(kwargs_fit['f_sampler']): os.remove(kwargs_fit['f_sampler'])
        if os.path.isfile(kwargs_fit['f_plot_param_corner']): os.remove(kwargs_fit['f_plot_param_corner'])

    # ---------------------------------------------------

    # Setup file redirect logging:
    if kwargs_fit['f_log'] is not None:
        loggerfile = logging.FileHandler(kwargs_fit['f_log'])
        loggerfile.setLevel(logging.INFO)
        logger.addHandler(loggerfile)

    # ++++++++++++++++++++++++++++++

    # --------------------------------
    # Initialize emcee sampler
    kwargs_dict_mcmc = {'fitdispersion':kwargs_fit['fitdispersion'],
                    'fitflux':kwargs_fit['fitflux'],
                    'blob_name': kwargs_fit['blob_name'],
                    'model_key_re':kwargs_fit['model_key_re'],
                    'model_key_halo': kwargs_fit['model_key_halo'],
                    'red_chisq': kwargs_fit['red_chisq'],
                    'oversampled_chisq': kwargs_fit['oversampled_chisq']}

    kwargs_dict = {**kwargs_dict_mcmc, **kwargs_galmodel}

    # kwargs_dict = {**kwargs_fit, **kwargs_galmodel}

    nBurn_orig = kwargs_fit['nBurn']

    nDim = gal.model.nparams_free

    # --------------------------------
    # Start pool, moves, backend:
    if (kwargs_fit['nCPUs'] > 1):
        pool = Pool(kwargs_fit['nCPUs'])
    else:
        pool = None

    moves = emcee.moves.StretchMove(a=kwargs_fit['scale_param_a'])

    backend_burn = emcee.backends.HDFBackend(kwargs_fit['f_sampler'], name="burnin_mcmc")

    if kwargs_fit['overwrite']:
        backend_burn.reset(kwargs_fit['nWalkers'], nDim)

    sampler_burn = emcee.EnsembleSampler(kwargs_fit['nWalkers'], nDim, log_prob,
                backend=backend_burn, pool=pool, moves=moves,
                args=[gal], kwargs=kwargs_dict)

    nBurnCur = sampler_burn.iteration

    kwargs_fit['nBurn'] = nBurn_orig - nBurnCur

    # --------------------------------
    # Initialize walker starting positions
    if sampler_burn.iteration == 0:
        initial_pos = initialize_walkers(gal.model, nWalkers=kwargs_fit['nWalkers'])
    else:
        initial_pos = sampler_burn.get_last_sample()


    # --------------------------------
    # Output some fitting info to logger:
    logger.info("*************************************")
    logger.info(" Fitting: {} with MCMC".format(gal.name))
    if gal.data.filename_velocity is not None:
        logger.info("    velocity file: {}".format(gal.data.filename_velocity))
    if gal.data.filename_dispersion is not None:
        logger.info("    dispers. file: {}".format(gal.data.filename_dispersion))

    logger.info('\n'+'nCPUs: {}'.format(kwargs_fit['nCPUs']))
    logger.info('nWalkers: {}'.format(kwargs_fit['nWalkers']))
    logger.info('lnlike: red_chisq={}'.format(kwargs_fit['red_chisq']))
    logger.info('lnlike: oversampled_chisq={}'.format(kwargs_fit['oversampled_chisq']))

    logger.info('\n'+'blobs: {}'.format(kwargs_fit['blob_name']))


    logger.info('\n'+'mvirial_tied: {}'.format(gal.model.components['halo'].mvirial.tied))
    if 'mhalo_relation' in gal.model.components['disk+bulge'].__dict__.keys():
        logger.info('mhalo_relation: {}'.format(gal.model.components['disk+bulge'].mhalo_relation))
    if 'truncate_lmstar_halo' in gal.model.components['disk+bulge'].__dict__.keys():
        logger.info('truncate_lmstar_halo: {}'.format(gal.model.components['disk+bulge'].truncate_lmstar_halo))
    logger.info('nSubpixels: {}'.format(kwargs_galmodel['oversample']))


    ################################################################
    # --------------------------------
    # Run burn-in
    if kwargs_fit['nBurn'] > 0:
        logger.info('\nBurn-in:'+'\n'
                    'Start: {}\n'.format(datetime.datetime.now()))
        start = time.time()

        ####

        pos = initial_pos
        for k in six.moves.xrange(nBurn_orig):
            # --------------------------------
            # If recovering intermediate save, only start past existing chain length:

            if k < sampler_burn.iteration:
                continue

            logger.info(" k={}, time.time={}, a_frac={}".format( k, datetime.datetime.now(),
                        np.mean(sampler_burn.acceptance_fraction)  ) )
            ###

            # Run one sample step:
            pos = sampler_burn.run_mcmc(pos, 1)

        #####
        end = time.time()
        elapsed = end-start

        acor_time = sampler_burn.get_autocorr_time(tol=10, quiet=True)


        #######################################################################################
        # Return Burn-in info
        # ****
        endtime = str(datetime.datetime.now())
        nthingsmsg = 'nCPU, nParam, nWalker, nBurn = {}, {}, {}, {}'.format(kwargs_fit['nCPUs'],
            nDim, kwargs_fit['nWalkers'], kwargs_fit['nBurn'])
        scaleparammsg = 'Scale param a= {}'.format(kwargs_fit['scale_param_a'])
        timemsg = 'Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s)'.format( elapsed, np.floor(elapsed/60.),
                (elapsed/60.-np.floor(elapsed/60.))*60. )
        macfracmsg = "Mean acceptance fraction: {:0.3f}".format(np.mean(sampler_burn.acceptance_fraction))
        acortimemsg = "Autocorr est: "+str(acor_time)
        logger.info('\nEnd: '+endtime+'\n'
                    '\n******************\n'
                    ''+nthingsmsg+'\n'
                    ''+scaleparammsg+'\n'
                    ''+timemsg+'\n'
                    ''+macfracmsg+'\n'
                    "Ideal acceptance frac: 0.2 - 0.5\n"
                    ''+acortimemsg+'\n'
                    '******************')

        nBurn_nEff = 2
        try:
            if kwargs_fit['nBurn'] < np.max(acor_time) * nBurn_nEff:
                nburntimemsg = 'nBurn is less than {}*acorr time'.format(nBurn_nEff)
                logger.info('\n#################\n'
                            ''+nburntimemsg+'\n'
                            '#################\n')
                # Give warning if the burn-in is less than say 2-3 times the autocorr time
        except:
            logger.info('\n#################\n'
                        "acorr time undefined -> can't check convergence\n"
                        '#################\n')

        # --------------------------------
        # Plot burn-in trace, if output file set
        if (kwargs_fit['do_plotting']) & (kwargs_fit['f_plot_trace_burnin'] is not None):
            sampler_burn_dict = make_emcee_sampler_dict(sampler_burn, nBurn=0)
            mcmcResults_burn = MCMCResults(model=gal.model, sampler=sampler_burn_dict)
            plotting.plot_trace(mcmcResults_burn, fileout=kwargs_fit['f_plot_trace_burnin'], overwrite=kwargs_fit['overwrite'])


    else:
        # --------------------------------
        # No burn-in: set initial position:
        if nBurn_orig > 0:
            logger.info('\nUsing previously completed burn-in'+'\n')

        pos = initial_pos


    #######################################################################################
    # Setup sampler:
    # --------------------------------
    # Start backend:
    backend = emcee.backends.HDFBackend(kwargs_fit['f_sampler'], name="mcmc")

    if kwargs_fit['overwrite']:
        backend.reset(kwargs_fit['nWalkers'], nDim)

    sampler = emcee.EnsembleSampler(kwargs_fit['nWalkers'], nDim, log_prob,
                backend=backend, pool=pool, moves=moves,
                args=[gal], kwargs=kwargs_dict)

    #######################################################################################
    # *************************************************************************************
    # --------------------------------
    # Run sampler: Get start time
    logger.info('\nEnsemble sampling:\n'
                'Start: {}\n'.format(datetime.datetime.now()))
    start = time.time()

    if sampler.iteration > 0:
        logger.info('\n   Resuming with existing sampler chain'+'\n')

    # --------------------------------
    # Run sampler: output info at each step
    for ii in six.moves.xrange(kwargs_fit['nSteps']):

        # --------------------------------
        # If continuing chain, only start past existing chain length:
        if ii < sampler.iteration:
            continue


        # --------------------------------
        # Only do one step at a time:
        pos = sampler.run_mcmc(pos, 1)
        # --------------------------------

        # --------------------------------
        # Give output info about this step:
        nowtime = str(datetime.datetime.now())
        stepinfomsg = "ii={}, a_frac={}".format( ii, np.mean(sampler.acceptance_fraction) )
        timemsg = " time.time()={}".format(nowtime)
        logger.info( stepinfomsg+timemsg )

        acor_time = sampler.get_autocorr_time(tol=10, quiet=True)
        #acor_time = sampler.get_autocorr_time(quiet=True)
        logger.info( "{}: acor_time ={}".format(ii, np.array(acor_time) ) )


        # --------------------------------
        # Case: test for convergence and truncate early:
        # Criteria checked: whether acceptance fraction within (minAF, maxAF),
        #                   and whether total number of steps > nEff * average autocorrelation time:
        #                   to make sure the paramter space is well explored.
        if ((kwargs_fit['minAF'] is not None) & (kwargs_fit['maxAF'] is not None) & \
                (kwargs_fit['nEff'] is not None) & (acor_time is not None)):
            if ((kwargs_fit['minAF'] < np.mean(sampler.acceptance_fraction) < kwargs_fit['maxAF']) & \
                ( ii > np.max(acor_time) * kwargs_fit['nEff'] )):
                    if ii == acor_force_min:
                        logger.info(" Enforced min step limit: {}.".format(ii+1))
                    if ii >= acor_force_min:
                        logger.info(" Finishing calculations early at step {}.".format(ii+1))
                        break


    # --------------------------------
    # Check if it failed to converge before the max number of steps, if doing convergence testing
    finishedSteps= ii+1
    if (finishedSteps  == kwargs_fit['nSteps']) & ((kwargs_fit['minAF'] is not None) & \
                (kwargs_fit['maxAF'] is not None) & (kwargs_fit['nEff'] is not None)):
        logger.info(" Caution: no convergence within nSteps={}.".format(kwargs_fit['nSteps']))

    # --------------------------------
    # Finishing info for fitting:
    end = time.time()
    elapsed = end-start
    logger.info("Finished {} steps".format(finishedSteps)+"\n")

    acor_time = sampler.get_autocorr_time(tol=10, quiet=True)

    #######################################################################################
    # ***********
    # Consider overall acceptance fraction
    endtime = str(datetime.datetime.now())
    nthingsmsg = 'nCPU, nParam, nWalker, nSteps = {}, {}, {}, {}'.format(kwargs_fit['nCPUs'],
        nDim, kwargs_fit['nWalkers'], kwargs_fit['nSteps'])
    scaleparammsg = 'Scale param a= {}'.format(kwargs_fit['scale_param_a'])
    timemsg = 'Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s)'.format(elapsed, np.floor(elapsed/60.),
            (elapsed/60.-np.floor(elapsed/60.))*60. )
    macfracmsg = "Mean acceptance fraction: {:0.3f}".format(np.mean(sampler.acceptance_fraction))
    acortimemsg = "Autocorr est: "+str(acor_time)
    logger.info('\nEnd: '+endtime+'\n'
                '\n******************\n'
                ''+nthingsmsg+'\n'
                ''+scaleparammsg+'\n'
                ''+timemsg+'\n'
                ''+macfracmsg+'\n'
                "Ideal acceptance frac: 0.2 - 0.5\n"
                ''+acortimemsg+'\n'
                '******************')


    if kwargs_fit['nCPUs'] > 1:
        pool.close()
        sampler.pool.close()
        sampler_burn.pool.close()

    ##########################################
    ##########################################
    ##########################################

    # --------------------------------
    # Setup sampler dict:
    sampler_dict = make_emcee_sampler_dict(sampler, nBurn=0)

    # --------------------------------
    # Bundle the results up into a results class:
    mcmcResults = MCMCResults(model=gal.model, sampler=sampler_dict, **kwargs_fit)

    if kwargs_fit['oversampled_chisq']:
        mcmcResults.oversample_factor_chisq = gal.data.oversample_factor_chisq

    # Do all analysis, plotting, saving:
    kwargs_all = {**kwargs_galmodel, **kwargs_fit}
    mcmcResults.analyze_plot_save_results(gal, **kwargs_all)


    # Clean up logger:
    if kwargs_fit['f_log'] is not None:
        logger.removeHandler(loggerfile)

    return mcmcResults


def fit_mpfit(gal, **kwargs):
    """
    Fit observed kinematics using MPFIT and a DYSMALPY model set.
    """

    config_c_m_data = config.Config_create_model_data(**kwargs)
    config_sim_cube = config.Config_simulate_cube(**kwargs)
    kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

    config_fit = config.Config_fit_mpfit(**kwargs)
    kwargs_fit = config_fit.dict

    # Check the FOV is large enough to cover the data output:
    dpy_utils_io._check_data_inst_FOV_compatibility(gal)

    # Create output directory
    if len(kwargs_fit['outdir']) > 0:
        if kwargs_fit['outdir'][-1] != '/': kwargs_fit['outdir'] += '/'
    ensure_dir(kwargs_fit['outdir'])

    # If the output filenames aren't defined: use default output filenames

    if kwargs_fit['save_model'] and (kwargs_fit['f_model'] is None):
        kwargs_fit['f_model'] = kwargs_fit['outdir']+'galaxy_model.pickle'
    if kwargs_fit['save_bestfit_cube'] and (kwargs_fit['f_cube'] is None):
        kwargs_fit['f_cube'] = kwargs_fit['outdir']+'mpfit_bestfit_cube.fits'
    if kwargs_fit['f_plot_bestfit'] is None:
        kwargs_fit['f_plot_bestfit'] = kwargs_fit['outdir'] + 'mpfit_best_fit.{}'.format(kwargs_fit['plot_type'])
    if kwargs_fit['save_results'] & (kwargs_fit['f_results'] is None):
        kwargs_fit['f_results'] = kwargs_fit['outdir'] + 'mpfit_results.pickle'
    if kwargs_fit['save_vel_ascii'] & (kwargs_fit['f_vel_ascii'] is None):
        kwargs_fit['f_vel_ascii'] = kwargs_fit['outdir'] + 'galaxy_bestfit_vel_profile.dat'

    if kwargs_fit['save_model_bestfit'] & (kwargs_fit['f_model_bestfit'] is None):
        if gal.data.ndim == 1:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-1dplots.txt'
        elif gal.data.ndim == 2:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-velmaps.fits'
        elif gal.data.ndim == 3:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-cube.fits'
        elif gal.data.ndim == 0:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-0d.txt'

    # ---------------------------------------------------
    # Check for existing files if overwrite=False:
    if (not kwargs_fit['overwrite']):
        fnames = []
        fnames_opt = [ kwargs_fit['f_plot_bestfit'], kwargs_fit['f_results'], kwargs_fit['f_vel_ascii'],
                        kwargs_fit['f_model'], kwargs_fit['f_cube'] ]
        for fname in fnames_opt:
            if fname is not None:
                fnames.append(fname)

        for fname in fnames:
            if fname is not None:
                if os.path.isfile(fname):
                    logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(kwargs_fit['overwrite'], fname))

        # Return early if it won't save the results, sampler:
        if os.path.isfile(kwargs_fit['f_results']):
            msg = "overwrite={}, and 'f_results' won't be saved,".format(kwargs_fit['overwrite'])
            msg += " so the fit will not be saved.\n Specify new outfile or delete old files."
            logger.warning(msg)
            return None
    # ---------------------------------------------------

    # Setup file redirect logging:
    if kwargs_fit['f_log'] is not None:
        loggerfile = logging.FileHandler(kwargs_fit['f_log'])
        loggerfile.setLevel(logging.INFO)
        logger.addHandler(loggerfile)

    # Setup the parinfo dictionary that mpfit needs
    p_initial = gal.model.get_free_parameters_values()
    pkeys = gal.model.get_free_parameter_keys()
    nparam = len(p_initial)
    parinfo = [{'value':0, 'limited': [1, 1], 'limits': [0., 0.], 'fixed': 0, 'parname':''} for i in
               range(nparam)]

    for cmp in pkeys:
        for param_name in pkeys[cmp]:

            if pkeys[cmp][param_name] != -99:

                bounds = gal.model.components[cmp].bounds[param_name]
                k = pkeys[cmp][param_name]
                parinfo[k]['limits'][0] = bounds[0]
                parinfo[k]['limits'][1] = bounds[1]
                parinfo[k]['value'] = p_initial[k]
                parinfo[k]['parname'] = '{}:{}'.format(cmp, param_name)

    # Setup dictionary of arguments that mpfit_chisq needs

    fa_init = {'gal':gal, 'fitdispersion':kwargs_fit['fitdispersion'],
                'fitflux':kwargs_fit['fitflux'], 'use_weights': kwargs_fit['use_weights']}
    fa = {**fa_init, **kwargs_galmodel}

    # Run mpfit
    # Output some fitting info to logger:
    logger.info("*************************************")
    logger.info(" Fitting: {} using MPFIT".format(gal.name))
    if gal.data.filename_velocity is not None:
        logger.info("    velocity file: {}".format(gal.data.filename_velocity))
    if gal.data.filename_dispersion is not None:
        logger.info("    dispers. file: {}".format(gal.data.filename_dispersion))

    #logger.info('\n')
    logger.info('\n'+'mvirial_tied: {}'.format(gal.model.components['halo'].mvirial.tied))
    if 'mhalo_relation' in gal.model.components['disk+bulge'].__dict__.keys():
        logger.info('mhalo_relation: {}'.format(gal.model.components['disk+bulge'].mhalo_relation))
    if 'truncate_lmstar_halo' in gal.model.components['disk+bulge'].__dict__.keys():
        logger.info('truncate_lmstar_halo: {}'.format(gal.model.components['disk+bulge'].truncate_lmstar_halo))
    logger.info('nSubpixels: {}'.format(kwargs_galmodel['oversample']))

    logger.info('\nMPFIT Fitting:\n'
                'Start: {}\n'.format(datetime.datetime.now()))
    start = time.time()

    m = mpfit(mpfit_chisq, parinfo=parinfo, functkw=fa, maxiter=kwargs_fit['maxiter'],
              iterfunct=mpfit_printer, iterkw={'logger': logger})

    end = time.time()
    elapsed = end - start
    endtime = str(datetime.datetime.now())
    timemsg = 'Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s)'.format(elapsed, np.floor(elapsed / 60.),
                                                                  (elapsed / 60. - np.floor(
                                                                      elapsed / 60.)) * 60.)
    statusmsg = 'MPFIT Status = {}'.format(m.status)
    if m.status <= 0:
        errmsg = 'MPFIT Error/Warning Message = {}'.format(m.errmsg)
    elif m.status == 5:
        errmsg = 'MPFIT Error/Warning Message = Maximum number of iterations reached. Fit may not have converged!'
    else:
        errmsg = 'MPFIT Error/Warning Message = None'

    logger.info('\nEnd: ' + endtime + '\n'
                '\n******************\n'
                '' + timemsg + '\n'
                '' + statusmsg + '\n'
                '' + errmsg + '\n'
                '******************')

    # Save all of the fitting results in an MPFITResults object
    mpfitResults = MPFITResults(model=gal.model,
                                f_plot_bestfit=kwargs_fit['f_plot_bestfit'],
                                f_results=kwargs_fit['f_results'],
                                blob_name=kwargs_fit['blob_name'])

    mpfitResults.input_results(m, gal=gal, model_key_re=kwargs_fit['model_key_re'],
                    model_key_halo=kwargs_fit['model_key_halo'])

    # Update theta to best-fit:
    gal.model.update_parameters(mpfitResults.bestfit_parameters)

    gal.create_model_data(**kwargs_galmodel)

    ###
    mpfitResults.bestfit_redchisq = chisq_red(gal, fitdispersion=kwargs_fit['fitdispersion'],
                    fitflux=kwargs_fit['fitflux'],
                    model_key_re=kwargs_fit['model_key_re'])
    mpfitResults.bestfit_chisq = chisq_eval(gal, fitdispersion=kwargs_fit['fitdispersion'],
                    fitflux=kwargs_fit['fitflux'],
                    model_key_re=kwargs_fit['model_key_re'])

    # Get vmax and vrot
    if kwargs_fit['model_key_re'] is not None:
        comp = gal.model.components.__getitem__(kwargs_fit['model_key_re'][0])
        param_i = comp.param_names.index(kwargs_fit['model_key_re'][1])
        r_eff = comp.parameters[param_i]
        mpfitResults.vrot_bestfit = gal.model.velocity_profile(1.38 * r_eff, compute_dm=False)

    mpfitResults.vmax_bestfit = gal.model.get_vmax()

    if kwargs_fit['f_results'] is not None:
        mpfitResults.save_results(filename=kwargs_fit['f_results'], overwrite=kwargs_fit['overwrite'])

    if kwargs_fit['f_model'] is not None:
        # Save model w/ updated theta equal to best-fit:
        gal.preserve_self(filename=kwargs_fit['f_model'], save_data=kwargs_fit['save_data'],
                    overwrite=kwargs_fit['overwrite'])

    if kwargs_fit['f_model_bestfit'] is not None:
        gal.save_model_data(filename=kwargs_fit['f_model_bestfit'], overwrite=kwargs_fit['overwrite'])

    if kwargs_fit['save_bestfit_cube']:
        gal.model_cube.data.write(kwargs_fit['f_cube'], overwrite=kwargs_fit['overwrite'])

    if kwargs_fit['do_plotting'] & (kwargs_fit['f_plot_bestfit'] is not None):
        plotting.plot_bestfit(mpfitResults, gal, fitdispersion=kwargs_fit['fitdispersion'],
                        fitflux=kwargs_fit['fitflux'], fileout=kwargs_fit['f_plot_bestfit'],
                        overwrite=kwargs_fit['overwrite'], **kwargs_galmodel)

    # Save velocity / other profiles to ascii file:
    if kwargs_fit['f_vel_ascii'] is not None:
        mpfitResults.save_bestfit_vel_ascii(gal, filename=kwargs_fit['f_vel_ascii'],
                model_key_re=kwargs_fit['model_key_re'], overwrite=kwargs_fit['overwrite'])

        # Clean up logger:
    if kwargs_fit['f_log'] is not None:
        logger.removeHandler(loggerfile)


    return mpfitResults


class FitResults(object):
    """
    General class to hold the results of any fitting
    """

    def __init__(self,
                 model=None,
                 f_plot_bestfit=None,
                 f_results=None,
                 fit_method=None):

        self.bestfit_parameters = None
        self.bestfit_parameters_err = None
        self.bestfit_redchisq = None
        self._fixed = None

        if model is not None:
            self.set_model(model)
        else:
            self.param_names = OrderedDict()
            self._param_keys = OrderedDict()
            self.nparams = None
            self.free_param_names = OrderedDict()
            self._free_param_keys = OrderedDict()
            self.nparams_free = None
            self.chain_param_names = None

        self.f_plot_bestfit = f_plot_bestfit
        self.f_results = f_results
        self.fit_method = fit_method

    def set_model(self, model):

        self.param_names = model.param_names.copy()
        self._param_keys = model._param_keys.copy()
        self.nparams = model.nparams
        self._fixed = model.fixed

        self.free_param_names = OrderedDict()
        self._free_param_keys = OrderedDict()
        self.chain_param_names = None

        self.nparams_free = model.nparams_free
        self.init_free_param_info(model)

    def init_free_param_info(self, model):
        """
        Initialize the free parameter info
        (similar to all params for ModelSet, but just the free parameters.)
        """
        freeparam = model.get_free_parameter_keys()

        dictfreecomp = OrderedDict()
        dictfreenames = OrderedDict()

        for key in freeparam.keys():
            dictparams = OrderedDict()
            tuplefreenames = ()
            for k in freeparam[key].keys():
                if freeparam[key][k] >= 0:
                    dictparams[k] = freeparam[key][k]
                    tuplefreenames = tuplefreenames + (k,)
            if len(dictparams) > 0:
                dictfreecomp[key] = dictparams
                dictfreenames[key] = tuplefreenames

        self.free_param_names = dictfreenames
        self._free_param_keys = dictfreecomp

        self.chain_param_names = make_arr_cmp_params(self)

    def save_results(self, filename=None, overwrite=False):
        # Check for existing file:
        if (not overwrite) and (filename is not None):
            if os.path.isfile(filename):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
                return None

        if filename is not None:
            dump_pickle(self, filename=filename, overwrite=overwrite)  # Save FitResults class to a pickle file

    def save_bestfit_vel_ascii(self, gal, filename=None, model_key_re=['disk+bulge', 'r_eff_disk'], overwrite=False):
        if filename is not None:
            try:
                # RE needs to be in kpc
                comp = gal.model.components.__getitem__(model_key_re[0])
                param_i = comp.param_names.index(model_key_re[1])
                r_eff = comp.parameters[param_i]
            except:
                r_eff = 10. / 3.
            rmax = np.max([3. * r_eff, 10.])
            stepsize = 0.1  # stepsize 0.1 kpc
            r = np.arange(0., rmax + stepsize, stepsize)

            gal.model.write_vrot_vcirc_file(r=r, filename=filename, overwrite=overwrite)

    @abc.abstractmethod
    def plot_results(self, *args, **kwargs):
        """
        Method to produce all of the necessary plots showing the results of the fitting.
        :param args:
        :param kwargs:
        :return:
        """

    def plot_bestfit(self, gal, fitdispersion=True, fitflux=False, fileout=None, overwrite=False, **kwargs_galmodel):
        """Plot/replot the bestfit for the MCMC fitting"""
        #if fileout is None:
        #    fileout = self.f_plot_bestfit
        # Check for existing file:
        if (not overwrite) and (fileout is not None):
            if os.path.isfile(fileout):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
                return None
        plotting.plot_bestfit(self, gal, fitdispersion=fitdispersion, fitflux=fitflux,
                             fileout=fileout, overwrite=overwrite, **kwargs_galmodel)

    def reload_results(self, filename=None):
        """Reload MCMC results saved earlier: the whole object"""
        if filename is None:
            filename = self.f_results
        resultsSaved = load_pickle(filename)
        for key in resultsSaved.__dict__.keys():
            try:
                self.__dict__[key] = resultsSaved.__dict__[key]
            except:
                pass

    def results_report(self, gal=None, filename=None, params=None,
                    report_type='pretty', overwrite=False, **kwargs):
        """Return a result report string, or save to file.
           report_type = 'pretty':   More human-readable
                       = 'machine':  Machine-readable ascii table (though with mixed column types)

           **kwargs: can pass other setting values: eg zcalc_truncate.
        """

        report = dpy_utils_io.create_results_report(gal, self, report_type=report_type,
                        params=params, **kwargs)

        if filename is not None:
            with open(filename, 'w') as f:
                f.write(report)
        else:
            return report



class MCMCResults(FitResults):
    """
    Class to hold results of MCMC fitting to DYSMALPY models.

    Note: emcee sampler object is ported to a dictionary in
            mcmcResults.sampler

        The name of the free parameters in the chain are accessed through:
            mcmcResults.chain_param_names,
                or more generally (separate model + parameter names) through
                mcmcResults.free_param_names
    """
    def __init__(self, model=None,
                 sampler=None,
                 f_plot_trace_burnin=None,
                 f_plot_trace=None,
                 f_burn_sampler=None,
                 f_sampler=None,
                 f_plot_param_corner=None,
                 f_plot_bestfit=None,
                 f_results=None,
                 f_chain_ascii=None,
                 linked_posterior_names=None,
                 blob_name=None,
                 **kwargs):

        self.sampler = sampler
        self.linked_posterior_names = linked_posterior_names

        self.bestfit_parameters_l68_err = None
        self.bestfit_parameters_u68_err = None
        self.bestfit_parameters_l68 = None
        self.bestfit_parameters_u68 = None

        # Filenames that are specific to MCMC fitting
        self.f_plot_trace_burnin = f_plot_trace_burnin
        self.f_plot_trace = f_plot_trace
        self.f_burn_sampler = f_burn_sampler
        self.f_sampler = f_sampler
        self.f_plot_param_corner = f_plot_param_corner
        self.f_chain_ascii = f_chain_ascii

        self.blob_name = blob_name

        super(MCMCResults, self).__init__(model=model, f_plot_bestfit=f_plot_bestfit,
                                          f_results=f_results, fit_method='MCMC')


    def analyze_plot_save_results(self, gal,
                linked_posterior_names=None,
                nPostBins=50,
                model_key_re=None,
                model_key_halo=None,
                fitdispersion=True,
                fitflux=False,
                save_data=True,
                save_bestfit_cube=False,
                f_cube=None,
                f_model=None,
                f_model_bestfit = None,
                f_vel_ascii = None,
                do_plotting = True,
                overwrite=False,
                **kwargs_galmodel):
        """
        Wrapper for post-sample analysis + plotting -- in case code broke and only have sampler saved.

        """

        if self.f_chain_ascii is not None:
            self.save_chain_ascii(filename=self.f_chain_ascii, overwrite=overwrite)

        # Get the best-fit values, uncertainty bounds from marginalized posteriors
        self.analyze_posterior_dist(gal=gal, linked_posterior_names=linked_posterior_names,
                    nPostBins=nPostBins)

        # Update theta to best-fit:
        gal.model.update_parameters(self.bestfit_parameters)

        if self.blob_name is not None:
            if isinstance(self.blob_name, str):
                blob_names = [self.blob_name]
            else:
                blob_names = self.blob_name[:]

            for blobn in blob_names:
                if blobn.lower() == 'fdm':
                    self.analyze_dm_posterior_dist(gal=gal, model_key_re=model_key_re, blob_name=self.blob_name)  # here blob_name should be the *full* list
                elif blobn.lower() == 'mvirial':
                    self.analyze_mvirial_posterior_dist(gal=gal, model_key_halo=model_key_halo, blob_name=self.blob_name)
                elif blobn.lower() == 'alpha':
                    self.analyze_alpha_posterior_dist(gal=gal, model_key_halo=model_key_halo, blob_name=self.blob_name)
                elif blobn.lower() == 'rb':
                    self.analyze_rb_posterior_dist(gal=gal, model_key_halo=model_key_halo, blob_name=self.blob_name)


        gal.create_model_data(**kwargs_galmodel)

        self.bestfit_redchisq = chisq_red(gal, fitdispersion=fitdispersion, fitflux=fitflux,
                        model_key_re=model_key_re)
        self.bestfit_chisq = chisq_eval(gal, fitdispersion=fitdispersion, fitflux=fitflux,
                                model_key_re=model_key_re)

        if model_key_re is not None:
            comp = gal.model.components.__getitem__(model_key_re[0])
            param_i = comp.param_names.index(model_key_re[1])
            r_eff = comp.parameters[param_i]
            self.vrot_bestfit = gal.model.velocity_profile(1.38*r_eff, compute_dm=False)


        self.vmax_bestfit = gal.model.get_vmax()

        if self.f_results is not None:
            self.save_results(filename=self.f_results, overwrite=overwrite)

        if f_model is not None:
            # Save model w/ updated theta equal to best-fit:
            gal.preserve_self(filename=f_model, save_data=save_data, overwrite=overwrite)



        if f_model_bestfit is not None:
            gal.save_model_data(filename=f_model_bestfit, overwrite=overwrite)

        if save_bestfit_cube:
            gal.model_cube.data.write(f_cube, overwrite=overwrite)

        # --------------------------------
        # Plot trace, if output file set
        if (do_plotting) & (self.f_plot_trace is not None) :
            plotting.plot_trace(self, fileout=self.f_plot_trace, overwrite=overwrite)

        # --------------------------------
        # Plot results: corner plot, best-fit
        if (do_plotting) & (self.f_plot_param_corner is not None):
            plotting.plot_corner(self, gal=gal, fileout=self.f_plot_param_corner, blob_name=self.blob_name, overwrite=overwrite)

        if (do_plotting) & (self.f_plot_bestfit is not None):
            plotting.plot_bestfit(self, gal, fitdispersion=fitdispersion, fitflux=fitflux,
                                  fileout=self.f_plot_bestfit, overwrite=overwrite, **kwargs_galmodel)

        # --------------------------------
        # Save velocity / other profiles to ascii file:
        if f_vel_ascii is not None:
            self.save_bestfit_vel_ascii(gal, filename=f_vel_ascii, model_key_re=model_key_re, overwrite=overwrite)


    def mod_linear_param_posterior(self, gal=None):
        linear_posterior = []
        j = -1
        for cmp in gal.model.fixed:
            # pkeys[cmp] = OrderedDict()
            for pm in gal.model.fixed[cmp]:
                if gal.model.fixed[cmp][pm] | np.bool(gal.model.tied[cmp][pm]):
                    pass
                else:
                    j += 1
                    if isinstance(gal.model.components[cmp].__getattribute__(pm).prior, UniformLinearPrior):
                        self.sampler['flatchain'][:,j] = np.power(10.,self.sampler['flatchain'][:,j])
                        linear_posterior.append(True)
                    else:
                        linear_posterior.append(False)

        self.linear_posterior = linear_posterior

    def back_map_linear_param_bestfits(self, mcmc_param_bestfit, mcmc_limits, mcmc_limits_percentile):
        mcmc_param_bestfit_linear = mcmc_param_bestfit.copy()
        mcmc_limits_linear = mcmc_limits.copy()

        for j in range(len(mcmc_param_bestfit)):
            if self.linear_posterior[j]:
                mcmc_param_bestfit[j] = np.log10(mcmc_param_bestfit[j])
                mcmc_limits[:,j] = np.log10(mcmc_limits[:, j])
                mcmc_limits_percentile[:,j] = np.log10(mcmc_limits_percentile[:, j])

        return mcmc_param_bestfit, mcmc_param_bestfit_linear, mcmc_limits, mcmc_limits_linear, mcmc_limits_percentile

    def analyze_posterior_dist(self, gal=None, linked_posterior_names=None, nPostBins=50):
        """
        Default analysis of posterior distributions from MCMC fitting:
            look at marginalized posterior distributions, and
            extract the best-fit value (peak of KDE), and extract the +- 1 sigma uncertainty bounds
            (eg, the 16%/84% distribution of posteriors)

        Optional input:
        linked_posterior_names: indicate if best-fit of parameters
                                should be measured in multi-D histogram space
                                format: set of linked parameter sets, with each linked parameter set
                                        consisting of len-2 tuples/lists of the
                                        component+parameter names.


        Structure explanation:
        (1) Want to analyze component+param 1 and 2 together, and then
            3 and 4 together.

            Input structure would be:
                linked_posterior_names = [ joint_param_bundle1, joint_param_bundle2 ]
                with
                join_param_bundle1 = [ [cmp1, par1], [cmp2, par2] ]
                jont_param_bundle2 = [ [cmp3, par3], [cmp4, par4] ]
                for a full array of:
                linked_posterior_names =
                    [ [ [cmp1, par1], [cmp2, par2] ], [ [cmp3, par3], [cmp4, par4] ] ]

        (2) Want to analyze component+param 1 and 2 together:
            linked_posterior_names = [ joint_param_bundle1 ]
            with
            join_param_bundle1 = [ [cmp1, par1], [cmp2, par2] ]

            for a full array of:
                linked_posterior_names = [ [ [cmp1, par1], [cmp2, par2] ] ]

                eg: look at halo: mvirial and disk+bulge: total_mass together
                    linked_posterior_names = [[['halo', 'mvirial'], ['disk+bulge', 'total_mass']]]
                    or linked_posterior_names = [[('halo', 'mvirial'), ('disk+bulge', 'total_mass')]]

        """

        if self.sampler is None:
            raise ValueError("MCMC.sampler must be set to analyze the posterior distribution.")

        self.mod_linear_param_posterior(gal=gal)

        # Unpack MCMC samples: lower, upper 1, 2 sigma
        mcmc_limits_percentile = np.percentile(self.sampler['flatchain'], [15.865, 84.135], axis=0)

        mcmc_limits = shortest_span_bounds(self.sampler['flatchain'], percentile=0.6827)


        ## location of peaks of *marginalized histograms* for each parameter
        mcmc_peak_hist = np.zeros(self.sampler['flatchain'].shape[1])
        for i in six.moves.xrange(self.sampler['flatchain'].shape[1]):
            yb, xb = np.histogram(self.sampler['flatchain'][:,i], bins=nPostBins)
            wh_pk = np.where(yb == yb.max())[0][0]
            mcmc_peak_hist[i] = np.average([xb[wh_pk], xb[wh_pk+1]])

        ## Use max prob as guess to get peak value of the gaussian KDE, to find 'best-fit' of the posterior:
        mcmc_param_bestfit = find_peak_gaussian_KDE(self.sampler['flatchain'], mcmc_peak_hist)

        # --------------------------------------------
        if linked_posterior_names is not None:
            # Make sure the param of self is updated
            #   (for ref. when reloading saved mcmcResult objects)

            self.linked_posterior_names = linked_posterior_names
            linked_posterior_ind_arr = get_linked_posterior_indices(self,
                            linked_posterior_names=linked_posterior_names)

            guess = mcmc_param_bestfit.copy()

            bestfit_theta_linked = get_linked_posterior_peak_values(self.sampler['flatchain'],
                            guess=guess,
                            linked_posterior_ind_arr=linked_posterior_ind_arr,
                            nPostBins=nPostBins)

            for k in six.moves.xrange(len(linked_posterior_ind_arr)):
                for j in six.moves.xrange(len(linked_posterior_ind_arr[k])):
                    mcmc_param_bestfit[linked_posterior_ind_arr[k][j]] = bestfit_theta_linked[k][j]



        # --------------------------------------------
        # Uncertainty bounds are currently determined from marginalized posteriors
        #   (even if the best-fit is found from linked posterior).

        # --------------------------------------------
        # Save best-fit results in the MCMCResults instance

        self.bestfit_parameters = mcmc_param_bestfit
        self.bestfit_redchisq = None

        # ++++++++++++++++++++++++=
        # Original 68% percentile interval:
        mcmc_stack_percentile = np.concatenate(([mcmc_param_bestfit], mcmc_limits_percentile), axis=0)
        # Order: best fit value, lower 1sig bound, upper 1sig bound

        mcmc_uncertainties_1sig_percentile = np.array(list(map(lambda v: (v[0]-v[1], v[2]-v[0]),
                            list(zip(*mcmc_stack_percentile)))))

        # 1sig lower, upper uncertainty
        self.bestfit_parameters_err_percentile = mcmc_uncertainties_1sig_percentile

        # Bound limits (in case it's useful)
        self.bestfit_parameters_l68_percentile = mcmc_limits_percentile[0]
        self.bestfit_parameters_u68_percentile = mcmc_limits_percentile[1]

        # Separate 1sig l, u uncertainty, for utility:
        self.bestfit_parameters_l68_err_percentile = mcmc_param_bestfit - mcmc_limits_percentile[0]
        self.bestfit_parameters_u68_err_percentile = mcmc_limits_percentile[1] - mcmc_param_bestfit


        # ++++++++++++++++++++++++=
        # From new shortest credible interval:
        mcmc_stack = np.concatenate(([mcmc_param_bestfit], mcmc_limits), axis=0)
        # Order: best fit value, lower 1sig bound, upper 1sig bound

        mcmc_uncertainties_1sig = np.array(list(map(lambda v: (v[0]-v[1], v[2]-v[0]),
                            list(zip(*mcmc_stack)))))

        # 1sig lower, upper uncertainty
        self.bestfit_parameters_err = mcmc_uncertainties_1sig

        # Bound limits (in case it's useful)
        self.bestfit_parameters_l68 = mcmc_limits[0]
        self.bestfit_parameters_u68 = mcmc_limits[1]

        # Separate 1sig l, u uncertainty, for utility:
        self.bestfit_parameters_l68_err = mcmc_param_bestfit - mcmc_limits[0]
        self.bestfit_parameters_u68_err = mcmc_limits[1] - mcmc_param_bestfit


    def analyze_blob_posterior_dist(self, bestfit=None, parname=None, blob_name=None):
        # Eg: parname = 'fdm' / 'mvirial' / 'alpha'
        if self.sampler is None:
            raise ValueError("MCMC.sampler must be set to analyze the posterior distribution.")

        if ('flatblobs' not in self.sampler.keys()):
            if len(self.sampler['blobs'].shape) == 2:
                # Only 1 blob: nSteps, nWalkers:
                flatblobs = self.sampler['blobs'].reshape(-1)
            elif len(self.sampler['blobs'].shape) == 3:
                # Multiblobs; nSteps, nWalkers, nBlobs
                flatblobs = self.sampler['blobs'].reshape(-1,self.sampler['blobs'].shape[2])
            else:
                raise ValueError("Sampler blob length not recognized")

            self.sampler['flatblobs'] = flatblobs

        if isinstance(blob_name, str):
            blobs = self.sampler['flatblobs']
            pname = parname.strip()
        else:
            pname = parname.strip()
            indv = blob_name.index(pname)
            blobs = self.sampler['flatblobs'][:,indv]

        # Unpack MCMC samples: lower, upper 1, 2 sigma
        mcmc_limits_percentile = np.percentile(blobs, [15.865, 84.135], axis=0)

        mcmc_limits = shortest_span_bounds(blobs, percentile=0.6827)

        # --------------------------------------------
        # Save best-fit results in the MCMCResults instance
        self.__dict__['bestfit_{}'.format(pname)] = bestfit
        self.__dict__['bestfit_{}_l68_err'.format(pname)] = bestfit - mcmc_limits[0]
        self.__dict__['bestfit_{}_u68_err'.format(pname)] = mcmc_limits[1] - bestfit


        self.__dict__['bestfit_{}_l68_err_percentile'.format(pname)] = bestfit - mcmc_limits_percentile[0]
        self.__dict__['bestfit_{}_u68_err_percentile'.format(pname)] = mcmc_limits_percentile[1] - bestfit


    def analyze_dm_posterior_dist(self, gal=None, model_key_re=None, blob_name=None):
        """
        Default analysis of posterior distributions of fDM from MCMC fitting:
            look at marginalized posterior distributions, and
            extract the best-fit value (peak of KDE), and extract the +- 1 sigma uncertainty bounds
            (eg, the 16%/84% distribution of posteriors)

        """
        fdm_mcmc_param_bestfit = gal.model.get_dm_frac_effrad(model_key_re=model_key_re)
        self.analyze_blob_posterior_dist(bestfit=fdm_mcmc_param_bestfit, parname='fdm', blob_name=blob_name)

    def analyze_mvirial_posterior_dist(self, gal=None, model_key_halo=None, blob_name=None):
        mvirial_mcmc_param_bestfit = gal.model.get_mvirial(model_key_halo=model_key_halo)
        self.analyze_blob_posterior_dist(bestfit=mvirial_mcmc_param_bestfit, parname='mvirial', blob_name=blob_name)

    def analyze_alpha_posterior_dist(self, gal=None, model_key_halo=None, blob_name=None):
        alpha_mcmc_param_bestfit = gal.model.get_halo_alpha(model_key_halo=model_key_halo)
        self.analyze_blob_posterior_dist(bestfit=alpha_mcmc_param_bestfit, parname='alpha', blob_name=blob_name)

    def analyze_rb_posterior_dist(self, gal=None, model_key_halo=None, blob_name=None):
        rb_mcmc_param_bestfit = gal.model.get_halo_rb(model_key_halo=model_key_halo)
        self.analyze_blob_posterior_dist(bestfit=rb_mcmc_param_bestfit, parname='rb', blob_name=blob_name)

    def get_uncertainty_ellipse(self, namex=None, namey=None, bins=50):
        """
        Using component name, get sampler chain for param x and y, and estimate joint uncertainty ellipse
        Input:
            name[x,y]:      List: ['flatchain', ind] or ['flatblobs', ind]
        """
        try:
            chain_x = self.sampler[namex[0]][:,namex[1]]
        except:
            # eg, Single blob value flatblobs
            chain_x = self.sampler[namex[0]]
        try:
            chain_y = self.sampler[namey[0]][:,namey[1]]
        except:
            # eg, Single blob value flatblobs
            chain_y = self.sampler[namey[0]]

        PA, stddev_x, stddev_y  = fit_uncertainty_ellipse(chain_x, chain_y, bins=bins)
        return PA, stddev_x, stddev_y


    def save_chain_ascii(self, filename=None, overwrite=False):
        # Check for existing file:
        if (not overwrite) and (filename is not None):
            if os.path.isfile(filename):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
                return None
        if filename is not None:
            try:
                blobs = self.sampler['blobs']
                blobset = True
            except:
                blobset = False

            if ('flatblobs' not in self.sampler.keys()) & (blobset):
                if len(self.sampler['blobs'].shape) == 2:
                    # Only 1 blob: nSteps, nWalkers:
                    flatblobs = self.sampler['blobs'].reshape(-1)
                elif len(self.sampler['blobs'].shape) == 3:
                    # Multiblobs; nSteps, nWalkers, nBlobs
                    flatblobs = self.sampler['blobs'].reshape(-1,self.sampler['blobs'].shape[2])
                else:
                    raise ValueError("Sampler blob length not recognized")

                self.sampler['flatblobs'] = flatblobs

            with open(filename, 'w') as f:
                namestr = '#'
                namestr += '  '.join(map(str, self.chain_param_names))
                if blobset:
                    # Currently assuming blob only returns DM fraction
                    if isinstance(self.blob_name, str):
                        namestr += '  {}'.format(self.blob_name)
                    else:
                        for blobn in self.blob_name:
                            namestr += '  {}'.format(blobn)
                f.write(namestr+'\n')

                # flatchain shape: (flat)step, params
                for i in six.moves.xrange(self.sampler['flatchain'].shape[0]):
                    datstr = '  '.join(map(str, self.sampler['flatchain'][i,:]))
                    if blobset:
                        if isinstance(self.blob_name, str):
                            datstr += '  {}'.format(self.sampler['flatblobs'][i])
                        else:
                            for k in range(len(self.blob_name)):
                                datstr += '  {}'.format(self.sampler['flatblobs'][i,k])

                    f.write(datstr+'\n')



    def reload_sampler(self, filename=None):
        """Reload the MCMC sampler saved earlier"""
        if filename is None:
            filename = self.f_sampler

        hdf5_aliases = ['h5', 'hdf5']
        pickle_aliases = ['pickle', 'pkl', 'pcl']
        if (filename.split('.')[-1].lower() in hdf5_aliases):
            self.sampler = _reload_sampler_hdf5(filename=filename)

        elif (filename.split('.')[-1].lower() in pickle_aliases):
            self.sampler = _reload_sampler_pickle(filename=filename)



    def plot_results(self, gal, fitdispersion=True, fitflux=False,
                     f_plot_param_corner=None, f_plot_bestfit=None, f_plot_trace=None,
                     overwrite=False, **kwargs_galmodel):
        """Plot/replot the corner plot and bestfit for the MCMC fitting"""
        self.plot_corner(gal=gal, fileout=f_plot_param_corner, overwrite=overwrite)
        self.plot_bestfit(gal, fitdispersion=fitdispersion, fitflux=fitflux,
                fileout=f_plot_bestfit, overwrite=overwrite, **kwargs_galmodel)
        self.plot_trace(fileout=f_plot_trace, overwrite=overwrite)


    def plot_corner(self, gal=None, fileout=None, overwrite=False):
        """Plot/replot the corner plot for the MCMC fitting"""
        plotting.plot_corner(self, gal=gal, fileout=fileout, blob_name=self.blob_name, overwrite=overwrite)


    def plot_trace(self, fileout=None, overwrite=False):
        """Plot/replot the trace for the MCMC fitting"""
        plotting.plot_trace(self, fileout=fileout, overwrite=overwrite)


class MPFITResults(FitResults):
    """
    Class to hold results of using MPFIT to fit to DYSMALPY models.
    """
    def __init__(self, model=None, f_plot_bestfit=None, f_results=None, blob_name=None, **kwargs):

        self._mpfit_object = None


        self.blob_name = blob_name

        super(MPFITResults, self).__init__(model=model, f_plot_bestfit=f_plot_bestfit,
                                         f_results=f_results, fit_method='MPFIT')

    def input_results(self, mpfit_obj, gal=None,
                    model_key_re=None, model_key_halo=None):
        """
        Save the best fit results from MPFIT in the MPFITResults object
        """

        self._mpfit_object = mpfit_obj
        if 'blas_enorm' in self._mpfit_object.__dict__.keys():
            # Can't pickle this object if this is a FORTRAN OBJECT // eg as defined in mpfit.py
            self._mpfit_object.blas_enorm = None
        self.status = mpfit_obj.status
        self.errmsg = mpfit_obj.errmsg
        self.niter = mpfit_obj.niter

        # Populate the self.bestfit_parameters attribute with the bestfit values for the
        # free parameters
        self.bestfit_parameters = mpfit_obj.params
        self.bestfit_parameters_err = mpfit_obj.perror

        if mpfit_obj.status > 0:
            self.bestfit_redchisq = mpfit_obj.fnorm/mpfit_obj.dof


        # Add "blob" bestfit:
        if self.blob_name is not None:
            if isinstance(self.blob_name, str):
                blob_names = [self.blob_name]
            else:
                blob_names = self.blob_name[:]

            for blobn in blob_names:
                if blobn.lower() == 'fdm':
                    param_bestfit = gal.model.get_dm_frac_effrad(model_key_re=model_key_re)
                elif blobn.lower() == 'mvirial':
                    param_bestfit = gal.model.get_mvirial(model_key_halo=model_key_halo)
                elif blobn.lower() == 'alpha':
                    param_bestfit = gal.model.get_halo_alpha(model_key_halo=model_key_halo)
                elif blobn.lower() == 'rb':
                    param_bestfit = gal.model.get_halo_rb(model_key_halo=model_key_halo)

                self.analyze_blob_value(bestfit=param_bestfit, parname=blobn.lower())




    def analyze_blob_value(self, bestfit=None, parname=None):
        # Eg: parname = 'fdm' / 'mvirial' / 'alpha'
        pname = parname.strip()
        # In case ever want to do error propagation here
        err_fill = -99.
        # --------------------------------------------
        # Save best-fit results in the MCMCResults instance
        self.__dict__['bestfit_{}'.format(pname)] = bestfit
        self.__dict__['bestfit_{}_err'.format(pname)] = err_fill


    def plot_results(self, gal, fitdispersion=True, fitflux=False,
                     f_plot_bestfit=None, overwrite=False, **kwargs_galmodel):
        """Plot/replot the corner plot and bestfit for the MCMC fitting"""
        self.plot_bestfit(gal, fitdispersion=fitdispersion, fitflux=fitflux,
                         fileout=f_plot_bestfit, overwrite=overwrite, **kwargs_galmodel)


def log_prob(theta, gal,
             red_chisq=False,
             oversampled_chisq=None,
             fitdispersion=True,
             fitflux=False,
             blob_name = None,
             model_key_re=None,
             model_key_halo=None,
             **kwargs_galmodel):
    """
    Evaluate the log probability of the given model
    """

    # Update the parameters
    gal.model.update_parameters(theta)

    # Evaluate prior prob of theta
    lprior = gal.model.get_log_prior()

    # First check to see if log prior is finite
    if not np.isfinite(lprior):
        if blob_name is not None:
            if isinstance(blob_name, str):
                return -np.inf, -np.inf
            else:
                return -np.inf, [-np.inf]*len(blob_name)
        else:
            return -np.inf
    else:
        # Update the model data
        gal.create_model_data(**kwargs_galmodel)

        # Evaluate likelihood prob of theta
        llike = log_like(gal, red_chisq=red_chisq,
                    oversampled_chisq=oversampled_chisq,
                    fitdispersion=fitdispersion,
                    fitflux=fitflux,
                    blob_name=blob_name,
                    model_key_re=model_key_re, model_key_halo=model_key_halo)

        if blob_name is not None:
            lprob = lprior + llike[0]
        else:
            lprob = lprior + llike

        if not np.isfinite(lprob):
            # Make sure the non-finite ln_prob is -Inf, for emcee handling
            lprob = -np.inf

        if blob_name is not None:
            if len(llike) == 2:
                return lprob, llike[1]
            else:
                return lprob, llike[1:]
        else:
            return lprob


def log_like(gal, red_chisq=False,
                oversampled_chisq=None,
                fitdispersion=True,
                fitflux=False,
                blob_name=None,
                model_key_re=None,
                model_key_halo=None):

    # Temporary: testing:
    if oversampled_chisq is None:
        raise ValueError

    if red_chisq:
        raise ValueError("red_chisq=True is currently *DISABLED* to test lnlike impact vs lnprior")


    if gal.data.ndim == 3:
        # Will have problem with vel shift: data, model won't match...

        msk = gal.data.mask
        dat = gal.data.data.unmasked_data[:].value[msk]
        mod = gal.model_data.data.unmasked_data[:].value[msk]
        err = gal.data.error.unmasked_data[:].value[msk]

        # Weights:
        wgt = 1.
        if hasattr(gal.data, 'weight'):
            if gal.data.weight is not None:
                wgt = gal.data.weight[msk]

        # Artificially mask zero errors which are masked
        #err[((err==0) & (msk==0))] = 99.
        chisq_arr_raw = (((dat - mod)/err)**2) * wgt + np.log( (2.*np.pi*err**2) / wgt )
        if oversampled_chisq:
            invnu = 1. / gal.data.oversample_factor_chisq
        elif red_chisq:
            if gal.model.nparams_free > np.sum(msk) :
                raise ValueError("More free parameters than data points!")
            invnu = 1./ (1.*(np.sum(msk) - gal.model.nparams_free))
        else:
            invnu = 1.
        llike = -0.5*chisq_arr_raw.sum() * invnu



    elif (gal.data.ndim == 1) or (gal.data.ndim ==2):

        #msk = gal.data.mask
        if hasattr(gal.data, 'mask_velocity'):
            if gal.data.mask_velocity is not None:
                msk = gal.data.mask_velocity
            else:
                msk = gal.data.mask
        else:
            msk = gal.data.mask

        vel_dat = gal.data.data['velocity'][msk]
        vel_mod = gal.model_data.data['velocity'][msk]
        vel_err = gal.data.error['velocity'][msk]

        if hasattr(gal.data, 'mask_vel_disp'):
            if gal.data.mask_vel_disp is not None:
                msk = gal.data.mask_vel_disp
            else:
                msk = gal.data.mask
        else:
            msk = gal.data.mask

        disp_dat = gal.data.data['dispersion'][msk]
        disp_mod = gal.model_data.data['dispersion'][msk]
        disp_err = gal.data.error['dispersion'][msk]

        if fitflux:
            msk = gal.data.mask
            flux_dat = gal.data.data['flux'][msk]
            flux_mod = gal.model_data.data['flux'][msk]
            if gal.data.error['flux'] is not None:
                flux_err = gal.data.error['flux'][msk]
            else:
                flux_err = 0.1 * gal.data.data['flux'][msk] # PLACEHOLDER

        wgt = 1.
        if hasattr(gal.data, 'weight'):
            if gal.data.weight is not None:
                wgt = gal.data.weight[msk]


        # Correct model for instrument dispersion if the data is instrument corrected:
        if 'inst_corr' in gal.data.data.keys():
            if gal.data.data['inst_corr']:
                disp_mod = np.sqrt(disp_mod**2 -
                                   gal.instrument.lsf.dispersion.to(u.km/u.s).value**2)
                disp_mod[~np.isfinite(disp_mod)] = 0   # Set the dispersion to zero when its below
                                                       # below the instrumental dispersion

        # Includes velocity shift
        chisq_arr_raw_vel = ((((vel_dat - vel_mod)/vel_err)**2) * wgt +
                               np.log( (2.*np.pi*vel_err**2) / wgt ))

        #####
        # Data includes velocity
        fac_mask = 1
        chisq_arr_sum = chisq_arr_raw_vel.sum()

        if fitdispersion:
            fac_mask += 1
            chisq_arr_raw_disp = ((((disp_dat - disp_mod)/disp_err)**2) * wgt +
                                    np.log( (2.*np.pi*disp_err**2) / wgt))
            chisq_arr_sum += chisq_arr_raw_disp.sum()

        if fitflux:
            fac_mask += 1
            chisq_arr_raw_flux = ((((flux_dat - flux_mod)/flux_err)**2) * wgt +
                                    np.log( (2.*np.pi*flux_err**2) / wgt))
            chisq_arr_sum += chisq_arr_raw_flux.sum()

        ####

        if oversampled_chisq:
            invnu = 1. / gal.data.oversample_factor_chisq
        elif red_chisq:
            if gal.model.nparams_free > fac_mask*np.sum(msk) :
                raise ValueError("More free parameters than data points!")
            invnu = 1./ (1.*(fac_mask*np.sum(msk) - gal.model.nparams_free))
        else:
            invnu = 1.


        ####
        llike = -0.5*(chisq_arr_sum) * invnu


    elif gal.data.ndim == 0:

        msk = gal.data.mask
        data = gal.data.data
        mod = gal.model_data.data
        err = gal.data.error

        wgt = 1.
        if hasattr(gal.data, 'weight'):
            if gal.data.weight is not None:
                wgt = gal.data.weight


        chisq_arr = ((((data - mod)/err)**2) * wgt + np.log((2.*np.pi*err**2) / wgt))
        #
        if oversampled_chisq:
            invnu = 1. / gal.data.oversample_factor_chisq
        elif red_chisq:
            if gal.model.nparams_free > np.sum(msk):
                raise ValueError("More free parameters than data points!")

            invnu = 1. / (1. * (np.sum(msk) - gal.model.nparams_free))

        else:
            invnu = 1.

        llike = -0.5*chisq_arr.sum() * invnu


    else:
        logger.warning("ndim={} not supported!".format(gal.data.ndim))
        raise ValueError

    ####
    if blob_name is not None:
        if isinstance(blob_name, str):
            # Single blob
            blob_arr = [blob_name]
        else:
            # Array of blobs
            blob_arr = blob_name[:]

        #
        blobvals = []
        for blobn in blob_arr:
            if blobn.lower() == 'fdm':
                blobv = gal.model.get_dm_frac_effrad(model_key_re=model_key_re)
            elif blobn.lower() == 'mvirial':
                blobv = gal.model.get_mvirial(model_key_halo=model_key_halo)
            elif blobn.lower() == 'alpha':
                blobv = gal.model.get_halo_alpha(model_key_halo=model_key_halo)
            elif blobn.lower() == 'rb':
                blobv = gal.model.get_halo_rb(model_key_halo=model_key_halo)
            #
            blobvals.append(blobv)

        # Preserve old behavior if there's only a single blob value: return float blob
        if isinstance(blob_name, str):
            blobvals = blobvals[0]


        return llike, blobvals

    else:
        return llike

def chisq_eval(gal, fitdispersion=True, fitflux=False,
                use_weights=False,
                model_key_re=['disk+bulge','r_eff_disk']):
    #
    if gal.data.ndim == 3:
        # Will have problem with vel shift: data, model won't match...

        msk = gal.data.mask
        dat = gal.data.data.unmasked_data[:].value[msk]
        mod = gal.model_data.data.unmasked_data[:].value[msk]
        err = gal.data.error.unmasked_data[:].value[msk]

        # Weights:
        wgt = 1.
        if use_weights:
            if hasattr(gal.data, 'weight'):
                if gal.data.weight is not None:
                    wgt = gal.data.weight[msk]


        # Artificially mask zero errors which are masked
        #err[((err==0) & (msk==0))] = 99.
        chisq_arr_raw = (((dat - mod)/err)**2) * wgt
        invnu = 1.
        chsq = chisq_arr_raw.sum() * invnu

    elif (gal.data.ndim == 1) or (gal.data.ndim ==2):

        #msk = gal.data.mask
        if hasattr(gal.data, 'mask_velocity'):
            if gal.data.mask_velocity is not None:
                msk = gal.data.mask_velocity
            else:
                msk = gal.data.mask
        else:
            msk = gal.data.mask

        vel_dat = gal.data.data['velocity'][msk]
        vel_mod = gal.model_data.data['velocity'][msk]
        vel_err = gal.data.error['velocity'][msk]

        if hasattr(gal.data, 'mask_vel_disp'):
            if gal.data.mask_vel_disp is not None:
                msk = gal.data.mask_vel_disp
            else:
                msk = gal.data.mask
        else:
            msk = gal.data.mask

        disp_dat = gal.data.data['dispersion'][msk]
        disp_mod = gal.model_data.data['dispersion'][msk]
        disp_err = gal.data.error['dispersion'][msk]

        if fitflux:
            msk = gal.data.mask
            flux_dat = gal.data.data['flux'][msk]
            flux_mod = gal.model_data.data['flux'][msk]
            try:
                flux_err = gal.data.error['flux'][msk]
            except:
                flux_err = 0.1*gal.data.data['flux'][msk] # PLACEHOLDER

        # Weights:
        wgt = 1.
        if use_weights:
            if hasattr(gal.data, 'weight'):
                if gal.data.weight is not None:
                    wgt = gal.data.weight[msk]

        # Correct model for instrument dispersion if the data is instrument corrected:
        if 'inst_corr' in gal.data.data.keys():
            if gal.data.data['inst_corr']:
                disp_mod = np.sqrt(disp_mod**2 -
                                   gal.instrument.lsf.dispersion.to(u.km/u.s).value**2)
                disp_mod[~np.isfinite(disp_mod)] = 0   # Set the dispersion to zero when its below
                                                       # below the instrumental dispersion

        # Includes velocity shift

        chisq_arr_raw_vel = (((vel_dat - vel_mod)/vel_err)**2) * wgt
        invnu = 1.

        chisq_sum = chisq_arr_raw_vel.sum()

        if fitdispersion:
            chisq_arr_raw_disp = (((disp_dat - disp_mod)/disp_err)**2) * wgt
            chisq_sum += chisq_arr_raw_disp.sum()
        if fitflux:
            chisq_arr_raw_flux = (((flux_dat - flux_mod)/flux_err)**2) * wgt
            chisq_sum += chisq_arr_raw_flux.sum()


        chsq = ( chisq_sum ) * invnu



    elif gal.data.ndim == 0:

        msk = gal.data.mask
        data = gal.data.data
        mod = gal.model_data.data
        err = gal.data.error

        # Weights:
        wgt = 1.
        if use_weights:
            if hasattr(gal.data, 'weight'):
                if gal.data.weight is not None:
                    wgt = gal.data.weight


        chisq_arr = ((data - mod)/err)**2 * wgt

        chsq = chisq_arr.sum()

    else:
        logger.warning("ndim={} not supported!".format(gal.data.ndim))
        raise ValueError


    return chsq


def chisq_red(gal, fitdispersion=True, fitflux=False, use_weights=False,
                model_key_re=['disk+bulge','r_eff_disk']):
    red_chisq = True
    if gal.data.ndim == 3:
        # Will have problem with vel shift: data, model won't match...

        msk = gal.data.mask
        dat = gal.data.data.unmasked_data[:].value[msk]
        mod = gal.model_data.data.unmasked_data[:].value[msk]
        err = gal.data.error.unmasked_data[:].value[msk]

        # Weights:
        wgt = 1.
        if use_weights:
            if hasattr(gal.data, 'weight'):
                if gal.data.weight is not None:
                    wgt = gal.data.weight[msk]


        # Artificially mask zero errors which are masked
        #err[((err==0) & (msk==0))] = 99.
        chisq_arr_raw = (((dat - mod)/err)**2) * wgt
        if red_chisq:
            if gal.model.nparams_free > np.sum(msk) :
                raise ValueError("More free parameters than data points!")
            invnu = 1./ (1.*(np.sum(msk) - gal.model.nparams_free))
        else:
            invnu = 1.
        redchsq = chisq_arr_raw.sum() * invnu



    elif (gal.data.ndim == 1) or (gal.data.ndim ==2):

        #msk = gal.data.mask
        if hasattr(gal.data, 'mask_velocity'):
            if gal.data.mask_velocity is not None:
                msk = gal.data.mask_velocity
            else:
                msk = gal.data.mask
        else:
            msk = gal.data.mask

        vel_dat = gal.data.data['velocity'][msk]
        vel_mod = gal.model_data.data['velocity'][msk]
        vel_err = gal.data.error['velocity'][msk]

        #
        if hasattr(gal.data, 'mask_vel_disp'):
            if gal.data.mask_vel_disp is not None:
                msk = gal.data.mask_vel_disp
            else:
                msk = gal.data.mask
        else:
            msk = gal.data.mask
        disp_dat = gal.data.data['dispersion'][msk]
        disp_mod = gal.model_data.data['dispersion'][msk]
        disp_err = gal.data.error['dispersion'][msk]

        if fitflux:
            msk = gal.data.mask
            flux_dat = gal.data.data['flux'][msk]
            flux_mod = gal.model_data.data['flux'][msk]
            try:
                flux_err = gal.data.error['flux'][msk]
            except:
                flux_err = 0.1*gal.data.data['flux'][msk] # PLACEHOLDER



        # Weights:
        wgt = 1.
        if use_weights:
            if hasattr(gal.data, 'weight'):
                if gal.data.weight is not None:
                    wgt = gal.data.weight[msk]


        # Correct model for instrument dispersion if the data is instrument corrected:
        if 'inst_corr' in gal.data.data.keys():
            if gal.data.data['inst_corr']:
                disp_mod = np.sqrt(disp_mod**2 -
                                   gal.instrument.lsf.dispersion.to(u.km/u.s).value**2)
                disp_mod[~np.isfinite(disp_mod)] = 0   # Set the dispersion to zero when its below
                                                       # below the instrumental dispersion


        #####

        ### Data includes velocity
        # Includes velocity shift
        chisq_arr_raw_vel = (((vel_dat - vel_mod)/vel_err)**2) * wgt
        fac_mask = 1
        chisq_arr_sum = chisq_arr_raw_vel.sum()

        if fitdispersion:
            fac_mask += 1
            chisq_arr_raw_disp = (((disp_dat - disp_mod)/disp_err)**2) * wgt
            chisq_arr_sum += chisq_arr_raw_disp.sum()

        if fitflux:
            fac_mask += 1
            chisq_arr_raw_flux = (((flux_dat - flux_mod)/flux_err)**2) * wgt
            chisq_arr_sum += chisq_arr_raw_flux.sum()

        ####
        if red_chisq:
            if gal.model.nparams_free > fac_mask*np.sum(msk) :
                raise ValueError("More free parameters than data points!")
            invnu = 1./ (1.*(fac_mask*np.sum(msk) - gal.model.nparams_free))


        ####
        redchsq = (chisq_arr_sum) * invnu


    elif gal.data.ndim == 0:

        msk = gal.data.mask
        data = gal.data.data
        mod = gal.model_data.data
        err = gal.data.error

        # Weights:
        wgt = 1.
        if use_weights:
            if hasattr(gal.data, 'weight'):
                if gal.data.weight is not None:
                    wgt = gal.data.weight


        chisq_arr = (((data - mod)/err)**2) * wgt
        if red_chisq:
            if gal.model.nparams_free > np.sum(msk):
                raise ValueError("More free parameters than data points!")

            invnu = 1. / (1. * (np.sum(msk) - gal.model.nparams_free))

        else:
            invnu = 1.

        redchsq = -0.5*chisq_arr.sum() * invnu

    else:
        logger.warning("ndim={} not supported!".format(gal.data.ndim))
        raise ValueError


    return redchsq

def mpfit_chisq(theta, fjac=None, gal=None,fitdispersion=True, fitflux=False,
                use_weights=False, **kwargs_galmodel):

    gal.model.update_parameters(theta)
    gal.create_model_data(**kwargs_galmodel)

    if gal.data.ndim == 3:
        dat = gal.data.data.unmasked_data[:].value
        mod = gal.model_data.data.unmasked_data[:].value
        err = gal.data.error.unmasked_data[:].value
        msk = gal.data.mask

        # Weights:
        if use_weights:
            if hasattr(gal.data, 'weight'):
                if gal.data.weight is not None:
                    wgt = gal.data.weight
                else:
                    wgt = 1.
            else:
                wgt = 1.
        else:
            wgt = 1.

        # Artificially mask zero errors which are masked
        err[((err == 0) & (msk == 0))] = 99.
        chisq_arr_raw = msk * (((dat - mod) / err)) * np.sqrt(wgt)
        chisq_arr_raw = chisq_arr_raw.flatten()

    elif (gal.data.ndim == 1) or (gal.data.ndim == 2):

        #msk = gal.data.mask
        if hasattr(gal.data, 'mask_velocity'):
            if gal.data.mask_velocity is not None:
                msk = gal.data.mask_velocity
            else:
                msk = gal.data.mask
        else:
            msk = gal.data.mask

        # Weights:
        if use_weights:
            if hasattr(gal.data, 'weight'):
                if gal.data.weight is not None:
                    wgt = gal.data.weight[msk]
                else:
                    wgt = 1.
            else:
                wgt = 1.
        else:
            wgt = 1.

        vel_dat = gal.data.data['velocity'][msk]
        vel_mod = gal.model_data.data['velocity'][msk]
        vel_err = gal.data.error['velocity'][msk]

        if hasattr(gal.data, 'mask_vel_disp'):
            if gal.data.mask_vel_disp is not None:
                msk = gal.data.mask_vel_disp
            else:
                msk = gal.data.mask
        else:
            msk = gal.data.mask
        disp_dat = gal.data.data['dispersion'][msk]
        disp_mod = gal.model_data.data['dispersion'][msk]
        disp_err = gal.data.error['dispersion'][msk]


        if fitflux:
            msk = gal.data.mask
            flux_dat = gal.data.data['flux'][msk]
            flux_mod = gal.model_data.data['flux'][msk]
            try:
                flux_err = gal.data.error['flux'][msk]
            except:
                flux_err = 0.1*gal.data.data['flux'][msk] # PLACEHOLDER

        # Correct model for instrument dispersion if the data is instrument corrected:
        if 'inst_corr' in gal.data.data.keys():
            if gal.data.data['inst_corr']:
                disp_mod = np.sqrt(
                    disp_mod ** 2 - gal.instrument.lsf.dispersion.to(
                        u.km / u.s).value ** 2)

        chisq_arr_raw_vel = ((vel_dat - vel_mod) / vel_err) * np.sqrt(wgt)
        if fitdispersion:
            chisq_arr_raw_disp = ((disp_dat - disp_mod) / disp_err) * np.sqrt(wgt)
            if fitflux:
                chisq_arr_raw_flux = ((flux_dat - flux_mod) / flux_err) * np.sqrt(wgt)
                chisq_arr_raw = np.hstack([chisq_arr_raw_vel.flatten(),
                                           chisq_arr_raw_disp.flatten(),
                                           chisq_arr_raw_flux.flatten()])
            else:
                chisq_arr_raw = np.hstack([chisq_arr_raw_vel.flatten(),
                                           chisq_arr_raw_disp.flatten()])
        else:
            if fitflux:
                chisq_arr_raw_flux = ((flux_dat - flux_mod) / flux_err) * np.sqrt(wgt)
                chisq_arr_raw = np.hstack([chisq_arr_raw_vel.flatten(),
                                           chisq_arr_raw_flux.flatten()])
            else:
                chisq_arr_raw = chisq_arr_raw_vel.flatten()
    else:
        logger.warning("ndim={} not supported!".format(gal.data.ndim))
        raise ValueError

    status = 0

    return [status, chisq_arr_raw]

def setup_oversampled_chisq(gal):
    # Setup for oversampled_chisq:
    if isinstance(gal.instrument.beam, GaussianBeam):
        try:
            PSF_FWHM = gal.instrument.beam.major.value
        except:
            PSF_FWHM = gal.instrument.beam.major
    elif isinstance(gal.instrument.beam, Moffat):
        try:
            PSF_FWHM = gal.instrument.beam.major_fwhm.value
        except:
            PSF_FWHM = gal.instrument.beam.major_fwhm
    elif isinstance(gal.instrument.beam, DoubleBeam):
        try:
            PSF_FWHM = np.max([gal.instrument.beam.beam1.major.value, gal.instrument.beam.beam2.major.value])
        except:
            PSF_FWHM = np.max([gal.instrument.beam.beam1.major, gal.instrument.beam.beam2.major])


    if gal.data.ndim == 1:
        rarrtmp = gal.data.rarr.copy()
        rarrtmp.sort()
        spacing_avg = np.abs(np.average(rarrtmp[1:]-rarrtmp[:-1]))
        gal.data.oversample_factor_chisq = PSF_FWHM /spacing_avg
    elif gal.data.ndim == 2:
        gal.data.oversample_factor_chisq = (PSF_FWHM / gal.instrument.pixscale.value)**2
    elif gal.data.ndim == 3:
        raise ValueError("need to implement!")

    return gal

def initialize_walkers(model, nWalkers=None):
    """
    Initialize a set of MCMC walkers by randomly drawing from the
    model set parameter priors
    """
    # nDim = len(model.get_free_parameters_values())
    stack_rand = []
    pfree_dict = model.get_free_parameter_keys()
    comps_names = pfree_dict.keys()

    for compn in comps_names:
        comp = model.components.__getitem__(compn)
        params_names = pfree_dict[compn].keys()
        for paramn in params_names:
            if (pfree_dict[compn][paramn] >= 0) :
                # Free parameter: randomly sample from prior nWalker times:
                param_rand = comp.__getattribute__(paramn).prior.sample_prior(comp.__getattribute__(paramn), N=nWalkers)
                stack_rand.append(param_rand)
    pos = np.array(list(zip(*stack_rand)))        # should have shape:   (nWalkers, nDim)
    return pos


def create_default_mcmc_options():
    """
    Create a default dictionary of MCMC options.
    These are used when calling fit, eg:
        mcmc_options = fitting.create_default_mcmc_options()
        mcmcResults = fitting.fit(gal, **mcmc_options)

    This dictionary is provides the full set of keywords that fit() can take,
        and some potentially useful values for these parameters.

    Now superceded by the functionality of `config.Config_fit_mcmc().dict`
    """
    config_fit = config.Config_fit_mcmc()
    mcmc_options = config_fit.dict
    # mcmc_options = dict(nWalkers=10,
    #    cpuFrac=None,
    #    nCPUs = 1,
    #    scale_param_a = 3.,
    #    nBurn = 2,
    #    nSteps = 10,
    #    minAF = 0.2,
    #    maxAF = 0.5,
    #    nEff = 10,
    #    do_plotting = True,
    #    outdir = 'mcmc_fit_results/',
    #    f_plot_trace_burnin = None,
    #    f_plot_trace = None,
    #    f_sampler = None,
    #    f_burn_sampler = None,
    #    f_plot_param_corner = None,
    #    f_plot_bestfit = None,
    #    f_mcmc_results = None)


    return mcmc_options


def find_peak_gaussian_KDE(flatchain, initval):
    """
    Return chain parameters that give peak of the posterior PDF, using KDE.
    """
    try:
        nparams = flatchain.shape[1]
        nrows = nparams
    except:
        nparams = 1
        nrows = 0

    if nrows > 0:
        peakvals = np.zeros(nparams)
        for i in six.moves.xrange(nparams):
            kern = gaussian_kde(flatchain[:,i])
            peakvals[i] = fmin(lambda x: -kern(x), initval[i],disp=False)
        return peakvals
    else:

        kern = gaussian_kde(flatchain)
        peakval = fmin(lambda x: -kern(x), initval,disp=False)

        try:
            return peakval[0]
        except:
            return peakval


def find_peak_gaussian_KDE_multiD(flatchain, linked_inds, initval):
    """
    Return chain parameters that give peak of the posterior PDF *FOR LINKED PARAMETERS, using KDE.
    """

    nparams = len(linked_inds)
    kern = gaussian_kde(flatchain[:,linked_inds].T)
    peakvals = fmin(lambda x: -kern(x), initval,disp=False)

    return peakvals


def find_multiD_pk_hist(flatchain, linked_inds, nPostBins=25):
    H2, edges = np.histogramdd(flatchain[:,linked_inds], bins=nPostBins)

    wh_pk = np.where(H2 == H2.max())[0][0]

    pk_vals = np.zeros(len(linked_inds))

    for k in six.moves.xrange(len(linked_inds)):
        pk_vals[k] = np.average([edges[k][wh_pk], edges[k][wh_pk+1]])

    return pk_vals



def get_linked_posterior_peak_values(flatchain,
                guess = None,
                linked_posterior_ind_arr=None,
                nPostBins=50):
    """
    Get linked posterior best-fit values using a multi-D histogram for the
    given linked parameter indices.

    Input:
        flatchain:                  sampler flatchain, shape (Nwalkers, Nparams)
        linked_posterior_inds_arr:  array of arrays of parameters to be analyzed together

                                    eg: analyze ind1+ind2 together, and then ind3+ind4 together
                                    linked_posterior_inds_arr = [ [ind1, ind2], [ind3, ind4] ]

        nPostBins:                  number of bins on each parameter "edge" of the multi-D histogram

    Output:
        bestfit_theta_linked:       array of the linked bestfit paramter values from multiD param space
                                    eg:
                                    bestfit_theta_linked = [ [best1, best2], [best3, best4] ]
    """

    # Use gaussian KDE to get bestfit linked:
    bestfit_theta_linked = np.array([])

    for k in six.moves.xrange(len(linked_posterior_ind_arr)):
        bestfit_thetas = find_peak_gaussian_KDE_multiD(flatchain, linked_posterior_ind_arr[k],
                guess[linked_posterior_ind_arr[k]])
        if len(bestfit_theta_linked) >= 1:
            bestfit_theta_linked = np.stack(bestfit_theta_linked, np.array([bestfit_thetas]) )
        else:
            bestfit_theta_linked = np.array([bestfit_thetas])


    return bestfit_theta_linked



def get_linked_posterior_indices(mcmcResults, linked_posterior_names=None):
    """
    Convert the input set of linked posterior names to set of indices:

    Input:
        (example structure)

        To analyze all parameters together:
        linked_posterior_names = 'all'


        Alternative: only link some parameters:

        linked_posterior_names = [ joint_param_bundle1, joint_param_bundle2 ]
        with
        join_param_bundle1 = [ [cmp1, par1], [cmp2, par2] ]
        jont_param_bundle2 = [ [cmp3, par3], [cmp4, par4] ]
        for a full array of:
        linked_posterior_names =
            [ [ [cmp1, par1], [cmp2, par2] ], [ [cmp3, par3], [cmp4, par4] ] ]


        Also if doing single bundle must have:
        linked_posterior_names = [ [ [cmp1, par1], [cmp2, par2] ] ]

    Output:
        linked_posterior_inds = [ joint_bundle1_inds, joint_bundle2_inds ]
        with joint_bundle1_inds = [ ind1, ind2 ], etc

        ex:
            output = [ [ind1, ind2], [ind3, ind4] ]

    """
    linked_posterior_ind_arr = None
    try:
        if linked_posterior_names.strip().lower() == 'all':
            linked_posterior_ind_arr = [range(len(mcmcResults.free_param_names))]
    except:
        pass
    if linked_posterior_ind_arr is None:
        free_cmp_param_arr = make_arr_cmp_params(mcmcResults)

        linked_posterior_ind_arr = []
        for k in six.moves.xrange(len(linked_posterior_names)):
            # Loop over *sets* of linked posteriors:
            # This is an array of len-2 arrays/tuples with cmp, param names
            linked_post_inds = []
            for j in six.moves.xrange(len(linked_posterior_names[k])):

                indp = get_param_index(mcmcResults, linked_posterior_names[k][j],
                            free_cmp_param_arr=free_cmp_param_arr)
                linked_post_inds.append(indp)


            linked_posterior_ind_arr.append(linked_post_inds)


    return linked_posterior_ind_arr

def get_param_index(mcmcResults, param_name, free_cmp_param_arr=None):
    if free_cmp_param_arr is None:
        free_cmp_param_arr = make_arr_cmp_params(mcmcResults)

    cmp_param = param_name[0].strip().lower()+':'+param_name[1].strip().lower()

    try:
        whmatch = np.where(free_cmp_param_arr == cmp_param)[0][0]
    except:
        raise ValueError(cmp_param+' component+parameter not found in free parameters of mcmcResults')
    return whmatch


def make_arr_cmp_params(mcmcResults):
    arr = np.array([])
    for cmp in mcmcResults.free_param_names.keys():
        for i in six.moves.xrange(len(mcmcResults.free_param_names[cmp])):
            param = mcmcResults.free_param_names[cmp][i]
            arr = np.append( arr, cmp.strip().lower()+':'+param.strip().lower() )

    return arr


def make_emcee_sampler_dict(sampler, nBurn=0, emcee_vers=3):
    """
    Save chain + key results from emcee sampler instance to a dict,
    as the emcee samplers aren't pickleable.
    """

    if emcee_vers == 3:
        return _make_emcee_sampler_dict_v3(sampler, nBurn=nBurn)
    elif emcee_vers == 2:
        return _make_emcee_sampler_dict_v2(sampler, nBurn=nBurn)
    else:
        raise ValueError("Emcee version {} not supported!".format(emcee_vers))


def _make_emcee_sampler_dict_v2(sampler, nBurn=0):
    """ Syntax for emcee v2.2.1 """
    # Cut first nBurn steps, to avoid the edge cases that are rarely explored.
    chain = sampler.chain[:, nBurn:, :]
    flatchain = chain.reshape((-1, sampler.dim))
    # Walkers, iterations
    probs =     sampler.lnprobability[:, nBurn:]
    flatprobs = probs.reshape((-1))

    try:
        acor_time = sampler.get_autocorr_time(low=5, c=10)
    except:
        acor_time = None


    # Make a dictionary:
    sampler_dict = { 'chain':             chain,
                     'flatchain':         flatchain,
                     'lnprobability':     probs,
                     'flatlnprobability': flatprobs,
                     'nIter':             sampler.iterations,
                     'nParam':            sampler.dim,
                     'nCPU':              sampler.threads,
                     'nWalkers':          len(sampler.chain),
                     'acceptance_fraction': sampler.acceptance_fraction,
                     'acor_time': acor_time }

    if len(sampler.blobs) > 0:
        sampler_dict['blobs'] = np.array(sampler.blobs[nBurn:])

        if len(np.shape(sampler.blobs)) == 2:
            # Only 1 blob: nSteps, nWalkers:
            sampler_dict['flatblobs'] = np.array(sampler_dict['blobs']).reshape(-1)
        elif len(np.shape(sampler.blobs)) == 3:
            # Multiblobs; nSteps, nWalkers, nBlobs
            sampler_dict['flatblobs'] = np.array(sampler_dict['blobs']).reshape(-1,np.shape(sampler.blobs)[2])
        else:
            raise ValueError("Sampler blob length not recognized")


    return sampler_dict


def _make_emcee_sampler_dict_v3(sampler, nBurn=0):
    """ Syntax for emcee v3 """

    # Cut first nBurn steps, to avoid the edge cases that are rarely explored.
    samples = sampler.chain[:, nBurn:, :].reshape((-1, sampler.ndim))
    # Walkers, iterations
    probs = sampler.lnprobability[:, nBurn:].reshape((-1))

    acor_time = sampler.get_autocorr_time(tol=10, quiet=True)

    try:
        nCPUs = sampler.pool._processes   # sampler.threads
    except:
        nCPUs = 1

    # Make a dictionary:
    sampler_dict = { 'chain':               sampler.chain[:, nBurn:, :],
                     'lnprobability':       sampler.lnprobability[:, nBurn:],
                    'flatchain':            samples,
                    'flatlnprobability':    probs,
                    'nIter':                sampler.iteration,
                    'nParam':               sampler.ndim,
                    'nCPU':                 nCPUs,
                    'nWalkers':             sampler.nwalkers,
                    'acceptance_fraction':  sampler.acceptance_fraction,
                    'acor_time':            acor_time }

    if len(sampler.blobs) > 0:
        if len(np.shape(sampler.blobs)) == 2:
            # Only 1 blob: nSteps, nWalkers:
            sampler_dict['blobs'] = sampler.blobs[nBurn:, :]
            flatblobs = np.array(sampler_dict['blobs']).reshape(-1)
        elif len(np.shape(sampler.blobs)) == 3:
            # Multiblobs; nSteps, nWalkers, nBlobs
            sampler_dict['blobs'] = sampler.blobs[nBurn:, :, :]
            flatblobs = np.array(sampler_dict['blobs']).reshape(-1,np.shape(sampler.blobs)[2])
        else:
            raise ValueError("Sampler blob shape not recognized")

        sampler_dict['flatblobs'] = flatblobs

    return sampler_dict


def _reload_sampler_hdf5(filename=None, backend_name='mcmc'):
    # Load backend from file
    backend = emcee.backends.HDFBackend(filename, name=backend_name)
    return _make_sampler_dict_from_hdf5(backend)

def _make_sampler_dict_from_hdf5(b):
    """  Construct a dysmalpy 'sampler_dict' out of the chain info stored in the emcee v3 HDF5 file """
    nwalkers =  b.shape[0]
    ndim =      b.shape[1]

    chain =     np.swapaxes(b.get_chain(), 0, 1)
    flatchain = chain.reshape((-1, ndim))

    # Walkers, iterations
    probs =     np.swapaxes(b.get_log_prob(), 0, 1)
    flatprobs = probs.reshape(-1)

    acor_time = b.get_autocorr_time(tol=10, quiet=True)

    # Make a dictionary:
    sampler_dict = { 'chain':                chain,
                     'flatchain':            flatchain,
                     'lnprobability':        probs,
                     'flatlnprobability':    flatprobs,
                     'nIter':                b.iteration,
                     'nParam':               ndim,
                     'nCPU':                 None,
                     'nWalkers':             nwalkers,
                     'acceptance_fraction':  b.accepted / float(b.iteration),
                     'acor_time':            acor_time }

    if b.has_blobs() :
        sampler_dict['blobs'] = b.get_blobs()
        if len(b.get_blobs().shape) == 2:
            # Only 1 blob: nSteps, nWalkers:
            flatblobs = np.array(sampler_dict['blobs']).reshape(-1)
        elif len(b.get_blobs().shape) == 3:
            # Multiblobs; nSteps, nWalkers, nBlobs
            flatblobs = np.array(sampler_dict['blobs']).reshape(-1,np.shape(sampler_dict['blobs'])[2])
        else:
            raise ValueError("Sampler blob shape not recognized")

        sampler_dict['flatblobs'] = flatblobs


    return sampler_dict

def _reload_sampler_pickle(filename=None):
    return load_pickle(filename)


def reinitialize_emcee_sampler(sampler_dict, gal=None, kwargs_dict=None,
                    scale_param_a=None):
    """
    Re-setup emcee sampler, using existing chain / etc, so more steps can be run.
    """

    # This will break for updated version of emcee
    # works for emcee v2.2.1
    if emcee.__version__ == '2.2.1':

        sampler = emcee.EnsembleSampler(sampler_dict['nWalkers'], sampler_dict['nParam'],
                    log_prob, args=[gal], kwargs=kwargs_dict, a=kwargs_fit['scale_param_a'],
                    threads=sampler_dict['nCPU'])

        sampler._chain = copy.deepcopy(sampler_dict['chain'])
        sampler._blobs = list(copy.deepcopy(sampler_dict['blobs']))
        sampler._lnprob = copy.deepcopy(sampler_dict['lnprobability'])
        sampler.iterations = sampler_dict['nIter']
        sampler.naccepted = np.array(sampler_dict['nIter']*copy.deepcopy(sampler_dict['acceptance_fraction']),
                            dtype=np.int64)
    ###
    elif np.int(emcee.__version__[0]) >= 3:
        # This is based off of HDF5 files, which automatically makes it easy to reload + resetup the sampler
        raise ValueError

    ###
    else:
        try:
            backend = emcee.Backend()
            backend.nwalkers = sampler_dict['nWalkers']
            backend.ndim = sampler_dict['nParam']
            backend.iteration = sampler_dict['nIter']
            backend.accepted = np.array(sampler_dict['nIter']*sampler_dict['acceptance_fraction'],
                                dtype=np.int64)
            backend.chain = sampler_dict['chain']
            backend.log_prob = sampler_dict['lnprobability']
            backend.blobs = sampler_dict['blobs']
            backend.initialized = True


            sampler = emcee.EnsembleSampler(sampler_dict['nWalkers'],
                        sampler_dict['nParam'],
                        log_prob,
                        args=[gal], kwargs=kwargs_dict,
                        backend=backend,
                        a=kwargs_fit['scale_param_a'],
                        threads=sampler_dict['nCPU'])

        except:
            raise ValueError



    return sampler



def mpfit_printer(fcn, x, iter, fnorm, functkw=None,
                  quiet=0, parinfo=None,
                  pformat='%.10g', dof=None,
                  logger=None):

        if quiet:
            return

        # Determine which parameters to print
        nprint = len(x)
        iter_line = "Iter {}  CHI-SQUARE = {:.10g}  DOF = {:}".format(iter, fnorm, dof)
        param_lines = '\n'
        for i in range(nprint):
            if (parinfo is not None) and ('parname' in parinfo[i]):
                p = '   ' + parinfo[i]['parname'] + ' = '
            else:
                p = '   P' + str(i) + ' = '
            if (parinfo is not None) and ('mpprint' in parinfo[i]):
                iprint = parinfo[i]['mpprint']
            else:
                iprint = 1
            if iprint:

                param_lines += p + (pformat % x[i]) + '  ' + '\n'

        if logger is None:
            print(iter_line+param_lines)
        else:
            logger.info(iter_line+param_lines)

        return 0


def ensure_dir(dir):
    """ Short function to ensure dir is a directory; if not, make the directory."""
    if not os.path.exists(dir):
        logger.info( "Making path="+dir)
        os.makedirs(dir)
    return None


def load_pickle(filename):
    """ Small wrapper function to load a pickled structure """
    with open(filename, 'rb') as f:
        data = copy.deepcopy(_pickle.load(f))
    return data


def dump_pickle(data, filename=None, overwrite=False):
    """ Small wrapper function to pickle a structure """
    # Check for existing file:
    if (not overwrite) and (filename is not None):
        if os.path.isfile(filename):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
            return None

    with open(filename, 'wb') as f:
        _pickle.dump(data, f )
    return None



def reload_all_fitting(filename_galmodel=None, filename_results=None, fit_method=None):
    """
    Utility to reload the Galaxy and Results object from a previous fit.

    Parameters
    ----------
    filename_galmodel : str
            Full path to the file storing the Galaxy object
    filename_results :  str
            Full path to the file storing the FitResults object
    fit_method : str
            Fitting method that was run. Used to determine the subclass of FitResults for reloading.
            Can be set to `mpfit` or `mcmc`.

    Output
    ------
    gal : obj
            Galaxy instance, including model with the current best-fit parameters
    retults : obj
            MCMCResults or MPFITResults instance, containing all fit results and analysis
    """

    if fit_method is None:
        raise ValueError("Must set 'fit_method'! Options are 'mpfit' or 'mcmc'.")

    if fit_method.lower().strip() == 'mcmc':
        return _reload_all_fitting_mcmc(filename_galmodel=filename_galmodel, filename_results=filename_results)
    elif fit_method.lower().strip() == 'mpfit':
        return _reload_all_fitting_mpfit(filename_galmodel=filename_galmodel, filename_results=filename_results)
    else:
        raise ValueError("Fit type {} not recognized!".format(fit_method))


def _reload_all_fitting_mcmc(filename_galmodel=None, filename_results=None):
    gal = galaxy.load_galaxy_object(filename=filename_galmodel)
    results = MCMCResults()
    results.reload_results(filename=filename_results)
    return gal, results

def _reload_all_fitting_mpfit(filename_galmodel=None, filename_results=None):
    gal = galaxy.load_galaxy_object(filename=filename_galmodel)
    results = MPFITResults()
    results.reload_results(filename=filename_results)
    return gal, results



def norm(x): # Euclidean norm
    return np.sqrt(np.sum(x**2))

######

def find_shortest_conf_interval(xarr, percentile_frac):
    # Canonical 1sigma: 0.6827
    xsort = np.sort(xarr)

    N = len(xarr)
    i_max = np.int64(np.round(percentile_frac*N))
    len_arr = xsort[i_max:] - xsort[0:N-i_max]

    argmin = np.argmin(len_arr)
    l_val, u_val = xsort[argmin], xsort[argmin+i_max-1]

    return l_val, u_val

def shortest_span_bounds(arr, percentile=0.6827):
    if len(arr.shape) == 1:
        limits = find_shortest_conf_interval(arr, percentile)
    else:
        limits = np.ones((2, arr.shape[1]))
        for j in six.moves.xrange(arr.shape[1]):
            limits[:, j] = find_shortest_conf_interval(arr[:,j], percentile)

    return limits
