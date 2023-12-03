# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Classes and functions for fitting DYSMALPY kinematic models
#   to the observed data using MCMC
#
# Some handling of MCMC / posterior distribution analysis inspired by speclens,
#    with thanks to Matt George:
#    https://github.com/mrgeorge/speclens/blob/master/speclens/fit.py

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

## Standard library
import logging
try:
    from multiprocess import cpu_count, Pool
except:
    # Old python versions:
    from multiprocessing import cpu_count, Pool

# DYSMALPY code
from dysmalpy.data_io import load_pickle, dump_pickle
from dysmalpy import plotting
from dysmalpy import galaxy
from dysmalpy import utils as dpy_utils
from dysmalpy.fitting import base
from dysmalpy.fitting import utils as fit_utils


# Third party imports
import os
import numpy as np
from collections import OrderedDict
import astropy.units as u
import copy

import time, datetime


__all__ = ['MCMCFitter', 'MCMCResults']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

try:
    import emcee
    _emcee_loaded = True
    _emcee_version = int(emcee.__version__[0])
except:
    _emcee_loaded = False
    logger.warn("emcee installation not found!")



class MCMCFitter(base.Fitter):
    """
    Class to hold the MCMC fitter attributes + methods
    """
    def __init__(self, **kwargs):
        if not _emcee_loaded:
            raise ValueError("emcee was not loaded!")

        self._emcee_version = _emcee_version

        self._set_defaults()
        super(MCMCFitter, self).__init__(fit_method='MCMC', **kwargs)

    def _set_defaults(self):
        # MCMC specific defaults
        self.nWalkers = 10
        self.nCPUs = 1
        self.cpuFrac = None
        self.scale_param_a = 3.
        self.nBurn = 2.
        self.nSteps = 10.
        self.minAF = 0.2
        self.maxAF = 0.5
        self.nEff = 10

        self.oversampled_chisq = True

        # self.red_chisq = False   # Option not used

        self.save_burn = False
        self.save_intermediate_sampler_results_chain = True
        self.nStep_intermediate_save = 5
        self.continue_steps = False

        self.nPostBins = 50
        self.linked_posterior_names = None

        self.input_sampler_results = None

        # ACOR SETTINGS
        # Force it to run for at least N steps, otherwise acor times might be completely wrong.
        self.acor_force_min = 49


    def fit(self, gal, output_options):
        """
        Fit observed kinematics using MCMC and a DYSMALPY model set.

        Parameters
        ----------
            gal : `Galaxy` instance
                observed galaxy, including kinematics.
                also contains instrument the galaxy was observed with (gal.instrument)
                and the DYSMALPY model set, with the parameters to be fit (gal.model)

            output_options : `config.OutputOptions` instance
                instance holding ouptut options for MCMC fitting.

        Returns
        -------
            mcmcResults : `MCMCResults` instance
                MCMCResults class instance containing the bestfit parameters, sampler_results information, etc.
        """

        # --------------------------------
        # Check option validity:
        if self.blob_name is not None:
            valid_blobnames = ['fdm', 'mvirial', 'alpha', 'rb']
            if isinstance(self.blob_name, str):
                # Single blob
                blob_arr = [self.blob_name]
            else:
                # Array of blobs
                blob_arr = self.blob_name[:]

            for blobn in blob_arr:
                if blobn.lower().strip() not in valid_blobnames:
                    raise ValueError("blob_name={} not recognized as option!".format(blobn))


        # # Temporary: testing:
        # if self.red_chisq:
        #     raise ValueError("red_chisq=True is currently *DISABLED* to test lnlike impact vs lnprior")

        # Check the FOV is large enough to cover the data output:
        dpy_utils._check_data_inst_FOV_compatibility(gal)

        # Pre-calculate instrument kernels:
        gal = dpy_utils._set_instrument_kernels(gal)

        # --------------------------------
        # Basic setup:

        # For compatibility with Python 2.7:
        mod_in = copy.deepcopy(gal.model)
        gal.model = mod_in

        #if nCPUs is None:
        if self.cpuFrac is not None:
            self.nCPUs = int(np.floor(cpu_count()*self.cpuFrac))

        # +++++++++++++++++++++++
        # Setup for oversampled_chisq:
        if self.oversampled_chisq:
            gal = fit_utils.setup_oversampled_chisq(gal)
        # +++++++++++++++++++++++

        # Set output options: filenames / which to save, etc
        output_options.set_output_options(gal, self)

        # MUST INCLUDE MCMC-SPECIFICS NOW!
        fit_utils._check_existing_files_overwrite(output_options, 
                                                  fit_type='mcmc', 
                                                  fitter=self)

        # --------------------------------
        # Setup file redirect logging:
        if output_options.f_log is not None:
            loggerfile = logging.FileHandler(output_options.f_log)
            loggerfile.setLevel(logging.INFO)
            logger.addHandler(loggerfile)

        # --------------------------------
        # Split by emcee version:
        if self._emcee_version >= 3:
            mcmcResults = self._fit_emcee_3(gal, output_options)
        else:
            mcmcResults = self._fit_emcee_221(gal, output_options)

        # Clean up logger:
        if output_options.f_log is not None:
            logger.removeHandler(loggerfile)
            loggerfile.close()

        return mcmcResults






    def _fit_emcee_221(self, gal, output_options):
        # --------------------------------
        # Initialize emcee sampler_results
        kwargs_dict = {'fitter': self}

        nBurn_orig = output_options['nBurn']

        nDim = gal.model.nparams_free


        if (not self.continue_steps) & ((not self.save_intermediate_sampler_results_chain) \
            | (not os.path.isfile(output_options.f_sampler_results_tmp))):
            sampler_results = emcee.EnsembleSampler(self.nWalkers, nDim, base.log_prob,
                        args=[gal], kwargs=kwargs_dict,
                        a = self.scale_param_a, threads = self.nCPUs)

            # --------------------------------
            # Initialize walker starting positions
            initial_pos = initialize_walkers(gal.model, nWalkers=self.nWalkers)

        elif self.continue_steps:
            self.nBurn = 0
            if self.input_sampler_results is None:
                try:
                    self.input_sampler_results = load_pickle(output_options.f_sampler_results)
                except:
                    message = "Couldn't find existing sampler_results in {}.".format(output_options.f_sampler_results)
                    message += '\n'
                    message += "Must set input_sampler_results if you will restart the sampler_results."
                    raise ValueError(message)

            sampler_results = reinitialize_emcee_sampler_results(self.input_sampler_results, gal=gal,
                                kwargs_dict=kwargs_dict,
                                scale_param_a=self.scale_param_a)

            initial_pos = self.input_sampler_results['chain'][:,-1,:]
            if self.blob_name is not None:
                blob = self.input_sampler_results['blobs']

            # Close things
            self.input_sampler_results = None

        elif self.save_intermediate_sampler_results_chain & (os.path.isfile(output_options.f_sampler_results_tmp)):
            self.input_sampler_results = load_pickle(output_options.f_sampler_results_tmp)

            sampler_results = reinitialize_emcee_sampler_results(self.input_sampler_results, gal=gal,
                                                 fitter=self)
            self.nBurn = nBurn_orig - (self.input_sampler_results['burn_step_cur'] + 1)

            initial_pos = self.input_sampler_results['chain'][:,-1,:]
            if self.blob_name is not None:
                blob = self.input_sampler_results['blobs']

            # If it saved after burn finished, but hasn't saved any of the normal steps: reset sampler_results
            if ((self.nBurn == 0) & (self.input_sampler_results['step_cur'] < 0)):
                blob = None
                sampler_results.reset()
                if self.blob_name is not None:
                     sampler_results.clear_blobs()

            # Close things
            input_sampler_results = None


        # --------------------------------
        # Output some fitting info to logger:
        logger.info("*************************************")
        logger.info(" Fitting: {} with MCMC".format(gal.name))
        for obs_name in gal.observations:
            obs = gal.observations[obs_name]
            logger.info("    obs: {}".format(obs.name))
            if obs.data.filename_velocity is not None:
                logger.info("        velocity file: {}".format(obs.data.filename_velocity))
            if obs.data.filename_dispersion is not None:
                logger.info("        dispers. file: {}".format(obs.data.filename_dispersion))

            logger.info('        nSubpixels: {}'.format(obs.mod_options.oversample))

        logger.info('\n'+'nCPUs: {}'.format(self.nCPUs))
        logger.info('nWalkers: {}'.format(self.nWalkers))
        # logger.info('lnlike: red_chisq={}'.format(self.red_chisq))
        logger.info('lnlike: oversampled_chisq={}'.format(self.oversampled_chisq))

        logger.info('\n'+'blobs: {}'.format(self.blob_name))


        if ('halo' in gal.model.components.keys()):
            logger.info('\n'+'mvirial_tied: {}'.format(gal.model.components['halo'].mvirial.tied))

        if ('disk+bulge' in gal.model.components.keys()):
            if 'mhalo_relation' in gal.model.components['disk+bulge'].__dict__.keys():
                logger.info('mhalo_relation: {}'.format(gal.model.components['disk+bulge'].mhalo_relation))
            if 'truncate_lmstar_halo' in gal.model.components['disk+bulge'].__dict__.keys():
                logger.info('truncate_lmstar_halo: {}'.format(gal.model.components['disk+bulge'].truncate_lmstar_halo))

        ################################################################
        # --------------------------------
        # Run burn-in
        if self.nBurn > 0:
            logger.info('\nBurn-in:'+'\n'
                        'Start: {}\n'.format(datetime.datetime.now()))
            start = time.time()

            ####
            pos = initial_pos
            prob = None
            state = None
            blob = None
            for k in range(nBurn_orig):
                # --------------------------------
                # If recovering intermediate save, only start past existing chain length:
                if self.save_intermediate_sampler_results_chain:
                    if k < sampler_results.chain.shape[1]:
                        continue

                logger.info(" k={}, time.time={}, a_frac={}".format( k, datetime.datetime.now(),
                            np.mean(sampler_results.acceptance_fraction)  ) )
                ###
                pos_cur = pos.copy()    # copy just in case things are set strangely

                # Run one sample step:
                if self.blob_name is not None:
                    pos, prob, state, blob = sampler_results.run_mcmc(pos_cur, 1, lnprob0=prob,
                            rstate0=state, blobs0 = blob)
                else:
                    pos, prob, state = sampler_results.run_mcmc(pos_cur, 1, lnprob0=prob, rstate0=state)


                # --------------------------------
                # Save intermediate steps if set:
                if self.save_intermediate_sampler_results_chain:
                    if ((k+1) % self.nStep_intermediate_save == 0):
                        sampler_results_dict_tmp = make_emcee_sampler_results_dict(sampler_results, nBurn=0, emcee_vers=2)
                        sampler_results_dict_tmp['burn_step_cur'] = k
                        sampler_results_dict_tmp['step_cur'] = -99
                        if output_options.f_sampler_results_tmp is not None:
                            # Save stuff to file, for future use:
                            dump_pickle(sampler_results_dict_tmp, filename=output_options.f_sampler_results_tmp, overwrite=True)
                # --------------------------------

            #####
            end = time.time()
            elapsed = end-start


            try:
                acor_time = sampler_results.get_autocorr_time(low=5, c=10)
            except:
                acor_time = "Undefined, chain did not converge"

            #######################################################################################
            # Return Burn-in info
            # ****
            endtime = str(datetime.datetime.now())
            nthingsmsg = 'nCPU, nParam, nWalker, nBurn = {}, {}, {}, {}'.format(self.nCPUs,
                nDim, self.nWalkers, self.nBurn)
            scaleparammsg = 'Scale param a= {}'.format(self.scale_param_a)
            timemsg = 'Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s)'.format( elapsed, np.floor(elapsed/60.),
                    (elapsed/60.-np.floor(elapsed/60.))*60. )
            macfracmsg = "Mean acceptance fraction: {:0.3f}".format(np.mean(sampler_results.acceptance_fraction))
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
                if self.nBurn < np.max(acor_time) * nBurn_nEff:
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
            # Save burn-in sampler_results, if desired
            if (self.save_burn) & (output_options.f_burn_sampler_results is not None):
                sampler_results_burn = make_emcee_sampler_results_dict(sampler_results, nBurn=0, emcee_vers=2)
                # Save stuff to file, for future use:
                dump_pickle(sampler_results_burn, filename=output_options.f_burn_sampler_results, overwrite=output_options.overwrite)


            # --------------------------------
            # Plot burn-in trace, if output file set
            if (output_options.do_plotting) & (output_options.f_plot_trace_burnin is not None):
                sampler_results_burn = make_emcee_sampler_results_dict(sampler_results, nBurn=0, emcee_vers=2)
                mcmcResultsburn = MCMCResults(model=gal.model, sampler_results=sampler_results_burn)
                plotting.plot_trace(mcmcResultsburn, fileout=output_options.f_plot_trace_burnin,
                            overwrite=output_options.overwrite)

            # Reset sampler_results after burn-in:
            sampler_results.reset()
            if self.blob_name is not None:
                 sampler_results.clear_blobs()

        else:
            # --------------------------------
            # No burn-in: set initial position:
            if nBurn_orig > 0:
                logger.info('\nUsing previously completed burn-in'+'\n')

            pos = np.array(initial_pos)
            prob = None
            state = None

            if (not self.continue_steps) | (not self.save_intermediate_sampler_results_chain):
                blob = None

        #######################################################################################
        # ****
        # --------------------------------
        # Run sampler_results: Get start time
        logger.info('\nEnsemble sampling:\n'
                    'Start: {}\n'.format(datetime.datetime.now()))
        start = time.time()

        if sampler_results.chain.shape[1] > 0:
            logger.info('\n   Resuming with existing sampler_results chain at iteration ' +
                        str(sampler_results.iteration) + '\n')
            pos = sampler_results['chain'][:,-1,:]

        # --------------------------------
        # Run sampler_results: output info at each step
        for ii in range(self.nSteps):

            # --------------------------------
            # If continuing chain, only start past existing chain length:
            if self.continue_steps | self.save_intermediate_sampler_results_chain:
                if ii < sampler_results.chain.shape[1]:
                    continue

            pos_cur = pos.copy()    # copy just in case things are set strangely

            # --------------------------------
            # Only do one step at a time:
            if self.blob_name is not None:
                pos, prob, state, blob = sampler_results.run_mcmc(pos_cur, 1, lnprob0=prob,
                        rstate0=state, blobs0 = blob)
            else:
                pos, prob, state = sampler_results.run_mcmc(pos_cur, 1, lnprob0=prob, rstate0=state)
            # --------------------------------

            # --------------------------------
            # Give output info about this step:
            nowtime = str(datetime.datetime.now())
            stepinfomsg = "ii={}, a_frac={}".format( ii, np.mean(sampler_results.acceptance_fraction) )
            timemsg = " time.time()={}".format(nowtime)
            logger.info( stepinfomsg+timemsg )

            try:
                acor_time = sampler_results.get_autocorr_time(low=5, c=10)
                logger.info( "{}: acor_time ={}".format(ii, np.array(acor_time) ) )
            except:
                acor_time = "Undefined, chain did not converge"
                logger.info(" {}: Chain too short for acor to run".format(ii) )

            # --------------------------------
            # Case: test for convergence and truncate early:
            # Criteria checked: whether acceptance fraction within (minAF, maxAF),
            #                   and whether total number of steps > nEff * average autocorrelation time:
            #                   to make sure the paramter space is well explored.
            if ((self.minAF is not None) & (self.maxAF is not None) & \
                    (self.nEff is not None) & (acor_time is not None)):
                if ((self.minAF < np.mean(sampler_results.acceptance_fraction) < self.maxAF) & \
                    ( ii > np.max(acor_time) * self.nEff )):
                        if ii == self.acor_force_min:
                            logger.info(" Enforced min step limit: {}.".format(ii+1))
                        if ii >= self.acor_force_min:
                            logger.info(" Finishing calculations early at step {}.".format(ii+1))
                            break

            # --------------------------------
            # Save intermediate steps if set:
            if self.save_intermediate_sampler_results_chain:
                if ((ii+1) % self.nStep_intermediate_save == 0):
                    sampler_results_dict_tmp = make_emcee_sampler_results_dict(sampler_results, nBurn=0, emcee_vers=2)
                    sampler_results_dict_tmp['burn_step_cur'] = nBurn_orig - 1
                    sampler_results_dict_tmp['step_cur'] = ii
                    if output_options.f_sampler_results_tmp is not None:
                        # Save stuff to file, for future use:
                        dump_pickle(sampler_results_dict_tmp, filename=output_options.f_sampler_results_tmp, overwrite=True)
            # --------------------------------

        # --------------------------------
        # Check if it failed to converge before the max number of steps, if doing convergence testing
        finishedSteps= ii+1
        if (finishedSteps  == self.nSteps) & ((self.minAF is not None) & \
                (self.maxAF is not None) & (self.nEff is not None)):
            logger.info(" Caution: no convergence within nSteps={}.".format(self.nSteps))

        # --------------------------------
        # Finishing info for fitting:
        end = time.time()
        elapsed = end-start
        logger.info("Finished {} steps".format(finishedSteps)+"\n")

        try:
            acor_time = sampler_results.get_autocorr_time(low=5, c=10)
        except:
            acor_time = "Undefined, chain did not converge"

        #######################################################################################
        # ***********
        # Consider overall acceptance fraction
        endtime = str(datetime.datetime.now())
        nthingsmsg = 'nCPU, nParam, nWalker, nSteps = {}, {}, {}, {}'.format(self.nCPUs,
            nDim, self.nWalkers, self.nSteps)
        scaleparammsg = 'Scale param a= {}'.format(self.scale_param_a)
        timemsg = 'Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s)'.format(elapsed, np.floor(elapsed/60.),
                (elapsed/60.-np.floor(elapsed/60.))*60. )
        macfracmsg = "Mean acceptance fraction: {:0.3f}".format(np.mean(sampler_results.acceptance_fraction))
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
        # Save sampler_results, if output file set:
        #   Burn-in is already cut by resetting the sampler_results at the beginning.
        # Get pickleable format:  # _fit_io.make_emcee_sampler_results_dict
        sampler_results_dict = make_emcee_sampler_results_dict(sampler_results, nBurn=0, emcee_vers=2)


        if output_options.f_sampler_results is not None:
            # Save stuff to file, for future use:
            dump_pickle(sampler_results_dict, filename=output_options.f_sampler_results, overwrite=output_options.overwrite)


        # --------------------------------
        # Cleanup intermediate saves:
        if self.save_intermediate_sampler_results_chain & (output_options.f_sampler_results_tmp is not None):
            if os.path.isfile(output_options.f_sampler_results_tmp):
                os.remove(output_options.f_sampler_results_tmp)
        # --------------------------------

        if self.nCPUs > 1:
            sampler_results.pool.close()

        ##########################################
        ##########################################
        ##########################################

        # --------------------------------
        # Bundle the results up into a results class:
        mcmcResults = MCMCResults(model=gal.model, sampler_results=sampler_results_dict,
                                  linked_posterior_names=self.linked_posterior_names,
                                  blob_name=self.blob_name,
                                  nPostBins=self.nPostBins)
        if self.oversampled_chisq:
            mcmcResults.oversample_factor_chisq = OrderedDict()
            for obs_name in gal.observations:
                obs = gal.observations[obs_name]
                mcmcResults.oversample_factor_chisq[obs_name] = obs.data.oversample_factor_chisq

        # Do all analysis, plotting, saving:
        mcmcResults.analyze_plot_save_results(gal, output_options=output_options)

        return mcmcResults

    def _fit_emcee_3(self, gal, output_options):

        # Check length of sampler_results if not overwriting:
        if (not output_options.overwrite):
            if os.path.isfile(output_options.f_sampler_results):
                backend = emcee.backends.HDFBackend(output_options.f_sampler_results, name='mcmc')
                try:
                    if backend.get_chain().shape[0] >= self.nSteps:
                        if output_options.f_results is not None:
                            if os.path.isfile(output_options.f_results):
                                msg = "overwrite={}, and 'f_sampler_results' already contains {} steps,".format(output_options.overwrite,
                                                        backend.get_chain().shape[0])
                                msg += " so the fit will not be saved.\n Specify new outfile or delete old files."
                                logger.warning(msg)
                                return None
                    else:
                        pass
                except:
                    pass


        # --------------------------------
        # Initialize emcee sampler_results

        nBurn_orig = self.nBurn

        nDim = gal.model.nparams_free
        kwargs_dict = {'fitter': self}

        # --------------------------------
        # Start pool, moves, backend:
        if (self.nCPUs > 1):
            pool = Pool(self.nCPUs)
        else:
            pool = None

        moves = emcee.moves.StretchMove(a=self.scale_param_a)

        backend_burn = emcee.backends.HDFBackend(output_options.f_sampler_results, name="burnin_mcmc")

        if output_options.overwrite:
            backend_burn.reset(self.nWalkers, nDim)

        sampler_results_burn = emcee.EnsembleSampler(self.nWalkers, nDim, base.log_prob,
                    backend=backend_burn, pool=pool, moves=moves,
                    args=[gal], kwargs=kwargs_dict)

        nBurnCur = sampler_results_burn.iteration

        self.nBurn = nBurn_orig - nBurnCur


        # --------------------------------
        # Initialize walker starting positions
        if sampler_results_burn.iteration == 0:
            initial_pos = initialize_walkers(gal.model, nWalkers=self.nWalkers)
        else:
            initial_pos = sampler_results_burn.get_last_sample()


        # --------------------------------
        # Output some fitting info to logger:
        logger.info("*************************************")
        logger.info(" Fitting: {} with MCMC".format(gal.name))
        for obs_name in gal.observations:
            obs = gal.observations[obs_name]
            logger.info("    obs: {}".format(obs.name))
            if obs.data.filename_velocity is not None:
                logger.info("        velocity file: {}".format(obs.data.filename_velocity))
            if obs.data.filename_dispersion is not None:
                logger.info("        dispers. file: {}".format(obs.data.filename_dispersion))

            logger.info('        nSubpixels: {}'.format(obs.mod_options.oversample))
        logger.info('\n'+'nCPUs: {}'.format(self.nCPUs))
        logger.info('nWalkers: {}'.format(self.nWalkers))
        # logger.info('lnlike: red_chisq={}'.format(self.red_chisq))
        logger.info('lnlike: oversampled_chisq={}'.format(self.oversampled_chisq))

        logger.info('\n'+'blobs: {}'.format(self.blob_name))


        if ('halo' in gal.model.components.keys()):
            logger.info('\n'+'mvirial_tied: {}'.format(gal.model.components['halo'].mvirial.tied))

        if ('disk+bulge' in gal.model.components.keys()):
            if 'mhalo_relation' in gal.model.components['disk+bulge'].__dict__.keys():
                logger.info('mhalo_relation: {}'.format(gal.model.components['disk+bulge'].mhalo_relation))
            if 'truncate_lmstar_halo' in gal.model.components['disk+bulge'].__dict__.keys():
                logger.info('truncate_lmstar_halo: {}'.format(gal.model.components['disk+bulge'].truncate_lmstar_halo))


        ################################################################
        # --------------------------------
        # Run burn-in
        if self.nBurn > 0:
            logger.info('\nBurn-in:'+'\n'
                        'Start: {}\n'.format(datetime.datetime.now()))
            start = time.time()
            ####

            pos = initial_pos
            for k in range(nBurn_orig):
                # --------------------------------
                # If recovering intermediate save, only start past existing chain length:

                if k < sampler_results_burn.iteration:
                    continue

                logger.info(" k={}, time.time={}, a_frac={}".format( k, datetime.datetime.now(),
                            np.mean(sampler_results_burn.acceptance_fraction)  ) )
                ###

                # Run one sample step:
                pos = sampler_results_burn.run_mcmc(pos, 1)

            #####
            end = time.time()
            elapsed = end-start

            acor_time = sampler_results_burn.get_autocorr_time(tol=10, quiet=True)


            #######################################################################################
            # Return Burn-in info
            # ****
            endtime = str(datetime.datetime.now())
            nthingsmsg = 'nCPU, nParam, nWalker, nBurn = {}, {}, {}, {}'.format(self.nCPUs,
                nDim, self.nWalkers, self.nBurn)
            scaleparammsg = 'Scale param a= {}'.format(self.scale_param_a)
            timemsg = 'Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s)'.format( elapsed, np.floor(elapsed/60.),
                    (elapsed/60.-np.floor(elapsed/60.))*60. )
            macfracmsg = "Mean acceptance fraction: {:0.3f}".format(np.mean(sampler_results_burn.acceptance_fraction))
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
                if self.nBurn < np.max(acor_time) * nBurn_nEff:
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
            if (output_options.do_plotting) & (output_options.f_plot_trace_burnin is not None):
                sampler_results_burn_dict = make_emcee_sampler_results_dict(sampler_results_burn, nBurn=0)
                mcmcResults_burn = MCMCResults(model=gal.model, sampler_results=sampler_results_burn_dict)
                plotting.plot_trace(mcmcResults_burn, fileout=output_options.f_plot_trace_burnin,
                                    overwrite=output_options.overwrite)


        else:
            # --------------------------------
            # No burn-in: set initial position:
            if nBurn_orig > 0:
                logger.info('\nUsing previously completed burn-in'+'\n')

            pos = initial_pos


        #######################################################################################
        # Setup sampler_results:
        # --------------------------------
        # Start backend:
        backend = emcee.backends.HDFBackend(output_options.f_sampler_results, name="mcmc")

        if output_options.overwrite:
            backend.reset(self.nWalkers, nDim)

        sampler_results = emcee.EnsembleSampler(self.nWalkers, nDim, base.log_prob,
                    backend=backend, pool=pool, moves=moves,
                    args=[gal], kwargs=kwargs_dict)

        #######################################################################################
        # *************************************************************************************
        # --------------------------------
        # Run sampler_results: Get start time
        logger.info('\nEnsemble sampling:\n'
                    'Start: {}\n'.format(datetime.datetime.now()))
        start = time.time()

        if sampler_results.iteration > 0:
            logger.info('\n   Resuming with existing sampler_results chain at iteration ' +
                        str(sampler_results.iteration) + '\n')
            pos = sampler_results.get_last_sample()

        # --------------------------------
        # Run sampler_results: output info at each step
        for ii in range(self.nSteps):
            # --------------------------------
            # If continuing chain, only start past existing chain length:
            if ii < sampler_results.iteration:
                continue
            # --------------------------------
            # Only do one step at a time:
            pos = sampler_results.run_mcmc(pos, 1)
            # --------------------------------

            # --------------------------------
            # Give output info about this step:
            nowtime = str(datetime.datetime.now())
            stepinfomsg = "ii={}, a_frac={}".format( ii, np.mean(sampler_results.acceptance_fraction) )
            timemsg = " time.time()={}".format(nowtime)
            logger.info( stepinfomsg+timemsg )

            acor_time = sampler_results.get_autocorr_time(tol=10, quiet=True)
            #acor_time = sampler_results.get_autocorr_time(quiet=True)
            logger.info( "{}: acor_time ={}".format(ii, np.array(acor_time) ) )

            # --------------------------------
            # Case: test for convergence and truncate early:
            # Criteria checked: whether acceptance fraction within (minAF, maxAF),
            #                   and whether total number of steps > nEff * average autocorrelation time:
            #                   to make sure the paramter space is well explored.
            if ((self.minAF is not None) & (self.maxAF is not None) & \
                    (self.nEff is not None) & (acor_time is not None)):
                if ((self.minAF < np.mean(sampler_results.acceptance_fraction) < self.maxAF) & \
                    ( ii > np.max(acor_time) * self.nEff )):
                        if ii == self.acor_force_min:
                            logger.info(" Enforced min step limit: {}.".format(ii+1))
                        if ii >= self.acor_force_min:
                            logger.info(" Finishing calculations early at step {}.".format(ii+1))
                            break


        # --------------------------------
        # Check if it failed to converge before the max number of steps, if doing convergence testing
        finishedSteps= ii+1
        if (finishedSteps  == self.nSteps) & ((self.minAF is not None) & \
                    (self.maxAF is not None) & (self.nEff is not None)):
            logger.info(" Caution: no convergence within nSteps={}.".format(self.nSteps))

        # --------------------------------
        # Finishing info for fitting:
        end = time.time()
        elapsed = end-start
        logger.info("Finished {} steps".format(finishedSteps)+"\n")

        acor_time = sampler_results.get_autocorr_time(tol=10, quiet=True)

        #######################################################################################
        # ***********
        # Consider overall acceptance fraction
        endtime = str(datetime.datetime.now())
        nthingsmsg = 'nCPU, nParam, nWalker, nSteps = {}, {}, {}, {}'.format(self.nCPUs,
            nDim, self.nWalkers, self.nSteps)
        scaleparammsg = 'Scale param a= {}'.format(self.scale_param_a)
        timemsg = 'Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s)'.format(elapsed, np.floor(elapsed/60.),
                (elapsed/60.-np.floor(elapsed/60.))*60. )
        macfracmsg = "Mean acceptance fraction: {:0.3f}".format(np.mean(sampler_results.acceptance_fraction))
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


        if self.nCPUs > 1:
            pool.close()
            sampler_results.pool.close()
            sampler_results_burn.pool.close()

        ##########################################
        ##########################################
        ##########################################
        # --------------------------------
        # Setup sampler_results dict:
        sampler_results_dict = make_emcee_sampler_results_dict(sampler_results, nBurn=0)

        # --------------------------------
        # Bundle the results up into a results class:
        mcmcResults = MCMCResults(model=gal.model, sampler_results=sampler_results_dict,
                                  linked_posterior_names=self.linked_posterior_names,
                                  blob_name=self.blob_name,
                                  nPostBins=self.nPostBins)

        if self.oversampled_chisq:
            mcmcResults.oversample_factor_chisq = OrderedDict()
            for obs_name in gal.observations:
                obs = gal.observations[obs_name]
                mcmcResults.oversample_factor_chisq[obs_name] = obs.data.oversample_factor_chisq

        # Do all analysis, plotting, saving:
        mcmcResults.analyze_plot_save_results(gal, output_options=output_options)


        return mcmcResults



class MCMCResults(base.BayesianFitResults, base.FitResults):
    """
    Class to hold results of MCMC fitting to DYSMALPY models.

    Notes:
    ------
        `emcee` sampler_results object is ported to a dictionary in
            `mcmcResults.sampler_results`

        The name of the free parameters in the chain are accessed through `mcmcResults.chain_param_names`, or more generally (separate model + parameter names) through `mcmcResults.free_param_names`

    Optional Attribute:
    ----------------------
        `linked_posterior_names`
            Indicates if best-fit parameters should be measured in multi-dimensional histogram space.
            It takes a list of linked parameter sets, where each set consists of len-2 tuples/lists of
            the component + parameter names.


    Structure Explanation:
    ----------------------
    #. To analyze component + param 1 and 2 together, and then 3 and 4 together: `linked_posterior_names = [joint_param_bundle1, joint_param_bundle2]` with `joint_param_bundle1 = [[cmp1, par1], [cmp2, par2]]` and `joint_param_bundle2 = [[cmp3, par3], [cmp4, par4]]`, for a full array of: `linked_posterior_names = [[[cmp1, par1], [cmp2, par2]],[[cmp3, par3], [cmp4, par4]]]`.

    #. To analyze component + param 1 and 2 together: `linked_posterior_names = [joint_param_bundle1]` with `joint_param_bundle1 = [[cmp1, par1], [cmp2, par2]]`, for a full array of `linked_posterior_names = [[[cmp1, par1], [cmp2, par2]]]`.
            Example: Look at halo: mvirial and disk+bulge: total_mass together
                `linked_posterior_names = [[['halo', 'mvirial'], ['disk+bulge', 'total_mass']]]`
    """
    
    def __init__(self, model=None, sampler_results=None,
                 linked_posterior_names=None,
                 blob_name=None, nPostBins=50):
        
        super(MCMCResults, self).__init__(model=model, blob_name=blob_name,
                                          fit_method='MCMC', 
                                          linked_posterior_names=linked_posterior_names, 
                                          sampler_results=sampler_results, 
                                          nPostBins=nPostBins)

    def __setstate__(self, state):
        # Compatibility hacks
        super(MCMCResults, self).__setstate__(state)

        # # ---------
        # if ('sampler' not in state.keys()) & ('sampler_results' in state.keys()):
        #     self._setup_samples_blobs()


    def _setup_samples_blobs(self):
        # Note: 
        # self.sampler.samples replaces self.sampler_results['flatchain'], and
        # self.sampler.blobs   replaces self.sampler_results['flatblobs']

        if 'blobs' in self.sampler_results.keys():
            blobset = True
        else:
            blobset = False

        if ('flatblobs' not in self.sampler_results.keys()) & (blobset):
            if len(self.sampler_results['blobs'].shape) == 2:
                # Only 1 blob: nSteps, nWalkers:
                flatblobs = self.sampler_results['blobs'].reshape(-1)
            elif len(self.sampler_results['blobs'].shape) == 3:
                # Multiblobs; nSteps, nWalkers, nBlobs
                flatblobs = self.sampler_results['blobs'].reshape(-1,self.sampler_results['blobs'].shape[2])
            else:
                raise ValueError("sampler_results blob length not recognized")
        elif (not blobset):
            flatblobs = None
        else: 
            flatblobs = self.sampler_results['flatblobs']

        self.sampler = base.BayesianSampler(samples=self.sampler_results['flatchain'], 
                                            blobs=flatblobs)


    def reload_sampler_results(self, filename=None):
        """Reload the MCMC sampler_results saved earlier"""
        if filename is None:
            #filename = self.f_sampler_results
            raise ValueError

        hdf5_aliases = ['h5', 'hdf5']
        pickle_aliases = ['pickle', 'pkl', 'pcl']
        if (filename.split('.')[-1].lower() in hdf5_aliases):
            self.sampler_results = _reload_sampler_results_hdf5(filename=filename)

        elif (filename.split('.')[-1].lower() in pickle_aliases):
            self.sampler_results = _reload_sampler_results_pickle(filename=filename)





def initialize_walkers(model, nWalkers=None):
    """
    Initialize a set of MCMC walkers by randomly drawing from the
    model set parameter priors
    """
    stack_rand = []
    pfree_dict = model.get_free_parameter_keys()
    comps_names = pfree_dict.keys()

    for compn in comps_names:
        comp = model.components.__getitem__(compn)
        params_names = pfree_dict[compn].keys()
        for paramn in params_names:
            if (pfree_dict[compn][paramn] >= 0) :
                # Free parameter: randomly sample from prior nWalker times:
                param_rand = comp.__getattribute__(paramn).prior.sample_prior(comp.__getattribute__(paramn),
                                    modelset=model, N=nWalkers)
                stack_rand.append(param_rand)
    pos = np.array(list(zip(*stack_rand)))        # should have shape:   (nWalkers, nDim)
    return pos

def make_emcee_sampler_results_dict(sampler_results, nBurn=0, emcee_vers=3):
    """
    Save chain + key results from emcee sampler_results instance to a dict,
    as the emcee sampler_resultss aren't pickleable.
    """

    if emcee_vers == 3:
        return _make_emcee_sampler_results_dict_v3(sampler_results, nBurn=nBurn)
    elif emcee_vers == 2:
        return _make_emcee_sampler_results_dict_v2(sampler_results, nBurn=nBurn)
    else:
        raise ValueError("Emcee version {} not supported!".format(emcee_vers))


def _make_emcee_sampler_results_dict_v2(sampler_results, nBurn=0):
    """ Syntax for emcee v2.2.1 """
    # Cut first nBurn steps, to avoid the edge cases that are rarely explored.
    chain = sampler_results.chain[:, nBurn:, :]
    flatchain = chain.reshape((-1, sampler_results.dim))
    # Walkers, iterations
    probs =     sampler_results.lnprobability[:, nBurn:]
    flatprobs = probs.reshape((-1))

    try:
        acor_time = sampler_results.get_autocorr_time(low=5, c=10)
    except:
        acor_time = None


    # Make a dictionary:
    sampler_results_dict = { 'chain':             chain,
                     'flatchain':         flatchain,
                     'lnprobability':     probs,
                     'flatlnprobability': flatprobs,
                     'nIter':             sampler_results.iterations,
                     'nParam':            sampler_results.dim,
                     'nCPU':              sampler_results.threads,
                     'nWalkers':          len(sampler_results.chain),
                     'acceptance_fraction': sampler_results.acceptance_fraction,
                     'acor_time': acor_time }

    if sampler_results.blobs is not None:
        if len(sampler_results.blobs) > 0:
            sampler_results_dict['blobs'] = np.array(sampler_results.blobs[nBurn:])

            if len(np.shape(sampler_results.blobs)) == 2:
                # Only 1 blob: nSteps, nWalkers:
                sampler_results_dict['flatblobs'] = np.array(sampler_results_dict['blobs']).reshape(-1)
            elif len(np.shape(sampler_results.blobs)) == 3:
                # Multiblobs; nSteps, nWalkers, nBlobs
                sampler_results_dict['flatblobs'] = np.array(sampler_results_dict['blobs']).reshape(-1,np.shape(sampler_results.blobs)[2])
            else:
                raise ValueError("sampler_results blob length not recognized")


    return sampler_results_dict


def _make_emcee_sampler_results_dict_v3(sampler_results, nBurn=0):
    """ Syntax for emcee v3 """
    # Cut first nBurn steps, to avoid the edge cases that are rarely explored.
    samples = np.swapaxes(
        sampler_results.get_chain(),0,1
        )[:, nBurn:, :].reshape((-1, sampler_results.ndim))
    # Walkers, iterations
    probs = sampler_results.get_log_prob()[:, nBurn:].reshape((-1))

    acor_time = sampler_results.get_autocorr_time(tol=10, quiet=True)

    try:
        nCPUs = sampler_results.pool._processes   # sampler_results.threads
    except:
        nCPUs = 1

    # Make a dictionary:
    sampler_results_dict = { 
        'chain':   np.swapaxes(sampler_results.get_chain(),0,1)[:, nBurn:, :],
        'lnprobability':       sampler_results.get_log_prob()[:, nBurn:],
        'flatchain':            samples,
        'flatlnprobability':    probs,
        'nIter':                sampler_results.iteration,
        'nParam':               sampler_results.ndim,
        'nCPU':                 nCPUs,
        'nWalkers':             sampler_results.nwalkers,
        'acceptance_fraction':  sampler_results.acceptance_fraction,
        'acor_time':            acor_time 
    }

    if sampler_results.get_blobs() is not None:
        if len(sampler_results.get_blobs()) > 0:
            if len(np.shape(sampler_results.get_blobs())) == 2:
                # Only 1 blob: nSteps, nWalkers:
                sampler_results_dict['blobs'] = sampler_results.get_blobs()[nBurn:, :]
                flatblobs = np.array(sampler_results_dict['blobs']).reshape(-1)
            elif len(np.shape(sampler_results.get_blobs())) == 3:
                # Multiblobs; nSteps, nWalkers, nBlobs
                sampler_results_dict['blobs'] = sampler_results.get_blobs()[nBurn:, :, :]
                flatblobs = np.array(sampler_results_dict['blobs']).reshape(-1,np.shape(sampler_results.get_blobs())[2])
            else:
                raise ValueError("sampler_results blob shape not recognized")

            sampler_results_dict['flatblobs'] = flatblobs

    return sampler_results_dict


def _reload_sampler_results_hdf5(filename=None, backend_name='mcmc'):
    # Load backend from file
    backend = emcee.backends.HDFBackend(filename, name=backend_name)
    return _make_sampler_results_dict_from_hdf5(backend)

def _make_sampler_results_dict_from_hdf5(b):
    """  Construct a dysmalpy 'sampler_results_dict' out of the chain info stored in the emcee v3 HDF5 file """
    nwalkers =  b.shape[0]
    ndim =      b.shape[1]

    chain =     np.swapaxes(b.get_chain(), 0, 1)
    flatchain = chain.reshape((-1, ndim))

    # Walkers, iterations
    probs =     np.swapaxes(b.get_log_prob(), 0, 1)
    flatprobs = probs.reshape(-1)

    acor_time = b.get_autocorr_time(tol=10, quiet=True)

    # Make a dictionary:
    sampler_results_dict = { 'chain':                chain,
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
        sampler_results_dict['blobs'] = b.get_blobs()
        if len(b.get_blobs().shape) == 2:
            # Only 1 blob: nSteps, nWalkers:
            flatblobs = np.array(sampler_results_dict['blobs']).reshape(-1)
        elif len(b.get_blobs().shape) == 3:
            # Multiblobs; nSteps, nWalkers, nBlobs
            flatblobs = np.array(sampler_results_dict['blobs']).reshape(-1,np.shape(sampler_results_dict['blobs'])[2])
        else:
            raise ValueError("sampler_results blob shape not recognized")

        sampler_results_dict['flatblobs'] = flatblobs


    return sampler_results_dict

def _reload_sampler_results_pickle(filename=None):
    return load_pickle(filename)


def reinitialize_emcee_sampler_results(sampler_results_dict, gal=None, fitter=None):
    """
    Re-setup emcee sampler_results, using existing chain / etc, so more steps can be run.
    """

    kwargs_dict = {'fitter': fitter}

    # This will break for updated version of emcee
    # works for emcee v2.2.1
    if emcee.__version__ == '2.2.1':

        sampler_results = emcee.EnsembleSampler(fitter.nWalkers, fitter.nParam,
                    base.log_prob, args=[gal], kwargs=kwargs_dict, a=fitter.scale_param_a,
                    threads=sampler_results_dict['nCPU'])

        sampler_results._chain = copy.deepcopy(sampler_results_dict['chain'])
        sampler_results._blobs = list(copy.deepcopy(sampler_results_dict['blobs']))
        sampler_results._lnprob = copy.deepcopy(sampler_results_dict['lnprobability'])
        sampler_results.iterations = sampler_results_dict['nIter']
        sampler_results.naccepted = np.array(sampler_results_dict['nIter']*copy.deepcopy(sampler_results_dict['acceptance_fraction']),
                            dtype=np.int64)
    ###
    elif int(emcee.__version__[0]) >= 3:
        # This is based off of HDF5 files, which automatically makes it easy to reload + resetup the sampler_results
        raise ValueError("emcee >=3 uses HDF5 files, so re-initialization not necessary!")

    ###
    else:
        try:
            backend = emcee.Backend()
            backend.nwalkers = sampler_results_dict['nWalkers']
            backend.ndim = sampler_results_dict['nParam']
            backend.iteration = sampler_results_dict['nIter']
            backend.accepted = np.array(sampler_results_dict['nIter']*sampler_results_dict['acceptance_fraction'],
                                dtype=np.int64)
            backend.chain = sampler_results_dict['chain']
            backend.log_prob = sampler_results_dict['lnprobability']
            backend.blobs = sampler_results_dict['blobs']
            backend.initialized = True


            sampler_results = emcee.EnsembleSampler(sampler_results_dict['nWalkers'],
                        sampler_results_dict['nParam'],
                        base.log_prob,
                        args=[gal], kwargs=kwargs_dict,
                        backend=backend,
                        a=fitter.scale_param_a,
                        threads=sampler_results_dict['nCPU'])

        except:
            raise ValueError



    return sampler_results


def _reload_all_fitting_mcmc(filename_galmodel=None, filename_results=None):
    gal = galaxy.load_galaxy_object(filename=filename_galmodel)
    results = MCMCResults()
    results.reload_results(filename=filename_results)
    return gal, results

