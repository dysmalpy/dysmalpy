# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
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
from multiprocessing import cpu_count, Pool

# DYSMALPY code
from dysmalpy.data_io import ensure_dir, load_pickle, dump_pickle
from dysmalpy import plotting
from dysmalpy import galaxy
from dysmalpy.parameters import UniformLinearPrior
from dysmalpy.instrument import DoubleBeam, Moffat, GaussianBeam
# from dysmalpy.utils import fit_uncertainty_ellipse
# from dysmalpy import utils_io as dpy_utils_io
from dysmalpy import utils as dpy_utils
from dysmalpy.fitting import base


# Third party imports
import os
import numpy as np
from collections import OrderedDict
import six
import astropy.units as u
import copy
import emcee

_emcee_version = int(emcee.__version__[0])
if _emcee_version >= 3:
    import h5py

import time, datetime

from scipy.stats import gaussian_kde
from scipy.optimize import fmin


__all__ = ['MCMCFitter', 'MCMCResults']


# ACOR SETTINGS
acor_force_min = 49
# Force it to run for at least 50 steps, otherwise acor times might be completely wrong.

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')



class MCMCFitter(base.Fitter):
    """
    Class to hold the MCMC fitter attributes + methods
    """
    def __init__(self, **kwargs):

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
        self.save_intermediate_sampler_chain = True
        self.nStep_intermediate_save = 5
        self.continue_steps = False

        self.nPostBins = 50
        self.linked_posterior_names = None

        self.input_sampler = None


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
                MCMCResults class instance containing the bestfit parameters, sampler information, etc.
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
            gal = setup_oversampled_chisq(gal)
        # +++++++++++++++++++++++

        # Set output options: filenames / which to save, etc
        output_options.set_output_options(gal, self)

        # MUST INCLUDE MCMC-SPECIFICS NOW!
        base._check_existing_files_overwrite(output_options, fit_type='mcmc', fitter=self)

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

        return mcmcResults






    def _fit_emcee_221(self, gal, output_options):
        # --------------------------------
        # Initialize emcee sampler
        kwargs_dict = {'fitter': self}

        nBurn_orig = kwargs_fit['nBurn']

        nDim = gal.model.nparams_free


        if (not self.continue_steps) & ((not self.save_intermediate_sampler_chain) \
            | (not os.path.isfile(output_options.f_sampler_tmp))):
            sampler = emcee.EnsembleSampler(self.nWalkers, nDim, log_prob,
                        args=[gal], kwargs=kwargs_dict,
                        a = self.scale_param_a, threads = self.nCPUs)

            # --------------------------------
            # Initialize walker starting positions
            initial_pos = initialize_walkers(gal.model, nWalkers=self.nWalkers)

        elif self.continue_steps:
            self.nBurn = 0
            if self.input_sampler is None:
                try:
                    self.input_sampler = load_pickle(output_options.f_sampler)
                except:
                    message = "Couldn't find existing sampler in {}.".format(output_options.f_sampler)
                    message += '\n'
                    message += "Must set input_sampler if you will restart the sampler."
                    raise ValueError(message)

            sampler = reinitialize_emcee_sampler(self.input_sampler, gal=gal,
                                kwargs_dict=kwargs_dict,
                                scale_param_a=self.scale_param_a)

            initial_pos = self.input_sampler['chain'][:,-1,:]
            if self.blob_name is not None:
                blob = self.input_sampler['blobs']

            # Close things
            self.input_sampler = None

        elif self.save_intermediate_sampler_chain & (os.path.isfile(output_options.f_sampler_tmp)):
            self.input_sampler = load_pickle(output_options.f_sampler_tmp)

            sampler = reinitialize_emcee_sampler(self.input_sampler, gal=gal,
                                                 fitter=self)
            self.nBurn = nBurn_orig - (self.input_sampler['burn_step_cur'] + 1)

            initial_pos = self.input_sampler['chain'][:,-1,:]
            if self.blob_name is not None:
                blob = self.input_sampler['blobs']

            # If it saved after burn finished, but hasn't saved any of the normal steps: reset sampler
            if ((self.nBurn == 0) & (self.input_sampler['step_cur'] < 0)):
                blob = None
                sampler.reset()
                if self.blob_name is not None:
                     sampler.clear_blobs()

            # Close things
            input_sampler = None


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
            for k in six.moves.xrange(nBurn_orig):
                # --------------------------------
                # If recovering intermediate save, only start past existing chain length:
                if self.save_intermediate_sampler_chain:
                    if k < sampler.chain.shape[1]:
                        continue

                logger.info(" k={}, time.time={}, a_frac={}".format( k, datetime.datetime.now(),
                            np.mean(sampler.acceptance_fraction)  ) )
                ###
                pos_cur = pos.copy()    # copy just in case things are set strangely

                # Run one sample step:
                if self.blob_name is not None:
                    pos, prob, state, blob = sampler.run_mcmc(pos_cur, 1, lnprob0=prob,
                            rstate0=state, blobs0 = blob)
                else:
                    pos, prob, state = sampler.run_mcmc(pos_cur, 1, lnprob0=prob, rstate0=state)


                # --------------------------------
                # Save intermediate steps if set:
                if self.save_intermediate_sampler_chain:
                    if ((k+1) % self.nStep_intermediate_save == 0):
                        sampler_dict_tmp = make_emcee_sampler_dict(sampler, nBurn=0, emcee_vers=2)
                        sampler_dict_tmp['burn_step_cur'] = k
                        sampler_dict_tmp['step_cur'] = -99
                        if output_options.f_sampler_tmp is not None:
                            # Save stuff to file, for future use:
                            dump_pickle(sampler_dict_tmp, filename=output_options.f_sampler_tmp, overwrite=True)
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
            nthingsmsg = 'nCPU, nParam, nWalker, nBurn = {}, {}, {}, {}'.format(self.nCPUs,
                nDim, self.nWalkers, self.nBurn)
            scaleparammsg = 'Scale param a= {}'.format(self.scale_param_a)
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
            if (self.save_burn) & (output_options.f_burn_sampler is not None):
                sampler_burn = make_emcee_sampler_dict(sampler, nBurn=0, emcee_vers=2)
                # Save stuff to file, for future use:
                dump_pickle(sampler_burn, filename=output_options.f_burn_sampler, overwrite=output_options.overwrite)


            # --------------------------------
            # Plot burn-in trace, if output file set
            if (output_options.do_plotting) & (output_options.f_plot_trace_burnin is not None):
                sampler_burn = make_emcee_sampler_dict(sampler, nBurn=0, emcee_vers=2)
                mcmcResultsburn = MCMCResults(model=gal.model, sampler=sampler_burn)
                plotting.plot_trace(mcmcResultsburn, fileout=output_options.f_plot_trace_burnin,
                            overwrite=output_options.overwrite)

            # Reset sampler after burn-in:
            sampler.reset()
            if self.blob_name is not None:
                 sampler.clear_blobs()

        else:
            # --------------------------------
            # No burn-in: set initial position:
            if nBurn_orig > 0:
                logger.info('\nUsing previously completed burn-in'+'\n')

            pos = np.array(initial_pos)
            prob = None
            state = None

            if (not self.continue_steps) | (not self.save_intermediate_sampler_chain):
                blob = None

        #######################################################################################
        # ****
        # --------------------------------
        # Run sampler: Get start time
        logger.info('\nEnsemble sampling:\n'
                    'Start: {}\n'.format(datetime.datetime.now()))
        start = time.time()

        if sampler.chain.shape[1] > 0:
            logger.info('\n   Resuming with existing sampler chain at iteration ' +
                        str(sampler.iteration) + '\n')
            pos = sampler['chain'][:,-1,:]

        # --------------------------------
        # Run sampler: output info at each step
        for ii in six.moves.xrange(self.nSteps):

            # --------------------------------
            # If continuing chain, only start past existing chain length:
            if self.continue_steps | self.save_intermediate_sampler_chain:
                if ii < sampler.chain.shape[1]:
                    continue

            pos_cur = pos.copy()    # copy just in case things are set strangely

            # --------------------------------
            # Only do one step at a time:
            if self.blob_name is not None:
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
            if ((self.minAF is not None) & (self.maxAF is not None) & \
                    (self.nEff is not None) & (acor_time is not None)):
                if ((self.minAF < np.mean(sampler.acceptance_fraction) < self.maxAF) & \
                    ( ii > np.max(acor_time) * self.nEff )):
                        if ii == acor_force_min:
                            logger.info(" Enforced min step limit: {}.".format(ii+1))
                        if ii >= acor_force_min:
                            logger.info(" Finishing calculations early at step {}.".format(ii+1))
                            break

            # --------------------------------
            # Save intermediate steps if set:
            if self.save_intermediate_sampler_chain:
                if ((ii+1) % self.nStep_intermediate_save == 0):
                    sampler_dict_tmp = make_emcee_sampler_dict(sampler, nBurn=0, emcee_vers=2)
                    sampler_dict_tmp['burn_step_cur'] = nBurn_orig - 1
                    sampler_dict_tmp['step_cur'] = ii
                    if output_options.f_sampler_tmp is not None:
                        # Save stuff to file, for future use:
                        dump_pickle(sampler_dict_tmp, filename=output_options.f_sampler_tmp, overwrite=True)
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
            acor_time = sampler.get_autocorr_time(low=5, c=10)
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


        if output_options.f_sampler is not None:
            # Save stuff to file, for future use:
            dump_pickle(sampler_dict, filename=output_options.f_sampler, overwrite=output_options.overwrite)


        # --------------------------------
        # Cleanup intermediate saves:
        if self.save_intermediate_sampler_chain & (output_options.f_sampler_tmp is not None):
            if os.path.isfile(output_options.f_sampler_tmp):
                os.remove(output_options.f_sampler_tmp)
        # --------------------------------

        if self.nCPUs > 1:
            sampler.pool.close()

        ##########################################
        ##########################################
        ##########################################

        # --------------------------------
        # Bundle the results up into a results class:
        mcmcResults = MCMCResults(model=gal.model, sampler=sampler_dict,
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

        # Check length of sampler if not overwriting:
        if (not output_options.overwrite):
            if os.path.isfile(output_options.f_sampler):
                backend = emcee.backends.HDFBackend(output_options.f_sampler, name='mcmc')
                try:
                    if backend.get_chain().shape[0] >= self.nSteps:
                        if output_options.f_results is not None:
                            if os.path.isfile(output_options.f_results):
                                msg = "overwrite={}, and 'f_sampler' already contains {} steps,".format(output_options.overwrite,
                                                        backend.get_chain().shape[0])
                                msg += " so the fit will not be saved.\n Specify new outfile or delete old files."
                                logger.warning(msg)
                                return None
                    else:
                        pass
                except:
                    pass


        # --------------------------------
        # Initialize emcee sampler

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

        backend_burn = emcee.backends.HDFBackend(output_options.f_sampler, name="burnin_mcmc")

        if output_options.overwrite:
            backend_burn.reset(self.nWalkers, nDim)

        sampler_burn = emcee.EnsembleSampler(self.nWalkers, nDim, log_prob,
                    backend=backend_burn, pool=pool, moves=moves,
                    args=[gal], kwargs=kwargs_dict)

        nBurnCur = sampler_burn.iteration

        self.nBurn = nBurn_orig - nBurnCur


        # --------------------------------
        # Initialize walker starting positions
        if sampler_burn.iteration == 0:
            initial_pos = initialize_walkers(gal.model, nWalkers=self.nWalkers)
        else:
            initial_pos = sampler_burn.get_last_sample()


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
            nthingsmsg = 'nCPU, nParam, nWalker, nBurn = {}, {}, {}, {}'.format(self.nCPUs,
                nDim, self.nWalkers, self.nBurn)
            scaleparammsg = 'Scale param a= {}'.format(self.scale_param_a)
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
            if (output_options.do_plotting) & (output_options.f_plot_trace_burnin is not None):
                sampler_burn_dict = make_emcee_sampler_dict(sampler_burn, nBurn=0)
                mcmcResults_burn = MCMCResults(model=gal.model, sampler=sampler_burn_dict)
                plotting.plot_trace(mcmcResults_burn, fileout=output_options.f_plot_trace_burnin,
                                    overwrite=output_options.overwrite)


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
        backend = emcee.backends.HDFBackend(output_options.f_sampler, name="mcmc")

        if output_options.overwrite:
            backend.reset(self.nWalkers, nDim)

        sampler = emcee.EnsembleSampler(self.nWalkers, nDim, log_prob,
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
            logger.info('\n   Resuming with existing sampler chain at iteration ' +
                        str(sampler.iteration) + '\n')
            pos = sampler.get_last_sample()

        # --------------------------------
        # Run sampler: output info at each step
        for ii in six.moves.xrange(self.nSteps):
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
            if ((self.minAF is not None) & (self.maxAF is not None) & \
                    (self.nEff is not None) & (acor_time is not None)):
                if ((self.minAF < np.mean(sampler.acceptance_fraction) < self.maxAF) & \
                    ( ii > np.max(acor_time) * self.nEff )):
                        if ii == acor_force_min:
                            logger.info(" Enforced min step limit: {}.".format(ii+1))
                        if ii >= acor_force_min:
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

        acor_time = sampler.get_autocorr_time(tol=10, quiet=True)

        #######################################################################################
        # ***********
        # Consider overall acceptance fraction
        endtime = str(datetime.datetime.now())
        nthingsmsg = 'nCPU, nParam, nWalker, nSteps = {}, {}, {}, {}'.format(self.nCPUs,
            nDim, self.nWalkers, self.nSteps)
        scaleparammsg = 'Scale param a= {}'.format(self.scale_param_a)
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


        if self.nCPUs > 1:
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
        mcmcResults = MCMCResults(model=gal.model, sampler=sampler_dict,
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



class MCMCResults(base.FitResults):
    """
    Class to hold results of MCMC fitting to DYSMALPY models.

    Note: emcee sampler object is ported to a dictionary in
            mcmcResults.sampler

        The name of the free parameters in the chain are accessed through:
            mcmcResults.chain_param_names,
                or more generally (separate model + parameter names) through
                mcmcResults.free_param_names

        Optional attribute:
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
    def __init__(self, model=None, sampler=None,
                 linked_posterior_names=None,
                 blob_name=None, nPostBins=50):

        self.sampler = sampler
        self.linked_posterior_names = linked_posterior_names
        self.nPostBins =nPostBins

        self.bestfit_parameters_l68_err = None
        self.bestfit_parameters_u68_err = None
        self.bestfit_parameters_l68 = None
        self.bestfit_parameters_u68 = None


        super(MCMCResults, self).__init__(model=model, blob_name=blob_name,
                                          fit_method='MCMC')


    def analyze_plot_save_results(self, gal, output_options=None):
        """
        Wrapper for post-sample analysis + plotting -- in case code broke and only have sampler saved.

        """

        if output_options.f_chain_ascii is not None:
            self.save_chain_ascii(filename=output_options.f_chain_ascii, overwrite=output_options.overwrite)

        # Get the best-fit values, uncertainty bounds from marginalized posteriors
        self.analyze_posterior_dist(gal=gal)

        # Update theta to best-fit:
        gal.model.update_parameters(self.bestfit_parameters)

        if self.blob_name is not None:
            if isinstance(self.blob_name, str):
                blob_names = [self.blob_name]
            else:
                blob_names = self.blob_name[:]

            for blobn in blob_names:
                if blobn.lower() == 'fdm':
                    self.analyze_dm_posterior_dist(gal=gal, blob_name=self.blob_name)  # here blob_name should be the *full* list
                elif blobn.lower() == 'mvirial':
                    self.analyze_mvirial_posterior_dist(gal=gal, blob_name=self.blob_name)
                elif blobn.lower() == 'alpha':
                    self.analyze_alpha_posterior_dist(gal=gal, blob_name=self.blob_name)
                elif blobn.lower() == 'rb':
                    self.analyze_rb_posterior_dist(gal=gal, blob_name=self.blob_name)


        gal.create_model_data()

        self.bestfit_redchisq = base.chisq_red(gal)
        self.bestfit_chisq = base.chisq_eval(gal)

        # self.vmax_bestfit = gal.model.get_vmax()

        if output_options.f_results is not None:
            self.save_results(filename=output_options.f_results, overwrite=output_options.overwrite)

        if output_options.f_model is not None:
            # Save model w/ updated theta equal to best-fit:
            gal.preserve_self(filename=output_options.f_model,
                              save_data=output_options.save_data,
                              overwrite=output_options.overwrite)



        if output_options.save_model_bestfit & (output_options.f_model_bestfit is not None):
            gal.save_model_data(filenames=output_options.f_model_bestfit, overwrite=output_options.overwrite)


        if output_options.save_bestfit_cube & (output_options.f_bestfit_cube is not None):
            for obs_name in gal.observations:
                obs = gal.observations[obs_name]
                fcube = output_options.f_bestfit_cube[obs_name]
                obs.model_cube.data.write(fcube, overwrite=output_options.overwrite)

        # --------------------------------
        # Plot trace, if output file set
        if (output_options.do_plotting) & (output_options.f_plot_trace is not None) :
            self.plot_trace(fileout=output_options.f_plot_trace,
                            overwrite=output_options.overwrite)

        # --------------------------------
        # Plot results: corner plot, best-fit
        if (output_options.do_plotting) & (output_options.f_plot_param_corner is not None):
            self.plot_corner(gal=gal, fileout=output_options.f_plot_param_corner, overwrite=output_options.overwrite)

        if (output_options.do_plotting) & (output_options.f_plot_bestfit is not None):
            self.plot_bestfit(gal, fileout=output_options.f_plot_bestfit,
                              overwrite=output_options.overwrite)

        # --------------------------------
        # Save velocity / other profiles to ascii file:
        if output_options.save_vel_ascii & (output_options.f_vel_ascii is not None):
            for tracer in gal.model.dispersions:
                self.save_bestfit_vel_ascii(tracer, gal.model,
                                            filename=output_options.f_vel_ascii[tracer],
                                            overwrite=output_options.overwrite)


        if ((output_options.save_vel_ascii)) & ((output_options.f_vcirc_ascii is not None) or \
             (output_options.f_mass_ascii is not None)):
            self.save_bestfit_vcirc_mass_profiles(gal, outpath=output_options.outdir,
                    fname_intrinsic=output_options.f_vcirc_ascii,
                    fname_intrinsic_m=output_options.f_mass_ascii,
                    overwrite=output_options.overwrite)

        if (output_options.save_reports):
            if (output_options.f_report_pretty is not None):
                self.results_report(gal=gal, filename=output_options.f_report_pretty,
                                    report_type='pretty',
                                    output_options=output_options,
                                    overwrite=output_options.overwrite)
            if (output_options.f_report_machine is not None):
                self.results_report(gal=gal, filename=output_options.f_report_machine,
                                    report_type='machine',
                                    output_options=output_options,
                                    overwrite=output_options.overwrite)



    def mod_linear_param_posterior(self, gal=None):
        linear_posterior = []
        j = -1
        for cmp in gal.model.fixed:
            for pm in gal.model.fixed[cmp]:
                if gal.model.fixed[cmp][pm] | bool(gal.model.tied[cmp][pm]):
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

    def analyze_posterior_dist(self, gal=None):
        """
        Default analysis of posterior distributions from MCMC fitting:
            look at marginalized posterior distributions, and
            extract the best-fit value (peak of KDE), and extract the +- 1 sigma uncertainty bounds
            (eg, the 16%/84% distribution of posteriors)

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
            yb, xb = np.histogram(self.sampler['flatchain'][:,i], bins=self.nPostBins)
            wh_pk = np.where(yb == yb.max())[0][0]
            mcmc_peak_hist[i] = np.average([xb[wh_pk], xb[wh_pk+1]])

        ## Use max prob as guess to get peak value of the gaussian KDE, to find 'best-fit' of the posterior:
        mcmc_param_bestfit = find_peak_gaussian_KDE(self.sampler['flatchain'], mcmc_peak_hist)

        # --------------------------------------------
        if self.linked_posterior_names is not None:
            # Make sure the param of self is updated
            #   (for ref. when reloading saved mcmcResult objects)
            linked_posterior_ind_arr = get_linked_posterior_indices(self)
            guess = mcmc_param_bestfit.copy()
            bestfit_theta_linked = get_linked_posterior_peak_values(self.sampler['flatchain'],
                            guess=guess, linked_posterior_ind_arr=linked_posterior_ind_arr,
                            nPostBins=self.nPostBins)

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


    def analyze_dm_posterior_dist(self, gal=None, blob_name=None):
        """
        Default analysis of posterior distributions of fDM from MCMC fitting:
            look at marginalized posterior distributions, and
            extract the best-fit value (peak of KDE), and extract the +- 1 sigma uncertainty bounds
            (eg, the 16%/84% distribution of posteriors)

        """
        fdm_mcmc_param_bestfit = gal.model.get_dm_frac_r_ap()
        self.analyze_blob_posterior_dist(bestfit=fdm_mcmc_param_bestfit, parname='fdm', blob_name=blob_name)

    def analyze_mvirial_posterior_dist(self, gal=None, blob_name=None):
        mvirial_mcmc_param_bestfit = gal.model.get_mvirial()
        self.analyze_blob_posterior_dist(bestfit=mvirial_mcmc_param_bestfit, parname='mvirial', blob_name=blob_name)

    def analyze_alpha_posterior_dist(self, gal=None, blob_name=None):
        alpha_mcmc_param_bestfit = gal.model.get_halo_alpha()
        self.analyze_blob_posterior_dist(bestfit=alpha_mcmc_param_bestfit, parname='alpha', blob_name=blob_name)

    def analyze_rb_posterior_dist(self, gal=None, blob_name=None):
        rb_mcmc_param_bestfit = gal.model.get_halo_rb()
        self.analyze_blob_posterior_dist(bestfit=rb_mcmc_param_bestfit, parname='rb', blob_name=blob_name)

    def get_uncertainty_ellipse(self, namex=None, namey=None, bins=50):
        r"""
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

        PA, stddev_x, stddev_y  = dpy_utils.fit_uncertainty_ellipse(chain_x, chain_y, bins=bins)
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



    def plot_results(self, gal, f_plot_param_corner=None, f_plot_bestfit=None,
                     f_plot_trace=None, overwrite=False):
        """Plot/replot the corner plot and bestfit for the MCMC fitting"""
        self.plot_corner(gal=gal, fileout=f_plot_param_corner, overwrite=overwrite)
        self.plot_bestfit(gal, fileout=f_plot_bestfit, overwrite=overwrite)
        self.plot_trace(fileout=f_plot_trace, overwrite=overwrite)


    def plot_corner(self, gal=None, fileout=None, overwrite=False):
        """Plot/replot the corner plot for the MCMC fitting"""
        plotting.plot_corner(self, gal=gal, fileout=fileout, blob_name=self.blob_name, overwrite=overwrite)

    def plot_trace(self, fileout=None, overwrite=False):
        """Plot/replot the trace for the MCMC fitting"""
        plotting.plot_trace(self, fileout=fileout, overwrite=overwrite)




def log_prob(theta, gal, fitter=None):
    """
    Evaluate the log probability of the given model
    """

    # Update the parameters
    gal.model.update_parameters(theta)

    # Evaluate prior prob of theta
    lprior = gal.model.get_log_prior()

    # First check to see if log prior is finite
    if not np.isfinite(lprior):
        if fitter.blob_name is not None:
            if isinstance(fitter.blob_name, str):
                return -np.inf, -np.inf
            else:
                return -np.inf, [-np.inf]*len(fitter.blob_name)
        else:
            return -np.inf
    else:
        # Update the model data
        gal.create_model_data()

        # Evaluate likelihood prob of theta
        llike = log_like(gal, fitter=fitter)

        if fitter.blob_name is not None:
            lprob = lprior + llike[0]
        else:
            lprob = lprior + llike

        if not np.isfinite(lprob):
            # Make sure the non-finite ln_prob is -Inf, for emcee handling
            lprob = -np.inf

        if fitter.blob_name is not None:
            if len(llike) == 2:
                return lprob, llike[1]
            else:
                return lprob, llike[1:]
        else:
            return lprob


def log_like(gal, fitter=None):

    # Temporary: testing:
    if fitter.oversampled_chisq is None:
        raise ValueError

    # if fitter.red_chisq:
    #     raise ValueError("red_chisq=True is currently *DISABLED* to test lnlike impact vs lnprior")


    llike = 0.0

    for obs_name in gal.observations:
        obs = gal.observations[obs_name]

        if obs.fit_options.fit:
            if obs.instrument.ndim == 3:
                msk = obs.data.mask
                dat = obs.data.data.unmasked_data[:].value[msk]
                mod = obs.model_data.data.unmasked_data[:].value[msk]
                err = obs.data.error.unmasked_data[:].value[msk]

                # Weights:
                wgt_data = 1.
                if hasattr(obs.data, 'weight'):
                    if obs.data.weight is not None:
                        wgt_data = obs.data.weight[msk]

                # Artificially mask zero errors which are masked
                #err[((err==0) & (msk==0))] = 99.
                chisq_arr_raw = (((dat - mod)/err)**2) * wgt_data + np.log( (2.*np.pi*err**2) / wgt_data )
                if fitter.oversampled_chisq:
                    invnu = 1. / obs.data.oversample_factor_chisq
                # elif fitter.red_chisq:
                #     if gal.model.nparams_free > np.sum(msk) :
                #         raise ValueError("More free parameters than data points!")
                #     invnu = 1./ (1.*(np.sum(msk) - gal.model.nparams_free))
                else:
                    invnu = 1.
                llike += -0.5*chisq_arr_raw.sum() * invnu * obs.weight


            elif (obs.instrument.ndim == 1) or (obs.instrument.ndim ==2):
                # VELOCITY:
                if obs.fit_options.fit_velocity:
                    msk = obs.data.mask
                    # If specific velocity mask: use that instead
                    if hasattr(obs.data, 'mask_velocity'):
                        if obs.data.mask_velocity is not None:
                            msk = obs.data.mask_velocity

                    vel_dat = obs.data.data['velocity'][msk]
                    vel_mod = obs.model_data.data['velocity'][msk]
                    vel_err = obs.data.error['velocity'][msk]

                # DISPERSION:
                if obs.fit_options.fit_dispersion:
                    msk = obs.data.mask
                    # If specific dispersion mask: use that instead
                    if hasattr(obs.data, 'mask_vel_disp'):
                        if obs.data.mask_vel_disp is not None:
                            msk = obs.data.mask_vel_disp

                    disp_dat = obs.data.data['dispersion'][msk]
                    disp_mod = obs.model_data.data['dispersion'][msk]
                    disp_err = obs.data.error['dispersion'][msk]

                # FLUX:
                if obs.fit_options.fit_flux:
                    msk = obs.data.mask
                    flux_dat = obs.data.data['flux'][msk]
                    flux_mod = obs.model_data.data['flux'][msk]
                    if obs.data.error['flux'] is not None:
                        flux_err = obs.data.error['flux'][msk]
                    else:
                        flux_err = 0.1 * obs.data.data['flux'][msk] # PLACEHOLDER

                # Data weights:
                wgt_data = 1.
                if hasattr(obs.data, 'weight'):
                    if obs.data.weight is not None:
                        wgt_data = obs.data.weight[msk]


                # Correct model for instrument dispersion if the data is instrument corrected:
                if obs.fit_options.fit_dispersion & ('inst_corr' in obs.data.data.keys()):
                    if obs.data.data['inst_corr']:
                        disp_mod = np.sqrt(disp_mod**2 -
                                           obs.instrument.lsf.dispersion.to(u.km/u.s).value**2)
                        disp_mod[~np.isfinite(disp_mod)] = 0   # Set the dispersion to zero when its below
                                                               # below the instrumental dispersion

                #####
                fac_mask = 0.
                chisq_arr_sum = 0.
                if obs.fit_options.fit_velocity:
                    # Data includes velocity
                    fac_mask += 1
                    # Includes velocity shift
                    chisq_arr_raw_vel = ((((vel_dat - vel_mod)/vel_err)**2) * wgt_data +
                                           np.log( (2.*np.pi*vel_err**2) / wgt_data ))
                    chisq_arr_sum += chisq_arr_raw_vel.sum()
                if obs.fit_options.fit_dispersion:
                    fac_mask += 1
                    chisq_arr_raw_disp = ((((disp_dat - disp_mod)/disp_err)**2) * wgt_data +
                                            np.log( (2.*np.pi*disp_err**2) / wgt_data))
                    chisq_arr_sum += chisq_arr_raw_disp.sum()

                if obs.fit_options.fit_flux:
                    fac_mask += 1
                    chisq_arr_raw_flux = ((((flux_dat - flux_mod)/flux_err)**2) * wgt_data +
                                            np.log( (2.*np.pi*flux_err**2) / wgt_data))
                    chisq_arr_sum += chisq_arr_raw_flux.sum()

                ####

                if fitter.oversampled_chisq:
                    invnu = 1. / obs.data.oversample_factor_chisq
                # elif fitter.red_chisq:
                #     if gal.model.nparams_free > fac_mask*np.sum(msk) :
                #         raise ValueError("More free parameters than data points!")
                #     invnu = 1./ (1.*(fac_mask*np.sum(msk) - gal.model.nparams_free))
                else:
                    invnu = 1.

                ####
                llike += -0.5*(chisq_arr_sum) * invnu * obs.weight

            elif obs.data.ndim == 0:

                msk = obs.data.mask
                data = obs.data.data
                mod = obs.model_data.data
                err = obs.data.error

                wgt_data = 1.
                if hasattr(obs.data, 'weight'):
                    if obs.data.weight is not None:
                        wgt_data = obs.data.weight


                chisq_arr = ((((data - mod)/err)**2) * wgt_data + np.log((2.*np.pi*err**2) / wgt_data))

                if fitter.oversampled_chisq:
                    invnu = 1. / obs.data.oversample_factor_chisq
                # elif fitter.red_chisq:
                #     if gal.model.nparams_free > np.sum(msk):
                #         raise ValueError("More free parameters than data points!")
                #
                #     invnu = 1. / (1. * (np.sum(msk) - gal.model.nparams_free))

                else:
                    invnu = 1.

                llike += -0.5*chisq_arr.sum() * invnu * obs.weight


            else:
                logger.warning("ndim={} not supported!".format(gal.data.ndim))
                raise ValueError


    ####
    # CALCULATE THE BLOB VALUE(S):
    if fitter.blob_name is not None:
        if isinstance(fitter.blob_name, str):
            # Single blob
            blob_arr = [fitter.blob_name]
        else:
            # Array of blobs
            blob_arr = fitter.blob_name[:]

        #
        blobvals = []
        for blobn in blob_arr:
            if blobn.lower() == 'fdm':
                blobv = gal.model.get_dm_frac_r_ap()
            elif blobn.lower() == 'mvirial':
                blobv = gal.model.get_mvirial()
            elif blobn.lower() == 'alpha':
                blobv = gal.model.get_haglo_alpha()
            elif blobn.lower() == 'rb':
                blobv = gal.model.get_halo_rb()
            #
            blobvals.append(blobv)

        # Preserve old behavior if there's only a single blob value: return float blob
        if isinstance(fitter.blob_name, str):
            blobvals = blobvals[0]


        return llike, blobvals

    else:
        return llike



def setup_oversampled_chisq(gal):
    # Setup for oversampled_chisq:
    for obs_name in gal.observations:
        obs = gal.observations[obs_name]

        if isinstance(obs.instrument.beam, GaussianBeam):
            try:
                PSF_FWHM = obs.instrument.beam.major.value
            except:
                PSF_FWHM = obs.instrument.beam.major
        elif isinstance(obs.instrument.beam, Moffat):
            try:
                PSF_FWHM = obs.instrument.beam.major_fwhm.value
            except:
                PSF_FWHM = obs.instrument.beam.major_fwhm
        elif isinstance(obs.instrument.beam, DoubleBeam):
            try:
                PSF_FWHM = np.max([obs.instrument.beam.beam1.major.value, obs.instrument.beam.beam2.major.value])
            except:
                PSF_FWHM = np.max([obs.instrument.beam.beam1.major, obs.instrument.beam.beam2.major])

        if obs.instrument.ndim == 1:
            rarrtmp = obs.data.rarr.copy()
            rarrtmp.sort()
            spacing_avg = np.abs(np.average(rarrtmp[1:]-rarrtmp[:-1]))
            obs.data.oversample_factor_chisq = PSF_FWHM /spacing_avg

        elif obs.instrument.ndim == 2:
            obs.data.oversample_factor_chisq = (PSF_FWHM / obs.instrument.pixscale.value)**2

        elif obs.instrument.ndim == 3:
            spec_step = obs.instrument.spec_step.to(u.km/u.s).value
            LSF_FWHM = obs.instrument.lsf.dispersion.to(u.km/u.s).value * (2.*np.sqrt(2.*np.log(2.)))
            obs.data.oversample_factor_chisq = (LSF_FWHM / spec_step) * (PSF_FWHM / obs.instrument.pixscale.value)**2

    return gal

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
    Return chain parameters that give peak of the posterior PDF *FOR LINKED PARAMETERS*, using KDE.
    """

    nparams = len(linked_inds)
    kern = gaussian_kde(flatchain[:,linked_inds].T)
    peakvals = fmin(lambda x: -kern(x), initval,disp=False)

    return peakvals


def find_multiD_pk_hist(flatchain, linked_inds, nPostBins=50):
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



def get_linked_posterior_indices(mcmcResults):
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
        if mcmcResults.linked_posterior_names.strip().lower() == 'all':
            linked_posterior_ind_arr = [range(len(mcmcResults.free_param_names))]
    except:
        pass
    if linked_posterior_ind_arr is None:
        free_cmp_param_arr = base.make_arr_cmp_params(mcmcResults)

        linked_posterior_ind_arr = []
        for k in six.moves.xrange(len(mcmcResults.linked_posterior_names)):
            # Loop over *sets* of linked posteriors:
            # This is an array of len-2 arrays/tuples with cmp, param names
            linked_post_inds = []
            for j in six.moves.xrange(len(mcmcResults.linked_posterior_names[k])):

                indp = get_param_index(mcmcResults, mcmcResults.linked_posterior_names[k][j],
                            free_cmp_param_arr=free_cmp_param_arr)
                linked_post_inds.append(indp)

            linked_posterior_ind_arr.append(linked_post_inds)

    return linked_posterior_ind_arr


def get_param_index(mcmcResults, param_name, free_cmp_param_arr=None):
    if free_cmp_param_arr is None:
        free_cmp_param_arr = base.make_arr_cmp_params(mcmcResults)

    cmp_param = param_name[0].strip().lower()+':'+param_name[1].strip().lower()

    try:
        whmatch = np.where(free_cmp_param_arr == cmp_param)[0][0]
    except:
        raise ValueError(cmp_param+' component+parameter not found in free parameters of mcmcResults')
    return whmatch


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

    if sampler.blobs is not None:
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

    if sampler.blobs is not None:
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


def reinitialize_emcee_sampler(sampler_dict, gal=None, fitter=None):
    """
    Re-setup emcee sampler, using existing chain / etc, so more steps can be run.
    """

    kwargs_dict = {'fitter': fitter}

    # This will break for updated version of emcee
    # works for emcee v2.2.1
    if emcee.__version__ == '2.2.1':

        sampler = emcee.EnsembleSampler(fitter.nWalkers, fitter.nParam,
                    log_prob, args=[gal], kwargs=kwargs_dict, a=fitter.scale_param_a,
                    threads=sampler_dictfitter.nCPU)

        sampler._chain = copy.deepcopy(sampler_dict['chain'])
        sampler._blobs = list(copy.deepcopy(sampler_dict['blobs']))
        sampler._lnprob = copy.deepcopy(sampler_dict['lnprobability'])
        sampler.iterations = sampler_dict['nIter']
        sampler.naccepted = np.array(sampler_dict['nIter']*copy.deepcopy(sampler_dict['acceptance_fraction']),
                            dtype=np.int64)
    ###
    elif int(emcee.__version__[0]) >= 3:
        # This is based off of HDF5 files, which automatically makes it easy to reload + resetup the sampler
        raise ValueError("emcee >=3 uses HDF5 files, so re-initialization not necessary!")

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
                        a=fitter.scale_param_a,
                        threads=sampler_dict['nCPU'])

        except:
            raise ValueError



    return sampler


def _reload_all_fitting_mcmc(filename_galmodel=None, filename_results=None):
    gal = galaxy.load_galaxy_object(filename=filename_galmodel)
    results = MCMCResults()
    results.reload_results(filename=filename_results)
    return gal, results


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
