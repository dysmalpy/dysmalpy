# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Main classes and functions for fitting DYSMALPY kinematic models
#   to the observed data using MCMC

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

## Standard library
import logging
from multiprocessing import cpu_count

# DYSMALPY code
from dysmalpy import plotting
from dysmalpy import galaxy

# Third party imports
import os
import numpy as np
from collections import OrderedDict
from astropy.extern import six
import astropy.units as u
import dill as _pickle
import copy
from dysmalpy.extern.cap_mpfit import mpfit
import emcee
import acor


import time, datetime

from scipy.stats import gaussian_kde
from scipy.optimize import fmin


__all__ = ['fit', 'MCMCResults']


# ACOR SETTINGS
acor_force_min = 49
# Force it to run for at least 50 steps, otherwise acor times might be completely wrong.

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')



def fit(gal, nWalkers=10,
           cpuFrac=None,
           nCPUs = 1,
           scale_param_a = 3.,
           nBurn = 2,
           nSteps = 10,
           minAF = 0.2,
           maxAF = 0.5,
           nEff = 10,
           oversample = 1,
           oversize = 1,
           red_chisq = False,
           profile1d_type='circ_ap_cube',
           fitdispersion = True,
           compute_dm = False, 
           model_key_re = ['disk+bulge','r_eff_disk'],  
           do_plotting = True,
           save_burn = False,
           save_model = True, 
           save_data = True, 
           out_dir = 'mcmc_fit_results/',
           linked_posterior_names= None,
           nPostBins = 50,
           continue_steps = False,
           input_sampler = None,
           f_plot_trace_burnin = None,
           f_plot_trace = None,
           f_model = None, 
           f_sampler = None,
           f_burn_sampler = None,
           f_plot_param_corner = None,
           f_plot_bestfit = None,
           f_mcmc_results = None,
           f_chain_ascii = None,
           f_vel_ascii = None, 
           f_log = None ):
    """
    Fit observed kinematics using DYSMALPY model set.

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
    # --------------------------------
    # Basic setup:
    
    # For compatibility with Python 2.7:
    mod_in = copy.deepcopy(gal.model)
    gal.model = mod_in
    
    #if nCPUs is None:
    if cpuFrac is not None:
        nCPUs = np.int(np.floor(cpu_count()*cpuFrac))

    nDim = gal.model.nparams_free
    #len(model.get_free_parameters_values())

    # Output filenames
    if (len(out_dir) > 0):
        if (out_dir[-1] != '/'): out_dir += '/'
    ensure_dir(out_dir)

    # Check to make sure previous sampler won't be overwritten: custom if continue_steps:
    if continue_steps and (f_sampler is None):  f_sampler = out_dir+'mcmc_sampler_continue.pickle'

    # If the output filenames aren't defined: use default output filenames
    if f_plot_trace_burnin is None:  f_plot_trace_burnin = out_dir+'mcmc_burnin_trace.pdf'
    if f_plot_trace is None:         f_plot_trace = out_dir+'mcmc_trace.pdf'
    if save_model and (f_model is None): f_model = out_dir+'galaxy_model.pickle'
    if f_sampler is None:            f_sampler = out_dir+'mcmc_sampler.pickle'
    if save_burn and (f_burn_sampler is None):  f_burn_sampler = out_dir+'mcmc_burn_sampler.pickle'
    if f_plot_param_corner is None:  f_plot_param_corner = out_dir+'mcmc_param_corner.pdf'
    if f_plot_bestfit is None:       f_plot_bestfit = out_dir+'mcmc_best_fit.pdf'
    if f_mcmc_results is None:       f_mcmc_results = out_dir+'mcmc_results.pickle'
    if f_chain_ascii is None:        f_chain_ascii = out_dir+'mcmc_chain_blobs.dat'
    if f_vel_ascii is None:          f_vel_ascii = out_dir+'galaxy_bestfit_vel_profile.dat'
    
    # Setup file redirect logging:
    if f_log is not None:
        loggerfile = logging.FileHandler(f_log)
        loggerfile.setLevel(logging.INFO)
        logger.addHandler(loggerfile)
    
    # ++++++++++++++++++++++++++++++

    if not continue_steps:
        # --------------------------------
        # Initialize walker starting positions
        initial_pos = initialize_walkers(gal.model, nWalkers=nWalkers)
    else:
        nBurn = 0
        if input_sampler is None:
            raise ValueError("Must set input_sampler if you will restart the sampler.")
        initial_pos = input_sampler['chain'][:,-1,:]
        

    # --------------------------------
    # Initialize emcee sampler
    kwargs_dict = {'oversample':oversample, 'oversize':oversize, 'fitdispersion':fitdispersion,
                    'compute_dm':compute_dm, 'model_key_re':model_key_re, 
                    'red_chisq': red_chisq, 'profile1d_type':profile1d_type}
    sampler = emcee.EnsembleSampler(nWalkers, nDim, log_prob,
                args=[gal], kwargs=kwargs_dict,
                a = scale_param_a, threads = nCPUs)

    # --------------------------------
    # Output some fitting info to logger:
    logger.info("*************************************")
    logger.info(" Fitting: {}".format(gal.name))
    if gal.data.filename_velocity is not None:
        logger.info("    velocity file: {}".format(gal.data.filename_velocity))
    if gal.data.filename_dispersion is not None:
        logger.info("    dispers. file: {}".format(gal.data.filename_dispersion))
    
    logger.info('\n  nCPUs: {}'.format(nCPUs))
    #logger.info('nSubpixels = %s' % (model.nSubpixels))

    ################################################################
    # --------------------------------
    # Run burn-in
    if nBurn > 0:
        logger.info('\nBurn-in:'+'\n'
                    'Start: {}\n'.format(datetime.datetime.now()))
        start = time.time()

        ####
        pos = initial_pos
        prob = None
        state = None
        dm_frac = None
        for k in six.moves.xrange(nBurn):
            #logger.info(" k={}, time.time={}".format( k, datetime.datetime.now() ) )
            # Temp for debugging:
            logger.info(" k={}, time.time={}, a_frac={}".format( k, datetime.datetime.now(), 
                        np.mean(sampler.acceptance_fraction)  ) )
            ###
            if compute_dm:
                pos, prob, state, dm_frac = sampler.run_mcmc(pos, 1, lnprob0=prob,
                                                    rstate0=state, blobs0=dm_frac)
            else:
                pos, prob, state = sampler.run_mcmc(pos, 1, lnprob0=prob,
                                                    rstate0=state)
        #####
        ## This would run in one go:
        #pos, prob, state = sampler.run_mcmc(initial_pos,fitEmis2D.mcmcOptions.nBurn)
        end = time.time()
        elapsed = end-start

        try:
            #acor_time = sampler.acor
            acor_time = [acor.acor(sampler.chain[:,:,jj])[0] for jj in range(sampler.dim)]
        except:
            acor_time = "Undefined, chain did not converge"


        #######################################################################################
        # Return Burn-in info
        # ****
        endtime = str(datetime.datetime.now())
        nthingsmsg = 'nCPU, nParam, nWalker, nBurn = {}, {}, {}, {}'.format(nCPUs,
            nDim, nWalkers, nBurn)
        scaleparammsg = 'Scale param a= {}'.format(scale_param_a)
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
        if (save_burn) & (f_burn_sampler is not None):
            sampler_burn = make_emcee_sampler_dict(sampler, nBurn=0)
            # Save stuff to file, for future use:
            dump_pickle(sampler_burn, filename=f_burn_sampler)


        # --------------------------------
        # Plot burn-in trace, if output file set
        if (do_plotting) & (f_plot_trace_burnin is not None):
            sampler_burn = make_emcee_sampler_dict(sampler, nBurn=0)
            mcmcResultsburn = MCMCResults(model=gal.model, sampler=sampler_burn)
            plotting.plot_trace(mcmcResultsburn, fileout=f_plot_trace_burnin)

        # Reset sampler after burn-in:
        sampler.reset()
        if compute_dm:
             sampler.clear_blobs()

    else:
        # --------------------------------
        # No burn-in: set initial position:
        pos = np.array(initial_pos)
        prob = None
        state = None
        dm_frac = None



    #######################################################################################
    # ****
    # --------------------------------
    # Run sampler: Get start time
    logger.info('\nEnsemble sampling:\n'
                'Start: {}\n'.format(datetime.datetime.now()))
    start = time.time()


    # --------------------------------
    # Run sampler: output info at each step
    for ii in six.moves.xrange(nSteps):
        pos_cur = pos.copy()    # copy just in case things are set strangely

        # --------------------------------
        # 1: only do one step at a time.
        if compute_dm:
            pos, prob, state, dm_frac = sampler.run_mcmc(pos_cur, 1, lnprob0=prob, 
                        rstate0=state, blobs0 = dm_frac)
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
            #acor_time = sampler.acor
            acor_time = [acor.acor(sampler.chain[:,:,jj])[0] for jj in range(sampler.dim)]
            logger.info( "{}: acor_time ={}".format(ii, np.array(acor_time) ) )
        except:
            logger.info(" {}: Chain too short for acor to run".format(ii) )
            acor_time = None
            
                     
        # --------------------------------
        # Case: test for convergence and truncate early:
        if((minAF is not None) & (maxAF is not None) & (nEff is not None)):
            if(minAF < np.mean(sampler.acceptance_fraction) < maxAF):
                if acor_time is not None:
                    if ( ii > np.max(acor_time) * nEff ):
                        if ii == acor_force_min:
                            logger.info(" Enforced min step limit: {}.".format(ii+1))
                        if ii >= acor_force_min:
                            logger.info(" Breaking chain at step {}.".format(ii+1))
                            break

    # --------------------------------
    # Check if it failed to converge before the max number of steps, if doing convergence testing
    finishedSteps= ii+1
    if (finishedSteps  == nSteps) & ((minAF is not None) & (maxAF is not None) & (nEff is not None)):
        logger.info(" Warning: chain did not converge after nSteps.")

    # --------------------------------
    # Finishing info for fitting:
    end = time.time()
    elapsed = end-start
    logger.info("Finished {} steps".format(finishedSteps)+"\n")
    try:
        #acor_time = sampler.acor
        acor_time = [acor.acor(sampler.chain[:,:,jj])[0] for jj in range(sampler.dim)]
    except:
        acor_time = "Undefined, chain did not converge"

    #######################################################################################
    # ***********
    # Consider overall acceptance fraction
    endtime = str(datetime.datetime.now())
    nthingsmsg = 'nCPU, nParam, nWalker, nSteps = {}, {}, {}, {}'.format(nCPUs,
        nDim, nWalkers, nSteps)
    scaleparammsg = 'Scale param a= {}'.format(scale_param_a)
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
    sampler_dict = make_emcee_sampler_dict(sampler, nBurn=0)

    if f_sampler is not None:
        # Save stuff to file, for future use:
        dump_pickle(sampler_dict, filename=f_sampler)


    if nCPUs > 1:
        sampler.pool.close()

    ##########################################
    ##########################################
    ##########################################


    # --------------------------------
    # Bundle the results up into a results class:
    mcmcResults = MCMCResults(model=gal.model, sampler=sampler_dict,
                f_plot_trace_burnin = f_plot_trace_burnin,
                f_plot_trace = f_plot_trace,
                f_sampler = f_sampler,
                f_plot_param_corner = f_plot_param_corner,
                f_plot_bestfit = f_plot_bestfit,
                f_mcmc_results = f_mcmc_results,
                f_chain_ascii = f_chain_ascii)

    # Get the best-fit values, uncertainty bounds from marginalized posteriors
    mcmcResults.analyze_posterior_dist(linked_posterior_names=linked_posterior_names,
                nPostBins=nPostBins)

            
    # Update theta to best-fit:
    gal.model.update_parameters(mcmcResults.bestfit_parameters)
    
    gal.create_model_data(oversample=oversample, oversize=oversize, 
                              line_center=gal.model.line_center)
    
    mcmcResults.bestfit_redchisq = -2.*log_like(gal, red_chisq=True, fitdispersion=fitdispersion, 
                    compute_dm=False, model_key_re=model_key_re)
    
    #
    if model_key_re is not None:
        comp = gal.model.components.__getitem__(model_key_re[0])
        param_i = comp.param_names.index(model_key_re[1])
        r_eff = comp.parameters[param_i] 
        mcmcResults.vrot_bestfit = gal.model.velocity_profile(1.38*r_eff, compute_dm=False)
    
    
    mcmcResults.vmax_bestfit = gal.model.get_vmax()
    
    
    if f_mcmc_results is not None:
        mcmcResults.save_results(filename=f_mcmc_results)
        
    if f_chain_ascii is not None:
        mcmcResults.save_chain_ascii(filename=f_chain_ascii)
        
    if f_model is not None:
        #mcmcResults.save_galaxy_model(galaxy=gal, filename=f_model)
        # Save model w/ updated theta equal to best-fit:
        gal.preserve_self(filename=f_model, save_data=save_data)
        
    # --------------------------------
    # Plot trace, if output file set
    if (do_plotting) & (f_plot_trace is not None) :
        plotting.plot_trace(mcmcResults, fileout=f_plot_trace)

    # --------------------------------
    # Plot results: corner plot, best-fit
    if (do_plotting) & (f_plot_param_corner is not None):
        plotting.plot_corner(mcmcResults, fileout=f_plot_param_corner)

    if (do_plotting) & (f_plot_bestfit is not None):
        plotting.plot_bestfit(mcmcResults, gal, fitdispersion=fitdispersion,
                              oversample=oversample, oversize=oversize, fileout=f_plot_bestfit)
                              
    # --------------------------------
    # Save velocity / other profiles to ascii file:
    if f_vel_ascii is not None:
        mcmcResults.save_bestfit_vel_ascii(gal, filename=f_vel_ascii, model_key_re=model_key_re)

    # Clean up logger:
    if f_log is not None:
        logger.removeHandler(loggerfile)
        
        

    return mcmcResults



def fit_mpfit(gal, fit_dispersion=True, profile1d_type='circ_ap_cube'):
    """
    A real simple function for fitting with least squares instead of MCMC.
    Right now being used for testing.
    """

    p_initial = gal.model.get_free_parameters_values()
    pkeys = gal.model.get_free_parameter_keys()
    nparam = len(p_initial)
    parinfo = [{'value':0, 'limited': [1, 1], 'limits': [0., 0.], 'fixed': 0} for i in
               range(nparam)]

    for cmp in pkeys:
        for param_name in pkeys[cmp]:

            if pkeys[cmp][param_name] != -99:

                bounds = gal.model.components[cmp].bounds[param_name]
                k = pkeys[cmp][param_name]
                parinfo[k]['limits'][0] = bounds[0]
                parinfo[k]['limits'][1] = bounds[1]
                parinfo[k]['value'] = p_initial[k]

    fa = {'gal':gal, 'fitdispersion':fit_dispersion, 'profile1d_type':profile1d_type}
    m = mpfit(mpfit_chisq, parinfo=parinfo, functkw=fa)

    return m


class MCMCResults(object):
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
            f_plot_trace_burnin = None,
            f_plot_trace = None,
            f_burn_sampler = None,
            f_sampler = None,
            f_plot_param_corner = None,
            f_plot_bestfit = None,
            f_mcmc_results = None,
            f_chain_ascii = None, 
            linked_posterior_names=None):

        self.sampler = sampler
        self.linked_posterior_names = linked_posterior_names

        #self.components = OrderedDict()

        self.bestfit_parameters = None
        self.bestfit_parameters_err = None
        self.bestfit_parameters_l68_err = None
        self.bestfit_parameters_u68_err = None

        self.bestfit_parameters_l68 = None
        self.bestfit_parameters_u68 = None
        
        self.bestfit_redchisq = None

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


        # Save what the filenames are for reference - eg, if they were defined by default.
        self.f_plot_trace_burnin = f_plot_trace_burnin
        self.f_plot_trace = f_plot_trace
        self.f_burn_sampler = f_burn_sampler
        self.f_sampler = f_sampler
        self.f_plot_param_corner = f_plot_param_corner
        self.f_plot_bestfit = f_plot_bestfit
        self.f_mcmc_results = f_mcmc_results
        self.f_chain_ascii = f_chain_ascii




    def set_model(self, model):
        self.param_names = model.param_names.copy()
        self._param_keys = model._param_keys.copy()
        self.nparams = model.nparams

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
        
        


    def analyze_posterior_dist(self, linked_posterior_names=None, nPostBins=40):
        """
        Default analysis of posterior distributions from MCMC fitting:
            look at marginalized posterior distributions, and extract the best-fit value (peak of KDE),
            and extract the +- 1 sigma uncertainty bounds (eg, the 16%/84% distribution of posteriors)

        Optional input:
                    linked_posterior_names:  indicate if best-fit of parameters
                                             should be measured in multi-D histogram space
                                format:  set of linked parameter sets, with each linked parameter set
                                         consisting of len-2 tuples/lists of the component+parameter names.
                                
                                
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
                            
                        eg:  look at halo: mvirial and disk+bulge: total_mass together
                            linked_posterior_names = [ [ ['halo', 'mvirial'], ['disk+bulge', 'total_mass'] ] ]
                         or linked_posterior_names = [ [ ('halo', 'mvirial'), ('disk+bulge', 'total_mass') ] ]

        """

        if self.sampler is None:
            raise ValueError("MCMC.sampler must be set to analyze the posterior distribution.")

        # Unpack MCMC samples: lower, upper 1, 2 sigma
        mcmc_limits = np.percentile(self.sampler['flatchain'], [15.865, 84.135], axis=0)

        ## location of peaks of *marginalized histograms* for each parameter
        mcmc_peak_hist = np.zeros(self.sampler['flatchain'].shape[1])
        for i in six.moves.xrange(self.sampler['flatchain'].shape[1]):
            yb, xb = np.histogram(self.sampler['flatchain'][:,i], bins=nPostBins)
            wh_pk = np.where(yb == yb.max())[0][0]
            mcmc_peak_hist[i] = np.average([xb[wh_pk], xb[wh_pk+1]])

        ## Use max prob as guess to get peakKDE value,
        ##      the peak of the marginalized posterior distributions (following M. George's speclens)
        mcmc_param_bestfit = getPeakKDE(self.sampler['flatchain'], mcmc_peak_hist)

        # --------------------------------------------
        if linked_posterior_names is not None:
            # Make sure the param of self is updated
            #   (for ref. when reloading saved mcmcResult objects)
            self.linked_posterior_names = linked_posterior_names
            linked_posterior_ind_arr = get_linked_posterior_indices(self,
                            linked_posterior_names=linked_posterior_names)

            bestfit_theta_linked = get_linked_posterior_peak_values(self.sampler['flatchain'],
                            guess=mcmc_param_bestfit, 
                            linked_posterior_ind_arr=linked_posterior_ind_arr,
                            nPostBins=nPostBins)

            for k in six.moves.xrange(len(linked_posterior_ind_arr)):
                for j in six.moves.xrange(len(linked_posterior_ind_arr[k])):
                    mcmc_param_bestfit[linked_posterior_ind_arr[k][j]] = bestfit_theta_linked[k][j]



        # --------------------------------------------
        # Uncertainty bounds are currently determined from marginalized posteriors
        #   (even if the best-fit is found from linked posterior).

        mcmc_stack = np.concatenate(([mcmc_param_bestfit], mcmc_limits), axis=0)
        # Order: best fit value, lower 1sig bound, upper 1sig bound

        mcmc_uncertainties_1sig = np.array(list(map(lambda v: (v[0]-v[1], v[2]-v[0]),
                            list(zip(*mcmc_stack)))))

        # --------------------------------------------
        # Save best-fit results in the MCMCResults instance
        self.bestfit_parameters = mcmc_param_bestfit

        # 1sig lower, upper uncertainty
        self.bestfit_parameters_err = mcmc_uncertainties_1sig

        # Bound limits (in case it's useful)
        self.bestfit_parameters_l68 = mcmc_limits[0]
        self.bestfit_parameters_u68 = mcmc_limits[1]

        # Separate 1sig l, u uncertainty, for utility:
        self.bestfit_parameters_l68_err = mcmc_param_bestfit - mcmc_limits[0]
        self.bestfit_parameters_u68_err = mcmc_limits[1] - mcmc_param_bestfit
        
        self.bestfit_redchisq = None
        
        
        
        
    def save_results(self, filename=None):
        if filename is not None:
            dump_pickle(self, filename=filename) # Save mcmcResults class
            
    def save_chain_ascii(self, filename=None):
        if filename is not None:
            try:
                blobs = self.sampler['blobs']
                blobset = True
            except:
                blobset = False
            
            if ('flatblobs' not in self.sampler.keys()) & (blobset):
                self.sampler['flatblobs'] = np.hstack(np.stack(self.sampler['blobs'], axis=1))
            
            with open(filename, 'w') as f:
                namestr = '#'
                namestr += '  '.join(map(str, self.chain_param_names))
                if blobset:
                    # Currently assuming blob only returns DM fraction
                    namestr += '  f_DM'
                f.write(namestr+'\n')
                
                # flatchain shape: (flat)step, params
                for i in six.moves.xrange(self.sampler['flatchain'].shape[0]):
                    datstr = '  '.join(map(str, self.sampler['flatchain'][i,:]))
                    if blobset:
                        datstr += '  {}'.format(self.sampler['flatblobs'][i])
                    f.write(datstr+'\n')
                    
    def save_bestfit_vel_ascii(gal, filename=None, model_key_re = ['disk+bulge','r_eff_disk']):
        if filename is not None:
            try:
                # RE needs to be in kpc
                comp = gal.model.components.__getitem__(model_key_re[0])
                param_i = comp.param_names.index(model_key_re[1])
                r_eff = comp.parameters[param_i]
            except:
                r_eff = 10./3.
            rmax = np.max([3.*r_eff, 10.])
            stepsize = 0.1 # stepsize 0.1 kpc
            r = np.arange(0., rmax+stepsize, stepsize)
            
            gal.model.write_vrot_vcirc_file(r=r, filename=filename)
            
            
    def reload_mcmc_results(self, filename=None):
        """Reload MCMC results saved earlier: the whole object"""
        if filename is None:
            filename = self.f_mcmc_results
        mcmcSaved = load_pickle(filename)
        for key in self.__dict__.keys():
            try:
                self.__dict__[key] = mcmcSaved.__dict__[key]
            except:
                pass

    def reload_sampler(self, filename=None):
        """Reload the MCMC sampler saved earlier"""
        if filename is None:
            filename = self.f_sampler
        self.sampler = load_pickle(filename)

    def plot_results(self, gal, fitdispersion=True, oversample=1, oversize=1,
                     f_plot_param_corner=None, f_plot_bestfit=None, f_plot_trace=None):
        """Plot/replot the corner plot and bestfit for the MCMC fitting"""
        self.plot_corner(fileout=f_plot_param_corner)
        self.plot_bestfit(gal, fitdispersion=fitdispersion,
                oversample=oversample, oversize=oversize, fileout=f_plot_bestfit)
        self.plot_trace(fileout=f_plot_trace)
    def plot_corner(self, fileout=None):
        """Plot/replot the corner plot for the MCMC fitting"""
        if fileout is None:
            fileout = self.f_plot_param_corner
        plotting.plot_corner(self, fileout=fileout)
    def plot_bestfit(self, gal, fitdispersion=True, oversample=1, oversize=1, fileout=None):
        """Plot/replot the bestfit for the MCMC fitting"""
        if fileout is None:
            fileout = self.f_plot_bestfit
        plotting.plot_bestfit(self, gal, fitdispersion=fitdispersion,
                              oversample=oversample, oversize=oversize, fileout=fileout)
    def plot_trace(self, fileout=None):
        """Plot/replot the trace for the MCMC fitting"""
        if fileout is None:
            fileout = self.f_plot_trace
        plotting.plot_trace(self, fileout=fileout)


def log_prob(theta, gal,
             oversample=1,
             oversize=1,
             red_chisq=False, 
             fitdispersion=True,
             compute_dm=False,
             profile1d_type='circ_ap_cube',
             model_key_re=['disk+bulge','r_eff_disk']):
    """
    Evaluate the log probability of the given model
    """

    # Update the parameters
    gal.model.update_parameters(theta)

    # Evaluate prior prob of theta
    lprior = gal.model.get_log_prior()

    # First check to see if log prior is finite
    if not np.isfinite(lprior):
        if compute_dm:
            return -np.inf, -np.inf
        else:
            return -np.inf
    else:
        # Update the model data
        gal.create_model_data(oversample=oversample, oversize=oversize,
                              line_center=gal.model.line_center, profile1d_type=profile1d_type)
                              
        # Evaluate likelihood prob of theta
        llike = log_like(gal, red_chisq=red_chisq, fitdispersion=fitdispersion, 
                    compute_dm=compute_dm, model_key_re=model_key_re)

        if compute_dm:
            lprob = lprior + llike[0]
        else:
            lprob = lprior + llike

        if not np.isfinite(lprob):
            # Make sure the non-finite ln_prob is -Inf,
            #    as this can be escaped in the next step
            lprob = -np.inf
            
        if compute_dm:
            return lprob, llike[1]
        else:
            return lprob


def log_like(gal, red_chisq=False, fitdispersion=True, 
                compute_dm=False, model_key_re=['disk+bulge','r_eff_disk']):

    if gal.data.ndim == 3:
        # Will have problem with vel shift: data, model won't match...

        msk = gal.data.mask
        dat = gal.data.data.unmasked_data[:].value[msk]
        mod = gal.model_data.data.unmasked_data[:].value[msk]
        err = gal.data.error.unmasked_data[:].value[msk]

        # Artificially mask zero errors which are masked
        #err[((err==0) & (msk==0))] = 99.
        chisq_arr_raw = ((dat - mod)/err)**2 + np.log(2.*np.pi*err**2)
        if red_chisq:
            if gal.model.nparams_free > np.sum(msk) :
                raise ValueError("More free parameters than data points!")
            invnu = 1./ (1.*(np.sum(msk) - gal.model.nparams_free))
        else:
            invnu = 1.
        llike = -0.5*chisq_arr_raw.sum() * invnu
        


    elif (gal.data.ndim == 1) or (gal.data.ndim ==2):

        msk = gal.data.mask
        vel_dat = gal.data.data['velocity'][msk]
        vel_mod = gal.model_data.data['velocity'][msk]
        vel_err = gal.data.error['velocity'][msk]

        disp_dat = gal.data.data['dispersion'][msk]
        disp_mod = gal.model_data.data['dispersion'][msk]
        disp_err = gal.data.error['dispersion'][msk]

        # Correct model for instrument dispersion if the data is instrument corrected:
        if 'inst_corr' in gal.data.data.keys():
            if gal.data.data['inst_corr']:
                disp_mod = np.sqrt(disp_mod**2 -
                                   gal.instrument.lsf.dispersion.to(u.km/u.s).value**2)
                disp_mod[~np.isfinite(disp_mod)] = 0   # Set the dispersion to zero when its below
                                                       # below the instrumental dispersion
                                                       
        # Includes velocity shift
        chisq_arr_raw_vel = (((vel_dat - vel_mod)/vel_err)**2 +
                               np.log(2.*np.pi*vel_err**2))
        if fitdispersion:
            if red_chisq:
                if gal.model.nparams_free > 2.*np.sum(msk) :
                    raise ValueError("More free parameters than data points!")
                invnu = 1./ (1.*(2.*np.sum(msk) - gal.model.nparams_free))
            else:
                invnu = 1.
            chisq_arr_raw_disp = (((disp_dat - disp_mod)/disp_err)**2 +
                                    np.log(2.*np.pi*disp_err**2))
            llike = -0.5*( chisq_arr_raw_vel.sum() + chisq_arr_raw_disp.sum()) * invnu
        else:
            if red_chisq:
                if gal.model.nparams_free > np.sum(msk) :
                    raise ValueError("More free parameters than data points!")
                invnu = 1./ (1.*(np.sum(msk) - gal.model.nparams_free))
            else:
                invnu = 1.
            llike = -0.5*chisq_arr_raw_vel.sum() * invnu
    else:
        logger.warning("ndim={} not supported!".format(gal.data.ndim))
        raise ValueError
        
    if compute_dm:
        dm_frac = gal.model.get_dm_frac_effrad(model_key_re=model_key_re)
        return llike, dm_frac
    else:
        return llike


def mpfit_chisq(theta, fjac=None, gal=None, fitdispersion=True, profile1d_type='circ_ap_cube'):

    gal.model.update_parameters(theta)
    gal.create_model_data(profile1d_type=profile1d_type)

    if gal.data.ndim == 3:
        dat = gal.data.data.unmasked_data[:].value
        mod = gal.model_data.data.unmasked_data[:].value
        err = gal.data.error.unmasked_data[:].value
        msk = gal.data.mask
        # Artificially mask zero errors which are masked
        err[((err == 0) & (msk == 0))] = 99.
        chisq_arr_raw = msk * (
        ((dat - mod) / err))
        chisq_arr_raw = chisq_arr_raw.flatten()

    elif (gal.data.ndim == 1) or (gal.data.ndim == 2):

        msk = gal.data.mask
        vel_dat = gal.data.data['velocity'][msk]
        vel_mod = gal.model_data.data['velocity'][msk]
        vel_err = gal.data.error['velocity'][msk]

        disp_dat = gal.data.data['dispersion'][msk]
        disp_mod = gal.model_data.data['dispersion'][msk]
        disp_err = gal.data.error['dispersion'][msk]

        # Correct model for instrument dispersion if the data is instrument corrected:
        if 'inst_corr' in gal.data.data.keys():
            if gal.data.data['inst_corr']:
                disp_mod = np.sqrt(
                    disp_mod ** 2 - gal.instrument.lsf.dispersion.to(
                        u.km / u.s).value ** 2)

        chisq_arr_raw_vel = ((vel_dat - vel_mod) / vel_err)
        if fitdispersion:
            chisq_arr_raw_disp = (((disp_dat - disp_mod) / disp_err))
            chisq_arr_raw = np.hstack([chisq_arr_raw_vel.flatten(),
                                       chisq_arr_raw_disp.flatten()])
        else:
            chisq_arr_raw = chisq_arr_raw_vel.flatten()
    else:
        logger.warning("ndim={} not supported!".format(gal.data.ndim))
        raise ValueError

    status = 0

    return [status, chisq_arr_raw]


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
                param_rand = comp.prior[paramn].sample_prior(comp.__getattribute__(paramn), N=nWalkers)
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

    """
    mcmc_options = dict(nWalkers=10,
       cpuFrac=None,
       nCPUs = 1,
       scale_param_a = 3.,
       nBurn = 2,
       nSteps = 10,
       minAF = 0.2,
       maxAF = 0.5,
       nEff = 10,
       do_plotting = True,
       out_dir = 'mcmc_fit_results/',
       f_plot_trace_burnin = None,
       f_plot_trace = None,
       f_sampler = None,
       f_burn_sampler = None,
       f_plot_param_corner = None,
       f_plot_bestfit = None,
       f_mcmc_results = None)


    return mcmc_options



def getPeakKDE(flatchain, guess):
    """
    Return chain pars that give peak of posterior PDF, using KDE.
    From speclens: https://github.com/mrgeorge/speclens/blob/master/speclens/fit.py
    """
    if(len(flatchain.shape)==1):
        nPars=1
        kern=gaussian_kde(flatchain)
        peakKDE=fmin(lambda x: -kern(x), guess,disp=False)
        return peakKDE
    else:
        nPars=flatchain.shape[1]
        peakKDE=np.zeros(nPars)
        for ii in range(nPars):
            kern=gaussian_kde(flatchain[:,ii])
            peakKDE[ii]=fmin(lambda x: -kern(x), guess[ii],disp=False)
        return peakKDE
        
def getPeakKDEmultiD(flatchain, inds, guess):
    """
    Return chain pars that give peak of posterior PDF *FOR LINKED PARAMETERS, using KDE.
    From speclens: https://github.com/mrgeorge/speclens/blob/master/speclens/fit.py
    """
    nPars = len(inds)
    
    kern = gaussian_kde(flatchain[:,inds].T)
    peakKDE = fmin(lambda x: -kern(x), guess, disp=False)
    
    return peakKDE
    

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
    # if nPostBins % 2 == 0:
    #     nPostBinsOdd = nPostBins+1
    # else:
    #     nPostBinsOdd = nPostBins
    # 
    # bestfit_theta_linked = np.array([])
    # 
    # for k in six.moves.xrange(len(linked_posterior_ind_arr)):
    #     H, edges = np.histogramdd(flatchain[:,linked_posterior_ind_arr[k]], bins=nPostBinsOdd)
    #     wh_H_peak = np.column_stack(np.where(H == H.max()))[0]
    # 
    #     bestfit_thetas = np.array([])
    #     for j in six.moves.xrange(len(linked_posterior_ind_arr[k])):
    #         bestfit_thetas = np.append(bestfit_thetas, np.average([edges[j][wh_H_peak[j]],
    #                                                         edges[j][wh_H_peak[j]+1]]))
    #     if len(bestfit_theta_linked) >= 1:
    #         bestfit_theta_linked = np.stack(bestfit_theta_linked, np.array([bestfit_thetas]) )
    #     else:
    #         bestfit_theta_linked = np.array([bestfit_thetas])

    # Use KDE to get bestfit linked:
    bestfit_theta_linked = np.array([])

    for k in six.moves.xrange(len(linked_posterior_ind_arr)):
        bestfit_thetas = getPeakKDEmultiD(flatchain, linked_posterior_ind_arr[k], 
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
                cmp_param = linked_posterior_names[k][j][0].strip().lower()+':'+\
                            linked_posterior_names[k][j][1].strip().lower()
                try:
                    whmatch = np.where(free_cmp_param_arr == cmp_param)[0][0]
                    linked_post_inds.append(whmatch)
                except:
                    raise ValueError(cmp_param+' component+parameter not found in free parameters of mcmcResults')

            # # SORT THIS TO GET ACENDING ORDER
            # linked_post_inds = sorted(linked_post_inds)
    
            linked_posterior_ind_arr.append(linked_post_inds)
        
        
    return linked_posterior_ind_arr

def make_arr_cmp_params(mcmcResults):
    arr = np.array([])
    for cmp in mcmcResults.free_param_names.keys():
        for i in six.moves.xrange(len(mcmcResults.free_param_names[cmp])):
            param = mcmcResults.free_param_names[cmp][i]
            arr = np.append( arr, cmp.strip().lower()+':'+param.strip().lower() )

    return arr

def make_emcee_sampler_dict(sampler, nBurn=0):
    """
    Save chain + key results from emcee sampler instance to a dict,
    as the emcee samplers aren't pickleable.
    """
    # Cut first nBurn steps, to avoid the edge cases that are rarely explored.
    samples = sampler.chain[:, nBurn:, :].reshape((-1, sampler.dim))
    # Walkers, iterations
    probs = sampler.lnprobability[:, nBurn:].reshape((-1))
    
    #
    try:
        #acor_time = sampler.acor
        acor_time = [acor.acor(sampler.chain[:,nBurn:,jj])[0] for jj in range(sampler.dim)]
    except:
        acor_time = None
        
        
    # Make a dictionary:
    df = { 'chain': sampler.chain[:, nBurn:, :],
           'lnprobability': sampler.lnprobability[:, nBurn:],
           'flatchain': samples,
           'flatlnprobability': probs,
           'nIter': sampler.iterations,
           'nParam': sampler.dim,
           'nCPU': sampler.threads,
           'nWalkers': len(sampler.chain), 
           'acceptance_fraction': sampler.acceptance_fraction,
           'acor_time': acor_time }

    if len(sampler.blobs) > 0:
        df['blobs'] = sampler.blobs
        df['flatblobs'] = np.hstack(np.stack(sampler.blobs, axis=1))

    return df

def ensure_dir(dir):
    """ Short function to ensure dir is a directory; if not, make the directory."""
    if not os.path.exists(dir):
        logger.info( "Making path="+dir)
        os.makedirs(dir)
    return None

def load_pickle(filename):
    """ Small wrapper function to load a pickled structure """
    data = _pickle.load(open(filename, "rb"))
    return data

def dump_pickle(data, filename=None):
    """ Small wrapper function to pickle a structure """
    _pickle.dump(data, open(filename, "wb") )
    return None



def reload_all_fitting(filename_galmodel=None, filename_mcmc_results=None):
    #, filename_sampler=None):
    gal = galaxy.load_galaxy_object(filename=filename_galmodel)
    
    mcmcResults = MCMCResults() #model=gal.model
    
    mcmcResults.reload_mcmc_results(filename=filename_mcmc_results)
    #mcmcResults.reload_sampler(filename=filename_sampler)
    
    return gal, mcmcResults




