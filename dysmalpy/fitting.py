# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Main classes and functions for fitting DYSMALPY kinematic models
#   to the observed data using MCMC

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# DYSMALPY code
from dysmalpy import plotting


# Third party imports
import os
import numpy as np
from collections import OrderedDict
from astropy.extern import six
import dill as _pickle

import emcee
import psutil
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
           fitdispersion = True,
           do_plotting = True,
           save_burn = False, 
           out_dir = 'mcmc_fit_results/',
           linked_posterior_names= None, 
           nPostBins = 50, 
           continue_steps = False, 
           input_sampler = None, 
           f_plot_trace_burnin = None,
           f_plot_trace = None,
           f_sampler = None,
           f_burn_sampler = None, 
           f_plot_param_corner = None,
           f_plot_bestfit = None,
           f_mcmc_results = None ):
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
    #if nCPUs is None:
    if cpuFrac is not None:
        nCPUs = np.int(np.floor(psutil.cpu_count()*cpuFrac))

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
    if f_sampler is None:            f_sampler = out_dir+'mcmc_sampler.pickle'
    if save_burn and (f_burn_sampler is None):  f_burn_sampler = out_dir+'mcmc_burn_sampler.pickle'
    if f_plot_param_corner is None:  f_plot_param_corner = out_dir+'mcmc_param_corner.pdf'
    if f_plot_bestfit is None:       f_plot_bestfit = out_dir+'mcmc_best_fit.pdf'
    if f_mcmc_results is None:       f_mcmc_results = out_dir+'mcmc_results.pickle'
    
    
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
    kwargs_dict = {'oversample':oversample, 'fitdispersion':fitdispersion}
    sampler = emcee.EnsembleSampler(nWalkers, nDim, log_prob,
                args=[gal], kwargs=kwargs_dict,
                a = scale_param_a, threads = nCPUs)

    # --------------------------------
    # Output some fitting info to logger:
    logger.info(' nCPUs: '+str(nCPUs))
    #logger.info('nSubpixels = %s' % (model.nSubpixels))

    ################################################################
    # --------------------------------
    # Run burn-in
    if nBurn > 0:
        logger.info('\nBurn-in:'+'\n'
                    'Start: '+str(datetime.datetime.now()))
        start = time.time()

        ####
        pos = initial_pos
        prob = None
        state = None
        for k in six.moves.xrange(nBurn):
            logger.info(" k, time.time="+str(k)+" "+str(datetime.datetime.now()) )
            pos, prob, state = sampler.run_mcmc(pos, 1, lnprob0=prob,
                                                rstate0=state)
        #####
        ## This would run in one go:
        #pos, prob, state = sampler.run_mcmc(initial_pos,fitEmis2D.mcmcOptions.nBurn)
        end = time.time()
        elapsed = end-start

        try:
            acor_time = [acor.acor(sampler.chain[:,:,jj])[0] for jj in range(sampler.dim)]
        except:
            acor_time = "Undefined, chain did not converge"


        #######################################################################################
        # Return Burn-in info
        # ****
        endtime = str(datetime.datetime.now())
        nthingsmsg = 'nCPU, nParam, nWalker, nBurn = %s, %s, %s, %s'%(nCPUs,
            nDim, nWalkers, nBurn)
        scaleparammsg = 'Scale param a= %s' % scale_param_a
        timemsg = 'Time= %3.2f (sec), %3.0f:%3.2f (m:s)' % (elapsed, np.floor(elapsed/60.),
                (elapsed/60.-np.floor(elapsed/60.))*60.)
        macfracmsg = "Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))
        acortimemsg = "Autocorr est: "+str(acor_time)
        logger.info('\nEnd: '+endtime+'\n'
                    '******************\n'
                    ''+nthingsmsg+'\n'
                    ''+scaleparammsg+'\n'
                    ''+timemsg+''
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

    else:
        # --------------------------------
        # No burn-in: set initial position:
        pos = np.array(initial_pos)
        prob = None
        state = None



    #######################################################################################
    # ****
    # --------------------------------
    # Run sampler: Get start time
    logger.info('\nEnsemble sampling:\n'
                'Start: '+str(datetime.datetime.now()))
    start = time.time()

    # --------------------------------
    # Case: test for convergence and truncate early:
    if((minAF is not None) & (maxAF is not None) & (nEff is not None)):
        for ii in six.moves.xrange(nSteps):
            pos_cur = pos.copy()    # copy just in case things are set strangely

            # --------------------------------
            # 1: only do one step at a time.
            pos, prob, state = sampler.run_mcmc(pos_cur, 1, lnprob0=prob, rstate0=state)
            # --------------------------------

            # --------------------------------
            # Test for convergence
            nowtime = str(datetime.datetime.now())
            stepinfomsg = "ii=%s, a_frac=%s" % (ii, np.mean(sampler.acceptance_fraction))
            logger.info( " time.time()="+nowtime+'\n'
                         ''+stepinfomsg+'')
            if(minAF < np.mean(sampler.acceptance_fraction) < maxAF):
                try:
                    acor_time = [acor.acor(sampler.chain[:,:,jj])[0] for jj in range(sampler.dim)]

                    logger.info( str(ii)+": acor_time =" + str(np.array(acor_time)))
                    if ( ii > np.max(acor_time) * nEff ):
                        if ii == acor_force_min:
                            logger.info(" Enforced min step limit: {}.".format(ii+1))
                        if ii >= 49:
                            logger.info(" Breaking chain at step {}.".format(ii+1))
                            break
                except RuntimeError:
                    # acor raises exception if the chain isn't long
                    # enough to compute the acor time. However, could also be other
                    #   runtime errors..... need to be careful!
                    logger.info( " "+str(ii)+": Chain is too short for acor to run")
                    pass

        # --------------------------------
        # Check if it failed to converge before the max number of steps
        finishedSteps= ii+1
        if (finishedSteps  == nSteps):
            logger.info(" Warning: chain did not converge after nSteps.")


    # --------------------------------
    # Case: don't do convergence testing with early truncation: just run max number of steps
    else:
        sampler.run_mcmc(pos, nSteps)
        finishedSteps = nSteps

    # --------------------------------
    # Finishing info for fitting:
    end = time.time()
    elapsed = end-start
    logger.info("Finished {} steps".format(finishedSteps)+"\n")
    try:
        acor_time = [acor.acor(sampler.chain[:,:,jj])[0] for jj in range(sampler.dim)]
    except:
        acor_time = "Undefined, chain did not converge"

    #######################################################################################
    # ***********
    # Consider overall acceptance fraction
    endtime = str(datetime.datetime.now())
    nthingsmsg = 'nCPU, nParam, nWalker, nSteps = %s, %s, %s, %s'%(nCPUs,
        nDim, nWalkers, nSteps)
    scaleparammsg = 'Scale param a= %s' % scale_param_a
    timemsg = 'Time= %3.2f (sec), %3.0f:%3.2f (m:s)' % (elapsed, np.floor(elapsed/60.),
            (elapsed/60.-np.floor(elapsed/60.))*60.)
    macfracmsg = "Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))
    acortimemsg = "Autocorr est: "+str(acor_time)
    logger.info('\nEnd: '+endtime+'\n'
                '******************\n'
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
                f_mcmc_results = f_mcmc_results)
    
    # Get the best-fit values, uncertainty bounds from marginalized posteriors
    mcmcResults.analyze_posterior_dist(linked_posterior_names=linked_posterior_names, 
                nPostBins=nPostBins)   
    
    if f_mcmc_results is not None:
        dump_pickle(mcmcResults, filename=f_mcmc_results) # Save mcmcResults class
    #

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
                    oversample=oversample, fileout=f_plot_bestfit)

    return mcmcResults





class MCMCResults(object):
    """
    Class to hold results of MCMC fitting to DYSMALPY models
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
            linked_posterior_names=None,
            reload_mcmc_results_file=None):

        self.sampler = sampler
        self.linked_posterior_names = linked_posterior_names

        #self.components = OrderedDict()

        self.bestfit_parameters = None
        self.bestfit_parameters_err = None
        self.bestfit_parameters_l68_err = None
        self.bestfit_parameters_u68_err = None

        self.bestfit_parameters_l68 = None
        self.bestfit_parameters_u68 = None

        if model is not None:
            self.set_model(model)
        else:
            self.param_names = OrderedDict()
            self._param_keys = OrderedDict()
            self.nparams = None 
            self.free_param_names = OrderedDict()
            self._free_param_keys = OrderedDict()
            self.nparams_free = None


        # Save what the filenames are for reference - eg, if they were defined by default.
        self.f_plot_trace_burnin = f_plot_trace_burnin
        self.f_plot_trace = f_plot_trace
        self.f_burn_sampler = f_burn_sampler
        self.f_sampler = f_sampler
        self.f_plot_param_corner = f_plot_param_corner
        self.f_plot_bestfit = f_plot_bestfit
        self.f_mcmc_results = f_mcmc_results
        
        
        

    def set_model(self, model):
        self.param_names = model.param_names.copy()
        self._param_keys = model._param_keys.copy()
        self.nparams = model.nparams

        self.free_param_names = OrderedDict()
        self._free_param_keys = OrderedDict()
        
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


    def analyze_posterior_dist(self, linked_posterior_names=None, nPostBins=50):
        """
        Default analysis of posterior distributions from MCMC fitting:
            look at marginalized posterior distributions, and extract the best-fit value (peak of KDE),
            and extract the +- 1 sigma uncertainty bounds (eg, the 16%/84% distribution of posteriors)
            
        Optional input:
                    linked_posterior_names:  indicate if best-fit of parameters 
                                             should be measured in multi-D histogram space
                                format:  set of linked parameter sets, with each linked parameter set
                                         consisting of len-2 tuples/lists of the component+parameter names.
                                eg:  look at halo: mvirial and disk+bulge: total_mass together
                                linked_posterior_names = [ [ [cmp1, par1], [cmp2, par2] ] ]
                        eg:
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

    def plot_results(self, gal, fitdispersion=True, oversample=1,
                f_plot_param_corner=None, f_plot_bestfit=None):
        """Plot/replot the corner plot and bestfit for the MCMC fitting"""
        self.plot_corner(fileout=f_plot_param_corner)
        self.plot_bestfit(gal, fitdispersion=fitdispersion,
                oversample=oversample, fileout=f_plot_bestfit)
    def plot_corner(self, fileout=None):
        """Plot/replot the corner plot for the MCMC fitting"""
        if fileout is None:
            fileout = self.f_plot_param_corner
        plotting.plot_corner(self, fileout=fileout)
    def plot_bestfit(self, gal, fitdispersion=True, oversample=1, fileout=None):
        """Plot/replot the bestfit for the MCMC fitting"""
        if fileout is None:
            fileout = self.f_plot_bestfit
        plotting.plot_bestfit(self, gal, fitdispersion=fitdispersion,
                    oversample=oversample, fileout=fileout)


def log_prob(theta, gal,
            oversample=1,
            fitdispersion=True):
    """
    Evaluate the log probability of the given model
    """

    # Update the parameters
    gal.model.update_parameters(theta)

    # Evaluate prior prob of theta
    lprior = gal.model.get_log_prior()

    # First check to see if log prior is finite
    if not np.isfinite(lprior):
        return -np.inf
    else:
        # Update the model data
        gal.create_model_data(oversample=oversample,
                              line_center=gal.model.line_center)

        # Evaluate likelihood prob of theta
        llike = log_like(gal, fitdispersion=fitdispersion)
        lprob = lprior + llike

        if not np.isfinite(lprob):
            # Make sure the non-finite ln_prob is -Inf,
            #    as this can be escaped in the next step
            lprob = -np.inf
        return lprob


def log_like(gal, fitdispersion=True):

    if gal.data.ndim == 3:
        dat = gal.data.data.unmasked_data
        mod = gal.model_data.data.unmasked_data
        err = gal.data.error.unmasked_data
        msk = gal.data.mask
        chisq_arr_raw = msk * (((dat - mod)/err)**2 + np.log(2.*np.pi*err**2))
        llike = -0.5*chisq_arr_raw.sum()

    elif (gal.data.ndim == 1) or (gal.data.ndim ==2):
        vel_dat = gal.data.data['velocity']
        vel_mod = gal.model_data.data['velocity']
        vel_err = gal.data.error['velocity']

        disp_dat = gal.data.data['dispersion']
        disp_mod = gal.model_data.data['dispersion']
        disp_err = gal.data.error['dispersion']

        msk = gal.data.mask

        chisq_arr_raw_vel = msk * (((vel_dat - vel_mod)/vel_err)**2 +
                                   np.log(2.*np.pi*vel_err**2))
        if fitdispersion:
            chisq_arr_raw_disp = msk * (((disp_dat - disp_mod)/disp_err)**2 +
                                          np.log(2.*np.pi*disp_err**2))
            llike = -0.5*( chisq_arr_raw_vel.sum() + chisq_arr_raw_disp.sum())
        else:
            llike = -0.5*chisq_arr_raw_vel.sum()
    else:
        logger.warning("ndim="+str(gal.data.ndim)+" not supported!")
        raise ValueError


    return llike

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


def get_linked_posterior_peak_values(flatchain, 
                linked_posterior_ind_arr=None,
                nPostBins=50):
    """
    Get linked posterior best-fit values using a multi-D histogram for the 
    given linked parameter indices. 
    """
    if nPostBins % 2 == 0:
        nPostBinsOdd = nPostBins+1
    else:
        nPostBinsOdd = nPostBins
    
    bestfit_theta_linked = np.array([])
    
    for k in six.moves.xrange(len(linked_posterior_ind_arr)):
        H, edges = np.histogramdd(flatchain[:,linked_posterior_ind_arr[k]], bins=nPostBinsOdd)
        wh_H_peak = np.column_stack(np.where(H == H.max()))[0]
        
        bestfit_thetas = np.array([])
        for j in six.moves.xrange(len(linked_posterior_ind_arr[k])):
            bestfit_thetas = np.append(bestfit_thetas, np.average([edges[j][wh_H_peak[j]],
                                                            edges[j][wh_H_peak[j]+1]]))
        bestfit_theta_linked = np.append(bestfit_theta_linked, [bestfit_thetas])
        
    return bestfit_theta_linked
    
def get_linked_posterior_indices(mcmcResults, linked_posterior_names=None):
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

    # Make a dictionary:
    df = { 'chain': sampler.chain[:, nBurn:, :],
           'lnprobability': sampler.lnprobability[:, nBurn:],
           'flatchain': samples,
           'flatlnprobability': probs,
           'nIter': sampler.iterations,
           'nParam': sampler.dim,
           'nCPU': sampler.threads,
           'nWalkers': len(sampler.chain) }

    return df

def ensure_dir(dir):
    """ Short function to ensure dir is a directory; if not, make the directory."""
    if not os.path.exists(dir):
        logger.info( "Making path=", dir)
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





