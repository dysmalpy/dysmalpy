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
from . import plotting


# Third party imports
import os
import numpy as np

import pickle as _pickle

import emcee
import psutil
import acor

import time, datetime

# ACOR SETTINGS
acor_force_min = 49
# Force it to run for at least 50 steps, otherwise acor times might be completely wrong.

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


def ensure_dir(dir):
    if not os.path.exists(dir):
        logger.info( "Making path=", dir)
        os.makedirs(dir)
    return None
    
    

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

def load_pickle(filename):
    data = _pickle.load(open(filename, "rb"))
    return data
    
def dump_pickle(data, filename=None):
    _pickle.dump(data, open(filename, "wb") )
    return None



class MCMCResults(object):
    def __init__(self, model, 
            sampler=None):
        self.sampler = sampler
        
    def analyze_posterior_dist(self):
        
        logger.info('WRITE THE MARGINALIZED POSTERIOR ANALYSIS')
        
        
        pass



def log_prob(theta, gal, inst, model):
    """
    Evaluate the log probability of the given model
    """
    model.update_parameters(theta)      # Update the parameters
    log_prior = model.get_log_prior()   # Evaluate prior prob of theta
    
    log_like = FIXTHISEVENTUALLY
    
    log_prob = log_prior + log_like
    
    if not np.isfinite(log_prob):
        # Make sure the non-finite ln_prob is -Inf, 
        #    as this can be escaped in the next step
        log_prob = -np.inf
    return log_prob


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
    pos = np.array(zip(*stack_rand))            # should have shape:   (nWalkers, nDim)
    return pos
    
    
    
def create_default_mcmc_options():
    
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
       filename_plot_trace_burnin = None,
       filename_plot_trace = None,
       filename_sampler = None, 
       filename_plot_param_corner = None, 
       filename_plot_bestfit = None,
       filename_mcmc_results = None)
    
    
    return mcmc_options
    
    
    
    
def fit(gal, inst, model, 
           nWalkers=10,
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
           filename_plot_trace_burnin = None,
           filename_plot_trace = None,
           filename_sampler = None, 
           filename_plot_param_corner = None, 
           filename_plot_bestfit = None, 
           filename_mcmc_results = None ):
    """
    Fit observed kinematics using DYSMALPY model set.
    Input:
            gal:            observed galaxy, including kinematics
            inst:           instrument galaxy was observed with
            model:          DSYMALPY model set, with parameters to be fit
            
            mcmc_options:   dictionary with MCMC fitting options
                            ** potentially expand this in the future, and force this to 
                            be an explicit set of parameters -- might be smarter!!!
    """
    # --------------------------------
    # Basic setup:
    if nCPUs is None:
        nCPUs = np.int(np.floor(psutil.cpu_count()*cpuFrac)) 
        
    nDim = len(model.get_free_parameters_values())
        
    # Output filenames
    if (len(out_dir) > 0): 
        if (out_dir[-1] != '/'): out_dir += '/'
    ensure_dir(out_dir)
    
    if filename_plot_trace_burnin is None:  filename_plot_trace_burnin = out_dir+'mcmc_burnin_trace.png'
    if filename_plot_trace is None:         filename_plot_trace = out_dir+'mcmc_trace.png'
    if filename_sampler is None:            filename_sampler = out_dir+'mcmc_sampler.pickle'
    if filename_plot_param_corner is None:  filename_plot_param_corner = out_dir+'mcmc_param_corner.png'
    if filename_plot_bestfit is None:       filename_plot_bestfit = out_dir+'mcmc_best_fit.png'
    if filename_mcmc_results is None:       filename_mcmc_results = out_dir+'mcmc_results.pickle'
    
    # --------------------------------
    # Initialize walker starting positions
    initial_pos = initialize_walkers(model, nWalkers=None)
    
    # --------------------------------
    # Initialize emcee sampler
    sampler = emcee.EnsembleSampler(nWalkers, nDim, log_prob, 
                args=(gal, inst, model,), a = scale_param_a, threads = nCPUs)
    
    # --------------------------------
    # Output some fitting info to logger:
    logger.info('nCPUs: '+str(nCPUs))
    #logger.info('nSubpixels = %s' % (model.nSubpixels))
    logger.info('')
    
    ################################################################
    # --------------------------------
    # Run burn-in
    if nBurn > 0:
        logger.info('Burn-in:'
                    'Start: '+str(datetime.datetime.now()))
        start = time.time()
        
        ####
        pos = initial_pos
        prob = None
        state = None
        for k in xrange(nBurn):
            logger.info("k, time.time=", k, datetime.datetime.now())
            pos, prob, state = sampler.run_mcmc(pos, 1, lnprob0=prob, rstate0=state)
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
        logger.info('End: '+str(datetime.datetime.now())
                    ''
                    '******************'
                    'nCPU, nParam, nWalker, nBurn = %s, %s, %s, %s'%(nCPUs,
                        nDim, nWalkers, nBurn)
                    'Scale param a= %s' % scale_param_a
                    'Time= %3.2f (sec), %3.0f:%3.2f (m:s)' % (elapsed, _np.floor(elapsed/60.), 
                            (elapsed/60.-_np.floor(elapsed/60.))*60.)
                    "Mean acceptance fraction: {0:.3f}".format(_np.mean(sampler.acceptance_fraction))
                    "Ideal acceptance frac: 0.2 - 0.5"
                    "Autocorr est: "+str(acor_time)
                    '******************')
        
        nBurn_nEff = 2
        try:
            if nBurn < _np.max(acor_time) * nBurn_nEff:
                logger.info('#################'
                            'nBurn is less than {}*acorr time'.format(nBurn_nEff)
                            '#################'+'\n')
                # Give warning if the burn-in is less than say 2-3 times the autocorr time
        except:
            logger.info('#################'
                        "acorr time undefined -> can't check convergence"
                        '#################')
        
        # --------------------------------
        # Plot burn-in trace, if output file set
        if (do_plotting) & (filename_plot_trace_burnin is not None):
            logger.info('WRITE THE PLOT TRACE BURNIN')
            # print('FIX THIS! NEED TO HAVE PLOTTING SPECIFIC TO THIS....')
            # raise ValueError
            # 
            # sampler_dict_burnin = make_emcee_sampler_dict(sampler, nBurn=0)
            # _misfit_plot.plot_trace(sampler_dict_burnin, fitEmis2D, 
            #                 fileout=filename_plot_trace_burnin)
        
        # Reset sampler after burn-in:
        sampler.reset()
            
    else:
        # --------------------------------
        # No burn-in: set initial position:
        pos = _np.array(initial_pos)
        prob = None
        state = None
        
        
    
    #######################################################################################
    # ****
    # --------------------------------
    # Run sampler: Get start time
    logger.info('Ensemble sampling:'
                'Start: '+str(datetime.datetime.now()))
    start = time.time()
    
    # --------------------------------
    # Case: test for convergence and truncate early:
    if((minAF is not None) & (maxAF is not None) & (nEff is not None)):
        for ii in xrange(nSteps):
            pos_cur = pos.copy()    # copy just in case things are set strangely
            
            # --------------------------------
            # 1: only do one step at a time.
            pos, prob, state = sampler.run_mcmc(pos_cur, 1, lnprob0=prob, rstate0=state)
            # --------------------------------
            
            # --------------------------------
            # Test for convergence
            logger.info( "time.time()=", datetime.datetime.now()
                         "ii=%s, a_frac=%s" % (ii, _np.mean(sampler.acceptance_fraction)))
            if(minAF < _np.mean(sampler.acceptance_fraction) < maxAF):
                try:
                    acor_time = [acor.acor(sampler.chain[:,:,jj])[0] for jj in range(sampler.dim)]
                    
                    logger.info( ii, ": acor_time =", _np.array(acor_time))
                    if ( ii > _np.max(acor_time) * nEff ):
                        if ii == acor_force_min:
                            logger.info("Enforced min step limit: {}.".format(ii+1))
                        if ii >= 49:
                            logger.info("Breaking chain at step {}.".format(ii+1))
                            break
                except RuntimeError:
                    # acor raises exception if the chain isn't long
                    # enough to compute the acor time. However, could also be other 
                    #   runtime errors..... need to be careful!
                    logger.info( ii, ": Chain is too short for acor to run")
                    pass
        
        # --------------------------------
        # Check if it failed to converge before the max number of steps
        finishedSteps= ii+1
        if (finishedSteps == nSteps):
            logger.info("Warning: chain did not converge after nSteps.")
        
        
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
    logger.info('End: '+str(datetime.datetime.now())
                ''
                '******************'
                'nCPU, nParam, nWalker, nSteps = %s, %s, %s, %s' % (nCPUs,
                    nDim, nWalkers, nSteps)
                'Scale param a= %s' % scale_param_a
                'Time= %3.2f (sec), %3.0f:%3.2f (m:s)' % (elapsed, _np.floor(elapsed/60.), 
                                    (elapsed/60.-_np.floor(elapsed/60.))*60. ) 
                "Mean acceptance fraction: {0:.3f}".format(_np.mean(sampler.acceptance_fraction))
                "Ideal acceptance frac: 0.2 - 0.5"
                "Autocorr est: "+str(acor_time) 
                '******************')
    
    
    # --------------------------------
    # Save sampler, if output file set:
    #   Burn-in is already cut by resetting the sampler at the beginning.
    # Get pickleable format:  # _fit_io.make_emcee_sampler_dict
    sampler_dict = make_emcee_sampler_dict(sampler, nBurn=0)
    
    if filename_sampler is not None:
        # Save stuff to file, for future use:
        dump_pickle(sampler_dict, filename=filename_sampler)
        
            
        
    ##########################################
    ##########################################
    ##########################################
    
    # --------------------------------
    # Plot trace, if output file set
    if (do_plotting) & (filename_plot_trace is not None) :
        logger.info('WRITE THE PLOT TRACE')
        
        #_misfit_plot.plot_trace(sampler_dict, fitEmis2D, fileout=filename_plot_trace)
    
    # --------------------------------
    # Bundle the results up into a results class:
    mcmcResults = MCMCResults(model, sampler=sampler_dict)
    mcmcResults.analyze_posterior_dist()   # Get the best-fit values, 
                                           # uncertainty bounds from marginalized posteriors
   if filename_mcmc_results is not None:
       dump_pickle(mcmcResults, filename=filename_mcmc_results) # Save mcmcResults class 
    
    # --------------------------------
    # Plot results: corner plot, best-fit
    if (do_plotting) & (filename_plot_param_corner is not None):
        logger.info('WRITE THE PLOT PARAM CORNER')
        pass
        
    if (do_plotting) & (filename_plot_bestfit is not None):
        logger.info('WRITE THE PLOT PARAM BESTFIT')
        pass
                         
    return mcmcResults
    
    
    
    
    
    
    
    
    
    
    
    
    
    
