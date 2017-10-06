# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Main classes and functions for fitting DYSMALPY kinematic models 
#   to the observed data using MCMC

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


class MCMCResults(object):
    def __init__():
        
        pass



def initialize_walkers(modset, nWalkers=None):
    """ 
    Initialize a set of MCMC walkers by randomly drawing from the 
    model set parameter priors
    """
    # nDim = len(modset.get_free_parameters_values())
    stack_rand = []
    pfree_dict = modset.get_free_parameter_keys()
    comps_names = pfree_dict.keys()
    
    for compn in comps_names:
        comp = modset.components.__getitem__(compn)
        params_names = pfree_dict[compn].keys()
        for paramn in params_names:
            if (pfree_dict[compn][paramn] >= 0) :
                # Free parameter: randomly sample from prior nWalker times:
                param_rand = comp.prior[paramn].sample_prior(comp.__getattribute__(paramn), N=nWalkers)
                stack_rand.append(param_rand)
    pos = np.array(zip(*stack_rand))            # should have shape:   (nWalkers, nDim)
    return pos
    
    
    
def create_default_mcmc_options():
    
    
    
    
    
    return mcmc_options
    
    
    
    
def fit(gal, inst, modset, 
        nWalkers=10):
    """
    Fit observed kinematics using DYSMALPY model set.
    Input:
            gal:            observed galaxy, including kinematics
            inst:           instrument galaxy was observed with
            modset:         DSYMALPY model set, with parameters to be fit
            
            mcmc_options:   dictionary with MCMC fitting options
                            ** potentially expand this in the future, and force this to 
                            be an explicit set of parameters -- might be smarter!!!
    """
    
    
    
    
    print('Write this function!')
    raise ValueError



