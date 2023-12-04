# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Top-level module for fitting DYSMALPY kinematic models to the observed data
#   Fitting method-specific classes/functions are found under the
#   'dysmalpy.fitting' submodule.
#

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

## Standard library
import logging

# DYSMALPY code
# from dysmalpy.data_io import ensure_dir, load_pickle, dump_pickle

# Local imports:
from .base import chisq_red, chisq_eval, chisq_red_per_type
from .utils import setup_oversampled_chisq
from .mcmc import MCMCFitter, MCMCResults, _reload_all_fitting_mcmc
from .nested_sampling import NestedFitter, NestedResults, _reload_all_fitting_nested
from .mpfit import MPFITFitter, MPFITResults, _reload_all_fitting_mpfit

__all__ = ['MCMCFitter', 'NestedFitter', 'MPFITFitter',
           'MCMCResults', 'NestedResults', 'MPFITResults',
           'reload_all_fitting',
           'chisq_red', 'chisq_eval', 'chisq_red_per_type',
           'setup_oversampled_chisq']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)


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
            Can be set to `mpfit`, `mcmc`, or `nested`.

    Returns
    -------
    gal : obj
            Galaxy instance, including model with the current best-fit parameters
    retults : obj
            MCMCResults or MPFITResults instance, containing all fit results and analysis

    """

    if fit_method is None:
        raise ValueError("Must set 'fit_method'! Options are 'mpfit', 'mcmc', or 'nested.")

    if fit_method.lower().strip() == 'mcmc':
        return _reload_all_fitting_mcmc(filename_galmodel=filename_galmodel, filename_results=filename_results)
    elif fit_method.lower().strip() == 'nested':
        return _reload_all_fitting_nested(filename_galmodel=filename_galmodel, filename_results=filename_results)
    elif fit_method.lower().strip() == 'mpfit':
        return _reload_all_fitting_mpfit(filename_galmodel=filename_galmodel, filename_results=filename_results)
    
    else:
        raise ValueError("Fit type {} not recognized!".format(fit_method))
