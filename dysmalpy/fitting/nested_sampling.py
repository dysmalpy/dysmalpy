# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information.
#
# Classes and functions for fitting DYSMALPY kinematic models
#   to the observed data using Nested sampling, with Dynesty:
#   Speagle 2020, https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract
#   dynesty.readthedocs.io

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
from dysmalpy.data_io import load_pickle, dump_pickle, pickle_module
from dysmalpy import plotting
from dysmalpy import galaxy
# from dysmalpy.utils import fit_uncertainty_ellipse
# from dysmalpy import utils_io as dpy_utils_io
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


__all__ = ['NestedFitter', 'NestedResults']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

try:
    import dynesty
    import dynesty.utils
    dynesty.utils.pickle_module = pickle_module
    _dynesty_loaded = True
except:
    _dynesty_loaded = False
    logger.warn("dynesty installation not found!")


class NestedFitter(base.Fitter):
    """
    Class to hold the Nested sampling fitter attributes + methods
    """
    def __init__(self, **kwargs):
        if not _dynesty_loaded:
            raise ValueError("dynesty was not loaded!")

        self._set_defaults()
        super(NestedFitter, self).__init__(fit_method='Nested', **kwargs)

    def _set_defaults(self):
        # Nested sampling specific defaults
        self.maxiter = None     # No limit to iterations

        self.bound = 'multi'
        self.sample = 'unif'

        self.nlive_init = 1000
        self.nlive_batch = 1000
        self.use_stop = True
        self.pfrac = 1.0

        self.nCPUs = 1.0
        self.cpuFrac = None

        self.oversampled_chisq = True

        self.nPostBins = 50
        self.linked_posterior_names = None


        self.print_func = None


    def fit(self, gal, output_options):
        """
        Fit observed kinematics using nested sampling and a DYSMALPY model set.

        Parameters
        ----------
            gal : `Galaxy` instance
                observed galaxy, including kinematics.
                also contains instrument the galaxy was observed with (gal.instrument)
                and the DYSMALPY model set, with the parameters to be fit (gal.model)

            output_options : `config.OutputOptions` instance
                instance holding ouptut options for nested sampling fitting.

        Returns
        -------
            nestedResults : `NestedResults` instance
                NestedResults class instance containing the bestfit parameters, sampler_results information, etc.
        """

        # --------------------------------
        # Check option validity:

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


        # MUST INCLUDE NESTED-SPECIFICS NOW!
        fit_utils._check_existing_files_overwrite(output_options,
                                                  fit_type='nested',
                                                  fitter=self)


        # --------------------------------
        # Setup file redirect logging:
        if output_options.f_log is not None:
            loggerfile = logging.FileHandler(output_options.f_log)
            loggerfile.setLevel(logging.INFO)
            logger.addHandler(loggerfile)

        # ++++++++++++++++++++++++++++++++
        # Run Dynesty:

        # Keywords for log likelihood:
        logl_kwargs = {'gal': gal,
                       'fitter': self}

        # Keywords for prior transform:
        # This needs to include the gal object,
        #   so we can get the appropriate per-free-param priors
        ptform_kwargs = {'gal': gal}

        ndim = gal.model.nparams_free

        # Set blob switch for Dynesty:
        if self.blob_name is not None:
            _calc_blob = True
        else:
            _calc_blob = False

        # --------------------------------
        # Start pool
        if (self.nCPUs > 1):
            pool = Pool(self.nCPUs)
            queue_size = self.nCPUs
        else:
            pool = queue_size = None


        dsampler = dynesty.DynamicNestedSampler(log_like_dynesty,
                                                prior_transform_dynsety,
                                                ndim,
                                                bound=self.bound,
                                                sample=self.sample,
                                                logl_kwargs=logl_kwargs,
                                                ptform_kwargs=ptform_kwargs,
                                                blob=_calc_blob,
                                                pool=pool,
                                                queue_size=queue_size)


        resume = False

        # If not overwriting, check to see if there is a file to restore:
        if not output_options.overwrite:
            if output_options.f_checkpoint is not None:
                if os.path.isfile(output_options.f_checkpoint):
                    logger.info("Reloading checkpoint file: {}".format(output_options.f_checkpoint))

                    # Resume the checkpointed sampler from file:
                    resume = True
                    dsampler = dynesty.DynamicNestedSampler.restore(output_options.f_checkpoint,
                                                pool=pool)


        dsampler.run_nested(nlive_init=self.nlive_init,
                            nlive_batch=self.nlive_batch,
                            maxiter=self.maxiter,
                            use_stop=self.use_stop,
                            resume=resume,
                            print_func=self.print_func,
                            checkpoint_file=output_options.f_checkpoint,
                            wt_kwargs={'pfrac': self.pfrac})

        res = dsampler.results

        if output_options.f_sampler_results is not None:
            # Save stuff to file, for future use:
            dump_pickle(res, filename=output_options.f_sampler_results,
                        overwrite=output_options.overwrite)


        # --------------------------------
        # Bundle the results up into a results class:
        nestedResults = NestedResults(model=gal.model, sampler_results=res,
                    linked_posterior_names=self.linked_posterior_names,
                    blob_name=self.blob_name,
                    nPostBins=self.nPostBins)

        if self.oversampled_chisq:
            nestedResults.oversample_factor_chisq = OrderedDict()
            for obs_name in gal.observations:
                obs = gal.observations[obs_name]
                nestedResults.oversample_factor_chisq[obs_name] = obs.data.oversample_factor_chisq

        # Do all analysis, plotting, saving:
        nestedResults.analyze_plot_save_results(gal, output_options=output_options)

        # --------------------------------
        # Clean up logger:
        if output_options.f_log is not None:
            logger.removeHandler(loggerfile)
            loggerfile.close()

        return nestedResults





class NestedResults(base.BayesianFitResults, base.FitResults):
    """
    Class to hold results of nested sampling fitting to DYSMALPY models.

    Notes:
    ------
        The dynesty *Results* object (containing the results of the run) is stored in `nestedResults.sampler_results`.

        The names of free parameters in the chain are accessed through `nestedResults.chain_param_names` or more generally (separate model + parameter names) through `nestedResults.free_param_names`

    Optional Attribute:
    ----------------------
        `linked_posterior_names`
            Indicates if best-fit parameters should be measured in multi-dimensional histogram space.
            It takes a list of linked parameter sets, where each set consists of len-2 tuples/lists of
            the component + parameter names.

    Structure Explanation:
    ----------------------
    1. To analyze component + param 1 and 2 together, and then 3 and 4 together: `linked_posterior_names = [joint_param_bundle1, joint_param_bundle2]` with `joint_param_bundle1 = [[cmp1, par1], [cmp2, par2]]` and `joint_param_bundle2 = [[cmp3, par3], [cmp4, par4]]`, for a full array of: `linked_posterior_names = [[[cmp1, par1], [cmp2, par2]],[[cmp3, par3], [cmp4, par4]]]`.

    2. To analyze component + param 1 and 2 together: `linked_posterior_names = [joint_param_bundle1]` with `joint_param_bundle1 = [[cmp1, par1], [cmp2, par2]]`, for a full array of `linked_posterior_names = [[[cmp1, par1], [cmp2, par2]]]`.
            Example: Look at halo: mvirial and disk+bulge: total_mass together
                `linked_posterior_names = [[['halo', 'mvirial'], ['disk+bulge', 'total_mass']]]`
    """
    def __init__(self, model=None, sampler_results=None,
                 linked_posterior_names=None,
                 blob_name=None, nPostBins=50):

        super(NestedResults, self).__init__(model=model, blob_name=blob_name,
                                            fit_method='Nested',
                                            linked_posterior_names=linked_posterior_names,
                                            sampler_results=sampler_results, nPostBins=nPostBins)

    def __setstate__(self, state):
        # Compatibility hacks
        super(NestedResults, self).__setstate__(state)

        # # ---------
        # if ('sampler' not in state.keys()) & ('sampler_results' in state.keys()):
        #     self._setup_samples_blobs()


    def _setup_samples_blobs(self):

        # Extract weighted samples, as in
        # https://dynesty.readthedocs.io/en/v1.2.3/quickstart.html?highlight=resample_equal#basic-post-processing

        samples_unweighted = self.sampler_results.samples
        blobs_unweighted = self.sampler_results.blob

        # weights = np.exp(self.sampler.logwt - self.sampler.logz[-1])

        # Updated, see https://dynesty.readthedocs.io/en/v2.0.3/quickstart.html#basic-post-processing
        weights = self.sampler_results.importance_weights()

        samples = dynesty.utils.resample_equal(samples_unweighted, weights)

        # Check if blobs_unweighted is None?
        blobs = dynesty.utils.resample_equal(blobs_unweighted, weights)

        self.sampler = base.BayesianSampler(samples=samples, blobs=blobs,
                                            weights=weights,
                                            samples_unweighted=samples_unweighted,
                                            blobs_unweighted=blobs_unweighted)

    def plot_run(self, fileout=None, overwrite=False):
        """Plot/replot the trace for the Bayesian fitting"""
        plotting.plot_run(self, fileout=fileout, overwrite=overwrite)

    def reload_sampler_results(self, filename=None):
        """Reload the Nested sampling results saved earlier"""
        if filename is None:
            #filename = self.f_sampler_results
            raise ValueError

        #hdf5_aliases = ['h5', 'hdf5']
        pickle_aliases = ['pickle', 'pkl', 'pcl']
        # if (filename.split('.')[-1].lower() in hdf5_aliases):
        #     self.sampler_results = _reload_sampler_results_hdf5(filename=filename)

        # elif (filename.split('.')[-1].lower() in pickle_aliases):
        if (filename.split('.')[-1].lower() in pickle_aliases):
            self.sampler_results = _reload_sampler_results_pickle(filename=filename)



def log_like_dynesty(theta, gal=None, fitter=None):

    # Update the parameters
    gal.model.update_parameters(theta)

    # Update the model data
    gal.create_model_data()

    # Evaluate likelihood prob of theta
    llike = base.log_like(gal, fitter=fitter)

    return llike


def prior_transform_dynsety(u, gal=None):
    """
    From Dynesty documentation:
    Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest.
    """
    # NEEDS TO BE IN ORDER OF THE VARIABLES
    # -- which means we need to construct this from the gal.model method

    v = gal.model.get_prior_transform(u)

    return v



def _reload_sampler_results_pickle(filename=None):
    return load_pickle(filename)



def _reload_all_fitting_nested(filename_galmodel=None, filename_results=None):
    gal = galaxy.load_galaxy_object(filename=filename_galmodel)
    results = NestedResults()
    results.reload_results(filename=filename_results)
    return gal, results
