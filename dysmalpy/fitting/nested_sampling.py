# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Classes and functions for fitting DYSMALPY kinematic models
#   to the observed data using Nested sampling, with Dynesty
#    (REF TO SPEAGLE, ET AL XXXX)

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
from dysmalpy import config
from dysmalpy.utils import fit_uncertainty_ellipse
from dysmalpy import utils_io as dpy_utils_io

# Local imports:
from .base import FitResults, Fitter, make_arr_cmp_params, \
                  chisq_eval, chisq_red, chisq_red_per_type
from .mcmc import shortest_span_bounds

# Third party imports
import os
import numpy as np
from collections import OrderedDict
import six
import astropy.units as u
import copy
import h5py

import time, datetime

from scipy.stats import gaussian_kde
from scipy.optimize import fmin


import dynesty
from dynesty import plotting as dyplot



__all__ = ['fit_nested', 'NestedFitter', 'NestedResults']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')



class NestedFitter(Fitter):
    """
    Class to hold the Nested sampling fitter attributes + methods
    Uses Dynesty
    """
    def __init__(self, **kwargs):
        self._set_defaults()
        super(NestedFitter, self).__init__(fit_method='nested', **kwargs)

    def _set_defaults(self):
        self.maxiter=None

        self.bound = 'multi'
        self.sample = 'unif'

        self.nlive_init = 100
        self.nlive_batch = 100
        self.use_stop = False
        self.pfrac = 1.0

        self.nCPUs = 1.0


    def fit():
  
        res = dsampler.results

        
        nestedResults = NestedResults(model=gal.model, res=res, **kwargs_fit)

        return nestedResults





class NestedResults(FitResults):
    """
    Class to hold results of Nested sampling fitting to DYSMALPY models.

        The name of the free parameters in the chain are accessed through:
            mcmcResults.chain_param_names,
                or more generally (separate model + parameter names) through
                mcmcResults.free_param_names
    """
    def __init__(self, model=None,
                 res=None,
                 f_plot_trace_burnin=None,
                 f_plot_trace=None,
                 f_burn_sampler=None,
                 f_sampler=None,
                 f_plot_param_corner=None,
                 f_plot_bestfit=None,
                 f_plot_spaxel=None,
                 f_plot_aperture=None,
                 f_plot_channel=None,
                 f_results=None,
                 f_chain_ascii=None,
                 linked_posterior_names=None,
                 blob_name=None,
                 **kwargs):

        self.res = res
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
                    f_plot_spaxel=f_plot_spaxel, f_plot_aperture=f_plot_aperture, f_plot_channel=f_plot_channel,
                    f_results=f_results, fit_method='MCMC')


    def analyze_plot_save_results(self, gal,
                linked_posterior_names=None,
                nPostBins=50,
                model_aperture_r=None,
                model_key_halo=None,
                fitvelocity=True,
                fitdispersion=True,
                fitflux=False,
                save_data=True,
                save_bestfit_cube=False,
                f_cube=None,
                f_model=None,
                f_model_bestfit = None,
                f_vel_ascii = None,
                f_vcirc_ascii = None,
                f_mass_ascii = None,
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
                    self.analyze_dm_posterior_dist(gal=gal, model_aperture_r=model_aperture_r, blob_name=self.blob_name)  # here blob_name should be the *full* list
                elif blobn.lower() == 'mvirial':
                    self.analyze_mvirial_posterior_dist(gal=gal, model_key_halo=model_key_halo, blob_name=self.blob_name)
                elif blobn.lower() == 'alpha':
                    self.analyze_alpha_posterior_dist(gal=gal, model_key_halo=model_key_halo, blob_name=self.blob_name)
                elif blobn.lower() == 'rb':
                    self.analyze_rb_posterior_dist(gal=gal, model_key_halo=model_key_halo, blob_name=self.blob_name)


        gal.create_model_data(**kwargs_galmodel)

        self.bestfit_redchisq = chisq_red(gal, fitvelocity=fitvelocity,
                        fitdispersion=fitdispersion, fitflux=fitflux)
        self.bestfit_chisq = chisq_eval(gal, fitvelocity=fitvelocity,
                                fitdispersion=fitdispersion, fitflux=fitflux)

        if ((gal.data.ndim == 1) or (gal.data.ndim ==2)):
            kwargs_fit = {'fitvelocity': fitvelocity,
                          'fitdispersion': fitdispersion,
                          'fitflux': fitflux}
            for k in ['velocity', 'dispersion', 'flux']:
                if kwargs_fit['fit{}'.format(k)]:
                    self.__dict__['bestfit_redchisq_{}'.format(k)] = chisq_red_per_type(gal, type=k)



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
            plotting.plot_bestfit(self, gal, fitvelocity=fitvelocity,
                                  fitdispersion=fitdispersion, fitflux=fitflux,
                                  fileout=self.f_plot_bestfit, overwrite=overwrite, **kwargs_galmodel)

        # --------------------------------
        # Save velocity / other profiles to ascii file:
        if f_vel_ascii is not None:
            self.save_bestfit_vel_ascii(gal, filename=f_vel_ascii,
                                        model_aperture_r=model_aperture_r, overwrite=overwrite)

        if (f_vcirc_ascii is not None) or (f_mass_ascii is not None):
            self.save_bestfit_vcirc_mass_profiles(gal, fname_intrinsic=f_vcirc_ascii,
                fname_intrinsic_m=f_mass_ascii, overwrite=overwrite)


    def mod_linear_param_posterior(self, gal=None):
        linear_posterior = []
        j = -1
        for cmp in gal.model.fixed:
            # pkeys[cmp] = OrderedDict()
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

        PA, stddev_x, stddev_y  = fit_uncertainty_ellipse(chain_x, chain_y, bins=bins)
        return PA, stddev_x, stddev_y




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



    def plot_results(self, gal, fitvelocity=True, fitdispersion=True, fitflux=False,
                     f_plot_param_corner=None, f_plot_bestfit=None,
                     f_plot_spaxel=None, f_plot_aperture=None, f_plot_channel=None,
                     f_plot_trace=None,
                     overwrite=False, **kwargs_galmodel):
        """Plot/replot the corner plot and bestfit for the MCMC fitting"""
        # Specific to 3D: 'f_plot_spaxel', 'f_plot_aperture', 'f_plot_channel'
        self.plot_corner(gal=gal, fileout=f_plot_param_corner, overwrite=overwrite)
        self.plot_bestfit(gal, fitvelocity=fitvelocity,
                fitdispersion=fitdispersion, fitflux=fitflux,
                fileout=f_plot_bestfit, fileout_aperture=f_plot_aperture,
                fileout_spaxel=f_plot_spaxel, fileout_channel=f_plot_channel,
                overwrite=overwrite, **kwargs_galmodel)
        self.plot_trace(fileout=f_plot_trace, overwrite=overwrite)


    def plot_corner(self, gal=None, fileout=None, overwrite=False):
        """Plot/replot the corner plot for the MCMC fitting"""
        plotting.plot_corner(self, gal=gal, fileout=fileout, blob_name=self.blob_name, overwrite=overwrite)


    def plot_trace(self, fileout=None, overwrite=False):
        """Plot/replot the trace for the MCMC fitting"""
        plotting.plot_trace(self, fileout=fileout, overwrite=overwrite)

