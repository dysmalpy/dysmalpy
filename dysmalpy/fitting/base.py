# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Base classes / methods for fitting / fit results

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

## Standard library
import logging
import abc

# DYSMALPY code
from dysmalpy import plotting
from dysmalpy import utils_io as dpy_utils_io
from dysmalpy import utils as dpy_utils
from dysmalpy.data_io import load_pickle, dump_pickle
from dysmalpy.fitting import utils as fit_utils

from dysmalpy.parameters import UniformLinearPrior

# Third party imports
import os, copy
import numpy as np
from collections import OrderedDict
import astropy.units as u


_bayesian_fitting_methods = ['mcmc', 'nested']


__all__ =  ['Fitter', 'FitResults',
            'log_prob', 'log_like', 
            'chisq_red', 'chisq_eval', 
            'chisq_red_per_type']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)


class Fitter(object):
    """
    General class to hold the fitter attributes + methods
    """

    def __init__(self, **kwargs):
        self._set_defaults_base()
        self._fill_values(**kwargs)

    def _set_defaults_base(self):
        self.fit_method = None
        self.blob_name = None

    def _fill_values(self, **kwargs):
        for key in self.__dict__.keys():
            if key in kwargs.keys():
                self.__dict__[key] = kwargs[key]

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit a galaxy with output options"""
        pass



class FitResults(object):
    """
    General class to hold the results of any fitting
    """

    def __init__(self, model=None, fit_method=None, blob_name=None):

        self.bestfit_parameters = None
        self.bestfit_parameters_err = None
        self.bestfit_redchisq = None
        self._fixed = None
        self.blob_name = blob_name

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

        self.chain_param_names = fit_utils.make_arr_cmp_params(self)

    def save_results(self, filename=None, overwrite=False):
        # Check for existing file:
        if (not overwrite) and (filename is not None):
            if os.path.isfile(filename):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
                return None

        if filename is not None:
            dump_pickle(self, filename=filename, overwrite=overwrite)  # Save FitResults class to a pickle file

    def save_bestfit_vel_ascii(self, tracer, model, filename=None, overwrite=False):
        if filename is not None:
            try:
                r_ap = model._model_aperture_r()
            except:
                r_ap = 10./3.
            rmax = np.max([3. * r_ap, 10.])
            stepsize = 0.1  # stepsize 0.1 kpc
            r = np.arange(0., rmax + stepsize, stepsize)

            model.write_vrot_vcirc_file(r=r, filename=filename, overwrite=overwrite, tracer=tracer)


    def save_bestfit_vcirc_mass_profiles(self, gal, outpath=None,
            fname_intrinsic=None, fname_intrinsic_m=None, overwrite=False):
        """Save the best-fit vcirc, enclosed mass profiles"""
        dpy_utils_io.create_vel_profile_files_intrinsic(gal=gal, outpath=outpath,
                    fname_intrinsic=fname_intrinsic, fname_intrinsic_m=fname_intrinsic_m,
                    overwrite=overwrite)


    @abc.abstractmethod
    def analyze_plot_save_results(self, *args, **kwargs):
        """Method to do finishing analysis, plotting, and result saving after fitting."""

    @abc.abstractmethod
    def plot_results(self, *args, **kwargs):
        """
        Method to produce all of the necessary plots showing the results of the fitting.
        :param args:
        :param kwargs:
        :return:
        """

    def plot_bestfit(self, gal, fileout=None, overwrite=False):
        """Plot/replot the bestfit for the fitting"""

        plotting.plot_bestfit(self, gal, fileout=fileout, overwrite=overwrite)


    def reload_results(self, filename=None):
        """Reload results saved earlier: the whole object"""
        if filename is None:
            #filename = self.f_results
            raise ValueError
        resultsSaved = load_pickle(filename)
        for key in resultsSaved.__dict__.keys():
            try:
                self.__dict__[key] = resultsSaved.__dict__[key]
            except:
                pass

    def results_report(self, gal=None, filename=None, output_options=None,
                       report_type='pretty', overwrite=False):
        """Return a result report string, or save to file.
           report_type = 'pretty':   More human-readable
                       = 'machine':  Machine-readable ascii table (though with mixed column types)

           **kwargs: can pass other setting values: eg zcalc_truncate.
        """

        report = dpy_utils_io.create_results_report(gal, self, report_type=report_type,
                                                    output_options=output_options)

        if filename is not None:
            if overwrite & os.path.isfile(filename):
                os.remove(filename)
            with open(filename, 'w') as f:
                f.write(report)
        else:
            return report


class BayesianFitResults(FitResults):
    """
    Class to hold the results of any Bayesian fitting, exending FitResults.
    Holds extra methods to extend FitResults for specific cases
    """
    def __init__(self, model=None, fit_method=None, 
                 blob_name=None, 
                 sampler_results=None, 
                 sampler=None, # backwards compatibility
                 linked_posterior_names=None, 
                 nPostBins=50):

        # Ensure the fit method is a valid Bayesian one:
        if fit_method.lower().strip() not in _bayesian_fitting_methods:
            msg = "Fit method {} is not a supported Bayesian method".format(fit_method)
            raise ValueError(msg)


        super(BayesianFitResults, self).__init__(model=model,
                                                 fit_method=fit_method, 
                                                 blob_name=blob_name)

        # Initialize some Bayesian result-specific attributes, 
        # including the BayesianSampler class instance 'sampler'
        self.sampler_results = sampler_results

        if (sampler_results is None) & (sampler is not None):
            self.sampler_results = sampler

        self.linked_posterior_names = linked_posterior_names
        self.nPostBins = nPostBins

        self.bestfit_parameters_l68_err = None
        self.bestfit_parameters_u68_err = None
        self.bestfit_parameters_l68 = None
        self.bestfit_parameters_u68 = None


    def __setstate__(self, state):
        # Compatibility hacks

        state_new = copy.deepcopy(state)

        # ---------
        # CHANGED THE NAMING SCHEME: Sampler is now a separate class 
        # to unify handling of emcee and dynesty specific outputs 
        if ('sampler' in state.keys()) & ('_sampler_results' not in state.keys()):
            state_new.pop('sampler', None)
            state_new['_sampler_results'] = state['sampler']

        self.__dict__ = state_new


        # ---------
        # If the old style: setup the samples and blobs:
        if ('sampler' in state.keys()) & ('_sampler_results' not in state.keys()):
            self._setup_samples_blobs()


    @property
    def sampler_results(self):
        return self._sampler_results

    @sampler_results.setter
    def sampler_results(self, value):
        self._sampler_results = value

        # Set up samples, and blobs if blob_name != None
        if self._sampler_results is not None:
            self._setup_samples_blobs()

    @abc.abstractmethod
    def _setup_samples_blobs(self, *args, **kwargs):
        """
        Method to set up the posterior samples + blob samples
        """


    def analyze_plot_save_results(self, gal, output_options=None):
        """
        Wrapper for post-sample analysis + plotting

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

        self.bestfit_redchisq = chisq_red(gal)
        self.bestfit_chisq = chisq_eval(gal)

        # self.vmax_bestfit = gal.model.get_vmax()

        if output_options.f_results is not None:
            self.save_results(filename=output_options.f_results, 
                              overwrite=output_options.overwrite)

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
        # Plot results, if output files set:

        self.plot_results(gal, f_plot_param_corner=output_options.f_plot_param_corner, 
                          f_plot_bestfit=output_options.f_plot_bestfit,
                          f_plot_trace=output_options.f_plot_trace, 
                          f_plot_run=output_options.f_plot_run, 
                          overwrite=output_options.overwrite, 
                          only_if_fname_set=True)

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

    def plot_results(self, gal, f_plot_param_corner=None, f_plot_bestfit=None,
                     f_plot_trace=None, f_plot_run=None, 
                     overwrite=False, only_if_fname_set=False):
        """Plot/replot the corner plot and bestfit for the Bayesian fitting"""

        if (not only_if_fname_set) | (f_plot_param_corner is not None):
            self.plot_corner(gal=gal, fileout=f_plot_param_corner, overwrite=overwrite)
        if (not only_if_fname_set) | (f_plot_bestfit is not None):
            self.plot_bestfit(gal, fileout=f_plot_bestfit, overwrite=overwrite)
        if (not only_if_fname_set) | (f_plot_trace is not None):
            self.plot_trace(fileout=f_plot_trace, overwrite=overwrite)
        if (not only_if_fname_set) | (f_plot_run is not None):
            if hasattr(self, 'plot_run'):
                self.plot_run(fileout=f_plot_run, overwrite=overwrite)


    def plot_corner(self, gal=None, fileout=None, overwrite=False):
        """Plot/replot the corner plot for the Bayesian fitting"""
        plotting.plot_corner(self, gal=gal, fileout=fileout, overwrite=overwrite)

    def plot_trace(self, fileout=None, overwrite=False):
        """Plot/replot the trace for the Bayesian fitting"""
        plotting.plot_trace(self, fileout=fileout, overwrite=overwrite)


    @abc.abstractmethod
    def reload_sampler_results(self, *args, **kwargs):
        """
        Method to reload the sampler results
        :param args:
        :param kwargs:
        :return:
        """

    # Backwards compatibility:
    def reload_sampler(self, *args, **kwargs):
        msg = "FitResults.reload_sampler() is now depreciated in favor of \n"
        msg += "FitResults.reload_sampler_results()"
        logger.warning(msg)
        return self.reload_sampler_results(*args, **kwargs)
    

    # General methods:
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
                        self.sampler.samples[:,j] = np.power(10.,self.sampler.samples[:,j])
                        linear_posterior.append(True)
                    else:
                        linear_posterior.append(False)

        self.linear_posterior = linear_posterior

    def back_map_linear_param_bestfits(self, param_bestfit, limits, limits_percentile):
        param_bestfit_linear = param_bestfit.copy()
        limits_linear = limits.copy()

        for j in range(len(param_bestfit)):
            if self.linear_posterior[j]:
                param_bestfit[j] = np.log10(param_bestfit[j])
                limits[:,j] = np.log10(limits[:, j])
                limits_percentile[:,j] = np.log10(limits_percentile[:, j])

        return param_bestfit, param_bestfit_linear, limits, limits_linear, limits_percentile

    def analyze_posterior_dist(self, gal=None):
        """
        Default analysis of posterior distributions from Bayesian fitting:
            look at marginalized posterior distributions, and
            extract the best-fit value (peak of KDE), and extract the +- 1 sigma uncertainty bounds
            (eg, the 16%/84% distribution of posteriors)

        """

        if self.sampler_results is None:
            raise ValueError("results.sampler_results must be set to analyze the posterior distribution.")

        self.mod_linear_param_posterior(gal=gal)

        # Unpack samples: lower, upper 1, 2 sigma
        limits_percentile = np.percentile(self.sampler.samples, 
                                                 [15.865, 84.135], axis=0)

        limits = fit_utils.shortest_span_bounds(self.sampler.samples, 
                                                       percentile=0.6827)


        ## location of peaks of *marginalized histograms* for each parameter
        peak_hist = np.zeros(self.sampler.samples.shape[1])
        for i in range(self.sampler.samples.shape[1]):
            yb, xb = np.histogram(self.sampler.samples[:,i], bins=self.nPostBins)
            wh_pk = np.where(yb == yb.max())[0][0]
            peak_hist[i] = np.average([xb[wh_pk], xb[wh_pk+1]])

        ## Use max prob as guess to get peak value of the gaussian KDE, to find 'best-fit' of the posterior:
        param_bestfit = fit_utils.find_peak_gaussian_KDE(self.sampler.samples, peak_hist)

        # --------------------------------------------
        if self.linked_posterior_names is not None:
            # Make sure the param of self is updated
            #   (for ref. when reloading saved BayesianFitResult objects)
            linked_posterior_ind_arr = fit_utils.get_linked_posterior_indices(self)
            guess = param_bestfit.copy()
            bestfit_theta_linked = fit_utils.get_linked_posterior_peak_values(self.sampler.samples,
                            guess=guess, linked_posterior_ind_arr=linked_posterior_ind_arr,
                            nPostBins=self.nPostBins)

            for k in range(len(linked_posterior_ind_arr)):
                for j in range(len(linked_posterior_ind_arr[k])):
                    param_bestfit[linked_posterior_ind_arr[k][j]] = bestfit_theta_linked[k][j]

        # --------------------------------------------
        # Uncertainty bounds are currently determined from marginalized posteriors
        #   (even if the best-fit is found from linked posterior).

        # --------------------------------------------
        # Save best-fit results in the BayesianFitResult instance

        self.bestfit_parameters = param_bestfit
        self.bestfit_redchisq = None

        # ++++++++++++++++++++++++=
        # Original 68% percentile interval:
        stack_percentile = np.concatenate(([param_bestfit], limits_percentile), axis=0)
        # Order: best fit value, lower 1sig bound, upper 1sig bound

        uncertainties_1sig_percentile = np.array(list(map(lambda v: (v[0]-v[1], v[2]-v[0]),
                            list(zip(*stack_percentile)))))

        # 1sig lower, upper uncertainty
        self.bestfit_parameters_err_percentile = uncertainties_1sig_percentile

        # Bound limits (in case it's useful)
        self.bestfit_parameters_l68_percentile = limits_percentile[0]
        self.bestfit_parameters_u68_percentile = limits_percentile[1]

        # Separate 1sig l, u uncertainty, for utility:
        self.bestfit_parameters_l68_err_percentile = param_bestfit - limits_percentile[0]
        self.bestfit_parameters_u68_err_percentile = limits_percentile[1] - param_bestfit


        # ++++++++++++++++++++++++=
        # From new shortest credible interval:
        stack = np.concatenate(([param_bestfit], limits), axis=0)
        # Order: best fit value, lower 1sig bound, upper 1sig bound

        uncertainties_1sig = np.array(list(map(lambda v: (v[0]-v[1], v[2]-v[0]),
                            list(zip(*stack)))))

        # 1sig lower, upper uncertainty
        self.bestfit_parameters_err = uncertainties_1sig

        # Bound limits (in case it's useful)
        self.bestfit_parameters_l68 = limits[0]
        self.bestfit_parameters_u68 = limits[1]

        # Separate 1sig l, u uncertainty, for utility:
        self.bestfit_parameters_l68_err = param_bestfit - limits[0]
        self.bestfit_parameters_u68_err = limits[1] - param_bestfit

    def analyze_blob_posterior_dist(self, bestfit=None, parname=None, blob_name=None):
        # Eg: parname = 'fdm' / 'mvirial' / 'alpha'
        if self.sampler_results is None:
            raise ValueError("results.sampler_results must be set to analyze the posterior distribution.")

        if isinstance(blob_name, str):
            blobs = self.sampler.blobs
            pname = parname.strip()
        else:
            pname = parname.strip()
            indv = blob_name.index(pname)
            blobs = self.sampler.blobs[:,indv]

        # Unpack samples: lower, upper 1, 2 sigma
        limits_percentile = np.percentile(blobs, [15.865, 84.135], axis=0)

        limits = fit_utils.shortest_span_bounds(blobs, percentile=0.6827)

        # --------------------------------------------
        # Save best-fit results in the BayesianFitResult instance
        self.__dict__['bestfit_{}'.format(pname)] = bestfit
        self.__dict__['bestfit_{}_l68_err'.format(pname)] = bestfit - limits[0]
        self.__dict__['bestfit_{}_u68_err'.format(pname)] = limits[1] - bestfit


        self.__dict__['bestfit_{}_l68_err_percentile'.format(pname)] = bestfit - limits_percentile[0]
        self.__dict__['bestfit_{}_u68_err_percentile'.format(pname)] = limits_percentile[1] - bestfit


    def analyze_dm_posterior_dist(self, gal=None, blob_name=None):
        """
        Default analysis of posterior distributions of fDM from Bayesian fitting:
            look at marginalized posterior distributions, and
            extract the best-fit value (peak of KDE), and extract the +- 1 sigma uncertainty bounds
            (eg, the 16%/84% distribution of posteriors)

        """
        fdm_param_bestfit = gal.model.get_dm_frac_r_ap()
        self.analyze_blob_posterior_dist(bestfit=fdm_param_bestfit, parname='fdm', blob_name=blob_name)

    def analyze_mvirial_posterior_dist(self, gal=None, blob_name=None):
        mvirial_param_bestfit = gal.model.get_mvirial()
        self.analyze_blob_posterior_dist(bestfit=mvirial_param_bestfit, parname='mvirial', blob_name=blob_name)

    def analyze_alpha_posterior_dist(self, gal=None, blob_name=None):
        alpha_param_bestfit = gal.model.get_halo_alpha()
        self.analyze_blob_posterior_dist(bestfit=alpha_param_bestfit, parname='alpha', blob_name=blob_name)

    def analyze_rb_posterior_dist(self, gal=None, blob_name=None):
        rb_param_bestfit = gal.model.get_halo_rb()
        self.analyze_blob_posterior_dist(bestfit=rb_param_bestfit, parname='rb', blob_name=blob_name)


    def get_uncertainty_ellipse(self, namex=None, namey=None, bins=50):
        r"""
        Using component name, get sampler_results chain for param x and y, and estimate joint uncertainty ellipse

        Input:
            name[x,y]:      List: ['flatchain', ind] or ['flatblobs', ind]

        """
        # Try to support backwards compatibility:
        if (namex[0] == 'flatchain'):
            namex[0] = 'samples'
        elif (namex[0] == 'flatblobs'):
            namex[0] = 'blobs'
        if (namey[0] == 'flatchain'):
            namey[0] = 'samples'
        elif (namey[0] == 'flatblobs'):
            namey[0] = 'blobs'
        
        try:
            chain_x = self.__dict__[namex[0]][:,namex[1]]
        except:
            # eg, Single blob value flatblobs
            chain_x = self.__dict__[namex[0]]
        try:
            chain_y = self.__dict__[namey[0]][:,namey[1]]
        except:
            # eg, Single blob value flatblobs
            chain_y = self.__dict__[namey[0]]

        PA, stddev_x, stddev_y  = dpy_utils.fit_uncertainty_ellipse(chain_x, chain_y, bins=bins)
        return PA, stddev_x, stddev_y


    def save_chain_ascii(self, filename=None, overwrite=False):
        # NOTE: for nested sampling / other methods with weights, 
        #       this only saves the *weighted*, redrawn samples & blobs to file:

        # Check for existing file:
        if (not overwrite) and (filename is not None):
            if os.path.isfile(filename):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
                return None
        if filename is not None:
            try:
                _ = self.sampler_results['blobs']
                blobset = True
            except:
                blobset = False

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
                for i in range(self.sampler.samples.shape[0]):
                    datstr = '  '.join(map(str, self.sampler.samples[i,:]))
                    if blobset:
                        if isinstance(self.blob_name, str):
                            datstr += '  {}'.format(self.sampler.blobs[i])
                        else:
                            for k in range(len(self.blob_name)):
                                datstr += '  {}'.format(self.sampler.blobs[i,k])

                    f.write(datstr+'\n')



class BayesianSampler(object):
    """
    Class to hold the basic attributes of a Bayesian sample.
    Uses a unified syntax, allowing results from eg, emcee and dynesty, 
    to be moved to a single structure.
    """
    def __init__(self, samples=None, blobs=None, weights=None,
                 samples_unweighted=None, blobs_unweighted=None):

        self.samples = samples
        self.blobs = blobs
        self.weights = weights

        self.samples_unweighted = samples_unweighted
        self.blobs_unweighted = blobs_unweighted

    # Backwards compatibility: implement a "dict-like" calling method
    def __getitem__(self, name_in):
        # Old MCMC: Results.sampler['flatchain', 'flatblobs', 'chain', 'blobs']
        # maps to samples, blobs, ....

        if name_in in ['flatchain', 'flatblobs', 'chain', 'blobs']:
            if (name_in == 'flatchain'):
                name = 'samples'
            elif (name_in == 'flatblobs'):
                name = 'blobs'
            elif (name_in in ['chain', 'blobs']):
                raise ValueError("Can't access 'chain' or 'blobs' from the new, generalized sampler class")
        else:
            name = name_in

        item = getattr(self, name, None)
        if item is None:
            raise KeyError(name)
        else:
            return item


###############################################################

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
                logger.warning("ndim={} not supported!".format(obs.data.ndim))
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




def chisq_eval(gal):
    """
    Evaluate chi square of model, relative to the data.
    """
    return fit_utils._chisq_generalized(gal, red_chisq=False)


def chisq_red(gal):
    """
    Evaluate reduced chi square of model, relative to the data.
    """
    return fit_utils._chisq_generalized(gal,red_chisq=True)

def chisq_red_per_type(obs, nparams_free, type=None, **kwargs):
    """
    Evaluate reduced chi square of the model velocity/dispersion/flux map/profile
    """
    if type is None:
        raise ValueError("'type' mustu be 'velocity', 'dispersion', or 'flux'!")
    return fit_utils._chisq_general_per_type(obs, type=type, red_chisq=True, nparams_free=nparams_free)




