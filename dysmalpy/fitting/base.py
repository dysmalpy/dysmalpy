# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Base classes / methods for fitting / fit results

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

## Standard library
import logging
import abc
import six

# DYSMALPY code
from dysmalpy import plotting
from dysmalpy import config
from dysmalpy import utils_io as dpy_utils_io
from dysmalpy.data_io import ensure_dir, load_pickle, dump_pickle

# Third party imports
import os
import numpy as np
from collections import OrderedDict
import astropy.units as u
import copy

__all__ =  ['FitResults',
            'chisq_red', 'chisq_eval', 'chisq_red_per_type']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


class FitResults(object):
    """
    General class to hold the results of any fitting
    """

    def __init__(self,
                 model=None,
                 f_plot_bestfit=None,
                 f_plot_spaxel=None,
                 f_plot_aperture=None,
                 f_plot_channel=None,
                 f_results=None,
                 fit_method=None):

        self.bestfit_parameters = None
        self.bestfit_parameters_err = None
        self.bestfit_redchisq = None
        self._fixed = None

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

        self.f_plot_bestfit = f_plot_bestfit

        # Specific to 3D: 'f_plot_spaxel', 'f_plot_aperture', 'f_plot_channel'
        self.f_plot_spaxel = f_plot_spaxel
        self.f_plot_aperture = f_plot_aperture
        self.f_plot_channel = f_plot_channel

        self.f_results = f_results
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

        self.chain_param_names = make_arr_cmp_params(self)

    def save_results(self, filename=None, overwrite=False):
        # Check for existing file:
        if (not overwrite) and (filename is not None):
            if os.path.isfile(filename):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
                return None

        if filename is not None:
            dump_pickle(self, filename=filename, overwrite=overwrite)  # Save FitResults class to a pickle file

    def save_bestfit_vel_ascii(self, gal, filename=None,
                               model_aperture_r=config._model_aperture_r, overwrite=False):
        if filename is not None:
            try:
                r_ap = model_aperture_r(self)
            except:
                r_ap = 10./3.
            rmax = np.max([3. * r_ap, 10.])
            stepsize = 0.1  # stepsize 0.1 kpc
            r = np.arange(0., rmax + stepsize, stepsize)

            gal.model.write_vrot_vcirc_file(r=r, filename=filename, overwrite=overwrite)

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

    def plot_bestfit(self, gal, fitvelocity=True, fitdispersion=True,
                     fitflux=False, fileout=None, overwrite=False, **kwargs_galmodel):
        """Plot/replot the bestfit for the MCMC fitting"""
        # Specific to 3D: 'f_plot_spaxel', 'f_plot_aperture', 'f_plot_channel'
        # f_plot_spaxel=None,
        # f_plot_aperture=None,
        # f_plot_channel=None,
        #if fileout is None:
        #    fileout = self.f_plot_bestfit
        # Check for existing file:
        if (not overwrite) and (fileout is not None):
            if os.path.isfile(fileout):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fileout))
                return None
        plotting.plot_bestfit(self, gal, fitvelocity=fitvelocity,
                              fitdispersion=fitdispersion, fitflux=fitflux,
                              fileout=fileout, overwrite=overwrite, **kwargs_galmodel)

    def reload_results(self, filename=None):
        """Reload MCMC results saved earlier: the whole object"""
        if filename is None:
            filename = self.f_results
        resultsSaved = load_pickle(filename)
        for key in resultsSaved.__dict__.keys():
            try:
                self.__dict__[key] = resultsSaved.__dict__[key]
            except:
                pass

    def results_report(self, gal=None, filename=None, params=None,
                    report_type='pretty', overwrite=False, **kwargs):
        """Return a result report string, or save to file.
           report_type = 'pretty':   More human-readable
                       = 'machine':  Machine-readable ascii table (though with mixed column types)

           **kwargs: can pass other setting values: eg zcalc_truncate.
        """

        report = dpy_utils_io.create_results_report(gal, self, report_type=report_type,
                        params=params, **kwargs)

        if filename is not None:
            with open(filename, 'w') as f:
                f.write(report)
        else:
            return report


###############################################################


def _chisq_generalized(gal, red_chisq=None):

    if red_chisq is None:
        raise ValueError("'red_chisq' must be True or False!")

    chsq_general = 0.0

    for obs_name in gal.observations:
        obs = gal.observations[obs_name]

        if obs.fit_options.fit:

            # 3D observation
            if obs.instrument.ndim == 3:
                # Will have problem with vel shift: data, model won't match...

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
                chisq_arr_raw = (((dat - mod)/err)**2) * wgt_data
                if red_chisq:
                    if gal.model.nparams_free > np.sum(msk) :
                        raise ValueError("More free parameters than data points!")
                    invnu = 1./ (1.*(np.sum(msk) - gal.model.nparams_free)) * obs.weight
                else:
                    invnu = 1.
                chsq_general += chisq_arr_raw.sum() * invnu * obs.weight

            elif ((obs.instrument.ndim == 1) or (obs.instrument.ndim ==2)):

                if obs.fit_options.fit_velocity:
                    #msk = obs.data.mask
                    if hasattr(obs.data, 'mask_velocity'):
                        if obs.data.mask_velocity is not None:
                            msk = obs.data.mask_velocity
                        else:
                            msk = obs.data.mask
                    else:
                        msk = obs.data.mask

                    vel_dat = obs.data.data['velocity'][msk]
                    vel_mod = obs.model_data.data['velocity'][msk]
                    vel_err = obs.data.error['velocity'][msk]

                if obs.fit_options.fit_dispersion:
                    if hasattr(obs.data, 'mask_vel_disp'):
                        if obs.data.mask_vel_disp is not None:
                            msk = obs.data.mask_vel_disp
                        else:
                            msk = obs.data.mask
                    else:
                        msk = obs.data.mask
                    disp_dat = obs.data.data['dispersion'][msk]
                    disp_mod = obs.model_data.data['dispersion'][msk]
                    disp_err = obs.data.error['dispersion'][msk]

                    # Correct model for instrument dispersion if the data is instrument corrected:
                    if 'inst_corr' in obs.data.data.keys():
                        if obs.data.data['inst_corr']:
                            disp_mod = np.sqrt(disp_mod**2 -
                                               obs.instrument.lsf.dispersion.to(u.km/u.s).value**2)
                            disp_mod[~np.isfinite(disp_mod)] = 0   # Set the dispersion to zero when its below
                                                                   # below the instrumental dispersion

                if obs.fit_options.fit_flux:
                    msk = obs.data.mask
                    flux_dat = obs.data.data['flux'][msk]
                    flux_mod = obs.model_data.data['flux'][msk]
                    try:
                        flux_err = obs.data.error['flux'][msk]
                    except:
                        flux_err = 0.1*obs.data.data['flux'][msk] # PLACEHOLDER



                # Weights:
                wgt_data = 1.
                if hasattr(obs.data, 'weight'):
                    if obs.data.weight is not None:
                        wgt_data = obs.data.weight[msk]

                #####
                fac_mask = 0
                chisq_arr_sum = 0

                if obs.fit_options.fit_velocity:
                    fac_mask += 1
                    ### Data includes velocity
                    # Includes velocity shift
                    chisq_arr_raw_vel = (((vel_dat - vel_mod)/vel_err)**2) * wgt_data
                    chisq_arr_sum += chisq_arr_raw_vel.sum()

                if obs.fit_options.fit_dispersion:
                    fac_mask += 1
                    chisq_arr_raw_disp = (((disp_dat - disp_mod)/disp_err)**2) * wgt_data
                    chisq_arr_sum += chisq_arr_raw_disp.sum()

                if obs.fit_options.fit_flux:
                    fac_mask += 1
                    chisq_arr_raw_flux = (((flux_dat - flux_mod)/flux_err)**2) * wgt_data
                    chisq_arr_sum += chisq_arr_raw_flux.sum()

                ####
                if red_chisq:
                    if gal.model.nparams_free > fac_mask*np.sum(msk) :
                        raise ValueError("More free parameters than data points!")
                    invnu = 1./ (1.*(fac_mask*np.sum(msk) - gal.model.nparams_free)) * obs.weight
                else:
                    invnu = 1. * obs.weight

                ####
                chsq_general += (chisq_arr_sum) * invnu

            elif obs.instrument.ndim == 0:

                msk = obs.data.mask
                data = obs.data.data
                mod = obs.model_data.data
                err = obs.data.error

                # Weights:
                wgt_data = 1.
                if hasattr(obs.data, 'weight'):
                    if obs.data.weight is not None:
                        wgt_data = obs.data.weight

                chisq_arr = (((data - mod)/err)**2) * wgt_data
                if red_chisq:
                    if gal.model.nparams_free > np.sum(msk):
                        raise ValueError("More free parameters than data points!")
                    invnu = 1. / (1. * (np.sum(msk) - gal.model.nparams_free)) * obs.weight
                else:
                    invnu = 1. * obs.weight

                chsq_general += chisq_arr.sum() * invnu

            else:
                logger.warning("ndim={} not supported!".format(obs.instrument.ndim))
                raise ValueError

    return chsq_general


def chisq_eval(gal):
    """
    Evaluate chi square of model, relative to the data.
    """
    return _chisq_generalized(gal, red_chisq=False)


def chisq_red(gal):
    """
    Evaluate reduced chi square of model, relative to the data.
    """
    return _chisq_generalized(gal,red_chisq=True)


def _chisq_general_per_type(obs, type=None, red_chisq=True, nparams_free=None, **kwargs):
    """
    Evaluate reduced chi square of model for one specific map/profile
    (i.e., flux/velocity/dispersion), relative to the data.
    """
    # type = 'velocity', 'disperesion', or 'flux'

    if ((obs.data.ndim != 1) & (obs.data.ndim != 2)):
        msg = "_chisq_general_per_type() can only be called when\n"
        msg += "obs.data.ndim = 1 or 2!"
        raise ValueError(msg)


    if (type.strip().lower() == 'velocity'):
        #msk = obs.data.mask
        if hasattr(obs.data, 'mask_velocity'):
            if obs.data.mask_velocity is not None:
                msk = obs.data.mask_velocity
            else:
                msk = obs.data.mask
        else:
            msk = obs.data.mask

        vel_dat = obs.data.data['velocity'][msk]
        vel_mod = obs.model_data.data['velocity'][msk]
        vel_err = obs.data.error['velocity'][msk]

    if (type.strip().lower() == 'dispersion'):
        if hasattr(obs.data, 'mask_vel_disp'):
            if obs.data.mask_vel_disp is not None:
                msk = obs.data.mask_vel_disp
            else:
                msk = obs.data.mask
        else:
            msk = obs.data.mask
        disp_dat = obs.data.data['dispersion'][msk]
        disp_mod = obs.model_data.data['dispersion'][msk]
        disp_err = obs.data.error['dispersion'][msk]

        # Correct model for instrument dispersion if the data is instrument corrected:
        if 'inst_corr' in obs.data.data.keys():
            if obs.data.data['inst_corr']:
                disp_mod = np.sqrt(disp_mod**2 -
                                   obs.instrument.lsf.dispersion.to(u.km/u.s).value**2)
                disp_mod[~np.isfinite(disp_mod)] = 0   # Set the dispersion to zero when its below
                                                       # below the instrumental dispersion

    if (type.strip().lower() == 'flux'):
        msk = obs.data.mask
        flux_dat = obs.data.data['flux'][msk]
        flux_mod = obs.model_data.data['flux'][msk]
        try:
            flux_err = obs.data.error['flux'][msk]
        except:
            flux_err = 0.1*obs.data.data['flux'][msk] # PLACEHOLDER

    # Weights:
    wgt_data = 1.
    if hasattr(obs.data, 'weight'):
        if obs.data.weight is not None:
            wgt_data = obs.data.weight[msk]

    #####
    fac_mask = 0
    chisq_arr_sum = 0

    if (type.strip().lower() == 'velocity'):
        fac_mask += 1
        ### Data includes velocity
        # Includes velocity shift
        chisq_arr_raw_vel = (((vel_dat - vel_mod)/vel_err)**2) * wgt
        chisq_arr_sum += chisq_arr_raw_vel.sum()

    if (type.strip().lower() == 'dispersion'):
        fac_mask += 1
        chisq_arr_raw_disp = (((disp_dat - disp_mod)/disp_err)**2) * wgt
        chisq_arr_sum += chisq_arr_raw_disp.sum()

    if (type.strip().lower() == 'flux'):
        fac_mask += 1
        chisq_arr_raw_flux = (((flux_dat - flux_mod)/flux_err)**2) * wgt
        chisq_arr_sum += chisq_arr_raw_flux.sum()

    ####
    if red_chisq:
        if nparams_free is None:
            raise ValueError("If `red_chisq` = TRUE, then must set `nparams_free`.")
        if nparams_free > fac_mask*np.sum(msk) :
            raise ValueError("More free parameters than data points!")
        invnu = 1./ (1.*(fac_mask*np.sum(msk) - nparams_free)) * obs.weight
    else:
        invnu = 1. * obs.weight

    ####
    chsq_general = (chisq_arr_sum) * invnu

    return chsq_general

def chisq_red_per_type(obs, nparams_free, type=None, **kwargs):
    """
    Evaluate reduced chi square of the model velocity/dispersion/flux map/profile
    """
    if type is None:
        raise ValueError("'type' mustu be 'velocity', 'dispersion', or 'flux'!")
    return _chisq_general_per_type(obs, type=type, red_chisq=True, nparams_free=nparams_free)


def make_arr_cmp_params(results):
    arr = np.array([])
    for cmp in results.free_param_names.keys():
        for i in six.moves.xrange(len(results.free_param_names[cmp])):
            param = results.free_param_names[cmp][i]
            arr = np.append( arr, cmp.strip().lower()+':'+param.strip().lower() )

    return arr
