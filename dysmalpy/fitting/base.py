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
from dysmalpy import utils_io as dpy_utils_io
from dysmalpy.data_io import ensure_dir, load_pickle, dump_pickle

# Third party imports
import os
import numpy as np
from collections import OrderedDict
import astropy.units as u
import dill as _pickle
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

    def save_bestfit_vel_ascii(self, gal, filename=None, model_key_re=['disk+bulge', 'r_eff_disk'], overwrite=False):
        if filename is not None:
            try:
                # RE needs to be in kpc
                comp = gal.model.components.__getitem__(model_key_re[0])
                param_i = comp.param_names.index(model_key_re[1])
                r_eff = comp.parameters[param_i]
            except:
                r_eff = 10. / 3.
            rmax = np.max([3. * r_eff, 10.])
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


def _chisq_generalized(gal, fitvelocity=True, fitdispersion=True, fitflux=False,
              use_weights=False, red_chisq=None, **kwargs):
    if red_chisq is None:
        raise ValueError("'red_chisq' must be True or False!")
    if gal.data.ndim == 3:
        # Will have problem with vel shift: data, model won't match...

        msk = gal.data.mask
        dat = gal.data.data.unmasked_data[:].value[msk]
        mod = gal.model_data.data.unmasked_data[:].value[msk]
        err = gal.data.error.unmasked_data[:].value[msk]

        # Weights:
        wgt = 1.
        if use_weights:
            if hasattr(gal.data, 'weight'):
                if gal.data.weight is not None:
                    wgt = gal.data.weight[msk]


        # Artificially mask zero errors which are masked
        #err[((err==0) & (msk==0))] = 99.
        chisq_arr_raw = (((dat - mod)/err)**2) * wgt
        if red_chisq:
            if gal.model.nparams_free > np.sum(msk) :
                raise ValueError("More free parameters than data points!")
            invnu = 1./ (1.*(np.sum(msk) - gal.model.nparams_free))
        else:
            invnu = 1.
        chsq_general = chisq_arr_raw.sum() * invnu

    elif ((gal.data.ndim == 1) or (gal.data.ndim ==2)):

        if fitvelocity:
            #msk = gal.data.mask
            if hasattr(gal.data, 'mask_velocity'):
                if gal.data.mask_velocity is not None:
                    msk = gal.data.mask_velocity
                else:
                    msk = gal.data.mask
            else:
                msk = gal.data.mask

            vel_dat = gal.data.data['velocity'][msk]
            vel_mod = gal.model_data.data['velocity'][msk]
            vel_err = gal.data.error['velocity'][msk]

        if fitdispersion:
            if hasattr(gal.data, 'mask_vel_disp'):
                if gal.data.mask_vel_disp is not None:
                    msk = gal.data.mask_vel_disp
                else:
                    msk = gal.data.mask
            else:
                msk = gal.data.mask
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

        if fitflux:
            msk = gal.data.mask
            flux_dat = gal.data.data['flux'][msk]
            flux_mod = gal.model_data.data['flux'][msk]
            try:
                flux_err = gal.data.error['flux'][msk]
            except:
                flux_err = 0.1*gal.data.data['flux'][msk] # PLACEHOLDER



        # Weights:
        wgt = 1.
        if use_weights:
            if hasattr(gal.data, 'weight'):
                if gal.data.weight is not None:
                    wgt = gal.data.weight[msk]


        #####
        fac_mask = 0
        chisq_arr_sum = 0

        if fitvelocity:
            fac_mask += 1
            ### Data includes velocity
            # Includes velocity shift
            chisq_arr_raw_vel = (((vel_dat - vel_mod)/vel_err)**2) * wgt
            chisq_arr_sum += chisq_arr_raw_vel.sum()

        if fitdispersion:
            fac_mask += 1
            chisq_arr_raw_disp = (((disp_dat - disp_mod)/disp_err)**2) * wgt
            chisq_arr_sum += chisq_arr_raw_disp.sum()

        if fitflux:
            fac_mask += 1
            chisq_arr_raw_flux = (((flux_dat - flux_mod)/flux_err)**2) * wgt
            chisq_arr_sum += chisq_arr_raw_flux.sum()

        ####
        if red_chisq:
            if gal.model.nparams_free > fac_mask*np.sum(msk) :
                raise ValueError("More free parameters than data points!")
            invnu = 1./ (1.*(fac_mask*np.sum(msk) - gal.model.nparams_free))
        else:
            invnu = 1.

        ####
        chsq_general = (chisq_arr_sum) * invnu

    elif gal.data.ndim == 0:

        msk = gal.data.mask
        data = gal.data.data
        mod = gal.model_data.data
        err = gal.data.error

        # Weights:
        wgt = 1.
        if use_weights:
            if hasattr(gal.data, 'weight'):
                if gal.data.weight is not None:
                    wgt = gal.data.weight

        chisq_arr = (((data - mod)/err)**2) * wgt
        if red_chisq:
            if gal.model.nparams_free > np.sum(msk):
                raise ValueError("More free parameters than data points!")
            invnu = 1. / (1. * (np.sum(msk) - gal.model.nparams_free))
        else:
            invnu = 1.

        chsq_general = chisq_arr.sum() * invnu

    else:
        logger.warning("ndim={} not supported!".format(gal.data.ndim))
        raise ValueError

    return chsq_general


def chisq_eval(gal, fitvelocity=True, fitdispersion=True, fitflux=False,
               use_weights=False, **kwargs):
    """
    Evaluate chi square of model, relative to the data.
    """
    return _chisq_generalized(gal, fitvelocity=fitvelocity, fitdispersion=fitdispersion,
                  fitflux=fitflux, use_weights=use_weights, red_chisq=False)


def chisq_red(gal, fitvelocity=True, fitdispersion=True, fitflux=False,
              use_weights=False, **kwargs):
    """
    Evaluate reduced chi square of model, relative to the data.
    """
    return _chisq_generalized(gal, fitvelocity=fitvelocity, fitdispersion=fitdispersion,
                  fitflux=fitflux, use_weights=use_weights, red_chisq=True)


def _chisq_general_per_type(gal, type=None, use_weights=False, red_chisq=True, **kwargs):
    """
    Evaluate reduced chi square of model for one specific map/profile
    (i.e., flux/velocity/dispersion), relative to the data.
    """
    # type = 'velocity', 'disperesion', or 'flux'

    if ((gal.data.ndim != 1) & (gal.data.ndim != 2)):
        msg = "_chisq_general_per_type() can only be called when\n"
        msg += "gal.data.ndim = 1 or 2!"
        raise ValueError(msg)


    if (type.strip().lower() == 'velocity'):
        #msk = gal.data.mask
        if hasattr(gal.data, 'mask_velocity'):
            if gal.data.mask_velocity is not None:
                msk = gal.data.mask_velocity
            else:
                msk = gal.data.mask
        else:
            msk = gal.data.mask

        vel_dat = gal.data.data['velocity'][msk]
        vel_mod = gal.model_data.data['velocity'][msk]
        vel_err = gal.data.error['velocity'][msk]

    if (type.strip().lower() == 'dispersion'):
        if hasattr(gal.data, 'mask_vel_disp'):
            if gal.data.mask_vel_disp is not None:
                msk = gal.data.mask_vel_disp
            else:
                msk = gal.data.mask
        else:
            msk = gal.data.mask
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

    if (type.strip().lower() == 'flux'):
        msk = gal.data.mask
        flux_dat = gal.data.data['flux'][msk]
        flux_mod = gal.model_data.data['flux'][msk]
        try:
            flux_err = gal.data.error['flux'][msk]
        except:
            flux_err = 0.1*gal.data.data['flux'][msk] # PLACEHOLDER

    # Weights:
    wgt = 1.
    if use_weights:
        if hasattr(gal.data, 'weight'):
            if gal.data.weight is not None:
                wgt = gal.data.weight[msk]


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
        if gal.model.nparams_free > fac_mask*np.sum(msk) :
            raise ValueError("More free parameters than data points!")
        invnu = 1./ (1.*(fac_mask*np.sum(msk) - gal.model.nparams_free))
    else:
        invnu = 1.

    ####
    chsq_general = (chisq_arr_sum) * invnu

    return chsq_general

def chisq_red_per_type(gal, type=None, use_weights=False, **kwargs):
    """
    Evaluate reduced chi square of the model velocity/dispersion/flux map/profile
    """
    if type is None:
        raise ValueError("'type' mustu be 'velocity', 'dispersion', or 'flux'!")
    return _chisq_general_per_type(gal, type=type, use_weights=use_weights, red_chisq=True)


def make_arr_cmp_params(results):
    arr = np.array([])
    for cmp in results.free_param_names.keys():
        for i in six.moves.xrange(len(results.free_param_names[cmp])):
            param = results.free_param_names[cmp][i]
            arr = np.append( arr, cmp.strip().lower()+':'+param.strip().lower() )

    return arr
