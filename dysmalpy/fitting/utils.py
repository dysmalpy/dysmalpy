# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Utility functions for fitting

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

## Standard library
import logging

# Third party imports
import os
import numpy as np
import astropy.units as u

from scipy.stats import gaussian_kde
from scipy.optimize import fmin


# Dysmalpy imports:
from dysmalpy.instrument import DoubleBeam, Moffat, GaussianBeam

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)


_bayesian_fitting_methods = ['mcmc', 'nested']

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
                    invnu = 1./ (1.*(np.sum(msk) - gal.model.nparams_free))
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
                    invnu = 1./ (1.*(fac_mask*np.sum(msk) - gal.model.nparams_free))
                else:
                    invnu = 1.

                ####
                chsq_general += (chisq_arr_sum) * invnu * obs.weight

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
                    invnu = 1. / (1. * (np.sum(msk) - gal.model.nparams_free))
                else:
                    invnu = 1.

                chsq_general += chisq_arr.sum() * invnu * obs.weight

            else:
                logger.warning("ndim={} not supported!".format(obs.instrument.ndim))
                raise ValueError

    return chsq_general

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
        chisq_arr_raw_vel = (((vel_dat - vel_mod)/vel_err)**2) * wgt_data
        chisq_arr_sum += chisq_arr_raw_vel.sum()

    if (type.strip().lower() == 'dispersion'):
        fac_mask += 1
        chisq_arr_raw_disp = (((disp_dat - disp_mod)/disp_err)**2) * wgt_data
        chisq_arr_sum += chisq_arr_raw_disp.sum()

    if (type.strip().lower() == 'flux'):
        fac_mask += 1
        chisq_arr_raw_flux = (((flux_dat - flux_mod)/flux_err)**2) * wgt_data
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



def find_peak_gaussian_KDE(flatchain, initval, weights=None):
    """
    Return chain parameters that give peak of the posterior PDF, using KDE.
    """
    try:
        nparams = flatchain.shape[1]
        nrows = nparams
    except:
        nparams = 1
        nrows = 0

    if nrows > 0:
        peakvals = np.zeros(nparams)
        for i in range(nparams):
            kern = gaussian_kde(flatchain[:,i], weights=weights)
            peakvals[i] = fmin(lambda x: -kern(x), initval[i],disp=False)
        return peakvals
    else:

        kern = gaussian_kde(flatchain, weights=weights)
        peakval = fmin(lambda x: -kern(x), initval,disp=False)

        try:
            return peakval[0]
        except:
            return peakval


def find_peak_gaussian_KDE_multiD(flatchain, linked_inds, initval, weights=None):
    """
    Return chain parameters that give peak of the posterior PDF *FOR LINKED PARAMETERS*, using KDE.
    """

    if weights is not None:
        raise ValueError("TEST")

    # nparams = len(linked_inds)
    kern = gaussian_kde(flatchain[:,linked_inds].T, weights=weights)
    peakvals = fmin(lambda x: -kern(x), initval,disp=False)

    return peakvals


def find_multiD_pk_hist(flatchain, linked_inds, nPostBins=50):
    H2, edges = np.histogramdd(flatchain[:,linked_inds], bins=nPostBins)

    wh_pk = np.where(H2 == H2.max())[0][0]

    pk_vals = np.zeros(len(linked_inds))

    for k in range(len(linked_inds)):
        pk_vals[k] = np.average([edges[k][wh_pk], edges[k][wh_pk+1]])

    return pk_vals



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

    # Use gaussian KDE to get bestfit linked:
    bestfit_theta_linked = np.array([])

    for k in range(len(linked_posterior_ind_arr)):
        bestfit_thetas = find_peak_gaussian_KDE_multiD(flatchain, linked_posterior_ind_arr[k],
                guess[linked_posterior_ind_arr[k]])
        if len(bestfit_theta_linked) >= 1:
            bestfit_theta_linked = np.stack(bestfit_theta_linked, np.array([bestfit_thetas]) )
        else:
            bestfit_theta_linked = np.array([bestfit_thetas])


    return bestfit_theta_linked



def get_linked_posterior_indices(nestedResults):
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


        Also if doing single bundle must have:
        linked_posterior_names = [ [ [cmp1, par1], [cmp2, par2] ] ]

    Output:
        linked_posterior_inds = [ joint_bundle1_inds, joint_bundle2_inds ]
        with joint_bundle1_inds = [ ind1, ind2 ], etc

        ex:
            output = [ [ind1, ind2], [ind3, ind4] ]

    """
    linked_posterior_ind_arr = None
    try:
        if nestedResults.linked_posterior_names.strip().lower() == 'all':
            linked_posterior_ind_arr = [range(len(nestedResults.free_param_names))]
    except:
        pass
    if linked_posterior_ind_arr is None:
        free_cmp_param_arr = make_arr_cmp_params(nestedResults)

        linked_posterior_ind_arr = []
        for k in range(len(nestedResults.linked_posterior_names)):
            # Loop over *sets* of linked posteriors:
            # This is an array of len-2 arrays/tuples with cmp, param names
            linked_post_inds = []
            for j in range(len(nestedResults.linked_posterior_names[k])):

                indp = get_param_index(nestedResults, nestedResults.linked_posterior_names[k][j],
                            free_cmp_param_arr=free_cmp_param_arr)
                linked_post_inds.append(indp)

            linked_posterior_ind_arr.append(linked_post_inds)

    return linked_posterior_ind_arr


def get_param_index(nestedResults, param_name, free_cmp_param_arr=None):
    if free_cmp_param_arr is None:
        free_cmp_param_arr = make_arr_cmp_params(nestedResults)

    cmp_param = param_name[0].strip().lower()+':'+param_name[1].strip().lower()

    try:
        whmatch = np.where(free_cmp_param_arr == cmp_param)[0][0]
    except:
        raise ValueError(cmp_param+' component+parameter not found in free parameters of nestedResults')
    return whmatch




############################################################
# UTILITY FUNCTIONS
####################

def make_arr_cmp_params(results):
    arr = np.array([])
    for cmp in results.free_param_names.keys():
        for i in range(len(results.free_param_names[cmp])):
            param = results.free_param_names[cmp][i]
            arr = np.append( arr, cmp.strip().lower()+':'+param.strip().lower() )

    return arr

def setup_oversampled_chisq(gal):
    # Setup for oversampled_chisq:
    for obs_name in gal.observations:
        obs = gal.observations[obs_name]

        if isinstance(obs.instrument.beam, GaussianBeam):
            try:
                PSF_FWHM = obs.instrument.beam.major.value
            except:
                PSF_FWHM = obs.instrument.beam.major
        elif isinstance(obs.instrument.beam, Moffat):
            try:
                PSF_FWHM = obs.instrument.beam.major_fwhm.value
            except:
                PSF_FWHM = obs.instrument.beam.major_fwhm
        elif isinstance(obs.instrument.beam, DoubleBeam):
            try:
                PSF_FWHM = np.max([obs.instrument.beam.beam1.major.value, obs.instrument.beam.beam2.major.value])
            except:
                PSF_FWHM = np.max([obs.instrument.beam.beam1.major, obs.instrument.beam.beam2.major])

        if obs.instrument.ndim == 1:
            rarrtmp = obs.data.rarr.copy()
            rarrtmp.sort()
            spacing_avg = np.abs(np.average(rarrtmp[1:]-rarrtmp[:-1]))
            obs.data.oversample_factor_chisq = PSF_FWHM /spacing_avg

        elif obs.instrument.ndim == 2:
            obs.data.oversample_factor_chisq = (PSF_FWHM / obs.instrument.pixscale.value)**2

        elif obs.instrument.ndim == 3:
            spec_step = obs.instrument.spec_step.to(u.km/u.s).value
            LSF_FWHM = obs.instrument.lsf.dispersion.to(u.km/u.s).value * (2.*np.sqrt(2.*np.log(2.)))
            obs.data.oversample_factor_chisq = (LSF_FWHM / spec_step) * (PSF_FWHM / obs.instrument.pixscale.value)**2

    return gal

def find_shortest_conf_interval(xarr, percentile_frac):
    # Canonical 1sigma: 0.6827
    xsort = np.sort(xarr)

    N = len(xarr)
    i_max = np.int64(np.round(percentile_frac*N))
    len_arr = xsort[i_max:] - xsort[0:N-i_max]

    argmin = np.argmin(len_arr)
    l_val, u_val = xsort[argmin], xsort[argmin+i_max-1]

    return l_val, u_val

def shortest_span_bounds(arr, percentile=0.6827):
    if len(arr.shape) == 1:
        limits = find_shortest_conf_interval(arr, percentile)
    else:
        limits = np.ones((2, arr.shape[1]))
        for j in range(arr.shape[1]):
            limits[:, j] = find_shortest_conf_interval(arr[:,j], percentile)

    return limits



def _check_existing_files_overwrite(output_options, fit_type=None, fitter=None):
    # ---------------------------------------------------
    # Check for existing files if overwrite=False:
    if (not output_options.overwrite):

        fnames = []
        fnames_opt = [ output_options.f_model, output_options.f_vcirc_ascii,
                       output_options.f_mass_ascii, output_options.f_results,
                       output_options.f_sampler_results, 
                       output_options.f_plot_bestfit ]

        if (fit_type.lower() in _bayesian_fitting_methods):
            fnames_ext = [output_options.f_plot_trace, 
                          output_options.f_plot_param_corner, 
                          output_options.f_chain_ascii]
            for fn in fnames_ext:
                fnames_opt.append(fn)

        if (fit_type.lower() == 'mcmc'):
            fnames_ext = [output_options.f_plot_trace_burnin]
            for fn in fnames_ext:
                fnames_opt.append(fn)
        elif (fit_type.lower() == 'nested'):
            fnames_ext = [output_options.f_checkpoint,
                          output_options.f_plot_run]
            for fn in fnames_ext:
                fnames_opt.append(fn)

        for fname in fnames_opt:
            if fname is not None:
                fnames.append(fname)

        file_bundle_names = ['f_model_bestfit', 'f_vel_ascii', 'f_bestfit_cube']
        for fbunname in file_bundle_names:
            for obsn in output_options.__dict__[fbunname]:
                if output_options.__dict__[fbunname][obsn] is not None:
                    fnames.append(output_options.__dict__[fbunname][obsn])


        for fname in fnames:
            if fname is not None:
                if os.path.isfile(fname):
                    logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(output_options.overwrite, fname))

        # Return early if it won't save the results, sampler:
        if output_options.f_results is not None:
            if os.path.isfile(output_options.f_results):
                msg = "overwrite={}, and 'f_results' won't be saved,".format(output_options.overwrite)
                msg += " so the fit will not be saved.\n Specify new outfile or delete old files."
                logger.warning(msg)
                return None


    else:
        # Overwrite=True: remove old file versions

        if (fit_type.lower() in _bayesian_fitting_methods):
            fnames_ext = [output_options.f_plot_trace, 
                          output_options.f_plot_param_corner, 
                          output_options.f_chain_ascii]
            for fn in fnames_ext:
                if os.path.isfile(fn): os.remove(fn)
                
        if (fit_type.lower() == 'mcmc'):
            fnames_ext = [output_options.f_plot_trace_burnin]
            for fn in fnames_ext:
                if os.path.isfile(fn): os.remove(fn)
                
        elif (fit_type.lower() == 'nested'):
            fnames_ext = [output_options.f_checkpoint,
                          output_options.f_plot_run]
            for fn in fnames_ext:
                if os.path.isfile(fn): os.remove(fn)
                

    # ---------------------------------------------------



