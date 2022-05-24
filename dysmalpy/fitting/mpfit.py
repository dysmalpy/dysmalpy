# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Classes and functions for fitting DYSMALPY kinematic models
#   to the observed data using MPFIT
#

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

## Standard library
import logging

# DYSMALPY code
from dysmalpy.data_io import ensure_dir, load_pickle, dump_pickle
from dysmalpy import plotting
from dysmalpy import galaxy
from dysmalpy import config
from dysmalpy import utils_io as dpy_utils_io

# Local imports:
from .base import FitResults, chisq_eval, chisq_red, chisq_red_per_type

# Third party imports
import os
import numpy as np
import astropy.units as u
import copy
from dysmalpy.extern.mpfit import mpfit

import time, datetime


__all__ = ['fit_mpfit', 'MPFITResults']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


def fit_mpfit(gal, **kwargs):
    """
    Fit observed kinematics using MPFIT and a DYSMALPY model set.
    """

    #config_c_m_data = config.Config_create_model_data(**kwargs)
    #config_sim_cube = config.Config_simulate_cube(**kwargs)
    #kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}
    logger.warning("MAKE SURE PARAMS GET INTO obs.mod_options, ETC!!")



    config_fit = config.Config_fit_mpfit(**kwargs)
    kwargs_fit = config_fit.dict
    logger.warning("RETHINK FIT OPTIONS CONFIG CLASSES!!")


    # Check the FOV is large enough to cover the data output:
    dpy_utils_io._check_data_inst_FOV_compatibility(gal)

    # Create output directory
    if len(kwargs_fit['outdir']) > 0:
        if kwargs_fit['outdir'][-1] != '/': kwargs_fit['outdir'] += '/'
    ensure_dir(kwargs_fit['outdir'])

    # If the output filenames aren't defined: use default output filenames

    if kwargs_fit['save_model'] and (kwargs_fit['f_model'] is None):
        kwargs_fit['f_model'] = kwargs_fit['outdir']+'galaxy_model.pickle'
    if kwargs_fit['save_bestfit_cube'] and (kwargs_fit['f_cube'] is None):
        kwargs_fit['f_cube'] = kwargs_fit['outdir']+'mpfit_bestfit_cube.fits'
    if kwargs_fit['f_plot_bestfit'] is None:
        kwargs_fit['f_plot_bestfit'] = kwargs_fit['outdir'] + 'mpfit_best_fit.{}'.format(kwargs_fit['plot_type'])


    # TEMPORARY KLUDGE
    obs_names = list(gal.observations.keys())
    obs = gal.observations[obs_names[0]]
    logger.warning("TEMP HORRIBLE FNAME KLUDGE: ONLY WORKS FOR 1 OBS")

    # Specific to 3D: 'f_plot_spaxel', 'f_plot_aperture', 'f_plot_channel'
    if (obs.data.ndim == 3) & (kwargs_fit['f_plot_spaxel'] is None):
        kwargs_fit['f_plot_spaxel'] = kwargs_fit['outdir']+'mpfit_best_fit_spaxels.{}'.format(kwargs_fit['plot_type'])
    if (obs.data.ndim == 3) & (kwargs_fit['f_plot_aperture'] is None):
        kwargs_fit['f_plot_aperture'] = kwargs_fit['outdir']+'mpfit_best_fit_apertures.{}'.format(kwargs_fit['plot_type'])
    if (obs.data.ndim == 3) & (kwargs_fit['f_plot_channel'] is None):
        kwargs_fit['f_plot_channel'] = kwargs_fit['outdir']+'mpfit_best_fit_channel.{}'.format(kwargs_fit['plot_type'])

    if kwargs_fit['save_results'] & (kwargs_fit['f_results'] is None):
        kwargs_fit['f_results'] = kwargs_fit['outdir'] + 'mpfit_results.pickle'
    if kwargs_fit['save_vel_ascii'] & (kwargs_fit['f_vel_ascii'] is None):
        kwargs_fit['f_vel_ascii'] = kwargs_fit['outdir'] + 'galaxy_bestfit_vel_profile.dat'
    if kwargs_fit['save_vel_ascii'] & (kwargs_fit['f_vcirc_ascii'] is None):
        kwargs_fit['f_vcirc_ascii'] = kwargs_fit['outdir']+'galaxy_bestfit_vcirc.dat'
    if kwargs_fit['save_vel_ascii'] & (kwargs_fit['f_mass_ascii'] is None):
        kwargs_fit['f_mass_ascii'] = kwargs_fit['outdir']+'galaxy_bestfit_menc.dat'

    if kwargs_fit['save_model_bestfit'] & (kwargs_fit['f_model_bestfit'] is None):
        if obs.data.ndim == 1:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-1dplots.txt'
        elif obs.data.ndim == 2:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-velmaps.fits'
        elif obs.data.ndim == 3:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-cube.fits'
        elif obs.data.ndim == 0:
            kwargs_fit['f_model_bestfit'] = kwargs_fit['outdir']+'galaxy_out-0d.txt'

    # ---------------------------------------------------
    # Null filenames if not saving:
    save_keys = ['save_model', 'save_bestfit_cube',  'save_results',
                'save_vel_ascii', 'save_model_bestfit']
    fname_keys = ['f_model', 'f_cube', 'f_results',
                    'f_vel_ascii', 'f_vcirc_ascii', 'f_mass_ascii', 'f_model_bestfit']
    for sk, fk in zip(save_keys, fname_keys):
        if not kwargs_fit[sk]:
            kwargs_fit[fk] = None

    if not kwargs_fit['do_plotting']:
        fname_keys = ['f_plot_bestfit']
        # Specific to 3D: 'f_plot_spaxel', 'f_plot_aperture', 'f_plot_channel'
        if (obs.data.ndim == 3):
            for kw in ['f_plot_spaxel', 'f_plot_aperture', 'f_plot_channel']:
                fname_keys.append(kw)
        for fk in fname_keys:
            kwargs_fit[fk] = None


    # ---------------------------------------------------
    # Check for existing files if overwrite=False:
    if (not kwargs_fit['overwrite']):
        fnames = []
        fnames_opt = [ kwargs_fit['f_plot_bestfit'], kwargs_fit['f_results'],
                        kwargs_fit['f_vel_ascii'], kwargs_fit['f_vcirc_ascii'],
                        kwargs_fit['f_mass_ascii'],
                        kwargs_fit['f_model'], kwargs_fit['f_cube'] ]
        for fname in fnames_opt:
            if fname is not None:
                fnames.append(fname)

        for fname in fnames:
            if fname is not None:
                if os.path.isfile(fname):
                    logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(kwargs_fit['overwrite'], fname))

        # Return early if it won't save the results, sampler:
        if kwargs_fit['f_results'] is not None:
            if os.path.isfile(kwargs_fit['f_results']):
                msg = "overwrite={}, and 'f_results' won't be saved,".format(kwargs_fit['overwrite'])
                msg += " so the fit will not be saved.\n Specify new outfile or delete old files."
                logger.warning(msg)
                return None

    # ---------------------------------------------------

    # Setup file redirect logging:
    if kwargs_fit['f_log'] is not None:
        loggerfile = logging.FileHandler(kwargs_fit['f_log'])
        loggerfile.setLevel(logging.INFO)
        logger.addHandler(loggerfile)


    # ---------------------------------------------------

    # Setup the parinfo dictionary that mpfit needs
    p_initial = gal.model.get_free_parameters_values()
    pkeys = gal.model.get_free_parameter_keys()
    nparam = len(p_initial)
    parinfo = [{'value':0, 'limited': [1, 1], 'limits': [0., 0.], 'fixed': 0, 'parname':''} for i in
               range(nparam)]

    for cmp in pkeys:
        for param_name in pkeys[cmp]:

            if pkeys[cmp][param_name] != -99:

                bounds = gal.model.components[cmp].bounds[param_name]
                k = pkeys[cmp][param_name]
                parinfo[k]['limits'][0] = bounds[0]
                parinfo[k]['limits'][1] = bounds[1]
                parinfo[k]['value'] = p_initial[k]
                parinfo[k]['parname'] = '{}:{}'.format(cmp, param_name)

    # Setup dictionary of arguments that mpfit_chisq needs

    # fa_init = {'gal':gal, 'fitvelocity':kwargs_fit['fitvelocity'],
    #             'fitdispersion':kwargs_fit['fitdispersion'],
    #             'fitflux':kwargs_fit['fitflux'], 'use_weights': kwargs_fit['use_weights']}
    # fa = {**fa_init, **kwargs_galmodel}
    fa = {'gal':gal}

    # Run mpfit
    # Output some fitting info to logger:
    logger.info("*************************************")
    logger.info(" Fitting: {} using MPFIT".format(gal.name))
    for obs_name in gal.observations:
        obs = gal.observations[obs_name]
        logger.info("    obs: {}".format(obs.name))
        if obs.data.filename_velocity is not None:
            logger.info("        velocity file: {}".format(obs.data.filename_velocity))
        if obs.data.filename_dispersion is not None:
            logger.info("        dispers. file: {}".format(obs.data.filename_dispersion))

        logger.info('        nSubpixels: {}'.format(obs.mod_options.oversample))

    if ('halo' in gal.model.components.keys()):
        logger.info('\n'+'mvirial_tied: {}'.format(gal.model.components['halo'].mvirial.tied))
    if 'disk+bulge' in gal.model.components.keys():
        if 'mhalo_relation' in gal.model.components['disk+bulge'].__dict__.keys():
            logger.info('mhalo_relation: {}'.format(gal.model.components['disk+bulge'].mhalo_relation))
        if 'truncate_lmstar_halo' in gal.model.components['disk+bulge'].__dict__.keys():
            logger.info('truncate_lmstar_halo: {}'.format(gal.model.components['disk+bulge'].truncate_lmstar_halo))

    # ----------------------------------

    logger.info('\nMPFIT Fitting:\n'
                'Start: {}\n'.format(datetime.datetime.now()))
    start = time.time()

    m = mpfit(mpfit_chisq, parinfo=parinfo, functkw=fa, maxiter=kwargs_fit['maxiter'],
              iterfunct=mpfit_printer, iterkw={'logger': logger})

    end = time.time()
    elapsed = end - start
    endtime = str(datetime.datetime.now())
    timemsg = 'Time= {:3.2f} (sec), {:3.0f}:{:3.2f} (m:s)'.format(elapsed, np.floor(elapsed / 60.),
                                                                  (elapsed / 60. - np.floor(
                                                                      elapsed / 60.)) * 60.)
    statusmsg = 'MPFIT Status = {}'.format(m.status)
    if m.status <= 0:
        errmsg = 'MPFIT Error/Warning Message = {}'.format(m.errmsg)
    elif m.status == 5:
        errmsg = 'MPFIT Error/Warning Message = Maximum number of iterations reached. Fit may not have converged!'
    else:
        errmsg = 'MPFIT Error/Warning Message = None'

    logger.info('\nEnd: ' + endtime + '\n'
                '\n******************\n'
                '' + timemsg + '\n'
                '' + statusmsg + '\n'
                '' + errmsg + '\n'
                '******************')

    # Save all of the fitting results in an MPFITResults object
    mpfitResults = MPFITResults(model=gal.model,
                                f_plot_bestfit=kwargs_fit['f_plot_bestfit'],
                                f_results=kwargs_fit['f_results'],
                                blob_name=kwargs_fit['blob_name'])

    mpfitResults.input_results(m, model=gal.model,
                    model_aperture_r=kwargs_fit['model_aperture_r'],
                    model_key_halo=kwargs_fit['model_key_halo'])

    #####
    # Do all analysis, plotting, saving:
    mpfitResults.analyze_plot_save_results(gal, kwargs_fit=kwargs_fit)

    # Clean up logger:
    if kwargs_fit['f_log'] is not None:
        logger.removeHandler(loggerfile)


    return mpfitResults


class MPFITResults(FitResults):
    """
    Class to hold results of using MPFIT to fit to DYSMALPY models.
    """
    def __init__(self, model=None, f_plot_bestfit=None, f_plot_spaxel=None,
                    f_plot_aperture=None, f_plot_channel=None,
                    f_results=None, blob_name=None, **kwargs):

        self._mpfit_object = None

        self.blob_name = blob_name

        super(MPFITResults, self).__init__(model=model, f_plot_bestfit=f_plot_bestfit,
                        f_plot_spaxel=f_plot_spaxel, f_plot_aperture=f_plot_aperture, f_plot_channel=f_plot_channel,
                        f_results=f_results, fit_method='MPFIT')

    def analyze_plot_save_results(self, gal, kwargs_fit=None):
        """
        Wrapper for analyzing MPFIT results and all remaining saving / plotting after fit.
        """

        # Update theta to best-fit:
        gal.model.update_parameters(self.bestfit_parameters)

        gal.create_model_data()

        ###
        self.bestfit_redchisq = chisq_red(gal)
        self.bestfit_chisq = chisq_eval(gal)

        if kwargs_fit['f_results'] is not None:
            self.save_results(filename=kwargs_fit['f_results'], overwrite=kwargs_fit['overwrite'])

        if kwargs_fit['f_model'] is not None:
            # Save model w/ updated theta equal to best-fit:
            gal.preserve_self(filename=kwargs_fit['f_model'], save_data=kwargs_fit['save_data'],
                        overwrite=kwargs_fit['overwrite'])

        if kwargs_fit['f_model_bestfit'] is not None:
            gal.save_model_data(filename=kwargs_fit['f_model_bestfit'], overwrite=kwargs_fit['overwrite'])

        if kwargs_fit['save_bestfit_cube']:
            for obs_name in gal.observations:
                logger.warning("CANT SAVE MORE THAN ONE RIGHT NOW")
                obs = gal.observations[obs_name]
                obs.model_cube.data.write(kwargs_fit['f_cube'], overwrite=kwargs_fit['overwrite'])

        if kwargs_fit['do_plotting'] & (kwargs_fit['f_plot_bestfit'] is not None):
            plotting.plot_bestfit(self, gal, fileout=kwargs_fit['f_plot_bestfit'],
                            fileout_aperture=kwargs_fit['f_plot_aperture'],
                            fileout_spaxel=kwargs_fit['f_plot_spaxel'],
                            fileout_channel=kwargs_fit['f_plot_channel'],
                            overwrite=kwargs_fit['overwrite'])

        # Save velocity / other profiles to ascii file:
        if kwargs_fit['f_vel_ascii'] is not None:
            logger.warning("Only writing best-fit vel ascii file for 1 observation! FIX ME!!!!!")

            for obs_name in gal.observations:
                obs = gal.observations[obs_name]
                self.save_bestfit_vel_ascii(obs, gal.model, filename=kwargs_fit['f_vel_ascii'],
                        model_aperture_r=kwargs_fit['model_aperture_r'], overwrite=kwargs_fit['overwrite'])

        if (kwargs_fit['f_vcirc_ascii'] is not None) or (kwargs_fit['f_mass_ascii'] is not None):
            self.save_bestfit_vcirc_mass_profiles(gal, outpath=kwargs_fit['outdir'],
                    fname_intrinsic=kwargs_fit['f_vcirc_ascii'],
                    fname_intrinsic_m=kwargs_fit['f_mass_ascii'], overwrite=kwargs_fit['overwrite'])


    def input_results(self, mpfit_obj, model=None,
                    model_aperture_r=None, model_key_halo=None):
        """
        Save the best fit results from MPFIT in the MPFITResults object
        """

        self._mpfit_object = mpfit_obj
        if 'blas_enorm' in self._mpfit_object.__dict__.keys():
            # Can't pickle this object if this is a FORTRAN OBJECT // eg as defined in mpfit.py
            self._mpfit_object.blas_enorm = None
        self.status = mpfit_obj.status
        self.errmsg = mpfit_obj.errmsg
        self.niter = mpfit_obj.niter

        # Populate the self.bestfit_parameters attribute with the bestfit values for the
        # free parameters
        self.bestfit_parameters = mpfit_obj.params
        self.bestfit_parameters_err = mpfit_obj.perror

        if mpfit_obj.status > 0:
            self.bestfit_redchisq = mpfit_obj.fnorm/mpfit_obj.dof


        # Add "blob" bestfit:
        if self.blob_name is not None:
            if isinstance(self.blob_name, str):
                blob_names = [self.blob_name]
            else:
                blob_names = self.blob_name[:]

            for blobn in blob_names:
                if blobn.lower() == 'fdm':
                    param_bestfit = model.get_dm_frac_effrad(model_aperture_r=model_aperture_r)
                elif blobn.lower() == 'mvirial':
                    param_bestfit = model.get_mvirial(model_key_halo=model_key_halo)
                elif blobn.lower() == 'alpha':
                    param_bestfit = model.get_halo_alpha(model_key_halo=model_key_halo)
                elif blobn.lower() == 'rb':
                    param_bestfit = model.get_halo_rb(model_key_halo=model_key_halo)

                self.analyze_blob_value(bestfit=param_bestfit, parname=blobn.lower())




    def analyze_blob_value(self, bestfit=None, parname=None):
        # Eg: parname = 'fdm' / 'mvirial' / 'alpha'
        pname = parname.strip()
        # In case ever want to do error propagation here
        err_fill = -99.
        # --------------------------------------------
        # Save best-fit results in the MCMCResults instance
        self.__dict__['bestfit_{}'.format(pname)] = bestfit
        self.__dict__['bestfit_{}_err'.format(pname)] = err_fill


    def plot_results(self, gal, f_plot_bestfit=None,
                     f_plot_spaxel=None, f_plot_aperture=None, f_plot_channel=None,
                     overwrite=False):
        """Plot/replot the bestfit for the MPFIT fitting"""
        # Specific to 3D: 'f_plot_spaxel', 'f_plot_aperture', 'f_plot_channel'
        logger.warning("NEED TO FIX FILENAMES FOR MULTI-OBS")
        self.plot_bestfit(gal, fileout=f_plot_bestfit,
                         fileout_aperture=f_plot_aperture,
                         fileout_spaxel=f_plot_spaxel,
                         fileout_channel=f_plot_channel,
                         overwrite=overwrite)


def mpfit_chisq(theta, fjac=None, gal=None):

    gal.model.update_parameters(theta)
    gal.create_model_data()

    chisq_arr_raw_allobs =  []
    obs_count = 0

    for obs_name in gal.observations:
        obs = gal.observations[obs_name]

        if obs.data.ndim == 3:
            dat = obs.data.data.unmasked_data[:].value
            mod = obs.model_data.data.unmasked_data[:].value
            err = obs.data.error.unmasked_data[:].value
            msk = obs.data.mask

            # Weights:
            wgt_data = 1.
            if hasattr(obs.data, 'weight'):
                if obs.data.weight is not None:
                    wgt_data = obs.data.weight


            # Artificially mask zero errors which are masked
            err[((err == 0) & (msk == 0))] = 99.
            chisq_arr_raw = msk * (((dat - mod) / err)) * np.sqrt(wgt_data) * obs.weight

            chisq_arr_raw = chisq_arr_raw.flatten()

        elif (obs.data.ndim == 1) or (obs.data.ndim == 2):
            # Weights:
            wgt_data = 1.
            if hasattr(obs.data, 'weight'):
                if obs.data.weight is not None:
                    wgt_data = obs.data.weight[msk]

            if obs.fit_options.fit_velocity:
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
                        disp_mod = np.sqrt(
                            disp_mod ** 2 - obs.instrument.lsf.dispersion.to(
                                u.km / u.s).value ** 2)


            if obs.fit_options.fit_flux:
                msk = obs.data.mask
                flux_dat = obs.data.data['flux'][msk]
                flux_mod = obs.model_data.data['flux'][msk]
                try:
                    flux_err = obs.data.error['flux'][msk]
                except:
                    flux_err = 0.1*obs.data.data['flux'][msk] # PLACEHOLDER

            chisq_arr_stack = []
            count = 0
            if obs.fit_options.fit_velocity:
                count += 1
                chisq_arr_raw_vel = ((vel_dat - vel_mod) / vel_err) * np.sqrt(wgt_data) * obs.weight
                chisq_arr_stack.append(chisq_arr_raw_vel.flatten())
            if obs.fit_options.fit_dispersion:
                count += 1
                chisq_arr_raw_disp = ((disp_dat - disp_mod) / disp_err) * np.sqrt(wgt_data) * obs.weight
                chisq_arr_stack.append(chisq_arr_raw_disp.flatten())
            if obs.fit_options.fit_flux:
                count += 1
                chisq_arr_raw_flux = ((flux_dat - flux_mod) / flux_err) * np.sqrt(wgt_data) * obs.weight
                chisq_arr_stack.append(chisq_arr_raw_flux.flatten())

            if count > 1:
                chisq_arr_raw = np.hstack(chisq_arr_stack)
            else:
                chisq_arr_raw = chisq_arr_stack[0]

        else:
            logger.warning("ndim={} not supported!".format(gal.data.ndim))
            raise ValueError

        chisq_arr_raw_allobs.append(chisq_arr_raw)
        obs_count += 1

    if obs_count > 1:
        chisq_arr_raw_allobs = np.hstack(chisq_arr_raw_allobs)
    else:
        chisq_arr_raw_allobs = chisq_arr_raw_allobs[0]



    status = 0

    return [status, chisq_arr_raw_allobs]



def mpfit_printer(fcn, x, iter, fnorm, functkw=None,
                  quiet=0, parinfo=None,
                  pformat='%.10g', dof=None,
                  logger=None):

        if quiet:
            return

        # Determine which parameters to print
        nprint = len(x)
        iter_line = "Iter {}  CHI-SQUARE = {:.10g}  DOF = {:}".format(iter, fnorm, dof)
        param_lines = '\n'
        for i in range(nprint):
            if (parinfo is not None) and ('parname' in parinfo[i]):
                p = '   ' + parinfo[i]['parname'] + ' = '
            else:
                p = '   P' + str(i) + ' = '
            if (parinfo is not None) and ('mpprint' in parinfo[i]):
                iprint = parinfo[i]['mpprint']
            else:
                iprint = 1
            if iprint:

                param_lines += p + (pformat % x[i]) + '  ' + '\n'

        if logger is None:
            print(iter_line+param_lines)
        else:
            logger.info(iter_line+param_lines)

        return 0




def norm(x): # Euclidean norm
    return np.sqrt(np.sum(x**2))



def _reload_all_fitting_mpfit(filename_galmodel=None, filename_results=None):
    gal = galaxy.load_galaxy_object(filename=filename_galmodel)
    results = MPFITResults()
    results.reload_results(filename=filename_results)
    return gal, results
