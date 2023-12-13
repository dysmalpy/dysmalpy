# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Classes and functions for fitting DYSMALPY kinematic models
#   to the observed data using MPFIT
#

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

## Standard library
import logging

# DYSMALPY code
from dysmalpy import galaxy
from dysmalpy import utils as dpy_utils
from dysmalpy.fitting import base
from dysmalpy.fitting import utils as fit_utils

# Third party imports
import os
import numpy as np
import astropy.units as u
from dysmalpy.extern.mpfit import mpfit

import time, datetime


__all__ = ['MPFITFitter', 'MPFITResults']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)



class MPFITFitter(base.Fitter):
    """
    Class to hold the MPFIT fitter attributes + methods
    """
    def __init__(self, **kwargs):
        self._set_defaults()
        super(MPFITFitter, self).__init__(fit_method='MPFIT', **kwargs)

    def _set_defaults(self):
        self.maxiter=200


    def fit(self, gal, output_options):
        """
        Fit observed kinematics using MPFIT and a DYSMALPY model set.

        Parameters
        ----------
            gal : `Galaxy` instance
                observed galaxy, including kinematics.
                also contains instrument the galaxy was observed with (gal.instrument)
                and the DYSMALPY model set, with the parameters to be fit (gal.model)

            output_options : `config.OutputOptions` instance
                instance holding ouptut options for MCMC fitting.

        Returns
        -------
            mpfitResults : `MPFITResults` instance
                MPFITResults class instance containing the bestfit parameters, fit information, etc.

        """

        # Check the FOV is large enough to cover the data output:
        dpy_utils._check_data_inst_FOV_compatibility(gal)

        # Pre-calculate instrument kernels:
        gal = dpy_utils._set_instrument_kernels(gal)

        # Set output options: filenames / which to save, etc
        output_options.set_output_options(gal, self)

        fit_utils._check_existing_files_overwrite(output_options, fit_type='mpfit')


        # Setup file redirect logging:
        if output_options.f_log is not None:
            loggerfile = logging.FileHandler(output_options.f_log)
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
            logger.info('    mvirial_tied: {}'.format(gal.model.components['halo'].mvirial.tied))
            #logger.info('    fdm_tied: {}'.format(gal.model.components['halo'].fdm.tied))
        if 'disk+bulge' in gal.model.components.keys():
            if 'mhalo_relation' in gal.model.components['disk+bulge'].__dict__.keys():
                logger.info('mhalo_relation: {}'.format(gal.model.components['disk+bulge'].mhalo_relation))
            if 'truncate_lmstar_halo' in gal.model.components['disk+bulge'].__dict__.keys():
                logger.info('truncate_lmstar_halo: {}'.format(gal.model.components['disk+bulge'].truncate_lmstar_halo))

        # ----------------------------------

        logger.info('\nMPFIT Fitting:\n'
                    'Start: {}\n'.format(datetime.datetime.now()))
        start = time.time()

        m = mpfit(mpfit_chisq, parinfo=parinfo, functkw=fa, maxiter=self.maxiter,
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
        mpfitResults = MPFITResults(model=gal.model, blob_name=self.blob_name)

        mpfitResults.input_results(m, gal.model)

        #####
        # Do all analysis, plotting, saving:
        mpfitResults.analyze_plot_save_results(gal, output_options=output_options)

        # Clean up logger:
        if output_options.f_log is not None:
            logger.removeHandler(loggerfile)
            loggerfile.close()

        return mpfitResults



class MPFITResults(base.FitResults):
    """
    Class to hold results of using MPFIT to fit to DYSMALPY models.
    """
    def __init__(self, model=None, blob_name=None):

        self._mpfit_object = None

        super(MPFITResults, self).__init__(model=model, blob_name=blob_name,
                                           fit_method='MPFIT')

    def analyze_plot_save_results(self, gal, output_options=None):
        """
        Wrapper for analyzing MPFIT results and all remaining saving / plotting after fit.
        """

        # Update theta to best-fit:
        gal.model.update_parameters(self.bestfit_parameters)

        gal.create_model_data()

        ###
        self.bestfit_redchisq = base.chisq_red(gal)
        self.bestfit_chisq = base.chisq_eval(gal)

        if output_options.save_results & (output_options.f_results is not None):
            self.save_results(filename=output_options.f_results, overwrite=output_options.overwrite)

        if output_options.save_model & (output_options.f_model is not None):
            # Save model w/ updated theta equal to best-fit:
            gal.preserve_self(output_options.f_model,
                              save_data=output_options.save_data,
                              overwrite=output_options.overwrite)

        if output_options.save_model_bestfit & (output_options.f_model_bestfit is not None):
            gal.save_model_data(filenames=output_options.f_model_bestfit, overwrite=output_options.overwrite)

        if output_options.save_bestfit_cube & (output_options.f_bestfit_cube is not None):
            for obs_name in gal.observations:
                obs = gal.observations[obs_name]
                fcube = output_options.f_bestfit_cube[obs_name]
                obs.model_cube.data.write(fcube, overwrite=output_options.overwrite)

        if output_options.do_plotting & (output_options.f_plot_bestfit is not None):
            self.plot_bestfit(gal, fileout=output_options.f_plot_bestfit,
                              overwrite=output_options.overwrite)

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



    def input_results(self, mpfit_obj, model):
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
                    param_bestfit = model.get_dm_frac_r_ap()
                elif blobn.lower() == 'mvirial':
                    param_bestfit = model.get_mvirial()
                elif blobn.lower() == 'alpha':
                    param_bestfit = model.get_halo_alpha()
                elif blobn.lower() == 'rb':
                    param_bestfit = model.get_halo_rb()

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



    def plot_results(self, gal, f_plot_bestfit=None, overwrite=False):
        """Plot/replot the bestfit for the MPFIT fitting"""
        self.plot_bestfit(gal, fileout=f_plot_bestfit, overwrite=overwrite)


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
