# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Module containing some useful utility functions

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import os
import re
import logging

# Third party imports
import numpy as np
from scipy import interpolate

import copy

from astropy.io import fits
import astropy.units as u

import datetime

try:
    import aperture_classes, data_classes
except:
    from . import aperture_classes
    from . import data_classes


_bayesian_fitting_methods = ['mcmc', 'nested']

#--------------------------------------------------------------
# Get DysmalPy version:
dir_path = os.path.dirname(os.path.realpath(__file__))
init_string = open(os.path.join(dir_path, '__init__.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, init_string, re.M)
__dpy_version__ = mo.group(1)
#--------------------------------------------------------------

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

# Class for intrinsic rot curve
class RotCurveInt(object):
    def __init__(self, r=None, vcirc_tot=None, vcirc_bar=None, vcirc_dm=None):

        self.rarr = r

        data = {'vcirc_tot': vcirc_tot,
                'vcirc_bar': vcirc_bar,
                'vcirc_dm': vcirc_dm}


        self.data = data


def read_model_intrinsic_profile(filename=None):
    # Load the data set to be fit
    dat_arr =   np.loadtxt(filename) #datadir+'{}.obs_prof.txt'.format(galID))
    gal_r      = dat_arr[:,0]
    vcirc_tot  = dat_arr[:,1]
    vcirc_bar  = dat_arr[:,2]
    vcirc_dm   = dat_arr[:,3]

    model_int = RotCurveInt(r=gal_r, vcirc_tot=vcirc_tot, vcirc_bar=vcirc_bar, vcirc_dm=vcirc_dm)

    return model_int



def read_bestfit_1d_obs_file(filename=None):
    """
    Short function to save load space 1D obs profile for a galaxy (eg, for plotting, etc)
    Follows form of H.Ü. example.
    """

    # Load the model file
    dat_arr =   np.loadtxt(filename)
    gal_r =     dat_arr[:,0]
    gal_flux =  dat_arr[:,1]
    gal_vel =   dat_arr[:,2]
    gal_disp =  dat_arr[:,3]

    model_data = data_classes.Data1D(r=gal_r, velocity=gal_vel,
                                     vel_disp=gal_disp, flux=gal_flux)

    return model_data

def write_model_obs_file(obs=None, model=None, fname=None, overwrite=False):

    ndim = obs.instrument.ndim
    if ndim == 1:
        write_model_1d_obs_file(obs=obs, fname=fname, overwrite=overwrite)
    elif ndim == 2:
        write_model_2d_obs_file(obs=obs, model=model, fname=fname, overwrite=overwrite)
    elif ndim == 3:
        write_model_3d_obs_file(obs=obs, fname=fname, overwrite=overwrite)
    elif ndim == 0:
        write_model_0d_obs_file(obs=obs, fname=fname, overwrite=overwrite)
    else:
        raise ValueError("ndim={} not recognized!".format(ndim))


def write_model_1d_obs_file(obs=None, fname=None, overwrite=False):
    """
    Short function to save *observed* space 1D model profile for a obsaxy (eg, for plotting, etc)
    Follows form of H.Ü. example.
    """
    if (not overwrite) and (fname is not None):
        if os.path.isfile(fname):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fname))
            return None

    model_r = obs.model_data.rarr
    model_flux = obs.model_data.data['flux']
    model_vel = obs.model_data.data['velocity']
    model_disp = obs.model_data.data['dispersion']

    # Correct dispersion for instrumental resolution of data is inst-corrected:
    if 'inst_corr' in obs.data.data.keys():
        if obs.data.data['inst_corr']:
            model_disp = np.sqrt( model_disp**2 - obs.instrument.lsf.dispersion.to(u.km/u.s).value**2 )
            model_disp[~np.isfinite(model_disp)] = 0

    # Write 1D profiles to text file
    np.savetxt(fname, np.transpose([model_r, model_flux, model_vel, model_disp]),
               fmt='%2.4f\t%2.4f\t%5.4f\t%5.4f',
               header='r [arcsec], flux [...], vel [km/s], disp [km/s]')

    return None


def write_model_2d_obs_file(obs=None, model=None, fname=None, overwrite=False):
    """
    Method to save the model 2D maps for a obsaxy.
    """

    data_mask = obs.data.mask

    vel_mod =  obs.model_data.data['velocity']

    if obs.model_data.data['flux'] is not None:
        flux_mod = obs.model_data.data['flux']
    else:
        flux_mod = np.ones(vel_mod.shape) * np.NaN

    if obs.model_data.data['dispersion'] is not None:
        disp_mod = obs.model_data.data['dispersion']
    else:
        disp_mod = np.ones(vel_mod.shape) * np.NaN

    # Correct model for instrument dispersion if the data is instrument corrected:
    if ('inst_corr' in obs.data.data.keys()) & (obs.model_data.data['dispersion'] is not None):
        if obs.data.data['inst_corr']:
            disp_mod = np.sqrt(disp_mod**2 -
                               obs.instrument.lsf.dispersion.to(u.km/u.s).value**2)
            disp_mod[~np.isfinite(disp_mod)] = 0   # Set the dispersion to zero when its below
                                                   # below the instrumental dispersion

    try:
        spec_unit = obs.instrument.spec_start.unit.to_string()
    except:
        spec_unit = 'km/s'  # Assume default

    hdr = fits.Header()

    hdr['NAXIS'] = (2, '2D map')
    hdr['NAXIS1'] = (flux_mod.shape[0], 'x size')
    hdr['NAXIS2'] = (flux_mod.shape[1], 'y size')

    hdr['WCSAXES'] = (2, 'Number of coordinate axes')

    try:
        hdr['PIXSCALE'] = (obs.instrument.pixscale.value / 3600., 'pixel scale [deg]') # Convert pixscale from arcsec to deg
    except:
        hdr['PIXSCALE'] = (obs.instrument.pixscale / 3600., 'pixel scale [deg]')  # Convert pixscale from arcsec to deg

    hdr['CUNIT1'] = ('deg', 'Units of coordinate increment and value')
    hdr['CUNIT2'] = ('deg', 'Units of coordinate increment and value')
    hdr['CDELT1'] = hdr['CDELT2'] = (hdr['PIXSCALE'], 'Units of coordinate increment and value')


    hdr['CRVAL1'] = (0., '[deg] Coordinate value at reference point')
    hdr['CRVAL2'] = (0., '[deg] Coordinate value at reference point')
    # Uses FITS standard where first pixel is (1,1)
    if obs.mod_options.xcenter is not None:
        xcenter = obs.mod_options.xcenter + 1
    else:
        xcenter = (vel_mod.shape[1]-1)/2. + 1
    if obs.mod_options.ycenter is not None:
        ycenter = obs.mod_options.ycenter + 1
    else:
        ycenter = (vel_mod.shape[0]-1)/2. + 1

    if len(model.geometries) > 0:
        hdr['CRPIX1'] = (xcenter + model.geometries[obs.name].xshift.value, 'Pixel coordinate of reference point')
        hdr['CRPIX2'] = (ycenter + model.geometries[obs.name].yshift.value, 'Pixel coordinate of reference point')
    else:
        key = next(iter(model.higher_order_geometries))
        hdr['CRPIX1'] = (xcenter + model.higher_order_geometries[key].xshift.value,
                         'Pixel coordinate of reference point')
        hdr['CRPIX2'] = (ycenter + model.higher_order_geometries[key].yshift.value,
                         'Pixel coordinate of reference point')

    hdr['BUNIT'] = (spec_unit, 'Spectral unit')

    hdr['HISTORY'] = 'Written by dysmalpy v{} on {}'.format(__dpy_version__, datetime.datetime.now())

    hdr_flux = hdr.copy()
    hdr_flux['BUNIT'] = ('', 'Arbitrary flux units')

    hdu_flux = fits.ImageHDU(data=flux_mod * data_mask, header=hdr_flux, name='flux')
    hdu_vel =  fits.ImageHDU(data=vel_mod * data_mask,  header=hdr, name='velocity')
    hdu_disp = fits.ImageHDU(data=disp_mod * data_mask, header=hdr, name='dispersion')

    hdul = fits.HDUList()
    hdul.append(hdu_flux)
    hdul.append(hdu_vel)
    hdul.append(hdu_disp)

    hdul.writeto(fname, overwrite=overwrite)

    return None

def write_model_3d_obs_file(obs=None, fname=None, overwrite=False):

    obs.model_data.data.write(fname, overwrite=overwrite)

    return None

def write_model_0d_obs_file(obs=None, fname=None, overwrite=False, spec_type=None):
    if (not overwrite) and (fname is not None):
        if os.path.isfile(fname):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fname))
            return None

    #
    if spec_type is None:
        spec_type = obs.instrument.spec_orig_type

    try:
        spec_unit = obs.instrument.spec_start.unit
    except:
        spec_unit = '??'

    x = obs.model_data.x
    mod = obs.model_data.data

    if spec_type.lower() == 'velocity':
        hdr = 'vel [{}], flux [...]'.format(spec_unit)

    elif spec_type.lower() == 'wavelength':
        hdr = 'wavelength [{}], flux [...]'.format(spec_unit)

    # Write 0D integrated spectrum to text file
    np.savetxt(fname, np.transpose([x, mod]),
               fmt='%2.4f\t%5.4f',
               header=hdr)


    return None

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Report(object):
    def __init__(self, report_type='pretty', fit_method=None):

        self.report_type = report_type
        self.fit_method = fit_method

        self._report = ''

    @property
    def report(self):
        return self._report

    def add_line(self, line):
        self._report += line + '\n'

    def add_string(self, string):
        self._report += string


    def create_results_report(self, gal, results, output_options=None):
        if self.report_type.lower().strip() == 'pretty':
            self._create_results_report_pretty(gal, results, output_options=output_options)
        elif self.report_type.lower().strip() == 'machine':
            self._create_results_report_machine(gal, results, output_options=output_options)


    def _create_results_report_pretty(self, gal, results, output_options=None):
        # --------------------------------------------
        if results.blob_name is not None:
            if isinstance(results.blob_name, str):
                blob_names = [results.blob_name]
            else:
                blob_names = results.blob_name[:]
        else:
            blob_names = None

        # --------------------------------------------

        self.add_line( '###############################' )
        self.add_line( ' Fitting for {}'.format(gal.name) )
        self.add_line( '' )

        self.add_line( "Date: {}".format(datetime.datetime.now()) )
        self.add_line( '' )


        for obs_name in gal.observations:
            obs = gal.observations[obs_name]
            self.add_line( "    obs: {}".format(obs.name) )
            self.add_line( '         Datafiles:' )
            if hasattr(obs.data, 'filename_velocity'):
                if (obs.data.filename_velocity is not None):
                    self.add_line( '             vel :  {}'.format(obs.data.filename_velocity) )
            if hasattr(obs.data, 'filename_dispersion'):
                if (obs.data.filename_dispersion is not None):
                    self.add_line( '             disp: {}'.format(obs.data.filename_dispersion) )
            if hasattr(obs.data, 'file_flux'):
                if obs.data.file_flux is not None:
                    self.add_line( '             flux: {}'.format(gal.data.file_flux) )

            if hasattr(obs.data, 'file_cube'):
                if (obs.data.file_cube is not None):
                    self.add_line( '             cube: {}'.format(obs.data.file_cube) )

            if hasattr(obs.data, 'file_mask'):
                if (obs.data.file_mask is not None):
                    self.add_line( '             mask: {}'.format(obs.data.file_mask) )


            # --------------------------------------
            if 'apertures' in obs.instrument.__dict__.keys():
                if obs.instrument.apertures is not None:
                    aper_type = str(type(obs.instrument.apertures)).split('.')[-1][:-2]
                    self.add_line( '         apertures:        {}'.format(aper_type) )


            # Save info on fit_dispersion / fit_flux
            if (obs.instrument.ndim == 1) | (obs.instrument.ndim == 2):
                if obs.fit_options.fit_velocity is not None:
                    self.add_line( '         fit_velocity:           {}'.format(obs.fit_options.fit_velocity))
                if obs.fit_options.fit_dispersion is not None:
                    self.add_line( '         fit_dispersion:         {}'.format(obs.fit_options.fit_dispersion))
                if obs.fit_options.fit_flux is not None:
                    self.add_line( '         fit_flux:               {}'.format(obs.fit_options.fit_flux))


            # Save info on weighting / moments :
            if hasattr(obs.fit_options, 'weighting_method'):
                if obs.fit_options.weighting_method  is not None:
                    self.add_line( '         weighting_method:      {}'.format(obs.fit_options.weighting_method))

            if (obs.instrument.ndim == 1) | (obs.instrument.ndim == 2):
                self.add_line( '         moment:           {}'.format(obs.instrument.moment))

            if 'apertures' in obs.instrument.__dict__.keys():
                if obs.instrument.apertures is not None:
                    partial_weight = obs.instrument.apertures.apertures[0].partial_weight
                    self.add_line( '         partial_weight:        {}'.format(partial_weight))

            # Save info on z calculation / oversampling:
            if obs.mod_options.zcalc_truncate is not None:
                self.add_line( '         zcalc_truncate:        {}'.format(obs.mod_options.zcalc_truncate))
            if obs.mod_options.n_wholepix_z_min is not None:
                self.add_line( '         n_wholepix_z_min:      {}'.format(obs.mod_options.n_wholepix_z_min))
            if obs.mod_options.oversample is not None:
                self.add_line( '         oversample:            {}'.format(obs.mod_options.oversample))
            if obs.mod_options.oversize is not None:
                self.add_line( '         oversize:              {}'.format(obs.mod_options.oversize))

            self.add_line( '' )

        if output_options is not None:
            if (output_options.f_params is not None):
                self.add_line( 'Paramfile: {}'.format(output_options.f_params) )

        self.add_line( '' )
        self.add_line( 'Fitting method: {}'.format(results.fit_method.upper()))
        if 'status' in results.__dict__.keys():
            self.add_line( '    fit status: {}'.format(results.status))
        self.add_line( '' )


        # INFO on pressure support type:
        self.add_line( 'pressure_support:      {}'.format(gal.model.kinematic_options.pressure_support))
        if gal.model.kinematic_options.pressure_support:
            self.add_line( 'pressure_support_type: {}'.format(gal.model.kinematic_options.pressure_support_type))


        # --------------------------------------
        # if 'status' in results.__dict__.keys():
        #     results_status = results.status
        # else:
        #     results_status = -99
        # if (((self.fit_method.upper() == 'MPFIT') & (results_status >= 0) | (self.fit_method.upper() == 'MCMC')):
        #if True:
        self.add_line( '' )
        self.add_line( '###############################' )
        self.add_line( ' Fitting results' )

        for cmp_n in gal.model.param_names.keys():
            self.add_line( '-----------' )
            self.add_line( ' {}'.format(cmp_n) )

            nfixedtied = 0
            nfree = 0

            for param_n in gal.model.param_names[cmp_n]:
                if '{}:{}'.format(cmp_n.lower(),param_n.lower()) in results.chain_param_names:
                    nfree += 1
                    whparam = np.where(results.chain_param_names == '{}:{}'.format(cmp_n.lower(),param_n.lower()))[0][0]
                    best = results.bestfit_parameters[whparam]

                    # Bayesian:
                    if self.fit_method.lower() in _bayesian_fitting_methods:
                        l68 = results.bestfit_parameters_l68_err[whparam]
                        u68 = results.bestfit_parameters_u68_err[whparam]
                        datstr = '    {: <13}    {:9.4f}  -{:9.4f} +{:9.4f}'.format(param_n, best, l68, u68)

                    # MPFIT
                    elif self.fit_method.upper() == 'MPFIT':
                        if results.bestfit_parameters_err is not None:
                            err = results.bestfit_parameters_err[whparam]
                            datstr = '    {: <13}    {:9.4f}  +/-{:9.4f}'.format(param_n, best, err)
                        else:
                            datstr = '    {: <13}    FITTING ERROR'.format(param_n,)

                    self.add_line( datstr )
                else:
                    nfixedtied += 1
            #
            if (nfree > 0) & (nfixedtied > 0):
                self.add_line( '' )

            for param_n in gal.model.param_names[cmp_n]:
                if '{}:{}'.format(cmp_n.lower(),param_n.lower()) not in results.chain_param_names:
                    best = getattr(gal.model.components[cmp_n], param_n).value

                    if not getattr(gal.model.components[cmp_n], param_n).tied:
                        if getattr(gal.model.components[cmp_n], param_n).fixed:
                            fix_tie = '[FIXED]'
                        else:
                            fix_tie = '[UNKNOWN]'
                    else:
                        fix_tie = '[TIED]'

                    datstr = '    {: <13}    {:9.4f}  {}'.format(param_n, best, fix_tie)

                    self.add_line( datstr )

            if '_noord_flat' in gal.model.components[cmp_n].__dict__.keys():
                self.add_line( '' )
                datstr = '    {: <13}       {}'.format('noord_flat', gal.model.components[cmp_n].noord_flat)
                self.add_line( datstr )

        ####
        if blob_names is not None:
            # MCMC
            if self.fit_method.lower() in _bayesian_fitting_methods:
                self.add_line( '' )
                self.add_line( '-----------' )
                for blobn in blob_names:
                    blob_best = results.__dict__['bestfit_{}'.format(blobn)]
                    l68_blob = results.__dict__['bestfit_{}_l68_err'.format(blobn)]
                    u68_blob = results.__dict__['bestfit_{}_u68_err'.format(blobn)]
                    datstr = '    {: <13}    {:9.4f}  -{:9.4f} +{:9.4f}'.format(blobn, blob_best, l68_blob, u68_blob)
                    self.add_line( datstr )

        # End "if true"

        ####
        self.add_line( '' )
        self.add_line( '-----------' )
        datstr = 'Adiabatic contraction: {}'.format(gal.model.kinematic_options.adiabatic_contract)
        self.add_line( datstr )

        self.add_line( '' )
        self.add_line( '-----------' )
        if results.bestfit_redchisq is not None:
            datstr = 'Red. chisq: {:0.4f}'.format(results.bestfit_redchisq)
        else:
            datstr = 'Red. chisq: {}'.format(results.bestfit_redchisq)
        self.add_line( datstr )


        self.add_line( '' )

        # If 2D data: Rmaxout2D:
        ncount_2d = 0
        for obs_name in gal.observations:
            obs = gal.observations[obs_name]
            if (obs.instrument.ndim == 2) & (len(gal.model.geometries) > 0):
                if ncount_2d == 0:
                    self.add_line( '-----------' )
                ncount_2d += 1
                Routmax2D = _calc_Rout_max_2D(model=gal.model, obs=obs, results=results, dscale=gal.dscale)
                datstr = 'obs {}: Rout,max,2D: {:0.4f}'.format(obs.name, Routmax2D)
                self.add_line( datstr )


        self.add_line( '' )




    def _create_results_report_machine(self, gal, results, output_options=None):
        # --------------------------------------------
        if results.blob_name is not None:
            if isinstance(results.blob_name, str):
                blob_names = [results.blob_name]
            else:
                blob_names = results.blob_name[:]
        else:
            blob_names = None

        # --------------------------------------------


        namestr = '# component             param_name      fixed       best_value   l68_err     u68_err'
        self.add_line( namestr )


        # if 'status' in results.__dict__.keys():
        #     results_status = results.status
        # else:
        #     results_status = -99
        #if (((self.fit_method.upper() == 'MPFIT') & results_status>=0) | (self.fit_method.upper() == 'MCMC')):
        # if True:
        datstr_noord = None
        for cmp_n in gal.model.param_names.keys():
            for param_n in gal.model.param_names[cmp_n]:

                if '{}:{}'.format(cmp_n.lower(),param_n.lower()) in results.chain_param_names:
                    # whparam = np.where(results.chain_param_names == '{}:{}'.format(cmp_n, param_n))[0][0]
                    # best = results.bestfit_parameters[whparam]
                    # try:
                    #     l68 = results.bestfit_parameters_l68_err[whparam]
                    #     u68 = results.bestfit_parameters_u68_err[whparam]
                    # except:
                    #     if results.bestfit_parameters_err is not None:
                    #         l68 = u68 = results.bestfit_parameters_err[whparam]
                    #     else:
                    #         # Fitting failed
                    #         best = l68 = u68 = np.NaN

                    whparam = np.where(results.chain_param_names == '{}:{}'.format(cmp_n.lower(),param_n.lower()))[0][0]
                    best = results.bestfit_parameters[whparam]

                    # Bayesian:
                    if self.fit_method.lower() in _bayesian_fitting_methods:
                        l68 = results.bestfit_parameters_l68_err[whparam]
                        u68 = results.bestfit_parameters_u68_err[whparam]

                    # MPFIT
                    elif self.fit_method.upper() == 'MPFIT':
                        if results.bestfit_parameters_err is not None:
                            l68 = u68 = results.bestfit_parameters_err[whparam]
                    else:
                        raise ValueError

                else:
                    best = getattr(gal.model.components[cmp_n], param_n).value
                    l68 = -99.
                    u68 = -99.

                #######
                if not getattr(gal.model.components[cmp_n], param_n).tied:
                    if getattr(gal.model.components[cmp_n], param_n).fixed:
                        fix_tie = 'True'
                    else:
                        fix_tie = 'False'
                else:
                    fix_tie = 'TIED'


                datstr = '{: <21}   {: <13}   {: <5}   {:12.4f}   {:9.4f}   {:9.4f}'.format(cmp_n, param_n,
                            fix_tie, best, l68, u68)
                self.add_line( datstr )


                # INFO on Noordermeer flattening:
                if '_noord_flat' in gal.model.components[cmp_n].__dict__.keys():
                    datstr_noord = '{: <21}   {: <13}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format('noord_flat', '-----',
                            '-----', str(gal.model.components[cmp_n].noord_flat), -99, -99)
            
            

        ###

        if blob_names is not None:
            for blobn in blob_names:
                blob_best = results.__dict__['bestfit_{}'.format(blobn)]
                try:
                    l68_blob = results.__dict__['bestfit_{}_l68_err'.format(blobn)]
                    u68_blob = results.__dict__['bestfit_{}_u68_err'.format(blobn)]
                except:
                    l68_blob = u68_blob = results.__dict__['bestfit_{}_err'.format(blobn)]

                datstr = '{: <21}   {: <13}   {: <5}   {:12.4f}   {:9.4f}   {:9.4f}'.format(blobn, '-----',
                            '-----', blob_best, l68_blob, u68_blob)
                self.add_line( datstr )
        # end "if True"

        ###
        if 'status' in results.__dict__.keys():
            datstr = '{: <21}   {: <13}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format('fit_status', '-----',
                            '-----', str(results.status), -99, -99)
            self.add_line( datstr )

        datstr = '{: <21}   {: <13}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format('adiab_contr', '-----',
                    '-----', str(gal.model.kinematic_options.adiabatic_contract), -99, -99)
        self.add_line( datstr )

        if results.bestfit_redchisq is not None:
            datstr = '{: <21}   {: <13}   {: <5}   {:12.4f}   {:9.4f}   {:9.4f}'.format('redchisq', '-----',
                    '-----', results.bestfit_redchisq, -99, -99)
        else:
            datstr = '{: <21}   {: <13}   {: <5}   {:12.4f}   {:9.4f}   {:9.4f}'.format('redchisq', '-----',
                    '-----', results.bestfit_redchisq, -99, -99)
        self.add_line( datstr )

        
        # INFO on Noordermeer flattening:
        if datstr_noord is not None:
            self.add_line( datstr_noord )


        # INFO on pressure support type:
        datstr = '{: <21}   {: <13}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format('pressure_support', '-----',
                    '-----', str(gal.model.kinematic_options.pressure_support), -99, -99)
        self.add_line( datstr )
        if gal.model.kinematic_options.pressure_support:
            datstr = '{: <21}   {: <13}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format('pressure_support_type', '-----',
                        '-----', str(gal.model.kinematic_options.pressure_support_type), -99, -99)
            self.add_line( datstr )

        # If 2D data: Rmaxout2D:
        ncount_2d = 0

        for obs_name in gal.observations:
            obs = gal.observations[obs_name]


            if 'apertures' in obs.instrument.__dict__.keys():
                if obs.instrument.apertures is not None:
                    aper_type = str(type(obs.instrument.apertures)).split('.')[-1][:-2]
                    lblstr = 'obs:{}:apertures'.format(''.join(obs_name.split(' ')))
                    datstr = '{: <21}   {: <13}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format(lblstr, '-----',
                                '-----', aper_type, -99, -99)
                    self.add_line( datstr )

            if hasattr(obs.fit_options, 'weighting_method'):
                if obs.fit_options.weighting_method  is not None:
                    lblstr = 'obs:{}:weighting_method'.format(''.join(obs_name.split(' ')))
                    datstr = '{: <21}   {: <13}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format(lblstr, '-----',
                                '-----', str(obs.fit_options.weighting_method), -99, -99)
                    self.add_line( datstr )


            if (obs.instrument.ndim == 1) | (obs.instrument.ndim == 2):
                lblstr = 'obs:{}:moment'.format(''.join(obs_name.split(' ')))
                datstr = '{: <21}   {: <13}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format(lblstr, '-----',
                            '-----', str(obs.instrument.moment), -99, -99)
                self.add_line( datstr )

            if 'apertures' in obs.instrument.__dict__.keys():
                if obs.instrument.apertures is not None:
                    lblstr = 'obs:{}:partial_weight'.format(''.join(obs_name.split(' ')))
                    datstr = '{: <21}   {: <13}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format(lblstr, '-----',
                                '-----', str(obs.instrument.apertures.apertures[0].partial_weight), -99, -99)
                    self.add_line( datstr )



            # If 2D data: Rmaxout2D:
                if (obs.instrument.ndim == 2) & (len(gal.model.geometries) > 0):
                    ncount_2d += 1
                    Routmax2D = _calc_Rout_max_2D(model=gal.model, obs=obs, results=results, dscale=gal.dscale)
                    lblstr = 'obs:{}:Routmax2D'.format(''.join(obs_name.split(' ')))
                    datstr = '{: <21}   {: <13}   {: <5}   {:12.4f}   {:9.4f}   {:9.4f}'.format(lblstr, '-----',
                                '-----', Routmax2D, -99, -99)
                    self.add_line( datstr )


        ########




def create_results_report(gal, results, output_options=None, report_type='pretty'):

    if results.fit_method is None:
        return None

    # Catch backwards compatibility:
    if (report_type.lower().strip() == 'short'):
        raise FutureWarning("Report type 'short' has been depreciated. Use 'pretty' in the future.")
        report_type = 'pretty'
    if (report_type.lower().strip() == 'long'):
        raise FutureWarning("Report type 'long' has been depreciated. Use 'machine' in the future.")
        report_type = 'machine'


    report = Report(report_type=report_type, fit_method=results.fit_method)
    report.create_results_report(gal, results, output_options=output_options)

    return report.report



#########################

# BETTER METHOD: ALONG MAJOR AXIS:
def _calc_Rout_max_2D(model=None, obs=None, results=None, dscale=None):
    model.update_parameters(results.bestfit_parameters)
    nx_sky = obs.data.data['velocity'].shape[1]
    ny_sky = obs.data.data['velocity'].shape[0]

    try:
        center_pixel_kin = (obs.mod_options.xcenter + model.geometries[obs.name].xshift.value,
                            obs.mod_options.ycenter + model.geometries[obs.name].yshift.value)
    except:
        center_pixel_kin = (int(nx_sky/ 2.) + model.geometries[obs.name].xshift.value,
                            int(ny_sky/ 2.) + model.geometries[obs.name].yshift.value)

    # Start going to neg, pos of center, at PA, and check if mask True/not
    #   in steps of pix, then rounding. if False: stop, and set 1 less as the end.
    cPA = np.cos(model.geometries[obs.name].pa.value * np.pi/180.)
    sPA = np.sin(model.geometries[obs.name].pa.value * np.pi/180.)

    rstep_A = 0.25
    rMA_tmp = 0
    rMA_arr = []
    # PA is to Blue; rMA_arr is [Blue (neg), Red (pos)]
    # but for PA definition blue will be pos step; invert at the end
    for fac in [1.,-1.]:
        ended_MA = False
        while not ended_MA:
            rMA_tmp += fac * rstep_A
            xtmp = rMA_tmp * -sPA + center_pixel_kin[0]
            ytmp = rMA_tmp * cPA  + center_pixel_kin[1]
            if (xtmp < 0) | (xtmp >nx_sky-1) | (ytmp < 0) | (ytmp >ny_sky-1):
                rMA_arr.append(-1.*(rMA_tmp - fac*rstep_A))  # switch sign: pos / blue for calc becomes neg
                rMA_tmp = 0
                ended_MA = True
            elif not obs.data.mask[int(np.round(ytmp)), int(np.round(xtmp))]:
                rMA_arr.append(-1.*rMA_tmp)  # switch sign: pos / blue for calc becomes neg
                rMA_tmp = 0
                ended_MA = True

    Routmax2D = np.max(np.abs(np.array(rMA_arr)))

    # In pixels. Convert to arcsec then to kpc:
    Routmax2D_kpc = Routmax2D * obs.instrument.pixscale.value / dscale
    return Routmax2D_kpc


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def create_vel_profile_files(gal=None, outpath=None,
            moment=False,
            partial_weight=True,
            fname_model_matchdata=None,
            fname_finer=None,
            fname_intrinsic=None,
            fname_intrinsic_m = None,
            overwrite=False,
            **kwargs):

    ####
    for obs_name in gal.observations:

        obs = gal.observations[obs_name]

        if obs.data.ndim == 1:
            create_vel_profile_files_obs1d(obs=obs, model=gal.model,
                        gal_name=gal.name, dscale=gal.dscale,
                        outpath=outpath,
                        fname_model_matchdata=fname_model_matchdata, fname_finer=fname_finer,
                        overwrite=overwrite)
    ####
    create_vel_profile_files_intrinsic(gal=gal, outpath=outpath,
                fname_intrinsic=fname_intrinsic, fname_intrinsic_m = fname_intrinsic_m,
                overwrite=overwrite, **kwargs)

    return None

def create_vel_profile_files_obs1d(obs=None, model=None, dscale=None,
            gal_name=None, outpath=None,
            fname_model_matchdata=None, fname_finer=None,
            overwrite=False):

    if outpath is None:
        raise ValueError

    if fname_model_matchdata is None:
        fname_model_matchdata = "{}{}_{}_out-1dplots.txt".format(outpath, gal_name, obs.name)
    if fname_finer is None:
        fname_finer = "{}{}_{}_out-1dplots_finer_sampling.txt".format(outpath, gal_name, obs.name)

    # ---------------------------------------------------------------------------
    obsin = copy.deepcopy(obs)
    obsin.create_single_obs_model_data(model, dscale)

    # --------------------------------------------------------------------------
    if (not os.path.isfile(fname_model_matchdata)) | (overwrite):
        write_model_1d_obs_file(obs=obs, fname=fname_model_matchdata, overwrite=overwrite)

    # Try finer scale:
    if (not os.path.isfile(fname_finer)) | (overwrite):
        # Reload galaxy object: reset things
        dummy_obs = copy.deepcopy(obs)
        dummy_obs.data = None
        write_1d_obs_finer_scale(obs=dummy_obs, model=model, dscale=dscale,
                                 fname=fname_finer, overwrite=overwrite,
                                 inst_corr=obs.data.data['inst_corr'])

    return None

def create_vel_profile_files_intrinsic(gal=None, outpath=None,
            fname_intrinsic=None, fname_intrinsic_m = None,
            overwrite=False):

    if ((outpath is None) and (fname_intrinsic_m is None) and (fname_intrinsic is None)):
        raise ValueError("Must set 'outpath' if 'fname_intrinsic' or 'fname_intrinsic_m' are not specified!")

    if (fname_intrinsic is None) and (outpath is not None):
        fname_intrinsic = '{}{}_vcirc_tot_bary_dm.dat'.format(outpath, gal.name)
    if (fname_intrinsic_m is None) and (outpath is not None):
        fname_intrinsic_m = '{}{}_menc_tot_bary_dm.dat'.format(outpath, gal.name)

    # ---------------------------------------------------------------------------
    galin = copy.deepcopy(gal)

    # -------------------
    # Save Bary/DM vcirc:
    if (not os.path.isfile(fname_intrinsic)) | (overwrite):
        write_vcirc_tot_bar_dm(gal=gal, fname=fname_intrinsic,
            fname_m=fname_intrinsic_m, overwrite=overwrite)

    return None

def write_1d_obs_finer_scale(obs=None, model=None, dscale=None, fname=None,
                             overwrite=False, inst_corr=None):

    # profile1d_type = obs.instrument.profile1d_type
    if isinstance(obs.instrument.apertures, aperture_classes.RectApertures):
        profile1d_type = 'rect_ap_cube'
    elif isinstance(obs.instrument.apertures, aperture_classes.CircApertures):
        profile1d_type = 'circ_ap_cube'
    elif isinstance(obs.instrument.apertures, aperture_classes.CircularPVApertures):
        profile1d_type = 'circ_ap_pv'

    try:
        aperture_radius = obs.instrument.apertures.pix_perp[0]*obs.instrument.pixscale.value
    except:
        aperture_radius=None

    # Try finer scale:
    rmax_abs = np.max([2.5, np.max(np.abs(obs.model_data.rarr))])
    r_step = 0.05
    aper_centers_interp = np.arange(-rmax_abs, rmax_abs+r_step, r_step)


    # Get slit-pa from model or data.
    slit_pa = model.geometries[obs.name].pa.value

    do_extract = True
    if profile1d_type == 'rect_ap_cube':
        f_par = interpolate.interp1d(obs.instrument.apertures.rarr, obs.instrument.apertures.pix_parallel,
                        kind='slinear', fill_value='extrapolate')
        f_perp = interpolate.interp1d(obs.instrument.apertures.rarr, obs.instrument.apertures.pix_perp,
                        kind='slinear', fill_value='extrapolate')

        pix_parallel_interp = f_par(aper_centers_interp)
        pix_perp_interp = f_perp(aper_centers_interp)

        obs.instrument.apertures = aperture_classes.setup_aperture_types(obs=obs,
                    profile1d_type=profile1d_type,
                    aper_centers = aper_centers_interp,
                    aperture_radius=1.,
                    pix_perp=pix_perp_interp, pix_parallel=pix_parallel_interp,
                    slit_pa=slit_pa,
                    partial_weight=obs.instrument.apertures.apertures[0].partial_weight)
    elif profile1d_type == 'circ_ap_cube':
        obs.instrument.apertures = aperture_classes.setup_aperture_types(obs=obs,
                    profile1d_type=profile1d_type,
                    aper_centers = aper_centers_interp,
                    aperture_radius=aperture_radius,
                    pix_perp=None, pix_parallel=None,
                    slit_pa=slit_pa,
                    partial_weight=obs.instrument.apertures.apertures[0].partial_weight)
    elif profile1d_type == 'circ_ap_pv':
        obs.instrument.apertures = aperture_classes.setup_aperture_types(obs=obs,
                    profile1d_type=profile1d_type,
                    aper_centers = aper_centers_interp,
                    aperture_radius=aperture_radius,
                    pix_perp=None, pix_parallel=None,
                    slit_pa=slit_pa, slit_width=obs.instrument.slit_width)
    else:
        do_extract = False

    if do_extract:
        # Create model:
        obs.create_single_obs_model_data(model, dscale)

        # Dummy to pass the inst corr settings:
        obs._data = data_classes.Data1D(obs.model_data.rarr, obs.model_data.data['velocity'],
                                        inst_corr=inst_corr)

        write_model_1d_obs_file(obs=obs, fname=fname, overwrite=overwrite)

    return None

def write_vcirc_tot_bar_dm(gal=None, fname=None, fname_m=None, overwrite=False):
    # -------------------
    # Save Bary/DM vcirc:

    rstep = 0.1
    rmax = 40.   # kpc
    rarr = np.arange(0, rmax+rstep, rstep)

    vcirc_tot, vcirc_bar, vcirc_dm = gal.model.circular_velocity(rarr, compute_baryon=True, compute_dm=True)

    menc_tot, menc_bar, menc_dm = gal.model.enclosed_mass(rarr, compute_baryon=True, compute_dm=True)

    vcirc_bar[~np.isfinite(vcirc_bar)] = 0.
    vcirc_dm[~np.isfinite(vcirc_dm)] = 0.

    vcirc_tot[~np.isfinite(vcirc_tot)] = 0.

    profiles = np.zeros((len(rarr), 4))
    profiles[:,0] = rarr
    profiles[:,1] = vcirc_tot
    profiles[:,2] = vcirc_bar
    profiles[:,3] = vcirc_dm

    profiles_m = np.zeros((len(rarr), 4))
    profiles_m[:,0] = rarr
    profiles_m[:,1] = np.log10(menc_tot)
    profiles_m[:,2] = np.log10(menc_bar)
    profiles_m[:,3] = np.log10(menc_dm)
    profiles_m[~np.isfinite(profiles_m)] = 0.

    save_vcirc_tot_bar_dm_files(fname=fname, fname_m=fname_m,
                    profiles=profiles, profiles_m=profiles_m, overwrite=overwrite)

    return None


#
def save_vcirc_tot_bar_dm_files(fname=None, fname_m=None, profiles=None, profiles_m=None, overwrite=False):
    save_fname = True
    save_fname_m = True
    if (not overwrite) and (fname is not None):
        if os.path.isfile(fname):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fname))
            save_fname = False
    if (not overwrite) and (fname_m is not None):
        if os.path.isfile(fname):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fname_m))
            save_fname_m = False
    if save_fname:
        with open(fname, 'w') as f:
            namestr = '#   r   vcirc_tot vcirc_bar   vcirc_dm'
            f.write(namestr+'\n')
            unitstr = '#   [kpc]   [km/s]   [km/s]   [km/s]'
            f.write(unitstr+'\n')
            for i in range(profiles.shape[0]):
                datstr = '    '.join(["{0:0.3f}".format(p) for p in profiles[i,:]])
                f.write(datstr+'\n')

    if save_fname_m:
        with open(fname_m, 'w') as f:
            namestr = '#   r   lmenc_tot   lmenc_bar   lmenc_dm'
            f.write(namestr+'\n')
            unitstr = '#   [kpc]   [log10Msun]   [log10Msun]   [log10Msun]'
            f.write(unitstr+'\n')
            for i in range(profiles.shape[0]):
                datstr = '    '.join(["{0:0.3f}".format(p) for p in profiles_m[i,:]])
                f.write(datstr+'\n')

    return None
