# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
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
    import aperture_classes
    import data_classes
except:
    from . import aperture_classes
    from . import data_classes

try:
    from config import Config_simulate_cube, Config_create_model_data
except:
    from .config import Config_simulate_cube, Config_create_model_data

# try:
#     from dysmalpy._version import __version__ as __dpy_version__
# except:
#     from ._version import __version__ as __dpy_version__


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

    slit_width = None
    slit_pa = None


    #
    model_data = data_classes.Data1D(r=gal_r, velocity=gal_vel,
                             vel_disp=gal_disp, flux=gal_flux,
                             slit_width=slit_width,
                             slit_pa=slit_pa)
    model_data.apertures = None

    return model_data

def write_model_obs_file(gal=None, fname=None, ndim=None, overwrite=False):
    if ndim == 1:
        write_model_1d_obs_file(gal=gal, fname=fname, overwrite=overwrite)
    elif ndim == 2:
        write_model_2d_obs_file(gal=gal, fname=fname, overwrite=overwrite)
    elif ndim == 3:
        write_model_3d_obs_file(gal=gal, fname=fname, overwrite=overwrite)
    elif ndim == 0:
        write_model_0d_obs_file(gal=gal, fname=fname, overwrite=overwrite)
    else:
        raise ValueError("ndim={} not recognized!".format(ndim))


def write_model_1d_obs_file(gal=None, fname=None, overwrite=False):
    """
    Short function to save *observed* space 1D model profile for a galaxy (eg, for plotting, etc)
    Follows form of H.Ü. example.
    """
    if (not overwrite) and (fname is not None):
        if os.path.isfile(fname):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fname))
            return None

    model_r = gal.model_data.rarr
    model_flux = gal.model_data.data['flux']
    model_vel = gal.model_data.data['velocity']
    model_disp = gal.model_data.data['dispersion']

    # Write 1D profiles to text file
    np.savetxt(fname, np.transpose([model_r, model_flux, model_vel, model_disp]),
               fmt='%2.4f\t%2.4f\t%5.4f\t%5.4f',
               header='r [arcsec], flux [...], vel [km/s], disp [km/s]')

    return None


def write_model_2d_obs_file(gal=None, fname=None, overwrite=False):
    """
    Method to save the model 2D maps for a galaxy.
    """

    data_mask = gal.data.mask

    vel_mod =  gal.model_data.data['velocity']

    if gal.model_data.data['flux'] is not None:
        flux_mod = gal.model_data.data['flux']
    else:
        flux_mod = np.ones(vel_mod.shape) * np.NaN

    if gal.model_data.data['dispersion'] is not None:
        disp_mod = gal.model_data.data['dispersion']
    else:
        disp_mod = np.ones(vel_mod.shape) * np.NaN

    # Correct model for instrument dispersion if the data is instrument corrected:
    if ('inst_corr' in gal.data.data.keys()) & (gal.model_data.data['dispersion'] is not None):
        if gal.data.data['inst_corr']:
            disp_mod = np.sqrt(disp_mod**2 -
                               gal.instrument.lsf.dispersion.to(u.km/u.s).value**2)
            disp_mod[~np.isfinite(disp_mod)] = 0   # Set the dispersion to zero when its below
                                                   # below the instrumental dispersion

    try:
        spec_unit = gal.instrument.spec_start.unit.to_string()
    except:
        spec_unit = 'km/s'  # Assume default

    hdr = fits.Header()

    hdr['NAXIS'] = (2, '2D map')
    hdr['NAXIS1'] = (flux_mod.shape[0], 'x size')
    hdr['NAXIS2'] = (flux_mod.shape[1], 'y size')

    hdr['WCSAXES'] = (2, 'Number of coordinate axes')

    try:
        hdr['PIXSCALE'] = (gal.data.pixscale.value / 3600., 'pixel scale [deg]') # Convert pixscale from arcsec to deg
    except:
        hdr['PIXSCALE'] = (gal.data.pixscale / 3600., 'pixel scale [deg]')  # Convert pixscale from arcsec to deg

    hdr['CUNIT1'] = ('deg', 'Units of coordinate increment and value')
    hdr['CUNIT2'] = ('deg', 'Units of coordinate increment and value')
    hdr['CDELT1'] = hdr['CDELT2'] = (hdr['PIXSCALE'], 'Units of coordinate increment and value')


    hdr['CRVAL1'] = (0., '[deg] Coordinate value at reference point')
    hdr['CRVAL2'] = (0., '[deg] Coordinate value at reference point')
    # Uses FITS standard where first pixel is (1,1)
    if gal.data.xcenter is not None:
        xcenter = gal.data.xcenter + 1
    else:
        xcenter = (vel_mod.shape[1]-1)/2. + 1
    if gal.data.ycenter is not None:
        ycenter = gal.data.ycenter + 1
    else:
        ycenter = (vel_mod.shape[0]-1)/2. + 1

    hdr['CRPIX1'] = (xcenter + gal.model.geometry.xshift.value, 'Pixel coordinate of reference point')
    hdr['CRPIX2'] = (ycenter + gal.model.geometry.yshift.value, 'Pixel coordinate of reference point')

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

def write_model_3d_obs_file(gal=None, fname=None, overwrite=False):

    gal.model_data.data.write(fname, overwrite=overwrite)

    return None

def write_model_0d_obs_file(gal=None, fname=None, overwrite=False, spec_type=None):
    if (not overwrite) and (fname is not None):
        if os.path.isfile(fname):
            logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, fname))
            return None

    #
    if spec_type is None:
        spec_type = gal.instrument.spec_orig_type

    try:
        spec_unit = gal.instrument.spec_start.unit
    except:
        spec_unit = '??'

    x = gal.model_data.x
    mod = gal.model_data.data

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


    def create_results_report(self, gal, results, params=None, **kwargs):
        if self.report_type.lower().strip() == 'pretty':
            self._create_results_report_pretty(gal, results, params=params, **kwargs)
        elif self.report_type.lower().strip() == 'machine':
            self._create_results_report_machine(gal, results, params=params, **kwargs)


    def _create_results_report_pretty(self, gal, results, params=None,**kwargs):
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


        if hasattr(gal.data, 'filename_velocity') & hasattr(gal.data, 'filename_dispersion'):
            if (gal.data.filename_velocity is not None) & (gal.data.filename_dispersion is not None):
                self.add_line( 'Datafiles:' )
                self.add_line( ' vel:  {}'.format(gal.data.filename_velocity) )
                self.add_line( ' disp: {}'.format(gal.data.filename_dispersion) )
                if hasattr(gal.data, 'file_flux'):
                    if gal.data.file_flux is not None:
                        self.add_line( ' flux: {}'.format(gal.data.file_flux) )
            elif (gal.data.filename_velocity is not None):
                self.add_line( 'Datafile: {}'.format(gal.data.filename_velocity) )
        elif hasattr(gal.data, 'filename_velocity'):
            if (gal.data.filename_velocity is not None):
                self.add_line( 'Datafile: {}'.format(gal.data.filename_velocity) )
        else:
            if params is not None:
                try:
                    self.add_line( 'Datafiles:' )
                    self.add_line( ' vel:  {}'.format(params['fdata_vel']) )
                    self.add_line( ' verr: {}'.format(params['fdata_verr']) )
                    self.add_line( ' disp: {}'.format(params['fdata_disp']) )
                    self.add_line( ' derr: {}'.format(params['fdata_derr']) )
                    if params['fitflux']:
                        try:
                            self.add_line( ' flux: {}'.format(params['fdata_flux']) )
                            self.add_line( ' ferr: {}'.format(params['fdata_ferr']) )
                        except:
                            pass
                    try:
                        self.add_line( ' mask: {}'.format(params['fdata_mask']) )
                    except:
                        pass
                except:
                    try:
                        self.add_line( 'Datafiles:' )
                        self.add_line( ' cube:  {}'.format(params['fdata_cube']) )
                        self.add_line( ' err:   {}'.format(params['fdata_err']) )
                    except:
                        pass

        if params is not None:
            self.add_line( 'Paramfile: {}'.format(params['param_filename']) )

        self.add_line( '' )
        self.add_line( 'Fitting method: {}'.format(results.fit_method.upper()))
        self.add_line( '' )
        if params is not None:
            if 'fit_module' in params.keys():
                self.add_line( '   fit_module: {}'.format(params['fit_module']))
                #self.add_line( '' )
        #self.add_line( '' )
        # --------------------------------------
        if 'profile1d_type' in gal.data.__dict__.keys():
            self.add_line( 'profile1d_type:        {}'.format(gal.data.profile1d_type) )


        fitdispersion = fitflux = None
        weighting_method = moment_calc = partial_weight = zcalc_truncate = None
        n_wholepix_z_min = None
        if params is not None:
            if 'fitdispersion' in params.keys():
                fitdispersion = params['fitdispersion']
            if 'fitflux' in params.keys():
                fitflux = params['fitflux']
            if 'weighting_method' in params.keys():
                weighting_method = params['weighting_method']
            if 'moment_calc' in params.keys():
                moment_calc = params['moment_calc']
            if 'partial_weight' in params.keys():
                partial_weight = params['partial_weight']

            if 'zcalc_truncate' in params.keys():
                zcalc_truncate = params['zcalc_truncate']

            if 'n_wholepix_z_min' in params.keys():
                n_wholepix_z_min = params['n_wholepix_z_min']

        if 'apertures' in gal.data.__dict__.keys():
            if gal.data.apertures is not None:
                if moment_calc is None:
                    moment_calc = gal.data.apertures.apertures[0].moment
                if partial_weight is None:
                    partial_weight = gal.data.apertures.apertures[0].partial_weight

        if zcalc_truncate is None:
            if 'zcalc_truncate' in kwargs.keys():
                zcalc_truncate = kwargs['zcalc_truncate']
            else:
                config_sim_cube = Config_simulate_cube()
                zcalc_truncate = "[Default: {}]".format(config_sim_cube.zcalc_truncate)

            if n_wholepix_z_min is None:
                if 'n_wholepix_z_min' in kwargs.keys():
                    n_wholepix_z_min = kwargs['n_wholepix_z_min']
                else:
                    config_sim_cube = Config_simulate_cube()
                    n_wholepix_z_min = "[Default: {}]".format(config_sim_cube.n_wholepix_z_min)

        # Save info on fitdispersion / fitflux
        if fitdispersion is not None:
            self.add_line( 'fitdispersion:         {}'.format(fitdispersion))
        if fitflux is not None:
            self.add_line( 'fitflux:               {}'.format(fitflux))
        if ((fitdispersion is not None) or (fitflux is not None)):
            self.add_line( '' )

        # Save info on weighting / moments:
        if weighting_method is not None:
            self.add_line( 'weighting_method:      {}'.format(weighting_method))
        if moment_calc is not None:
            self.add_line( 'moment_calc:           {}'.format(moment_calc))
        if partial_weight is not None:
            self.add_line( 'partial_weight:        {}'.format(partial_weight))
        if zcalc_truncate is not None:
            self.add_line( 'zcalc_truncate:        {}'.format(zcalc_truncate))
        if n_wholepix_z_min is not None:
            self.add_line( 'n_wholepix_z_min:      {}'.format(n_wholepix_z_min))
        # INFO on pressure support type:
        self.add_line( 'pressure_support:      {}'.format(gal.model.kinematic_options.pressure_support))
        if gal.model.kinematic_options.pressure_support:
            self.add_line( 'pressure_support_type: {}'.format(gal.model.kinematic_options.pressure_support_type))

        # --------------------------------------
        self.add_line( '' )
        self.add_line( '###############################' )
        self.add_line( ' Fitting results' )

        for cmp_n in gal.model.param_names.keys():
            self.add_line( '-----------' )
            self.add_line( ' {}'.format(cmp_n) )

            nfixedtied = 0
            nfree = 0

            for param_n in gal.model.param_names[cmp_n]:

                if '{}:{}'.format(cmp_n,param_n) in results.chain_param_names:
                    nfree += 1
                    whparam = np.where(results.chain_param_names == '{}:{}'.format(cmp_n, param_n))[0][0]
                    best = results.bestfit_parameters[whparam]

                    # MCMC
                    if self.fit_method.upper() == 'MCMC':
                        l68 = results.bestfit_parameters_l68_err[whparam]
                        u68 = results.bestfit_parameters_u68_err[whparam]
                        datstr = '    {: <11}    {:9.4f}  -{:9.4f} +{:9.4f}'.format(param_n, best, l68, u68)

                    # MPFIT
                    elif self.fit_method.upper() == 'MPFIT':
                        err = results.bestfit_parameters_err[whparam]
                        datstr = '    {: <11}    {:9.4f}  +/-{:9.4f}'.format(param_n, best, err)

                    self.add_line( datstr )
                else:
                    nfixedtied += 1
            #
            if (nfree > 0) & (nfixedtied > 0):
                self.add_line( '' )

            for param_n in gal.model.param_names[cmp_n]:

                if '{}:{}'.format(cmp_n,param_n) not in results.chain_param_names:
                    best = getattr(gal.model.components[cmp_n], param_n).value

                    if not getattr(gal.model.components[cmp_n], param_n).tied:
                        if getattr(gal.model.components[cmp_n], param_n).fixed:
                            fix_tie = '[FIXED]'
                        else:
                            fix_tie = '[UNKNOWN]'
                    else:
                        fix_tie = '[TIED]'

                    datstr = '    {: <11}    {:9.4f}  {}'.format(param_n, best, fix_tie)

                    self.add_line( datstr )


        ####
        if blob_names is not None:
            # MCMC
            if self.fit_method.upper() == 'MCMC':
                self.add_line( '' )
                self.add_line( '-----------' )
                for blobn in blob_names:
                    blob_best = results.__dict__['bestfit_{}'.format(blobn)]
                    l68_blob = results.__dict__['bestfit_{}_l68_err'.format(blobn)]
                    u68_blob = results.__dict__['bestfit_{}_u68_err'.format(blobn)]
                    datstr = '    {: <11}    {:9.4f}  -{:9.4f} +{:9.4f}'.format(blobn, blob_best, l68_blob, u68_blob)
                    self.add_line( datstr )


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

        if gal.data.ndim == 2:
            Routmax2D = _calc_Rout_max_2D(gal=gal, results=results)
            self.add_line( '' )
            self.add_line( '-----------' )
            datstr = 'Rout,max,2D: {:0.4f}'.format(Routmax2D)
            self.add_line( datstr )

        self.add_line( '' )



    def _create_results_report_machine(self, gal, results, params=None, **kwargs):
        # --------------------------------------------
        if results.blob_name is not None:
            if isinstance(results.blob_name, str):
                blob_names = [results.blob_name]
            else:
                blob_names = results.blob_name[:]
        else:
            blob_names = None

        # --------------------------------------------

        namestr = '# component             param_name    fixed       best_value   l68_err     u68_err'
        self.add_line( namestr )

        for cmp_n in gal.model.param_names.keys():
            for param_n in gal.model.param_names[cmp_n]:

                if '{}:{}'.format(cmp_n,param_n) in results.chain_param_names:
                    whparam = np.where(results.chain_param_names == '{}:{}'.format(cmp_n, param_n))[0][0]
                    best = results.bestfit_parameters[whparam]
                    try:
                        l68 = results.bestfit_parameters_l68_err[whparam]
                        u68 = results.bestfit_parameters_u68_err[whparam]
                    except:
                        l68 = u68 = results.bestfit_parameters_err[whparam]
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


                datstr = '{: <21}   {: <11}   {: <5}   {:12.4f}   {:9.4f}   {:9.4f}'.format(cmp_n, param_n,
                            fix_tie, best, l68, u68)
                self.add_line( datstr )

        ###

        if blob_names is not None:
            for blobn in blob_names:
                blob_best = results.__dict__['bestfit_{}'.format(blobn)]
                try:
                    l68_blob = results.__dict__['bestfit_{}_l68_err'.format(blobn)]
                    u68_blob = results.__dict__['bestfit_{}_u68_err'.format(blobn)]
                except:
                    l68_blob = u68_blob = results.__dict__['bestfit_{}_err'.format(blobn)]

                datstr = '{: <21}   {: <11}   {: <5}   {:12.4f}   {:9.4f}   {:9.4f}'.format(blobn, '-----',
                            '-----', blob_best, l68_blob, u68_blob)
                self.add_line( datstr )


        ###
        datstr = '{: <21}   {: <11}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format('adiab_contr', '-----',
                    '-----', str(gal.model.kinematic_options.adiabatic_contract), -99, -99)
        self.add_line( datstr )

        if results.bestfit_redchisq is not None:
            datstr = '{: <21}   {: <11}   {: <5}   {:12.4f}   {:9.4f}   {:9.4f}'.format('redchisq', '-----',
                    '-----', results.bestfit_redchisq, -99, -99)
        else:
            datstr = '{: <21}   {: <11}   {: <5}   {:12.4f}   {:9.4f}   {:9.4f}'.format('redchisq', '-----',
                    '-----', results.bestfit_redchisq, -99, -99)
        self.add_line( datstr )


        if 'profile1d_type' in gal.data.__dict__.keys():
            datstr = '{: <21}   {: <11}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format('profile1d_type', '-----',
                        '-----', gal.data.profile1d_type, -99, -99)
            self.add_line( datstr )


        #############
        weighting_method = moment_calc = partial_weight = None
        if params is not None:
            if 'weighting_method' in params.keys():
                if params['weighting_method'] is not None:
                    weighting_method = params['weighting_method']

            if 'moment_calc' in params.keys():
                if params['moment_calc'] is not None:
                    moment_calc = params['moment_calc']

            if 'partial_weight' in params.keys():
                if params['partial_weight'] is not None:
                    partial_weight = params['partial_weight']

        if 'apertures' in gal.data.__dict__.keys():
            if gal.data.apertures is not None:
                if moment_calc is None:
                     moment_calc = gal.data.apertures.apertures[0].moment
                if partial_weight is None:
                    partial_weight = gal.data.apertures.apertures[0].partial_weight

        # Apply some defaults:
        if (partial_weight is None) and (gal.data.ndim == 1):
            partial_weight = True

        # Write settings:
        if weighting_method is not None:
            datstr = '{: <21}   {: <11}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format('weighting_method', '-----',
                        '-----', str(weighting_method), -99, -99)
            self.add_line( datstr )
        if moment_calc is not None:
            datstr = '{: <21}   {: <11}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format('moment_calc', '-----',
                        '-----', str(moment_calc), -99, -99)
            self.add_line( datstr )
        if partial_weight is not None:
            datstr = '{: <21}   {: <11}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format('partial_weight', '-----',
                        '-----', str(partial_weight), -99, -99)
            self.add_line( datstr )


        # INFO on pressure support type:
        datstr = '{: <21}   {: <11}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format('pressure_support', '-----',
                    '-----', str(gal.model.kinematic_options.pressure_support), -99, -99)
        self.add_line( datstr )
        if gal.model.kinematic_options.pressure_support:
            datstr = '{: <21}   {: <11}   {: <5}   {: >12}   {:9.4f}   {:9.4f}'.format('pressure_support_type', '-----',
                        '-----', str(gal.model.kinematic_options.pressure_support_type), -99, -99)
            self.add_line( datstr )


        # If 2D data: Rmaxout2D:
        if gal.data.ndim == 2:
            Routmax2D = _calc_Rout_max_2D(gal=gal, results=results)
            datstr = '{: <21}   {: <11}   {: <5}   {:12.4f}   {:9.4f}   {:9.4f}'.format('Routmax2D', '-----',
                        '-----', Routmax2D, -99, -99)
            self.add_line( datstr )

        ########

    # Backwards compatibility:
    def create_results_report_short(self, gal, results, params=None):
        # Depreciated:
        wrn_msg = "Method report.create_results_report_short() depreciated.\n"
        wrn_msg += "Use report.create_results_report() in the future."
        raise FutureWarning(wrn_msg)

        self._create_results_report_pretty(gal, results, params=params)

    def create_results_report_long(self, gal, results, params=None):
        # Depreciated:
        wrn_msg = "Method report.create_results_report_long() depreciated.\n"
        wrn_msg += "Use report.create_results_report() in the future."
        raise FutureWarning(wrn_msg)

        self._create_results_report_machine(gal, results, params=params)





def create_results_report(gal, results, params=None, report_type='pretty', **kwargs):

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
    report.create_results_report(gal, results, params=params, **kwargs)

    return report.report



#########################
def _check_data_inst_FOV_compatibility(gal):
    logger_msg = None
    if gal.data.ndim == 1:
        if min(gal.instrument.fov)/2. <  np.abs(gal.data.rarr).max() / gal.instrument.pixscale.value:
            logger_msg = "FOV smaller than the maximum data extent!\n"
            logger_msg += "FOV=[{},{}] pix; max(abs(data.rarr))={} pix".format(gal.instrument.fov[0],
                            gal.instrument.fov[1], np.abs(gal.data.rarr).max()/ gal.instrument.pixscale)

    if logger_msg is not None:
        logger.warning(logger_msg)

    return None



#########################
## OLD METHOD: NOT AS GOOD, LOOKS AT NON-MAJOR AXIS
# def _calc_Rout_max_2D(gal=None, results=None):
#     gal.model.update_parameters(results.bestfit_parameters)
#     inc_gal = gal.model.geometry.inc.value
#
#     ###############
#     # Get grid of data coords:
#     nx_sky = gal.data.data['velocity'].shape[1]
#     ny_sky = gal.data.data['velocity'].shape[0]
#     nz_sky = 1 #np.int(np.max([nx_sky, ny_sky]))
#     rstep = gal.data.pixscale
#
#     xcenter = gal.data.xcenter
#     ycenter = gal.data.ycenter
#
#     if xcenter is None:
#         xcenter = (nx_sky - 1) / 2.
#     if ycenter is None:
#         ycenter = (ny_sky - 1) / 2.
#
#
#     sh = (nz_sky, ny_sky, nx_sky)
#     zsky, ysky, xsky = np.indices(sh)
#     zsky = zsky - (nz_sky - 1) / 2.
#     ysky = ysky - ycenter
#     xsky = xsky - xcenter
#
#     # Apply the geometric transformation to get galactic coordinates
#     xgal, ygal, zgal = gal.model.geometry(xsky, ysky, zsky)
#
#     # Get the 4 corners sets:
#     gal.model.geometry.inc = 0
#     xskyp_ur, yskyp_ur, zskyp_ur = gal.model.geometry(xsky+0.5, ysky+0.5, zsky)
#     xskyp_ll, yskyp_ll, zskyp_ll = gal.model.geometry(xsky-0.5, ysky-0.5, zsky)
#     xskyp_lr, yskyp_lr, zskyp_lr = gal.model.geometry(xsky+0.5, ysky-0.5, zsky)
#     xskyp_ul, yskyp_ul, zskyp_ul = gal.model.geometry(xsky-0.5, ysky+0.5, zsky)
#
#     #Reset:
#     gal.model.geometry.inc = inc_gal
#
#     yskyp_ur_flat = yskyp_ur[0,:,:]
#     yskyp_ll_flat = yskyp_ll[0,:,:]
#     yskyp_lr_flat = yskyp_lr[0,:,:]
#     yskyp_ul_flat = yskyp_ul[0,:,:]
#
#     val_sgns = np.zeros(yskyp_ur_flat.shape)
#     val_sgns += np.sign(yskyp_ur_flat)
#     val_sgns += np.sign(yskyp_ll_flat)
#     val_sgns += np.sign(yskyp_lr_flat)
#     val_sgns += np.sign(yskyp_ul_flat)
#
#     whgood = np.where( ( np.abs(val_sgns) < 4. ) & (gal.data.mask) )
#
#     xgal_flat = xgal[0,:,:]
#     ygal_flat = ygal[0,:,:]
#     xgal_list = xgal_flat[whgood]
#     ygal_list = ygal_flat[whgood]
#
#     # The circular velocity at each position only depends on the radius
#     # Convert to kpc
#     rgal = np.sqrt(xgal_list ** 2 + ygal_list ** 2) * rstep / gal.dscale
#
#     Routmax2D = np.max(rgal.flatten())
#
#     return Routmax2D

# BETTER METHOD: ALONG MAJOR AXIS:
def _calc_Rout_max_2D(gal=None, results=None):
    gal.model.update_parameters(results.bestfit_parameters)
    nx_sky = gal.data.data['velocity'].shape[1]
    ny_sky = gal.data.data['velocity'].shape[0]

    try:
        center_pixel_kin = (gal.data.xcenter + gal.model.geometry.xshift.value,
                            gal.data.ycenter + gal.model.geometry.yshift.value)
    except:
        center_pixel_kin = (np.int(nx_sky/ 2.) + gal.model.geometry.xshift.value,
                            np.int(ny_sky/ 2.) + gal.model.geometry.yshift.value)

    # Start going to neg, pos of center, at PA, and check if mask True/not
    #   in steps of pix, then rounding. if False: stop, and set 1 less as the end.
    cPA = np.cos(gal.model.components['geom'].pa.value * np.pi/180.)
    sPA = np.sin(gal.model.components['geom'].pa.value * np.pi/180.)

    ## but just considering +- MA -> all in y.
    ## xnew = -rMA * sPA
    ## ynew = rMA * cPA
    ## then for MINA -> all in x
    ## xnew2 = rMINA * cPA
    ## ynew2 = rMINA * sPA

    #rstep_A = 1.
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
            elif not gal.data.mask[np.int(np.round(ytmp)), np.int(np.round(xtmp))]:
                rMA_arr.append(-1.*rMA_tmp)  # switch sign: pos / blue for calc becomes neg
                rMA_tmp = 0
                ended_MA = True

    Routmax2D = np.max(np.abs(np.array(rMA_arr)))

    # In pixels. Convert to arcsec then to kpc:
    Routmax2D_kpc = Routmax2D * gal.instrument.pixscale.value / gal.dscale
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
    #
    config_c_m_data = Config_create_model_data(**kwargs)
    config_sim_cube = Config_simulate_cube(**kwargs)
    kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

    if outpath is None:
        raise ValueError

    if fname_model_matchdata is None:
        #fname_model_matchdata = outpath + '{}_out-1dplots_{}.txt'.format(gal.name, monthday)
        fname_model_matchdata = "{}{}_out-1dplots.txt".format(outpath, gal.name)
    if fname_finer is None:
        #fname_finer = outpath + '{}_out-1dplots_{}_finer_sampling.txt'.format(gal.name, monthday)
        fname_finer = "{}{}_out-1dplots_finer_sampling.txt".format(outpath, gal.name)

    if fname_intrinsic is None:
        fname_intrinsic = '{}{}_vcirc_tot_bary_dm.dat'.format(outpath, gal.name)
    if fname_intrinsic_m is None:
        fname_intrinsic_m = '{}{}_menc_tot_bary_dm.dat'.format(outpath, gal.name)

    ###
    galin = copy.deepcopy(gal)

    # ---------------------------------------------------------------------------
    gal.create_model_data(**kwargs_galmodel)

    # -------------------
    # Save Bary/DM vcirc:
    write_vcirc_tot_bar_dm(gal=gal, fname=fname_intrinsic, fname_m=fname_intrinsic_m)

    # --------------------------------------------------------------------------
    if (not os.path.isfile(fname_model_matchdata)) | (overwrite):
        write_model_1d_obs_file(gal=gal, fname=fname_model_matchdata)


    # Try finer scale:
    if (not os.path.isfile(fname_finer)) | (overwrite):
        # Reload galaxy object: reset things
        gal = copy.deepcopy(galin)

        write_1d_obs_finer_scale(gal=gal, fname=fname_finer, moment=moment,
                partial_weight=partial_weight, overwrite=overwrite, **kwargs_galmodel)


    return None


def write_1d_obs_finer_scale(gal=None, fname=None,
            partial_weight=True,
            moment=False,
            overwrite=False, **kwargs):
    config_c_m_data = Config_create_model_data(**kwargs)
    config_sim_cube = Config_simulate_cube(**kwargs)
    kwargs_galmodel = {**config_c_m_data.dict, **config_sim_cube.dict}

    profile1d_type = kwargs_galmodel['profile1d_type']
    aperture_radius = kwargs_galmodel['aperture_radius']

    # Try finer scale:
    rmax_abs = np.max([2.5, np.max(np.abs(gal.model_data.rarr))])
    r_step = 0.025 #0.05
    if rmax_abs > 4.:
        r_step = 0.05
    aper_centers_interp = np.arange(0, rmax_abs+r_step, r_step)

    if kwargs_galmodel['slit_pa'] is None:
        try:
            kwargs_galmodel['slit_pa'] = gal.model.geometry.pa.value
        except:
            kwargs_galmodel['slit_pa'] = gal.data.slit_pa

    if profile1d_type == 'rect_ap_cube':
        f_par = interpolate.interp1d(gal.data.rarr, gal.data.apertures.pix_parallel,
                        kind='slinear', fill_value='extrapolate')
        f_perp = interpolate.interp1d(gal.data.rarr, gal.data.apertures.pix_perp,
                        kind='slinear', fill_value='extrapolate')

        pix_parallel_interp = f_par(aper_centers_interp)
        pix_perp_interp = f_perp(aper_centers_interp)

        gal.data.apertures = aperture_classes.setup_aperture_types(gal=gal,
                    profile1d_type=profile1d_type,
                    aper_centers = aper_centers_interp,
                    aperture_radius=1.,
                    pix_perp=pix_perp_interp, pix_parallel=pix_parallel_interp,
                    pix_length=None, from_data=False,
                    slit_pa=kwargs_galmodel['slit_pa'],
                    partial_weight=partial_weight,
                    moment=moment)
    elif profile1d_type == 'circ_ap_cube':
        gal.data.apertures = aperture_classes.setup_aperture_types(gal=gal,
                    profile1d_type=profile1d_type,
                    aper_centers = aper_centers_interp,
                    aperture_radius=aperture_radius,
                    pix_perp=None, pix_parallel=None,
                    pix_length=None, from_data=False,
                    slit_pa=kwargs_galmodel['slit_pa'],
                    partial_weight=partial_weight,
                    moment=moment)

    if (profile1d_type == 'circ_ap_cube') | ( profile1d_type == 'rect_ap_cube'):
        gal.create_model_data(**kwargs_galmodel)
    else:
        gal.instrument.slit_width = gal.data.slit_width
        kwargs_galmodel_in = kwargs_galmodel.copy()
        kwargs_galmodel_in['from_data'] = False
        kwargs_galmodel_in['from_instrument'] = True
        kwargs_galmodel_in['ndim_final'] = 1
        kwargs_galmodel_in['aper_centers'] = aper_centers_interp
        kwargs_galmodel_in['slit_width'] = gal.data.slit_width
        kwargs_galmodel_in['slit_pa'] = gal.data.slit_pa
        kwargs_galmodel_in['aperture_radius'] = aperture_radius
        gal.create_model_data(**kwargs_galmodel_in)


    write_model_1d_obs_file(gal=gal, fname=fname, overwrite=overwrite)

    return None

def write_vcirc_tot_bar_dm(gal=None, fname=None, fname_m=None):
    # -------------------
    # Save Bary/DM vcirc:

    rstep = 0.1
    rmax = 40.   #17.2   # kpc
    rarr = np.arange(0, rmax+rstep, rstep)

    vcirc_bar = gal.model.components['disk+bulge'].circular_velocity(rarr)
    vcirc_dm  = gal.model.components['halo'].circular_velocity(rarr)
    vcirc_tot = gal.model.circular_velocity(rarr)

    menc_tot, menc_bar, menc_dm = gal.model.enclosed_mass(rarr)

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

    save_vcirc_tot_bar_dm_files(gal=gal, fname=fname, fname_m=fname_m,
                    profiles=profiles, profiles_m=profiles_m)

    return None


#
def save_vcirc_tot_bar_dm_files(gal=None, fname=None, fname_m=None, profiles=None, profiles_m=None):
    with open(fname, 'w') as f:
        namestr = '#   r   vcirc_tot vcirc_bar   vcirc_dm'
        f.write(namestr+'\n')
        unitstr = '#   [kpc]   [km/s]   [km/s]   [km/s]'
        f.write(unitstr+'\n')
        for i in range(profiles.shape[0]):
            datstr = '    '.join(["{0:0.3f}".format(p) for p in profiles[i,:]])
            f.write(datstr+'\n')

    with open(fname_m, 'w') as f:
        namestr = '#   r   lmenc_tot   lmenc_bar   lmenc_dm'
        f.write(namestr+'\n')
        unitstr = '#   [kpc]   [log10Msun]   [log10Msun]   [log10Msun]'
        f.write(unitstr+'\n')
        for i in range(profiles.shape[0]):
            datstr = '    '.join(["{0:0.3f}".format(p) for p in profiles_m[i,:]])
            f.write(datstr+'\n')

    return None
