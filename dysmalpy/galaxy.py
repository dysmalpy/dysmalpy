# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Main classes and functions for DYSMALPY for simulating the kinematics of
# a model galaxy and fitting it to observed data.


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import time
import logging
import copy

# Third party imports
import numpy as np
import astropy.cosmology as apy_cosmo
import astropy.units as u
from astropy.extern import six
import scipy.optimize as scp_opt
import scipy.interpolate as scp_interp


import dill as _pickle
#import pickle as _pickle

# Local imports
# Package imports
from dysmalpy.instrument import Instrument
from dysmalpy.models import ModelSet, calc_1dprofile, calc_1dprofile_circap_pv
from dysmalpy.data_classes import Data0D, Data1D, Data2D, Data3D
from dysmalpy.utils import apply_smoothing_2D, apply_smoothing_3D
from dysmalpy import aperture_classes
# from dysmalpy.utils import measure_1d_profile_apertures

__all__ = ['Galaxy']

# Default cosmology
_default_cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')

# Function to rebin a cube in the spatial dimension
def rebin(arr, new_2dshape):
    shape = (arr.shape[0],
             new_2dshape[0], arr.shape[1] // new_2dshape[0],
             new_2dshape[1], arr.shape[2] // new_2dshape[1])
    return arr.reshape(shape).sum(-1).sum(-2)


class Galaxy:
    """
    The main object for simulating the kinematics of a galaxy based on
    user provided mass components.
    """

    def __init__(self, z=0, cosmo=_default_cosmo, model=None, instrument=None,
                 data=None, name='galaxy', 
                 data1d=None, data2d=None, data3d=None):

        self._z = z
        self.name = name
        if model is None:
            self.model = ModelSet()
        else:
            self.model = model
        self.data = data
        
        self.data1d = data1d
        self.data2d = data2d
        self.data3d = data3d
        
        self.instrument = instrument
        self._cosmo = cosmo
        self.dscale = self._cosmo.arcsec_per_kpc_proper(self._z).value
        self.model_data = None
        self.model_cube = None

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        if value < 0:
            raise ValueError("Redshift can't be negative!")
        self._z = value

    @property
    def cosmo(self):
        return self._cosmo

    @cosmo.setter
    def cosmo(self, new_cosmo):
        if isinstance(apy_cosmo.FLRW, new_cosmo):
            raise TypeError("Cosmology must be an astropy.cosmology.FLRW "
                            "instance.")
        if new_cosmo is None:
            self._cosmo = _default_cosmo
        self._cosmo = new_cosmo
        
        
    # def get_vmax(self, r=None):
    #     if r is None:
    #         r = np.linspace(0., 25., num=251, endpoint=True)
    # 
    #     vel = self.velocity_profile(r, compute_dm=False)
    # 
    #     vmax = vel.max()
    #     return vmax

    def create_model_data(self, ndim_final=3, nx_sky=None, ny_sky=None,
                          rstep=None, spec_type='velocity', spec_step=10.,
                          spec_start=-1000., nspec=201, line_center=None,
                          spec_unit=(u.km/u.s), aper_centers=None, aper_dist=None,
                          slit_width=None, slit_pa=None, profile1d_type='circ_ap_cube',
                          from_instrument=True, from_data=True,
                          oversample=1, oversize=1, debug=False,
                          aperture_radius=None, pix_perp=None, pix_parallel=None,
                          pix_length=None,
                          skip_downsample=False, partial_aperture_weight=False, 
                          xcenter=None, ycenter=None):

        """
        Simulate an IFU cube then optionally collapse it down to a 2D
        velocity/dispersion field or 1D velocity/dispersion profile.
        
        Convention:
            slit_pa is angle of slit to left side of major axis (eg, neg r is E)
        """

        # Pull parameters from the observed data if specified
        if from_data:

            ndim_final = self.data.ndim

            if ndim_final == 3:

                nx_sky = self.data.shape[2]
                ny_sky = self.data.shape[1]
                nspec = self.data.shape[0]
                spec_ctype = self.data.data.wcs.wcs.ctype[-1]
                if spec_ctype == 'WAVE':
                    spec_type = 'wavelength'
                elif spec_ctype == 'VOPT':
                    spec_type = 'velocity'
                spec_start = self.data.data.spectral_axis[0].value
                spec_unit = self.data.data.spectral_axis.unit
                spec_step = (self.data.data.spectral_axis[1].value -
                             self.data.data.spectral_axis[0].value)
                rstep = self.data.data.wcs.wcs.cdelt[0]*3600.
                
                try:
                    xcenter = self.data.xcenter
                except:
                    pass
                try:
                    ycenter = self.data.ycenter
                except:
                    pass
                
            elif ndim_final == 2:

                nx_sky = self.data.data['velocity'].shape[1]
                ny_sky = self.data.data['velocity'].shape[0]
                rstep = self.data.pixscale
                try:
                    xcenter = self.data.xcenter
                except:
                    pass
                try:
                    ycenter = self.data.ycenter
                except:
                    pass
                if from_instrument:
                    spec_type = self.instrument.spec_type
                    spec_start = self.instrument.spec_start.value
                    spec_step = self.instrument.spec_step.value
                    spec_unit = self.instrument.spec_start.unit
                    nspec = self.instrument.nspec

            elif ndim_final == 1:

                if from_instrument:
                    nx_sky = self.instrument.fov[0]
                    ny_sky = self.instrument.fov[1]
                    spec_type = self.instrument.spec_type
                    spec_start = self.instrument.spec_start.value
                    spec_step = self.instrument.spec_step.value
                    spec_unit = self.instrument.spec_start.unit
                    nspec = self.instrument.nspec
                    rstep = self.instrument.pixscale.value
                else:

                    maxr = 1.5*np.max(np.abs(self.data.rarr))
                    if rstep is None:
                        rstep = np.mean(self.data.rarr[1:] -
                                        self.data.rarr[0:-1])/3.
                    if nx_sky is None:
                        nx_sky = int(np.ceil(maxr/rstep))
                    if ny_sky is None:
                        ny_sky = int(np.ceil(maxr/rstep))

                slit_width = self.data.slit_width
                slit_pa = self.data.slit_pa
                aper_centers = self.data.rarr
                
                try:
                    xcenter = self.data.xcenter
                except:
                    pass
                try:
                    ycenter = self.data.ycenter
                except:
                    pass

            elif ndim_final == 0:

                if from_instrument:
                    nx_sky = self.instrument.fov[0]
                    ny_sky = self.instrument.fov[1]
                    spec_type = self.instrument.spec_type
                    spec_start = self.instrument.spec_start.value
                    spec_step = self.instrument.spec_step.value
                    spec_unit = self.instrument.spec_start.unit
                    nspec = self.instrument.nspec
                    rstep = self.instrument.pixscale.value

                else:

                    if (nx_sky is None) | (ny_sky is None) | (rstep is None):

                        raise ValueError("At minimum, nx_sky, ny_sky, and rstep must "
                                         "be set if from_instrument and/or from_data"
                                         " is False.")

                slit_width = self.data.slit_width
                slit_pa = self.data.slit_pa
                xarr = self.data.x

        # Pull parameters from the instrument
        elif from_instrument:

            nx_sky = self.instrument.fov[0]
            ny_sky = self.instrument.fov[1]
            spec_type = self.instrument.spec_type
            spec_start = self.instrument.spec_start.value
            spec_step = self.instrument.spec_step.value
            spec_unit = self.instrument.spec_start.unit
            nspec = self.instrument.nspec
            rstep = self.instrument.pixscale.value
            
            try:
                slit_width = self.instrument.slit_width
            except:
                pass
            
        else:

            if (nx_sky is None) | (ny_sky is None) | (rstep is None):

                raise ValueError("At minimum, nx_sky, ny_sky, and rstep must "
                                 "be set if from_instrument and/or from_data"
                                 " is False.")
                                 
        sim_cube, spec = self.model.simulate_cube(nx_sky=nx_sky,
                                                  ny_sky=ny_sky,
                                                  dscale=self.dscale,
                                                  rstep=rstep,
                                                  spec_type=spec_type,
                                                  spec_step=spec_step,
                                                  nspec=nspec,
                                                  spec_start=spec_start,
                                                  spec_unit=spec_unit,
                                                  oversample=oversample,
                                                  oversize=oversize, 
                                                  xcenter=xcenter, 
                                                  ycenter=ycenter)
                                                  
        # Correct for any oversampling
        if (oversample > 1) & (not skip_downsample): 
            sim_cube_nooversamp = rebin(sim_cube, (ny_sky*oversize, nx_sky*oversize))
        else:
            sim_cube_nooversamp = sim_cube
        
        if skip_downsample:
            rstep /= (1.*oversample)
            nx_sky *= oversample
            ny_sky *= oversample
            # Fix instrument:
            self.instrument.pixscale = rstep * u.arcsec
            self.instrument.fov = [nx_sky, ny_sky]
            self.instrument.set_beam_kernel()

        #if debug:
        #self.model_cube_no_convolve = sim_cube_nooversamp

        # Apply beam smearing and/or instrumental spreading
        if self.instrument is not None:
            sim_cube_obs = self.instrument.convolve(cube=sim_cube_nooversamp,
                                                    spec_center=line_center)
        else:
            sim_cube_obs = sim_cube_nooversamp


        # Re-size the cube back down
        if oversize > 1:
            nx_oversize = sim_cube_obs.shape[2]
            ny_oversize = sim_cube_obs.shape[1]
            sim_cube_final = sim_cube_obs[:,
                np.int(ny_oversize/2 - ny_sky/2):np.int(ny_oversize/2+ny_sky/2),
                np.int(nx_oversize/2 - nx_sky/2):np.int(nx_oversize/2+nx_sky/2)]

        else:
            sim_cube_final = sim_cube_obs

            
        self.model_cube = Data3D(cube=sim_cube_final, pixscale=rstep,
                                 spec_type=spec_type, spec_arr=spec,
                                 spec_unit=spec_unit)

        if ndim_final == 3:
            # sim_cube_flat = np.sum(sim_cube_obs*self.data.mask, axis=0)
            # data_cube_flat = np.sum(self.data.data.unmasked_data[:].value*self.data.mask, axis=0)
            # errsq_cube_flat = np.sum( ( self.data.error.unmasked_data[:].value**2 )*self.data.mask, axis=0)
            #
            # # Fill errsq_cube_flat == 0 of *masked* parts with 99.,
            # #   so that later (data*sim/errsq) * mask is finite (and contributes nothing)
            # # Potentially make this a *permanent mask* that can be accessed for faster calculations?
            # mask_flat = np.sum(self.data.mask, axis=0)/self.data.mask.shape[0]
            # mask_flat[mask_flat != 0] = 1.
            # errsq_cube_flat[((errsq_cube_flat == 0.) & (mask_flat==0))] = 99.
            #
            # if self.model.per_spaxel_norm_3D:
            # Do normalization on a per-spaxel basis -- eg, don't care about preserving
            #   M/L ratio information from model.
            # collapse in spectral dimension only: axis 0

            if from_data:
                if self.data.flux_map is None:
                    num = np.sum(self.data.mask*(self.data.data.unmasked_data[:].value*
                                     sim_cube_final/(self.data.error.unmasked_data[:].value**2)), axis=0)
                    den = np.sum(self.data.mask*
                                    (sim_cube_final**2/(self.data.error.unmasked_data[:].value**2)), axis=0)
                    scale = np.abs(num/den)
                    scale3D = np.zeros(shape=(1, scale.shape[0], scale.shape[1],))
                    scale3D[0, :, :] = scale
                    sim_cube_final *= scale3D
                    # else:
                    #     scale = np.sum( mask_flat*(data_cube_flat*sim_cube_flat / errsq_cube_flat) )/\
                    #                 np.sum( mask_flat*(sim_cube_flat**2 / errsq_cube_flat) )
                    #     sim_cube_obs *= scale

                    # Throw a non-implemented error if smoothing + 3D model:
                    if from_data:
                        if self.data.smoothing_type is not None:
                            raise NotImplementedError('Smoothing for 3D output not implemented yet!')

                else:

                    model_peak = np.nanmax(sim_cube_final, axis=0)
                    scale = self.data.flux_map/model_peak
                    scale3D = np.zeros((1, scale.shape[0], scale.shape[1]))
                    scale3D[0, :, :] = scale
                    sim_cube_final *= scale3D
            
            self.model_data = Data3D(cube=sim_cube_final, pixscale=rstep,
                                     spec_type=spec_type, spec_arr=spec,
                                     spec_unit=spec_unit)

        elif ndim_final == 2:
            
            if from_data:
                if self.data.smoothing_type is not None:
                    self.model_cube.data = apply_smoothing_3D(self.model_cube.data,
                                smoothing_type=self.data.smoothing_type,
                                smoothing_npix=self.data.smoothing_npix)
                                
            if spec_type == "velocity":
                if self.data.moment_calc:
                    vel = self.model_cube.data.moment1().to(u.km/u.s).value
                    disp = self.model_cube.data.linewidth_sigma().to(u.km/u.s).value
                else:
                    vel = lksjdfldksf
                    disp = klsjdflkdf
            elif spec_type == "wavelength":

                cube_with_vel = self.model_cube.data.with_spectral_unit(u.km/u.s, 
                    velocity_convention='optical',
                    rest_value=line_center)

                if self.data.moment_calc:
                    vel = cube_with_vel.moment1().value
                    disp = cube_with_vel.linewidth_sigma().value
                else:
                    vel = lksjdflksdjf
                    disp = lksjdflksdjflskd
                    
                disp[np.isnan(disp)] = 0.

            else:
                raise ValueError("spec_type can only be 'velocity' or "
                                 "'wavelength.'")
            
            # if from_data:
            #     if self.data.smoothing_type is not None:
            #         vel, disp = apply_smoothing_2D(vel, disp,
            #                     smoothing_type=self.data.smoothing_type,
            #                     smoothing_npix=self.data.smoothing_npix)
            
            self.model_data = Data2D(pixscale=rstep, velocity=vel,
                                     vel_disp=disp)

        elif ndim_final == 1:

            if spec_type == 'wavelength':

                cube_with_vel = self.model_cube.data.with_spectral_unit(
                    u.km / u.s, velocity_convention='optical',
                    rest_value=line_center)

                cube_data = cube_with_vel.unmasked_data[:]
                vel_arr = cube_with_vel.spectral_axis.to(u.km/u.s).value

            elif spec_type == 'velocity':

                cube_data = sim_cube_obs
                vel_arr = spec

            else:
                raise ValueError("spec_type can only be 'velocity' or "
                                 "'wavelength.'")

            if profile1d_type == 'circ_ap_pv':
                r1d, flux1d, vel1d, disp1d = calc_1dprofile_circap_pv(cube_data, slit_width,
                                                    slit_pa-180., rstep, vel_arr)
                vinterp = scp_interp.interp1d(r1d, vel1d,
                                              fill_value='extrapolate')
                disp_interp = scp_interp.interp1d(r1d, disp1d,
                                                  fill_value='extrapolate')
                vel1d = vinterp(aper_centers)
                disp1d = disp_interp(aper_centers)
                # flux1d = aper_centers*0. + np.NaN
                flux_interp = scp_interp.interp1d(r1d, flux1d,
                                                  fill_value='extrapolate')
                flux1d = flux_interp(aper_centers)
                aper_model = None
            
            elif profile1d_type == 'single_pix_pv':
                r1d, flux1d, vel1d, disp1d = calc_1dprofile(cube_data, slit_width,
                                                    slit_pa-180., rstep, vel_arr)
                vinterp = scp_interp.interp1d(r1d, vel1d,
                                              fill_value='extrapolate')
                disp_interp = scp_interp.interp1d(r1d, disp1d,
                                                  fill_value='extrapolate')
                vel1d = vinterp(aper_centers)
                disp1d = disp_interp(aper_centers)
                # flux1d = aper_centers*0. + np.NaN
                
                flux_interp = scp_interp.interp1d(r1d, flux1d,
                                                  fill_value='extrapolate')
                flux1d = flux_interp(aper_centers)
                
                aper_model = None
            else:
                
                if from_data:
                    if (self.data.aper_center_pix_shift is not None):
                        center_pixel = (np.int(nx_sky / 2) + self.data.aper_center_pix_shift[0],
                                        np.int(ny_sky / 2) + self.data.aper_center_pix_shift[1])
                    else:
                        center_pixel = None
                else:
                    center_pixel = None
                
                # raise ValueError
                
                #----------------------------------------------------------
                #try:
                if from_data:
                    aper_centers, flux1d, vel1d, disp1d = self.data.apertures.extract_1d_kinematics(spec_arr=vel_arr, 
                            cube=cube_data, center_pixel = center_pixel, pixscale=rstep)
                    aper_model = None
                    
                # except:
                #     raise TypeError('Unknown method for measuring the 1D profiles.')
                
                #----------------------------------------------------------
                else:
                    
                    aper_model = aperture_classes.setup_aperture_types(gal=self, 
                                profile1d_type=profile1d_type, 
                                slit_width = slit_width, aper_centers=aper_centers, slit_pa=slit_pa, 
                                aperture_radius=aperture_radius, 
                                pix_perp=pix_perp, pix_parallel=pix_parallel,
                                pix_length=pix_length, 
                                partial_weight=partial_aperture_weight, 
                                from_data=False)
                    
                    
                    aper_centers, flux1d, vel1d, disp1d = aper_model.extract_1d_kinematics(spec_arr=vel_arr, 
                            cube=cube_data, center_pixel = center_pixel, pixscale=rstep)



            self.model_data = Data1D(r=aper_centers, velocity=vel1d,
                                     vel_disp=disp1d, flux=flux1d, 
                                     slit_width=slit_width,
                                     slit_pa=slit_pa)
            self.model_data.apertures = aper_model

        elif ndim_final == 0:

            if self.data.integrate_cube:

                # Integrate over the spatial dimensions of the cube
                flux = np.nansum(np.nansum(self.model_cube.data.unmasked_data[:], axis=2), axis=1)

                # Normalize to the maximum of the spectrum
                flux /= np.nanmax(flux)
                flux = flux.value

            else:

                # Place slit down on cube
                raise NotImplementedError('Using slits to create spectrum not implemented yet!')

            self.model_data = Data0D(x=spec, flux=flux, slit_pa=self.data.slit_pa,
                                     slit_width=self.data.slit_width, integrate_cube=self.data.integrate_cube,
                                    )
                                    
        #
        # Reset instrument to orig value
        if skip_downsample:
            rstep *= oversample
            nx_sky /= (1.*oversample)
            ny_sky /= (1.*oversample)
            # Fix instrument:
            self.instrument.pixscale = rstep * u.arcsec
            self.instrument.fov = [nx_sky, ny_sky]
            self.instrument.set_beam_kernel()




    #
    def preserve_self(self, filename=None, save_data=True):
        # def save_galaxy_model(self, galaxy=None, filename=None):
        if filename is not None:
            galtmp = copy.deepcopy(self)
            
            galtmp.filename_velocity = copy.deepcopy(galtmp.data.filename_velocity)
            galtmp.filename_dispersion = copy.deepcopy(galtmp.data.filename_dispersion)
            
            if not save_data:
                galtmp.data = None
                galtmp.model_data = None
                galtmp.model_cube = None
            
            # galtmp.instrument = copy.deepcopy(galaxy.instrument)
            # galtmp.model = modtmp
            
            #dump_pickle(galtmp, filename=filename) # Save mcmcResults class
            _pickle.dump(galtmp, open(filename, "wb") )
            
            return None
            
    def load_self(self, filename=None):
        if filename is not None:
            galtmp = _pickle.load(open(filename, "rb"))
            # Reset
            #self = copy.deepcopy(galtmp)
            #return self
            return galtmp
            
            
def load_galaxy_object(filename=None):
    gal = Galaxy()
    gal = gal.load_self(filename=filename)
    return gal
    