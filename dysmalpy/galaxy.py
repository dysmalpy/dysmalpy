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

# Third party imports
import numpy as np
import astropy.cosmology as apy_cosmo
import astropy.units as u
from astropy.extern import six
import scipy.optimize as scp_opt
import scipy.interpolate as scp_interp

# Local imports
# Package imports
from .instrument import Instrument
from .models import ModelSet, calc_1dprofile
from .data_classes import Data1D, Data2D, Data3D

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
                 data=None, name='galaxy'):

        self._z = z
        self.name = name
        if model is None:
            self.model = ModelSet()
        else:
            self.model = model
        self.data = data
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

    def create_model_data(self, ndim_final=3, nx_sky=None, ny_sky=None,
                          rstep=None, spec_type='velocity', spec_step=10.,
                          spec_start=-1000., nspec=201, line_center=None,
                          spec_unit=(u.km/u.s), aper_centers=None,
                          slit_width=None, slit_pa=None,
                          from_instrument=True, from_data=True,
                          oversample=1):

        """Simulate an IFU cube then optionally collapse it down to a 2D
        velocity/dispersion field or 1D velocity/dispersion profile."""

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

            elif ndim_final == 2:

                nx_sky = self.data.data['velocity'].shape[1]
                ny_sky = self.data.data['velocity'].shape[0]
                rstep = self.data.pixscale
                if from_instrument:
                    spec_type = 'wavelength'
                    spec_start = self.instrument.wave_start.value
                    spec_step = self.instrument.wave_step.value
                    spec_unit = self.instrument.wave_start.unit
                    nspec = self.instrument.nwave

            elif ndim_final == 1:

                if from_instrument:
                    nx_sky = self.instrument.fov[0]
                    ny_sky = self.instrument.fov[1]
                    spec_type = 'wavelength'
                    spec_start = self.instrument.wave_start.value
                    spec_step = self.instrument.wave_step.value
                    spec_unit = self.instrument.wave_start.unit
                    nspec = self.instrument.nwave
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

        # Pull parameters from the instrument
        elif from_instrument:

            nx_sky = self.instrument.fov[0]
            ny_sky = self.instrument.fov[1]
            spec_type = 'wavelength'
            spec_start = self.instrument.wave_start.value
            spec_step = self.instrument.wave_step.value
            spec_unit = self.instrument.wave_start.unit
            nspec = self.instrument.nwave
            rstep = self.instrument.pixscale.value

        else:

            if (nx_sky is None) | (ny_sky is None) | (rstep is None):

                raise ValueError("At minimum, nx_sky, ny_sky, and rstep must "
                                 "be set if from_instrument and/or from_data"
                                 " is False.")
        time1 = time.time()
        sim_cube, spec = self.model.simulate_cube(nx_sky=nx_sky,
                                                  ny_sky=ny_sky,
                                                  dscale=self.dscale,
                                                  rstep=rstep,
                                                  spec_type=spec_type,
                                                  spec_step=spec_step,
                                                  nspec=nspec,
                                                  spec_start=spec_start,
                                                  line_center=line_center,
                                                  oversample=oversample)
        time2 = time.time()

        # Correct for any oversampling
        if oversample > 1:
            sim_cube_nooversamp = rebin(sim_cube, (ny_sky, nx_sky))
        else:
            sim_cube_nooversamp = sim_cube

        # Apply beam smearing and/or instrumental spreading
        if self.instrument is not None:
            sim_cube_obs = self.instrument.convolve(cube=sim_cube_nooversamp,
                                                    spec_type=spec_type,
                                                    spec_step=spec_step,
                                                    spec_center=line_center)
            time3 = time.time()
        else:
            sim_cube_obs = sim_cube_nooversamp

        self.model_cube = Data3D(cube=sim_cube_obs, pixscale=rstep,
                                 spec_type=spec_type, spec_arr=spec,
                                 spec_unit=spec_unit)

        if ndim_final == 3:
            sim_cube_flat = np.sum(sim_cube_obs*self.data.mask, axis=0)
            data_cube_flat = np.sum(self.data.data.unmasked_data*self.data.mask, axis=0)
            errsq_cube_flat = np.sum( ( self.data.error.unmasked_data**2 )*self.data.mask, axis=0)

            scale = np.sum( data_cube_flat*sim_cube_flat / errsq_cube_flat )/\
                        np.sum( sim_cube_flat**2 / errsq_cube_flat )
            sim_cube_obs *= scale
            self.model_data = Data3D(cube=sim_cube_obs, pixscale=rstep,
                                     spec_type=spec_type, spec_arr=spec,
                                     spec_unit=spec_unit)

        elif ndim_final == 2:

            if spec_type == "velocity":

                vel = self.model_cube.data.moment1().value
                disp = self.model_cube.data.linewidth_sigma().value

            elif spec_type == "wavelength":

                cube_with_vel = self.model_cube.data.with_spectral_unit(
                    u.km/u.s, velocity_convention='optical',
                    rest_value=line_center*spec_unit)

                vel = cube_with_vel.moment1().value
                disp = cube_with_vel.linewidth_sigma().value
                disp[np.isnan(disp)] = 0.

            else:
                raise ValueError("spec_type can only be 'velocity' or "
                                 "'wavelength.'")

            self.model_data = Data2D(pixscale=rstep, velocity=vel,
                                     vel_disp=disp)

        elif ndim_final == 1:

            if spec_type == 'wavelength':

                cube_with_vel = self.model_cube.data.with_spectral_unit(
                    u.km / u.s, velocity_convention='optical',
                    rest_value=line_center * spec_unit)

                cube_data = cube_with_vel.unmasked_data[:]
                vel_arr = cube_with_vel.spectral_axis.to(u.km/u.s).value

            elif spec_type == 'velocity':

                cube_data = sim_cube
                vel_arr = spec

            else:
                raise ValueError("spec_type can only be 'velocity' or "
                                 "'wavelength.'")

            r1d, vel1d, disp1d = calc_1dprofile(cube_data, slit_width,
                                                slit_pa, rstep, vel_arr)

            vinterp = scp_interp.interp1d(r1d, vel1d,
                                          fill_value='extrapolate')
            disp_interp = scp_interp.interp1d(r1d, disp1d,
                                              fill_value='extrapolate')
            vel1d = vinterp(aper_centers)
            disp1d = disp_interp(aper_centers)
            time4 = time.time()
            self.model_data = Data1D(r=aper_centers, velocity=vel1d,
                                     vel_disp=disp1d, slit_width=slit_width,
                                     slit_pa=slit_pa)

            print(time2-time1)
            print(time3-time2)
            print(time4-time3)
