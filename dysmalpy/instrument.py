# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# File containing classes for defining different instruments
# to "observe" a model galaxy.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np
import astropy.convolution as apy_conv
from scipy.signal import fftconvolve
import astropy.units as u
import astropy.constants as c
from radio_beam import Beam

__all__ = ["Instrument", "Beam", "LSF"]

# CONSTANTS
sig_to_fwhm = 2.*np.sqrt(2.*np.log(2.))

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


class Instrument:
    """Base Class to define an instrument to observe a model galaxy with."""

    def __init__(self, beam=None, beam_type=None, lsf=None, pixscale=None,
                 spec_type='velocity', spec_start=-1000*u.km/u.s,
                 spec_step=10*u.km/u.s, nspec=201,
                 fov=None, name='Instrument'):

        self.name = name
        self.pixscale = pixscale
        
        # Case of two beams: analytic and empirical: if beam_type==None, assume analytic
        self.beam = beam
        self.beam_type = beam_type
            
        self._beam_kernel = None
        self.lsf = lsf
        self._lsf_kernel = None
        self.fov = fov
        self.spec_type = spec_type
        self.spec_start = spec_start
        self.spec_step = spec_step
        self.nspec = nspec

    def convolve(self, cube, spec_center=None):
        """
        Method to perform the convolutions in both the spatial and
        spectral space. Cube is assumed to be 3D with the first dimension
        corresponding to the spectral dimension.
        First convolve with the instrument PSF, then with the LSF.
        """

        if self.beam is None and self.lsf is None:
            # Nothing to do if a PSF and LSF aren't set for the instrument
            logger.warning("No convolution being performed since PSF and LSF "
                           "haven't been set!")
            return cube

        elif self.lsf is None:

            cube = self.convolve_with_beam(cube)

        elif self.beam is None:

            cube = self.convolve_with_lsf(cube, spec_center=spec_center)

        else:

            cube_conv_beam = self.convolve_with_beam(cube)
            cube = self.convolve_with_lsf(cube_conv_beam,
                                          spec_center=spec_center)

        return cube

    def convolve_with_lsf(self, cube, spec_center=None):
        """Convolve cube with the LSF"""

        if self._lsf_kernel is None:
            self.set_lsf_kernel(spec_center=spec_center)

        cube = fftconvolve(cube, self._lsf_kernel, mode='same')

        return cube

    def convolve_with_beam(self, cube):

        if self.pixscale is None:
            raise ValueError("Pixelscale for this instrument has not "
                             "been set yet. Can't convolve with beam.")

        if self._beam_kernel is None:
            self.set_beam_kernel()

        cube = fftconvolve(cube, self._beam_kernel, mode='same')

        return cube

    def set_beam_kernel(self):
        if (self.beam_type == 'analytic') | (self.beam_type == None):
            kernel = self.beam.as_kernel(self.pixscale)
            kern2D = kernel.array
        elif self.beam_type == 'empirical':
            if len(self.beam.shape) == 1:
                raise ValueError("1D beam/PSF not currently supported")
            kern2D = self.beam.copy()
            # Replace NaNs/non-finite with zero:
            kern2D[~np.isfinite(kern2D)] = 0.
            # Replace < 0 with zero:
            kern2D[kern2D<0.] = 0.
            kern2D /= np.sum(kern2D[np.isfinite(kern2D)])  # need to normalize
        kern3D = np.zeros(shape=(1, kern2D.shape[0], kern2D.shape[1],))
        kern3D[0, :, :] = kern2D

        self._beam_kernel = kern3D

    def set_lsf_kernel(self, spec_center=None):

        if (self.spec_step is None):
            raise ValueError("Spectral step not defined.")

        elif (self.spec_step is not None) and (self.spec_type == 'velocity'):

            kernel = self.lsf.as_velocity_kernel(self.spec_step)

        elif (self.spec_step is not None) and (self.spec_type == 'wavelength'):

            if (self.spec_center is None) and (spec_center is None):
                raise ValueError("Center wavelength not defined in either "
                                 "the instrument or call to convolve.")

            elif (spec_center is not None):
                #logger.info("Overriding the instrument central wavelength "
                #            "with {}.".format(spec_center))

                kernel = self.lsf.as_wave_kernel(self.spec_step, spec_center)

            else:

                kernel = self.lsf.as_wave_kernel(self.spec_step,
                                                 self.spec_center)

        kern1D = kernel.array
        kern3D = np.zeros(shape=(kern1D.shape[0], 1, 1,))
        kern3D[:, 0, 0] = kern1D

        self._lsf_kernel = kern3D


    @property
    def beam(self):
        return self._beam

    @beam.setter
    def beam(self, new_beam):
        if isinstance(new_beam, Beam):
            self._beam = new_beam
        elif new_beam is None:
            self._beam = None
        else:
            raise TypeError("Beam must be an instance of "
                            "radio_beam.beam.Beam")

    @property
    def lsf(self):
        return self._lsf

    @lsf.setter
    def lsf(self, new_lsf):
        self._lsf = new_lsf

    @property
    def pixscale(self):
        return self._pixscale

    @pixscale.setter
    def pixscale(self, value):
        if value is None:
            self._pixscale = value
        elif not isinstance(value, u.Quantity):
            logger.warning("No units on pixscale. Assuming arcseconds.")
            self._pixscale = value*u.arcsec
        else:
            if (u.arcsec).is_equivalent(value):
                self._pixscale = value
            else:
                raise u.UnitsError("pixscale not in equivalent units to "
                                   "arcseconds.")

    @property
    def spec_step(self):
        return self._spec_step

    @spec_step.setter
    def spec_step(self, value):
        if value is None:
            self._spec_step = value
        elif not isinstance(value, u.Quantity):
            logger.warning("No units on spec_step. Assuming km/s.")
            self._spec_step = value * u.km/u.s
        else:
            self._spec_step = value


    @property
    def spec_center(self):
        if (self.spec_start is None) | (self.nspec is None):
            return None
        else:
            return (self.spec_start + np.round(self.nspec/2)*self.spec_step)



class LSF(u.Quantity):
    """
    An object to handle line spread functions.
    """

    def __new__(cls, dispersion=None, default_unit=u.km/u.s, meta=None):
        """
        Create a new Gaussian Line Spread Function
        Parameters
        ----------
        dispersion : :class:`~astropy.units.Quantity` with speed equivalency
        default_unit : :class:`~astropy.units.Unit`
            The unit to impose on dispersion if they are specified as floats
        """

        # TODO: Allow for wavelength dispersion to be specified

        # error checking

        # give specified values priority
        if dispersion is not None:
            if (u.km/u.s).is_equivalent(dispersion):
                dispersion = dispersion
            else:
                logger.warning("Assuming dispersion has been specified in "
                               "km/s.")
                dispersion = dispersion * default_unit

        self = super(LSF, cls).__new__(cls, dispersion.value, u.km/u.s)
        self._dispersion = dispersion
        self.default_unit = default_unit

        if meta is None:
            self.meta = {}
        elif isinstance(meta, dict):
            self.meta = meta
        else:
            raise TypeError("metadata must be a dictionary")

        return self

    def __repr__(self):
        return "LSF: Vel. Disp. = {0}".format(
            self.dispersion.to(self.default_unit))

    def __str__(self):
        return self.__repr__()

    @property
    def dispersion(self):
        return self._dispersion

    def vel_to_lambda(self, wave):
        """
        Convert from velocity dispersion to wavelength dispersion for
        a given central wavelength.
        """

        if not isinstance(wave, u.Quantity):
            raise TypeError("wave must be a Quantity object. "
                            "Try 'wave*u.Angstrom' or another equivalent unit.")
        return (self.dispersion/c.c.to(self.dispersion.unit))*wave

    def as_velocity_kernel(self, velstep, **kwargs):
        """
        Return a Gaussian convolution kernel in velocity space
        """

        sigma_pixel = (self.dispersion.value /
                       velstep.to(self.dispersion.unit).value)

        return apy_conv.Gaussian1DKernel(sigma_pixel, **kwargs)

    def as_wave_kernel(self, wavestep, wavecenter, **kwargs):
        """
        Return a Gaussian convolution kernel in wavelength space
        """

        sigma_pixel = self.vel_to_lambda(wavecenter).value/wavestep.value

        return apy_conv.Gaussian1DKernel(sigma_pixel, **kwargs)
