# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# File containing classes for defining different instruments
# to "observe" a model galaxy.

# Standard library
import logging

# Third party imports
import numpy as np
import astropy.convolution as apy_conv
import astropy.units as u
import astropy.constants as c
from radio_beam import Beam

__all__ = ["Instrument", "Beam"]

# CONSTANTS
sig_to_fwhm = 2.*np.sqrt(2.*np.log(2.))


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


class Instrument:
    """Base Class to define an instrument to observe a model galaxy with."""

    def __init__(self, beam=None, lsf=None, pixscale=None, center_wave=None,
                 wavestep=None, name='Instrument'):

        self.name = name
        self.pixscale = pixscale
        self.center_wave = center_wave
        self.wavestep = wavestep
        self.beam = beam
        self.lsf = lsf

    def convolve(self, cube):
        """
        Method to perform the convolutions in both the spatial and
        spectral space. Cube is assumed to be 3D with the first dimension
        corresponding to the spectral dimension.
        First convolve with the instrument PSF, the with the LSF.
        """

        if self.beam is None and self.lsf is None:
            # Nothing to do if a PSF and LSF aren't set for the instrument
            logger.warning("No convolution being performed since PSF and LSF "
                           "haven't been set!")
            return cube

        elif self.lsf is None:

            if self.pixscale is None:
                raise ValueError("Pixelscale for this instrument has not "
                                 "been set yet. Can't convolve with beam.")
            kernel = self.beam.as_kernel(self.pixscale)
            for i in range(cube.shape[0]):
                cube[i, :, :] = apy_conv.convolve_fft(cube[i, :, :], kernel)

        #elif self.beam is None:



        return cube

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
    def center_wave(self):
        return self._center_wave

    @center_wave.setter
    def center_wave(self, value):
        if value is None:
            self._center_wave = value
        elif not isinstance(value, u.Quantity):
            logger.warning("No units on center_wave. Assuming Angstroms.")
            self._center_wave = value * u.Angstrom
        else:
            if (u.Angstrom).is_equivalent(value):
                self._center_wave = value
            else:
                raise u.UnitsError("center_wave not in equivalent units to "
                                   "Angstoms.")

    @property
    def wavestep(self):
        return self._wavestep

    @wavestep.setter
    def wavestep(self, value):
        if value is None:
            self._wavestep = value
        elif not isinstance(value, u.Quantity):
            logger.warning("No units on wavestep. Assuming Angstoms.")
            self._wavestep = value * u.Angstrom
        else:
            if (u.Angstrom).is_equivalent(value):
                self._wavestep = value
            else:
                raise u.UnitsError("wavestep not in equivalent units to "
                                   "Angstrom.")



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
