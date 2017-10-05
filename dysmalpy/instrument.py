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

    def convolve(self, cube, spec_units='velocity', spec_step=None,
                 spec_center=None):
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

            cube = self.convolve_with_lsf(cube, spec_units=spec_units,
                                          spec_step=spec_step,
                                          spec_center=spec_center)

        else:

            cube_conv_beam = self.convolve_with_beam(cube)
            cube = self.convolve_with_lsf(cube_conv_beam, spec_units=spec_units,
                                          spec_step=spec_step,
                                          spec_center=spec_center)\

        return cube

    def convolve_with_lsf(self, cube, spec_units='velocity', spec_step=None,
                          spec_center=None):
        """Convolve cube with the LSF"""

        if (spec_units != 'velocity') | (spec_units != 'wavelength'):
            raise ValueError("spec_units must be either 'velocity' or "
                             "'wavelength'.")

        if (self.wavestep is None) and (spec_step is None):
            raise ValueError("Spectral step not defined. Either set "
                             "'wavestep' for this instrument or specify in"
                             " 'spec_step'.")

        elif (spec_step is not None) and (spec_units == 'velocity'):

            kernel = self.lsf.as_velocity_kernel(spec_step)

        elif (spec_step is not None) and (spec_units == 'wavelength'):

            if (self.center_wave is None) and (spec_center is None):
                raise ValueError("Center wavelength not defined in either "
                                 "the instrument or call to convolve.")

            elif (spec_center is not None):
                logger.info("Overriding the instrument central wavelength "
                            "with {}.".format(spec_center))

                kernel = self.lsf.as_wave_kernel(spec_step, spec_center)

            else:

                kernel = self.lsf.as_wave_kernel(spec_step,
                                                 self.center_wave)

        elif (self.wavestep is not None) and (spec_units == 'velocity'):

            if (self.center_wave is None) and (spec_center is None):
                raise ValueError("Center wavelength not defined in either "
                                 "the instrument or call to convolve.")

            elif (spec_center is not None):

                logger.info("Overriding the instrument central wavelength "
                            "with {}.".format(spec_center))
                velstep = ((self.wavestep /
                            spec_center.to(self.wavestep.unit)) *
                            c.c.to(u.km / u.s))

            else:

                velstep = ((self.wavestep /
                            self.center_wave.to(self.wavestep.unit)) *
                           c.c.to(u.km / u.s))

            kernel = self.lsf.as_velocity_kernel(velstep)

        elif (self.wavestep is not None) and (spec_units == 'wavelength'):

            if (self.center_wave is None) and (spec_center is None):
                raise ValueError("Center wavelength not defined in either "
                                 "the instrument or call to convolve.")

            elif (spec_center is not None):

                logger.info("Overriding the instrument central wavelength "
                            "with {}.".format(spec_center))
                kernel = self.lsf.as_wave_kernel(self.wavestep,
                                                 spec_center)

            else:

                kernel = self.lsf.as_wave_kernel(self.wavestep,
                                                 self.center_wave)

        for i in range(cube.shape[1]):
            for j in range(cube.shape[2]):
                cube[:, i, j] = apy_conv.convolve_fft(cube[:, i, j], kernel)

        return cube

    def convolve_with_beam(self, cube):

        if self.pixscale is None:
            raise ValueError("Pixelscale for this instrument has not "
                             "been set yet. Can't convolve with beam.")
        kernel = self.beam.as_kernel(self.pixscale)
        for i in range(cube.shape[0]):
            cube[i, :, :] = apy_conv.convolve_fft(cube[i, :, :], kernel)

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
