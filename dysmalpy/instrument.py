# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# File containing classes for defining different instruments
# to "observe" a model galaxy.

# Standard library
import logging
import abc

# Third party imports
import numpy as np
import astropy.convolution as apy_conv
import astropy.units as u
from radio_beam import Beam

__all__ = ["Instrument", "Beam"]

# CONSTANTS
sig_to_fwhm = 2.*np.sqrt(2.*np.log(2.))

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')

class Instrument:
    """Base Class to define an instrument to observe a model galaxy with."""

    def __init__(self, beam=None, lsf=None, pixscale=None,
                 name='Instrument'):

        self.name = name
        self.pixscale = pixscale
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
            kernel = self.beam.as_kernel(self.pixscale*u.arcsec)
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
        else:
            raise TypeError("Beam must be an instance of"
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
        self._pixscale = value


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
                dispersion= dispersion
            else:
                warnings.warn("Assuming dispersion has been specified in "
                              "km/s.")
                dispersion = dispersion * default_unit

        # some sensible defaults
        if minor is None:
            minor = major

        self = super(Beam, cls).__new__(cls, _to_area(major,minor).value, u.sr)
        self._major = major
        self._minor = minor
        self._pa = pa
        self.default_unit = default_unit

        if meta is None:
            self.meta = {}
        elif isinstance(meta, dict):
            self.meta = meta
        else:
            raise TypeError("metadata must be a dictionary")

        return self