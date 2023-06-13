# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Extinction models for DysmalPy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import abc
import logging

# Third party imports
import numpy as np

import astropy.units as u
import astropy.cosmology as apy_cosmo


__all__ = ['ConstantDimming', 'CosmologicalDimming']

# DEFAULT COSMOLOGY
_default_cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")



class Dimming:
    r"""
    Base model for dimming component, that converts the light profile from luminosity to flux

    Notes
    -----
    This model applies the same dimming everywhere in the galaxy.

    """
    @abc.abstractmethod
    def __call__(*args, **kwargs):
        """Evaluate the luminosity to flux"""

    @abc.abstractmethod
    def luminosity_to_flux(self, *args, **kwargs):
        """Evaluate the luminosity to flux"""



class ConstantDimming(Dimming):
    r"""
    Model for constant dimming.

    Parameters
    ----------
    amp_lumtoflux : float
        Luminsoty to flux conversinon factor

    """

    def __init__(self, amp_lumtoflux=1.e-10, **kwargs):
        self.amp_lumtoflux = amp_lumtoflux
        super(ConstantDimming, self).__init__(**kwargs)


    def __call__(self, x, y, z):
        return np.ones(x.shape) * self.amp_lumtoflux

    def luminosity_to_flux(self, x, y, z):
        """Evaluate luminosity to flux"""
        return self.__call__(x, y, z)



class CosmologicalDimming(Dimming):
    r"""
    Model for constant cosmological dimming.

    """

    def __init__(self, z=0, cosmo=_default_cosmo, **kwargs):
        self._z = z
        self._cosmo = cosmo
        self._set_amp_lumtoflux()
        super(CosmologicalDimming, self).__init__(**kwargs)


    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        if value < 0:
            raise ValueError("Redshift can't be negative!")
        self._z = value

        # Reset hz:
        self._set_amp_lumtoflux()

    @property
    def cosmo(self):
        return self._cosmo

    @cosmo.setter
    def cosmo(self, new_cosmo):
        if not isinstance(new_cosmo, apy_cosmo.FLRW):
            raise TypeError("Cosmology must be an astropy.cosmology.FLRW "
                            "instance.")
        if new_cosmo is None:
            self._cosmo = _default_cosmo
        self._cosmo = new_cosmo

        # Reset amp_lumtoflux:
        self._set_amp_lumtoflux()

    def _set_amp_lumtoflux(self):
        dL = self.cosmo.luminosity_distance(self.z).to(u.cm).value
        self.amp_lumtoflux = 1. / (4.*np.pi * dL**2)

    def __call__(self, x, y, z):
        return np.ones(x.shape) * self.amp_lumtoflux

    def luminosity_to_flux(self, x, y, z):
        """Evaluate luminosity to flux"""
        return self.__call__(x, y, z)
