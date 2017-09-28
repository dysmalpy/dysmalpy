# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Main classes and functions for DYSMALPY for simulating the kinematics of
# a model galaxy and fitting it to observed data.


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Package imports
from .instrument import Instrument
from .models import Model

# Third party imports
import numpy as np
import astropy.cosmology as apy_cosmo
from astropy.extern import six

__all__ = ['Galaxy']

# Default cosmology
_default_cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)


class Galaxy:
    """
    The main object for simulating the kinematics of a galaxy based on
    user provided mass components.
    """

    def __init__(self, z=0, cosmo=None, model=None, instrument=None,
                 data=None, name='galaxy'):

        self._z = z
        self.name = name
        self.model = model
        self.data = data
        self.instrument = instrument
        self._cosmo = cosmo
        self.dscale = self._cosmo.arcsec_per_kpc_proper(self._z).value

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
