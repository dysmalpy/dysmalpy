# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# ZHeight models for DysmalPy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np

# Local imports
from .base import _DysmalFittable1DModel
from dysmalpy.parameters import DysmalParameter

__all__ = ['ZHeightGauss', 'ZHeightExp']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')

np.warnings.filterwarnings('ignore')


# ******* Z-Height Profiles ***************
class ZHeightProfile(_DysmalFittable1DModel):
    """Base object for flux profiles in the z-direction"""
    _type = 'zheight'

    # Must set property z_scalelength for each subclass,
    #   for use with getting indices ai for filling simulated cube

class ZHeightGauss(ZHeightProfile):
    r"""
    Gaussian flux distribution in the z-direction

    Parameters
    ----------
    sigmaz : float
        Dispersion of the Gaussian in kpc

    Notes
    -----
    Model formula:

    .. math::

        F_z = \exp\left\{\frac{-z^2}{2\sigma_z^2}\right\}
    """
    sigmaz = DysmalParameter(default=1.0, fixed=True, bounds=(0, 10))

    def __init__(self, **kwargs):
        super(ZHeightGauss, self).__init__(**kwargs)

    @staticmethod
    def evaluate(z, sigmaz):
        return np.exp(-0.5*(z/sigmaz)**2)

    @property
    def z_scalelength(self):
        return self.sigmaz


class ZHeightExp(ZHeightProfile):
    r"""
    Exponential flux distribution in the z-direction

    Parameters
    ----------
    hz : float
        Scale length of the exponential in kpc

    Notes
    -----
    Model formula:

    .. math::

        F_z = \exp\left\{\frac{-z}{h_z}\right\}
    """
    hz = DysmalParameter(default=1.0, fixed=True, bounds=(0, 10))

    def __init__(self, **kwargs):
        super(ZHeightExp, self).__init__(**kwargs)

    @staticmethod
    def evaluate(z, hz):
        return np.exp(-z/hz)

    @property
    def z_scalelength(self):
        return self.hz
