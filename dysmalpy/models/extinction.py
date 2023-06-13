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

# Local imports
from .base import _DysmalFittable3DModel
from dysmalpy.parameters import DysmalParameter

__all__ = ['ThinCentralPlaneDustExtinction', 'ForegroundConstantExtinction',
           'ForegroundExponentialExtinction']

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")



class DustExtinction(_DysmalFittable3DModel):
    r"""
    Base model for dust extinction component, that attenuates the light profile.

    Notes
    -----
    This model places the extinction within the model cube.
    All positions that will have their flux reduced by the returned attenuation cube, :math:`A` :

        .. math::

            F = A * F_{\rm intrinsic}

    where all values in :math:`A` are between 0 and 1.
    """

    _type = 'extinction'
    outputs = ('A',)

    @abc.abstractmethod
    def attenuation_cube(self, *args, **kwargs):
        """Evaluate the flux attenuation (linear multiplier) at all locations of the cube"""



class ThinCentralPlaneDustExtinction(DustExtinction):
    r"""
    Model for extinction due to a thin plane of dust

    Parameters
    ----------
    inc : float
        Inclination of the dust plane in deg

    pa : float
        Position angle of the dust plane in deg

    xshift, yshift : float
        Offset in pixels of the center of the dust plane

    amp_extinct : float
        Strength of the extinction through the dust plane. Expressed as the fraction of
        flux that is transmitted through the dust plane. `amp_extinct` = 1 means 100% attenuation.

    Notes
    -----
    This model places a dust plane within the model cube. All positions that
    are behind the dust plane relative to the line of sight will have their
    flux reduced by `amp_extinct`:

        .. math::

            F = AF_{\rm intrinsic}

    where :math:`A` is between 0 and 1.
    """

    inc = DysmalParameter(default=45.0, bounds=(0, 90))
    pa = DysmalParameter(default=0.0, bounds=(-180, 180))
    xshift = DysmalParameter(default=0.0)
    yshift = DysmalParameter(default=0.0)
    amp_extinct = DysmalParameter(default=0.0, bounds=(0., 1.))  # default: No attenuation

    @staticmethod
    def evaluate(x, y, z, inc, pa, xshift, yshift, amp_extinct):
        inc = np.pi / 180. * inc
        pa = np.pi / 180. * (pa - 90.)

        xsky = x - xshift
        ysky = y - yshift
        zsky = z

        ytmp = -xsky * np.sin(pa) + ysky * np.cos(pa)

        ydust = ytmp * np.cos(inc) - zsky * np.sin(inc)

        zsky_dust = ydust * np.sin(-inc)
        extinction = np.ones(x.shape)
        extinction[zsky <= zsky_dust] = (1.-amp_extinct)

        return extinction

    def attenuation_cube(self, x, y, z):
        """Evaluate the flux attenuation (linear multiplier) at all locations of the cube"""
        return self.evaluate(x, y, z, self.inc, self.pa, self.xshift, self.yshift, self.amp_extinct)

class ForegroundConstantExtinction(DustExtinction):
    r"""
    Model for extinction due to a foreground plane of dust

    Parameters
    ----------
    Alam : float
        Magnitude of attenuation of the dust plane (at the tracer wavelength).

    Notes
    -----
    This model places a dust dust as a foreground screen before the entire cube,
    and so is insenstive to the exact galaxy / sky geometry.

    All positions that will have their
    flux reduced by :math:`A=10^{-0.4A_{\lambda}}`, with

        .. math::

            F = AF_{\rm intrinsic}

    where :math:`A` is between 0 and 1.
    """

    Alam = DysmalParameter(default=0.0, bounds=(0., 10.))  # default: No attenuation

    @staticmethod
    def evaluate(x, y, z, Alam):
        amp_extinct = np.power(10., -0.4*Alam)
        extinction = np.ones(x.shape) * amp_extinct
        return extinction

    def attenuation_cube(self, x, y, z):
        """Evaluate the flux attenuation (linear multiplier) at all locations of the cube"""
        return self.evaluate(x, y, z, self.Alam)


class ForegroundExponentialExtinction(DustExtinction):
    r"""
    Model for extinction due to a foreground plane of dust

    Parameters
    ----------
    Alam0 : float
        Magnitude of attenuation of the dust plane (at the tracer wavelength) at the center.

    rd : float
        Scale-length of the dust attenuation exponential curve, in kpc

    Notes
    -----
    This model places a dust dust as a foreground screen before the entire cube,
    with an exponential profile in the midplane of the source geometry.

    All positions that will have their
    flux reduced by :math:`A=10^{-0.4A_{\lambda}(r)}`, with

        .. math::

            F = AF_{\rm intrinsic}

    where :math:`A` is between 0 and 1. Here :math:`A_{\lambda}(r)=A_{\lambda,0}e^{-r/rd}`,
    and :math:`r=\sqrt{x^2+y^2}`.
    """

    Alam0 = DysmalParameter(default=0.0, bounds=(0., 10.))  # default: No attenuation
    rd = DysmalParameter(default=1.0, bounds=(0., 10.))     # Exponential scale radius, in kpc

    @staticmethod
    def evaluate(x, y, z, Alam0, rd):
        # Geometry: must be in correct source plane already!
        # Consider exponential in midplane, regardless of z.
        r = np.sqrt(x**2 + y**2)
        Alam = Alam0 * np.exp(-(r/rd))
        extinction = np.power(10., -0.4*Alam)
        return extinction

    def attenuation_cube(self, x, y, z):
        """Evaluate the flux attenuation (linear multiplier) at all locations of the cube"""
        return self.evaluate(x, y, z, self.Alam0, self.rd)
