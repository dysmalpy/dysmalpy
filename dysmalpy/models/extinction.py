# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Extinction models for DysmalPy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np

# Local imports
from .parameters import DysmalParameter

__all__ = ['DustExtinction']

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')

np.warnings.filterwarnings('ignore')


class DustExtinction(_DysmalFittable3DModel):
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
        flux that is transmitted through the dust plane. `amp_extinct` = 1 means

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
    amp_extinct = DysmalParameter(default=0.0, bounds=(0., 1.))  # default: none

    _type = 'extinction'
    outputs = ('yp',)

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
        extinction[zsky <= zsky_dust] = amp_extinct

        return extinction
