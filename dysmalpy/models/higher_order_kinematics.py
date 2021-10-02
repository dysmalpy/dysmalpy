# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Higher-order kinematics models for DysmalPy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np

# Local imports
from .base import _DysmalFittable3DModel
from dysmalpy.parameters import DysmalParameter

__all__ = ['UniformRadialFlow']

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')

np.warnings.filterwarnings('ignore')


class UniformRadialFlow(_DysmalFittable3DModel):
    """
    Model for a uniform radial flow.

    Parameters
    ----------
    vr : float
        Radial velocity in km/s. vr > 0 for outflow, vr < 0 for inflow

    Notes
    -----
    This model simply adds a constant radial velocity component
    to all of the positions in the galaxy.
    """
    vr = DysmalParameter(default=30.)

    _type = 'higher_order'
    _spatial_type = 'resolved'
    outputs = ('vrad',)

    def __init__(self, **kwargs):

        super(UniformRadialFlow, self).__init__(**kwargs)

    def evaluate(self, x, y, z, vr):
        """Evaluate the radial velocity as a function of position x, y, z"""

        vel = np.ones(x.shape) * (vr)

        return vel


    def vel_vector_direction_emitframe(self, x, y, z):
        r"""
        Method to return the velocity direction in the outflow Cartesian frame.

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the radial flow reference frame.

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyz).

            For a uniform radial flow, this is the rhat direction, in spherical coordinates
            (r,phi,theta).
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2 )
        vel_dir_unit_vector = [ x/r, y/r, z/r ]
        return vel_dir_unit_vector
