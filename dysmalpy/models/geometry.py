# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Geometry models for DysmalPy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np
import scipy.ndimage as scp_ndi

try:
    from dysmalpy.utils import get_cin_cout
except:
    from ..utils import get_cin_cout


# Local imports
from .base import _DysmalFittable3DModel
from dysmalpy.parameters import DysmalParameter

__all__ = ['Geometry']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")


class Geometry(_DysmalFittable3DModel):
    """
    Model component defining the transformation from galaxy to sky coordinates

    Parameters
    ----------
    inc : float
        Inclination of the model in degrees

    pa : float
        Position angle East of North of the blueshifted side of the model in degrees

    xshift : float
        x-coordinate of the center of the model relative to center of data cube in pixels

    yshift : float
        y-coordinate of the center of the model relative to center of data cube in pixels

    vel_shift : float
        Systemic velocity shift that will be applied to the whole cube in km/s

    obs_name : string
        (Attribute): Name of the observation to which this geometry belongs.

    Methods
    -------
    coord_transform:
        Transform from sky to galaxy coordinates.

    inverse_coord_transform:
        Transform from galaxy to sky coordinates.

    Notes
    -----
    This model component takes as input sky coordinates and converts them
    to galaxy frame coordinates. `vel_shift` instead is used within `ModelSet.simulate_cube` to
    apply the necessary velocity shift.
    """

    inc = DysmalParameter(default=45.0, bounds=(0, 90))
    pa = DysmalParameter(default=0.0, bounds=(-180, 180))
    xshift = DysmalParameter(default=0.0)
    yshift = DysmalParameter(default=0.0)

    vel_shift = DysmalParameter(default=0.0, fixed=True)  # default: none

    obs_name = 'galaxy'

    _type = 'geometry'
    outputs = ('xp', 'yp', 'zp')

    def __init__(self, obs_name=None, **kwargs):
        if obs_name is None:
            raise ValueError("Geometries must have an 'obs_name' specified!")

        self.obs_name = obs_name

        super(Geometry, self).__init__(**kwargs)


    @staticmethod
    def evaluate(x, y, z, inc, pa, xshift, yshift, vel_shift):
        """Transform sky coordinates to galaxy/model reference frame"""
        inc = np.pi / 180. * inc
        pa = np.pi / 180. * (pa - 90.)

        # Apply the shifts in the sky system
        xsky = x - xshift
        ysky = y - yshift
        zsky = z

        xtmp = xsky * np.cos(pa) + ysky * np.sin(pa)
        ytmp = -xsky * np.sin(pa) + ysky * np.cos(pa)
        ztmp = zsky

        xgal = xtmp
        ygal = ytmp * np.cos(inc) - ztmp * np.sin(inc)
        zgal = ytmp * np.sin(inc) + ztmp * np.cos(inc)

        return xgal, ygal, zgal

    def coord_transform(self, x, y, z, inc=None, pa=None, xshift=None, yshift=None):
        """Transform sky coordinates to galaxy/model reference frame"""
        if inc is None:     inc = self.inc
        if pa is None:      pa = self.pa
        if xshift is None:  xshift = self.xshift
        if yshift is None:  yshift = self.yshift

        return self.evaluate(x, y, z, inc, pa, xshift, yshift, self.vel_shift)


    def inverse_coord_transform(self, xgal, ygal, zgal,
            inc=None, pa=None, xshift=None, yshift=None):
        """Transform galaxy/model reference frame to sky coordinates"""
        if inc is None:     inc = self.inc
        if pa is None:      pa = self.pa
        if xshift is None:  xshift = self.xshift
        if yshift is None:  yshift = self.yshift

        inc = np.pi / 180. * inc
        pa = np.pi / 180. * (pa - 90.)

        # Apply inlincation:
        xtmp =  xgal
        ytmp =  ygal * np.cos(inc) + zgal * np.sin(inc)
        ztmp = -ygal * np.sin(inc) + zgal * np.cos(inc)

        # Apply PA + shifts in sky system:
        xsky = xtmp * np.cos(pa) - ytmp * np.sin(pa) + xshift
        ysky = xtmp * np.sin(pa) + ytmp * np.cos(pa) + yshift
        zsky = ztmp

        return xsky, ysky, zsky


    def transform_cube_affine(self, cube, inc=None, pa=None, xshift=None, yshift=None,
                output_shape=None):
        """Incline and transform a cube from galaxy/model reference frame to sky frame.
            Use scipy.ndimage.affine_transform"""
        if inc is None:     inc = self.inc
        if pa is None:      pa = self.pa
        if xshift is None:  xshift = self.xshift
        if yshift is None:  yshift = self.yshift

        inc = np.pi / 180. * inc
        pa = np.pi / 180. * (pa - 90.)

        c_in =  get_cin_cout(cube.shape)
        if output_shape is not None:
            c_out = get_cin_cout(output_shape)
        else:
            c_out = get_cin_cout(cube.shape)

        # # CUBE: z, y, x
        minc = np.array([[np.cos(inc), np.sin(inc),  0.],
                         [-np.sin(inc), np.cos(inc), 0.],
                         [0., 0., 1.]])

        mpa = np.array([[1., 0., 0.],
                        [0., np.cos(pa), -np.sin(pa)],
                        [0., np.sin(pa), np.cos(pa)]])

        transf_matrix = np.matmul(minc, mpa)
        offset_arr = np.array([0., yshift.value, xshift.value])
        offset_transf = c_in-np.matmul(transf_matrix,c_out+offset_arr)

        cube_sky = scp_ndi.interpolation.affine_transform(cube, transf_matrix,
                    offset=offset_transf, order=3, output_shape=output_shape)

        return cube_sky



    def LOS_direction_emitframe(self, inc=None):
        r"""
        Method to return the LOS direction in terms of the emission frame.
        This just accounts for the inclination.

        Parameters
        ----------

        Returns
        -------
        LOS_unit_vector : 3-element array
            Direction of the LOS vector in (xyz).

            This describes the inclination, so returns [0, sin(i), -cos(i)]
            Note the zsky direction is TOWARDS from the observer, so the LOS is -zsky,
            where zsky is in the direction [0, -sin(i), cos(i)].
        """
        if inc is None:     inc = self.inc
        inc = np.pi / 180. * inc

        LOS_unit_vector = [ 0., np.sin(inc), -np.cos(inc) ]

        return LOS_unit_vector


    def project_velocity_along_LOS(self, model, vel, x, y, z, inc=None):
        r"""
        Method to project velocities in models' emission frame along the LOS (zsky direction).

        Parameters
        ----------
        model : `~dysmalpy.models._DysmalModel`
            Model component from which the velocity was calculated.

        vel : float or array or tuple of arrays
            Amplitude of the velocity.
            If model._multicoord_velocity is True, must pass a tuple of velocity components
            for each coordinate in the model's native geometry.

        x, y, z : float or array
            xyz position in the radial flow reference frame.

        inc : float, optional
            Specify a separate inclination. Default: uses self.inc

        Returns
        -------
        v_LOS : float or array
            Projection of velocity 'vel' along the LOS.
        """

        LOS_hat = self.LOS_direction_emitframe(inc=inc)

        vel_cartesian = model.velocity_vector(x, y, z, vel=vel, _save_memory=True)

        # Dot product of vel_cartesian, LOS_unit_vector
        v_LOS = vel_cartesian[0]*LOS_hat[0] + vel_cartesian[1]*LOS_hat[1] + vel_cartesian[2]*LOS_hat[2]

        return v_LOS
