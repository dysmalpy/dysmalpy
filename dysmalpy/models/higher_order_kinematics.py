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
from .base import _DysmalFittable1DModel, _DysmalFittable3DModel
from dysmalpy.parameters import DysmalParameter

__all__ = ['BiconicalOutflow', 'UnresolvedOutflow', 'UniformRadialFlow']

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')

np.warnings.filterwarnings('ignore')


class BiconicalOutflow(_DysmalFittable3DModel):
    r"""
    Model for a biconical outflow

    Parameters
    ----------
    n : float
        Power law index of the outflow velocity profile

    vmax : float
        Maximum velocity of the outflow in km/s

    rturn : float
        Turn-over radius in kpc of the velocty profile

    thetain : float
        Half inner opening angle in degrees. Measured from the bicone
        axis

    dtheta : float
        Difference between inner and outer opening angle in degrees

    rend : float
        Maximum radius of the outflow in kpc

    norm_flux : float
        Log flux amplitude of the outflow at r = 0

    tau_flux : float
        Exponential decay rate of the flux

    profile_type : {'both', 'increase', 'decrease', 'constant'}
        Type of velocity profile

    Notes
    -----
    This biconical outflow model is based on the model presented in Bae et al. (2016) [1]_.
    It consists of two symmetric cones joined at their apexes. `thetain` and `dtheta` control
    how hollow the cones are. `thetain` = 0 therefore would produce a fully filled cone.

    Within the cone, the velocity radial profile of the gas follows a power law with index `n`.
    Four different profile types can be selected. The simplest is 'constant' in which case
    the velocity of the gas is `vmax` at all radii.

    For a `profile_type` = 'increase':

        .. math::

            v = v_{\rm max}\left(\frac{r}{r_{\rm end}}\right)^n

    For a `profile_type` = 'decrease':

        .. math::

            v = v_{\rm max}\left(1 - \left(\frac{r}{r_{\rm end}}\right)^n\right)

    For a `profile_type` = 'both' the velocity first increases up to the turnover radius, `rturn`,
    then decreases back to 0 at 2 `rturn`.

    For :math:`r < r_{\rm turn}`:

        .. math::

            v =  v_{\rm max}\left(\frac{r}{r_{\rm turn}}\right)^n

    For :math:`r > r_{\rm turn}`:

        .. math::

            v = v_{\rm max}\left(2 - \frac{r}{r_{\rm turn}}\right)^n

    The flux radial profile of the outflow is described by a decreasing exponential:

        .. math::

            F = A\exp\left\{\frac{-\tau r}{r_{\rm end}}\right\}

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/2016ApJ...828...97B/abstract

    """

    n = DysmalParameter(default=0.5, fixed=True)
    vmax = DysmalParameter(min=0)
    rturn = DysmalParameter(default=0.5, min=0)
    thetain = DysmalParameter(bounds=(0, 90))
    dtheta = DysmalParameter(default=20.0, bounds=(0, 90))
    rend = DysmalParameter(default=1.0, min=0)
    norm_flux = DysmalParameter(default=0.0, fixed=True)
    tau_flux = DysmalParameter(default=5.0, fixed=True)

    _type = 'higher_order'
    _spatial_type = 'resolved'
    outputs = ('vout',)

    def __init__(self, profile_type='both', **kwargs):

        valid_profiles = ['increase', 'decrease', 'both', 'constant']

        if profile_type in valid_profiles:
            self.profile_type = profile_type
        else:
            logger.error("Invalid profile type. Must be one of 'increase',"
                         "'decrease', 'constant', or 'both.'")

        super(BiconicalOutflow, self).__init__(**kwargs)

    def evaluate(self, x, y, z, n, vmax, rturn, thetain, dtheta, rend, norm_flux, tau_flux):
        """Evaluate the outflow velocity as a function of position x, y, z"""

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.abs(z)/r)*180./np.pi
        theta[r == 0] = 0.
        vel = np.zeros(r.shape)

        if self.profile_type == 'increase':

            amp = vmax/rend**n
            vel[r <= rend] = amp*r[r <= rend]**n
            vel[r == 0] = 0

        elif self.profile_type == 'decrease':

            amp = -vmax/rend**n
            vel[r <= rend] = vmax + amp*r[r <= rend]** n


        elif self.profile_type == 'both':

            vel[r <= rturn] = vmax*(r[r <= rturn]/rturn)**n
            ind = (r > rturn) & (r <= 2*rturn)
            vel[ind] = vmax*(2 - r[ind]/rturn)**n

        elif self.profile_type == 'constant':

            vel[r <= rend] = vmax

        thetaout = np.min([thetain+dtheta, 90.])
        ind_zero = (theta < thetain) | (theta > thetaout) | (vel < 0)
        vel[ind_zero] = 0.

        return vel

    def light_profile(self, x, y, z):
        """Evaluate the outflow line flux as a function of position x, y, z"""

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.abs(z) / r) * 180. / np.pi
        theta[r == 0] = 0.
        flux = 10**self.norm_flux*np.exp(-self.tau_flux*(r/self.rend))
        thetaout = np.min([self.thetain + self.dtheta, 90.])
        ind_zero = ((theta < self.thetain) |
                    (theta > thetaout) |
                    (r > self.rend))
        flux[ind_zero] = 0.

        return flux


    def vel_vector_direction_emitframe(self, x, y, z):
        r"""
        Method to return the velocity direction in the outflow Cartesian frame.

        Parameters
        ----------
        x, y, z : float or array
            xyz position in the outflow reference frame.

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyz).

            For biconical outflows, this is the rhat direction, in spherical coordinates
            (r,phi,theta).
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2 )
        vel_dir_unit_vector = [ x/r, y/r, z/r ]
        return vel_dir_unit_vector

#
# class UnresolvedOutflow(_DysmalFittable1DModel):
#     """
#     Model for an unresolved outflow component described by a Gaussian
#
#     Parameters
#     ----------
#     vcenter : float
#         Central velocity of the Gaussian in km/s
#
#     fwhm : float
#         FWHM of the Gaussian in km/s
#
#     amplitude : float
#         Amplitude of the Gaussian
#
#     Notes
#     -----
#     This model simply produces a broad Gaussian spectrum that will be placed in the
#     central spectrum of the galaxy.
#     """
#
#     vcenter = DysmalParameter(default=0)
#     fwhm = DysmalParameter(default=1000.0, bounds=(0, None))
#     amplitude = DysmalParameter(default=1.0, bounds=(0, None))
#
#     _type = 'outflow'
#     _spatial_type = 'unresolved'
#     outputs = ('vout',)
#
#     @staticmethod
#     def evaluate(v, vcenter, fwhm, amplitude):
#
#         return amplitude*np.exp(-(v - vcenter)**2/(fwhm/2.35482)**2)


class UnresolvedOutflow(_DysmalFittable3DModel):
    """
    Model for an unresolved outflow component described by a Gaussian

    Parameters
    ----------
    vcenter : float
        Central velocity of the Gaussian in km/s

    fwhm : float
        FWHM of the Gaussian in km/s

    amplitude : float
        Amplitude of the Gaussian

    Notes
    -----
    This model simply produces a broad Gaussian spectrum that will be placed in the
    central spectrum of the galaxy.
    """

    vcenter = DysmalParameter(default=0)
    fwhm = DysmalParameter(default=1000.0, bounds=(0, None))
    amplitude = DysmalParameter(default=1.0, bounds=(0, None))

    _type = 'higher_order'
    _spatial_type = 'unresolved'
    outputs = ('vout',)

    @staticmethod
    def evaluate( x, y, z, vcenter, fwhm, amplitude):
        return np.ones(x.shape)*vcenter

    def dispersion_profile(self, x, y, z, fwhm=None):
        """Dispersion profile for the outflow"""
        if fwhm is None: fwhm = self.fwhm
        return np.ones(x.shape)*(fwhm/2.35482)

    def light_profile(self, x, y, z):
        """Evaluate the outflow line flux as a function of position x, y, z
           All the light will be deposited at the center pixel."""

        # The coordinates where the unresolved outflow is placed needs to be
        # an integer pixel so for now we round to nearest integer.

        r = np.sqrt(x**2 + y**2 + z**2)
        ind_min = r.argmin()
        flux = x*0.
        flux[ind_min] = self.amplitude

        return flux



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
