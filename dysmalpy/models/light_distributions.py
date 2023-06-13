# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Light distribution models for DysmalPy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np

# Local imports
from .base import LightModel, _DysmalFittable1DModel, _DysmalFittable3DModel, \
                  truncate_sersic_mr, sersic_mr, _I0_gaussring
from dysmalpy.parameters import DysmalParameter

try:
    from dysmalpy.models import utils
except:
   from . import utils

__all__ = ['LightTruncateSersic', 'LightGaussianRing', 'LightClump',
           'LightGaussianRingAzimuthal']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")


class LightTruncateSersic(LightModel, _DysmalFittable1DModel):
    """
    Light distribution following a Sersic profile. Can be truncted.

    Parameters
    ----------
    r_eff : float
        Effective (half-light) radius in kpc

    L_tot: float
        Total luminsoity of untruncated Sersic profile. Arbitrary units.

    n : float
        Sersic index

    r_inner : float
        Inner truncation radius in kpc. Default: 0 kpc (untruncated)

    r_outer : float
        Outer truncation radius in kpc. Default: np.inf kpc (untruncated)

    tracer : string
        (Attribute): Name of the dynamical tracer

    Notes
    -----
    Model formula:

    .. math::

        I(R) = I_e \exp \\left\{ -b_n \\left[ \\left( \\frac{R}{R_{\mathrm{eff}}} \\right)^{1/n} -1 \\right] \\right\}

    The constant :math:`b_n` is defined such that :math:`R_{\mathrm{eff}}` contains half the total
    light, and can be solved for numerically as: 

    .. math::

        \Gamma(2n) = 2\gamma (b_n,2n)

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from dysmalpy.models import LightTruncateSersic
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(111, xscale='log', yscale='log')
        ls1 = LightTruncateSersic(r_eff=5, n=1, r_inner=1, r_outer=20, L_tot=1.e11, tracer='halpha')
        r=np.arange(0, 100, .01)

        for n in range(1, 10):
             ls1.n = n
             plt.plot(r, ls1(r), color=str(float(n) / 15))

        plt.axis([0.8, 27, 1e5, 1e10])
        plt.xlabel('log Radius [kpc]')
        plt.ylabel('log Intensity Surface Density [log Lsun/kpc^2]')
        plt.text(1.1, 7.e8, 'n=1')
        plt.text(1.1, 3.e9, 'n=10')
        plt.show()

    """

    L_tot = DysmalParameter(default=1, bounds=(0, 50))
    r_eff = DysmalParameter(default=1, bounds=(0, 50))
    n = DysmalParameter(default=1, bounds=(0, 8))
    r_inner = DysmalParameter(default=0., bounds=(0, 10))
    r_outer = DysmalParameter(default=np.inf, bounds=(0, np.inf))

    def __init__(self, tracer=None, **kwargs):
        if tracer is None:
            raise ValueError("'tracer' for light profile must be specified!")
        self.tracer = tracer

        super(LightTruncateSersic, self).__init__(**kwargs)

    @staticmethod
    def evaluate(r, L_tot, r_eff, n, r_inner, r_outer):
        """
        Sersic light surface density. Same as self.light_profile
        """
        return truncate_sersic_mr(r, L_tot, n, r_eff, r_inner, r_outer)

    def light_profile(self, r):
        """
        Conversion from mass to light as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        light : float or array
            Relative line flux as a function of radius
        """
        return self.evaluate(r, self.L_tot, self.r_eff, self.n, self.r_inner, self.r_outer)


class LightGaussianRing(LightModel, _DysmalFittable1DModel):
    r"""
    Light distribution following a Gaussian ring profile.

    Parameters
    ----------
    R_peak : float
        Peak of gaussian (radius) in kpc

    FWHM: float
        FWHM of gaussian, in kpc

    L_tot: float
        Total luminsoity of component. Arbitrary units

    tracer : string
        (Attribute): Name of the dynamical tracer

    Notes
    -----
    Model formula:

    .. math::

        I(r)=I_0\exp\left[-\frac{(r-r_{peak})^2}{2\sigma_R^2}\right]


    """
    R_peak = DysmalParameter(default=1, bounds=(0, 50))
    FWHM = DysmalParameter(default=1, bounds=(0, 50))
    L_tot = DysmalParameter(default=1, bounds=(0, 50))

    def __init__(self, tracer=None, **kwargs):
        if tracer is None:
            raise ValueError("'tracer' for light profile must be specified!")
        self.tracer = tracer

        super(LightGaussianRing, self).__init__(**kwargs)

    def sigma_R(self):
        return self.FWHM.value / (2.*np.sqrt(2.*np.log(2.)))

    @staticmethod
    def evaluate(r, R_peak, FWHM, L_tot):
        """
        Gaussian ring light surface density.
        Radius r must be in kpc
        """
        sigma_R = FWHM/ (2.*np.sqrt(2.*np.log(2.)))
        I0 = _I0_gaussring(R_peak, sigma_R, L_tot)
        return I0*np.exp(-(r-R_peak)**2/(2.*sigma_R**2))

    def light_profile(self, r):
        """
        Conversion from mass to light as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass, in kpc

        Returns
        -------
        light : float or array
            Relative line flux as a function of radius
        """
        return self.evaluate(r, self.R_peak, self.FWHM, self.L_tot)


class LightClump(LightModel, _DysmalFittable3DModel):
    """
    Light distribution for a clump following a Sersic profile,
    at a given galaxy midplane R and azimuthal angle phi.

    Parameters
    ----------
    r_center : float
        Radial distance from galaxy center to clump center, in the galaxy midplane, in kpc

    phi : float
        Azimuthal angle of clump, counter-clockwise from blue major axis. In degrees.

    theta : float
        Polar angle of clump, from the +z direction. In degrees.

    L_tot: float
        Total luminsoity of clump. Arbitrary units.

    r_eff : float
        Effective (half-light) radius of clump in kpc

    n : float
        Sersic index of clump

    tracer : string
        (Attribute): Name of the dynamical tracer

    Notes
    -----
    Model formula:

    .. math::

        I(r) = I_e \exp \\left\{ -b_n \\left[ \\left( \\frac{r}{r_{\mathrm{eff}}} \\right)^{1/n} -1 \\right] \\right\}

    The constant :math:`b_n` is defined such that :math:`r_{\mathrm{eff}}` contains half the total
    light, and can be solved for numerically.

    .. math::

        \Gamma(2n) = 2\gamma (b_n,2n)

    """

    _axisymmetric = False
    L_tot = DysmalParameter(default=1, bounds=(0, 50))
    r_eff = DysmalParameter(default=1, bounds=(0, 50))
    n = DysmalParameter(default=1, bounds=(0, 8))
    r_center = DysmalParameter(default=0., bounds=(0, 30))
    phi = DysmalParameter(default=0., bounds=(0, 360))
    theta = DysmalParameter(default=90., bounds=(0, 180))

    def __init__(self, tracer=None, **kwargs):
        if tracer is None:
            raise ValueError("'tracer' for light profile must be specified!")
        self.tracer = tracer

        super(LightClump, self).__init__(**kwargs)

    @staticmethod
    def evaluate(x, y, z, L_tot, r_eff, n, r_center, phi, theta):
        """
        Light profile of the clump
        """
        phi_rad = np.pi / 180. * phi

        # INGORE THETA, and assume clump centered at midplane:
        r = np.sqrt( (x-r_center*np.cos(phi_rad))**2 + \
                     (y-r_center*np.sin(phi_rad))**2 )
        return sersic_mr(r, L_tot, n, r_eff)

    def light_profile(self, x, y, z):
        """
        Light profile of the clump

        Parameters
        ----------
        x, y, z : float or array
            Position at which to calculate the light profile

        Returns
        -------
        light : float or array
            Relative line flux as a function of radius
        """
        return self.evaluate(x, y, z, self.L_tot, self.r_eff, self.n,
                             self.r_center, self.phi, self.theta)


class LightGaussianRingAzimuthal(LightModel, _DysmalFittable3DModel):
    r"""
    Light distribution following a Gaussian ring profile, with azimuthal brightness variation.
    (Reflection symmetric about one axis)

    Parameters
    ----------
    R_peak : float
        Peak of gaussian (radius) in kpc

    FWHM: float
        FWHM of gaussian, in kpc

    L_tot: float
        Total luminsoity of component. Arbitrary units

    phi : float
        Azimuthal angle of bright side, counter-clockwise from blue major axis. In degrees.

    contrast : float
        Brightness contrast between dim and bright sides, dim/bright. Default: 1.

    gamma : float
        Scaling factor for how quickly the brightness changes occur from [0., abs(180.)].

    tracer : string
        (Attribute): Name of the dynamical tracer

    Notes
    -----
    Model formula, for the radial part:

    .. math::

        I(r)=I_0\exp\left[-\frac{(r-r_{peak})^2}{2\sigma_R^2}\right]


    """

    _axisymmetric = False
    R_peak = DysmalParameter(default=1, bounds=(0, 50))
    FWHM = DysmalParameter(default=1, bounds=(0, 50))
    L_tot = DysmalParameter(default=1, bounds=(0, 50))
    phi = DysmalParameter(default=0., bounds=(0, 360))
    contrast = DysmalParameter(default=1., bounds=(0., 1.))
    gamma = DysmalParameter(default=1., bounds=(0, 100.))

    def __init__(self, tracer=None, **kwargs):
        if tracer is None:
            raise ValueError("'tracer' for light profile must be specified!")
        self.tracer = tracer

        super(LightGaussianRingAzimuthal, self).__init__(**kwargs)

    def sigma_R(self):
        return self.FWHM.value / (2.*np.sqrt(2.*np.log(2.)))

    @staticmethod
    def evaluate(x, y, z, R_peak, FWHM, L_tot, phi, contrast, gamma):
        """
        Azimuthally varying Gaussian ring light surface density.
        Positions x,y,z in kpc
        """
        sigma_R = FWHM / (2.*np.sqrt(2.*np.log(2.)))
        I0 = _I0_gaussring(R_peak, sigma_R, L_tot)
        r = np.sqrt( x ** 2 + y ** 2 )
        gaus_symm = I0*np.exp(-(r-R_peak)**2/(2.*sigma_R**2))

        # Assume ring is in midplane
        phi_rad = phi * np.pi / 180.
        phi_gal_rad = utils.get_geom_phi_rad_polar(x, y)


        asymm_fac = 1. - (1.-contrast)*np.power(np.abs(np.sin(0.5 * (phi_gal_rad-phi_rad))), 1./gamma)
        gaus_asymm = gaus_symm * asymm_fac

        return gaus_asymm

    def light_profile(self, x, y, z):
        """
        Azimuthally varying Gaussian ring light surface density.

        Parameters
        ----------
        x, y, z : float or array
            Position at which to calculate the light profile, in kpc. In the galaxy frame.

        Returns
        -------
        light : float or array
            Relative line flux as a function of radius
        """
        return self.evaluate(x, y, z, self.R_peak, self.FWHM, self.L_tot,
                    self.phi, self.contrast, self.gamma)
