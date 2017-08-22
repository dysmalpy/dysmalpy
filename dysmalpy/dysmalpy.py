# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is the main module to run DYSMALPY.
"""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

__all__ = ['Galaxy', 'Sersic', 'NFW', 'calc_rvir']
__version__ = '0.1'
__author__ = ''

import numpy as np
import scipy.special as scp_spec
import scipy.interpolate as scp_interp
import scipy.io as scp_io
import scipy.optimize as scp_opt
import astropy.constants as apy_con
import astropy.units as u
import astropy.cosmology as apy_cosmo
from astropy.modeling import Fittable2DModel, Parameter
import astropy.convolution as apy_conv
import astropy.io.fits as fits

# Set the cosmology that will be assumed throughout
cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)

# Necessary constants
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc


class Galaxy:
    """
    The main object for simulating the kinematics of a galaxy based on
    user provided mass components.
    """

    def __init__(self, redshift, name='galaxy'):
        self.z = redshift
        self.name = name
        self.mass_model = None
        self._light = []
        self._thick = []
        self.dscale = cosmo.arcsec_per_kpc_proper(self.z)

    def addSersic(self, re, n, mass, invq=1.0, light=True, name=None):
        """
        Add a Sersic profile as a mass component.

        :param re: Effective radius which contains half the total luminosity
        :param n: Sersic index
        :param mass: Total mass of this component in  log(Msun)
        :param invq: Flattening parameter for thick disks
        :param light: Whether to include in the light profile
        :param name: Name of the model (optional)
        """

        if name is None:
            name = 'sersic'

        thick = 2 * re / (invq * 2.35482)
        serc_mod = Sersic(total_mass=mass, r_eff=re, n=n, thick=thick,
                          name=name)
        #self._thick.append(2 * re / (invq * 2.35482))
        self._light.append(light)

        if self.mass_model is None:
            self.mass_model = serc_mod
        else:
            self.mass_model += serc_mod

    def addNFW(self, conc, mvirial, rvirial=None, tie_rvir_mvir=False,
               name=None):
        """
        Add an NFW halo as a mass component.

        :param conc: NFW concentration parameter
        :param mvirial: Virial mass of the NFW halo in log(Msun)
        :param rvirial: Virial radius of the NFW halo in kpc
        :param tie_rvir_mvir: Option to tie the virial radius to the
                              virial mass and redshift of the source
                              Default = False
        :param name: Name of the model (optional)
        """

        if name is None:
            name = 'nfw'

        if (rvirial is None) & (not tie_rvir_mvir):
            raise ValueError('Either a a value for rvirial must be provided or '
                             'tie_rvir_mvir must be set to True')
        elif tie_rvir_mvir:
            print('Virial radius will be set based on the virial mass and '
                  'specific cosmology!')
            rvirial = calc_rvir(mvirial, self.z)
            nfw_mod = NFW(mvirial=mvirial, rvirial=rvirial, conc=conc, z=self.z,
                          name=name)
            nfw_mod.rvirial.tied = _tie_rvir_mvir

        else:
            nfw_mod = NFW(mvirial=mvirial, rvirial=rvirial, conc=conc,
                          name=name)

        self._thick.append(0)  # No flattening for DM halo
        self._light.append(False)  # No light component for DM halo

        if self.mass_model is None:
            self.mass_model = nfw_mod
        else:
            self.mass_model += nfw_mod


class Sersic(Fittable2DModel):
    """
    1D Sersic mass model with parameters defined by the total mass,
    Sersic index, and effective radius.
    """

    total_mass = Parameter(default=1)
    r_eff = Parameter(default=1)
    n = Parameter(default=4)
    thick = Parameter(default=1)

    @staticmethod
    def evaluate(r, z, total_mass, r_eff, n, thick):
        """2D Sersic profile parameterized by the total mass and scale height"""

        bn = scp_spec.gammaincinv(2. * n, 0.5)
        alpha = r_eff / (bn ** n)
        amp = (10**total_mass / (2 * np.pi) / alpha ** 2 / n /
               scp_spec.gamma(2. * n))
        radial = amp * np.exp(-bn * (r / r_eff) ** (1. / n))
        height = np.exp(-0.5*(z/thick)**2)
        return radial*height


class NFW(Fittable2DModel):
    """
    2D NFW mass model parameterized by the virial radius, virial mass, and
    concentration.
    """

    mvirial = Parameter(default=1.0)
    rvirial = Parameter(default=1.0)
    conc = Parameter(default=5.0)

    def __init__(self, mvirial, rvirial, conc, z=0, **kwargs):
        self.z = z
        super(NFW, self).__init__(mvirial, rvirial, conc, **kwargs)

    @staticmethod
    def evaluate(r, z, mvirial, rvirial, conc):
        """2D NFW model for a dark matter halo"""

        rho0 = (10**mvirial / (4 * np.pi * rvirial ** 3) * conc ** 3 /
                (np.log(1 + conc) - conc / (1 + conc)))
        rtrue = np.sqrt(r**2 + z**2)
        return 2 * np.pi * rho0 * rvirial / conc / (1 + conc * rtrue / rvirial) ** 2


def calc_rvir(mvirial, z):
    """
    Calculate the virial radius based on virial mass and redshift
    M_vir = 100*H(z)^2/G * R_vir^3

    :param mvirial: Virial mass in log(Msun)
    :param z: Redshift
    :return: rvirial: Virial radius in kpc
    """
    G_new_unit = G.to(u.pc/u.Msun*(u.km/u.s)**2).value
    Hz = cosmo.H(z).value
    rvir = ((10**mvirial * (G_new_unit * 1e-3) /
             (10 * Hz * 1e-3) ** 2) ** (1./3.))

    return rvir


def _tie_rvir_mvir(model):
    # Function that will tie the virial radius to the virial mass within
    # the model fitting

    return calc_rvir(model.mvirial, model.z)
