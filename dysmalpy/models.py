# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# File containing all of the available models to use build the
# galaxy

# Standard library
import abc


# Third party imports
import numpy as np
import scipy.special as scp_spec
import scipy.io as scp_io
import scipy.interpolate as scp_interp
import astropy.constants as apy_con
import astropy.units as u
from astropy.modeling import FittableModel, Fittable1DModel, Parameter

# Directory where Noordermeer flattening curves are located
_dir_noordermeer = "data/noordermeer/"

# Useful constants
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc


# Generic model container which tracks all components, parameters,
# parameter settings, model settings, etc.
class ModelSet:

    def __init__(self):

        self.mass_components = []  # List of all of the mass components
        self.mass_comp_names = []  # List of the names of the mass models
        self.parameters = None     # Array of the current parameter values
        self.fixed = {}            # Dict. of bools for fixed parameters
        self.param_names = {}      # Dict. of parameter names
        self.bounds = {}           # Dict. of bounds for each parameter
        self.priors = {}           # Dict. of prior functions for each parameter
        self._param_keys = {}      # Dict. of location of each parameters within
                                   # the parameter array

# ***** Mass Component Model Classes ******
# Base abstract mass model component class
class MassModel(Fittable1DModel):

    _type = 'mass'

    @abc.abstractmethod
    def enclosed_mass(self, *args, **kwargs):
        """Evaluate the enclosed mass as a function of radius"""

    def circular_velocity(self, r):
        """
        Default method to evaluate the circular velocity
        as a function of radius using the standard equation:
        v(r) = SQRT(GM(r)/r)
        """

        mass_enc = self.enclosed_mass(r)
        vcirc = np.sqrt(G.cgs.value * mass_enc * Msun.cgs.value /
                        (r * 1000. * pc.cgs.value))
        vcirc = vcirc/1e5

        return vcirc


class Sersic(MassModel):
    """
    1D Sersic mass model with parameters defined by the total mass,
    Sersic index, and effective radius.
    """

    total_mass = Parameter(default=1)
    r_eff = Parameter(default=1)
    n = Parameter(default=1)

    def __init__(self, total_mass, r_eff, n, invq=1.0, **kwargs):
        self.invq = invq
        super(Sersic, self).__init__(total_mass, r_eff, n, **kwargs)

    @staticmethod
    def evaluate(r, total_mass, r_eff, n):
        """
        1D Sersic profile parameterized by the total mass and
        effective radius
        """

        bn = scp_spec.gammaincinv(2.*n, 0.5)
        alpha = r_eff/(bn**n)
        amp = (10**total_mass / (2*np.pi) / alpha**2 / n /
               scp_spec.gamma(2.*n))
        radial = amp * np.exp(-bn*(r/r_eff)**(1./n))
        # height = np.exp(-0.5*(z/thick)**2)

        return radial

    def enclosed_mass(self, r):
        """
        Calculate the enclosed mass as a function of radius
        :param r: Radii at which to calculate the enclosed mass
        :return: 1D enclosed mass profile
        """

        bn = scp_spec.gammaincinv(2.*self.n, 0.5)
        integ = scp_spec.gammainc(2*self.n, bn*(r/self.r_eff)**(1./self.n))
        norm = 10**self.total_mass

        return norm*integ

    def circular_velocity(self, r, noord_flat=False):

        if noord_flat:
            noordermeer_n = np.arange(0.5, 8.1, 0.1)  # Sersic indices
            noordermeer_invq = np.array([1, 2, 3, 4, 5, 6, 8, 10, 20,
                                         100])  # 1:1, 1:2, 1:3, ...flattening

            nearest_n = noordermeer_n[
                np.argmin(np.abs(noordermeer_n - self.n))]
            nearest_q = noordermeer_invq[
                np.argmin(np.abs(noordermeer_invq - self.invq))]

            # Need to do this internally instead of relying on IDL save files!!
            file_noord = _dir_noordermeer + 'VC_n{0:3.1f}_invq{1}.save'.format(
                nearest_n, nearest_q)
            restNVC = scp_io.readsav(file_noord)
            N2008_vcirc = restNVC.N2008_vcirc
            N2008_rad = restNVC.N2008_rad
            N2008_Re = restNVC.N2008_Re
            N2008_mass = restNVC.N2008_mass

            v_interp = scp_interp.interp1d(N2008_rad, N2008_vcirc,
                                           fill_value="extrapolate")
            vcirc = (v_interp(r/self.r_eff*N2008_Re)*np.sqrt(
                10**self.total_mass/N2008_mass)*np.sqrt(N2008_Re/self.r_eff))

        else:

            vcirc = super(Sersic, self).circular_velocity(r)

        return vcirc


class NFW(MassModel):
    """
    1D NFW mass model parameterized by the virial radius, virial mass, and
    concentration.
    """

    mvirial = Parameter(default=1.0)
    rvirial = Parameter(default=1.0)
    conc = Parameter(default=5.0)

    def __init__(self, mvirial, rvirial, conc, z=0, **kwargs):
        self.z = z
        super(NFW, self).__init__(mvirial, rvirial, conc, **kwargs)

    @staticmethod
    def evaluate(r, mvirial, rvirial, conc):
        """1D NFW model for a dark matter halo"""

        rho0 = (10**mvirial / (4 * np.pi * rvirial ** 3) * conc ** 3 /
                (np.log(1 + conc) - conc / (1 + conc)))

        return (2*np.pi * rho0 * rvirial /
                conc / (1+conc * r / rvirial)**2)

    def enclosed_mass(self, r):
        """
        Calculate the enclosed mass as a function of radius
        :param r: Radii at which to calculate the enclosed mass
        :return: 1D enclosed mass profile
        """

        rho0 = self.calc_rho0()
        rs = self.rvirial/self.conc
        aa = 4.*np.pi*rho0*self.rvirial**3/self.conc**3
        bb = np.log((rs + r)/rs) - r/(rs + r)

        return aa*bb

    def calc_rho0(self):
        aa = 10**self.mvirial/(4.*np.pi*self.rvirial**3)
        bb = self.conc**3/(np.log(1.+self.conc) - (self.conc/(1.+self.conc)))

        return aa*bb


# ****** Geometric Model ********
class Fittable3DModel(FittableModel):

    inputs = ('x', 'y', 'z')
    outputs = ('xp', 'yp', 'zp')


class Geometry(Fittable3DModel):
    """
    Class to hold the geometric parameters that can be fit.
    Also takes as input the sky coordinates and returns the
    corresponding galaxy plane coordinates.
    """

    inc = Parameter(default=45.0, bounds=(0, 90))
    pa = Parameter(default=0.0, bounds=(-180, 180))
    xshift = Parameter(default=0.0)
    yshift = Parameter(default=0.0)

    _type = 'geometry'

    @staticmethod
    def evaluate(x, y, z, inc, pa, xshift, yshift):

        inc = np.pi / 180. * inc
        pa = np.pi / 180. * (pa - 90.)

        nz, ny, nx = x.shape
        zsky = z - (nz - 1) / 2.
        xsky = x - (nx - 1) / 2. - xshift
        ysky = y - (ny - 1) / 2. - yshift

        xtmp = xsky * np.cos(pa) + ysky * np.sin(pa)
        ytmp = -xsky * np.sin(pa) + ysky * np.cos(pa)
        ztmp = zsky

        xgal = xtmp
        ygal = ytmp * np.cos(inc) - ztmp * np.sin(inc)
        zgal = ytmp * np.sin(inc) + ztmp * np.cos(inc)

        return (xgal, ygal, zgal)


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


def _adiabatic(rprime, r_adi, adia_v_dm, adia_x_dm, adia_v_disk):
    if rprime < 0.:
        rprime = 0.1
    if rprime < adia_x_dm[1]:
        rprime = adia_x_dm[1]
    rprime_interp = scp_interp.interp1d(adia_x_dm, adia_v_dm,
                                        fill_value="extrapolate")
    result = (r_adi + r_adi * ((r_adi*adia_v_disk**2) /
                               (rprime*(rprime_interp(rprime))**2)) - rprime)
    return result

