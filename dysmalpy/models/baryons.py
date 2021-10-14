# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Baryon mass models for DysmalPy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import os
import logging

# Third party imports
import numpy as np
import scipy.special as scp_spec
import scipy.io as scp_io
import scipy.interpolate as scp_interp
import astropy.constants as apy_con
import astropy.units as u


from astropy.table import Table

# Local imports
from .base import MassModel, v_circular, menc_from_vcirc, sersic_mr
from dysmalpy.parameters import DysmalParameter

__all__ = ['Sersic', 'DiskBulge', 'LinearDiskBulge', 'ExpDisk', 'BlackHole',
           'surf_dens_exp_disk', 'menc_exp_disk', 'vcirc_exp_disk',
           'sersic_menc_2D_proj', 'apply_noord_flat']

# NOORDERMEER DIRECTORY
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
# located one up:
dir_path = '/'.join(dir_path.split('/')[:-1])
_dir_noordermeer = dir_path+"/data/noordermeer/"


# ALT NOORDERMEER DIRECTORY:
# TEMP:
_dir_sersic_profile_mass_VC_TMP = "/Users/sedona/data/sersic_profile_mass_VC/"
_dir_sersic_profile_mass_VC = os.getenv('SERSIC_PROFILE_MASS_VC_DATADIR', _dir_sersic_profile_mass_VC_TMP)

# try:
#     import sersic_profile_mass_VC.calcs as sersic_profile_mass_VC_calcs
#     _sersic_profile_mass_VC_loaded = True
# except:
#     _sersic_profile_mass_VC_loaded = False


# CONSTANTS
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc

# # +++++++++++++++++++++++++++++
# # TEMP:
# G = 6.67e-11 * u.m**3 / u.kg / (u.s**2)  #(unit='m3 / (kg s2)')
# Msun = 2e30 * u.kg
# pc = 3e16 * u.m
# # +++++++++++++++++++++++++++++

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')

np.warnings.filterwarnings('ignore')



def surf_dens_exp_disk(r, mass, rd):
    """
    Radial surface density function for an infinitely thin exponential disk

    Parameters
    ----------
    r : float or array
        Radius or radii at which to calculate the surface density

    mass : float
        Total mass of the disk

    rd : float
        Disk scale length.

    Returns
    -------
    Sigr : float or array
        Surface density of a thin exponential disk at `r`
    """

    Sig0 = mass / (2. * np.pi * rd**2)
    Sigr = Sig0 * np.exp(-r/rd)

    return Sigr


def menc_exp_disk(r, mass, rd):
    """
    Enclosed mass function for exponential disk

    Parameters
    ----------
    r : float or array
        Radius or radii at which to calculate the surface density

    mass : float
        Total mass of the disk

    rd : float
        Disk scale length.

    Returns
    -------
    menc : float or array
        Enclosed mass of an exponential disk for the given `r`
    """

    Sig0 = mass / (2. * np.pi * rd**2)

    menc = 2. * np.pi * Sig0 * rd**2 * ( 1 - np.exp(-r/rd)*(1.+r/rd) )

    return menc


def vcirc_exp_disk(r, mass, rd):
    """
    Rotation curve function for exponential disk

    Parameters
    ----------
    r : float or array
        Radius or radii at which to calculate the surface density

    mass : float
        Total mass of the disk

    rd : float
        Disk scale length.

    Returns
    -------
    vc : float or array
        Circular velocity of an exponential disk as a function of `r`
    """

    #b1 = 1.6783469900166612   # scp_spec.gammaincinv(2.*n, 0.5), n=1
    #rd = r_eff / b1
    Sig0 = mass / (2. * np.pi * rd**2)

    y = r / (2.*rd)
    expdisk = y**2 * ( scp_spec.i0(y) * scp_spec.k0(y) - scp_spec.i1(y)*scp_spec.k1(y) )
    VCsq = 4 * np.pi * G.cgs.value*Msun.cgs.value / (1000.*pc.cgs.value) * Sig0 * rd * expdisk

    VCsq[r==0] = 0.

    return np.sqrt(VCsq) / 1.e5


def sersic_menc_2D_proj(r, mass, n, r_eff):
    """
    Enclosed mass as a function of r for a generic sersic model

    Parameters
    ----------
    r : float or array
        Radius or radii at which to calculate the surface mass density

    mass : float
        Total mass of the Sersic component

    n : float
        Sersic index

    r_eff : float
        Effective radius

    Returns
    -------
    menc : float or array
        Enclosed mass as a function of `r`

    Notes
    -----
    This function is only valid in the case of an infinite cylinder
    """

    bn = scp_spec.gammaincinv(2. * n, 0.5)
    integ = scp_spec.gammainc(2 * n, bn * (r / r_eff) ** (1. / n))
    norm = mass
    menc = norm*integ

    return menc




def apply_noord_flat(r, r_eff, mass, n, invq):
    """
    Calculate circular velocity for a thick Sersic component

    Parameters
    ----------
    r : float or array
        Radius or radii at which to calculate the circular velocity in kpc

    r_eff : float
        Effective radius of the Sersic component in kpc

    mass : float
        Total mass of the Sersic component

    n : float
        Sersic index

    invq : float
        Ratio of the effective radius of the Sersic component in the midplane to the
        effective radius in the z-direction

    Returns
    -------
    vcirc : float or array
        Circular velocity at each given `r`

    Notes
    -----
    This function determines the circular velocity as a function of radius for
    a Sersic component with a total mass, `mass`, Sersic index, `n`, and
    an effective radius to scale height ratio, `invq`. This uses lookup tables
    numerically calculated from the derivations provided in Noordermeer 2008 [1]_ which
    properly accounted for the thickness of the mass component.

    The lookup table provides rotation curves for Sersic components with
    `n` = 0.5 - 8 at steps of 0.1 and `invq` = [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 100].
    If the given `n` and/or `invq` are not one of these values then the nearest
    ones are used.

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/abstract
    """

    noordermeer_n = np.arange(0.5, 8.1, 0.1)  # Sersic indices
    noordermeer_invq = np.array([1, 2, 3, 4, 5, 6, 8, 10, 20,
                                 100])  # 1:1, 1:2, 1:3, ...flattening

    nearest_n = noordermeer_n[
        np.argmin(np.abs(noordermeer_n - n))]
    nearest_q = noordermeer_invq[
        np.argmin(np.abs(noordermeer_invq - invq))]

    # Need to do this internally instead of relying on IDL save files!!
    file_noord = _dir_noordermeer + 'VC_n{0:3.1f}_invq{1}.save'.format(
        nearest_n, nearest_q)

    #try:
    if True:
        restNVC = scp_io.readsav(file_noord)
        N2008_vcirc = restNVC.N2008_vcirc
        N2008_rad = restNVC.N2008_rad
        N2008_Re = restNVC.N2008_Re
        N2008_mass = restNVC.N2008_mass

        v_interp = scp_interp.interp1d(N2008_rad, N2008_vcirc,
                                       fill_value="extrapolate")
        vcirc = (v_interp(r / r_eff * N2008_Re) * np.sqrt(
                 mass / N2008_mass) * np.sqrt(N2008_Re / r_eff))

    # except:
    #     vcirc = apply_noord_flat_new(r, r_eff, mass, n, invq)

    return vcirc


def get_sersic_VC_table_new(n, invq):
    # Use the "typical" collection of table values:
    table_n = np.arange(0.5, 8.1, 0.1)   # Sersic indices
    table_invq = np.array([1., 2., 3., 4., 5., 6., 7., 8., 10., 20., 100.,
                    1.11, 1.43, 1.67, 3.33, 0.5, 0.67])  # 1:1, 1:2, 1:3, ... flattening  [also prolate 2:1, 1.5:1]

    nearest_n = table_n[ np.argmin( np.abs(table_n - n) ) ]
    nearest_invq = table_invq[ np.argmin( np.abs( table_invq - invq) ) ]

    file_sersic = _dir_sersic_profile_mass_VC + 'mass_VC_profile_sersic_n{:0.1f}_invq{:0.2f}.fits'.format(nearest_n, nearest_invq)

    try:
        t = Table.read(file_sersic)
    except:
        raise ValueError("File {} not found. _dir_sersic_profile_mass_VC={}. Check that system var ${} is set correctly.".format(file_sersic,
                    _dir_sersic_profile_mass_VC, 'SERSIC_PROFILE_MASS_VC_DATADIR'))

    return t[0]

def apply_noord_flat_new(r, r_eff, mass, n, invq):
    # SHOULD BE EXACTLY, w/in numerical limitations, EQUIV TO OLD CALCULATION
    table = get_sersic_VC_table_new(n, invq)

    N2008_vcirc =   table['vcirc']
    N2008_rad =     table['r']
    N2008_Re =      table['Reff']
    N2008_mass =    table['total_mass']

    v_interp = scp_interp.interp1d(N2008_rad, N2008_vcirc,
                                   fill_value="extrapolate")
    vcirc = (v_interp(r / r_eff * N2008_Re) * np.sqrt(
             mass / N2008_mass) * np.sqrt(N2008_Re / r_eff))

    return vcirc

def sersic_curve_rho(r, Reff, total_mass, n, invq, interp_type='linear'):
    table = get_sersic_VC_table_new(n, invq)

    table_rho =     table['rho']
    table_rad =     table['r']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Drop nonfinite parts:
    whfin = np.where(np.isfinite(table_rho))[0]
    table_rho = table_rho[whfin]
    table_rad = table_rad[whfin]

    # Ensure it's an array:
    if isinstance(r*1., float):
        rarr = np.array([r])
    else:
        rarr = np.array(r)
    # Ensure all radii are 0. or positive:
    rarr = np.abs(rarr)


    # # UNIFIED INTERPOLATION/EXTRAPOLATION
    # r_interp = scp_interp.interp1d(table_rad, table_rho, bounds_error=False,
    #                                fill_value='extrapolate', kind='linear')
    #
    # rho_interp =  (r_interp(rarr / Reff * table_Reff) * (total_mass / table_mass) * (table_Reff / Reff)**3 )

    scale_fac = (total_mass / table_mass) * (table_Reff / Reff)**3

    if interp_type.lower().strip() == 'cubic':
        r_interp = scp_interp.interp1d(table_rad, table_rho, fill_value=np.NaN, bounds_error=False, kind='cubic')
        r_interp_extrap = scp_interp.interp1d(table_rad, table_rho, fill_value='extrapolate', kind='linear')

        rho_interp = np.zeros(len(rarr))
        wh_in =     np.where((rarr <= table_rad.max()) & (rarr >= table_rad.min()))[0]
        wh_extrap = np.where((rarr > table_rad.max()) | (rarr < table_rad.min()))[0]
        rho_interp[wh_in] =     (r_interp(rarr[wh_in] / Reff * table_Reff) * scale_fac )
        rho_interp[wh_extrap] = (r_interp_extrap(rarr[wh_extrap] / Reff * table_Reff) * scale_fac)
    elif interp_type.lower().strip() == 'linear':
        r_interp = scp_interp.interp1d(table_rad, table_rho, fill_value='extrapolate',
                                       bounds_error=False, kind='linear')
        rho_interp =     (r_interp(rarr / Reff * table_Reff) * scale_fac )

    else:
        raise ValueError("interp type '{}' unknown!".format(interp_type))



    if (len(rarr) > 1):
        return rho_interp
    else:
        if isinstance(r*1., float):
            # Float input
            return rho_interp[0]
        else:
            # Length 1 array input
            return rho_interp

    return rho_interp

def sersic_curve_dlnrho_dlnr(r, Reff, n, invq, interp_type='linear'):
    table = get_sersic_VC_table_new(n, invq)

    table_dlnrho_dlnr =     table['dlnrho_dlnr']
    table_rad =     table['r']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Drop nonfinite parts:
    whfin = np.where(np.isfinite(table_dlnrho_dlnr))[0]
    table_dlnrho_dlnr = table_dlnrho_dlnr[whfin]
    table_rad = table_rad[whfin]


    # Ensure it's an array:
    if isinstance(r*1., float):
        rarr = np.array([r])
    else:
        rarr = np.array(r)
    # Ensure all radii are 0. or positive:
    rarr = np.abs(rarr)

    #
    # # UNIFIED INTERPOLATION/EXTRAPOLATION
    # r_interp = scp_interp.interp1d(table_rad, table_dlnrho_dlnr, bounds_error=False,
    #                                fill_value='extrapolate', kind='linear')
    # dlnrho_dlnr_interp = (r_interp(rarr / Reff * table_Reff) )

    if interp_type.lower().strip() == 'cubic':
        r_interp = scp_interp.interp1d(table_rad, table_dlnrho_dlnr, fill_value=np.NaN, bounds_error=False, kind='cubic')
        r_interp_extrap = scp_interp.interp1d(table_rad, table_dlnrho_dlnr, fill_value='extrapolate', kind='linear')

        dlnrho_dlnr_interp = np.zeros(len(rarr))
        wh_in =     np.where((rarr <= table_rad.max()) & (rarr >= table_rad.min()))[0]
        wh_extrap = np.where((rarr > table_rad.max()) | (rarr < table_rad.min()))[0]
        dlnrho_dlnr_interp[wh_in] =     (r_interp(rarr[wh_in] / Reff * table_Reff) )
        dlnrho_dlnr_interp[wh_extrap] = (r_interp_extrap(rarr[wh_extrap] / Reff * table_Reff))
    elif interp_type.lower().strip() == 'linear':
        r_interp = scp_interp.interp1d(table_rad, table_dlnrho_dlnr, fill_value='extrapolate',
                                       bounds_error=False, kind='linear')
        dlnrho_dlnr_interp =     (r_interp(rarr / Reff * table_Reff)  )
    else:
        raise ValueError("interp type '{}' unknown!".format(interp_type))


    if (len(rarr) > 1):
        return dlnrho_dlnr_interp
    else:
        if isinstance(r*1., float):
            # Float input
            return dlnrho_dlnr_interp[0]
        else:
            # Length 1 array input
            return dlnrho_dlnr_interp

    return dlnrho_dlnr_interp





class BlackHole(MassModel):
    """
    Central black hole. Treated as a point source at r = 0.

    Parameters
    ----------
    BH_mass : float
        Log10 of the mass in solar units

    """
    BH_mass = DysmalParameter(default=1, bounds=(0., 12.))
    _subtype = 'baryonic'
    baryon_type = 'blackhole'

    def __init__(self, **kwargs):
        super(BlackHole, self).__init__(**kwargs)

    @staticmethod
    def evaluate(r, BH_mass):
        """
        Mass surface density of a BH (treat like delta function)
        """
        # Ensure it's an array:
        if isinstance(r*1., float):
            rarr = np.array([r])
        else:
            rarr = np.array(r)
        # Ensure all radii are 0. or positive:
        rarr = np.abs(rarr)

        mr = r * 0.

        wh0 = np.where((rarr == 0.))[0]
        mr[wh0] = BH_mass

        if (len(rarr) > 1):
            return mr
        else:
            if isinstance(r*1., float):
                # Float input
                return mr[0]
            else:
                # Length 1 array input
                return mr

    def enclosed_mass(self, r):
        """
        Central black hole enclosed mass (treat as step function)

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        menc : float or array
            Enclosed mass profile  (unit: Msun)
        """

        menc = r*0. + np.power(10.,self.BH_mass)

        return menc


    def projected_enclosed_mass(self, r):
        # Point source: 2D is same as 3D
        return self.enclosed_mass(r)

    def circular_velocity(self, r):
        """
        Circular velocity as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        vcirc : float or array
            Circular velocity in km/s
        """
        return super(BlackHole, self).circular_velocity(r)


    def light_profile(self, r):
        """
        Conversion from mass to light as a function of radius
            Assuming NO LIGHT emitted by central BH (eg, ignoring any emission in surrounding medium, eg flares)

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        light : float or array
            Relative line flux as a function of radius
        """
        return r * 0.


class ExpDisk(MassModel):
    """
    Infinitely thin exponential disk (i.e. Freeman disk)

    Parameters
    ----------
    total_mass : float
        Log of total mass of the disk in solar units

    r_eff : float
        Effective radius in kpc

    baryon_type : {'gas+stars', 'stars', 'gas'}
        What type of baryons are included. Used for dlnrhogas/dlnr

    """

    total_mass = DysmalParameter(default=1, bounds=(5, 14))
    r_eff = DysmalParameter(default=1, bounds=(0, 50))
    _subtype = 'baryonic'

    def __init__(self, baryon_type='gas+stars', **kwargs):
        self.baryon_type = baryon_type
        super(ExpDisk, self).__init__(**kwargs)

    @staticmethod
    def evaluate(r, total_mass, r_eff):
        """
        Mass surface density of a thin exponential disk
        """
        return surf_dens_exp_disk(r, 10.**total_mass, r_eff / 1.6783469900166612)

    @property
    def rd(self):
        #b1 = 1.6783469900166612   # scp_spec.gammaincinv(2.*n, 0.5), n=1
        return self.r_eff / 1.6783469900166612

    def enclosed_mass(self, r):
        """
        Enclosed mass as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        menc : float or array
            1D enclosed mass profile
        """
        return menc_exp_disk(r, 10**self.total_mass, self.rd)

    def circular_velocity(self, r):
        """
        Circular velocity as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        vcirc : float or array
            Circular velocity in km/s
        """
        vcirc = vcirc_exp_disk(r, 10**self.total_mass, self.rd)
        return vcirc

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
        light = surf_dens_exp_disk(r, 1.0, self.rd)
        return light

    def rhogas(self, r):
        """
        Mass surface density as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        surf_dens : float or array
            Mass surface density at `r` in units of Msun/kpc^2
        -------

        """
        return surf_dens_exp_disk(r, 10.**self.total_mass, self.rd)

    def dlnrhogas_dlnr(self, r):
        """
        Exponential disk asymmetric drift term

        Parameters
        ----------
        r : float or array
            Radius in kpc

        Returns
        -------
        log_drhodr : float or array
            Log surface density derivative as a function or radius

        Notes
        -----
        See [1]_ for derivation and specificall Equations 3-11

        References
        ----------
        .. [1] https://ui.adsabs.harvard.edu/abs/2010ApJ...725.2324B/abstract

        """
        # Shortcut for the exponential disk asymmetric drift term, from Burkert+10 eq 11:
        return -2. * (r / self.rd)


class Sersic(MassModel):
    """
    Mass distribution following a Sersic profile

    Parameters
    ----------
    total_mass : float
        Log10 of the total mass in solar units

    r_eff : float
        Effective (half-light) radius in kpc

    n : float
        Sersic index

    invq : float
        Ratio of the effective radius to the effective radius in the z-direction

    noord_flat : bool
        If True, use circular velocity profiles derived in Noordermeer 2008.
        If False, circular velocity is derived through `v_circular`

    baryon_type : {'gas+stars', 'stars', 'gas'}
        What type of baryons are included. Used for dlnrhogas/dlnr

    Notes
    -----
    Model formula:

    .. math::

        M(r) = M_e \exp \\left\{ -b_n \\left[ \\left( \\frac{r}{r_{\mathrm{eff}}} \\right)^{1/n} -1 \\right] \\right\}

    The constant :math:`b_n` is defined such that :math:`r_{\mathrm{eff}}` contains half the total
    mass, and can be solved for numerically.

    .. math::

        \Gamma(2n) = 2\gamma (b_n,2n)

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from dysmalpy.models import Sersic
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(111, xscale='log', yscale='log')
        s1 = Sersic(total_mass=10, r_eff=5, n=1)
        r=np.arange(0, 100, .01)

        for n in range(1, 10):
             s1.n = n
             plt.plot(r, s1(r), color=str(float(n) / 15))

        plt.axis([1e-1, 30, 1e5, 1e10])
        plt.xlabel('log Radius [kpc]')
        plt.ylabel('log Mass Surface Density [log Msun/kpc^2]')
        plt.text(.25, 8.e7, 'n=1')
        plt.text(.25, 3.e9, 'n=10')
        plt.show()

    """

    total_mass = DysmalParameter(default=1, bounds=(5, 14))
    r_eff = DysmalParameter(default=1, bounds=(0, 50))
    n = DysmalParameter(default=1, bounds=(0, 8))

    _subtype = 'baryonic'

    def __init__(self, invq=1.0, noord_flat=False, baryon_type='gas+stars', **kwargs):

        self.invq = invq
        self.noord_flat = noord_flat
        self.baryon_type = baryon_type
        super(Sersic, self).__init__(**kwargs)

    @staticmethod
    def evaluate(r, total_mass, r_eff, n):
        """
        Sersic mass surface density
        """

        return sersic_mr(r, 10**total_mass, n, r_eff)

    def enclosed_mass(self, r):
        """
        Sersic enclosed mass

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        menc : float or array
            Enclosed mass profile
        """

        if self.noord_flat:
            # Take Noordermeer+08 vcirc, and then get Menc from vcirc
            return menc_from_vcirc(apply_noord_flat(r, self.r_eff, 10**self.total_mass,
                                     self.n, self.invq), r)

        else:
            return sersic_menc_2D_proj(r, 10**self.total_mass, self.n, self.r_eff)

    def projected_enclosed_mass(self, r):
        return sersic_menc_2D_proj(r, 10**self.total_mass, self.n, self.r_eff)

    def circular_velocity(self, r):
        """
        Circular velocity as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        vcirc : float or array
            Circular velocity in km/s
        """
        if self.noord_flat:
            vcirc = apply_noord_flat(r, self.r_eff, 10**self.total_mass,
                                     self.n, self.invq)
        else:
            vcirc = super(Sersic, self).circular_velocity(r)

        return vcirc

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
        return sersic_mr(r, 1.0, self.n, self.r_eff)

    def rhogas(self, r):
        """
        Mass density as a function of radius (if noord_flat; otherwise surface density)

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        dens : float or array
            Mass density at `r` in units of Msun/kpc^3 (if noord_flat; otherwise surface density)
        """

        if 'gas' in self.baryon_type.lower().strip():

            if self.noord_flat:
                rhogas = sersic_curve_rho(r, self.r_eff, 10**self.total_mass, self.n, self.invq)

            else:
                rhogas = sersic_mr(r, 10**self.total_mass, self.n, self.r_eff)
        else:
            rhogas = r * 0.

        return rhogas

    def dlnrhogas_dlnr(self, r):
        """
        Sersic asymmetric drift term

        Parameters
        ----------
        r : float or array
            Radius in kpc

        Returns
        -------
        log_drhodr : float or array
            Log surface density derivative as a function or radius
        """
        if 'gas' in self.baryon_type.lower().strip():
            if self.noord_flat:
                dlnrhogas_dlnr_arr = sersic_curve_dlnrho_dlnr(r, self.r_eff, self.n, self.invq)

            else:
                bn = scp_spec.gammaincinv(2. * self.n, 0.5)
                dlnrhogas_dlnr_arr = -2. * (bn / self.n) * np.power(r/self.r_eff, 1./self.n)
        else:
            dlnrhogas_dlnr_arr = r * 0.

        return dlnrhogas_dlnr_arr


class DiskBulge(MassModel):
    """
    Mass distribution with a disk and bulge

    Parameters
    ----------
    total_mass : float
        Log10 of the combined disk and bulge in solar units

    r_eff_disk : float
        Effective radius of the disk in kpc

    n_disk : float
        Sersic index of the disk

    r_eff_bulge : float
        Effective radius of the bulge

    n_bulge : float
        Sersic index of the bulge

    bt : float
        Bulge-to-total mass ratio

    invq_disk : float
        Effective radius to effective height ratio for the disk

    invq_bulge : float
        Effective radius to effective height ratio for the bulge

    noord_flat : bool
        If True, use circular velocity profiles derived in Noordermeer 2008.
        If False, circular velocity is derived through `v_circular`

    light_component : {'disk', 'bulge', 'total'}
        Which component to use as the flux profile

    gas_component : {'disk', 'total'}
        Which component contributes to dlnrhogas/dlnr

    baryon_type : {'gas+stars', 'stars', 'gas'}
        What type of baryons are included. Used for dlnrhogas/dlnr

    Notes
    -----
    This model is the combination of 2 components, a disk and bulge, each described by
    a `Sersic`. The model is parametrized such that the B/T is a free parameter rather
    than the individual masses of the disk and bulge.
    """

    total_mass = DysmalParameter(default=10, bounds=(5, 14))
    r_eff_disk = DysmalParameter(default=1, bounds=(0, 50))
    n_disk = DysmalParameter(default=1, fixed=True, bounds=(0, 8))
    r_eff_bulge = DysmalParameter(default=1, bounds=(0, 50))
    n_bulge = DysmalParameter(default=4., fixed=True, bounds=(0, 8))
    bt = DysmalParameter(default=0.2, bounds=(0, 1))

    _subtype = 'baryonic'

    def __init__(self, invq_disk=5, invq_bulge=1, noord_flat=False,
                 light_component='disk', gas_component='disk', baryon_type='gas+stars',
                 **kwargs):

        self.invq_disk = invq_disk
        self.invq_bulge = invq_bulge
        self.noord_flat = noord_flat
        self.light_component = light_component
        self.gas_component = gas_component
        self.baryon_type = baryon_type

        super(DiskBulge, self).__init__(**kwargs)

    @staticmethod
    def evaluate(r, total_mass, r_eff_disk, n_disk, r_eff_bulge, n_bulge, bt):
        """Disk+Bulge mass surface density"""

        print("consider if Noord flat: this will be modified")
        mbulge_total = 10**total_mass*bt
        mdisk_total = 10**total_mass*(1 - bt)

        mr_bulge = sersic_mr(r, mbulge_total, n_bulge, r_eff_bulge)
        mr_disk = sersic_mr(r, mdisk_total, n_disk, r_eff_disk)

        return mr_bulge+mr_disk

    def enclosed_mass(self, r):
        """
        Disk+Bulge total enclosed mass

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        menc : float or array
            Enclosed mass profile
        """
        mbulge_total = 10 ** self.total_mass * self.bt
        mdisk_total = 10 ** self.total_mass * (1 - self.bt)

        if self.noord_flat:
            # TO FIX
            menc_bulge = menc_from_vcirc(apply_noord_flat(r, self.r_eff_bulge, mbulge_total,
                        self.n_bulge, self.invq_bulge), r)
            menc_disk =  menc_from_vcirc(apply_noord_flat(r, self.r_eff_disk,  mdisk_total,
                        self.n_disk,  self.invq_disk),  r)
        else:
            # 2D projected:
            menc_bulge = sersic_menc_2D_proj(r, mbulge_total, self.n_bulge, self.r_eff_bulge)
            menc_disk = sersic_menc_2D_proj(r, mdisk_total, self.n_disk, self.r_eff_disk)

        return menc_disk+menc_bulge

    def enclosed_mass_disk(self, r):
        """
        Enclosed mass of the disk component

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        menc : float or array
            Enclosed mass profile
        """
        mdisk_total = 10 ** self.total_mass * (1 - self.bt)

        if self.noord_flat:
            # TO FIX
            menc_disk =  menc_from_vcirc(apply_noord_flat(r, self.r_eff_disk,  mdisk_total,
                        self.n_disk,  self.invq_disk),  r)
        else:
            # 2D projected:
            menc_disk = sersic_menc_2D_proj(r, mdisk_total, self.n_disk, self.r_eff_disk)
        return menc_disk

    def enclosed_mass_bulge(self, r):
        """
        Enclosed mass of the bulge component

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        menc : float or array
            Enclosed mass profile
        """
        mbulge_total = 10 ** self.total_mass * self.bt

        if self.noord_flat:
            # TO FIX
            menc_bulge = menc_from_vcirc(apply_noord_flat(r, self.r_eff_bulge, mbulge_total,
                        self.n_bulge, self.invq_bulge), r)
        else:
            # 2D projected:
            menc_bulge = sersic_menc_2D_proj(r, mbulge_total, self.n_bulge, self.r_eff_bulge)

        return menc_bulge

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def projected_enclosed_mass(self, r):
        menc_disk = self.projected_enclosed_mass_disk(r)
        menc_bulge = self.projected_enclosed_mass_bulge(r)
        return menc_disk + menc_bulge

    def projected_enclosed_mass_disk(self, r):
        mdisk_total = 10 ** self.total_mass * (1 - self.bt)
        return sersic_menc_2D_proj(r, mdisk_total, self.n_disk, self.r_eff_disk)
    def projected_enclosed_mass_bulge(self, r):
        mbulge_total = 10 ** self.total_mass * self.bt
        return sersic_menc_2D_proj(r, mbulge_total, self.n_bulge, self.r_eff_bulge)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def circular_velocity_disk(self, r):
        """
        Circular velocity of the disk as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the circular velocity

        Returns
        -------
        vcirc : float or array
            Circular velocity in km/s
        """
        if self.noord_flat:
            mdisk_total = 10**self.total_mass*(1-self.bt)
            vcirc = apply_noord_flat(r, self.r_eff_disk, mdisk_total,
                                     self.n_disk, self.invq_disk)
        else:
            mass_enc = self.enclosed_mass_disk(r)
            vcirc = v_circular(mass_enc, r)

        return vcirc

    def circular_velocity_bulge(self, r):
        """
        Circular velocity of the bulge as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the circular velocity

        Returns
        -------
        vcirc : float or array
            Circular velocity in km/s
        """

        if self.noord_flat:
            mbulge_total = 10**self.total_mass*self.bt
            vcirc = apply_noord_flat(r, self.r_eff_bulge, mbulge_total,
                                     self.n_bulge, self.invq_bulge)
        else:
            mass_enc = self.enclosed_mass_bulge(r)
            vcirc = v_circular(mass_enc, r)

        return vcirc

    def circular_velocity(self, r):
        """
        Total circular velocity as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the circular velocity in kpc

        Returns
        -------
        vcirc : float or array
            Circular velocity in km/s
        """

        vbulge = self.circular_velocity_bulge(r)
        vdisk = self.circular_velocity_disk(r)

        vcirc = np.sqrt(vbulge**2 + vdisk**2)

        return vcirc

    def velocity_profile(self, r, modelset):
        """
        Total rotational velocity due to the disk+bulge

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the circular velocity in kpc

        modelset : `ModelSet`
            Full ModelSet this component belongs to

        Returns
        -------
        vrot : float or array
            Rotational velocity in km/s

        Notes
        -----
        This method requires a `ModelSet` input to be able to apply the pressure support
        correction due to the gas turbulence.

        """

        vcirc = self.circular_velocity(r)
        vrot = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc)
        return vrot

    def velocity_profile_disk(self, r, modelset):
        """
        Rotational velocity due to the disk

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the circular velocity in kpc

        modelset : `ModelSet`
            Full ModelSet this component belongs to

        Returns
        -------
        vrot : float or array
            Rotational velocity in km/s

        Notes
        -----
        This method requires a `ModelSet` input to be able to apply the pressure support
        correction due to the gas turbulence.

        """

        vcirc = self.circular_velocity_disk(r)
        vrot = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc)
        return vrot

    def velocity_profile_bulge(self, r, modelset):
        """
        Rotational velocity due to the bulge

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the circular velocity in kpc

        modelset : `ModelSet`
            Full ModelSet this component belongs to

        Returns
        -------
        vrot : float or array
            Rotational velocity in km/s

        Notes
        -----
        This method requires a `ModelSet` input to be able to apply the pressure support
        correction due to the gas turbulence.
        """

        vcirc = self.circular_velocity_bulge(r)
        vrot = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc)
        return vrot


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

        Notes
        -----
        The resulting light profile depends on what `DiskBulge.light_component` is set to.
        If 'disk' or 'bulge' then only the mass associated with the disk or bulge will
        be converted into light. If 'total', then both components will be used.
        """

        if self.light_component == 'disk':

            flux = sersic_mr(r, 1.0, self.n_disk, self.r_eff_disk)

        elif self.light_component == 'bulge':

            flux = sersic_mr(r, 1.0, self.n_bulge, self.r_eff_bulge)

        elif self.light_component == 'total':

            flux_disk = sersic_mr(r, 1.0-self.bt,
                                  self.n_disk, self.r_eff_disk)
            flux_bulge = sersic_mr(r, self.bt,
                                   self.n_bulge, self.r_eff_bulge)
            flux = flux_disk + flux_bulge

        else:

            raise ValueError("light_component can only be 'disk', 'bulge', "
                             "or 'total.'")

        return flux

    def rhogas_disk(self, r):
        """
        Mass density of the disk as a function of radius (if noord_flat; otherwise surface density)

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        dens : float or array
            Mass density at `r` in units of Msun/kpc^3 (if noord_flat; otherwise surface density)
        """

        if 'gas' in self.baryon_type.lower().strip():
            if self.gas_component in ['total', 'disk']:
                if self.noord_flat:
                    mdisk_total = 10**self.total_mass*(1 - self.bt)
                    rhogas = sersic_curve_rho(r, self.r_eff_disk, mdisk_total,
                                              self.n_disk, self.invq_disk)
                else:
                    mdisk_total = 10**self.total_mass*(1 - self.bt)
                    # Just use the surface density as "rho", as this is the razor-thin case
                    rhogas = sersic_mr(r, mdisk_total, self.n_disk, self.r_eff_disk)
            else:
                rhogas = r * 0.
        else:
            rhogas = r * 0.

        return rhogas


    def rhogas_bulge(self, r):
        """
        Mass density of the bulge as a function of radius (if noord_flat; otherwise surface density)

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        dens : float or array
            Mass density at `r` in units of Msun/kpc^3 (if noord_flat; otherwise surface density)
        """
        if 'gas' in self.baryon_type.lower().strip():
            # Only include bas in bulge if gas_component is 'total':
            if self.gas_component in ['total']:
                if self.noord_flat:
                    mbulge_total = 10**self.total_mass*self.bt
                    rhogas = sersic_curve_rho(r, self.r_eff_bulge, mbulge_total,
                                              self.n_bulge, self.invq_bulge)
                else:
                    mbulge_total = 10**self.total_mass*self.bt
                    # Just use the surface density as "rho", as this is the razor-thin case
                    rhogas = sersic_mr(r, mbulge_total, self.n_bulge, self.r_eff_bulge)

            else:
                rhogas = r * 0.
        else:
            rhogas = r * 0.

        return rhogas

    def rhogas(self, r):
        """
        Mass density as a function of radius (if noord_flat; otherwise surface density)

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        dens : float or array
            Mass density at `r` in units of Msun/kpc^3 (if noord_flat; otherwise surface density)
        """

        # All cases handled internally in rhogas_disk, rhogas_bulge
        rhogas = self.rhogas_disk(r) + self.rhogas_bulge(r)

        return rhogas

    def dlnrhogas_dlnr_disk(self, r):
        if self.noord_flat:
            dlnrhogas_dlnr_arr = sersic_curve_dlnrho_dlnr(r, self.r_eff_disk, self.n_disk, self.invq_disk)

            return dlnrhogas_dlnr_arr
        else:
            bn = scp_spec.gammaincinv(2. * self.n_disk, 0.5)
            return -2. * (bn / self.n_disk) * np.power(r/self.r_eff_disk, 1./self.n_disk)

    def dlnrhogas_dlnr_bulge(self, r):
        if self.noord_flat:
            dlnrhogas_dlnr_arr = sersic_curve_dlnrho_dlnr(r, self.r_eff_bulge, self.n_bulge, self.invq_bulge)

            return dlnrhogas_dlnr_arr
        else:
            bn = scp_spec.gammaincinv(2. * self.n_bulge, 0.5)
            return -2. * (bn / self.n_bulge) * np.power(r/self.r_eff_bulge, 1./self.n_bulge)

    def dlnrhogas_dlnr(self, r):
        """
        Asymmetric drift term for the combined disk and bulge

        Parameters
        ----------
        r : float or array
            Radius in kpc

        Returns
        -------
        log_drhodr : float or array
            Log surface density derivative as a function or radius
        """

        if 'gas' in self.baryon_type.lower().strip():
            if self.gas_component == 'total':
                rhogasD = self.rhogas_disk(r)
                rhogasB = self.rhogas_bulge(r)

                dlnrhogas_dlnr_tot = (1./(rhogasD + rhogasB)) * \
                            (rhogasD*self.dlnrhogas_dlnr_disk(r) + rhogasB*self.dlnrhogas_dlnr_bulge(r))
            elif self.gas_component == 'disk':
                dlnrhogas_dlnr_tot = self.dlnrhogas_dlnr_disk(r)
        else:
            dlnrhogas_dlnr_tot = r * 0.

        return dlnrhogas_dlnr_tot


class LinearDiskBulge(MassModel):
    """
    Mass distribution with a disk and bulge

    Parameters
    ----------
    total_mass : float
        Combined disk and bulge mass in solar units

    r_eff_disk : float
        Effective radius of the disk in kpc

    n_disk : float
        Sersic index of the disk

    r_eff_bulge : float
        Effective radius of the bulge

    n_bulge : float
        Sersic index of the bulge

    bt : float
        Bulge-to-total mass ratio

    invq_disk : float
        Effective radius to effective height ratio for the disk

    invq_bulge : float
        Effective radius to effective height ratio for the bulge

    noord_flat : bool
        If True, use circular velocity profiles derived in Noordermeer 2008.
        If False, circular velocity is derived through `v_circular`

    light_component : {'disk', 'bulge', 'total'}
        Which component to use as the flux profile

    baryon_type : {'gas+stars', 'stars', 'gas'}
        What type of baryons are included. Used for dlnrhogas/dlnr

    Notes
    -----
    This model is the exactly the same as `DiskBulge` except that `total_mass`
    is in linear units instead of log.
    """

    total_mass = DysmalParameter(default=10, bounds=(5, 14))
    r_eff_disk = DysmalParameter(default=1, bounds=(0, 50))
    n_disk = DysmalParameter(default=1, fixed=True, bounds=(0, 8))
    r_eff_bulge = DysmalParameter(default=1, bounds=(0, 50))
    n_bulge = DysmalParameter(default=4., fixed=True, bounds=(0, 8))
    bt = DysmalParameter(default=0.2, bounds=(0, 1))

    _subtype = 'baryonic'

    def __init__(self, invq_disk=5, invq_bulge=1, noord_flat=False,
                 light_component='disk', baryon_type='gas+stars', **kwargs):

        self.invq_disk = invq_disk
        self.invq_bulge = invq_bulge
        self.noord_flat = noord_flat
        self.light_component = light_component
        self.baryon_type = baryon_type

        super(LinearDiskBulge, self).__init__(**kwargs)

    @staticmethod
    def evaluate(r, total_mass, r_eff_disk, n_disk, r_eff_bulge, n_bulge, bt):
        """Disk+Bulge mass surface density"""
        print("consider if Noord flat: this will be modified")
        mbulge_total = total_mass*bt
        mdisk_total = total_mass*(1 - bt)

        mr_bulge = sersic_mr(r, mbulge_total, n_bulge, r_eff_bulge)
        mr_disk = sersic_mr(r, mdisk_total, n_disk, r_eff_disk)

        return mr_bulge+mr_disk

    def enclosed_mass(self, r):
        """
        Disk+Bulge total enclosed mass

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        menc : float or array
            Enclosed mass profile
        """
        mbulge_total = self.total_mass * self.bt
        mdisk_total = self.total_mass * (1 - self.bt)

        if self.noord_flat:
            # TO FIX
            menc_bulge = menc_from_vcirc(apply_noord_flat(r, self.r_eff_bulge, mbulge_total,
                        self.n_bulge, self.invq_bulge), r)
            menc_disk =  menc_from_vcirc(apply_noord_flat(r, self.r_eff_disk,  mdisk_total,
                        self.n_disk,  self.invq_disk),  r)
        else:
            # 2D projected:
            menc_bulge = sersic_menc_2D_proj(r, mbulge_total, self.n_bulge, self.r_eff_bulge)
            menc_disk = sersic_menc_2D_proj(r, mdisk_total, self.n_disk, self.r_eff_disk)

        return menc_disk+menc_bulge

    def enclosed_mass_disk(self, r):
        """
        Enclosed mass of the disk component

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        menc : float or array
            Enclosed mass profile
        """
        mdisk_total = self.total_mass * (1 - self.bt)

        if self.noord_flat:
            # TO FIX
            menc_disk =  menc_from_vcirc(apply_noord_flat(r, self.r_eff_disk,  mdisk_total,
                        self.n_disk,  self.invq_disk),  r)
        else:
            # 2D projected:
            menc_disk = sersic_menc_2D_proj(r, mdisk_total, self.n_disk, self.r_eff_disk)

        return menc_disk

    def enclosed_mass_bulge(self, r):
        """
        Enclosed mass of the bulge component

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        menc : float or array
            Enclosed mass profile
        """
        mbulge_total = self.total_mass * self.bt

        if self.noord_flat:
            # TO FIX
            menc_bulge = menc_from_vcirc(apply_noord_flat(r, self.r_eff_bulge, mbulge_total,
                        self.n_bulge, self.invq_bulge), r)
        else:
            # 2D projected:
            menc_bulge = sersic_menc_2D_proj(r, mbulge_total, self.n_bulge, self.r_eff_bulge)

        return menc_bulge

    def circular_velocity_disk(self, r):
        """
        Circular velocity of the disk as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the circular velocity

        Returns
        -------
        vcirc : float or array
            Circular velocity in km/s
        """
        if self.noord_flat:
            mdisk_total = self.total_mass*(1-self.bt)
            vcirc = apply_noord_flat(r, self.r_eff_disk, mdisk_total,
                                     self.n_disk, self.invq_disk)
        else:
            mass_enc = self.enclosed_mass_disk(r)
            vcirc = v_circular(mass_enc, r)

        return vcirc

    def circular_velocity_bulge(self, r):
        """
        Circular velocity of the bulge as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the circular velocity

        Returns
        -------
        vcirc : float or array
            Circular velocity in km/s
        """
        if self.noord_flat:
            mbulge_total = self.total_mass*self.bt
            vcirc = apply_noord_flat(r, self.r_eff_bulge, mbulge_total,
                                     self.n_bulge, self.invq_bulge)
        else:
            mass_enc = self.enclosed_mass_bulge(r)
            vcirc = v_circular(mass_enc, r)

        return vcirc

    def circular_velocity(self, r):
        """
        Total Circular velocity as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the circular velocity

        Returns
        -------
        vcirc : float or array
            Circular velocity in km/s
        """
        vbulge = self.circular_velocity_bulge(r)
        vdisk = self.circular_velocity_disk(r)

        vcirc = np.sqrt(vbulge**2 + vdisk**2)

        return vcirc

    def velocity_profile(self, r, modelset):
        """
        Total rotational velocity due to the disk+bulge

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the circular velocity in kpc

        modelset : `ModelSet`
            Full ModelSet this component belongs to

        Returns
        -------
        vrot : float or array
            Rotational velocity in km/s

        Notes
        -----
        This method requires a `ModelSet` input to be able to apply the pressure support
        correction due to the gas turbulence.

        """
        vcirc = self.circular_velocity(r)
        vrot = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc)
        return vrot

    def velocity_profile_disk(self, r, modelset):
        """
        Rotational velocity due to the disk

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the circular velocity in kpc

        modelset : `ModelSet`
            Full ModelSet this component belongs to

        Returns
        -------
        vrot : float or array
            Rotational velocity in km/s

        Notes
        -----
        This method requires a `ModelSet` input to be able to apply the pressure support
        correction due to the gas turbulence.

        """
        vcirc = self.circular_velocity_disk(r)
        vrot = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc)
        return vrot

    def velocity_profile_bulge(self, r, modelset):
        """
        Rotational velocity due to the bulge

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the circular velocity in kpc

        modelset : `ModelSet`
            Full ModelSet this component belongs to

        Returns
        -------
        vrot : float or array
            Rotational velocity in km/s

        Notes
        -----
        This method requires a `ModelSet` input to be able to apply the pressure support
        correction due to the gas turbulence.

        """
        vcirc = self.circular_velocity_bulge(r)
        vrot = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc)
        return vrot


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

        Notes
        -----
        The resulting light profile depends on what `DiskBulge.light_component` is set to.
        If 'disk' or 'bulge' then only the mass associated with the disk or bulge will
        be converted into light. If 'total', then both components will be used.
        """

        if self.light_component == 'disk':

            flux = sersic_mr(r, 1.0, self.n_disk, self.r_eff_disk)

        elif self.light_component == 'bulge':

            flux = sersic_mr(r, 1.0, self.n_bulge, self.r_eff_bulge)

        elif self.light_component == 'total':

            flux_disk = sersic_mr(r, 1.0-self.bt,
                                  self.n_disk, self.r_eff_disk)
            flux_bulge = sersic_mr(r, self.bt,
                                   self.n_bulge, self.r_eff_bulge)
            flux = flux_disk + flux_bulge

        else:

            raise ValueError("light_component can only be 'disk', 'bulge', "
                             "or 'total.'")

        return flux
