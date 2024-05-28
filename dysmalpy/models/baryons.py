# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Baryon mass models for DysmalPy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import os, copy
import logging
import glob

# Third party imports
import numpy as np
import scipy.special as scp_spec
import scipy.integrate as scp_int
import scipy.optimize as scp_opt
import scipy.interpolate as scp_interp
import scipy.optimize as scp_opt
import scipy.integrate as scp_int
import astropy.constants as apy_con


from astropy.table import Table

# Local imports
from .base import MassModel, _LightMassModel, v_circular, \
                  sersic_mr, _I0_gaussring
# from .base import menc_from_vcirc
from dysmalpy.parameters import DysmalParameter

__all__ = ['Sersic', 'DiskBulge', 'LinearDiskBulge', 'ExpDisk', 'BlackHole',
           'GaussianRing',
           'surf_dens_exp_disk', 'menc_exp_disk', 'vcirc_exp_disk', 'sersic_menc_2D_proj',
           'mass_comp_conditional_ring',
           'NoordFlat', 'InfThinMassiveGaussianRing']

# NEW, RECALCULATED NOORDERMEER DIRECTORY: INCLUDES dlnrho/dlnr; MASS PROFILES
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
# located one up:
dir_path = os.sep.join(dir_path.split(os.sep)[:-1])
_dir_deprojected_sersic_models = os.sep.join([dir_path, "data",
                                "deprojected_sersic_models_tables", ""])



# MASSIVE RING DIRECTORIES:
_dir_gaussian_ring_tables = os.getenv('GAUSSIAN_RING_PROFILE_DATADIR', None)

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
logger.setLevel(logging.INFO)


import warnings
warnings.filterwarnings("ignore")



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


def mass_comp_conditional_ring(param, modelset):
    """
    Basic conditional prior on the mass of the other component(s) (i.e., a bulge or halo)
    when fitting both a massive ring and one or more other mass components, returning True/False

    This conditional could apply to the bulge mass, or to the halo mass (probably best to not do both).

    Intended to be passed as f_bounds function when setting, e.g.,
        sersic_comp.total_mass.prior = ConditionalEmpiricalUniformPrior(f_cond=mass_comp_conditional_bounds_ring)
    """
    # Double check param + model values are same:
    if param.value != modelset.components[param._model._name].__getattribute__(param._name).value:
        raise ValueError

    # Test radius array:
    rarr = np.arange(0., 10.1, 0.1)

    return np.all(np.isfinite(modelset.circular_velocity(rarr)))



##########################
class NoordFlat(object):
    """
    Class to handle circular velocities / enclosed mass profiles for a thick Sersic component.

    Lookup tables are numerically calculated from the derivations provided in
    `Noordermeer 2008 <https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/abstract>`_ which properly accounted for the thickness of the mass component.

    The lookup table provides rotation curves for Sersic components with
    `n` = 0.5 - 8 at steps of 0.1 and `invq` = [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 100].
    If the given `n` and/or `invq` are not one of these values then the nearest
    ones are used.

    References
    ----------
    `Noordermeer 2008 <https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/abstract>`_

    Parameters
    ----------
    n : float
        Sersic index

    invq: float
        Sersic index

    """

    def __init__(self, n=None, invq=None):
        self._n = n
        self._invq = invq
        self._n_current = None
        self._invq_current = None

        self.rho_interp_func = None
        self.dlnrhodlnr_interp_func = None
        self._rho_interp_type = None
        self._dlnrhodlnr_interp_type = None

        self._reset_interps()

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if value < 0:
            raise ValueError("Sersic index can't be negative!")
        self._n = value

        # Reset vcirc interp:
        self._reset_interps()

    @property
    def invq(self):
        return self._invq

    @invq.setter
    def invq(self, value):
        if value < 0:
            raise ValueError("Invq can't be negative!")
        self._invq = value

        # Reset vcirc interp:
        self._reset_interps()

    def _reset_interps(self):
        self._set_vcirc_interp()
        self._set_menc_interp()
        if self.rho_interp_func is not None:
            self._set_rho_interp(interp_type=self._rho_interp_type)
        if self.dlnrhodlnr_interp_func is not None:
            self._set_dlnrhodlnr_interp(interp_type=self._dlnrhodlnr_interp_type)



    def read_deprojected_sersic_table(self):
        # Use the "typical" collection of table values:
        table_n = np.arange(0.5, 8.1, 0.1)   # Sersic indices
        table_invq = np.array([1., 2., 3., 4., 5., 6., 7., 8., 10., 20., 100.,
                        1.11, 1.43, 1.67, 3.33, 0.5, 0.67])  # 1:1, 1:2, 1:3, ... flattening  [also prolate 2:1, 1.5:1]

        nearest_n = table_n[ np.argmin( np.abs(table_n - self.n) ) ]
        nearest_invq = table_invq[ np.argmin( np.abs( table_invq - self.invq) ) ]

        file_sersic = _dir_deprojected_sersic_models + 'deproj_sersic_model_n{:0.1f}_invq{:0.2f}.fits'.format(nearest_n, nearest_invq)

        try:
            t = Table.read(file_sersic)
        except:
            # REMOVE BACKWARDS COMPATIBILITY!
            raise ValueError("File {} not found. _dir_deprojected_sersic_models={}.".format(file_sersic))
        return t[0]


    def _set_vcirc_interp(self):
        # SHOULD BE EXACTLY, w/in numerical limitations, EQUIV TO OLD CALCULATION
        table = self.read_deprojected_sersic_table()

        N2008_vcirc =        table['vcirc']
        N2008_rad =          table['R']
        self.N2008_Re =      table['Reff']
        self.N2008_mass =    table['total_mass']

        self.vcirc_interp = scp_interp.interp1d(N2008_rad, N2008_vcirc,
                                       fill_value="extrapolate")
        # vcirc = (v_interp(r / r_eff * N2008_Re) * np.sqrt(
        #          mass / N2008_mass) * np.sqrt(N2008_Re / r_eff))

        # return vcirc

    def _set_menc_interp(self):
        table = self.read_deprojected_sersic_table()

        table_Rad =  table['R']
        table_menc = table['menc3D_sph']

        # Clean up values inside rmin:  Add the value at r=0: menc=0
        if table['R'][0] > 0.:
            table_Rad = np.append(0., table_Rad)
            table_menc = np.append(0., table_menc)

        self.menc_interp = scp_interp.interp1d(table_Rad, table_menc, 
                                        fill_value="extrapolate")

    def circular_velocity(self, r, r_eff, mass):
        """
        Calculate circular velocity for a thick Sersic component, by interpolating

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate the circular velocity in kpc

        r_eff : float
            Effective radius of the Sersic component in kpc

        mass : float
            Total mass of the Sersic component

        Returns
        -------
        vcirc : float or array
            Circular velocity at each given `r`, in km/s

        Notes
        -----
        This function determines the circular velocity as a function of radius for
        a Sersic component with a total mass, `mass`, Sersic index, `n`, and
        an effective radius to scale height ratio, `invq`. This uses lookup tables
        numerically calculated from the derivations provided in `Noordermeer 2008 <https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/abstract>`_ 
        which properly account for the thickness of the mass component.

        References
        ----------
        `Noordermeer 2008 <https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/abstract>`_
        """
        vcirc = (self.vcirc_interp(r / r_eff * self.N2008_Re) * np.sqrt(
                 mass / self.N2008_mass) * np.sqrt(self.N2008_Re / r_eff))

        return vcirc


    def enclosed_mass(self, r, r_eff, mass):
        """
        Calculate enclosed mass for a thick Sersic component, by interpolating

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate the circular velocity in kpc

        r_eff : float
            Effective radius of the Sersic component in kpc

        mass : float
            Total mass of the Sersic component

        Returns
        -------
        menc : float or array
            Enclosed mass (in a sphere) at each given `r`, in solar masses

        Notes
        -----
        This function determines the spherical enclosed mass as a function of radius for
        a Sersic component with a total mass, `mass`, Sersic index, `n`, and
        an effective radius to scale height ratio, `invq`. This uses lookup tables
        numerically calculated from the derivations provided in `Noordermeer 2008 <https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/abstract>`_ 
        (as extended in `Price et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022A%26A...665A.159P/abstract>`_)
        which properly account for the thickness of the mass component.

        References
        ----------
        `Noordermeer 2008 <https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/abstract>`_
        `Price et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022A%26A...665A.159P/abstract>`_
        """

        menc = self.menc_interp(r / r_eff * self.N2008_Re) * (mass / self.N2008_mass)
        
        # # TEST:
        # print("USING OLD vcirc from menc! (baryons.py)")
        # menc = menc_from_vcirc(self.circular_velocity(r, r_eff, mass), r)

        return menc

    def rho(self, r, Reff, total_mass, interp_type='linear'):
        if (self._rho_interp_type != interp_type) | (self.rho_interp_func is None):
            # Update rho funcs:
            self._set_rho_interp(interp_type=interp_type)

        # Ensure it's an array:
        if isinstance(r*1., float):
            rarr = np.array([r])
        else:
            rarr = np.array(r)
        # Ensure all radii are 0. or positive:
        rarr = np.abs(rarr)

        scale_fac = (total_mass / self.table_mass) * (self.table_Reff / Reff)**3

        if interp_type.lower().strip() == 'cubic':
            rho_interp = np.zeros(len(rarr))
            wh_in =     np.where((rarr <= self.table_rad_rho.max()) & (rarr >= self.table_rad_rho.min()))[0]
            wh_extrap = np.where((rarr > self.table_rad_rho.max()) | (rarr < self.table_rad_rho.min()))[0]
            rho_interp[wh_in] =  (self.rho_interp_func(rarr[wh_in] / Reff * self.table_Reff) * scale_fac )
            rho_interp[wh_extrap] = (self.rho_interp_extrap_func(rarr[wh_extrap] / Reff * self.table_Reff) * scale_fac)
        elif interp_type.lower().strip() == 'linear':
            rho_interp =  (self.rho_interp_func(rarr / Reff * self.table_Reff) * scale_fac )

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


    def dlnrho_dlnr(self, r, Reff, interp_type='linear'):

        """
        Calculate log mass density gradient for a thick Sersic component, by interpolating. 

        Can be used to determine an alternative pressure support correction. 

        References
        ----------
        `Noordermeer 2008 <https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/abstract>`_
        `Price et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022A%26A...665A.159P/abstract>`_

        """
        if (self._dlnrhodlnr_interp_type != interp_type) | (self.dlnrhodlnr_interp_func is None):
            # Update rho funcs:
            self._set_dlnrhodlnr_interp(interp_type=interp_type)

        # Ensure it's an array:
        if isinstance(r*1., float):
            rarr = np.array([r])
        else:
            rarr = np.array(r)
        # Ensure all radii are 0. or positive:
        rarr = np.abs(rarr)


        if interp_type.lower().strip() == 'cubic':
            dlnrho_dlnr_interp = np.zeros(len(rarr))
            wh_in =     np.where((rarr <= self.table_rad_dlnrhodlnr.max()) & \
                            (rarr >= self.table_rad_dlnrhodlnr.min()))[0]
            wh_extrap = np.where((rarr > self.table_rad_dlnrhodlnr.max()) | \
                            (rarr < self.table_rad_dlnrhodlnr.min()))[0]
            dlnrho_dlnr_interp[wh_in] = (self.dlnrhodlnr_interp_func(rarr[wh_in] / Reff * self.table_Reff) )
            dlnrho_dlnr_interp[wh_extrap] = (self.dlnrhodlnr_interp_func_extrap(rarr[wh_extrap] / Reff * self.table_Reff))
        elif interp_type.lower().strip() == 'linear':
            dlnrho_dlnr_interp = (self.dlnrhodlnr_interp_func(rarr / Reff * self.table_Reff)  )
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

    def _set_rho_interp(self, interp_type='linear'):
        self._rho_interp_type = interp_type

        table = self.read_deprojected_sersic_table()

        table_rho =     table['rho']
        table_rad =     table['R']
        self.table_Reff =    table['Reff']
        self.table_mass =    table['total_mass']

        # Drop nonfinite parts:
        whfin = np.where(np.isfinite(table_rho))[0]
        table_rho = table_rho[whfin]
        table_rad = table_rad[whfin]

        self.table_rad_rho  =    table_rad


        if interp_type.lower().strip() == 'cubic':
            self.rho_interp_func = scp_interp.interp1d(table_rad, table_rho,
                        fill_value=np.NaN, bounds_error=False, kind='cubic')
            self.rho_interp_extrap_func = scp_interp.interp1d(table_rad, table_rho,
                    fill_value='extrapolate', kind='linear')
        elif interp_type.lower().strip() == 'linear':
            self.rho_interp_func = scp_interp.interp1d(table_rad, table_rho, fill_value='extrapolate',
                                           bounds_error=False, kind='linear')

        else:
            raise ValueError("interp type '{}' unknown!".format(interp_type))


    def _set_dlnrhodlnr_interp(self, interp_type='linear'):
        self._dlnrhodlnr_interp_type = interp_type

        table = self.read_deprojected_sersic_table()

        table_dlnrho_dlnr =     table['dlnrho_dlnR']
        table_rad =             table['R']
        self.table_Reff =       table['Reff']
        self.table_mass =       table['total_mass']

        # Drop nonfinite parts:
        whfin = np.where(np.isfinite(table_dlnrho_dlnr))[0]
        table_dlnrho_dlnr = table_dlnrho_dlnr[whfin]
        table_rad = table_rad[whfin]

        self.table_rad_dlnrhodlnr = table_rad


        if interp_type.lower().strip() == 'cubic':
            self.dlnrhodlnr_interp_func = scp_interp.interp1d(table_rad, table_dlnrho_dlnr,
                    fill_value=np.NaN, bounds_error=False, kind='cubic')
            self.dlnrhodlnr_interp_func_extrap = scp_interp.interp1d(table_rad,
                    table_dlnrho_dlnr, fill_value='extrapolate', kind='linear')

        elif interp_type.lower().strip() == 'linear':
            self.dlnrhodlnr_interp_func = scp_interp.interp1d(table_rad, table_dlnrho_dlnr,
                    fill_value='extrapolate', bounds_error=False, kind='linear')
        else:
            raise ValueError("interp type '{}' unknown!".format(interp_type))



##########################
class InfThinMassiveGaussianRing(object):
    """
    Class to handle circular velocities / enclosed mass profiles for an infinitely thin, massive Gaussian ring.

    Lookup tables are numerically calculated, following B&T and Bovy online galaxies textbook.

    The lookup table provides rotation curves for Gaussian rings with
    `invh` = 0.5 - 8 at steps of 0.1 and `invq` = [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 100],
    where :math:`\mathrm{invh} = R_{\mathrm{peak}} / \mathrm{FWHM}_{\mathrm{ring}}`
    If the given `n` and/or `invq` are not one of these values then the nearest
    ones are used.

    Parameters
    ----------
    invh : float
        Ratio of `R_peak` / `ring_FWHM`

    """

    def __init__(self, invh=None):
        self._invh = invh
        self._invh_current = None

        self._reset_interps()

    @property
    def invh(self):
        return self._invh

    @invh.setter
    def invh(self, value):
        if value < 0:
            #raise ValueError("Invh can't be negative!")
            logger.warning('Invh is negative -- undefined, so interps will be all NaN!!')
        self._invh = value

        # Reset vcirc interp:
        self._reset_interps()

    def _reset_interps(self):
        # self._set_vcirc_interp()
        self._set_potential_gradient_interp()
        self._set_menc_interp()

    def read_ring_table(self):
        # Use the "typical" collection of table values:

        #--------------------------------
        # Glob values from path:
        table_invh = []
        name_base = 'gauss_ring_profile_invh'
        fnames_glob = glob.glob(_dir_gaussian_ring_tables+name_base+'*.fits')
        for fn in fnames_glob:
            invh_str = fn.split(name_base)[-1].split('.fits')[0]
            table_invh.append(float(invh_str))

        table_invh = np.array(table_invh)
        table_invh.sort()

        # #--------------------------------
        # table_invh =    np.array([0.01, 0.05,
        #                           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,
        #                           0.25, 0.75,
        #                           1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.,
        #                           2.25, 2.5, 2.75, 3., 3.5,
        #                           3.33, 6.67,
        #                           4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5,
        #                           8., 8.5, 9., 9.5, 10., 12.5,
        #                           15., 20., 25., 50.])
        # #--------------------------------

        nearest_invh = table_invh[ np.argmin( np.abs( table_invh - self.invh) ) ]

        if self.invh != nearest_invh:
            logger.warning('Using table for massive gausring with NON-EXACT invh!!'
                           'Model invh = {} ; table invh = {}'.format(self.invh, nearest_invh))

        file_gausring = _dir_gaussian_ring_tables + 'gauss_ring_profile_invh{:0.2f}.fits'.format(nearest_invh)
        try:
            t = Table.read(file_gausring)
        except:
            raise ValueError("File {} not found. _dir_gaussian_ring_tables={}. Check that system var ${} is set correctly.".format(file_gausring,
                        _dir_gaussian_ring_tables, 'GAUSSIAN_RING_PROFILE_DATADIR'))

        return t[0]

    # def _set_vcirc_interp(self):
    #     table = self.read_ring_table()
    #
    #     tab_rad =               table['R']
    #     tab_vcirc =             table['vcirc']
    #     self.tab_invh =         table['invh']
    #     self.tab_R_peak =       table['R_peak']
    #     self.tab_ring_FWHM =    table['ring_FWHM']
    #     self.tab_mass =         table['total_mass']
    #
    #     self.vcirc_interp = scp_interp.interp1d(tab_rad, tab_vcirc,
    #                                    fill_value="extrapolate")
    #
    #     # scale_fac = np.sqrt(total_mass / table_mass) * np.sqrt(table_Rpeak / R_peak)
    #     # vcirc_interp = v_interp(Rarr / R_peak * table_Rpeak) * scale_fac


    def _set_potential_gradient_interp(self):
        if self.invh < 0:
            self.potl_grad_interp = None
        else:
            table = self.read_ring_table()

            tab_rad =               table['R']
            tab_potl_grad =         table['potential_gradient']
            self.tab_invh =         table['invh']
            self.tab_R_peak =       table['R_peak']
            self.tab_ring_FWHM =    table['ring_FWHM']
            self.tab_mass =         table['total_mass']

            self.potl_grad_interp = scp_interp.interp1d(tab_rad, tab_potl_grad,
                                           fill_value="extrapolate")

            # scale_fac = (total_mass / table_mass) * (table_Rpeak / R_peak)**2
            # potential_gradient_interp = potl_grad_interp(Rarr / R_peak * table_Rpeak) * scale_fac

    def _set_menc_interp(self):
        if self.invh < 0:
            self.menc_interp = None
        else:
            table = self.read_ring_table()

            tab_rad =               table['R']
            tab_menc =              table['menc']
            self.tab_invh =         table['invh']
            self.tab_R_peak =       table['R_peak']
            self.tab_ring_FWHM =    table['ring_FWHM']
            self.tab_mass =         table['total_mass']

            self.menc_interp = scp_interp.interp1d(tab_rad, tab_menc, fill_value="extrapolate")

            # scale_fac = (total_mass / table_mass)
            # menc_interp = m_interp(Rarr / R_peak * table_Rpeak) * scale_fac

    # def vcirc(self, R, R_peak, total_mass):
    #     """
    #     Calculate circular velocity for a inf thin massive gaussian Ring
    #
    #     Parameters
    #     ----------
    #     R : float or array
    #         Radius or radii at which to calculate the circular velocity in kpc
    #
    #     R_peak : float
    #         Peak of Gaussian ring in kpc
    #
    #     total_mass : float
    #         Total mass of the Gaussian ring component
    #
    #     Returns
    #     -------
    #     vcirc : float or array
    #         Circular velocity at each given `R`
    #
    #     Notes
    #     -----
    #     This function determines the circular velocity as a function of radius for
    #     a massive Gaussian ring component with a total mass, `total_mass`,
    #     and a ring peak radius to ring FWHM ratio, `invh`.
    #     This uses numerically calculated lookup tables.
    #
    #     """
    #     scale_fac = np.sqrt(total_mass / self.tab_mass) * np.sqrt(self.tab_R_peak / R_peak)
    #     vcirc = self.vcirc_interp(R / R_peak * self.tab_R_peak) * scale_fac
    #
    #     # scale_fac = np.sqrt(total_mass / table_mass) * np.sqrt(table_Rpeak / R_peak)
    #     # vcirc_interp = v_interp(Rarr / R_peak * table_Rpeak) * scale_fac
    #
    #     return vcirc


    def potential_gradient(self, R, R_peak, total_mass):
        """
        Calculate potential gradient for a inf thin massive gaussian Ring

        Parameters
        ----------
        R : float or array
            Radius or radii at which to calculate the potential gradient in kpc

        R_peak : float
            Peak of Gaussian ring in kpc

        total_mass : float
            Total mass of the Gaussian ring component

        Returns
        -------
        potl_grad : float or array
            Potential gradient at each given `R`

        Notes
        -----
        This function determines the potential gradient as a function of radius for
        a massive Gaussian ring component with a total mass, `total_mass`,
        and a ring peak radius to ring FWHM ratio, `invh`.
        This uses numerically calculated lookup tables.

        """
        if self.potl_grad_interp is not None:
            scale_fac = total_mass / self.tab_mass * (self.tab_R_peak / R_peak)**2
            potential_gradient_interp = self.potl_grad_interp(R / R_peak * self.tab_R_peak) * scale_fac

            return potential_gradient_interp
        else:
            return R*np.NaN

    def enclosed_mass(self, R, R_peak, total_mass):
        """
        Calculate enclosed mass for a inf thin massive gaussian Ring

        Parameters
        ----------
        R : float or array
            Radius or radii at which to calculate the enclosed in kpc

        R_peak : float
            Peak of Gaussian ring in kpc

        total_mass : float
            Total mass of the Gaussian ring component

        Returns
        -------
        vcirc : float or array
            Circular velocity at each given `R`

        Notes
        -----
        This function determines the enclosed as a function of radius for
        a massive Gaussian ring component with a total mass, `total_mass`,
        and a ring peak radius to ring FWHM ratio, `invh`.
        This uses numerically calculated lookup tables.

        """
        if self.menc_interp is not None:
            scale_fac = total_mass / self.tab_mass
            menc = self.menc_interp(R / R_peak * self.tab_R_peak) * scale_fac

            return menc
        else:
            return R*np.NaN


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


class ExpDisk(MassModel, _LightMassModel):
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
    mass_to_light = DysmalParameter(default=1, fixed=True)
    _subtype = 'baryonic'
    tracer = 'mass'

    def __init__(self, baryon_type='gas+stars', **kwargs):
        self.baryon_type = baryon_type
        super(ExpDisk, self).__init__(**kwargs)

    @staticmethod
    def evaluate(r, total_mass, r_eff, mass_to_light):
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
        #light = surf_dens_exp_disk(r, 1.0, self.rd)

        light = surf_dens_exp_disk(r, (1./self.mass_to_light) * 10**self.total_mass, self.rd)
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


class Sersic(MassModel, _LightMassModel):
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
    mass_to_light = DysmalParameter(default=1, fixed=True)

    _subtype = 'baryonic'
    tracer = 'mass'

    def __init__(self, invq=1.0, noord_flat=False, baryon_type='gas+stars', **kwargs):

        self.invq = invq
        self.baryon_type = baryon_type
        self._noord_flat = noord_flat
        super(Sersic, self).__init__(**kwargs)

        self._initialize_noord_flatteners()

    def __setstate__(self, state):
        super(Sersic, self).__setstate__(state)

        if 'baryon_type' in state.keys():
            pass
        else:
            self.baryon_type = 'gas+stars'


        if '_noord_flat' in state.keys():
            pass
        else:
            self._noord_flat = state['noord_flat']
            self._initialize_noord_flatteners()

    @property
    def noord_flat(self):
        return self._noord_flat

    @noord_flat.setter
    def noord_flat(self, value):
        if type(value) is not bool:
            raise ValueError("noord_flat must be True/False!")
        self._noord_flat = value
        self._initialize_noord_flatteners()

    def _initialize_noord_flatteners(self):
        if self.noord_flat:
            # Initialize NoordFlat object:
            self.noord_flattener = NoordFlat(n=self.n.value, invq=self.invq)

    def _update_noord_flatteners(self):
        if self.n.value != self.noord_flattener._n:
            self.noord_flattener.n = self.n.value

        if self.invq != self.noord_flattener._invq:
            self.noord_flattener.invq = self.invq

    @staticmethod
    def evaluate(r, total_mass, r_eff, n, mass_to_light):
        """
        Sersic mass surface density
        """

        return sersic_mr(r, 10**total_mass, n, r_eff)

    def enclosed_mass(self, r):
        """
        Sersic enclosed mass (linear)

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

            # Check menc, invq are right:
            self._update_noord_flatteners()

            # Correct flattened mass profile: 
            return self.noord_flattener.enclosed_mass(r, self.r_eff, 10**self.total_mass)

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
            # Check v, invq are right:
            self._update_noord_flatteners()
            vcirc = self.noord_flattener.circular_velocity(r, self.r_eff, 10**self.total_mass)
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
        #return sersic_mr(r, 1.0, self.n, self.r_eff)
        return sersic_mr(r, (1./self.mass_to_light) * 10**self.total_mass, self.n, self.r_eff)

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
                #rhogas = sersic_curve_rho(r, self.r_eff, 10**self.total_mass, self.n, self.invq)

                # Check v, invq are right:
                self._update_noord_flatteners()
                rhogas = self.noord_flattener.rho(r, self.r_eff, 10**self.total_mass)
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
                #dlnrhogas_dlnr_arr = sersic_curve_dlnrho_dlnr(r, self.r_eff, self.n, self.invq)

                # Check v, invq are right:
                self._update_noord_flatteners()
                dlnrhogas_dlnr_arr = self.noord_flattener.dlnrho_dlnr(r, self.r_eff)

            else:
                bn = scp_spec.gammaincinv(2. * self.n, 0.5)
                dlnrhogas_dlnr_arr = -2. * (bn / self.n) * np.power(r/self.r_eff, 1./self.n)
        else:
            dlnrhogas_dlnr_arr = r * 0.

        return dlnrhogas_dlnr_arr


class DiskBulge(MassModel, _LightMassModel):
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
    mass_to_light = DysmalParameter(default=1, fixed=True)

    _subtype = 'baryonic'
    tracer = 'mass'

    def __init__(self, invq_disk=5, invq_bulge=1, noord_flat=False,
                 light_component='disk', gas_component='disk', baryon_type='gas+stars',
                 **kwargs):

        self.invq_disk = invq_disk
        self.invq_bulge = invq_bulge
        self.light_component = light_component
        self.gas_component = gas_component
        self.baryon_type = baryon_type

        self._noord_flat = noord_flat

        super(DiskBulge, self).__init__(**kwargs)

        self._initialize_noord_flatteners()

    def __setstate__(self, state):
        state_mod = copy.deepcopy(state)
        if 'noord_flat' in state.keys():
            del state_mod['noord_flat']
            state_mod['_noord_flat'] = state['noord_flat']

        super(DiskBulge, self).__setstate__(state_mod)

        if 'baryon_type' in state_mod.keys():
            pass
        else:
            self.baryon_type = 'gas+stars'
            self.gas_component = 'disk'

        if 'noord_flat' in state.keys():
            self._initialize_noord_flatteners()    

    @property
    def noord_flat(self):
        return self._noord_flat

    @noord_flat.setter
    def noord_flat(self, value):
        if type(value) is not bool:
            raise ValueError("noord_flat must be True/False!")
        self._noord_flat = value

        self._initialize_noord_flatteners()

    def _initialize_noord_flatteners(self):
        # Initialize NoordFlat objects:
        self.noord_flattener_disk = NoordFlat(n=self.n_disk.value, invq=self.invq_disk)
        self.noord_flattener_bulge = NoordFlat(n=self.n_bulge.value, invq=self.invq_bulge)

    def _update_noord_flatteners(self):
        if self.n_disk.value != self.noord_flattener_disk._n:
            self.noord_flattener_disk.n = self.n_disk.value

        if self.n_bulge.value != self.noord_flattener_bulge._n:
            self.noord_flattener_bulge.n = self.n_bulge.value


        if self.invq_disk != self.noord_flattener_disk._invq:
            self.noord_flattener_disk.invq = self.invq_disk

        if self.invq_bulge != self.noord_flattener_bulge._invq:
            self.noord_flattener_bulge.invq = self.invq_bulge


    @staticmethod
    def evaluate(r, total_mass, r_eff_disk, n_disk, r_eff_bulge, n_bulge, bt, mass_to_light):
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
            # Check menc, invq are right:
            self._update_noord_flatteners()

            # Correct flattened mass profiles: 
            menc_bulge = self.noord_flattener_bulge.enclosed_mass(r, self.r_eff_bulge, mbulge_total)
            menc_disk  = self.noord_flattener_disk.enclosed_mass(r, self.r_eff_disk, mdisk_total)

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
            # Check menc, invq are right:
            self._update_noord_flatteners()

            # Correct flattened mass profiles: 
            menc_disk  = self.noord_flattener_disk.enclosed_mass(r, self.r_eff_disk, mdisk_total)

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
            # Check menc, invq are right:
            self._update_noord_flatteners()

            # Correct flattened mass profiles: 
            menc_bulge = self.noord_flattener_bulge.enclosed_mass(r, self.r_eff_bulge, mbulge_total)

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

            # Check v, invq are right:
            self._update_noord_flatteners()
            vcirc = self.noord_flattener_disk.circular_velocity(r, self.r_eff_disk, mdisk_total)

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

            # Check v, invq are right:
            self._update_noord_flatteners()
            vcirc = self.noord_flattener_bulge.circular_velocity(r, self.r_eff_bulge, mbulge_total)
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
        vcirc_sq = self.vcirc_sq(r)
        vrot_sq = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc_sq)
        vrot = np.sqrt(vrot_sq)

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
        vcirc_sq = self.circular_velocity_disk(r) ** 2
        vrot_sq = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc_sq)
        vrot = np.sqrt(vrot_sq)

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
        vcirc_sq = self.circular_velocity_bulge(r) ** 2
        vrot_sq = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc_sq)
        vrot = np.sqrt(vrot_sq)

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

            #flux = sersic_mr(r, 1.0, self.n_disk, self.r_eff_disk)
            flux = sersic_mr(r, (1./self.mass_to_light) * (1.0-self.bt) * 10**self.total_mass,
                             self.n_disk, self.r_eff_disk)

        elif self.light_component == 'bulge':

            #flux = sersic_mr(r, 1.0, self.n_bulge, self.r_eff_bulge)
            flux = sersic_mr(r, (1./self.mass_to_light) * self.bt * 10**self.total_mass,
                             self.n_bulge, self.r_eff_bulge)

        elif self.light_component == 'total':

            # flux_disk = sersic_mr(r, 1.0-self.bt,
            #                       self.n_disk, self.r_eff_disk)
            # flux_bulge = sersic_mr(r, self.bt,
            #                        self.n_bulge, self.r_eff_bulge)

            flux_disk = sersic_mr(r,
                        (1./self.mass_to_light) * (1.0-self.bt) * 10**self.total_mass,
                         self.n_disk, self.r_eff_disk)
            flux_bulge = sersic_mr(r,
                        (1./self.mass_to_light) * self.bt * 10**self.total_mass,
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
                    # rhogas = sersic_curve_rho(r, self.r_eff_disk, mdisk_total,
                    #                           self.n_disk, self.invq_disk)

                    # Check v, invq are right:
                    self._update_noord_flatteners()
                    rhogas = self.noord_flattener_disk.rho(r, self.r_eff_disk, mdisk_total)
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
                    # rhogas = sersic_curve_rho(r, self.r_eff_bulge, mbulge_total,
                    #                           self.n_bulge, self.invq_bulge)

                    # Check v, invq are right:
                    self._update_noord_flatteners()
                    rhogas = self.noord_flattener_bulge.rho(r, self.r_eff_bulge, mbulge_total)
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
            #dlnrhogas_dlnr_arr = sersic_curve_dlnrho_dlnr(r, self.r_eff_disk, self.n_disk, self.invq_disk)

            # Check v, invq are right:
            self._update_noord_flatteners()
            dlnrhogas_dlnr_arr = self.noord_flattener_disk.dlnrho_dlnr(r, self.r_eff_disk)
            return dlnrhogas_dlnr_arr
        else:
            bn = scp_spec.gammaincinv(2. * self.n_disk, 0.5)
            return -2. * (bn / self.n_disk) * np.power(r/self.r_eff_disk, 1./self.n_disk)

    def dlnrhogas_dlnr_bulge(self, r):
        if self.noord_flat:
            #dlnrhogas_dlnr_arr = sersic_curve_dlnrho_dlnr(r, self.r_eff_bulge, self.n_bulge, self.invq_bulge)

            # Check v, invq are right:
            self._update_noord_flatteners()
            dlnrhogas_dlnr_arr = self.noord_flattener_bulge.dlnrho_dlnr(r, self.r_eff_bulge)

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


class LinearDiskBulge(MassModel, _LightMassModel):
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
    mass_to_light = DysmalParameter(default=1, fixed=True)

    _subtype = 'baryonic'
    tracer = 'mass'

    def __init__(self, invq_disk=5, invq_bulge=1, noord_flat=False,
                 light_component='disk', baryon_type='gas+stars', **kwargs):

        self.invq_disk = invq_disk
        self.invq_bulge = invq_bulge
        self.light_component = light_component
        self.baryon_type = baryon_type

        self._noord_flat = noord_flat

        super(LinearDiskBulge, self).__init__(**kwargs)


        self._initialize_noord_flatteners()

    def __setstate__(self, state):
        super(LinearDiskBulge, self).__setstate__(state)

        if 'baryon_type' in state.keys():
            pass
        else:
            self.baryon_type = 'gas+stars'

        if '_noord_flat' in state.keys():
            pass
        else:
            self._noord_flat = state['noord_flat']
            self._initialize_noord_flatteners()

    @property
    def noord_flat(self):
        return self._noord_flat

    @noord_flat.setter
    def noord_flat(self, value):
        if type(value) is not bool:
            raise ValueError("noord_flat must be True/False!")
        self._noord_flat = value

        self._initialize_noord_flatteners()

    def _initialize_noord_flatteners(self):
        # Initialize NoordFlat objects:
        self.noord_flattener_disk = NoordFlat(n=self.n_disk.value, invq=self.invq_disk)
        self.noord_flattener_bulge = NoordFlat(n=self.n_bulge.value, invq=self.invq_bulge)


    def _update_noord_flatteners(self):
        if self.n_disk.value != self.noord_flattener_disk._n:
            self.noord_flattener_disk.n = self.n_disk.value

        if self.n_bulge.value != self.noord_flattener_bulge._n:
            self.noord_flattener_bulge.n = self.n_bulge.value

        if self.invq_disk != self.noord_flattener_disk._invq:
            self.noord_flattener_disk.invq = self.invq_disk

        if self.invq_bulge != self.noord_flattener_bulge._invq:
            self.noord_flattener_bulge.invq = self.invq_bulge

    @staticmethod
    def evaluate(r, total_mass, r_eff_disk, n_disk, r_eff_bulge, n_bulge, bt, mass_to_light):
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
            # Check menc, invq are right:
            self._update_noord_flatteners()

            # Correct flattened mass profiles: 
            menc_bulge = self.noord_flattener_bulge.enclosed_mass(r, self.r_eff_bulge, mbulge_total)
            menc_disk  = self.noord_flattener_disk.enclosed_mass(r, self.r_eff_disk, mdisk_total)

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
            # Check menc, invq are right:
            self._update_noord_flatteners()

            # Correct flattened mass profiles: 
            menc_disk  = self.noord_flattener_disk.enclosed_mass(r, self.r_eff_disk, mdisk_total)

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
            # Check menc, invq are right:
            self._update_noord_flatteners()

            # Correct flattened mass profiles: 
            menc_bulge = self.noord_flattener_bulge.enclosed_mass(r, self.r_eff_bulge, mbulge_total)
            
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

            # Check v, invq are right:
            self._update_noord_flatteners()
            vcirc = self.noord_flattener_disk.circular_velocity(r, self.r_eff_disk, mdisk_total)
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

            # Check v, invq are right:
            self._update_noord_flatteners()
            vcirc = self.noord_flattener_bulge.circular_velocity(r, self.r_eff_bulge, mbulge_total)
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
        vcirc_sq = self.vcirc_sq(r)
        vrot_sq = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc_sq)
        vrot = np.sqrt(vrot_sq)

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
        vcirc_sq = self.circular_velocity_disk(r) ** 2
        vrot_sq = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc_sq)
        vrot = np.sqrt(vrot_sq)
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
        vcirc_sq = self.circular_velocity_bulge(r) ** 2
        vrot_sq = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc_sq)
        vrot = np.sqrt(vrot_sq)

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

            #flux = sersic_mr(r, 1.0, self.n_disk, self.r_eff_disk)
            flux = sersic_mr(r, (1./self.mass_to_light) * (1.0-self.bt) * self.total_mass,
                             self.n_disk, self.r_eff_disk)

        elif self.light_component == 'bulge':

            #flux = sersic_mr(r, 1.0, self.n_bulge, self.r_eff_bulge)
            flux = sersic_mr(r, (1./self.mass_to_light) * self.bt * self.total_mass,
                             self.n_bulge, self.r_eff_bulge)

        elif self.light_component == 'total':

            # flux_disk = sersic_mr(r, 1.0-self.bt,
            #                       self.n_disk, self.r_eff_disk)
            # flux_bulge = sersic_mr(r, self.bt,
            #                        self.n_bulge, self.r_eff_bulge)

            flux_disk = sersic_mr(r,
                        (1./self.mass_to_light) * (1.0-self.bt) * self.total_mass,
                         self.n_disk, self.r_eff_disk)
            flux_bulge = sersic_mr(r,
                        (1./self.mass_to_light) * self.bt * self.total_mass,
                         self.n_bulge, self.r_eff_bulge)
            flux = flux_disk + flux_bulge



        else:

            raise ValueError("light_component can only be 'disk', 'bulge', "
                             "or 'total.'")

        return flux


class GaussianRing(MassModel, _LightMassModel):
    r"""
    Mass distribution following an infinitely thin Gaussian ring profile.

    Parameters
    ----------

    total_mass : float
        Log10 of the total mass in solar units

    R_peak : float
        Peak of gaussian (radius) in kpc

    FWHM: float
        FWHM of gaussian, in kpc

    baryon_type : string
        What type of baryons are included. Used for dlnrhogas/dlnr.
        Options: {'gas+stars', 'stars', 'gas'}
        

    Notes
    -----
    Model formula:

    .. math::
        M(r)&=M_0\exp\left(\frac{(r-r_{\rm peak})^2}{2\sigma_R^2}\right)

        \sigma_R &= \mathrm{FWHM}/(2\sqrt{2\ln 2})

    """

    total_mass = DysmalParameter(default=1, bounds=(5, 14))
    R_peak = DysmalParameter(default=1, bounds=(0, 50))
    FWHM = DysmalParameter(default=1, bounds=(0, 50))
    mass_to_light = DysmalParameter(default=1, fixed=True)

    _subtype = 'baryonic'
    tracer = 'mass'

    def __init__(self, baryon_type='gas+stars', **kwargs):
        self.baryon_type = baryon_type

        super(GaussianRing, self).__init__(**kwargs)

        self._initialize_ring_table()

    def __setstate__(self, state):
        super(GaussianRing, self).__setstate__(state)

        if 'baryon_type' in state.keys():
            pass
        else:
            self.baryon_type = 'gas+stars'

        if 'ring_table' in state.keys():
            pass
        else:
            self._initialize_ring_table()

    def _initialize_ring_table(self):
        # Initialize InfThinMassiveGaussianRing object:
        self.ring_table = InfThinMassiveGaussianRing(invh=self.ring_invh())

    def _update_ring_table(self):
        if self.ring_invh() != self.ring_table._invh:
            self.ring_table.invh = self.ring_invh()

    def sigma_R(self):
        return self.FWHM.value / (2.*np.sqrt(2.*np.log(2.)))

    def ring_invh(self):
        return self.R_peak.value / self.FWHM.value

    def ring_reff(self):
        # Find the effective radius explicitly by definition, using erf function
        # and in the dimensionless parameter x = r/Rpeak; with subbing u = 4*ln(2) h^2 (x-1)^2
        # Solving for:
        #     int_0^xeff t * exp{-4*ln(2)*invh**2*(t-1)**2) = 0.5 * int_0^inf t * exp{-4*ln(2)*invh**2*(t-1)**2)
        try:
            A = 4*np.log(2)
            func = lambda ueff: scp_spec.erf(np.sqrt(ueff)) - (np.pi * A * self.ring_invh() ** 2) ** -0.5 * np.exp(-ueff) - \
                                   0.5 * (1 - scp_spec.erf(np.sqrt(A * self.ring_invh() ** 2)) - (np.pi * A * self.ring_invh() ** 2) ** -0.5 * np.exp(-A * self.ring_invh() ** 2))
            u_eff = scp_opt.brentq(lambda t: func(t), a=0., b=1.)
            x_eff = 1 + np.sqrt(u_eff / (A * self.ring_invh() ** 2))
            return x_eff * self.R_peak.value
        except:
            logger.warning("Could not find ring_reff. Assuming reff=Rpeak instead...")
            return self.R_peak.value

    def r_eff(self):
        return self.ring_reff()

    @staticmethod
    def evaluate(r, total_mass, R_peak, FWHM, mass_to_light):
        """ Gaussian ring mass surface density """
        sigma_R = FWHM/ (2.*np.sqrt(2.*np.log(2.)))
        I0 = _I0_gaussring(R_peak, sigma_R, 10**total_mass)
        return I0*np.exp(-(r-R_peak)**2/(2.*sigma_R**2))

    def surface_density(self, r):
        """ Gaussian ring mass surface density """
        return self.evaluate(r, self.total_mass.value,
                             self.R_peak.value, self.FWHM.value, 1.)

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
        # Check invh is correct
        self._update_ring_table()

        return self.ring_table.enclosed_mass(r, self.R_peak.value, 10**self.total_mass.value)

    def projected_enclosed_mass(self, r):
        """ Same as enclosed mass as this is infinitely thin gaussian ring """
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

        return np.sqrt(self.vcirc_sq(r))

    def vcirc_sq(self, r):
        """
        Square of circular velocity as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        vcirc_sq : float or array
            Square of circular velocity in km^2/s^2

        Notes
        -----
        Calculated as :math:`v_{\mathrm{circ}}^2(R) = R * \partial \Phi / \partial R`
        from the gradient of the potential, as the potential gradient has negative values.
        """
        return r * self.potential_gradient(r)


    def potential_gradient(self, r):
        r"""
        Method to evaluate the gradient of the potential, :math:`\Delta\Phi(r)/\Delta r`.

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate circular velocity in kpc

        Returns
        -------
        dPhidr : float or array
            Gradient of the potential at `r`

        """
        # Check invh is correct
        self._update_ring_table()

        return self.ring_table.potential_gradient(r, self.R_peak.value, 10**self.total_mass.value)

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
        #return 1.*np.exp(-(r-self.R_peak.value)**2/(2.*self.sigma_R()**2))
        return self.surface_density(r) * (1./self.mass_to_light)

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
            # Inf thin: rho(R,z=0) = Sigma(R)
            rhogas = self.surface_density(r)
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
            # Inf thin: rho(R,z=0) = Sigma(R)
            dlnrhogas_dlnr_arr = - r * (r-self.R_peak.value) / (self.sigma_R()**2)
        else:
            dlnrhogas_dlnr_arr = r * 0.

        return dlnrhogas_dlnr_arr
