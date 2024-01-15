# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Halo mass models for DysmalPy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import abc
import logging
import copy

# Third party imports
import numpy as np
import scipy.special as scp_spec
import scipy.optimize as scp_opt
import astropy.constants as apy_con
import astropy.units as u
import astropy.cosmology as apy_cosmo


# Local imports
from .model_set import ModelSet
from .base import MassModel
from dysmalpy.parameters import DysmalParameter

__all__ = ['NFW', 'TwoPowerHalo', 'Burkert', 'Einasto', 'DekelZhao', 'LinearNFW']

# DEFAULT COSMOLOGY
_default_cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)


# CONSTANTS
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc

g_pc_per_Msun_kmssq = G.to(u.pc / u.Msun * (u.km / u.s) ** 2).value

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

_dict_lmvir_fac_test_z = {'zarr':    np.arange(0.5, 2.75, 0.25),
                          'fdmarr': np.array([1.e-3, 1.e-2, 0.1, 0.25, 0.5, 0.75, 0.9, 1-1.e-2, 1-1.e-3]),
                          'facarr': np.array([[ 1.87027726e-01,  6.83525784e-01,  1.70740388e+00,
                                                 2.43045482e+00,  3.27806760e+00,  4.19108979e+00,
                                                 5.13078742e+00,  7.20681515e+00,  9.21406239e+00],
                                               [ 1.42061858e-01,  6.00007080e-01,  1.53906815e+00,
                                                 2.21927776e+00,  3.03914007e+00,  3.93936616e+00,
                                                 4.87420697e+00,  6.94791177e+00,  8.95494591e+00],
                                               [ 1.00551748e-01,  5.24281952e-01,  1.38498363e+00,
                                                 2.02073465e+00,  2.80851657e+00,  3.69274241e+00,
                                                 4.62126978e+00,  6.69190395e+00,  8.69865476e+00],
                                               [ 6.31519829e-02,  4.57268141e-01,  1.24826995e+00,
                                                 1.84043178e+00,  2.59324412e+00,  3.45844497e+00,
                                                 4.37910433e+00,  6.44582962e+00,  8.45221694e+00],
                                               [ 2.97794509e-02,  3.98491637e-01,  1.12869340e+00,
                                                 1.67974956e+00,  2.39616708e+00,  3.23967562e+00,
                                                 4.15086381e+00,  6.21275987e+00,  8.21869377e+00],
                                               [ 5.36103016e-05,  3.46978470e-01,  1.02457472e+00,
                                                 1.53784436e+00,  2.21772071e+00,  3.03735532e+00,
                                                 3.93747250e+00,  5.99354714e+00,  7.99892789e+00],
                                               [-2.64841174e-02,  3.01676546e-01,  9.33823097e-01,
                                                 1.41290431e+00,  2.05708386e+00,  2.85121748e+00,
                                                 3.73871828e+00,  5.78791915e+00,  7.79263749e+00],
                                               [-5.02759808e-02,  2.61622377e-01,  8.54407920e-01,
                                                 1.30283654e+00,  1.91285041e+00,  2.68042250e+00,
                                                 3.55386264e+00,  5.59509028e+00,  7.59902746e+00],
                                               [-7.17141754e-02,  2.25990008e-01,  7.84535250e-01,
                                                 1.20560418e+00,  1.78340279e+00,  2.52388554e+00,
                                                 3.38196112e+00,  5.41408290e+00,  7.41711095e+00]])}


class DarkMatterHalo(MassModel):
    r"""
    Base model for dark matter halos

    Parameters
    ----------
    mvirial : float
        Virial mass

    conc : float
        Concentration parameter

    fdm : float
        Dark matter fraction

    """
    # Standard parameters for a dark matter halo profile
    mvirial = DysmalParameter(default=1.0, bounds=(5, 20))
    fdm = DysmalParameter(default=-99.9, fixed=True, bounds=(0,1))
    _subtype = 'dark_matter'

    def __init__(self, z=0, cosmo=_default_cosmo, **kwargs):
        self._z = z
        self._cosmo = cosmo
        self._set_hz()
        super(DarkMatterHalo, self).__init__(**kwargs)

    def __setstate__(self, state):
        super(DarkMatterHalo, self).__setstate__(state)

        # quick test if necessary to migrate:
        if '_z' in state.keys():
            pass
        else:
            migrate_keys = ['z', 'cosmo']
            for mkey in migrate_keys:
                if (mkey in state.keys()) and ('_{}'.format(mkey) not in state.keys()):
                    self.__dict__['_{}'.format(mkey)] = state[mkey]
                    del self.__dict__[mkey]

        if '_hz' not in state.keys():
            self._set_hz()

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        if value < 0:
            raise ValueError("Redshift can't be negative!")
        self._z = value

        # Reset hz:
        self._set_hz()

    @property
    def cosmo(self):
        return self._cosmo

    @cosmo.setter
    def cosmo(self, new_cosmo):
        if not isinstance(new_cosmo, apy_cosmo.FLRW):
            raise TypeError("Cosmology must be an astropy.cosmology.FLRW "
                            "instance.")
        if new_cosmo is None:
            self._cosmo = _default_cosmo
        self._cosmo = new_cosmo

        # Reset hz:
        self._set_hz()

    def _set_hz(self):
        self._hz = self.cosmo.H(self.z).value

    def calc_rvir(self):
        r"""
        Calculate the virial radius based on virial mass and redshift

        Returns
        -------
        rvir : float
            Virial radius

        Notes
        -----
        Formula:

        .. math::

            M_{\rm vir} = 100 \frac{H(z)^2 R_{\rm vir}^3}{G}

        This is based on Mo, Mao, & White (1998) [1]_ which defines the virial
        radius as the radius where the mean mass density is :math:`200\rho_{\rm crit}`.
        :math:`\rho_{\rm crit}` is the critical density for closure at redshift, :math:`z`.
        """
        # hz = self.cosmo.H(self.z).value
        # rvir = ((10 ** self.mvirial * (g_pc_per_Msun_kmssq * 1e-3) /
        #         (10 * hz * 1e-3) ** 2) ** (1. / 3.))
        rvir = ((10 ** self.mvirial * (g_pc_per_Msun_kmssq * 1e-3) /
                (10 * self._hz * 1e-3) ** 2) ** (1. / 3.))

        return rvir

    @abc.abstractmethod
    def calc_rho0(self, *args, **kwargs):
        """
        Method to calculate the scale density
        """

    def velocity_profile(self, r, model):
        """
        Calculate velocity profile, including any adiabatic contraction
        """

        if model.kinematic_options.adiabatic_contract:
            raise NotImplementedError("Adiabatic contraction not currently supported!")
        else:
            return self.circular_velocity(r)



    def calc_mvirial_from_fdm(self, baryons, r_fdm, adiabatic_contract=False):
        """
        Calculate virial mass given dark matter fraction and baryonic distribution

        Parameters
        ----------
        baryons : `~dysmalpy.models.MassModel` or dictionary
            Model component representing the baryons (assumed to be light emitting),
            or dictionary containing a list of the baryon components (baryons['components'])
            and a list of whether the baryon components are light emitting or not (baryons['light'])

        r_fdm : float
            Radius at which the dark matter fraction is determined

        Returns
        -------
        mvirial : float
            Virial mass in logarithmic solar units

        Notes
        -----
        This uses the current value of `fdm` together with
        the input baryon distribution to calculate the inferred `mvirial`.
        """
        if (self.fdm.value > self.bounds['fdm'][1]) | \
                ((self.fdm.value < self.bounds['fdm'][0])):
            mvirial = np.NaN
        elif (self.fdm.value == 1.):
            mvirial = np.inf
        elif (self.fdm.value == 0.):
            mvirial = -np.inf #-5.  # as a small but finite value
        elif (self.fdm.value < 1.e-10):
            mvirial = -np.inf
        elif (r_fdm < 0.):
            mvirial = np.NaN
        else:
            if isinstance(baryons, dict):
                vsqr_bar_re = 0
                bar_mtot_linear = 0
                for bcmp in baryons['components']:
                    vsqr_bar_re += bcmp.vcirc_sq(r_fdm)
                    bar_mtot_linear += 10**bcmp.total_mass.value
                bar_mtot = np.log10(bar_mtot_linear)
            else:
                vsqr_bar_re = baryons.vcirc_sq(r_fdm)
                bar_mtot = baryons.total_mass.value

            vsqr_dm_re_target = vsqr_bar_re / (1./self.fdm.value - 1)

            if not np.isfinite(vsqr_dm_re_target):
                mvirial = np.NaN
            else:
                #mtest = np.arange(-5, 50, 1.0)
                short_mtest = False
                try:
                    if ((bar_mtot >= 8.) & (bar_mtot <=13.)):
                        whminz = np.argmin(np.abs(_dict_lmvir_fac_test_z['zarr']-self.z))
                        whminfdm = np.argmin(np.abs(_dict_lmvir_fac_test_z['fdmarr']-self.fdm.value))

                        fac_lmvir = _dict_lmvir_fac_test_z['facarr'][whminz,whminfdm]
                        rough_mvir = fac_lmvir - np.log10(1./self.fdm.value-1)+np.log10(0.5)+bar_mtot

                        mtest = np.arange(rough_mvir-1., rough_mvir+1.5, 0.5)
                        mtest = np.append(-5., mtest)
                        mtest = np.append(mtest, 50.)
                        short_mtest = True
                    else:
                        mtest = np.arange(-5, 50, 1.0)
                except:
                    mtest = np.arange(-5, 50, 1.0)

                vtest = np.array([self._minfunc_vdm_mvir_from_fdm(m, vsqr_dm_re_target,
                        r_fdm, baryons, adiabatic_contract) for m in mtest])
                try:
                    a = mtest[vtest < 0][-1]
                    b = mtest[vtest > 0][0]
                    # # TEST
                    # if adiabatic_contract:
                    #     a_noAC = mtest[vtest_noAC < 0][-1]
                    #     b_noAC = mtest[vtest_noAC > 0][0]
                except:
                    print("adiabatic_contract={}".format(adiabatic_contract))
                    print("fdm={}".format(self.fdm.value))
                    print("r_fdm={}".format(r_fdm))
                    print(mtest, vtest)
                    raise ValueError

                # ------------------------------------------------------------------
                # Do a quick check to make sure it was inside the inner range:
                # If not, redo calc with better range:
                if short_mtest & ((a == mtest[0]) | (b == mtest[-1])):
                    mtest_orig = mtest[:]
                    if (a == mtest[0]):
                        mtest = np.arange(np.ceil(mtest_orig[1])-2., np.ceil(mtest_orig[1])+1., 1.0)
                        mtest = np.append(np.arange(np.ceil(mtest_orig[1])-22., np.ceil(mtest_orig[1])-2., 2.0), mtest)
                        mtest = np.append(-5., mtest)
                    elif (b == mtest[-1]):
                        mtest = np.arange(np.floor(mtest_orig[-2]), np.floor(mtest_orig[-2])+3., 1.0)
                        mtest = np.append(mtest, np.arange(np.floor(mtest_orig[-2])+3., np.floor(mtest_orig[-2])+23., 2.0))
                        mtest = np.append(mtest, 50.)

                    vtest = np.array([self._minfunc_vdm_mvir_from_fdm(m, vsqr_dm_re_target, r_fdm,
                                        baryons, adiabatic_contract) for m in mtest])
                    try:
                        a = mtest[vtest < 0][-1]
                        b = mtest[vtest > 0][0]
                    except:
                        print("adiabatic_contract={}".format(adiabatic_contract))
                        print("fdm={}".format(self.fdm.value))
                        print("r_fdm={}".format(r_fdm))
                        print(mtest, vtest)
                        raise ValueError

                # ------------------------------------------------------------------
                # Run optimizer:
                mvirial = scp_opt.brentq(self._minfunc_vdm_mvir_from_fdm, a, b, args=(vsqr_dm_re_target, r_fdm,
                                        baryons, adiabatic_contract))
        return mvirial


    def _minfunc_vdm_mvir_from_fdm(self, mvirial, vsqtarget, r_fdm, bary, adiabatic_contract):

        halotmp = self.copy()
        halotmp.__setattr__('mvirial', mvirial)

        if adiabatic_contract:
            modtmp = ModelSet()
            if isinstance(bary, dict):
                for bcmp,b_light in zip(bary['components'], bary['light']):
                    modtmp.add_component(bcmp, light=b_light)
            else:
                modtmp.add_component(bary, light=True)
            modtmp.add_component(halotmp)
            modtmp.kinematic_options.adiabatic_contract = True
            modtmp.kinematic_options.adiabatic_contract_modify_small_values = True
            modtmp._update_tied_parameters()

            vc_sq, vc_sq_dm = modtmp.vcirc_sq(r_fdm, compute_dm=True)
            return vc_sq_dm - vsqtarget
        else:
            return halotmp.vcirc_sq(r_fdm)- vsqtarget



class NFW(DarkMatterHalo):
    r"""
    Dark matter halo following an NFW profile

    Parameters
    ----------
    mvirial : float
        Virial mass in logarithmic solar units

    conc : float
        Concentration parameter

    fdm : float
        Dark matter fraction

    z : float
        Redshift

    cosmo : `~astropy.cosmology` object
        The cosmology to use for modelling.
        If this model component will be attached to a `~dysmalpy.galaxy.Galaxy` make sure
        the respective cosmologies are the same. Default is
        `~astropy.cosmology.FlatLambdaCDM` with H0=70., and Om0=0.3.

    Notes
    -----
    Model formula:

    The mass density follows Navarro, Frenk, & White (1995) [1]_:

    .. math::

        \rho = \frac{\rho_0}{(r/r_s)(1 + r/r_s)^2}

    :math:`r_s` is the scale radius defined as :math:`r_{\rm vir}/c`.
    :math:`\rho_0` is the normalization parameter.

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/1995MNRAS.275..720N/abstract
    """
    mvirial = DysmalParameter(default=1.0, bounds=(5, 20))
    conc = DysmalParameter(default=5.0, bounds=(2, 20))
    fdm = DysmalParameter(default=-99.9, fixed=True, bounds=(0,1))

    def __init__(self, z=0, cosmo=_default_cosmo, **kwargs):
        super(NFW, self).__init__(z=z, cosmo=cosmo, **kwargs)

    def evaluate(self, r, mvirial, conc, fdm):
        """Mass density as a function of radius"""

        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0(rvirial=rvirial)
        rs = rvirial / self.conc

        return rho0 / (r / rs * (1 + r / rs) ** 2)

    def enclosed_mass(self, r):
        """
        Enclosed mass as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        menc : float or array
            Enclosed mass in solar units
        """

        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0(rvirial=rvirial)
        rs = rvirial/self.conc
        aa = 4.*np.pi*rho0*rvirial**3/self.conc**3

        # For very small r, bb can be negative.
        bb = np.abs(np.log((rs + r)/rs) - r/(rs + r))

        return aa*bb

    def calc_rho0(self, rvirial=None):
        r"""
        Normalization of the density distribution

        Returns
        -------
        rho0 : float
            Mass density normalization in :math:`M_{\odot}/\rm{kpc}^3`
        """
        if rvirial is None:
            rvirial = self.calc_rvir()
        aa = 10**self.mvirial/(4.*np.pi*rvirial**3)*self.conc**3
        bb = 1./(np.log(1.+self.conc) - (self.conc/(1.+self.conc)))

        return aa * bb




class TwoPowerHalo(DarkMatterHalo):
    r"""
    Two power law density model for a dark matter halo

    Parameters
    ----------
    mvirial : float
        Virial mass in logarithmic solar units

    conc : float
        Concentration parameter

    alpha : float
        Power law index at small radii

    beta : float
        Power law index at large radii

    fdm : float
        Dark matter fraction

    z : float
        Redshift

    cosmo : `~astropy.cosmology` object
        The cosmology to use for modelling.
        If this model component will be attached to a `~dysmalpy.galaxy.Galaxy` make sure
        the respective cosmologies are the same. Default is
        `~astropy.cosmology.FlatLambdaCDM` with H0=70., and Om0=0.3.

    Notes
    -----
    Model formula:

    The mass density follows Equation 2.64 of Binney & Tremaine (2008) [1]_:

    .. math::

        \rho=\frac{\rho_0}{(r/r_s)^\alpha(1 + r/r_s)^{\beta - \alpha}}

    :math:`r_s` is the scale radius and defined as :math:`r_\mathrm{vir}/c` where
    :math:`r_\mathrm{vir}` is the virial radius and :math:`c` is the concentration
    parameter. :math:`\rho_0` is the normalization parameter.

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/2008gady.book.....B/abstract
    """

    # Powerlaw slopes for the density model
    mvirial = DysmalParameter(default=1.0, bounds=(5, 20))
    conc = DysmalParameter(default=5.0, bounds=(2, 20))
    alpha = DysmalParameter(default=1.0)
    beta = DysmalParameter(default=3.0)
    fdm = DysmalParameter(default=-99.9, fixed=True, bounds=(0,1))

    _subtype = 'dark_matter'

    def __init__(self, z=0, cosmo=_default_cosmo, **kwargs):
        super(TwoPowerHalo, self).__init__(z=z, cosmo=cosmo, **kwargs)

    def evaluate(self, r, mvirial, conc, alpha, beta, fdm):
        """ Mass density for the TwoPowerHalo"""

        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0()
        rs = rvirial / self.conc

        return rho0 / ((r/rs)**alpha * (1 + r/rs)**(beta - alpha))

    def enclosed_mass(self, r):
        """
        Enclosed mass as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        menc : float or array
            Enclosed mass in solar units
        """

        rvirial = self.calc_rvir()
        rs = rvirial/self.conc
        aa = 10**self.mvirial*(r/rvirial)**(3 - self.alpha)
        bb = (scp_spec.hyp2f1(3-self.alpha, self.beta-self.alpha, 4-self.alpha, -r/rs) /
              scp_spec.hyp2f1(3 - self.alpha, self.beta - self.alpha, 4 - self.alpha, -self.conc))

        return aa*bb

    def calc_rho0(self, rvirial=None):
        r"""
        Normalization of the density distribution

        Returns
        -------
        rho0 : float
            Mass density normalization in :math:`M_{\odot}/\rm{kpc}^3`
        """
        if rvirial is None:
            rvirial = self.calc_rvir()

        rs = rvirial/self.conc
        aa = -10**self.mvirial/(4*np.pi*self.conc**(3-self.alpha)*rs**3)
        bb = (self.alpha - 3) / scp_spec.hyp2f1(3-self.alpha, self.beta-self.alpha, 4-self.alpha, -self.conc)

        return aa*bb



    def calc_alpha_from_fdm(self, baryons, r_fdm):
        """
        Calculate alpha given dark matter fraction and baryonic distribution

        Parameters
        ----------
        baryons : `~dysmalpy.models.MassModel` or dictionary
            Model component representing the baryons (assumed to be light emitting),
            or dictionary containing a list of the baryon components (baryons['components'])
            and a list of whether the baryon components are light emitting or not (baryons['light'])

        r_fdm : float
            Radius at which the dark matter fraction is determined

        Returns
        -------
        alpha : float
            alpha value

        Notes
        -----
        This uses the current values of `fdm`, `mvirial`, and `beta` together with
        the input baryon distribution to calculate the necessary value of `alpha`.
        """
        if (self.fdm.value > self.bounds['fdm'][1]) | \
                ((self.fdm.value < self.bounds['fdm'][0])):
            alpha = np.NaN
        else:
            if isinstance(baryons, dict):
                vsqr_bar_re = 0
                for bcmp in baryons['components']:
                    vsqr_bar_re += bcmp.vcirc_sq(r_fdm)
            else:
                vsqr_bar_re = baryons.vcirc_sq(r_fdm)

            vsqr_dm_re_target = vsqr_bar_re / (1./self.fdm - 1)

            # alphtest = np.arange(-50, 50, 1.)
            alphtest = np.arange(0., 5., 1.0)
            alphtest = np.append(-5., alphtest)
            alphtest = np.append(alphtest, 50.)

            vtest = np.array([self._minfunc_vdm_alpha_from_fdm(alph, vsqr_dm_re_target, self.mvirial, self.conc,
                                    self.beta, self.z, r_fdm) for alph in alphtest])

            try:
                a = alphtest[vtest < 0][-1]
                try:
                    b = alphtest[vtest > 0][0]
                except:
                    a = alphtest[-2] # Even if not perfect, force in case of no convergence...
                    b = alphtest[-1]
            except:
                a = alphtest[0]    # Even if not perfect, force in case of no convergence...
                b = alphtest[1]

            alpha = scp_opt.brentq(self._minfunc_vdm_alpha_from_fdm, a, b, args=(vsqr_dm_re_target, self.mvirial, self.conc,
                                        self.beta, self.z, r_fdm))

        return alpha

    def _minfunc_vdm_alpha_from_fdm(self, alpha, vsqtarget, mass, conc, beta, z, r_fdm):
        halo = TwoPowerHalo(mvirial=mass, conc=conc, alpha=alpha, beta=beta, z=z)
        return halo.vcirc_sq(r_fdm)- vsqtarget



class Burkert(DarkMatterHalo):
    r"""
    Dark matter halo following a Burkert profile

    Parameters
    ----------
    mvirial : float
        Virial mass in logarithmic solar units

    rB : float
        Size of the dark matter core in kpc

    fdm : float
        Dark matter fraction

    z : float
        Redshift

    cosmo : `~astropy.cosmology` object
        The cosmology to use for modelling.
        If this model component will be attached to a `~dysmalpy.galaxy.Galaxy` make sure
        the respective cosmologies are the same. Default is
        `~astropy.cosmology.FlatLambdaCDM` with H0=70., and Om0=0.3.

    Notes
    -----
    Model formula:

    The mass density follows Burkert (1995) [1]_:

    .. math::

        \rho=\frac{\rho_0}{(1 + r/r_B)(1 + (r/r_B)^2)}

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/1995ApJ...447L..25B/abstract
    """

    mvirial = DysmalParameter(default=1.0, bounds=(5, 20))
    rB = DysmalParameter(default=1.0)
    fdm = DysmalParameter(default=-99.9, fixed=True, bounds=(0,1))

    _subtype = 'dark_matter'

    def __init__(self, z=0, cosmo=_default_cosmo, **kwargs):
        super(Burkert, self).__init__(z=z, cosmo=cosmo, **kwargs)

    def evaluate(self, r, mvirial, rB, fdm):
        """Mass density as a function of radius"""

        rho0 = self.calc_rho0()

        return rho0 / ((1 + r/rB) * (1 + (r/rB)**2))

    def I(self, r):
        Ival = 0.25 * (np.log(r**2 + self.rB**2) + 2.*np.log(r + self.rB)
                       - 2.*np.arctan(r/self.rB) - 4.*np.log(self.rB))
        return Ival

    def enclosed_mass(self, r):
        """
        Enclosed mass as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        menc : float or array
            Enclosed mass in solar units
        """
        rvir = self.calc_rvir()
        Irvir = self.I(rvir)

        aa = 10**self.mvirial / Irvir
        bb = self.I(r)
        return aa*bb

    def calc_rho0(self):
        r"""
        Normalization of the density distribution

        Returns
        -------
        rho0 : float
            Mass density normalization in :math:`M_{\odot}/\rm{kpc}^3`
        """
        rvir = self.calc_rvir()
        Irvir = self.I(rvir)

        aa = 10**self.mvirial / (4*np.pi* self.rB**3)
        bb = 1./Irvir

        return aa*bb

    def calc_conc(self):
        """
        Calculate the concentration parameter

        Returns
        -------
        conc : float
            Concentration based on the core radius, `rB`.
        """
        rvir = self.calc_rvir()
        conc = rvir/self.rB
        self.conc = conc
        return conc

    def calc_rB_from_fdm(self, baryons, r_fdm):
        """
        Calculate core radius given dark matter fraction and baryonic distribution

        Parameters
        ----------
        baryons : `~dysmalpy.models.MassModel` or dictionary
            Model component representing the baryons (assumed to be light emitting),
            or dictionary containing a list of the baryon components (baryons['components'])
            and a list of whether the baryon components are light emitting or not (baryons['light'])

        r_fdm : float
            Radius at which the dark matter fraction is determined

        Returns
        -------
        rB : float
            Core radius in kpc

        Notes
        -----
        This uses the current values of `fdm`, and `mvirial` together with
        the input baryon distribution to calculate the necessary value of `rB`.
        """
        if (self.fdm.value > self.bounds['fdm'][1]) | \
                ((self.fdm.value < self.bounds['fdm'][0])):
            rB = np.NaN
        else:
            if isinstance(baryons, dict):
                vsqr_bar_re = 0
                for bcmp in baryons['components']:
                    vsqr_bar_re += bcmp.vcirc_sq(r_fdm)
            else:
                vsqr_bar_re = baryons.vcirc_sq(r_fdm)
            vsqr_dm_re_target = vsqr_bar_re / (1./self.fdm - 1)

            rBtest = np.arange(0., 250., 5.0)
            vtest = np.array([self._minfunc_vdm_rB_from_fDM(rBt, vsqr_dm_re_target, self.mvirial, self.z, r_fdm) for rBt in rBtest])

            try:
                a = rBtest[vtest < 0][-1]
                try:
                    b = rBtest[vtest > 0][0]
                except:
                    a = rBtest[0]    # Even if not perfect, force in case of no convergence...
                    b = rBtest[1]
            except:
                a = rBtest[-2] # Even if not perfect, force in case of no convergence...
                b = rBtest[-1]

            try:
                rB = scp_opt.brentq(self._minfunc_vdm_rB_from_fDM, a, b, args=(vsqr_dm_re_target, self.mvirial, self.z, r_fdm))
            except:
                # SOMETHING, if it's failing...
                rB = np.average([a,b])

        return rB

    def _minfunc_vdm_rB_from_fDM(self, rB, vsqtarget, mass, z, r_fdm):
        halo = Burkert(mvirial=mass, rB=rB, z=z)
        return halo.vcirc_sq(r_fdm) - vsqtarget




class Einasto(DarkMatterHalo):
    r"""
    Dark matter halo following an Einasto profile

    Parameters
    ----------
    mvirial : float
        Virial mass in logarithmic solar units

    conc : float
        Concentration parameter

    nEinasto : float
        Inverse of the power law logarithmic slope

    alphaEinasto : float
        Power law logarithmic slope

    fdm : float
        Dark matter fraction

    z : float
        Redshift

    cosmo : `~astropy.cosmology` object
        The cosmology to use for modelling.
        If this model component will be attached to a `~dysmalpy.galaxy.Galaxy` make sure
        the respective cosmologies are the same. Default is
        `~astropy.cosmology.FlatLambdaCDM` with H0=70., and Om0=0.3.

    Einasto_param : {'None', 'nEinasto', 'alphaEinasto'}
        Which parameter to leave as the free parameter. If 'None', the model
        determines which parameter to use based on if `nEinasto` or `alphaEinasto`
        is None. Default is 'None'

    Notes
    -----
    Model formula following Retana-Montenegro et al (2012) [1]_:

    .. math::

        \rho = \rho_0 \exp\left\{-\left(\frac{r}{h}\right)^{1/n}\right\}

    where :math:`h=r_s/(2n)^n` is the scale length and
    :math:`r_s` is the scale radius defined as :math:`r_\mathrm{vir}/c`.

    In this model only `nEinasto` or `alphaEinasto` can be free since :math:`n=1/\alpha`.

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/2012A%26A...540A..70R/abstract

    """

    mvirial = DysmalParameter(default=1.0, bounds=(5, 20))
    conc = DysmalParameter(default=5.0, bounds=(2, 20))
    nEinasto = DysmalParameter(default=1.0)
    alphaEinasto = DysmalParameter(default=-99., fixed=True)
    fdm = DysmalParameter(default=-99.9, fixed=True, bounds=(0,1))

    _subtype = 'dark_matter'

    def __init__(self, z=0, cosmo=_default_cosmo,
            Einasto_param='None', alphaEinasto=None, nEinasto=None, **kwargs):

        # Check whether at least *one* of alphaEinasto and nEinasto is set:
        if (alphaEinasto is None) & (nEinasto is None):
            raise ValueError("Must set at least one of alphaEinasto and nEinasto!")
        if (alphaEinasto is not None) & (nEinasto is not None) & (Einasto_param == 'None'):
            raise ValueError("If both 'alphaEinasto' and 'nEinasto' are set, must specify which is the fit variable with 'Einasto_param'")

        super(Einasto, self).__init__(z=z, cosmo=cosmo, **kwargs)

        # Setup the "alternating" of whether to use nEinasto or alphaEinasto:
        if (Einasto_param.lower() == 'neinasto') | (alphaEinasto is None):
            self.Einasto_param = 'nEinasto'
            self.alphaEinasto.fixed = False
            self.alphaEinasto.tied = self.tie_alphaEinasto
        elif (Einasto_param.lower() == 'alphaeinasto') | (nEinasto is None):
            self.Einasto_param = 'alphaEinasto'
            self.nEinasto.fixed = False
            self.nEinasto.tied = self.tie_nEinasto
        else:
            raise ValueError("Einasto_param = {} not recognized! [options: 'nEinasto', 'alphaEinasto']".format(Einasto_param))

    def evaluate(self, r, mvirial, conc, alphaEinasto, nEinasto, fdm):
        """Mass density as a function of radius"""

        if self.Einasto_param.lower() == 'alphaeinasto':
            nEinasto = 1./alphaEinasto

        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0(rvirial=rvirial)
        rs = rvirial / conc
        h = rs / np.power(2.*nEinasto, nEinasto)

        # Return the density at a given radius:
        return rho0 * np.exp(- np.power(r/h, 1./nEinasto))

        # Equivalent to:
        #  rho0 * np.exp( - 2 * nEinasto * ( np.power(r/rs, 1./nEinasto) -1.) )
        # or
        #  rho0 * np.exp( - 2 / alphaEinasto * ( np.power(r/rs, alphaEinasto) -1.) )

    def enclosed_mass(self, r):
        """
        Enclosed mass as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        menc : float or array
            Enclosed mass in solar units
        """
        rvirial = self.calc_rvir()
        rs = rvirial/self.conc
        h = rs / np.power(2.*self.nEinasto, self.nEinasto)

        rho0 = self.calc_rho0(rvirial=rvirial)

        # Explicitly substituted for s = r/h before doing s^(1/nEinasto)
        incomp_gam =  scp_spec.gammainc(3*self.nEinasto, 2.*self.nEinasto * np.power(r/rs, 1./self.nEinasto) ) \
                        * scp_spec.gamma(3*self.nEinasto)

        Menc = 4.*np.pi * rho0 * np.power(h, 3.) * self.nEinasto * incomp_gam

        return Menc

    def calc_rho0(self, rvirial=None):
        r"""
        Density at the scale length

        Returns
        -------
        rho0 : float
            Mass density at the scale radius in :math:`M_{\odot}/\rm{kpc}^3`
        """
        if rvirial is None:
            rvirial = self.calc_rvir()
        rs = rvirial/self.conc
        h = rs / np.power(2.*self.nEinasto, self.nEinasto)

        incomp_gam =  scp_spec.gammainc(3*self.nEinasto, (2.*self.nEinasto) * \
                        np.power(self.conc, 1./self.nEinasto) ) \
                        * scp_spec.gamma(3*self.nEinasto)

        rho0 = 10**self.mvirial / (4.*np.pi*self.nEinasto * np.power(h, 3.) * incomp_gam)

        return rho0

    def calc_alphaEinasto_from_fdm(self, baryons, r_fdm):
        """
        Calculate alpha given dark matter fraction and baryonic distribution

        Parameters
        ----------
        baryons : `~dysmalpy.models.MassModel` or dictionary
            Model component representing the baryons (assumed to be light emitting),
            or dictionary containing a list of the baryon components (baryons['components'])
            and a list of whether the baryon components are light emitting or not (baryons['light'])

        r_fdm : float
            Radius at which the dark matter fraction is determined

        Returns
        -------
        alphaEinasto : float
            Power law logarithmic slope

        Notes
        -----
        This uses the current values of `fdm`, and `mvirial` together with
        the input baryon distribution to calculate the necessary value of `alphaEinasto`.
        """

        nEinasto = self.calc_nEinasto_from_fdm(baryons, r_fdm)
        if np.isfinite(nEinasto):
            return 1./nEinasto
        else:
            return np.NaN

    def calc_nEinasto_from_fdm(self, baryons, r_fdm):
        """
        Calculate n given the dark matter fraction and baryonic distribution

        Parameters
        ----------
        baryons : `~dysmalpy.models.MassModel` or dictionary
            Model component representing the baryons (assumed to be light emitting),
            or dictionary containing a list of the baryon components (baryons['components'])
            and a list of whether the baryon components are light emitting or not (baryons['light'])

        r_fdm : float
            Radius at which the dark matter fraction is determined

        Returns
        -------
        alphaEinasto : float
            Power law logarithmic slope

        Notes
        -----
        This uses the current values of `fdm`, and `mvirial` together with
        the input baryon distribution to calculate the necessary value of `nEinasto`.
        """

        if (self.fdm.value > self.bounds['fdm'][1]) | \
                ((self.fdm.value < self.bounds['fdm'][0])):
            nEinasto = np.NaN
        else:

            # NOTE: have not tested this yet

            if isinstance(baryons, dict):
                vsqr_bar_re = 0
                for bcmp in baryons['components']:
                    vsqr_bar_re += bcmp.vcirc_sq(r_fdm)
            else:
                vsqr_bar_re = baryons.vcirc_sq(r_fdm)

            vsqr_dm_re_target = vsqr_bar_re / (1./self.fdm - 1)

            nEinastotest = np.arange(-50, 50, 1.)
            vtest = np.array([self._minfunc_vdm_nEin_from_fdm(nEinast, vsqr_dm_re_target, self.mvirial, self.conc,
                                    self.alphaEinasto, self.z, r_fdm) for nEinast in nEinastotest])

            try:
                a = nEinastotest[vtest < 0][-1]
                try:
                    b = nEinastotest[vtest > 0][0]
                except:
                    a = nEinastotest[-2] # Even if not perfect, force in case of no convergence...
                    b = nEinastotest[-1]
            except:
                a = nEinastotest[0]    # Even if not perfect, force in case of no convergence...
                b = nEinastotest[1]

            alpha = scp_opt.brentq(self._minfunc_vdm_nEin_from_fdm, a, b, args=(vsqr_dm_re_target,
                                        self.mvirial, self.conc,
                                        self.alphaEinasto, self.z, r_fdm))

        return nEinasto

    def _minfunc_vdm_nEin_from_fdm(self, nEinasto, vsqtarget, mass, conc, alphaEinasto, z, r_fdm):
        halo = Einasto(mvirial=mass, conc=conc, nEinasto=nEinasto, alphaEinasto=alphaEinasto, z=z)
        return halo.vcirc_sq(r_fdm) - vsqtarget

    def tie_nEinasto(self, model_set):
        """
        Function to tie n to alpha

        Parameters
        ----------
        model_set : `ModelSet`
            `ModelSet` the component is a part of and will be used in the fitting

        Returns
        -------
        nEinasto : float
            `nEinastro` given the current value of `alphaEinasto`

        """
        if model_set.components['halo'].alphaEinasto.value != self.alphaEinasto:
            raise ValueError
        return 1./self.alphaEinasto

    def tie_alphaEinasto(self, model_set):
        """
        Function to tie alpha to n

        Parameters
        ----------
        model_set : `ModelSet`
            `ModelSet` the component is a part of and will be used in the fitting

        Returns
        -------
        alphaEinasto : float
            `alphaEinastro` given the current value of `nEinasto`

        """
        return 1./self.nEinasto



class DekelZhao(DarkMatterHalo):
    r"""
    Dekel-Zhao halo profile (Dekel et al. 2017, Freundlich et al. 2020)

    Parameters
    ----------
    mvirial : float
        Virial mass in logarithmic solar units

    s1 : float
        Inner logarithmic slope (at resolution r1=0.01*Rvir)

    c2 : float
        Concentration parameter (defined relative to c, a)

    fdm : float
        Dark matter fraction

    z : float
        Redshift

    cosmo : `~astropy.cosmology` object
        The cosmology to use for modelling.
        If this model component will be attached to a `~dysmalpy.galaxy.Galaxy` make sure
        the respective cosmologies are the same. Default is
        `~astropy.cosmology.FlatLambdaCDM` with H0=70., and Om0=0.3.

    Notes
    -----
    The formula for this implementation are given in Freundlich et al. 2020. [1]_
    Specifically, we use the forms where b=2, gbar=3 (see Eqns 4, 5, 14, 15)

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.2912F/abstract

    """

    # Powerlaw slopes for the density model
    mvirial = DysmalParameter(default=1.0, bounds=(5, 20))
    s1 = DysmalParameter(default=1.5, bounds=(0.0, 2.0))
    c2 = DysmalParameter(default=25., bounds=(0.0, 40.0))
    fdm = DysmalParameter(default=-99.9, fixed=True, bounds=(0,1))

    _subtype = 'dark_matter'

    def __init__(self, z=0, cosmo=_default_cosmo, **kwargs):
        super(DekelZhao, self).__init__(z=z, cosmo=cosmo, **kwargs)

    def evaluate(self, r, mvirial, s1, c2, fdm):
        """ Mass density for the DekelZhao halo profile"""

        rvirial = self.calc_rvir()
        a, c = self.calc_a_c()
        rhoc = self.calc_rho0(rvirial=rvirial, a=a, c=c)

        rc = rvirial / c
        x = r / rc

        return rhoc / (np.power(x, a) * np.power((1.+np.sqrt(x)), 2.*(3.5-a)))

    def enclosed_mass(self, r):
        """
        Enclosed mass as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        menc : float or array
            Enclosed mass in solar units
        """
        mvir = 10**self.mvirial
        rvirial = self.calc_rvir()
        a, c = self.calc_a_c()

        rc = rvirial / c
        x = r / rc
        mu = self.calc_mu(a=a, c=c)

        return mu * mvir / (np.power(x, a-3.)*np.power((1.+np.sqrt(x)), 2.*(3.-a)))

    def calc_a_c(self):
        r"""
        Calculate a, c from s1, c2 for the Dekel-Zhao halo.

        Returns
        -------
        a, c:   inner asymptotic slope, concentration parameter for DZ halo

        """
        #rvirial = self.calc_rvir()
        #r12 = np.sqrt(0.01*rvirial/rvirial)
        r12 = np.sqrt(0.01)
        c12 = np.sqrt(self.c2)
        a = (1.5*self.s1 - 2.*(3.5-self.s1)*r12*c12)/(1.5 - (3.5-self.s1)*r12*c12)
        c = ((self.s1-2.)/((3.5-self.s1)*r12 - 1.5/c12))**2

        return a, c

    def calc_rho0(self, rvirial=None, a=None, c=None):
        r"""
        Normalization of the density distribution, rho_c

        Returns
        -------
        rhoc : float
            Mass density normalization in :math:`M_{\odot}/\rm{kpc}^3`
        """
        if (a is None) or (c is None):
            a, c = self.calc_a_c()

        mu = self.calc_mu(a=a, c=c)
        rhovirbar = self.calc_rhovirbar(rvirial=rvirial)
        rhocbar = c**3 * mu * rhovirbar

        return (1.-a/3.)*rhocbar

    def calc_rhovirbar(self, rvirial=None):
        r"""
        Average density in the virial radius, in :math:`M_{\odot}/\rm{kpc}^3`
        """
        mvir = 10**self.mvirial
        if rvirial is None:
            rvirial = self.calc_rvir()

        rhovirbar = (3.*mvir)/(4.*np.pi*(rvirial**3))
        return rhovirbar

    def calc_mu(self, a=None, c=None):
        """
        Subfunction for describing DZ profile
        """
        if (a is None) or (c is None):
            a, c = self.calc_a_c()

        mu = np.power(c, a-3.) * np.power((1.+np.sqrt(c)), 2.*(3.-a))
        return mu

    def _minfunc_vdm_mvir_from_fdm(self, mvirial, vsqtarget, r_fdm, bary, adiabatic_contract):
        halotmp = self.copy()
        halotmp.__setattr__('mvirial', mvirial)

        modtmp = ModelSet()
        if isinstance(bary, dict):
            for bcmp,b_light in zip(bary['components'], bary['light']):
                modtmp.add_component(bcmp, light=b_light)
        else:
            modtmp.add_component(bary, light=True)
        modtmp.add_component(halotmp)
        modtmp.kinematic_options.adiabatic_contract = adiabatic_contract
        if adiabatic_contract:
            modtmp.kinematic_options.adiabatic_contract_modify_small_values = True
        modtmp._update_tied_parameters()

        if adiabatic_contract:
            vc_sq, vc_sq_dm = modtmp.vcirc_sq(r_fdm, compute_dm=True)
            return vc_sq_dm - vsqtarget
        else:
            return modtmp.components['halo'].vcirc_sq(r_fdm) - vsqtarget


class LinearNFW(DarkMatterHalo):
    r"""
    Same as `NFW` except with the virial mass in linear units

    Parameters
    ----------
    mvirial : float
        Virial mass in  solar units

    conc : float
        Concentration parameter

    fdm : float
        Dark matter fraction

    z : float
        Redshift

    cosmo : `~astropy.cosmology` object
        The cosmology to use for modelling.
        If this model component will be attached to a `~dysmalpy.galaxy.Galaxy` make sure
        the respective cosmologies are the same. Default is
        `~astropy.cosmology.FlatLambdaCDM` with H0=70., and Om0=0.3.

    Notes
    -----
    Model formula:

    The mass density follows Navarro, Frenk, & White (1995) [1]_:

    .. math::

        \rho = \frac{\rho_0}{(r/r_s)(1 + r/r_s)^2}

    :math:`r_s` is the scale radius defined as :math:`r_{\rm vir}/c`.
    :math:`\rho_0` is the normalization parameter. 

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/1995MNRAS.275..720N/abstract
    """
    mvirial = DysmalParameter(default=1.e1, bounds=(1.e5, 1.e20))
    conc = DysmalParameter(default=5.0, bounds=(2, 20))
    fdm = DysmalParameter(default=-99.9, fixed=True, bounds=(0,1))

    def __init__(self, z=0, cosmo=_default_cosmo, **kwargs):
        super(LinearNFW, self).__init__(z=z, cosmo=cosmo, **kwargs)

    def evaluate(self, r, mvirial, conc, fdm):
        """Mass density as a function of radius"""

        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0(rvirial=rvirial)
        rs = rvirial / self.conc

        return rho0 / (r / rs * (1 + r / rs) ** 2)

    def enclosed_mass(self, r):
        """
        Enclosed mass as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        menc : float or array
            Enclosed mass in solar units
        """

        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0(rvirial=rvirial)
        rs = rvirial/self.conc
        aa = 4.*np.pi*rho0*rvirial**3/self.conc**3

        # For very small r, bb can be negative.
        bb = np.abs(np.log((rs + r)/rs) - r/(rs + r))

        return aa*bb

    def calc_rho0(self, rvirial=None):
        r"""
        Normalization of the density distribution

        Returns
        -------
        rho0 : float
            Mass density normalization in :math:`M_{\odot}/\rm{kpc}^3`
        """
        
        if rvirial is None:
            rvirial = self.calc_rvir()
        aa = self.mvirial/(4.*np.pi*rvirial**3)*self.conc**3
        bb = 1./(np.log(1.+self.conc) - (self.conc/(1.+self.conc)))

        return aa * bb
    
    def calc_rvir(self):
        r"""
        Calculate the virial radius based on virial mass and redshift

        Returns
        -------
        rvir : float
            Virial radius

        Notes
        -----
        Formula:

        .. math::

            M_{\rm vir} = 100 \frac{H(z)^2 R_{\rm vir}^3}{G}

        This is based on Mo, Mao, & White (1998) [1]_ which defines the virial
        radius as the radius where the mean mass density is :math:`200\rho_{\rm crit}`.
        :math:`\rho_{\rm crit}` is the critical density for closure at redshift, :math:`z`.
        """
        # hz = self.cosmo.H(self.z).value
        # rvir = ((self.mvirial * (g_pc_per_Msun_kmssq * 1e-3) /
        #         (10 * hz * 1e-3) ** 2) ** (1. / 3.))
        rvir = ((self.mvirial * (g_pc_per_Msun_kmssq * 1e-3) /
                (10 * self._hz * 1e-3) ** 2) ** (1. / 3.))

        return rvir
