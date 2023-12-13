# Methods for tying parameters for fitting fitting_wrappers

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from functools import partial

import numpy as np

from dysmalpy import parameters

import scipy.optimize as scp_opt


def tie_rturn(model_set):
    # Define a function to tie the turnover radius to half the end radius
    rend = model_set.components['outflow'].rend.value

    return rend/2.

def get_param_value(mod_set, main_geom_name, param_name):
    return mod_set.components[main_geom_name].__dict__[param_name].value

def tied_geom_lambda(main_geom_name, param_name):
    return partial(get_param_value, main_geom_name=main_geom_name, param_name=param_name)

def tie_sigz_reff(model_set):
    #'sersic', 'disk+bulge', 'lsersic', 'GaussianRing'
    reff = None
    if 'disk+bulge' in model_set.light_components.keys():
        if model_set.light_components['disk+bulge']:
            reff = model_set.components['disk+bulge'].r_eff_disk.value
    elif 'sersic' in model_set.light_components.keys():
        if model_set.light_components['sersic']:
            reff = model_set.components['sersic'].r_eff.value
    elif 'lsersic' in model_set.light_components.keys():
        if model_set.light_components['lsersic']:
            reff = model_set.components['lsersic'].r_eff.value
    elif 'ring' in model_set.light_components.keys():
        if model_set.light_components['ring']:
            reff = model_set.components['ring'].ring_reff()

    if 'disk+bulge' in model_set.components.keys():
        invq = model_set.components['disk+bulge'].invq_disk
    elif 'sersic' in model_set.components.keys():
        invq = model_set.components['sersic'].invq
    else:
        invq = 5.  # USE A DEFAULT of q=0.2

    if reff is not None:
        sigz = 2.0*reff/invq/2.35482
    else:
        sigz = 1.

    return sigz



def m13_lshfm(lMh, z):
    # Moster + 2013, MNRAS 428, 3121
    # Table 1, Best fit
    M10 = 11.590
    M11 = 1.195
    N10 = 0.0351
    N11 = -0.0247
    b10 = 1.376
    b11 = -0.826
    g10 = 0.608
    g11 = 0.329

    zarg = z/(1.+z)
    M1 = np.power(10., M10 + M11*zarg)
    Nz = N10 + N11*zarg
    bz = b10 + b11*zarg
    gz = g10 + g11*zarg

    Mh = np.power(10., lMh)
    lmSMh = np.log10(2*Nz) - np.log10(np.power(Mh/M1 , -bz)  + np.power( Mh/M1, gz))

    return lmSMh

def lmstar_num_solver_moster(lMh, z, lmass):

    return m13_lshfm(lMh, z) - lmass + lMh


def moster13_halo_mass_num_solve(z=None, lmass=None, truncate_lmstar_halo=None):
    # Do zpt solver to get lmhalo given lmass:

    if truncate_lmstar_halo is None:
        raise ValueError

    if truncate_lmstar_halo:
        lmstar = min(lmass, 11.2)
    else:
        lmstar = lmass

    lmhalo = scp_opt.newton(lmstar_num_solver_moster, lmstar + 2.,
                        args=(z, lmstar),
                        maxiter=200)

    return lmhalo


def behroozi13_halo_mass(z=None, lmass=None):
    # From the inverted relation fit by Omri Ginzburg (A Dekel student; email from A Dekel 2020-05-15)
    # Valid for lM* = 10-12; z=0.5-3 at better than 0.5% accuracy
    # ** NO TRUNCATION **

    A0, A1, A2, A3, A4 = 13.3402, -1.8671, 1.3010, -0.4037, 0.0439
    B0, B1, B2, B3, B4 = -0.1814, 0.1791, -0.1020, 0.0270, -2.85e-3
    C0, C1, C2, C3 = 0.7361, 0.6427, -0.2737, 0.0280
    D0, D1, D2, D3 = 5.3744, 6.2722, -2.6661, 0.2503

    A_OGAD_z = A0 + A1*z + A2*(z**2) + A3*(z**3) + A4*(z**4)
    B_OGAD_z = B0 + B1*z + B2*(z**2) + B3*(z**3) + B4*(z**4)
    C_OGAD_z = C0 + C1*z + C2*(z**2) + C3*(z**3)
    D_OGAD_z = D0 + D1*z + D2*(z**2) + D3*(z**3)

    lmhalo = A_OGAD_z + B_OGAD_z*lmass * np.sin(C_OGAD_z * lmass - D_OGAD_z)

    return lmhalo

def moster13_halo_mass(z=None, lmass=None):
    # From the fitting relation from Moster, Naab & White 2013; from email from Thorsten Naab on 2020-05-21
    # ** NO TRUNCATION **

    log_m1 = 10.485 + 1.099 * (z/(z+1.))
    n  = np.power( 10., (1.425 + 0.328 * (z/(z+1.)) - 1.174 * ((z/(z+1.))**2)) )
    b  = -0.569 + 0.132 * (z/(z+1.))
    g  = 1.023 + 0.295 * (z/(z+1.)) - 2.768 * ((z/(z+1.))**2)

    lmhalo = lmass + np.log10(0.5) + np.log10(n) + np.log10( np.power( np.power(10., (lmass-log_m1)), b ) + np.power( np.power(10., (lmass-log_m1)), g ) )

    return lmhalo

def moster18_halo_mass(z=None, lmass=None):
    # From the updated fitting relation from Moster, Naab & White 2018; from email from Thorsten Naab on 2020-05-21
    # From stellar mass binned fitting result (avoids divergance at high lMstar)
    # ** NO TRUNCATION **

    log_m1 = 10.6
    n = np.power(10., (1.507 - 0.124 * (z/(z+1.)) ) )
    b = -0.621 - 0.059 * (z/(z+1.))
    g = 1.055 + 0.838 * (z/(z+1)) - 3.083 * ( ((z/(z+1)))**2 )

    lmhalo = lmass + np.log10(0.5) + np.log10(n) + np.log10( np.power( np.power(10., (lmass-log_m1)), b ) + np.power( np.power(10., (lmass-log_m1)), g ) )

    return lmhalo

def tied_mhalo_mstar(model_set):
    # Uses constant fgas to go from lMbar to the stellar mass for the SMHM calculation
    z = model_set.components['halo'].z

    lmbar = model_set.components['disk+bulge'].total_mass.value
    fgas = model_set.components['disk+bulge'].fgas
    Mbar = np.power(10., lmbar)
    Mstar = (1.-fgas)*Mbar

    try:
        mhalo_relation = model_set.components['disk+bulge'].mhalo_relation
    except:

        print("Missing mhalo_relation! setting mhalo_relation='Moster18' ! [options: 'Moster18', 'Behroozi13', 'Moster13']")
        mhalo_relation = 'Moster18'

    ########

    if mhalo_relation.lower().strip() == 'behroozi13':
        lmhalo = behroozi13_halo_mass(z=z, lmass=np.log10(Mstar))

    elif mhalo_relation.lower().strip() == 'moster18':
        lmhalo = moster18_halo_mass(z=z, lmass=np.log10(Mstar))

    elif mhalo_relation.lower().strip() == 'moster13':
        raise ValueError

        ## OLD VERSION, NUMERICAL SOLUTION TO MOSTER13
        try:
            truncate_lmstar_halo = model_set.components['disk+bulge'].truncate_lmstar_halo
        except:
            print("Missing truncate_lmstar_halo! setting truncate_lmstar_halo=True")
            truncate_lmstar_halo = True

        lmhalo = moster13_halo_mass_num_solve(z=z, lmass=np.log10(Mstar),
                                truncate_lmstar_halo=truncate_lmstar_halo)
    ####
    return lmhalo


def tied_mhalo_mstar_fixed_fgas(model_set):
    # Alias to old method:
    return tied_mhalo_mstar(model_set)

def tied_mhalo_mstar_fixed_lmstar(model_set):
    # Uses constant lMstar the SMHM calculation
    z = model_set.components['halo'].z

    Mstar = np.power(10., model_set.components['disk+bulge'].lmstar)

    try:
        mhalo_relation = model_set.components['disk+bulge'].mhalo_relation
    except:
        print("Missing mhalo_relation! setting mhalo_relation='Moster18' ! [options: 'Moster18', 'Behroozi13', 'Moster13']")
        mhalo_relation = 'Moster18'

    ########

    if mhalo_relation.lower().strip() == 'behroozi13':
        lmhalo = behroozi13_halo_mass(z=z, lmass=np.log10(Mstar))

    elif mhalo_relation.lower().strip() == 'moster18':
        lmhalo = moster18_halo_mass(z=z, lmass=np.log10(Mstar))

    elif mhalo_relation.lower().strip() == 'moster13':
        raise ValueError
        ## OLD VERSION, NUMERICAL SOLUTION TO MOSTER13
        try:
            truncate_lmstar_halo = model_set.components['disk+bulge'].truncate_lmstar_halo
        except:
            print("Missing truncate_lmstar_halo! setting truncate_lmstar_halo=True")
            truncate_lmstar_halo = True

        lmhalo = moster13_halo_mass_num_solve(z=z, lmass=np.log10(Mstar),
                                truncate_lmstar_halo=truncate_lmstar_halo)
    ####
    return lmhalo

############################################################################
# Tied functions for halo fitting:

# def tie_lmvirial_to_fdm(model_set):
#     comp_halo = model_set.components.__getitem__('halo')
#     comp_baryons = model_set.components.__getitem__('disk+bulge')
#     r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
#     mvirial = comp_halo.calc_mvirial_from_fdm(comp_baryons, r_fdm,
#                     adiabatic_contract=model_set.kinematic_options.adiabatic_contract)
#     return mvirial
#
#
# def tie_alpha_TwoPower(model_set):
#     comp_halo = model_set.components.__getitem__('halo')
#     comp_baryons = model_set.components.__getitem__('disk+bulge')
#     r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
#     alpha = comp_halo.calc_alpha_from_fdm(comp_baryons, r_fdm)
#     return alpha
#

def tie_lmvirial_NFW(model_set):
    return tie_lmvirial_to_fdm(model_set)

def tie_lmvirial_to_fdm(model_set):
    comp_halo = None
    comps_bar = []
    light_bar = []

    for cmp in model_set.mass_components:
        if model_set.mass_components[cmp]:
            mcomp = model_set.components.__getitem__(cmp)

            if (mcomp._subtype == 'dark_matter'):
                if comp_halo is not None:
                    raise ValueError("Overwriting halo component!")
                comp_halo = mcomp

            elif mcomp._subtype == 'baryonic':
                comps_bar.append(mcomp)
                light_bar.append(model_set.light_components[cmp])

    bar_dict = {'components': comps_bar,
                'light': light_bar}

    # r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    # Generalize r_fdm component (as in changes from AN):
    if 'disk+bulge' in model_set.mass_components:
        if model_set.mass_components['disk+bulge']:
            r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    elif 'disk' in model_set.mass_components:
        if model_set.mass_components['disk']:
            r_fdm = model_set.components['disk'].r_eff.value
    elif 'ring' in model_set.mass_components:
        if model_set.mass_components['ring']:
            r_fdm = model_set.components['ring'].ring_reff()
    else:
        r_fdm = 1.

    mvirial = comp_halo.calc_mvirial_from_fdm(bar_dict, r_fdm,
                    adiabatic_contract=model_set.kinematic_options.adiabatic_contract)
    return mvirial

def tie_alpha_TwoPower(model_set):
    comp_halo = None
    comps_bar = []
    light_bar = []

    for cmp in model_set.mass_components:
        if model_set.mass_components[cmp]:
            mcomp = model_set.components.__getitem__(cmp)

            if (mcomp._subtype == 'dark_matter'):
                if comp_halo is not None:
                    raise ValueError("Overwriting halo component!")
                comp_halo = mcomp

            elif mcomp._subtype == 'baryonic':
                comps_bar.append(mcomp)
                light_bar.append(model_set.light_components[cmp])

    bar_dict = {'components': comps_bar,
                'light': light_bar}

    r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    alpha = comp_halo.calc_alpha_from_fdm(bar_dict, r_fdm)
    return alpha

def tie_rB_Burkert(model_set):
    comp_halo = model_set.components.__getitem__('halo')
    comp_baryons = model_set.components.__getitem__('disk+bulge')
    r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    rB = comp_halo.calc_rB_from_fdm(comp_baryons, r_fdm)
    return rB


def tie_alphaEinasto_Einasto(model_set):
    comp_halo = model_set.components.__getitem__('halo')
    comp_baryons = model_set.components.__getitem__('disk+bulge')
    r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    alphaEinasto = comp_halo.calc_alphaEinasto_from_fdm(comp_baryons, r_fdm)
    return alphaEinasto

def tie_nEinasto_Einasto(model_set):
    comp_halo = model_set.components.__getitem__('halo')
    comp_baryons = model_set.components.__getitem__('disk+bulge')
    r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    nEinasto = comp_halo.calc_nEinasto_from_fdm(comp_baryons, r_fdm)
    return nEinasto



def _DZ_s1_MstarMhalo(Mstar, Mvir):
    """
    Function to tie s1 to M*/Mvir.
    From Freundlich et al. 2020, Eqs 45, 47, 48 + Table 1
    """
    cdmo = np.power(10., 1.025 - 0.097 * np.log10(Mvir * 0.671 / 1.e12))
    sdmo = (1. + 0.03*cdmo) / (1. + 0.01*cdmo)

    x = Mstar/Mvir
    x0 = 1.30e-3
    sprime = 1
    spp = 0.32
    nu = 2.86
    s1ratio = sprime / (1. + np.power(x/x0, nu)) + spp * np.log10(1. + np.power(x/x0, nu))

    return s1ratio * sdmo

def _DZ_c2_MstarMhalo(Mstar, Mvir):
    """
    Function to tie c1 to M*/Mvir.
    From Freundlich et al. 2020, Eqs 47, 49 + Table 1
    """
    cdmo = np.power(10., 1.025 - 0.097 * np.log10(Mvir * 0.671 / 1.e12))

    x = Mstar/Mvir
    x0 = 2.43e-2
    cprime = 1.14
    nu = 1.37
    mu = 0.142
    c2ratio = cprime * (1. + np.power(x/x0, nu)) - np.power(x, mu)

    return c2ratio * cdmo

def tie_DZ_s1_MstarMhalo(model_set):
    Mstar = np.power(10., model_set.components['disk+bulge'].lmstar)
    Mvir = np.power(10., model_set.components['halo'].mvirial.value)
    return _DZ_s1_MstarMhalo(Mstar, Mvir)

def tie_DZ_c2_MstarMhalo(model_set):
    Mstar = np.power(10., model_set.components['disk+bulge'].lmstar)
    Mvir = np.power(10., model_set.components['halo'].mvirial.value)
    return _DZ_c2_MstarMhalo(Mstar, Mvir)


def tie_fdm(model_set):
    # r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    # Generalize r_fdm component (as in changes from AN):
    if 'disk+bulge' in model_set.mass_components:
        if model_set.mass_components['disk+bulge']:
            r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    elif 'disk' in model_set.mass_components:
        if model_set.mass_components['disk']:
            r_fdm = model_set.components['disk'].r_eff.value
    elif 'ring' in model_set.mass_components:
        if model_set.mass_components['ring']:
            r_fdm = model_set.components['ring'].ring_reff()
    else:
        r_fdm = 1.
    # r_fdm = model_set.components['disk+bulge'].r_eff_disk.value
    fdm = model_set.get_dm_aper(r_fdm)
    return fdm

############################################################################



class TiedUniformPriorLowerTrunc(parameters.UniformPrior):
    def __init__(self, compn='disk+bulge', paramn='total_mass'):
        self.compn = compn
        self.paramn = paramn

        super(TiedUniformPriorLowerTrunc, self).__init__()
    def log_prior(self, param, modelset=None, **kwargs):

        pmin = modelset.components[self.compn].__getattribute__(self.paramn).value

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (param.value >= pmin) & (param.value <= pmax):
            return 0.
        else:
            return -np.inf


class TiedBoundedGaussianPriorLowerTrunc(parameters.BoundedGaussianPrior):
    def __init__(self, compn='disk+bulge', paramn='total_mass', center=0, stddev=1.0):
        self.compn = compn
        self.paramn = paramn

        super(TiedBoundedGaussianPriorLowerTrunc, self).__init__(center=center, stddev=stddev)

    def log_prior(self, param, modelset=None, **kwargs):

        pmin = modelset.components[self.compn].__getattribute__(self.paramn).value

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (param.value >= pmin) & (param.value <= pmax):
            return np.log(parameters.norm.pdf(param.value, loc=self.center, scale=self.stddev))
        else:
            return -np.inf
