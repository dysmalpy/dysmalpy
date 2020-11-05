# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# File containing all of the available models to use build the
# galaxy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import os
import abc
import logging
import six
from collections import OrderedDict

# Third party imports
import numpy as np
import scipy.special as scp_spec
import scipy.io as scp_io
import scipy.interpolate as scp_interp
import scipy.optimize as scp_opt
import scipy.ndimage as scp_ndi
import astropy.constants as apy_con
import astropy.units as u
from astropy.modeling import Model
import astropy.cosmology as apy_cosmo
import pyximport; pyximport.install()
from . import cutils

from astropy.table import Table

# Local imports
from .parameters import DysmalParameter

__all__ = ['ModelSet', 'Sersic', 'DiskBulge', 'LinearDiskBulge', 'ExpDisk',
           'NFW', 'LinearNFW', 'TwoPowerHalo', 'Burkert', 'Einasto',
           'DispersionConst', 'Geometry', 'BiconicalOutflow', 'UnresolvedOutflow',
           'UniformRadialInflow', 'DustExtinction',
           'KinematicOptions', 'ZHeightGauss',
           'surf_dens_exp_disk', 'menc_exp_disk', 'vcirc_exp_disk',
           'sersic_mr', 'sersic_menc', 'v_circular', 'menc_from_vcirc',
           'apply_noord_flat', 'calc_1dprofile', 'calc_1dprofile_circap_pv']

# NOORDERMEER DIRECTORY
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
_dir_noordermeer = dir_path+"/data/noordermeer/"


# ALT NOORDERMEER DIRECTORY:
# TEMP:
_dir_sersic_profile_mass_VC_TMP = "/Users/sedona/data/sersic_profile_mass_VC/"
_dir_sersic_profile_mass_VC = os.getenv('SERSIC_PROFILE_MASS_VC_DATADIR', _dir_sersic_profile_mass_VC_TMP)

try:
    import sersic_profile_mass_VC.calcs as sersic_profile_mass_VC_calcs
    _sersic_profile_mass_VC_loaded = True
except:
    _sersic_profile_mass_VC_loaded = False


# CONSTANTS
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc

# DEFAULT COSMOLOGY
_default_cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)

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

# TODO: Tied parameters are NOT automatically updated when variables change!!
# TODO: Need to keep track during the fitting!


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


def sersic_mr(r, mass, n, r_eff):
    """
    Radial surface mass density function for a generic sersic model

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
    mr : float or array
        Surface mass density as a function of `r`
    """

    bn = scp_spec.gammaincinv(2. * n, 0.5)
    alpha = r_eff / (bn ** n)
    amp = (mass / (2 * np.pi) / alpha ** 2 / n /
           scp_spec.gamma(2. * n))
    mr = amp * np.exp(-bn * (r / r_eff) ** (1. / n))

    return mr


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


def sersic_menc(r, mass, n, r_eff):
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
    return sersic_menc_2D_proj(r, mass, n, r_eff)

# def sersic_menc(r, mass, n, r_eff):
#     """Enclosed mass as a function of r for a sersic model: 3D Abel deprojection"""
#
#
#     raise ValueError("Implement: lookup table, or 3D integral")
#
#     menc = None
#
#     return menc


def v_circular(mass_enc, r):
    """
    Circular velocity given an enclosed mass and radius
    v(r) = SQRT(GM(r)/r)

    Parameters
    ----------
    mass_enc : float
        Enclosed mass in solar units

    r : float or array
        Radius at which to calculate the circular velocity in kpc

    Returns
    -------
    vcirc : float or array
        Circular velocity in km/s as a function of radius
    """
    vcirc = np.sqrt(G.cgs.value * mass_enc * Msun.cgs.value /
                    (r * 1000. * pc.cgs.value))
    vcirc = vcirc/1e5

    return vcirc


def menc_from_vcirc(vcirc, r):
    """
    Enclosed mass given a circular velocity and radius

    Parameters
    ----------
    vcirc : float or array
        Circular velocity in km/s

    r : float or array
        Radius at which to calculate the enclosed mass in kpc

    Returns
    -------
    menc : float or array
        Enclosed mass in solar units
    """
    menc = ((vcirc*1e5)**2.*(r*1000.*pc.cgs.value) /
                  (G.cgs.value * Msun.cgs.value))
    return menc


def _make_cube_ai(model, xgal, ygal, zgal, rstep=None, oversample=None, dscale=None,
            maxr=None, maxr_y=None):

    oversize = 1.5  # Padding factor for x trimming

    thick = model.zprofile.z_scalelength.value

    # # maxr, maxr_y are already in pixel units
    xsize = np.int(np.floor(2.*(maxr * oversize) +0.5))
    ysize = np.int(np.floor( 2.*maxr_y + 0.5))

    # Sample += 2 * scale length thickness
    # Modify: make sure there are at least 3 *whole* pixels sampled:
    zsize = np.max([ 3*oversample, np.int(np.floor(4.*thick/rstep*dscale + 0.5 )) ])

    if ( (xsize%2) < 0.5 ): xsize += 1
    if ( (ysize%2) < 0.5 ): ysize += 1
    if ( (zsize%2) < 0.5 ): zsize += 1

    zi, yi, xi = np.indices(xgal.shape)
    full_ai = np.vstack([xi.flatten(), yi.flatten(), zi.flatten()])

    origpos = np.vstack([xgal.flatten() - np.mean(xgal.flatten()) + xsize/2.,
                         ygal.flatten() - np.mean(ygal.flatten()) + ysize/2.,
                         zgal.flatten() - np.mean(zgal.flatten()) + zsize/2.])


    validpts = np.where( (origpos[0,:] >= -0.5) & (origpos[0,:] < xsize-0.5) & \
                         (origpos[1,:] >= -0.5) & (origpos[1,:] < ysize-0.5) & \
                         (origpos[2,:] >= -0.5) & (origpos[2,:] < zsize-0.5) )[0]

    ai = full_ai[:,validpts]

    return ai



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

    try:
        restNVC = scp_io.readsav(file_noord)
        N2008_vcirc = restNVC.N2008_vcirc
        N2008_rad = restNVC.N2008_rad
        N2008_Re = restNVC.N2008_Re
        N2008_mass = restNVC.N2008_mass

        v_interp = scp_interp.interp1d(N2008_rad, N2008_vcirc,
                                       fill_value="extrapolate")
        vcirc = (v_interp(r / r_eff * N2008_Re) * np.sqrt(
                 mass / N2008_mass) * np.sqrt(N2008_Re / r_eff))

    except:
        vcirc = apply_noord_flat_new(r, r_eff, mass, n, invq)
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

def sersic_curve_rho(r, Reff, total_mass, n, invq):
    table = get_sersic_VC_table_new(n, invq)

    table_rho =     table['rho']
    table_rad =     table['r']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Clean up values inside rmin:  Add the value at r=0: menc=0
    if table['r'][0] > 0.:
        #print("_sersic_profile_mass_VC_loaded={}".format(_sersic_profile_mass_VC_loaded))
        if _sersic_profile_mass_VC_loaded:
            try:
                table_rad = np.append(r.min() * table_Reff/Reff, table_rad)
                table_rho = np.append(sersic_profile_mass_VC_calcs.rho(r.min()* table_Reff/Reff,
                        n=n, total_mass=table_mass, Reff=table_Reff, q=table['q']), table_rho)
            except:
                pass

    r_interp = scp_interp.interp1d(table_rad, table_rho, fill_value=np.NaN, bounds_error=False, kind='cubic')
    r_interp_extrap = scp_interp.interp1d(table_rad, table_rho, fill_value='extrapolate', kind='linear')

    # Ensure it's an array:
    if isinstance(r*1., float):
        rarr = np.array([r])
    else:
        rarr = np.array(r)
    # Ensure all radii are 0. or positive:
    rarr = np.abs(rarr)

    rho_interp = np.zeros(len(rarr))
    wh_in =     np.where((r <= table_rad.max()) & (r >= table_rad.min()))[0]
    wh_extrap = np.where((r > table_rad.max()) | (r < table_rad.min()))[0]
    rho_interp[wh_in] =     (r_interp(rarr[wh_in] / Reff * table_Reff) * (total_mass / table_mass) * (table_Reff / Reff)**3 )
    rho_interp[wh_extrap] = (r_interp_extrap(rarr[wh_extrap] / Reff * table_Reff) * (total_mass / table_mass) * (table_Reff / Reff)**3 )

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

def sersic_curve_dlnrho_dlnr(r, Reff, n, invq):
    table = get_sersic_VC_table_new(n, invq)

    table_dlnrho_dlnr =     table['dlnrho_dlnr']
    table_rad =     table['r']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Drop nonfinite parts:
    whfin = np.where(np.isfinite(table_dlnrho_dlnr))[0]
    table_dlnrho_dlnr = table_dlnrho_dlnr[whfin]
    table_rad = table_rad[whfin]

    # Clean up values inside rmin:  Add the value at r=0: menc=0
    if table['r'][0] > 0.:
        if _sersic_profile_mass_VC_loaded:
            try:
                table_rad = np.append(r.min() * table_Reff/Reff, table_rad)
                table_dlnrho_dlnr = np.append(sersic_profile_mass_VC_calcs.dlnrho_dlnr(r.min()* table_Reff/Reff,
                            n=n, total_mass=table_mass, Reff=table_Reff, q=table['q']), table_dlnrho_dlnr)
            except:
                pass

    r_interp = scp_interp.interp1d(table_rad, table_dlnrho_dlnr, fill_value=np.NaN, bounds_error=False, kind='cubic')
    r_interp_extrap = scp_interp.interp1d(table_rad, table_dlnrho_dlnr, fill_value='extrapolate', kind='linear')

    # Ensure it's an array:
    if isinstance(r*1., float):
        rarr = np.array([r])
    else:
        rarr = np.array(r)
    # Ensure all radii are 0. or positive:
    rarr = np.abs(rarr)

    dlnrho_dlnr_interp = np.zeros(len(rarr))
    wh_in =     np.where((r <= table_rad.max()) & (r >= table_rad.min()))[0]
    wh_extrap = np.where((r > table_rad.max()) | (r < table_rad.min()))[0]
    dlnrho_dlnr_interp[wh_in] =     (r_interp(rarr[wh_in] / Reff * table_Reff) )
    dlnrho_dlnr_interp[wh_extrap] = (r_interp_extrap(rarr[wh_extrap] / Reff * table_Reff) )


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



def area_segm(rr, dd):

    return (rr**2 * np.arccos(dd/rr) -
            dd * np.sqrt(2. * rr * (rr-dd) - (rr-dd)**2))


#
def calc_1dprofile(cube, slit_width, slit_angle, pxs, vx, soff=0.):
    """
    Measure the 1D rotation curve from a cube using a pseudoslit.

    This function measures the 1D rotation curve by first creating a PV diagram based on the
    input slit properties. Fluxes, velocities, and dispersions are then measured from the spectra
    at each single position in the PV diagram by calculating the 0th, 1st, and 2nd moments
    of each spectrum.

    Parameters
    ----------
    cube : 3D array
        Data cube from which to measure the rotation curve. First dimension is assumed to
        be spectral direction.

    slit_width : float
        Slit width of the pseudoslit in arcseconds

    slit_angle : float
        Position angle of the pseudoslit

    pxs : float
        Pixelscale of the data cube in arcseconds/pixel

    vx : 1D array
        Values of the spectral axis. This array must have the same length as the
        first dimension of `cube`.

    soff : float, optional
        Offset of the slit from center in arcseconds. Default is 0.

    Returns
    -------
    xvec : 1D array
        Position along slit in arcseconds

    flux : 1D array
        Relative flux of the line at each position. Calculated as the sum of the spectrum.

    vel : 1D array
        Velocity at each position in same units as given by `vx`. Calculated as the first moment
        of the spectrum.

    disp : 1D array
        Velocity dispersion at each position in the same units as given by `vx`. Calculated as the
        second moment of the spectrum.

    """
    cube_shape = cube.shape
    psize = cube_shape[1]
    vsize = cube_shape[0]
    lin = np.arange(psize) - np.fix(psize/2.)
    veldata = scp_ndi.interpolation.rotate(cube, slit_angle, axes=(2, 1),
                                           reshape=False)
    tmpn = (((lin*pxs) <= (soff+slit_width/2.)) &
            ((lin*pxs) >= (soff-slit_width/2.)))
    data = np.zeros((psize, vsize))

    flux = np.zeros(psize)

    yvec = vx
    xvec = lin*pxs

    for i in range(psize):
        for j in range(vsize):
            data[i, j] = np.mean(veldata[j, i, tmpn])
        flux[i] = np.sum(data[i,:])

    flux = flux / np.max(flux) * 10.
    pvec = (flux < 0.)

    vel = np.zeros(psize)
    disp = np.zeros(psize)
    for i in range(psize):
        vel[i] = np.sum(data[i,:]*yvec)/np.sum(data[i,:])
        disp[i] = np.sqrt( np.sum( ((yvec-vel[i])**2) * data[i,:]) / np.sum(data[i,:]) )

    if np.sum(pvec) > 0.:
        vel[pvec] = -1.e3
        disp[pvec] = 0.

    return xvec, flux, vel, disp


def calc_1dprofile_circap_pv(cube, slit_width, slit_angle, pxs, vx, soff=0.):
    """
    Measure the 1D rotation curve from a cube using a pseudoslit

    This function measures the 1D rotation curve by first creating a PV diagram based on the
    input slit properties. Fluxes, velocities, and dispersions are then measured from spectra
    produced by integrating over circular apertures placed on the PV diagram with radii equal
    to 0.5*`slit_width`. The 0th, 1st, and 2nd moments of the integrated spectra are then calculated
    to determine the flux, velocity, and dispersion.

    Parameters
    ----------
    cube : 3D array
        Data cube from which to measure the rotation curve. First dimension is assumed to
        be spectral direction.

    slit_width : float
        Slit width of the pseudoslit in arcseconds

    slit_angle : float
        Position angle of the pseudoslit

    pxs : float
        Pixelscale of the data cube in arcseconds/pixel

    vx : 1D array
        Values of the spectral axis. This array must have the same length as the
        first dimension of `cube`.

    soff : float, optional
        Offset of the slit from center in arcseconds. Default is 0.

    Returns
    -------
    xvec : 1D array
        Position along slit in arcseconds

    flux : 1D array
        Relative flux of the line at each position. Calculated as the sum of the spectrum.

    vel : 1D array
        Velocity at each position in same units as given by `vx`. Calculated as the first moment
        of the spectrum.

    disp : 1D array
        Velocity dispersion at each position in the same units as given by `vx`. Calculated as the
        second moment of the spectrum.

    """
    cube_shape = cube.shape
    psize = cube_shape[1]
    vsize = cube_shape[0]
    lin = np.arange(psize) - np.fix(psize/2.)
    veldata = scp_ndi.interpolation.rotate(cube, slit_angle, axes=(2, 1),
                                           reshape=False)
    tmpn = (((lin*pxs) <= (soff+slit_width/2.)) &
            ((lin*pxs) >= (soff-slit_width/2.)))
    data = np.zeros((psize, vsize))
    flux = np.zeros(psize)

    yvec = vx
    xvec = lin*pxs

    for i in range(psize):
        for j in range(vsize):
            data[i, j] = np.mean(veldata[j, i, tmpn])
        tmp = data[i]
        flux[i] = np.sum(tmp)

    flux = flux / np.max(flux) * 10.
    pvec = (flux < 0.)

    # Calculate circular segments
    rr = 0.5 * slit_width
    pp = pxs

    nslice = np.int(1 + 2 * np.ceil((rr - 0.5 * pp) / pp))

    circaper_idx = np.arange(nslice) - 0.5 * (nslice - 1)
    circaper_sc = np.zeros(nslice)

    circaper_sc[int(0.5*nslice - 0.5)] = (np.pi*rr**2 -
                                          2.*area_segm(rr, 0.5*pp))

    if nslice > 1:
        circaper_sc[0] = area_segm(rr, (0.5*nslice - 1)*pp)
        circaper_sc[nslice-1] = circaper_sc[0]

    if nslice > 3:
        for cnt in range(1, int(0.5*(nslice-3))+1):
            circaper_sc[cnt] = (area_segm(rr, (0.5*nslice - 1. - cnt)*pp) -
                                area_segm(rr, (0.5*nslice - cnt)*pp))
            circaper_sc[nslice-1-cnt] = circaper_sc[cnt]

    circaper_vel = np.zeros(psize)
    circaper_disp = np.zeros(psize)
    circaper_flux = np.zeros(psize)

    nidx = len(circaper_idx)
    for i in range(psize):
        tot_vnum = 0.
        tot_denom = 0.
        cnt_idx = 0
        cnt_start = int(i + circaper_idx[0]) if (i + circaper_idx[0]) > 0 else 0
        cnt_end = (int(i + circaper_idx[nidx-1]) if (i + circaper_idx[nidx-1]) <
                                                    (psize-1) else (psize-1))
        for cnt in range(cnt_start, cnt_end+1):
            tmp = data[cnt]
            tot_vnum += circaper_sc[cnt_idx] * np.sum(tmp*yvec)
            tot_denom += circaper_sc[cnt_idx] * np.sum(tmp)
            cnt_idx = cnt_idx + 1

        circaper_vel[i] = tot_vnum / tot_denom
        circaper_flux[i] = tot_denom

        tot_dnum = 0.
        cnt_idx = 0
        for cnt in range(cnt_start, cnt_end+1):
            tmp = data[cnt]
            tot_dnum = (tot_dnum + circaper_sc[cnt_idx] *
                        np.sum(tmp*(yvec-circaper_vel[i])**2))
            cnt_idx = cnt_idx + 1

        circaper_disp[i] = np.sqrt(tot_dnum / tot_denom)

    if np.sum(pvec) > 0.:
        circaper_vel[pvec] = -1.e3
        circaper_disp[pvec] = 0.
        circaper_flux[pvec] = 0.

    return xvec, circaper_flux, circaper_vel, circaper_disp

# ############################################################################
# def tie_r_fdm(model_set):
#     return model_set.components['disk+bulge'].r_eff_disk.value

############################################################################

# Generic model container which tracks all components, parameters,
# parameter settings, model settings, etc.
class ModelSet:
    """
    Object that contains all model components, parameters, and settings

    `ModelSet` does not take any arguments. Instead, it first should be initialized
    and then :meth:`ModelSet.add_component` can be used to include specific model components.
    All included components can then be accessed through `ModelSet.components` which
    is a dictionary that has keys equal to the names of each component. The primary method
    of `ModelSet` is :meth:`ModelSet.simulate_cube` which produces a model data cube of
    line emission that follows the full kinematics of given model.
    """
    def __init__(self):

        self.mass_components = OrderedDict()
        self.components = OrderedDict()
        self.light_components = OrderedDict()
        self.geometry = None
        self.dispersion_profile = None
        self.zprofile = None
        self.outflow = None
        self.outflow_geometry = None
        self.outflow_dispersion = None
        self.outflow_flux = None
        self.inflow = None
        self.inflow_geometry = None
        self.inflow_dispersion = None
        self.inflow_flux = None
        self.extinction = None
        self.parameters = None
        self.fixed = OrderedDict()
        self.tied = OrderedDict()
        self.param_names = OrderedDict()
        self._param_keys = OrderedDict()
        self.nparams = 0
        self.nparams_free = 0
        self.nparams_tied = 0
        self.kinematic_options = KinematicOptions()
        self.line_center = None

        # Option for dealing with 3D data:
        self.per_spaxel_norm_3D = False

    def add_component(self, model, name=None, light=False, geom_type='galaxy',
                      disp_type='galaxy'):
        """
        Add a model component to the set

        Parameters
        ----------
        model : `~dysmalpy.models._DysmalModel`
            Model component to be added to the model set

        name : str
            Name of the model component

        light : bool
            If True, use the mass profile of the model component in the calculation of the
            flux of the line, i.e. setting the mass-to-light ratio equal to 1.

        geom_type : {'galaxy', 'outflow', 'inflow'}
            Specify which model components the geometry applies to.
            Only used if `model` is a `~Geometry`. If 'galaxy', then all included
            components except outflows and inflows will follow this geometry.
            Default is 'galaxy'.

        disp_type : {'galaxy', 'outflow', 'inflow'}
            Specify which model components the dispersion applies to.
            Only used if `model` is a `~DispersionProfile`. Default is 'galaxy'.
        """
        # Check to make sure its the correct class
        if isinstance(model, _DysmalModel):

            # Check to make sure it has a name
            if (name is None) & (model.name is None):
                raise ValueError('Please give this component a name!')

            elif name is not None:
                model = model.rename(name)

            if model._type == 'mass':

                # Make sure there isn't a mass component already named this
                if list(self.mass_components.keys()).count(model.name) > 0:
                    raise ValueError('Component already exists. Please give'
                                     'it a unique name.')
                else:
                    self.mass_components[model.name] = True

            elif model._type == 'geometry':

                if geom_type == 'galaxy':
                    if (self.geometry is not None):
                        logger.warning('Current Geometry model is being '
                                    'overwritten!')
                    self.geometry = model

                elif geom_type == 'outflow':

                    self.outflow_geometry = model

                elif geom_type == 'inflow':

                    self.inflow_geometry = model

                else:
                    logger.error("geom_type can only be either 'galaxy', "
                                 "'inflow', or 'outflow'.")

                self.mass_components[model.name] = False

            elif model._type == 'dispersion':

                if disp_type == 'galaxy':
                    if self.dispersion_profile is not None:
                        logger.warning('Current Dispersion model is being '
                                       'overwritten!')
                    self.dispersion_profile = model

                elif disp_type == 'outflow':

                    self.outflow_dispersion = model

                elif disp_type == 'inflow':

                    self.inflow_dispersion = model

                self.mass_components[model.name] = False

            elif model._type == 'zheight':
                if self.zprofile is not None:
                    logger.warning('Current z-height model is being '
                                   'overwritten!')
                self.zprofile = model
                self.mass_components[model.name] = False

            elif model._type == 'outflow':
                if self.outflow is not None:
                    logger.warning('Current outflow model is being '
                                   'overwritten!')
                self.outflow = model
                self.mass_components[model.name] = False

            elif model._type == 'inflow':
                if self.inflow is not None:
                    logger.warning('Current inflow model is being '
                                   'overwritten!')
                self.inflow = model
                self.mass_components[model.name] = False

            elif model._type == 'extinction':
                if self.extinction is not None:
                    logger.warning('Current extinction model is being overwritten!')
                self.extinction = model
                self.mass_components[model.name] = False

            else:
                raise TypeError("This model type is not known. Must be one of"
                                "'mass', 'geometry', 'dispersion', 'zheight',"
                                "'outflow', 'inflow', or 'extinction'.")

            if light:
                self.light_components[model.name] = True
            else:
                self.light_components[model.name] = False

            self._add_comp(model)

        else:

            raise TypeError('Model component must be a '
                            'dysmalpy.models.DysmalModel instance!')

    def _add_comp(self, model):
        """
        Update the `ModelSet` parameters with new model component

        Parameters
        ----------
        model : `~dysmalpy.models._DysmalModel`
            Model component to be added to the model set

        """

        # Update the components list
        self.components[model.name] = model

        # Update the parameters and parameters_free arrays
        if self.parameters is None:
            self.parameters = model.parameters
        else:

            self.parameters = np.concatenate([self.parameters,
                                              model.parameters])
        self.param_names[model.name] = model.param_names
        self.fixed[model.name] = model.fixed
        self.tied[model.name] = model.tied

        # Update the dictionaries containing the locations of the parameters
        # in the parameters array. Also count number of tied parameters
        key_dict = OrderedDict()
        ntied = 0
        for i, p in enumerate(model.param_names):
            key_dict[p] = i + self.nparams
            if model.tied[p]:
                ntied += 1

        self._param_keys[model.name] = key_dict
        self.nparams += len(model.param_names)
        self.nparams_free += (len(model.param_names) - sum(model.fixed.values())
                              - ntied)
        self.nparams_tied += ntied

    def set_parameter_value(self, model_name, param_name, value):
        """
        Change the value of a specific parameter

        Parameters
        ----------
        model_name : str
            Name of the model component the parameter belongs to.

        param_name : str
            Name of the parameter

        value : float
            Value to change the parameter to
        """

        try:
            comp = self.components[model_name]
        except KeyError:
            raise KeyError('Model not part of the set.')

        try:
            param_i = comp.param_names.index(param_name)
        except ValueError:
            raise ValueError('Parameter is not part of model.')

        self.components[model_name].__getattribute__(param_name).value = value
        self.parameters[self._param_keys[model_name][param_name]] = value

    def set_parameter_fixed(self, model_name, param_name, fix):
        """
        Change whether a specific parameter is fixed or not

        Parameters
        ----------
        model_name : str
            Name of the model component the parameter belongs to.

        param_name : str
            Name of the parameter

        fix : bool
            If True, the parameter will be fixed to its current value. If False, it will
            be a free parameter allowed to vary during fitting.
        """

        try:
            comp = self.components[model_name]
        except KeyError:
            raise KeyError('Model not part of the set.')

        try:
            param_i = comp.param_names.index(param_name)
        except ValueError:
            raise ValueError('Parameter is not part of model.')

        self.components[model_name].fixed[param_name] = fix
        self.fixed[model_name][param_name] = fix
        if fix:
            self.nparams_free -= 1
        else:
            self.nparams_free += 1

    def update_parameters(self, theta):
        """
        Update all of the free and tied parameters of the model

        Parameters
        ----------
        theta : array with length = `ModelSet.nparams_free`
            New values for the free parameters

        Notes
        -----
        The order of the values in `theta` is important.
        Use :meth:`ModelSet.get_free_parameter_keys` to determine the correct order.
        """

        # Sanity check to make sure the array given is the right length
        if len(theta) != self.nparams_free:
            raise ValueError('Length of theta is not equal to number '
                             'of free parameters, {}'.format(self.nparams_free))

        # Loop over all of the parameters
        pfree, pfree_keys = self._get_free_parameters()
        for cmp in pfree_keys:
            for pp in pfree_keys[cmp]:
                ind = pfree_keys[cmp][pp]
                if ind != -99:
                    self.set_parameter_value(cmp, pp, theta[ind])

        # Now update all of the tied parameters if there are any
        if self.nparams_tied > 0:
            for cmp in self.tied:
                for pp in self.tied[cmp]:
                    if self.tied[cmp][pp]:
                        new_value = self.tied[cmp][pp](self)
                        self.set_parameter_value(cmp, pp, new_value)

    # Methods to grab the free parameters and keys
    def _get_free_parameters(self):
        """
        Return the current values and indices of the free parameters

        Returns
        -------
        p : array
            Values of the free parameters

        pkeys : dictionary
            Dictionary of all model components with their parameters. If a model
            parameter is free, then it lists its index within `p`. Otherwise, -99.
        """
        p = np.zeros(self.nparams_free)
        pkeys = OrderedDict()
        j = 0
        for cmp in self.fixed:
            pkeys[cmp] = OrderedDict()
            for pm in self.fixed[cmp]:
                if self.fixed[cmp][pm] | np.bool(self.tied[cmp][pm]):
                    pkeys[cmp][pm] = -99
                else:
                    pkeys[cmp][pm] = j
                    p[j] = self.parameters[self._param_keys[cmp][pm]]
                    j += 1
        return p, pkeys

    def get_free_parameters_values(self):
        """
        Return the current values of the free parameters

        Returns
        -------
        pfree : array
            Values of the free parameters
        """
        pfree, pfree_keys = self._get_free_parameters()
        return pfree

    def get_free_parameter_keys(self):
        """
        Return the index within an array of each free parameter

        Returns
        -------
        pfree_keys : dictionary
            Dictionary of all model components with their parameters. If a model
            parameter is free, then it lists its index within `p`. Otherwise, -99.
        """
        pfree, pfree_keys = self._get_free_parameters()
        return pfree_keys

    def get_log_prior(self):
        """
        Return the total log prior based on current values

        Returns
        -------
        log_prior_model : float
            Summed log prior
        """
        log_prior_model = 0.
        pfree_dict = self.get_free_parameter_keys()
        comps_names = pfree_dict.keys()
        for compn in comps_names:
            comp = self.components.__getitem__(compn)
            params_names = pfree_dict[compn].keys()
            for paramn in params_names:
                if pfree_dict[compn][paramn] >= 0:
                    # Free parameter: add to total prior
                    log_prior_model += comp.__getattribute__(paramn).prior.log_prior(comp.__getattribute__(paramn), modelset=self)
        return log_prior_model

    def get_dm_aper(self, r):
        """
        Calculate the enclosed dark matter fraction

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc within which to calculate the dark matter fraction.
            Assumes a `DarkMatterHalo` component is included in the `ModelSet`.

        Returns
        -------
        dm_frac: array
            Enclosed dark matter fraction at `r`
        """
        vc, vdm = self.circular_velocity(r, compute_dm=True)
        dm_frac = vdm**2/vc**2
        return dm_frac

    def get_dm_frac_effrad(self, model_key_re=['disk+bulge', 'r_eff_disk']):
        """
        Calculate the dark matter fraction within the effective radius

        Parameters
        ----------
        model_key_re : list
            Two element list which contains the name of the model component
            and parameter to use for the effective radius.
            Default is ['disk+bulge', 'r_eff_disk'].

        Returns
        -------
        dm_frac : float
            Dark matter fraction within the specified effective radius
        """
        # RE needs to be in kpc
        comp = self.components.__getitem__(model_key_re[0])
        param_i = comp.param_names.index(model_key_re[1])
        r_eff = comp.parameters[param_i]
        dm_frac = self.get_dm_aper(r_eff)

        return dm_frac

    def get_mvirial(self, model_key_halo=['halo']):
        """
        Return the virial mass of the dark matter halo component

        Parameters
        ----------
        model_key_halo : list
            One element list with the name of the halo model component

        Returns
        -------
        mvir : float
            Virial mass of the dark matter halo in log(Msun)
        """
        comp = self.components.__getitem__(model_key_halo[0])
        try:
            mvir = comp.mvirial.value
        except:
            mvir = comp.mvirial

        return mvir

    def get_halo_alpha(self, model_key_halo=['halo']):
        """
        Return the alpha parameter value for a `TwoPowerHalo`

        Parameters
        ----------
        model_key_halo : list
            One element list with the name of the `TwoPowerHalo` model component

        Returns
        -------
        alpha : float or None
            Value of the alpha parameter. Returns None if the correct component
            does not exist.
        """
        comp = self.components.__getitem__(model_key_halo[0])
        try:
            return comp.alpha.value
        except:
            return None

    def get_halo_rb(self, model_key_halo=['halo']):
        """
        Return the Burkert radius parameter value for a `Burkert` dark matter halo

        Parameters
        ----------
        model_key_halo : list
            One element list with the name of the `Burkert` model component

        Returns
        -------
        rb : float or None
            Value of the Burkert radius. Returns None if the correct component
            does not exist.
        """
        comp = self.components.__getitem__(model_key_halo[0])
        try:
            return comp.rB.value
        except:
            return None

    def get_encl_mass_effrad(self, model_key_re=['disk+bulge', 'r_eff_disk']):
        """
        Calculate the total enclosed mass within the effective radius

        Parameters
        ----------
        model_key_re : list
            Two element list which contains the name of the model component
            and parameter to use for the effective radius.
            Default is ['disk+bulge', 'r_eff_disk'].

        Returns
        -------
        menc : float
            Total enclosed mass within the specified effective radius

        Notes
        -----
        This method uses the total circular velocity to determine the enclosed mass
        based on v^2 = GM/r.
        """

        comp = self.components.__getitem__(model_key_re[0])
        param_i = comp.param_names.index(model_key_re[1])
        r_eff = comp.parameters[param_i]
        r = r_eff

        vc, vdm = self.circular_velocity(r, compute_dm=True)
        menc = menc_from_vcirc(vc, r_eff)

        return menc

    def enclosed_mass(self, r, model_key_re=['disk+bulge', 'r_eff_disk'], step1d=0.2):
        """
        Calculate the total enclosed mass

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate the enclosed mass in kpc

        model_key_re : list, optional
            Two element list which contains the name of the model component
            and parameter to use for the effective radius. Only necessary
            if adiabatic contraction is used. Default is ['disk+bulge', 'r_eff_disk'].

        step1d : float, optional
            Step size in kpc to use during adiabatic contraction calculation

        Returns
        -------
        enc_mass, enc_bary, enc_dm : float or array
            The total, baryonic, and dark matter enclosed masses
        """

        # First check to make sure there is at least one mass component in the
        # model set.
        if len(self.mass_components) == 0:
            raise AttributeError("There are no mass components so an enclosed "
                                 "mass can't be calculated.")
        else:

            enc_mass = r*0.
            enc_dm = r*0.
            enc_bary = r*0.

            for cmp in self.mass_components:
                if self.mass_components[cmp]:
                    mcomp = self.components[cmp]
                    enc_mass_cmp = mcomp.enclosed_mass(r)
                    enc_mass += enc_mass_cmp

                    if mcomp._subtype == 'dark_matter':

                        enc_dm += enc_mass_cmp

                    elif mcomp._subtype == 'baryonic':

                        enc_bary += enc_mass_cmp

            if (np.sum(enc_dm) > 0) & self.kinematic_options.adiabatic_contract:

                vcirc, vhalo_adi = self.circular_velocity(r, compute_dm=True, model_key_re=model_key_re, step1d=step1d)
                # enc_dm_adi = ((vhalo_adi*1e5)**2.*(r*1000.*pc.cgs.value) /
                #               (G.cgs.value * Msun.cgs.value))
                enc_dm_adi = menc_from_vcirc(vhalo_adi, r)
                enc_mass = enc_mass - enc_dm + enc_dm_adi
                enc_dm = enc_dm_adi

        return enc_mass, enc_bary, enc_dm

    def get_dlnrhotot_dlnr(self, r):
        """
        Calculate the composite derivative dln(rho,tot) / dlnr

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        dlnrhotot_dlnr : float or array

        """

        # First check to make sure there is at least one mass component in the
        # model set.
        if len(self.mass_components) == 0:
            raise AttributeError("There are no mass components so a dlnrho/dlnr "
                                 "can't be calculated.")
        else:
            rhotot = r*0.
            drho_dr = r*0.

            for cmp in self.mass_components:

                if self.mass_components[cmp]:
                    mcomp = self.components[cmp]

                    cmpnt_rho = mcomp.rho(r)
                    cmpnt_drhodr = mcomp.drho_dr(r)

                    whfin = np.where(np.isfinite(cmpnt_drhodr))[0]
                    if len(whfin) < len(r):
                        # FLAG42
                        raise ValueError

                    rhotot = rhotot + cmpnt_rho
                    drho_dr = drho_dr + cmpnt_drhodr

        dlnrhotot_dlnr = r / rhotot * (drho_dr)

        ## FLAG42
        #raise ValueError

        return dlnrhotot_dlnr

    def circular_velocity(self, r, compute_dm=False, model_key_re=['disk+bulge', 'r_eff_disk'],
                          step1d=0.2):
        """
        Calculate the total circular velocity as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate the circular velocity in kpc

        compute_dm : bool
            If True, also return the circular velocity due to the halo

        model_key_re : list, optional
            Two element list which contains the name of the model component
            and parameter to use for the effective radius.
            Default is ['disk+bulge', 'r_eff_disk'].

        step1d : float, optional
            Step size in kpc to use during adiabatic contraction calculation

        Returns
        -------
        vel : float or array
            Total circular velocity in km/s

        vdm : float or array, only if `compute_dm` = True
            Circular velocity due to the halo
        """

        # First check to make sure there is at least one mass component in the
        # model set.
        if len(self.mass_components) == 0:
            raise AttributeError("There are no mass components so a velocity "
                                 "can't be calculated.")
        else:
            vdm = r*0.
            vbaryon = r*0.

            for cmp in self.mass_components:

                if self.mass_components[cmp]:
                    mcomp = self.components[cmp]

                    if isinstance(mcomp, DiskBulge) | isinstance(mcomp, LinearDiskBulge):
                        cmpnt_v = mcomp.circular_velocity(r)
                    else:
                        cmpnt_v = mcomp.circular_velocity(r)
                    if (mcomp._subtype == 'dark_matter') | (mcomp._subtype == 'combined'):

                        vdm = np.sqrt(vdm ** 2 + cmpnt_v ** 2)

                    elif mcomp._subtype == 'baryonic':

                        vbaryon = np.sqrt(vbaryon ** 2 + cmpnt_v ** 2)

                    else:
                        raise TypeError("{} mass model subtype not recognized"
                                        " for {} component. Only 'dark_matter'"
                                        " or 'baryonic' accepted.".format(
                                        mcomp._subtype, cmp))
            vels = self.kinematic_options.apply_adiabatic_contract(self, r, vbaryon, vdm,
                                                                   compute_dm=compute_dm,
                                                                   model_key_re=model_key_re,
                                                                   step1d=step1d)

            if compute_dm:
                vel = vels[0]
                vdm = vels[1]
            else:
                vel = vels
            if compute_dm:
                return vel, vdm
            else:
                return vel

    def velocity_profile(self, r, compute_dm=False):
        """
        Calculate the rotational velocity as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate the velocity in kpc

        compute_dm : bool
            If True also return the circular velocity due to the dark matter halo

        Returns
        -------
        vel : float or array
            Rotational velocity as a function of radius in km/s

        vdm : float or array
            Circular velocity due to the dark matter halo in km/s
            Only returned if `compute_dm` = True
       """
        vels = self.circular_velocity(r, compute_dm=compute_dm)
        if compute_dm:
            vcirc = vels[0]
            vdm = vels[1]
        else:
            vcirc = vels

        vel = self.kinematic_options.apply_pressure_support(r, self, vcirc)

        if compute_dm:
            return vel, vdm
        else:
            return vel

    def get_vmax(self, r=None):
        """
        Calculate the peak velocity of the rotation curve

        Parameters
        ----------
        r : array, optional
            Radii to sample to find the peak. If None, then a linearly
            spaced array from 0 to 25 kpc with 251 points will be used

        Returns
        -------
        vmax : float
            Peak velocity of the rotation curve in km/s

        Notes
        -----
        This simply finds the maximum of the rotation curve which is calculated at discrete
        radii, `r`.

        """
        if r is None:
            r = np.linspace(0., 25., num=251, endpoint=True)

        vel = self.velocity_profile(r, compute_dm=False)

        vmax = vel.max()
        return vmax

    def write_vrot_vcirc_file(self, r=None, filename=None, overwrite=False):
        """
        Output the rotational and circular velocities to a file

        Parameters
        ----------
        r : array, optional
            Radii to sample to find the peak. If None, then a linearly
            spaced array from 0 to 25 kpc with 251 points will be used

        filename : str, optional
            Name of file to output velocities to. Default is 'vout.txt'
        """
        # Check for existing file:
        if (not overwrite) and (filename is not None):
            if os.path.isfile(filename):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
                return None

        # Quick test for if vcirc defined:
        coltry = ['velocity_profile', 'circular_velocity']
        coltrynames = ['vrot', 'vcirc']
        coltryunits = ['[km/s]', '[km/s]']
        cols = []
        colnames = []
        colunits = []
        for c, cn, cu in zip(coltry, coltrynames, coltryunits):
            try:
                fnc = getattr(self, c)
                tmp = fnc(np.array([2.]))
                cols.append(c)
                colnames.append(cn)
                colunits.append(cu)
            except:
                pass

        if len(cols) >= 1:

            self.write_profile_file(r=r, filename=filename,
                cols=cols, prettycolnames=colnames, colunits=colunits, overwrite=overwrite)


    def write_profile_file(self, r=None, filename=None,
            cols=None, prettycolnames=None, colunits=None, overwrite=False):
        """
        Output various radial profiles of the `ModelSet`

        Parameters
        ----------
        r: array, optional
            Radii to sample to find the peak. If None, then a linearly
            spaced array from 0 to 10 kpc with a stepsize of 0.1 will be used

        filename: str, optional
            Output filename to write to. Will be written as ascii, w/ space delimiter.
            Default is 'rprofiles.txt'

        cols: list, optional
            Names of ModelSet methods that will be called as function of r,
            and to be saved as a column in the output file.
            Default is ['velocity_profile', 'circular_velocity', 'get_dm_aper'].

        prettycolnames:  list, optional
            Alternate column names for output in file header (eg, 'vrot' not 'velocity_profile')
            Default is `cols`.

        colunits: list, optional
            Units of each column. r is added by hand, and will always be in kpc.
        """
        # Check for existing file:
        if (not overwrite) and (filename is not None):
            if os.path.isfile(filename):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
                return None

        if cols is None:              cols = ['velocity_profile', 'circular_velocity', 'get_dm_aper']
        if prettycolnames is None:    prettycolnames = cols
        if r is None:                 r = np.arange(0., 10.+0.1, 0.1)  # stepsize 0.1 kpc

        profiles = np.zeros((len(r), len(cols)+1))
        profiles[:,0] = r
        for j in six.moves.xrange(len(cols)):
            try:
                fnc = getattr(self, cols[j])
                arr = fnc(r)
            except:
                arr = np.ones(len(r))*-99.
            profiles[:, j+1] = arr

        colsout = ['r']
        colsout.extend(prettycolnames)
        if colunits is not None:
            unitsout = ['kpc']
            unitsout.extend(colunits)

        with open(filename, 'w') as f:
            namestr = '#   ' + '   '.join(colsout)
            f.write(namestr+'\n')
            if colunits is not None:
                unitstr = '#   ' + '   '.join(unitsout)
                f.write(unitstr+'\n')
            for i in six.moves.xrange(len(r)):
                datstr = '    '.join(["{0:0.3f}".format(p) for p in profiles[i,:]])
                f.write(datstr+'\n')


    def simulate_cube(self, nx_sky, ny_sky, dscale, rstep,
                      spec_type, spec_step, spec_start, nspec,
                      spec_unit=u.km/u.s, oversample=1, oversize=1,
                      xcenter=None, ycenter=None,
                      zcalc_truncate=True):
                      #debug=False):
        """
        Simulate a line emission cube of this model set

        Parameters
        ----------
        nx_sky : int
            Number of pixels in the output cube in the x-direction

        ny_sky : int
            Number of pixels in the output cube in the y-direction

        dscale : float
            Conversion from sky to physical coordinates in arcsec/kpc

        rstep : float
            Pixel scale in arsec/pixel

        spec_type : {'velocity', 'wavelength'}
            Spectral axis type.

        spec_step : float
            Step size of the spectral axis

        spec_start : float
            Value of the first element of the spectral axis

        nspec : int
            Number of spectral channels

        spec_unit : `~astropy.units.Unit`
            Unit of the spectral axis

        oversample : int, optional
            Oversampling factor for creating the model cube. If `oversample` > 1, then
            the model cube will be generated at `rstep`/`oversample` pixel scale.

        oversize : int, optional
            Oversize factor for creating the model cube. If `oversize` > 1, then the model
            cube will be generated with `oversize`*`nx_sky` and `oversize`*`ny_sky`
            number of pixels in the x and y direction respectively.

        xcenter : float, optional
            The x-coordinate of the center of the galaxy. If None then the x-coordinate of the
            center of the cube will be used.

        ycenter : float, optional
            The y-coordinate of the center of the galaxy. If None then the x-coordinate of the
            center of the cube will be used.

        zcalc_truncate: bool
            Setting the default behavior of filling the model cube. If True,
            then the cube is only filled with flux
            to within +- 2 * scale length thickness above and below
            the galaxy midplane (minimum: 3 whole pixels; to speed up the calculation).
            If False, then no truncation is
            applied and the cube is filled over the full range of zgal.
            Default: True

        Returns
        -------
        cube_final : 3D array
            Line emission cube that incorporates all of the kinematics due to the components
            of the current `ModelSet`

        spec : 1D array
            Values of the spectral channels as determined by `spec_type`, `spec_start`,
            `spec_step`, `nspec`, and `spec_unit`
        """
        # Start with a 3D array in the sky coordinate system
        # x and y sizes are user provided so we just need
        # the z size where z is in the direction of the L.O.S.
        # We'll just use the maximum of the given x and y

        ### Don't need, just do later for {x/y}center_samp
        # if xcenter is None:
        #     xcenter = (nx_sky - 1) / 2.
        # if ycenter is None:
        #     ycenter = (nx_sky - 1) / 2.

        # Backwards compatibility:
        if 'outflow' not in self.__dict__.keys():
            self.outflow = None
            self.outflow_geometry = None
            self.outflow_dispersion = None
            self.outflow_flux = None
        if 'inflow' not in self.__dict__.keys():
            self.inflow = None
            self.inflow_geometry = None
            self.inflow_dispersion = None
            self.inflow_flux = None

        nx_sky_samp = nx_sky*oversample*oversize
        ny_sky_samp = ny_sky*oversample*oversize
        rstep_samp = rstep/oversample

        if (np.mod(nx_sky, 2) == 1) & (np.mod(oversize, 2) == 0) & (oversize > 1):
            nx_sky_samp = nx_sky_samp + 1

        if (np.mod(ny_sky, 2) == 1) & (np.mod(oversize, 2) == 0) & (oversize > 1):
            ny_sky_samp = ny_sky_samp + 1

        if xcenter is None:
            xcenter_samp = (nx_sky_samp - 1) / 2.
        else:
            xcenter_samp = (xcenter + 0.5)*oversample - 0.5
        if ycenter is None:
            ycenter_samp = (ny_sky_samp - 1) / 2.
        else:
            ycenter_samp = (ycenter + 0.5)*oversample - 0.5

        #nz_sky_samp = np.max([nx_sky_samp, ny_sky_samp])

        # Setup the final IFU cube
        spec = np.arange(nspec) * spec_step + spec_start
        if spec_type == 'velocity':
            vx = (spec * spec_unit).to(u.km / u.s).value
        elif spec_type == 'wavelength':
            if self.line_center is None:
                raise ValueError("line_center must be provided if spec_type is "
                                 "'wavelength.'")
            line_center_conv = self.line_center.to(spec_unit).value
            vx = (spec - line_center_conv) / line_center_conv * apy_con.c.to(
                u.km / u.s).value

        cube_final = np.zeros((nspec, ny_sky_samp, nx_sky_samp))


        # First construct the cube based on mass components
        if sum(self.mass_components.values()) > 0:

            # Create 3D arrays of the sky pixel coordinates
            cos_inc = np.cos(self.geometry.inc*np.pi/180.)
            maxr = np.sqrt(nx_sky_samp**2 + ny_sky_samp**2)
            maxr_y = np.max(np.array([maxr*1.5, np.min(
                np.hstack([maxr*1.5/ cos_inc, maxr * 5.]))]))
            #nz_sky_samp = np.int(np.max([nx_sky_samp, ny_sky_samp, maxr_y]))
            nz_sky_samp = np.int(np.max([nx_sky_samp, ny_sky_samp]))
            if np.mod(nz_sky_samp, 2) < 0.5:
                nz_sky_samp += 1

            sh = (nz_sky_samp, ny_sky_samp, nx_sky_samp)
            zsky, ysky, xsky = np.indices(sh)
            zsky = zsky - (nz_sky_samp - 1) / 2.
            ysky = ysky - ycenter_samp # (ny_sky_samp - 1) / 2.
            xsky = xsky - xcenter_samp # (nx_sky_samp - 1) / 2.

            # Apply the geometric transformation to get galactic coordinates
            # Need to account for oversampling in the x and y shift parameters
            self.geometry.xshift = self.geometry.xshift.value * oversample
            self.geometry.yshift = self.geometry.yshift.value * oversample
            xgal, ygal, zgal = self.geometry(xsky, ysky, zsky)

            # The circular velocity at each position only depends on the radius
            # Convert to kpc
            rgal = np.sqrt(xgal ** 2 + ygal ** 2) * rstep_samp / dscale
            vrot = self.velocity_profile(rgal)
            # L.O.S. velocity is then just vrot*sin(i)*cos(theta) where theta
            # is the position angle in the plane of the disk
            # cos(theta) is just xgal/rgal
            v_sys = self.geometry.vel_shift.value  # systemic velocity
            vobs_mass = v_sys + (vrot * np.sin(np.radians(self.geometry.inc.value)) *
                    xgal / (rgal / rstep_samp * dscale))

            vobs_mass[rgal == 0] = 0.

            #######
            if ((self.inflow is not None) & (self.inflow_geometry is None)):
                rgal3D = np.sqrt(xgal ** 2 + ygal ** 2 + zgal **2)
                # 3D radius, converted to kpc
                rgal3D_kpc = rgal3D * rstep_samp / dscale
                xgal_kpc = xgal * rstep_samp / dscale
                ygal_kpc = ygal * rstep_samp / dscale
                zgal_kpc = ygal * rstep_samp / dscale
                vin = self.inflow(xgal_kpc, ygal_kpc, zgal_kpc)

                # Negative of inflow is already included in the definition of the inflow
                #   No systemic velocity here bc this is relative to the center of the galaxy at rest already
                vin_obs = - vin * zsky/rgal3D
                vin_obs[rgal3D == 0] = vin[rgal3D == 0]
                vobs_mass += vin_obs
            #######

            # Calculate "flux" for each position

            flux_mass = np.zeros(vobs_mass.shape)

            for cmp in self.light_components:
                if self.light_components[cmp]:
                    zscale = self.zprofile(zgal * rstep_samp / dscale)
                    flux_mass += self.components[cmp].mass_to_light(rgal) * zscale

            # Apply extinction if a component exists
            if self.extinction is not None:

                flux_mass *= self.extinction(xsky, ysky, zsky)

            # The final spectrum will be a flux weighted sum of Gaussians at each
            # velocity along the line of sight.
            sigmar = self.dispersion_profile(rgal)

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if zcalc_truncate:
                ##print("    models.py: TRUE: zcalc_truncate={}".format(zcalc_truncate))
                # Truncate in the z direction by flagging what pixels to include in propogation
                ai = _make_cube_ai(self, xgal, ygal, zgal, rstep=rstep_samp, oversample=oversample,
                    dscale=dscale, maxr=maxr/2., maxr_y=maxr_y/2.)
                cube_final += cutils.populate_cube_ais(flux_mass, vobs_mass, sigmar, vx, ai)
            else:
                ##print("    models.py: FALSE: zcalc_truncate={}".format(zcalc_truncate))
                # Do complete cube propogation calculation
                cube_final += cutils.populate_cube(flux_mass, vobs_mass, sigmar, vx)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            self.geometry.xshift = self.geometry.xshift.value / oversample
            self.geometry.yshift = self.geometry.yshift.value / oversample

        if self.outflow is not None:

            if self.outflow._spatial_type == 'resolved':
                # Create 3D arrays of the sky pixel coordinates
                sin_inc = np.sin(self.outflow_geometry.inc * np.pi / 180.)
                maxr = np.sqrt(nx_sky_samp ** 2 + ny_sky_samp ** 2)
                maxr_y = np.max(np.array([maxr * 1.5, np.min(
                    np.hstack([maxr * 1.5 / sin_inc, maxr * 5.]))]))
                nz_sky_samp = np.int(np.max([nx_sky_samp, ny_sky_samp, maxr_y]))
                if np.mod(nz_sky_samp, 2) < 0.5:
                    nz_sky_samp += 1

                sh = (nz_sky_samp, ny_sky_samp, nx_sky_samp)
                zsky, ysky, xsky = np.indices(sh)
                zsky = zsky - (nz_sky_samp - 1) / 2.
                #ysky = ysky - (ny_sky_samp - 1) / 2.
                #xsky = xsky - (nx_sky_samp - 1) / 2.
                ysky = ysky - ycenter_samp # (ny_sky_samp - 1) / 2.
                xsky = xsky - xcenter_samp # (nx_sky_samp - 1) / 2.

                # Apply the geometric transformation to get outflow coordinates
                # Account for oversampling
                self.outflow_geometry.xshift = self.outflow_geometry.xshift.value * oversample
                self.outflow_geometry.yshift = self.outflow_geometry.yshift.value * oversample
                xout, yout, zout = self.outflow_geometry(xsky, ysky, zsky)

                # Convert to kpc
                xout_kpc = xout * rstep_samp / dscale
                yout_kpc = yout * rstep_samp / dscale
                zout_kpc = zout * rstep_samp / dscale

                rout = np.sqrt(xout**2 + yout**2 + zout**2)
                vout = self.outflow(xout_kpc, yout_kpc, zout_kpc)
                fout = self.outflow.light_profile(xout_kpc, yout_kpc, zout_kpc)

                # Apply extinction if it exists
                if self.extinction is not None:

                    fout *= self.extinction(xsky, ysky, zsky)

                # L.O.S. velocity is v*cos(alpha) = -v*zsky/rsky
                v_sys = self.outflow_geometry.vel_shift.value  # systemic velocity
                vobs = v_sys -vout * zsky/rout
                vobs[rout == 0] = vout[rout == 0]

                sigma_out = self.outflow_dispersion(rout)
                #for zz in range(nz_sky_samp):
                #    f_cube = np.tile(fout[zz, :, :], (nspec, 1, 1))
                #    vobs_cube = np.tile(vobs[zz, :, :], (nspec, 1, 1))
                #    sig_cube = np.tile(sigma_out[zz, :, :], (nspec, 1, 1))
                #    tmp_cube = np.exp(
                #        -0.5 * ((velcube - vobs_cube) / sig_cube) ** 2)
                #    cube_sum = np.nansum(tmp_cube, 0)
                #    cube_sum[cube_sum == 0] = 1
                #    cube_final += tmp_cube / cube_sum * f_cube
                cube_final += cutils.populate_cube(fout, vobs, sigma_out, vx)

                self.outflow_geometry.xshift = self.outflow_geometry.xshift.value / oversample
                self.outflow_geometry.yshift = self.outflow_geometry.yshift.value / oversample

            elif self.outflow._spatial_type == 'unresolved':

                # Set where the unresolved will be located and account for oversampling
                xshift = self.outflow_geometry.xshift.value * oversample
                yshift = self.outflow_geometry.yshift.value * oversample

                # The coordinates where the unresolved outflow is placed needs to be
                # an integer pixel so for now we round to nearest integer.
                # xpix = np.int(np.round(xshift)) + nx_sky_samp/2
                # ypix = np.int(np.round(yshift)) + ny_sky_samp/2

                xpix = np.int(np.round(xshift)) + np.int(np.round(xcenter_samp))
                ypix = np.int(np.round(yshift)) + np.int(np.round(ycenter_samp))

                voutflow = v_sys + self.outflow(vx)
                cube_final[:, ypix, xpix] += voutflow

                xshift = self.outflow_geometry.xshift.value / oversample
                yshift = self.outflow_geometry.yshift.value / oversample


        ####
        if (self.inflow is not None) & (self.inflow_geometry is not None):
            # If self.inflow_geometry is None:
            #   Just us the galaxy geometry and light profile: is just a superimposed kinematic signature
            #       on the normal disk rotation:
            # CALCULATED EARLIER

            if self.inflow._spatial_type == 'resolved':

                # Create 3D arrays of the sky pixel coordinates
                sin_inc = np.sin(self.inflow_geometry.inc * np.pi / 180.)

                maxr = np.sqrt(nx_sky_samp ** 2 + ny_sky_samp ** 2)
                maxr_y = np.max(np.array([maxr * 1.5, np.min(
                    np.hstack([maxr * 1.5 / sin_inc, maxr * 5.]))]))
                nz_sky_samp = np.int(np.max([nx_sky_samp, ny_sky_samp, maxr_y]))

                if np.mod(nz_sky_samp, 2) < 0.5:
                    nz_sky_samp += 1

                sh = (nz_sky_samp, ny_sky_samp, nx_sky_samp)
                zsky, ysky, xsky = np.indices(sh)
                zsky = zsky - (nz_sky_samp - 1) / 2.
                ysky = ysky - ycenter_samp
                xsky = xsky - xcenter_samp

                # Apply the geometric transformation to get inflow coordinates
                # Account for oversampling
                self.inflow_geometry.xshift = self.inflow_geometry.xshift.value * oversample
                self.inflow_geometry.yshift = self.inflow_geometry.yshift.value * oversample
                xin, yin, zin = self.inflow_geometry(xsky, ysky, zsky)

                # Convert to kpc
                xin_kpc = xin * rstep_samp / dscale
                yin_kpc = yin * rstep_samp / dscale
                zin_kpc = zin * rstep_samp / dscale

                rin = np.sqrt(xin**2 + yin**2 + zin**2)
                vin = self.inflow(xin_kpc, yin_kpc, zin_kpc)
                fin = self.inflow.light_profile(xin_kpc, yin_kpc, zin_kpc)

                # Apply extinction if it exists
                if self.extinction is not None:
                    fin *= self.extinction(xsky, ysky, zsky)

                # L.O.S. velocity is v*cos(alpha) = -v*zsky/rsky
                v_sys = self.inflow_geometry.vel_shift.value  # systemic velocity
                vobs = v_sys - vin * zsky/rin
                vobs[rin == 0] = vin[rin == 0]

                sigma_in = self.inflow_dispersion(rin)
                cube_final += cutils.populate_cube(fin, vobs, sigma_in, vx)

                self.inflow_geometry.xshift = self.inflow_geometry.xshift.value / oversample
                self.inflow_geometry.yshift = self.inflow_geometry.yshift.value / oversample


        #if debug:
        #    return cube_final, spec, flux_mass, vobs_mass

        return cube_final, spec


# ***** Mass Component Model Classes ******
# Base abstract mass model component class
class _DysmalModel(Model):
    """
    Base abstract `dysmalpy` model component class
    """

    parameter_constraints = DysmalParameter.constraints


class _DysmalFittable1DModel(_DysmalModel):
    """
    Base class for 1D model components
    """

    linear = False
    fit_deriv = None
    col_fit_deriv = True
    fittable = True

    inputs = ('x',)
    outputs = ('y',)


class MassModel(_DysmalFittable1DModel):
    """
    Base model for components that exert a gravitational influence
    """

    _type = 'mass'

    @abc.abstractmethod
    def enclosed_mass(self, *args, **kwargs):
        """Evaluate the enclosed mass as a function of radius"""

    @abc.abstractmethod
    def rho(self, *args, **kwargs):
        """Evaluate the density rho as a function of radius"""

    @abc.abstractmethod
    def dlnrho_dlnr(self, *args, **kwargs):
        """Evaluate the derivative dlnRho / dlnr as a function of radius"""

    @abc.abstractmethod
    def drho_dr(self, *args, **kwargs):
        """Evaluate the derivative dRho / dr as a function of radius"""

    def circular_velocity(self, r):
        r"""
        Default method to evaluate the circular velocity

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate circular velocity in kpc

        Returns
        -------
        vcirc : float or array
            Circular velocity at `r`

        Notes
        -----
        Calculates the circular velocity as a function of radius
        using the standard equation :math:`v(r) = \sqrt(GM(r)/r)`.
        This is only valid for a spherical mass distribution.
        """

        mass_enc = self.enclosed_mass(r)

        vcirc = v_circular(mass_enc, r)

        return vcirc


class ExpDisk(MassModel):
    """
    Infinitely thin exponential disk (i.e. Freeman disk)

    Parameters
    ----------
    total_mass : float
        Log of total mass of the disk in solar units

    r_eff : float
        Effective radius in kpc

    """

    total_mass = DysmalParameter(default=1, bounds=(5, 14))
    r_eff = DysmalParameter(default=1, bounds=(0, 50))
    _subtype = 'baryonic'

    def __init__(self, total_mass, r_eff, **kwargs):

        super(ExpDisk, self).__init__(total_mass, r_eff)

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

    def mass_to_light(self, r):
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

    def rho(self, r):
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

    def dlnrho_dlnr(self, r):
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

    def drho_dr(self, r):
        """
        Surface density radial derivative

        Parameters
        ----------
        r : float or array
            Radius in kpc

        Returns
        -------
        drhodr : float or array
            Surface density radial derivative at `r`
        """
        return self.dlnrho_dlnr(r) * self.rho(r) / r

class Sersic(MassModel):
    r"""
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

    Notes
    -----
    Model formula:

    .. math::

        M(r)=M_e\exp\left\{-b_n\left[\left(\frac{r}{r_{eff}}\right)^{(1/n)}-1\right]\right\}

    The constant :math:`b_n` is defined such that :math:`r_{eff}` contains half the total
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

        plt.axis([1e-1, 30, 1e5, 1e11])
        plt.xlabel('log Radius')
        plt.ylabel('log Mass Surface Density')
        plt.text(.25, 10**7.5, 'n=1')
        plt.text(.25, 1e11, 'n=10')
        plt.show()
    """

    total_mass = DysmalParameter(default=1, bounds=(5, 14))
    r_eff = DysmalParameter(default=1, bounds=(0, 50))
    n = DysmalParameter(default=1, bounds=(0, 8))

    _subtype = 'baryonic'

    def __init__(self, total_mass, r_eff, n, invq=1.0, noord_flat=False,
                 **kwargs):

        self.invq = invq
        self.noord_flat = noord_flat
        super(Sersic, self).__init__(total_mass, r_eff, n, **kwargs)

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
            #return sersic_menc(r, 10**self.total_mass, self.n, self.r_eff)
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

    def mass_to_light(self, r):
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

    def rho(self, r):
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
        if self.noord_flat:
            rho = sersic_curve_rho(r, self.r_eff, 10**self.total_mass, self.n, self.invq)

        else:
            rho = sersic_mr(r, 10**self.total_mass, self.n, self.r_eff)

        return rho

    def dlnrho_dlnr(self, r):
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

        if self.noord_flat:
            dlnrho_dlnr_arr = sersic_curve_dlnrho_dlnr(r, self.r_eff, self.n, self.invq)

            return dlnrho_dlnr_arr
        else:
            bn = scp_spec.gammaincinv(2. * self.n, 0.5)
            return -2. * (bn / self.n) * np.power(r/self.r_eff, 1./self.n)

    def drho_dr(self, r):
        """
        Surface density radial derivative

        Parameters
        ----------
        r : float or array
            Radius in kpc

        Returns
        -------
        drhodr : float or array
            Surface density radial derivative at `r`
        """
        return self.dlnrho_dlnr(r) * self.rho(r) / r


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

    def __init__(self, total_mass, r_eff_disk, n_disk, r_eff_bulge,
                 n_bulge, bt, invq_disk=5, invq_bulge=1, noord_flat=False,
                 light_component='disk', **kwargs):

        self.invq_disk = invq_disk
        self.invq_bulge = invq_bulge
        self.noord_flat = noord_flat
        self.light_component = light_component

        super(DiskBulge, self).__init__(total_mass, r_eff_disk, n_disk,
                                        r_eff_bulge, n_bulge, bt, **kwargs)

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
            #menc_bulge = sersic_menc(r, mbulge_total, self.n_bulge, self.r_eff_bulge)
            #menc_disk = sersic_menc(r, mdisk_total, self.n_disk, self.r_eff_disk)
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
            #menc_disk = sersic_menc(r, mdisk_total, self.n_disk, self.r_eff_disk)
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
            #menc_bulge = sersic_menc(r, mbulge_total, self.n_bulge, self.r_eff_bulge)
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


    def mass_to_light(self, r):
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

    def rho_disk(self, r):
        """
        Mass surface density of the disk as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        surf_dens : float or array
            Mass surface density at `r` in units of Msun/kpc^2
        """
        if self.noord_flat:
            mdisk_total = 10**self.total_mass*(1 - self.bt)
            rho = sersic_curve_rho(r, self.r_eff_disk, mdisk_total, self.n_disk, self.invq_disk)

            return rho
        else:
            mdisk_total = 10**self.total_mass*(1 - self.bt)
            mr_disk = sersic_mr(r, mdisk_total, self.n_disk, self.r_eff_disk)
            return mr_disk

    def rho_bulge(self, r):
        """
        Mass surface density of the bulge as a function of radius

        Parameters
        ----------
        r : float or array
            Radii at which to calculate the enclosed mass

        Returns
        -------
        surf_dens : float or array
            Mass surface density at `r` in units of Msun/kpc^2
        """
        if self.noord_flat:
            mbulge_total = 10**self.total_mass*self.bt

            rho = sersic_curve_rho(r, self.r_eff_bulge, mbulge_total, self.n_bulge, self.invq_bulge)
            return rho


        else:
            mbulge_total = 10**self.total_mass*self.bt
            mr_bulge = sersic_mr(r, mbulge_total, self.n_bulge, self.r_eff_bulge)
            return mr_bulge

    def rho(self, r):
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
        return self.rho_disk(r) + self.rho_bulge(r)

    def dlnrho_dlnr_disk(self, r):
        if self.noord_flat:
            dlnrho_dlnr_arr = sersic_curve_dlnrho_dlnr(r, self.r_eff_disk, self.n_disk, self.invq_disk)

            return dlnrho_dlnr_arr
        else:
            bn = scp_spec.gammaincinv(2. * self.n_disk, 0.5)
            return -2. * (bn / self.n_disk) * np.power(r/self.r_eff_disk, 1./self.n_disk)

    def dlnrho_dlnr_bulge(self, r):
        if self.noord_flat:
            dlnrho_dlnr_arr = sersic_curve_dlnrho_dlnr(r, self.r_eff_bulge, self.n_bulge, self.invq_bulge)

            return dlnrho_dlnr_arr
        else:
            bn = scp_spec.gammaincinv(2. * self.n_bulge, 0.5)
            return -2. * (bn / self.n_bulge) * np.power(r/self.r_eff_bulge, 1./self.n_bulge)

    def dlnrho_dlnr(self, r):
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

        return r / self.rho(r) * self.drho_dr(r)

    def drho_dr_disk(self, r):
        """
         Asymmetric drift term for the disk component

         Parameters
         ----------
         r : float or array
             Radius in kpc

         Returns
         -------
         log_drhodr : float or array
             Log surface density derivative as a function or radius
         """
        drhodr = self.rho_disk(r) / r * self.dlnrho_dlnr_disk(r)
        try:
            if len(r) > 1:
                drhodr[r==0] = 0.
            else:
                if r[0] == 0.:
                    drhodr[0] = 0.
        except:
            if r == 0:
                drhodr = 0.
        return drhodr

    def drho_dr_bulge(self, r):
        """
         Asymmetric drift term for the bulge component

         Parameters
         ----------
         r : float or array
             Radius in kpc

         Returns
         -------
         log_drhodr : float or array
             Log surface density derivative as a function or radius
         """
        drhodr = self.rho_bulge(r) / r * self.dlnrho_dlnr_bulge(r)
        try:
            if len(r) > 1:
                drhodr[r==0] = 0.
            else:
                if r[0] == 0.:
                    drhodr[0] = 0.
        except:
            if r == 0:
                drhodr = 0.

        return drhodr

    def drho_dr(self, r):
        """
        Surface density radial derivative

        Parameters
        ----------
        r : float or array
            Radius in kpc

        Returns
        -------
        drhodr : float or array
            Surface density radial derivative at `r`
        """

        return self.drho_dr_disk(r) + self.drho_dr_bulge(r)


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

    def __init__(self, total_mass, r_eff_disk, n_disk, r_eff_bulge,
                 n_bulge, bt, invq_disk=5, invq_bulge=1, noord_flat=False,
                 light_component='disk', **kwargs):

        self.invq_disk = invq_disk
        self.invq_bulge = invq_bulge
        self.noord_flat = noord_flat
        self.light_component = light_component

        super(LinearDiskBulge, self).__init__(total_mass, r_eff_disk, n_disk,
                                        r_eff_bulge, n_bulge, bt, **kwargs)

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
            #menc_bulge = sersic_menc(r, mbulge_total, self.n_bulge, self.r_eff_bulge)
            #menc_disk = sersic_menc(r, mdisk_total, self.n_disk, self.r_eff_disk)
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
            #menc_disk = sersic_menc(r, mdisk_total, self.n_disk, self.r_eff_disk)
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
            #menc_bulge = sersic_menc(r, mbulge_total, self.n_bulge, self.r_eff_bulge)
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


    def mass_to_light(self, r):
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


class DarkMatterHalo(MassModel):
    """
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
    conc = DysmalParameter(default=5.0, bounds=(2, 20))
    fdm = DysmalParameter(default=-99.9, fixed=True, bounds=(0,1))
    _subtype = 'dark_matter'

    @abc.abstractmethod
    def calc_rvir(self, *args, **kwargs):
        """
        Method to calculate the virial radius
        """

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

    def drho_dr(self, r):
        """
        Surface density radial derivative

        Parameters
        ----------
        r : float or array
            Radius in kpc

        Returns
        -------
        drhodr : float or array
            Surface density radial derivative at `r`
        """

        drhodr = self.rho(r) / r * self.dlnrho_dlnr(r)
        try:
            if len(r) > 1:
                drhodr[r == 0] = 0.
            else:
                if r[0] == 0.:
                    drhodr[0] = 0.
        except:
            if r == 0:
                drhodr = 0.
        return drhodr


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

    :math:`r_s` is the scale radius and defined as :math:`r_{vir}/c` where
    :math:`r_{vir}` is the virial radius and :math:`c` is the concentration
    parameter. :math:`rho_0` then is the density at :math:`r_s`.

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

    def __init__(self, mvirial, conc, alpha, beta, fdm = None,
            z=0, cosmo=_default_cosmo, **kwargs):

        self.z = z
        self.cosmo = cosmo
        super(TwoPowerHalo, self).__init__(mvirial, conc, alpha, beta, fdm, **kwargs)

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

    def calc_rho0(self):
        r"""
        Normalization of the density distribution

        Returns
        -------
        rho0 : float
            Mass density normalization in :math:`M_{\odot}/\rm{kpc}^3`
        """

        rvir = self.calc_rvir()
        rs = rvir/self.conc
        aa = -10**self.mvirial/(4*np.pi*self.conc**(3-self.alpha)*rs**3)
        bb = (self.alpha - 3) / scp_spec.hyp2f1(3-self.alpha, self.beta-self.alpha, 4-self.alpha, -self.conc)

        return aa*bb

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

        g_new_unit = G.to(u.pc / u.Msun * (u.km / u.s) ** 2).value
        hz = self.cosmo.H(self.z).value
        rvir = ((10 ** self.mvirial * (g_new_unit * 1e-3) /
                 (10 * hz * 1e-3) ** 2) ** (1. / 3.))

        return rvir

    def calc_alpha_from_fdm(self, baryons, r_fdm):
        """
        Calculate alpha given dark matter fraction and baryonic distribution

        Parameters
        ----------
        baryons : `~dysmalpy.models.MassModel`
            Model component representing the baryons

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
            vsqr_bar_re = baryons.circular_velocity(r_fdm)**2
            vsqr_dm_re_target = vsqr_bar_re / (1./self.fdm - 1)

            alphtest = np.arange(-50, 50, 1.)
            vtest = np.array([self._minfunc_vdm(alph, vsqr_dm_re_target, self.mvirial, self.conc,
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

            alpha = scp_opt.brentq(self._minfunc_vdm, a, b, args=(vsqr_dm_re_target, self.mvirial, self.conc,
                                        self.beta, self.z, r_fdm))

        return alpha

    def _minfunc_vdm(self, alpha, vtarget, mass, conc, beta, z, r_eff):
        halo = TwoPowerHalo(mvirial=mass, conc=conc, alpha=alpha, beta=beta, z=z)
        return halo.circular_velocity(r_eff) ** 2 - vtarget

    def rho(self, r):
        r"""
        Mass density as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        rho : float or array
            Mass density at `r` in :math:`M_{\odot}/\rm{kpc}^3`
        """
        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0()
        rs = rvirial / self.conc

        return rho0 / ((r/rs)**self.alpha * (1. + r/rs)**(self.beta - self.alpha))

    def dlnrho_dlnr(self, r):
        """
        Log gradient of rho as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        dlnrho_dlnr : float or array
            Log gradient of rho at `r`
        """
        rvirial = self.calc_rvir()
        rs = rvirial / self.conc
        return -self.alpha - (self.beta-self.alpha)*(r/rs)/(1. + r/rs)


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

    def __init__(self, mvirial, rB, fdm=None,
            z=0, cosmo=_default_cosmo, **kwargs):
        self.z = z
        self.cosmo = cosmo
        super(Burkert, self).__init__(mvirial, rB, fdm, **kwargs)

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

        g_new_unit = G.to(u.pc / u.Msun * (u.km / u.s) ** 2).value
        hz = self.cosmo.H(self.z).value
        rvir = ((10 ** self.mvirial * (g_new_unit * 1e-3) /
                 (10 * hz * 1e-3) ** 2) ** (1. / 3.))

        return rvir

    def calc_rB_from_fdm(self, baryons, r_fdm):
        """
        Calculate core radius given dark matter fraction and baryonic distribution

        Parameters
        ----------
        baryons : `~dysmalpy.models.MassModel`
            Model component representing the baryons

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
            vsqr_bar_re = baryons.circular_velocity(r_fdm)**2
            vsqr_dm_re_target = vsqr_bar_re / (1./self.fdm - 1)

            rBtest = np.arange(0., 250., 5.0)
            vtest = np.array([self._minfunc_vdm(rBt, vsqr_dm_re_target, self.mvirial, self.z, r_fdm) for rBt in rBtest])

            # a = rBtest[vtest < 0][-1]
            # b = rBtest[vtest > 0][0]

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
                rB = scp_opt.brentq(self._minfunc_vdm, a, b, args=(vsqr_dm_re_target, self.mvirial, self.z, r_fdm))
            except:
                # SOMETHING, if it's failing...
                rB = np.average([a,b])

        return rB

    def _minfunc_vdm(self, rB, vtarget, mass, z, r_eff):
        halo = Burkert(mvirial=mass, rB=rB, z=z)
        return halo.circular_velocity(r_eff) ** 2 - vtarget

    ##########
    def rho(self, r):
        r"""
        Mass density as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        rho : float or array
            Mass density at `r` in :math:`M_{\odot}/\rm{kpc}^3`
        """
        rho0 = self.calc_rho0()
        return rho0 / ( (1 + r/self.rB) * ( 1 + (r/self.rB)**2 ) )


    def dlnrho_dlnr(self, r):
        """
        Log gradient of rho as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        dlnrho_dlnr : float or array
            Log gradient of rho at `r`
        """
        return -(r/self.rB) /(1.+r/self.rB) - 2.*(r/self.rB)**2/(1.+(r/self.rB)**2)


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
    :math:`r_s` is the scale radius defined as :math:`r_{vir}/c`.

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

    def __init__(self, mvirial, conc, alphaEinasto=None, nEinasto=None, fdm=None,
            z=0, cosmo=_default_cosmo, Einasto_param='None', **kwargs):
        self.z = z
        self.cosmo = cosmo

        # Check whether at least *one* of alphaEinasto and nEinasto is set:
        if (alphaEinasto is None) & (nEinasto is None):
            raise ValueError("Must set at least one of alphaEinasto and nEinasto!")
        if (alphaEinasto is not None) & (nEinasto is not None) & (Einasto_param == 'None'):
            raise ValueError("If both 'alphaEinasto' and 'nEinasto' are set, must specify which is the fit variable with 'Einasto_param'")

        super(Einasto, self).__init__(mvirial, conc, alphaEinasto, nEinasto, fdm, **kwargs)

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
        rho0 = self.calc_rho0()
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

        rho0 = self.calc_rho0()

        # Explicitly substituted for s = r/h before doing s^(1/nEinasto)
        incomp_gam =  scp_spec.gammainc(3*self.nEinasto, 2.*self.nEinasto * np.power(r/rs, 1./self.nEinasto) ) \
                        * scp_spec.gamma(3*self.nEinasto)

        Menc = 4.*np.pi * rho0 * np.power(h, 3.) * self.nEinasto * incomp_gam

        return Menc

    def calc_rho0(self):
        r"""
        Density at the scale length

        Returns
        -------
        rho0 : float
            Mass density at the scale radius in :math:`M_{\odot}/\rm{kpc}^3`
        """
        rvir = self.calc_rvir()
        rs = rvir/self.conc
        h = rs / np.power(2.*self.nEinasto, self.nEinasto)

        incomp_gam =  scp_spec.gammainc(3*self.nEinasto, (2.*self.nEinasto) * np.power(self.conc, 1./self.nEinasto) ) \
                        * scp_spec.gamma(3*self.nEinasto)

        rho0 = 10**self.mvirial / (4.*np.pi*self.nEinasto * np.power(h, 3.) * incomp_gam)

        return rho0


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

        g_new_unit = G.to(u.pc / u.Msun * (u.km / u.s) ** 2).value
        hz = self.cosmo.H(self.z).value
        rvir = ((10 ** self.mvirial * (g_new_unit * 1e-3) /
                 (10 * hz * 1e-3) ** 2) ** (1. / 3.))

        return rvir

    def calc_alphaEinasto_from_fdm(self, baryons, r_fdm):
        """
        Calculate alpha given dark matter fraction and baryonic distribution

        Parameters
        ----------
        baryons : `~dysmalpy.models.MassModel`
            Model component representing the baryons

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
        baryons : `~dysmalpy.models.MassModel`
            Model component representing the baryons

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

            vsqr_bar_re = baryons.circular_velocity(r_fdm)**2
            vsqr_dm_re_target = vsqr_bar_re / (1./self.fdm - 1)

            nEinastotest = np.arange(-50, 50, 1.)
            vtest = np.array([self._minfunc_vdm(nEinast, vsqr_dm_re_target, self.mvirial, self.conc,
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

            alpha = scp_opt.brentq(self._minfunc_vdm, a, b, args=(vsqr_dm_re_target, self.mvirial, self.conc,
                                        self.alphaEinasto, self.z, r_fdm))

        return nEinasto

    def _minfunc_vdm(self, nEinasto, vtarget, mass, conc, alphaEinasto, z, r_eff):
        halo = Einasto(mvirial=mass, conc=conc, nEinasto=nEinasto, alphaEinasto=alphaEinasto, z=z)
        return halo.circular_velocity(r_eff) ** 2 - vtarget

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

    def rho(self, r):
        r"""
        Mass density as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        rho : float or array
            Mass density at `r` in :math:`M_{\odot}/\rm{kpc}^3`
        """
        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0()
        rs = rvirial / self.conc
        h = rs / np.power(2.*self.nEinasto, self.nEinasto)

        # Return the density at a given radius:
        return rho0 * np.exp(- np.power(r/h, 1./self.nEinasto))

    def dlnrho_dlnr(self, r):
        """
        Log gradient of rho as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        dlnrho_dlnr : float or array
            Log gradient of rho at `r`
        """
        rvirial = self.calc_rvir()
        rs = rvirial / self.conc
        # self.alphaEinasto = 1./self.nEinasto
        return -2. * np.power(r/rs, self.alphaEinasto)


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
    :math:`\rho_0` then is the mass density at :math:`r_s`.

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/1995MNRAS.275..720N/abstract
    """

    def __init__(self, mvirial, conc, fdm = None,
            z=0, cosmo=_default_cosmo, **kwargs):

        self.z = z
        self.cosmo = cosmo
        super(NFW, self).__init__(mvirial, conc, fdm, **kwargs)

    def evaluate(self, r, mvirial, conc, fdm):
        """Mass density as a function of radius"""

        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0()
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

        rho0 = self.calc_rho0()
        rvirial = self.calc_rvir()
        rs = rvirial/self.conc
        aa = 4.*np.pi*rho0*rvirial**3/self.conc**3

        # For very small r, bb can be negative.
        bb = np.abs(np.log((rs + r)/rs) - r/(rs + r))

        return aa*bb

    def calc_rho0(self):
        r"""
        Normalization of the density distribution

        Returns
        -------
        rho0 : float
            Mass density normalization in :math:`M_{\odot}/\rm{kpc}^3`
        """
        rvirial = self.calc_rvir()
        aa = 10**self.mvirial/(4.*np.pi*rvirial**3)*self.conc**3
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
        g_new_unit = G.to(u.pc / u.Msun * (u.km / u.s) ** 2).value
        hz = self.cosmo.H(self.z).value
        rvir = ((10 ** self.mvirial * (g_new_unit * 1e-3) /
                (10 * hz * 1e-3) ** 2) ** (1. / 3.))

        return rvir

    def calc_mvirial_from_fdm(self, baryons, r_fdm, adiabatic_contract=False):
        """
        Calculate virial mass given dark matter fraction and baryonic distribution

        Parameters
        ----------
        baryons : `~dysmalpy.models.MassModel`
            Model component representing the baryons

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
            vsqr_bar_re = baryons.circular_velocity(r_fdm)**2
            vsqr_dm_re_target = vsqr_bar_re / (1./self.fdm.value - 1)

            if not np.isfinite(vsqr_dm_re_target):
                mvirial = np.NaN
            else:
                mtest = np.arange(-5, 50, 1.0)
                if adiabatic_contract:
                    vtest = np.array([self._minfunc_vdm_AC(m, vsqr_dm_re_target, self.conc.value, self.z, r_fdm, baryons) for m in mtest])
                    # TEST
                    vtest_noAC = np.array([self._minfunc_vdm(m, vsqr_dm_re_target, self.conc.value, self.z, r_fdm) for m in mtest])
                else:
                    vtest = np.array([self._minfunc_vdm(m, vsqr_dm_re_target, self.conc.value, self.z, r_fdm) for m in mtest])

                try:
                    a = mtest[vtest < 0][-1]
                    b = mtest[vtest > 0][0]
                    # TEST
                    if adiabatic_contract:
                        a_noAC = mtest[vtest_noAC < 0][-1]
                        b_noAC = mtest[vtest_noAC > 0][0]
                except:
                    print("adiabatic_contract={}".format(adiabatic_contract))
                    print("fdm={}".format(self.fdm.value))
                    print("r_fdm={}".format(r_fdm))
                    print(mtest, vtest)
                    raise ValueError

                if adiabatic_contract:
                    # # TEST
                    # print("mtest={}".format(mtest))
                    # print("vtest={}".format(vtest))
                    mvirial = scp_opt.brentq(self._minfunc_vdm_AC, a, b, args=(vsqr_dm_re_target, self.conc.value, self.z, r_fdm, baryons))

                    # TEST
                    mvirial_noAC = scp_opt.brentq(self._minfunc_vdm, a_noAC, b_noAC, args=(vsqr_dm_re_target, self.conc.value, self.z, r_fdm))
                    print("mvirial={}, mvirial_noAC={}".format(mvirial, mvirial_noAC))
                else:
                    mvirial = scp_opt.brentq(self._minfunc_vdm, a, b, args=(vsqr_dm_re_target, self.conc.value, self.z, r_fdm))

        return mvirial

    def _minfunc_vdm(self, mass, vtarget, conc, z, r_eff):
        halo = NFW(mvirial=mass, conc=conc, z=z)
        return halo.circular_velocity(r_eff) ** 2 - vtarget

    def _minfunc_vdm_AC(self, mass, vtarget, conc, z, r_eff, bary):
        halo = NFW(mvirial=mass, conc=conc, z=z, name='halotmp')
        modtmp = ModelSet()
        modtmp.add_component(bary, light=True)
        modtmp.add_component(halo)
        modtmp.kinematic_options.adiabatic_contract = True
        modtmp.kinematic_options.adiabatic_contract_modify_small_values = True

        vc, vc_dm = modtmp.circular_velocity(r_eff, compute_dm=True)
        #print("mass={}, barymass={}, vc={}, vcdm={}, vcdm_noAC={}, vtarget={}".format(mass, bary.total_mass.value, vc,
        #            vc_dm, halo.circular_velocity(r_eff), np.sqrt(vtarget)))
        return vc_dm **2 - vtarget

    def rho(self, r):
        r"""
        Mass density as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        rho : float or array
            Mass density at `r` in :math:`M_{\odot}/\rm{kpc}^3`
        """
        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0()
        rs = rvirial / self.conc

        return rho0 / ((r/rs) * (1. + r/rs)**2)

    def dlnrho_dlnr(self, r):
        """
        Log gradient of rho as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        dlnrho_dlnr : float or array
            Log gradient of rho at `r`
        """

        rvirial = self.calc_rvir()
        rs = rvirial / self.conc
        return -1. - 2.*(r/rs)/(1. + r/rs)


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
    :math:`\rho_0` then is the mass density at :math:`r_s`.

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/1995MNRAS.275..720N/abstract
    """

    def __init__(self, mvirial, conc, fdm = None,
            z=0, cosmo=_default_cosmo, **kwargs):
        self.z = z
        self.cosmo = cosmo
        super(LinearNFW, self).__init__(mvirial, conc, fdm, **kwargs)

    def evaluate(self, r, mvirial, conc, fdm):
        """Mass density as a function of radius"""

        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0()
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

        rho0 = self.calc_rho0()
        rvirial = self.calc_rvir()
        rs = rvirial/self.conc
        aa = 4.*np.pi*rho0*rvirial**3/self.conc**3

        # For very small r, bb can be negative.
        bb = np.abs(np.log((rs + r)/rs) - r/(rs + r))

        return aa*bb

    def calc_rho0(self):
        r"""
        Density at the scale radius

        Returns
        -------
        rho0 : float
            Mass density at the scale radius in :math:`M_{\odot}/\rm{kpc}^3`
        """
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
        g_new_unit = G.to(u.pc / u.Msun * (u.km / u.s) ** 2).value
        hz = self.cosmo.H(self.z).value
        rvir = ((self.mvirial * (g_new_unit * 1e-3) /
                (10 * hz * 1e-3) ** 2) ** (1. / 3.))

        return rvir

    def rho(self, r):
        r"""
         Mass density as a function of radius

         Parameters
         ----------
         r : float or array
             Radius or radii in kpc

         Returns
         -------
         rho : float or array
             Mass density at `r` in :math:`M_{\odot}/\rm{kpc}^3`
         """
        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0()
        rs = rvirial / self.conc

        return rho0 / ((r/rs) * (1. + r/rs)**2)

    def dlnrho_dlnr(self, r):
        """
        Log gradient of rho as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        dlnrho_dlnr : float or array
            Log gradient of rho at `r`
        """
        rvirial = self.calc_rvir()
        rs = rvirial / self.conc
        return -1. - 2.*(r/rs)/(1. + r/rs)


# ****** Geometric Model ********
class _DysmalFittable3DModel(_DysmalModel):
    """
        Base class for 3D model components
    """

    linear = False
    fit_deriv = None
    col_fit_deriv = True
    fittable = True

    inputs = ('x', 'y', 'z')


class Geometry(_DysmalFittable3DModel):
    """
    Model component defining the transformation from galaxy to sky coordinates

    Parameters
    ----------
    inc : float
        Inclination of the modelin degrees

    pa : float
        Position angle East of North of the blueshifted side of the model in degrees

    xshift : float
        x-coordinate of the center of the model relative to center of data cube in pixels

    yshift : float
        y-coordinate of the center of the model relative to center of data cube in pixels

    vel_shift : float
        Systemic velocity shift that will be applied to the whole cube in km/s

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

    _type = 'geometry'
    outputs = ('xp', 'yp', 'zp')

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


# ******* Dispersion Profiles **************
class DispersionProfile(_DysmalFittable1DModel):
    """Base object for dispersion profile models"""
    _type = 'dispersion'


class DispersionConst(DispersionProfile):
    """
    Model for a constant dispersion

    Parameters
    ----------
    sigma0 : float
        Value of the dispersion at all radii
    """
    sigma0 = DysmalParameter(default=10., bounds=(0, None), fixed=True)

    @staticmethod
    def evaluate(r, sigma0):
        """Dispersion as a function of radius"""
        return np.ones(r.shape)*sigma0


# ******* Z-Height Profiles ***************
class ZHeightProfile(_DysmalFittable1DModel):
    """Base object for flux profiles in the z-direction"""
    _type = 'zheight'


    # Must set property z_scalelength for each subclass,
    #   for use with getting indices ai for filling simulated cube

class ZHeightGauss(ZHeightProfile):
    r"""
    Gaussian flux distribution in the z-direction

    Parameters
    ----------
    sigmaz : float
        Dispersion of the Gaussian in kpc

    Notes
    -----
    Model formula:

    .. math::

        F_z = \exp\left\{\frac{-z^2}{2\sigma_z^2}\right\}
    """
    sigmaz = DysmalParameter(default=1.0, fixed=True, bounds=(0, 10))

    def __init__(self, sigmaz, **kwargs):
        super(ZHeightGauss, self).__init__(sigmaz, **kwargs)

    @staticmethod
    def evaluate(z, sigmaz):
        return np.exp(-0.5*(z/sigmaz)**2)

    @property
    def z_scalelength(self):
        return self.sigmaz



# ****** Kinematic Options Class **********
class KinematicOptions:
    r"""
    Object for storing and applying kinematic corrections

    Parameters
    ----------
    adiabatic_contract : bool
        If True, apply adiabatic contraction when deriving the rotational velocity

    pressure_support : bool
        If True, apply asymmetric drift correction when deriving the rotational velocity

    pressure_support_type : {1, 2, 3}
        Type of asymmetric drift correction. Default is 1 (following Burkert et al. 2010).

    pressure_support_re : float
        Effective radius in kpc to use for asymmetric drift calculation

    pressure_support_n : float
        Sersic index to use for asymmetric drift calculation

    Notes
    -----
    **Adiabatic contraction** is applied following Burkert et al (2010) [1]_.
    The recipe involves numerically solving these two implicit equations:

    .. math::

        v^2_{\rm circ}(r) = v^2_{\rm disk}(r) + v^2_{\rm DM}(r^{\prime})

        r^{\prime} = r\left(1 + \frac{rv^2_{\rm disk}(r)}{r^{\prime} v^2_{\rm DM}(r^{\prime})}\right)

    Adiabatic contraction then can only be applied if there is a halo and baryon component
    in the `ModelSet`.


    **Pressure support** (i.e., asymmetric drift) can be calculated in three different ways.

    By default (`pressure_support_type=1`), the asymmetric drift derivation from
    Burkert et al. (2010) [1]_, Equation (11) is applied
    (assuming an exponential disk, with :math:`R_e=1.678R_e`):

    .. math::

        v^2_{\rm rot}(r) = v^2_{\rm circ} - 3.36 \sigma_0^2 \left(\frac{r}{R_e}\right)

    Alternatively, for `pressure_support_type=2`, the Sersic index can be taken into account beginning from
    Eq (9) of Burkert et al. (2010), so the asymmetric drift is then:

    .. math::

        v^2_{\rm rot}(r) = v^2_{\rm circ} - 2 \sigma_0^2 \frac{b_n}{n} \left(\frac{r}{R_e}\right)^{1/n}

    Finally, for `pressure_support_type=3`, the asymmetric drift is determined using
    the pressure gradient (assuming constant veloctiy dispersion :math:`\sigma_0`).
    This approach allows for explicitly incorporating different gradients
    :math:`d\ln{}\rho(r)/d\ln{}r` for different components (versus applying the disk geometry inherent in the
    in the later parts of the Burkert et al. derivation).
    For `pressure_support_type=3`, we follow Eq (3) of Burkert et al. (2010):

    .. math::

        v^2_{\rm rot}(r) = v^2_{\rm circ} + \sigma_0^2 \frac{d \ln \rho(r)}{d \ln r}



    Warnings
    --------
    Adiabatic contraction can significantly increase the computation time for a `ModelSet`
    to simulate a cube.

    References
    ----------
    .. [1] https://ui.adsabs.harvard.edu/abs/2010ApJ...725.2324B/abstract
    """

    def __init__(self, adiabatic_contract=False, pressure_support=False,
                 pressure_support_type=1, pressure_support_re=None,
                 pressure_support_n=None):

        self.adiabatic_contract = adiabatic_contract
        self.pressure_support = pressure_support
        self.pressure_support_re = pressure_support_re
        self.pressure_support_n = pressure_support_n
        self.pressure_support_type = pressure_support_type

    def apply_adiabatic_contract(self, model, r, vbaryon, vhalo,
                                 compute_dm=False, model_key_re=['disk+bulge', 'r_eff_disk'],
                                 step1d = 0.2):
        """
        Function that applies adiabatic contraction to a ModelSet

        Parameters
        ----------
        model : `ModelSet`
            ModelSet that adiabatic contraction will be applied to

        r : array
            Radii in kpc

        vbaryon : array
            Baryonic component circular velocities in km/s

        vhalo : array
            Dark matter halo circular velocities in km/s

        compute_dm : bool
            If True, will return the adiabatically contracted halo velocities.

        model_key_re : list
            Two element list which contains the name of the model component
            and parameter to use for the effective radius.
            Default is ['disk+bulge', 'r_eff_disk'].

        step1d : float
            Step size in kpc to use during adiabatic contraction calculation

        Returns
        -------
        vel : array
           Total circular velocity corrected for adiabatic contraction in km/s

        vhalo_adi : array
            Dark matter halo circular velocities corrected for adiabatic contraction.
            Only returned if `compute_dm` = True
        """

        if self.adiabatic_contract:
            #logger.info("Applying adiabatic contraction.")

            # Define 1d radius array for calculation
            #step1d = 0.2  # kpc
            # r1d = np.arange(step1d, np.ceil(r.max()/step1d)*step1d+ step1d, step1d, dtype=np.float64)
            try:
                rmaxin = r.max()
            except:
                rmaxin = r
            # Get reff:
            comp = model.components.__getitem__(model_key_re[0])
            param_i = comp.param_names.index(model_key_re[1])
            r_eff = comp.parameters[param_i]

            rmax_calc = max(5.* r_eff, rmaxin)

            # Wide enough radius range for full calculation -- out to 5*Reff, at least
            r1d = np.arange(step1d, np.ceil(rmax_calc/step1d)*step1d+ step1d, step1d, dtype=np.float64)


            rprime_all_1d = np.zeros(len(r1d))

            # Calculate vhalo, vbaryon on this 1D radius array [note r is a 3D array]
            vhalo1d = r1d * 0.
            vbaryon1d = r1d * 0.
            for cmp in model.mass_components:

                if model.mass_components[cmp]:
                    mcomp = model.components[cmp]
                    if isinstance(mcomp, DiskBulge) | isinstance(mcomp, LinearDiskBulge):
                        cmpnt_v = mcomp.circular_velocity(r1d)
                    else:
                        cmpnt_v = mcomp.circular_velocity(r1d)
                    if mcomp._subtype == 'dark_matter':

                        vhalo1d = np.sqrt(vhalo1d ** 2 + cmpnt_v ** 2)

                    elif mcomp._subtype == 'baryonic':

                        vbaryon1d = np.sqrt(vbaryon1d ** 2 + cmpnt_v ** 2)

                    elif mcomp._subtype == 'combined':

                        raise ValueError('Adiabatic contraction cannot be turned on when'
                                         'using a combined baryonic and halo mass model!')

                    else:
                        raise TypeError("{} mass model subtype not recognized"
                                        " for {} component. Only 'dark_matter'"
                                        " or 'baryonic' accepted.".format(mcomp._subtype, cmp))


            converged = np.zeros(len(r1d), dtype=np.bool)
            for i in range(len(r1d)):
                try:
                    result = scp_opt.newton(_adiabatic, r1d[i] + 1.,
                                        args=(r1d[i], vhalo1d, r1d, vbaryon1d[i]),
                                        maxiter=200)
                    converged[i] = True
                except:
                    result = r1d[i]
                    converged[i] = False

                # ------------------------------------------------------------------
                # HACK TO FIX WEIRD AC: If too weird: toss it...
                if ('adiabatic_contract_modify_small_values' in self.__dict__.keys()):
                    if self.adiabatic_contract_modify_small_values:
                        if ((result < 0.) | (result > 5*max(r1d))):
                            #print("tossing, mvir={}".format(model.components['halotmp'].mvirial.value))
                            result = r1d[i]
                            converged[i] = False
                # ------------------------------------------------------------------

                rprime_all_1d[i] = result


            vhalo_adi_interp_1d = scp_interp.interp1d(r1d, vhalo1d, fill_value='extrapolate', kind='linear')   # linear interpolation
            ####vhalo_adi_interp_1d = scp_interp.interp1d(r1d, vhalo1d, kind='linear')   # linear interpolation

            #print("r1d={}".format(r1d))
            #print("rprime_all_1d={}".format(rprime_all_1d))

            # Just calculations:
            if converged.sum() < len(r1d):
                if converged.sum() >= 0.9 *len(r1d):
                    rprime_all_1d = rprime_all_1d[converged]
                    r1d = r1d[converged]

            vhalo_adi_1d = vhalo_adi_interp_1d(rprime_all_1d)
            ####print("vhalo_adi_1d={}".format(vhalo_adi_1d))
            vhalo_adi_interp_map_3d = scp_interp.interp1d(r1d, vhalo_adi_1d, fill_value='extrapolate', kind='linear')

            vhalo_adi = vhalo_adi_interp_map_3d(r)

            vel = np.sqrt(vhalo_adi ** 2 + vbaryon ** 2)

        else:
            vel = np.sqrt(vhalo ** 2 + vbaryon ** 2)

        if compute_dm:
            if self.adiabatic_contract:
                return vel, vhalo_adi
            else:
                return vel, vhalo
        else:
            return vel

    def apply_pressure_support(self, r, model, vel):
        """
        Function to apply asymmetric drift correction

        Parameters
        ----------
        r : float or array
            Radius or radii at which to apply the correction

        model : `ModelSet`
            ModelSet for which the correction is applied to

        vel : float or array
            Circular velocity in km/s

        Returns
        -------
        vel : float or array
            Rotational velocity with asymmetric drift applied in km/s

        """
        if self.pressure_support:
            vel_asymm_drift = self.get_asymm_drift_profile(r, model)
            vel_squared = (vel **2 - vel_asymm_drift**2)

            # if array:
            try:
                vel_squared[vel_squared < 0] = 0.
            except:
                # if float single value:
                if vel_squared < 0:
                    vel_squared = 0.
            vel = np.sqrt(vel_squared)

        return vel

    def correct_for_pressure_support(self, r, model, vel):
        """
        Remove asymmetric drift effect from input velocities

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        model : `ModelSet`
            ModelSet the correction is applied to

        vel : float or array
            Rotational velocities in km/s from which to remove asymmetric drift

        Returns
        -------
        vel : float or array
            Circular velocity after asymmetric drift is removed in km/s
        """
        if self.pressure_support:
            #
            vel_asymm_drift = self.get_asymm_drift_profile(r, model)
            vel_squared = (vel **2 + vel_asymm_drift**2)

            # if array:
            try:
                vel_squared[vel_squared < 0] = 0.
            except:
                # if float single value:
                if (vel_squared < 0):
                    vel_squared = 0.
            vel = np.sqrt(vel_squared)

        return vel

    def get_asymm_drift_profile(self, r, model):
        """
        Calculate the asymmetric drift correction

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        model : `ModelSet`
            ModelSet the correction is applied to

        Returns
        -------
        vel_asymm_drift : float or array
            Velocity correction in km/s associated with asymmetric drift
        """
        pre = self.get_pressure_support_param(model, param='re')

        if model.dispersion_profile is None:
            raise AttributeError("Can't apply pressure support without "
                                 "a dispersion profile!")

        sigma = model.dispersion_profile(r)
        if self.pressure_support_type == 1:
            # Pure exponential derivation // n = 1
            # vel_squared = (vel ** 2 - 3.36 * (r / pre) * sigma ** 2)
            vel_asymm_drift = np.sqrt( 3.36 * (r / pre) * sigma ** 2 )
        elif self.pressure_support_type == 2:
            # Modified derivation that takes into account n_disk / n
            pn = self.get_pressure_support_param(model, param='n')

            bn = scp_spec.gammaincinv(2. * pn, 0.5)
            #vel_squared = (vel ** 2 - 2. * (bn/pn) * np.power((r/pre), 1./pn) * sigma**2 )
            vel_asymm_drift = np.sqrt( 2. * (bn/pn) * np.power((r/pre), 1./pn) * sigma**2 )

        elif self.pressure_support_type == 3:
            # Direct calculation from sig0^2 dlnrho/dlnr:

            if not _sersic_profile_mass_VC_loaded:
                raise ImportError("The module 'sersic_profile_mass_VC' is currently needed to use 'pressure_support_type=3'")

            dlnrhotot_dlnr = model.get_dlnrhotot_dlnr(r)

            vel_asymm_drift = np.sqrt( - dlnrhotot_dlnr * sigma**2 )

            #raise ValueError
            #FLAG42

        return vel_asymm_drift

    def get_pressure_support_param(self, model, param=None):
        """
        Return model parameters needed for asymmetric drift calculation

        Parameters
        ----------
        model : `ModelSet`
            ModelSet the correction is applied to

        param : {'n', 're'}
            Which parameter value to retrieve. Either the effective radius or Sersic index

        Returns
        -------
        p_val : float
            Parameter value
        """
        p_altnames = {'n': 'n',
                      're': 'r_eff'}
        if param not in ['n', 're']:
            raise ValueError("get_pressure_support_param() only works for param='n', 're'")

        paramkey = 'pressure_support_{}'.format(param)
        p_altname = p_altnames[param]

        if self.__dict__[paramkey] is None:
            p_val = None
            for cmp in model.mass_components:
                if model.mass_components[cmp]:
                    mcomp = model.components[cmp]
                    if (mcomp._subtype == 'baryonic') | (mcomp._subtype == 'combined'):
                        if (isinstance(mcomp, DiskBulge)) | (isinstance(mcomp, LinearDiskBulge)):
                            p_val = mcomp.__getattribute__('{}_disk'.format(p_altname)).value
                        else:
                            p_val = mcomp.__getattribute__('{}'.format(p_altname)).value
                        break

            if p_val is None:
                if param == 're':
                    logger.warning("No baryonic mass component found. Using "
                               "1 kpc as the pressure support effective"
                               " radius")
                    p_val = 1.0
                elif param == 'n':
                    logger.warning("No baryonic mass component found. Using "
                               "n=1 as the pressure support Sersic index")
                    p_val = 1.0

        else:
            p_val = self.__dict__[paramkey]

        return p_val

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

    _type = 'outflow'
    _spatial_type = 'resolved'
    outputs = ('vout',)

    def __init__(self, n, vmax, rturn, thetain, dtheta, rend, norm_flux, tau_flux,
                 profile_type='both', **kwargs):

        valid_profiles = ['increase', 'decrease', 'both', 'constant']

        if profile_type in valid_profiles:
            self.profile_type = profile_type
        else:
            logger.error("Invalid profile type. Must be one of 'increase',"
                         "'decrease', 'constant', or 'both.'")

        #self.tau_flux = tau_flux
        #self.norm_flux = norm_flux

        super(BiconicalOutflow, self).__init__(n, vmax, rturn, thetain,
                                               dtheta, rend, norm_flux, tau_flux, **kwargs)

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


class UnresolvedOutflow(_DysmalFittable1DModel):
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

    _type = 'outflow'
    _spatial_type = 'unresolved'
    outputs = ('vout',)

    @staticmethod
    def evaluate(v, vcenter, fwhm, amplitude):

        return amplitude*np.exp(-(v - vcenter)**2/(fwhm/2.35482)**2)


class UniformRadialInflow(_DysmalFittable3DModel):
    """
    Model for a uniform radial inflow.

    Parameters
    ----------
    vin : float
        Inflow velocity in km/s

    Notes
    -----
    This model simply adds a constant inflowing radial velocity component
    to all of the positions in the galaxy.
    """
    vin = DysmalParameter(default=30.)

    _type = 'inflow'
    _spatial_type = 'resolved'
    outputs = ('vinfl',)

    def __init__(self, vin, **kwargs):

        super(UniformRadialInflow, self).__init__(vin, **kwargs)

    def evaluate(self, x, y, z, vin):
        """Evaluate the inflow velocity as a function of position x, y, z"""

        vel = np.ones(x.shape) * (- vin)   # negative because of inflow

        return vel


class DustExtinction(_DysmalFittable3DModel):
    r"""
    Model for extinction due to a thin plane of dust

    Parameters
    ----------
    inc : float
        Inclination of the dust plane in deg

    pa : float
        Position angle of the dust plane in deg

    xshift, yshift : float
        Offset in pixels of the center of the dust plane

    amp_extinct : float
        Strength of the extinction through the dust plane. Expressed as the fraction of
        flux that is transmitted through the dust plane. `amp_extinct` = 1 means

    Notes
    -----
    This model places a dust plane within the model cube. All positions that
    are behind the dust plane relative to the line of sight will have their
    flux reduced by `amp_extinct`:

        .. math::

            F = AF_{\rm intrinsic}

    where :math:`A` is between 0 and 1.
    """

    inc = DysmalParameter(default=45.0, bounds=(0, 90))
    pa = DysmalParameter(default=0.0, bounds=(-180, 180))
    xshift = DysmalParameter(default=0.0)
    yshift = DysmalParameter(default=0.0)
    amp_extinct = DysmalParameter(default=0.0, bounds=(0., 1.))  # default: none

    _type = 'extinction'
    outputs = ('yp',)

    @staticmethod
    def evaluate(x, y, z, inc, pa, xshift, yshift, amp_extinct):
        inc = np.pi / 180. * inc
        pa = np.pi / 180. * (pa - 90.)

        xsky = x - xshift
        ysky = y - yshift
        zsky = z

        ytmp = -xsky * np.sin(pa) + ysky * np.cos(pa)

        ydust = ytmp * np.cos(inc) - zsky * np.sin(inc)

        zsky_dust = ydust * np.sin(-inc)
        extinction = np.ones(x.shape)
        extinction[zsky <= zsky_dust] = amp_extinct

        return extinction


def _adiabatic(rprime, r_adi, adia_v_dm, adia_x_dm, adia_v_disk):
    if rprime <= 0.:
        rprime = 0.1
    if rprime < adia_x_dm[1]:
        rprime = adia_x_dm[1]
    rprime_interp = scp_interp.interp1d(adia_x_dm, adia_v_dm,
                                        fill_value="extrapolate")
    result = (r_adi + r_adi * ((r_adi*adia_v_disk**2) /
                               (rprime*(rprime_interp(rprime))**2)) - rprime)
    # #
    # print("adia_x_dm={}, adia_v_dm={}, r_adi={}, rprime={}, rprime_interp(rprime)={}, result={}".format(adia_x_dm, adia_v_dm, r_adi, rprime,
    #             rprime_interp(rprime), result))
    return result
