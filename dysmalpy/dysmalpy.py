# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is the main module to run DYSMALPY.
"""

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

# Third Party
import numpy as np
import scipy.special as scp_spec
import scipy.interpolate as scp_interp
import scipy.io as scp_io
import scipy.optimize as scp_opt
import astropy.constants as apy_con
import astropy.units as u
import astropy.cosmology as apy_cosmo
import astropy.convolution as apy_conv
from astropy.modeling import Fittable1DModel, Parameter
from astropy.wcs import WCS
from radio_beam import Beam
from spectral_cube import SpectralCube


__all__ = ['Galaxy', 'Sersic', 'NFW', 'calc_rvir']
__version__ = '0.1'
__author__ = ''

# Set the cosmology that will be assumed throughout
cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)

# Necessary constants
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc

# Directories
_dir_noordermeer = '/Users/ttshimiz/Dropbox/Research/LLAMA/dysmal/noordermeer/'


class Galaxy:
    """
    The main object for simulating the kinematics of a galaxy based on
    user provided mass components.
    """

    def __init__(self, redshift, name='galaxy'):
        self.z = redshift
        self.name = name
        self.mass_model = None
        self._comp_names = []
        self._serc_comp = []
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

        serc_mod = Sersic(total_mass=mass, r_eff=re, n=n, invq=invq, name=name)
        self._comp_names.append(name)
        self._serc_comp.append(True)
        self._thick.append(2 * re / (invq * 2.35482))
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
            raise ValueError('Either a value for rvirial must be provided or '
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
        self._comp_names.append(name)
        self._serc_comp.append(False)
        self._thick.append(0)  # No flattening for DM halo
        self._light.append(False)  # No light component for DM halo

        if self.mass_model is None:
            self.mass_model = nfw_mod
        else:
            self.mass_model += nfw_mod

    def velocity1d(self, r, noord_flat=False, adi_contract=False,
                   pressure_support=False, psupport_re=None, turb=None):

        vhalo = np.zeros(len(r))
        vbaryon = np.zeros(len(r))

        for i, n in enumerate(self.mass_model.submodel_names):

            # If its not a Sersic component, assume its an NFW component
            # and add to the halo velocity component
            if not self._serc_comp[i]:

                cmpnt_v = self.mass_model[n].circular_velocity(r)
                vhalo = np.sqrt(vhalo**2 + cmpnt_v**2)

            else:

                cmpnt_v = self.mass_model[n].circular_velocity(
                    r, noord_flat=noord_flat)
                vbaryon = np.sqrt(vbaryon**2 + cmpnt_v**2)

        # Perform adiabatic contraction
        if adi_contract:
            rprime_all = np.zeros(len(r))
            for i in range(len(r)):
                result = scp_opt.newton(_adiabatic, r[i] + 1.,
                                        args=(r[i], vhalo, r, vbaryon[i]))
                rprime_all[i] = result

            vhalo_adi_interp = scp_interp.interp1d(r, vhalo,
                                                   fill_value='extrapolate')
            vhalo_adi = vhalo_adi_interp(rprime_all)
            vel = np.sqrt(vhalo_adi**2 + vbaryon**2)

        else:
            vel = np.sqrt(vhalo**2 + vbaryon**2)

        # Apply pressure support
        if pressure_support:

            if psupport_re is None:
                first_serc = np.array(self.mass_model.submodel_names)[self._serc_comp][0]
                psupport_re = self.mass_model[first_serc].r_eff.value

            if turb is None:
                raise ValueError("Pressure support can't be applied without a"
                                 "value for the turbulence.")

            else:

                sigma0 = turb / 2.35842
                vel_squared = (vel**2 - 3.36 * (r/psupport_re) * sigma0 ** 2)
                vel_squared[vel_squared < 0] = 0.
                vel = np.sqrt(vel_squared)

        return vel

    def simulate_cube(self, inc, pa, nx_sky, ny_sky, rstep, velmax, velstep,
                      xshift=0, yshift=0, vshift=0, vrad=0, noise=0,
                      beam_smear=True, beamsize=None, noord_flat=False,
                      adi_contract=False, pressure_support=False,
                      psupport_re=None, turb=None):

        # Simulate an IFU observation of this galaxy
        # Adjust position angle to be north of west instead of east of north
        inc = np.pi / 180. * inc
        pa = np.pi / 180. * (pa - 90.)

        # Start with a 3D array in the sky coordinate system
        # x and y sizes are user provided so we just need
        # the z size where z is in the direction of the L.O.S.
        # We'll just use the maximum of the given x and y
        nz_sky = np.max([nx_sky, ny_sky])

        # Create 3D arrays of the sky pixel coordinates
        # Shift them so that (0,0,0) is in the center of the galaxy
        sh = (nz_sky, ny_sky, nx_sky)
        zsky, ysky, xsky = np.indices(sh)
        zsky = zsky - (nz_sky - 1)/2.
        xsky = xsky - (nx_sky - 1)/2. - xshift
        ysky = ysky - (ny_sky - 1)/2. - yshift

        # Transform from sky coordinates to galaxy plane coordinates
        # 1. Rotate around zsky axis first so line of nodes lies along xsky axis
        # 2. Rotate around xsky axis to remove inclination
        xtmp = xsky*np.cos(pa) + ysky*np.sin(pa)
        ytmp = -xsky*np.sin(pa) + ysky*np.cos(pa)
        ztmp = zsky

        xgal = xtmp
        ygal = ytmp*np.cos(inc) - ztmp*np.sin(inc)
        zgal = ytmp*np.sin(inc) + ztmp*np.cos(inc)

        # The circular velocity at each position only depends on the radius
        # Convert to kpc
        rgal = np.sqrt(xgal**2 + ygal**2)*rstep/self.dscale
        vcirc = self.velocity1d(rgal, noord_flat=noord_flat,
                                adi_contract=adi_contract,
                                pressure_support=pressure_support,
                                psupport_re=psupport_re,
                                turb=turb)

        # L.O.S. velocity is then just vcirc*sin(i)*cos(theta) where theta
        # is the position angle in the plane of the disk
        # cos(theta) is just xgal/rgal
        vobs = vshift + vcirc*np.sin(inc)*xgal/(rgal/rstep*self.dscale)
        vobs[rgal == 0] = 0.

        # Calculate "flux" for each position
        flux = np.zeros(vobs.shape)
        for i, n in enumerate(self.mass_model.submodel_names):
            if self._light[i]:
                cpt_mass = 10**self.mass_model[n].total_mass.value
                hz = self._thick[i]*self.dscale/rstep
                zscale = np.exp(-0.5*(zgal/hz)**2)
                flux += self.mass_model[n](rgal)/cpt_mass*zscale

        # Begin constructing the IFU cube
        vx = np.arange(np.int(2 * velmax / velstep + 1)) * velstep - velmax
        nv = len(vx)
        velcube = np.tile(np.resize(vx, (nv, 1, 1)), (1, ny_sky, nx_sky))
        cube_final = np.zeros((nv, ny_sky, nx_sky))

        # The final spectrum will be a flux weighted sum of Gaussians at each
        # velocity along the line of sight.
        for zz in range(nz_sky):
            f_cube = np.tile(flux[zz, :, :], (nv, 1, 1))
            vobs_cube = np.tile(vobs[zz, :, :], (nv, 1, 1))
            tmp_cube = np.exp(-0.5*((velcube - vobs_cube)/(turb/2.355))**2)
            cube_final += tmp_cube/np.sum(tmp_cube, 0)*100.*f_cube

        # Apply beam smearing if requested
        if beam_smear:
            beam = Beam(major=beamsize * u.arcsec,
                        minor=beamsize * u.arcsec,
                        pa=0. * u.deg)
            beam_kernel = beam.as_kernel(rstep*u.arcsec)

            for i in range(nv):
                cube_final[i, :, :] = apy_conv.convolve_fft(cube_final[i, :, :],
                                                            beam_kernel)

        # Create a simple header for the simulate cube
        w = WCS(naxis=3)
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'VOPT']
        w.wcs.cdelt = [rstep / 3600., rstep / 3600., velstep]
        w.wcs.crpix = [1, 1, 1]
        w.wcs.cunit = ['deg', 'deg', 'km/s']
        w.wcs.crval = [xsky[0, 0, 0]*rstep, ysky[0, 0, 0]*rstep, vx[0]]
        spec_cube = SpectralCube(data=cube_final, wcs=w)


        return spec_cube


class Sersic(Fittable1DModel):
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
        """1D Sersic profile parameterized by the total mass and scale height"""

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

            # Mass scaling to test out code
            rscale = 16.94174001289003
            mscale = self.enclosed_mass(rscale)

            v_interp = scp_interp.interp1d(N2008_rad, N2008_vcirc,
                                           fill_value="extrapolate")
            vcirc = (v_interp(r/self.r_eff*N2008_Re)*np.sqrt(
                mscale/N2008_mass)*np.sqrt(N2008_Re/self.r_eff))

        else:

            mass_enc = self.enclosed_mass(r)
            vcirc = np.sqrt(G.cgs.value*mass_enc*Msun.cgs.value /
                            (r*1000.*pc.cgs.value))
            vcirc = vcirc/1e5

        return vcirc


class NFW(Fittable1DModel):
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

    def circular_velocity(self, r):

        mass_enc = self.enclosed_mass(r)
        vcirc = np.sqrt(G.cgs.value*mass_enc*Msun.cgs.value /
                        (r*1000.*pc.cgs.value))
        vcirc = vcirc/1e5

        return vcirc


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
