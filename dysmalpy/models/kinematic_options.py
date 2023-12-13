# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Kinematic options for DysmalPy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np
import scipy.special as scp_spec
import scipy.interpolate as scp_interp
import scipy.optimize as scp_opt

# Local imports
from .baryons import DiskBulge, LinearDiskBulge, Sersic, ExpDisk

__all__ = ['KinematicOptions']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")

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
    (assuming an exponential disk, with :math:`R_e=1.678r_d`):

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


    def apply_adiabatic_contract(self, model, r, vbaryon_sq, vhalo_sq,
                                 compute_dm=False, return_vsq=False,
                                 step1d = 0.2):
        """
        Function that applies adiabatic contraction to a ModelSet

        Parameters
        ----------
        model : `ModelSet`
            ModelSet that adiabatic contraction will be applied to

        r : array
            Radii in kpc

        vbaryon_sq : array
            Square of baryonic component circular velocities in km^2/s^2

        vhalo_sq : array
            Square of dark matter halo circular velocities in km^2/s^2

        compute_dm : bool
            If True, will return the adiabatically contracted halo velocities.

        return_vsq : bool
            If True, return square velocities instead of taking sqrt.

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

            # Define 1d radius array for calculation
            #step1d = 0.2  # kpc
            # r1d = np.arange(step1d, np.ceil(r.max()/step1d)*step1d+ step1d, step1d, dtype=np.float64)
            try:
                rmaxin = r.max()
            except:
                rmaxin = r

            try:
                r_ap = model._model_aperture_r()
            except:
                r_ap = 0.

            rmax_calc = max(5.* r_ap, rmaxin)

            # Wide enough radius range for full calculation -- out to 5*Reff, at least
            r1d = np.arange(step1d, np.ceil(rmax_calc/step1d)*step1d+ step1d, step1d, dtype=np.float64)


            rprime_all_1d = np.zeros(len(r1d))

            # Calculate vhalo, vbaryon on this 1D radius array [note r is a 3D array]
            vhalo1d_sq = r1d * 0.
            vbaryon1d_sq = r1d * 0.
            for cmp in model.mass_components:
                if model.mass_components[cmp]:
                    mcomp = model.components[cmp]

                    cmpnt_v_sq = mcomp.vcirc_sq(r1d)

                    if mcomp._subtype == 'dark_matter':
                        vhalo1d_sq = vhalo1d_sq + cmpnt_v_sq
                    elif mcomp._subtype == 'baryonic':
                        vbaryon1d_sq = vbaryon1d_sq + cmpnt_v_sq
                    elif mcomp._subtype == 'combined':
                        raise ValueError('Adiabatic contraction cannot be turned on when'
                                         'using a combined baryonic and halo mass model!')

                    else:
                        raise TypeError("{} mass model subtype not recognized"
                                        " for {} component. Only 'dark_matter'"
                                        " or 'baryonic' accepted.".format(mcomp._subtype, cmp))

            converged = np.zeros(len(r1d), dtype=bool)
            for i in range(len(r1d)):
                try:
                    result = scp_opt.newton(_adiabatic_sq, r1d[i] + 1.,
                                        args=(r1d[i], vhalo1d_sq, r1d, vbaryon1d_sq[i]),
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
                            result = r1d[i]
                            converged[i] = False
                # ------------------------------------------------------------------

                rprime_all_1d[i] = result

            ###########################
            vhalo_adi_interp_1d = scp_interp.interp1d(r1d, np.sqrt(vhalo1d_sq), fill_value='extrapolate', kind='linear')

            # Just calculations:
            if converged.sum() < len(r1d):
                if converged.sum() >= 0.9 *len(r1d):
                    rprime_all_1d = rprime_all_1d[converged]
                    r1d = r1d[converged]

            vhalo_adi_1d = vhalo_adi_interp_1d(rprime_all_1d)

            vhalo_adi_interp_map_3d = scp_interp.interp1d(r1d, vhalo_adi_1d, fill_value='extrapolate', kind='linear')

            vhalo_adi = vhalo_adi_interp_map_3d(r)

            vel_sq = vhalo_adi ** 2 + vbaryon_sq
        else:
            vel_sq = vhalo_sq + vbaryon_sq

        if return_vsq:
            if compute_dm:
                if self.adiabatic_contract:
                    return vel_sq, vhalo_adi ** 2
                else:
                    return vel_sq, vhalo_sq
            else:
                return vel_sq
        else:
            vel = np.sqrt(vel_sq)
            if compute_dm:
                if self.adiabatic_contract:
                    return vel, vhalo_adi
                else:
                    vhalo = np.sqrt(vhalo_sq)
                    return vel, vhalo
            else:
                return vel


    def apply_pressure_support(self, r, model, vel_sq, tracer=None):
        """
        Function to apply asymmetric drift correction

        Parameters
        ----------
        r : float or array
            Radius or radii at which to apply the correction

        model : `ModelSet`
            ModelSet for which the correction is applied to

        vel_sq : float or array
            Square of circular velocity in km^2/s^2

        tracer : string
            Name of the dynamical tracer (used to determine which is the appropriate dispersion profile).

        Returns
        -------
        vel_sq : float or array
            Square of rotational velocity with asymmetric drift applied, in km^2/s^2

        """
        if self.pressure_support:
            if tracer is None:
                raise ValueError("Must specify 'tracer' to determine pressure support!")

            vel_asymm_drift_sq = self.get_asymm_drift_profile(r, model, tracer=tracer)
            vel_squared = vel_sq - vel_asymm_drift_sq

            # if array:
            try:
                vel_squared[vel_squared < 0] = 0.
            except:
                # if float single value:
                if vel_squared < 0:
                    vel_squared = 0.
            #vel = np.sqrt(vel_squared)

        return vel_squared

    def correct_for_pressure_support(self, r, model, vel_sq, tracer=None):
        """
        Remove asymmetric drift effect from input velocities

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        model : `ModelSet`
            ModelSet the correction is applied to

        vel_sq : float or array
            Square of rotational velocities in km^2/s^2 from which to remove asymmetric drift

        tracer : string
            Name of the dynamical tracer (used to determine which is the appropriate dispersion profile).

        Returns
        -------
        vel_sq : float or array
            Square of circular velocity after asymmetric drift is removed, in km^2/s^2
        """
        if self.pressure_support:
            if tracer is None:
                raise ValueError("Must specify 'tracer' to determine pressure support!")

            vel_asymm_drift_sq = self.get_asymm_drift_profile(r, model, tracer=tracer)
            vel_squared = vel_sq + vel_asymm_drift_sq

            # if array:
            try:
                vel_squared[vel_squared < 0] = 0.
            except:
                # if float single value:
                if (vel_squared < 0):
                    vel_squared = 0.
            #vel = np.sqrt(vel_squared)

        return vel_squared

    def get_asymm_drift_profile(self, r, model, tracer=None):
        """
        Calculate the asymmetric drift correction

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        model : `ModelSet`
            ModelSet the correction is applied to

        tracer : string
            Name of the dynamical tracer (used to determine which is the appropriate dispersion profile).

        Returns
        -------
        vel_asymm_drift_sq : float or array
            Square velocity correction in km^2/s^2 associated with asymmetric drift
        """
        if tracer is None:
            raise ValueError("Must specify 'tracer' to determine pressure support!")

        # Compatibility hack, to handle the changed galaxy structure
        #    (properties, not attributes for data[*], instrument
        if 'pressure_support_type' not in self.__dict__.keys():
            # Set to default if missing
            self.pressure_support_type = 1
        if 'pressure_support_n' not in self.__dict__.keys():
            # Set to default if missing:
            self.pressure_support_n = None

        if (self.pressure_support_type == 1) | \
           (self.pressure_support_type == 2):
            pre = self.get_pressure_support_param(model, param='re')

        if tracer not in model.dispersions.keys():
            raise AttributeError("The dispersion profile for tracer={} not found!".format(tracer))

        sigma = model.dispersions[tracer](r)

        if self.pressure_support_type == 1:
            # Pure exponential derivation // n = 1
            vel_asymm_drift_sq = 3.36 * (r / pre) * sigma ** 2
        elif self.pressure_support_type == 2:
            # Modified derivation that takes into account n_disk / n
            pn = self.get_pressure_support_param(model, param='n')
            bn = scp_spec.gammaincinv(2. * pn, 0.5)

            vel_asymm_drift_sq = 2. * (bn/pn) * np.power((r/pre), 1./pn) * sigma**2

        elif self.pressure_support_type == 3:
            # Direct calculation from sig0^2 dlnrho/dlnr:
            # Assumes constant sig0 -- eg Eq 3, Burkert+10

            # NEEDS TO BE JUST RHO FOR THE GAS:
            dlnrhogas_dlnr = model.get_dlnrhogas_dlnr(r)
            vel_asymm_drift_sq = - dlnrhogas_dlnr * sigma**2

        return vel_asymm_drift_sq

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
                        elif (isinstance(mcomp, Sersic)) | (isinstance(mcomp, ExpDisk)):
                            p_val = mcomp.__getattribute__('{}'.format(p_altname)).value
                        break

            if p_val is None:
                if param == 're':
                    logger.warning("No disk baryonic mass component found. Using "
                               "1 kpc as the pressure support effective"
                               " radius")
                    p_val = 1.0
                elif param == 'n':
                    logger.warning("No disk baryonic mass component found. Using "
                               "n=1 as the pressure support Sersic index")
                    p_val = 1.0

        else:
            p_val = self.__dict__[paramkey]

        return p_val


def _adiabatic(rprime, r_adi, adia_v_dm, adia_x_dm, adia_v_disk):
    if rprime <= 0.:
        rprime = 0.1
    if rprime < adia_x_dm[1]:
        rprime = adia_x_dm[1]
    rprime_interp = scp_interp.interp1d(adia_x_dm, adia_v_dm,
                                        fill_value="extrapolate")
    result = (r_adi + r_adi * ((r_adi*adia_v_disk**2) /
                               (rprime*(rprime_interp(rprime))**2)) - rprime)

    return result

def _adiabatic_sq(rprime, r_adi, adia_v_dm_sq, adia_x_dm, adia_v_disk_sq):
    if rprime <= 0.:
        rprime = 0.1
    if rprime < adia_x_dm[1]:
        rprime = adia_x_dm[1]
    rprime_interp = scp_interp.interp1d(adia_x_dm, np.sqrt(adia_v_dm_sq),
                                        fill_value="extrapolate")
    result = (r_adi + r_adi * ((r_adi*adia_v_disk_sq) /
                               (rprime*(rprime_interp(rprime))**2)) - rprime)

    return result
