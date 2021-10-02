# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# File containing base classes for DysmalPy models

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import abc
import logging

# Third party imports
import numpy as np
from astropy.modeling import Model
import astropy.constants as apy_con
import scipy.special as scp_spec

# Local imports
from dysmalpy.parameters import DysmalParameter

__all__ = ['MassModel', 'LightModel', 'LightModel3D'
           'v_circular', 'sersic_mr', 'truncate_sersic_mr']


# CONSTANTS
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')




# ***** Mass Component Model Classes ******
# Base abstract mass model component class
class _DysmalModel(Model):
    """
    Base abstract `dysmalpy` model component class
    """

    parameter_constraints = DysmalParameter.constraints

    def __setstate__(self, state):
        # Compatibility hack, to handle the changed galaxy structure
        #    (properties, not attributes for data[*], instrument)

        self.__dict__ = state

        # Compatibility hack, to handle the changes in astropy.modeling from v3 to v4
        if not self.param_names:
            pass
        else:
            if self.param_names[0] not in self.__dict__.keys():
                # If self.__dict__ doesn't contain param names,
                #       need to do v3 to v4 migration
                for pname in self.param_names:
                    # Fill param with correct values:
                    param = self.__getattribute__(pname)
                    start = self._param_metrics[pname]['slice'].start
                    stop = self._param_metrics[pname]['slice'].stop
                    param._value = self._parameters[start:stop]

                    keys_migrate = ['fixed', 'bounds', 'tied', 'prior']
                    for km in keys_migrate:
                        param.__dict__['_'+km] = self._constraints[km][pname]

                    # Set size:
                    self._param_metrics[pname]['size'] = np.size(param._value)

                    # Set param as part of model dict (v4 "standard")
                    self.__dict__[pname] = param

                if '_model' in param.__dict__.keys():
                    if param._model is None:
                        # If param._model exists and is missing,
                        # Back-set the model for all parameters, after model complete.
                        for pname in self.param_names:
                            param = self.__getattribute__(pname)
                            param._model = self
                            self.__setattr__(pname, param)


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


class _DysmalFittable3DModel(_DysmalModel):
    """
        Base class for 3D model components
    """

    linear = False
    fit_deriv = None
    col_fit_deriv = True
    fittable = True

    inputs = ('x', 'y', 'z')



class MassModel(_DysmalFittable1DModel):
    """
    Base model for components that exert a gravitational influence
    """

    _type = 'mass'
    _axisymmetric = True
    _potential_gradient_has_neg = False

    @abc.abstractmethod
    def enclosed_mass(self, *args, **kwargs):
        """Evaluate the enclosed mass as a function of radius"""


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


    def potential_gradient(self, r):
        r"""
        Default method to evaluate the gradient of the potential, :math:`\del\Phi(r)/\del r`.

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate circular velocity in kpc

        Returns
        -------
        dPhidr : float or array
            Gradient of the potential at `r`

        Notes
        -----
        Calculates the gradient of the potential from the circular velocity
        using :math:`\del\Phi(r)/\del r = v_{c}^2(r)/r`.
        An alternative should be written for components where the
        potential gradient is ever *negative* (i.e., rings).

        Can be coupled with setting & checking `model._potential_gradient_has_neg` flag
        for mass models.
        """
        vcirc = self.circular_velocity(r)
        dPhidr = vcirc ** 2 / r

        return dPhidr





class LightModel(_DysmalFittable1DModel):
    """
    Base model for components that emit light, but are treated separately from any gravitational influence
    Case: 1D profile.
    """

    _type = 'light'
    _axisymmetric = True

    @abc.abstractmethod
    def mass_to_light(self, *args, **kwargs):
        """Evaluate the enclosed mass as a function of radius"""




class LightModel3D(_DysmalFittable3DModel):
    """
    Base model for components that emit light, but are treated separately from any gravitational influence
    Case: 3D profile.
    """

    _type = 'light'
    _axisymmetric = False

    @abc.abstractmethod
    def mass_to_light(self, *args, **kwargs):
        """Evaluate the enclosed mass as a function of radius"""



#########################################

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

    # -------------------------
    # Test for 0:
    try:
        if len(r) >= 1:
            vcirc[np.array(r) == 0.] = 0.
    except:
        if r == 0.:
            vcirc = 0.
    # -------------------------

    return vcirc

#

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

def truncate_sersic_mr(r, mass, n, r_eff, r_inner, r_outer):
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

    r_inner: float
        Inner truncation radius

    r_outer: float
        Outer truncation radius

    Returns
    -------
    mr : float or array
        Surface mass density as a function of `r`
    """
    # Ensure it's an array:
    if isinstance(r*1., float):
        rarr = np.array([r])
    else:
        rarr = np.array(r)
    # Ensure all radii are 0. or positive:
    rarr = np.abs(rarr)

    mr = sersic_mr(rarr, mass, n, r_eff)

    wh_out = np.where((rarr < r_inner) | (rarr > r_outer))
    mr[wh_out] = 0.

    if (len(rarr) > 1):
        return mr
    else:
        if isinstance(r*1., float):
            # Float input
            return mr[0]
        else:
            # Length 1 array input
            return mr

def _I0_gaussring(r_peak, sigma_r, L_tot):
    x = r_peak / (sigma_r * np.sqrt(2.))
    Ih = np.sqrt(np.pi)*x*(1.+scp_spec.erf(x)) + np.exp(-x**2)
    I0 = L_tot / (2.*np.pi*(sigma_r**2)*Ih)
    return I0
