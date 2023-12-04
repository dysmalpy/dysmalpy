# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
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
from dysmalpy.parameters import DysmalParameter, UniformPrior

try:
    from dysmalpy.models import utils
except:
   from . import utils

__all__ = ['MassModel', 'LightModel',
           'HigherOrderKinematicsSeparate', 'HigherOrderKinematicsPerturbation'
           'v_circular', 'menc_from_vcirc', 'sersic_mr', 'truncate_sersic_mr']


# CONSTANTS
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)




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

        # Compatibility hacks, to handle the changes in astropy.modeling from v3 to v4
        if '_inputs' not in state.keys():
            #inputs = ('x', 'y', 'z') or ('x',)
            if len(state['_input_units_strict'].keys()) == 1:
                self.__dict__['_inputs'] =  ('x',)
            elif len(state['_input_units_strict'].keys()) == 3:
                self.__dict__['_inputs'] =  ('x', 'y', 'z')
        if '_outputs' not in state.keys():
            # Should only be the 1D case
            if len(state['_input_units_strict'].keys()) == 1:
                self.__dict__['_outputs'] =  ('y',)


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

    # inputs = ('x',)
    # outputs = ('y',)
    n_inputs = 1
    n_outputs = 1


class _DysmalFittable3DModel(_DysmalModel):
    """
        Base class for 3D model components
    """

    linear = False
    fit_deriv = None
    col_fit_deriv = True
    fittable = True

    # inputs = ('x', 'y', 'z')
    n_inputs = 3



class MassModel(_DysmalFittable1DModel):
    """
    Base model for components that exert a gravitational influence
    """

    _type = 'mass'
    _axisymmetric = True
    _multicoord_velocity = False
    _native_geometry = 'cylindrical'  ## possibility for further vel direction abstraction

    @property
    @abc.abstractmethod
    def _subtype(self):
        pass

    @abc.abstractmethod
    def enclosed_mass(self, *args, **kwargs):
        """Evaluate the enclosed mass as a function of radius"""
        pass

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

    def vcirc_sq(self, r):
        r"""
        Default method to evaluate the square of the circular velocity

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate circular velocity in kpc

        Returns
        -------
        vcirc_sq : float or array
            Square of circular velocity at `r`

        Notes
        -----
        Calculates the circular velocity as a function of radius
        as just the square of self.circular_velocity().

        This can be overwritten for inheriting classes with negative potential gradients.
        """
        return self.circular_velocity(r)**2

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

        """
        vcirc = self.circular_velocity(r)
        dPhidr = vcirc ** 2 / r

        return dPhidr


    def vel_direction_emitframe(self, xgal, ygal, zgal, _save_memory=False):
        r"""
        Default method to return the velocity direction in the galaxy Cartesian frame.

        Parameters
        ----------
        xgal, ygal, zgal : float or array
            xyz position in the galaxy reference frame.

        _save_memory : bool, optional
            Option to save memory by only calculating the relevant matrices (eg during fitting).
            Default: False

        Returns
        -------
        vel_dir_unit_vector : 3-element array
            Direction of the velocity vector in (xyzgal).

            As this is the base mass model, assumes the velocity direction
            is the phi direction in cylindrical coordinates, (R,phi,z).
        """
        rgal = np.sqrt(xgal ** 2 + ygal ** 2)

        vhat_y = xgal/rgal

        # Excise rgal=0 values
        vhat_y = utils.replace_values_by_refarr(vhat_y, rgal, 0., 0.)

        if not _save_memory:
            vhat_x = -ygal/rgal
            vhat_z = 0.*zgal

            # Excise rgal=0 values
            vhat_x = utils.replace_values_by_refarr(vhat_x, rgal, 0., 0.)
            vhat_z = utils.replace_values_by_refarr(vhat_z, rgal, 0., 0.)

            vel_dir_unit_vector = np.array([vhat_x, vhat_y, vhat_z])
        else:
            # Only calculate y values
            vel_dir_unit_vector = [0., vhat_y, 0.]

        return vel_dir_unit_vector


    def velocity_vector(self, xgal, ygal, zgal, vel=None, _save_memory=False):
        """ Return the relevant velocity -- if not specified, call self.circular_velocity() --
            as a vector in the the reference Cartesian frame coordinates. """
        if vel is None:
            vel = self.circular_velocity(np.sqrt(xgal**2 + ygal**2))

        vel_hat = self.vel_direction_emitframe(xgal, ygal, zgal, _save_memory=_save_memory)

        if not _save_memory:
            vel_cartesian = vel * vel_hat
        else:
            # Only calculated y direction, as this is cylindrical only
            if self._native_geometry == 'cylindrical':
                vel_cartesian = [0., vel*vel_hat[1], 0.]
            else:
                raise ValueError("all mass models assumed to be cylindrical for memory saving!")

        return vel_cartesian


class LightModel(_DysmalModel):
    """
    Base model for components that emit light, but are treated separately from any gravitational influence
    """

    _type = 'light'
    _axisymmetric = True

    @abc.abstractmethod
    def light_profile(self, *args, **kwargs):
        """Evaluate the enclosed mass as a function of radius"""


class _LightMassModel(_DysmalModel):
    """
    Abstract model for mass model that also emits light
    """

    def __setstate__(self, state):
        if 'mass_to_light' not in state.keys():
            state = utils.insert_param_state(state, 'mass_to_light', value=1.,
                                             fixed=True, tied=False,
                                             bounds=(0.,10.), prior=UniformPrior)

        super(_LightMassModel, self).__setstate__(state)





class HigherOrderKinematics(_DysmalModel):
    """
    Base model for higher-order kinematic components
    """

    _type = 'higher_order'

    @property
    @abc.abstractmethod
    def _native_geometry(self):
        pass

    @property
    @abc.abstractmethod
    def _higher_order_type(self):
        pass

    @property
    @abc.abstractmethod
    def _separate_light_profile(self):
        pass

    @property
    @abc.abstractmethod
    def _spatial_type(self):
        pass

    @property
    @abc.abstractmethod
    def _multicoord_velocity(self):
        pass

    @abc.abstractmethod
    def velocity(self, *args, **kwargs):
        """Method to return the velocity amplitude (in the output geometry Cartesian frame,
           if self._multicoord_velocity==True)."""
        pass

    @abc.abstractmethod
    def vel_direction_emitframe(self, *args, **kwargs):
        """Method to return the velocity direction in the output geometry Cartesian frame."""
        pass


    def velocity_vector(self, x, y, z, vel=None, _save_memory=False):
        """ Return the velocity -- calling self.velocity() if vel is None -- of the higher order
            component as a vector in the the reference Cartesian frame coordinates. """
        if vel is None:
            vel = self.velocity(x, y, z)

        # Dot product of vel_hat, zsky_unit_vector
        if self._multicoord_velocity:
            # Matrix multiply the velocity direction matrix with the
            #   oritinal velocity tuple, then dot product with the zsky unit vector
            vel_dir_matrix = self.vel_direction_emitframe(x, y, z, _save_memory=_save_memory)

            # Need to explicity work this out, as this is a 3x3 matrix multiplication
            #   with a 3-element vector, where the elements themselves are arrays...
            vel_cartesian = [vel[0]*0., vel[1]*0., vel[2]*0.]
            for row in range(vel_dir_matrix.shape[0]):
                for col in range(vel_dir_matrix.shape[1]):
                    vel_cartesian[row] += vel_dir_matrix[row, col] * vel[col]

        else:
            # Simply apply magnitude to velhat
            vel_hat = self.vel_direction_emitframe(x, y, z, _save_memory=_save_memory)
            if not _save_memory:
                vel_cartesian = vel * vel_hat
            else:
                # Only calculated y,z directions
                vel_cartesian = [0., vel*vel_hat[1], vel*vel_hat[2]]

        return vel_cartesian


class HigherOrderKinematicsSeparate(HigherOrderKinematics):
    """
    Base model for higher-order kinematic components that are separate from the galaxy.
    Have separate light profiles, and can have separate geometry/dispersion components.
    """

    _higher_order_type = 'separate'
    _separate_light_profile = True


class HigherOrderKinematicsPerturbation(HigherOrderKinematics):
    """
    Base model for higher-order kinematic components that are perturbations to the galaxy.
    Cannot have a separate light/geometry/dispersion components.
    However, they can have light that is then *added* to the galaxy,
        by adding to the ModelSet with light=True.
    """

    _higher_order_type = 'perturbation'
    _separate_light_profile = False
    _axisymmetric = False


#########################################

def v_circular(mass_enc, r):
    r"""
    Circular velocity given an enclosed mass and radius

    .. math:: 
        v(r) = \sqrt{(GM(r)/r)}

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
