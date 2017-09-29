# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# File containing all of the available models to use build the
# galaxy

# Standard library
import abc
import logging
from collections import OrderedDict

# Third party imports
import numpy as np
import scipy.special as scp_spec
import scipy.io as scp_io
import scipy.interpolate as scp_interp
import astropy.constants as apy_con
import astropy.units as u
from astropy.modeling import Model

# Local imports
from parameters import DysmalParameter

__all__ = ['ModelSet', 'MassModel', 'Sersic', 'NFW', 'Geometry']

# NOORDERMEER DIRECTORY
_dir_noordermeer = "data/noordermeer/"

# CONSTANTS
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


# Generic model container which tracks all components, parameters,
# parameter settings, model settings, etc.
class ModelSet:

    def __init__(self):

        self.mass_components = OrderedDict()
        self.components = OrderedDict()
        self.geometry = None
        self.parameters = None
        self.fixed = OrderedDict()
        self.param_names = OrderedDict()
        self._param_keys = OrderedDict()
        self.nparams = 0
        self.nparams_free = 0

    def add_component(self, model, name=None):
        """Add a model component to the set"""

        # Check to make sure its the correct class
        if isinstance(model, _DysmalModel):
            if model._type == 'mass':

                if (name is None) & (model.name is None):
                    raise ValueError('Please give this component a name!')

                elif name is not None:
                    model = model.rename(name)

                # Make sure there isn't a mass component already named this
                if list(self.mass_components.keys()).count(model.name) > 0:
                    raise ValueError('Component already exists. Please give'
                                     'it a unique name.')
                else:
                    self.mass_components[model.name] = True

            elif model._type == 'geometry':
                if self.geometry is not None:
                    logger.warning('Current Geometry model is being '
                                   'overwritten!')
                self.geometry = model
                self.mass_components[model.name] = False

            else:
                raise TypeError("This model type is not known! Must be either"
                                "'mass' or 'geometry.'")

            self._add_comp(model)

        else:

            raise TypeError('Model component must be an an'
                            'dysmalpy.models.DysmalModel instance!')

    def _add_comp(self, model):

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

        # Update the dictionaries containing the locations of the parameters
        # in the full and only free arrays
        key_dict = OrderedDict()
        for i, p in enumerate(model.param_names):
            key_dict[p] = i + self.nparams

        self._param_keys[model.name] = key_dict
        self.nparams += len(model.param_names)
        self.nparams_free += len(model.param_names) - sum(model.fixed.values())

    def set_parameter_value(self, model_name, param_name, value):
        """Method to set a specific parameter value"""

        try:
            comp = self.components[model_name]
        except KeyError:
            raise KeyError('Model not part of the set.')

        try:
            param_i = comp.param_names.index(param_name)
        except ValueError:
            raise ValueError('Parameter is not part of model.')

        self.components[model_name].parameters[param_i] = value
        self.parameters[self._param_keys[model_name][param_name]] = value

    def set_parameter_fixed(self, model_name, param_name, fix):
        """Method to set a specific parameter fixed or not"""

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
        """Update all of the free parameters of the model"""

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

    def _get_free_parameters(self):
        p = np.zeros(self.nparams_free)
        pkeys = OrderedDict()
        j = 0
        for cmp in self.fixed:
            pkeys[cmp] = OrderedDict()
            for pm in self.fixed[cmp]:
                if self.fixed[cmp][pm]:
                    pkeys[cmp][pm] = -99
                else:
                    pkeys[cmp][pm] = j
                    p[j] = self.parameters[self._param_keys[cmp][pm]]
                    j += 1
        return p, pkeys

    def get_free_parameters(self):
        pfree, pfree_keys = self._get_free_parameters()
        return pfree

    def get_free_parameter_keys(self):
        pfree, pfree_keys = self._get_free_parameters()
        return pfree_keys



# ***** Mass Component Model Classes ******
# Base abstract mass model component class
class _DysmalModel(Model):

    parameter_constraints = DysmalParameter.constraints

    @property
    def prior(self):
        return self._constraints['prior']


class _DysmalFittable1DModel(_DysmalModel):

    linear = False
    fit_deriv = None
    col_fit_deriv = True
    fittable = True

    inputs = ('x',)
    outputs = ('y',)


class MassModel(_DysmalFittable1DModel):

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

    total_mass = DysmalParameter(default=1)
    r_eff = DysmalParameter(default=1)
    n = DysmalParameter(default=1)

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

    mvirial = DysmalParameter(default=1.0)
    rvirial = DysmalParameter(default=1.0)
    conc = DysmalParameter(default=5.0)

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
class _DysmalFittable3DModel(_DysmalModel):

    linear = False
    fit_deriv = None
    col_fit_deriv = True
    fittable = True

    inputs = ('x', 'y', 'z')
    outputs = ('xp', 'yp', 'zp')

    @property
    def prior(self):
        return self._constraints['prior']


class Geometry(_DysmalFittable3DModel):
    """
    Class to hold the geometric parameters that can be fit.
    Also takes as input the sky coordinates and returns the
    corresponding galaxy plane coordinates.
    """

    inc = DysmalParameter(default=45.0, bounds=(0, 90))
    pa = DysmalParameter(default=0.0, bounds=(-180, 180))
    xshift = DysmalParameter(default=0.0)
    yshift = DysmalParameter(default=0.0)

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

        return xgal, ygal, zgal


def calc_rvir(mvirial, z):
    """
    Calculate the virial radius based on virial mass and redshift
    M_vir = 100*H(z)^2/G * R_vir^3

    :param mvirial: Virial mass in log(Msun)
    :param z: Redshift
    :return: rvirial: Virial radius in kpc
    """
    g_new_unit = G.to(u.pc/u.Msun*(u.km/u.s)**2).value
    Hz = cosmo.H(z).value
    rvir = ((10**mvirial * (g_new_unit * 1e-3) /
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
