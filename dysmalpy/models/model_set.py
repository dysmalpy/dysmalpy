# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Handling of DysmalPy ModelSets (and Models) to use build the galaxy model

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import os
import logging
import warnings
from collections import OrderedDict

# Local imports
from .base import _DysmalModel, menc_from_vcirc
from .kinematic_options import KinematicOptions
from .dimming import ConstantDimming

try:
   import dysmalpy.models.utils as model_utils
except:
   from . import utils as model_utils


# Third party imports
import numpy as np
import astropy.constants as apy_con
import astropy.units as u
import pyximport; pyximport.install()
from . import cutils


__all__ = ['ModelSet']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")

def _make_cube_ai(model, xgal, ygal, zgal, n_wholepix_z_min = 3,
            pixscale=None, oversample=None, dscale=None,
            maxr=None, maxr_y=None):

    oversize = 1.5  # Padding factor for x trimming

    thick = model.zprofile.z_scalelength.value
    if not np.isfinite(thick):
        thick = 0.

    # # maxr, maxr_y are already in pixel units
    xsize = int(np.floor(2.*(maxr * oversize) +0.5))
    ysize = int(np.floor( 2.*maxr_y + 0.5))

    # Sample += 2 * scale length thickness
    # Modify: make sure there are at least 3 *whole* pixels sampled:
    zsize = np.max([ n_wholepix_z_min*oversample, int(np.floor(4.*thick/pixscale*dscale + 0.5 )) ])

    if ( (xsize%2) < 0.5 ): xsize += 1
    if ( (ysize%2) < 0.5 ): ysize += 1
    if ( (zsize%2) < 0.5 ): zsize += 1

    zi, yi, xi = np.indices(xgal.shape)
    full_ai = np.vstack([xi.flatten(), yi.flatten(), zi.flatten()])

    origpos = np.vstack([xgal.flatten() - np.mean(xgal.flatten()) + xsize/2.,
                         ygal.flatten() - np.mean(ygal.flatten()) + ysize/2.,
                         zgal.flatten() - np.mean(zgal.flatten()) + zsize/2.])


    validpts = np.where( (origpos[0,:] >= 0.) & (origpos[0,:] <= xsize) & \
                         (origpos[1,:] >= 0.) & (origpos[1,:] <= ysize) & \
                         (origpos[2,:] >= 0.) & (origpos[2,:] <= zsize) )[0]


    ai = full_ai[:,validpts]

    return ai


def _get_xyz_sky_gal(geom, sh, xc_samp, yc_samp, zc_samp):
    """ Get grids in xyz sky, galaxy frames, assuming regularly gridded in sky frame """
    zsky, ysky, xsky = np.indices(sh)
    zsky = zsky - zc_samp
    ysky = ysky - yc_samp
    xsky = xsky - xc_samp
    xgal, ygal, zgal = geom(xsky, ysky, zsky)
    return xgal, ygal, zgal, xsky, ysky, zsky

def _get_xyz_sky_gal_inverse(geom, sh, xc_samp, yc_samp, zc_samp):
    """ Get grids in xyz sky, galaxy frames, assuming regularly gridded in galaxy frame """
    zgal, ygal, xgal = np.indices(sh)
    zgal = zgal - zc_samp
    ygal = ygal - yc_samp
    xgal = xgal - xc_samp
    xsky, ysky, zsky = geom.inverse_coord_transform(xgal, ygal, zgal)
    return xgal, ygal, zgal, xsky, ysky, zsky

def _calculate_max_skyframe_extents(geom, nx_sky_samp, ny_sky_samp, transform_method, angle='cos'):
    """ Calculate max zsky sample size, given geometry """
    maxr = np.sqrt(nx_sky_samp**2 + ny_sky_samp**2)
    if transform_method.lower().strip() == 'direct':
        if angle.lower().strip() == 'cos':
            cos_inc = np.cos(geom.inc*np.pi/180.)
            geom_fac = cos_inc
        elif angle.lower().strip() == 'sin':
            sin_inc = np.sin(geom.inc*np.pi/180.)
            geom_fac = sin_inc
        else:
            raise ValueError
        maxr_y = np.max(np.array([maxr*1.5, np.min(
            np.hstack([maxr*1.5/ geom_fac, maxr * 5.]))]))
    else:
        maxr_y = maxr * 5. #1.5

    if angle.lower().strip() == 'cos':
        nz_sky_samp = int(np.max([nx_sky_samp, ny_sky_samp]))
    elif angle.lower().strip() == 'sin':
        nz_sky_samp = int(np.max([nx_sky_samp, ny_sky_samp, maxr_y]))
    if np.mod(nz_sky_samp, 2) < 0.5:
        nz_sky_samp += 1

    return nz_sky_samp, maxr, maxr_y



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
        self.geometries = OrderedDict()
        self.dispersions = OrderedDict()
        self.zprofile = None

        # Keep higher-order kinematic components, and their geom/disp/fluxes in OrderedDicts:
        #       (biconical) outflow / uniform flow / etc
        self.higher_order_components = OrderedDict()
        self.higher_order_geometries = OrderedDict()
        self.higher_order_dispersions = OrderedDict()

        self.dimming = None
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
        self.dimming = ConstantDimming()
        self.line_center = None

        # Option for dealing with 3D data:
        self.per_spaxel_norm_3D = False

        # Options for defining fDM apertures:
        # self.model_key_re = ['disk+bulge','r_eff_disk']
        self.model_key_halo=['halo']


    def _model_aperture_r(self, model_key_re = ['disk+bulge','r_eff_disk']):
        """
        Default radius aperture function: returns Reff of the disk
        Change to a function that returns a different radius (including const)
        to get other aperture values
        """
        # Default: use old behavior of model_key_re = ['disk+bulge','r_eff_disk']:

        comp = self.components.__getitem__(model_key_re[0])
        param_i = comp.param_names.index(model_key_re[1])
        r_ap = comp.parameters[param_i]

        return r_ap

    def __setstate__(self, state):
        # Compatibility hack, to handle the change to generalized
        #    higher order components in ModelSet.simulate_cube().
        self.__dict__ = state

        # Check param name order, in case it's changed since the object was pickled:
        for key in self.components.keys():
            if list(self.components[key]._param_metrics.keys()) != list(self.components[key].param_names):
                # Reset param name order to match slice order:
                self.components[key].param_names = tuple(self.components[key]._param_metrics.keys())

        # quick test if necessary to migrate:
        if 'higher_order_components' not in state.keys():
            new_keys = ['higher_order_components', 'higher_order_geometries',
                        'higher_order_dispersions']
            for nkey in new_keys:
                self.__dict__[nkey] = OrderedDict()

            # If there are nonzero higher-order kin components, migrate them:
            # migrate_keys = ['outflow', 'outflow_geometry', 'outflow_dispersion',
            #                 'flow', 'flow_geometry', 'flow_dispersion']
            migrate_keys = ['outflow', 'flow']
            ends = ['', '_geometry', '_dispersion']
            for mkeyb in migrate_keys:
                if mkeyb in state.keys():
                    for e in ends:
                        mkey = mkeyb+e
                        if state[mkey] is not None:
                            if e == '':
                                self.higher_order_components[state[mkey].name] = state[mkey]
                            elif e == '_geometry':
                                self.higher_order_geometries[state[mkeyb].name] = state[mkey]
                            elif e == '_dispersion':
                                self.higher_order_dispersions[state[mkeyb].name] = state[mkey]
                            self.mass_components[state[mkey].name] = False

            # Add dimming, if missing
            if 'dimming' not in state.keys():
                self.dimming = ConstantDimming()

            # Cleanup old names:
            del_keys = ['outflow', 'outflow_geometry', 'outflow_dispersion', 'outflow_flux',
                        'inflow', 'inflow_geometry', 'inflow_dispersion', 'inflow_flux',
                        'flow', 'flow_geometry', 'flow_dispersion', 'flow_flux']
            for dkey in del_keys:
                if dkey in self.__dict__.keys():
                    del self.__dict__[dkey]


        # Migrate to new-multi-obs/multi-tracer framework:
        if 'dispersion_profile' in state.keys():
            self.dispersions  = OrderedDict()
            self.dispersions['LINE'] = state['dispersion_profile']
            del self.__dict__['dispersion_profile']
        if 'geometry' in state.keys():
            self.geometries  = OrderedDict()
            self.geometries['OBS'] = state['geometry']
            del self.__dict__['geometry']


    def add_component(self, model, name=None, light=False,
                      geom_type='galaxy', disp_type='galaxy'):
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

        geom_type : {'galaxy', component name}
            Specify which model components the geometry applies to.
            Only used if `model` is a `~Geometry`. If 'galaxy', then all included
            components except named components with other geometries will follow this geometry
            (i.e., those with a separate geometry included that has 'geom_type'=the component name).
            Default is 'galaxy'.

        disp_type : {'galaxy', component name}
            Specify which model components the dispersion applies to
            (by name, with the exception of the default profiles).
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
                    if model.obs_name in self.geometries.keys():
                        if (self.geometries[model.obs_name] is not None):
                            wrn = "Current Geometry model '{}' is being ".format(model.obs_name)
                            wrn += "overwritten!"
                            logger.warning(wrn)
                    self.geometries[model.obs_name] = model
                else:
                    self.higher_order_geometries[geom_type] = model

                self.mass_components[model.name] = False

            elif model._type == 'dispersion':
                if disp_type == 'galaxy':
                    if model.tracer in self.dispersions.keys():
                        if (self.dispersions[model.tracer] is not None):
                            wrn = "Current Dispersion model '{}' is being ".format(model.tracer)
                            wrn += "overwritten!"
                            logger.warning(wrn)
                    self.dispersions[model.tracer] = model
                else:
                    self.higher_order_dispersions[disp_type] = model
                self.mass_components[model.name] = False

            elif model._type == 'zheight':
                if self.zprofile is not None:
                    logger.warning('Current z-height model is being '
                                   'overwritten!')
                self.zprofile = model
                self.mass_components[model.name] = False

            elif model._type == 'higher_order':
                self.higher_order_components[model.name] = model
                self.mass_components[model.name] = False

            elif model._type == 'extinction':
                if self.extinction is not None:
                    logger.warning('Current extinction model is being overwritten!')
                self.extinction = model
                self.mass_components[model.name] = False

            elif model._type == 'light':
                if not light:
                    light = True
                self.mass_components[model.name] = False
            else:
                raise TypeError("This model type is not known. Must be one of"
                                "'mass', 'geometry', 'dispersion', 'zheight',"
                                "'higher_order', or 'extinction'.")

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

        # Ensure the tied parameters have fixed=False:
        for pn in model.param_names:
            if not model.tied[pn]:
                # Parameter not tied
                pass
            else:
                # Parameter is tied
                model.fixed[pn] = False

        # Update the components list
        self.components[model.name] = model

        # Update the parameters and parameters_free arrays
        if self.parameters is None:
            self.parameters = model.parameters
        else:

            self.parameters = np.concatenate([self.parameters,
                                              model.parameters])
        self.param_names[model.name] = model.param_names
        self.tied[model.name] = model.tied
        self.fixed[model.name] = model.fixed

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

        # Now update all of the tied parameters if there are any
        # Wrap in try/except, to avoid issues in component adding order:
        try:
            self._update_tied_parameters()
        except:
            pass

    def set_parameter_value(self, model_name, param_name, value, skip_updated_tied=False):
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

        if not skip_updated_tied:
            # Now update all of the tied parameters if there are any
            self._update_tied_parameters()

            # if param_name in ['n', 'n_disk', 'n_bulge',
            #                   'invq', 'invq_disk', 'invq_bulge']:
            #     if getattr(self.components[model_name], 'noord_flat', False):
            #         self.components[model_name]._update_noord_flatteners()


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

        # Check to see if parameter was fixed or free previously:
        prevstate = self.fixed[model_name][param_name]

        self.components[model_name].fixed[param_name] = fix
        self.fixed[model_name][param_name] = fix

        if prevstate != fix:
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
                    self.set_parameter_value(cmp, pp, theta[ind],skip_updated_tied=True)

        # Now update all of the tied parameters if there are any
        self._update_tied_parameters()

    # Method to update tied parameters:
    def _update_tied_parameters(self):
        """
        Update all tied parameters of the model

        Notes
        -----
        Possibly this should just be invoked at the beginning of :meth:`ModelSet.simulate_cube`
        to ensure the correct tied parameters are used if not set using :meth:`ModelSet.update_parameters`.
        """
        if self.nparams_tied > 0:
            for cmp in self.tied:
                for pp in self.tied[cmp]:
                    if self.tied[cmp][pp]:
                        new_value = self.tied[cmp][pp](self)
                        self.set_parameter_value(cmp, pp, new_value,skip_updated_tied=True)


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
                if self.fixed[cmp][pm] | bool(self.tied[cmp][pm]):
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


    def get_prior_transform(self, u):

        v = np.zeros(len(u))
        ind = -1

        pfree_dict = self.get_free_parameter_keys()
        comps_names = pfree_dict.keys()
        for compn in comps_names:
            comp = self.components.__getitem__(compn)
            params_names = pfree_dict[compn].keys()
            for paramn in params_names:
                if pfree_dict[compn][paramn] >= 0:
                    # Free parameter: get unit prior transform
                    ind += 1
                    v[ind] = comp.__getattribute__(paramn).prior.prior_unit_transform(comp.__getattribute__(paramn), u[ind], modelset=self)
                                
        return v

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
        # vc, vdm = self.circular_velocity(r, compute_dm=True)
        # dm_frac = vdm**2/vc**2

        vcsq, vdmsq = self.vcirc_sq(r, compute_dm=True)
        dm_frac = vdmsq/vcsq

        return dm_frac


    def get_dm_frac_r_ap(self):
        """
        Calculate the dark matter fraction within an aperture radius

        Uses the method self._model_aperture_r to get the defined apeture radius.
        By default this function seturns the disk effective radius
        (i.e., self.components['disk+bulge'].__getattribute__['r_eff_disk'].value )

        Returns
        -------
        dm_frac : float
            Dark matter fraction within the specified effective radius
        """
        # r_ap needs to be in kpc
        r_ap = self._model_aperture_r()
        dm_frac = self.get_dm_aper(r_ap)

        return dm_frac


    def get_mvirial(self):
        """
        Return the virial mass of the dark matter halo component

        Returns
        -------
        mvir : float
            Virial mass of the dark matter halo in log(Msun)
        """
        comp = self.components.__getitem__(self.model_key_halo[0])
        try:
            mvir = comp.mvirial.value
        except:
            mvir = comp.mvirial

        return mvir

    def get_halo_alpha(self):
        """
        Return the alpha parameter value for a `TwoPowerHalo`

        Returns
        -------
        alpha : float or None
            Value of the alpha parameter. Returns None if the correct component
            does not exist.
        """
        comp = self.components.__getitem__(self.model_key_halo[0])
        try:
            return comp.alpha.value
        except:
            return None

    def get_halo_rb(self):
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
        comp = self.components.__getitem__(self.model_key_halo[0])
        try:
            return comp.rB.value
        except:
            return None

    def get_dlnrhogas_dlnr(self, r):
        """
        Calculate the composite derivative dln(rho,gas) / dlnr

        ** Assumes gas follows same distribution as total baryons.
           Based on slope, so do not need to rescale for fgas/Mgas under this assumption.**

        Parameters
        ----------
        r : float or array
            Radius or radii in kpc

        Returns
        -------
        dlnrhogas_dlnr : float or array

        """
        # First check to make sure there is at least one mass component in the model set.
        if len(self.mass_components) == 0:
            raise AttributeError("There are no mass components so a dlnrho/dlnr "
                                 "can't be calculated.")
        else:
            rhogastot = r*0.
            rho_dlnrhogas_dlnr_sum = r*0.

            for cmp in self.mass_components:

                if self.mass_components[cmp]:
                    mcomp = self.components[cmp]

                    # if (mcomp._subtype == 'baryonic') & (not isinstance(mcomp, BlackHole)):
                    if (mcomp._subtype == 'baryonic'):
                        if ('gas' in mcomp.baryon_type.lower().strip()):
                            cmpnt_rhogas = mcomp.rhogas(r)
                            cmpnt_dlnrhogas_dlnr = mcomp.dlnrhogas_dlnr(r)

                            whfin = np.where(np.isfinite(cmpnt_dlnrhogas_dlnr))[0]
                            try:
                                if len(whfin) < len(r):
                                    raise ValueError
                            except:
                                pass

                            rhogastot += cmpnt_rhogas
                            rho_dlnrhogas_dlnr_sum += cmpnt_rhogas * cmpnt_dlnrhogas_dlnr

        dlnrhogas_dlnr = (1./rhogastot) * rho_dlnrhogas_dlnr_sum

        return dlnrhogas_dlnr


    def get_encl_mass_r_ap(self):
        """
        Calculate the total enclosed mass within an aperture radius

        Uses the method self._model_aperture_r to get the defined apeture radius.
        By default this function seturns the disk effective radius
        (i.e., self.components['disk+bulge'].__getattribute__['r_eff_disk'].value )

        Returns
        -------
        menc : float
            Total enclosed mass within the specified effective radius

        Notes
        -----
        (OLD: This method uses the total circular velocity to determine the enclosed mass
        based on v^2 = GM/r.)
        """
        logger.warning("Using spherical approximation to get menc from vcirc!")

        # r_ap needs to be in kpc
        r_ap = self._model_aperture_r()

        # vc = self.circular_velocity(r_ap)
        # menc = menc_from_vcirc(vc, r_ap)

        menc = self.enclosed_mass(r_ap)

        return menc



    def enclosed_mass(self, r, compute_baryon=False, compute_dm=False, step1d=0.2):
        """
        Calculate the total enclosed mass

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate the enclosed mass in kpc

        compute_baryon : bool
            If True, also return the enclosed mass of the baryons

        compute_dm : bool
            If True, also return the enclosed mass of the halo

        step1d : float, optional
            Step size in kpc to use during adiabatic contraction calculation

        Returns
        -------
        enc_mass : float or array
            Total enclosed mass in Msun

        enc_bary : float or array, only if `compute_baryon` = True
            Enclosed mass of the baryons, in Msun

        enc_dm : float or array, only if `compute_dm` = True
            Enclosed mass of the halo, in Msun
        """

        # logger.warning("Using spherical approximation to get menc from vcirc!")

        # First check to make sure there is at least one mass component in the model set.
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
                vcirc, vhalo_adi = self.circular_velocity(r, compute_dm=True, step1d=step1d)
                enc_dm_adi = menc_from_vcirc(vhalo_adi, r)
                enc_mass = enc_mass - enc_dm + enc_dm_adi
                enc_dm = enc_dm_adi

        #return enc_mass, enc_bary, enc_dm
        if (compute_baryon and compute_dm):
            return enc_mass, enc_bary, enc_dm
        elif (compute_dm and (not compute_baryon)):
            return enc_mass, enc_dm
        elif (compute_baryon and (not compute_dm)):
            return enc_mass, enc_bary
        else:
            return enc_mass


    def circular_velocity(self, r, compute_baryon=False, compute_dm=False, step1d=0.2):
        """
        Calculate the total circular velocity as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate the circular velocity in kpc

        compute_baryon : bool
            If True, also return the circular velocity due to the baryons

        compute_dm : bool
            If True, also return the circular velocity due to the halo

        step1d : float, optional
            Step size in kpc to use during adiabatic contraction calculation

        Returns
        -------
        vel : float or array
            Total circular velocity in km/s

        vbaryon : float or array, only if `compute_baryon` = True
            Circular velocity due to the baryons

        vdm : float or array, only if `compute_dm` = True
            Circular velocity due to the halo
        """

        vels_sq = self.vcirc_sq(r, compute_baryon=compute_baryon,
                                compute_dm=compute_dm, step1d=step1d)

        if (compute_baryon and compute_dm):
            vel = np.sqrt(vels_sq[0])
            vbaryon = np.sqrt(vels_sq[1])
            vdm = np.sqrt(vels_sq[2])
            return vel, vbaryon, vdm
        elif (compute_dm and (not compute_baryon)):
            vel = np.sqrt(vels_sq[0])
            vdm = np.sqrt(vels_sq[1])
            return vel, vdm
        elif (compute_baryon and (not compute_dm)):
            vel = np.sqrt(vels_sq[0])
            vbaryon = np.sqrt(vels_sq[1])
            return vel, vbaryon
        else:
            vel = np.sqrt(vels_sq)
            return vel


    def vcirc_sq(self, r, compute_baryon=False, compute_dm=False, step1d=0.2):
        """
        Calculate the square of the total circular velocity as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate the circular velocity in kpc

        compute_baryon : bool
            If True, also return the circular velocity due to the baryons

        compute_dm : bool
            If True, also return the circular velocity due to the halo

        step1d : float, optional
            Step size in kpc to use during adiabatic contraction calculation

        Returns
        -------
        vel : float or array
            Total circular velocity in km/s

        vbaryon : float or array, only if `compute_baryon` = True
            Circular velocity due to the baryons

        vdm : float or array, only if `compute_dm` = True
            Circular velocity due to the halo
        """

        # First check to make sure there is at least one mass component in the
        # model set.
        if len(self.mass_components) == 0:
            raise AttributeError("There are no mass components so a velocity "
                                 "can't be calculated.")
        else:
            vdm_sq = r*0.
            vbaryon_sq = r*0.

            for cmp in self.mass_components:

                if self.mass_components[cmp]:
                    mcomp = self.components[cmp]

                    cmpnt_v_sq = mcomp.vcirc_sq(r)

                    if (mcomp._subtype == 'dark_matter') | (mcomp._subtype == 'combined'):
                        vdm_sq = vdm_sq + cmpnt_v_sq

                    elif mcomp._subtype == 'baryonic':
                        vbaryon_sq = vbaryon_sq + cmpnt_v_sq

                    else:
                        raise TypeError("{} mass model subtype not recognized"
                                        " for {} component. Only 'dark_matter'"
                                        " or 'baryonic' accepted.".format(
                                        mcomp._subtype, cmp))

            vels_sq = self.kinematic_options.apply_adiabatic_contract(self, r, vbaryon_sq, vdm_sq,
                                                                   compute_dm=compute_dm,
                                                                   return_vsq=True,
                                                                   step1d=step1d)

            if compute_dm:
                vel_sq = vels_sq[0]
                vdm_sq = vels_sq[1]
            else:
                vel_sq = vels_sq

            if (compute_baryon and compute_dm):
                return vel_sq, vbaryon_sq, vdm_sq
            elif (compute_dm and (not compute_baryon)):
                return vel_sq, vdm_sq
            elif (compute_baryon and (not compute_dm)):
                return vel_sq, vbaryon_sq
            else:
                return vel_sq

    def velocity_profile(self, r, tracer=None, compute_dm=False):
        """
        Calculate the rotational velocity as a function of radius

        Parameters
        ----------
        r : float or array
            Radius or radii at which to calculate the velocity in kpc

        tracer : string
            Name of the dynamical tracer (used to determine which is the appropriate dispersion profile).

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
        if tracer is None:
            raise ValueError("Must specify 'tracer' to calculate velocity profile!")

        vels_sq = self.vcirc_sq(r, compute_dm=compute_dm)
        if compute_dm:
            vcirc_sq = vels_sq[0]
            vdm_sq = vels_sq[1]

            vdm = np.sqrt(vdm_sq)
        else:
            vcirc_sq = vels_sq

        vel_sq = self.kinematic_options.apply_pressure_support(r, self, vcirc_sq, tracer=tracer)
        vel = np.sqrt(vel_sq)

        if compute_dm:
            return vel, vdm
        else:
            return vel



    def get_vmax(self, r=None, tracer=None):
        """
        Calculate the peak velocity of the rotation curve

        Parameters
        ----------
        r : array, optional
            Radii to sample to find the peak. If None, then a linearly
            spaced array from 0 to 25 kpc with 251 points will be used

        tracer : string
            Name of the dynamical tracer (used to determine which is the appropriate dispersion profile).

        Returns
        -------
        vmax : float
            Peak velocity of the rotation curve in km/s

        Notes
        -----
        This simply finds the maximum of the rotation curve which is calculated at discrete
        radii, `r`.

        """
        if tracer is None:
            raise ValueError("Must specify 'tracer' to calculate velocity profile!")

        if r is None:
            r = np.linspace(0., 25., num=251, endpoint=True)

        vel = self.velocity_profile(r, compute_dm=False, tracer=tracer)

        vmax = vel.max()
        return vmax

    def write_vrot_vcirc_file(self, r=None, filename=None, tracer=None, overwrite=False):
        """
        Output the rotational and circular velocities to a file

        Parameters
        ----------
        r : array, optional
            Radii to sample to find the peak. If None, then a linearly
            spaced array from 0 to 25 kpc with 251 points will be used

        filename : str, optional
            Name of file to output velocities to. Default is 'vout.txt'

        tracer : string
            Name of the dynamical tracer (used to determine which is the appropriate dispersion profile).

        """
        if tracer is None:
            raise ValueError("Must specify 'tracer' to calculate velocity profile!")

        # Check for existing file:
        if (not overwrite) and (filename is not None):
            if os.path.isfile(filename):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
                return None

        # # Quick test for if vcirc defined:
        # coltry = ['velocity_profile', 'circular_velocity']
        # coltrynames = ['vrot', 'vcirc']
        # coltryunits = ['[km/s]', '[km/s]']
        # cols = []
        # colnames = []
        # colunits = []
        # for c, cn, cu in zip(coltry, coltrynames, coltryunits):
        #     try:
        #         fnc = getattr(self, c)
        #         tmp = fnc(np.array([2.]))
        #         cols.append(c)
        #         colnames.append(cn)
        #         colunits.append(cu)
        #     except:
        #         pass

        cols = ['velocity_profile', 'circular_velocity']
        colnames = ['vrot', 'vcirc']
        colunits = ['[km/s]', '[km/s]']

        if len(cols) >= 1:
            self.write_profile_file(r=r, filename=filename,
                cols=cols, prettycolnames=colnames, colunits=colunits,
                tracer=tracer, overwrite=overwrite)


    def write_profile_file(self, r=None, filename=None,
            cols=None, prettycolnames=None, colunits=None,
            tracer=None, overwrite=False):
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

        tracer : string
            Name of the dynamical tracer (used to determine which is the appropriate dispersion profile).
        """
        if tracer is None:
            raise ValueError("Must specify 'tracer' to calculate velocity profile!")

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
        for j in range(len(cols)):
            if cols[j] != 'velocity_profile':
                try:
                    fnc = getattr(self, cols[j])
                    arr = fnc(r)
                    arr[~np.isfinite(arr)] = 0.
                except:
                    arr = np.ones(len(r))*-99.
            else:
                try:
                    fnc = getattr(self, cols[j])
                    arr = fnc(r, tracer=tracer)
                    arr[~np.isfinite(arr)] = 0.
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
            for i in range(len(r)):
                datstr = '    '.join(["{0:0.3f}".format(p) for p in profiles[i,:]])
                f.write(datstr+'\n')


    def simulate_cube(self, obs=None, dscale=None):
        r"""
        Simulate a line emission cube of this model set

        Parameters
        ----------
        obs : Observation class
            Instance holding the observation information

        dscale : float
            Conversion from sky to physical coordinates in arcsec/kpc

        Returns
        -------
        cube_final : 3D array
            Line emission cube that incorporates all of the kinematics due to the components
            of the current `ModelSet`

        spec : 1D array
            Values of the spectral channels as determined by `spec_type`, `spec_start`,
            `spec_step`, `nspec`, and `spec_unit`

        """

        # from . import cutils

        if obs is None:
            raise ValueError("Must pass 'obs' instance!")

        if obs.mod_options.transform_method.lower().strip() not in ['direct', 'rotate']:
            raise ValueError("Transform method {} unknown! "
                    "Must be 'direct' or 'rotate'!".format(transform_method))

        # ----------------------------------------------
        # Get key settings / variables out of obs:
        nx_sky = obs.instrument.fov[0]
        ny_sky = obs.instrument.fov[1]
        pixscale = obs.instrument.pixscale.to(u.arcsec).value
        spec_type = obs.instrument.spec_type
        nspec = obs.instrument.nspec
        spec_step = obs.instrument.spec_step.value
        spec_start = obs.instrument.spec_start.to(obs.instrument.spec_step.unit).value
        spec_unit = obs.instrument.spec_step.unit

        oversample = obs.mod_options.oversample
        oversize = obs.mod_options.oversize
        xcenter = obs.mod_options.xcenter
        ycenter = obs.mod_options.ycenter
        transform_method = obs.mod_options.transform_method
        zcalc_truncate = obs.mod_options.zcalc_truncate
        n_wholepix_z_min = obs.mod_options.n_wholepix_z_min
        # ----------------------------------------------


        # Get the galaxy geometry corresponding to the observation:
        if len(self.geometries) == 0:
            if sum(self.mass_components.values()) > 0:
                raise AttributeError('No geometry defined in your ModelSet!')
            else:
                geom = None
        else:
            geom = self.geometries[obs.name]


        # Start with a 3D array in the sky coordinate system
        # x and y sizes are user provided so we just need
        # the z size where z is in the direction of the L.O.S.
        # We'll just use the maximum of the given x and y

        nx_sky_samp = nx_sky*oversample*oversize
        ny_sky_samp = ny_sky*oversample*oversize
        pixscale_samp = pixscale/oversample
        to_kpc = pixscale_samp / dscale

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

            # Create 3D arrays of the sky / galaxy pixel coordinates
            nz_sky_samp, maxr, maxr_y = _calculate_max_skyframe_extents(geom,
                    nx_sky_samp, ny_sky_samp, transform_method)

            # Apply the geometric transformation to get galactic coordinates
            # Need to account for oversampling in the x and y shift parameters
            geom.xshift = geom.xshift.value * oversample
            geom.yshift = geom.yshift.value * oversample
            sh = (nz_sky_samp, ny_sky_samp, nx_sky_samp)

            # Regularly gridded in galaxy space
            #   -- just use the number values from sky space for simplicity
            if transform_method.lower().strip() == 'direct':
                xgal, ygal, zgal, xsky, ysky, zsky = _get_xyz_sky_gal(geom, sh,
                        xcenter_samp, ycenter_samp, (nz_sky_samp - 1) / 2.)

            # Regularly gridded in sky space, will be rotated later
            elif transform_method.lower().strip() == 'rotate':
                xgal, ygal, zgal, xsky, ysky, zsky = _get_xyz_sky_gal_inverse(geom, sh,
                        xcenter_samp, ycenter_samp, (nz_sky_samp - 1) / 2.)


            # The circular velocity at each position only depends on the radius
            rgal = np.sqrt(xgal ** 2 + ygal ** 2)

            vrot = self.velocity_profile(rgal*to_kpc, tracer=obs.tracer)
            # L.O.S. velocity is then just vrot*sin(i)*cos(theta) where theta
            # is the position angle in the plane of the disk
            # cos(theta) is just xgal/rgal
            v_sys = geom.vel_shift.value  # systemic velocity
            if transform_method.lower().strip() == 'direct':

                # #########################
                # # Get one of the mass components: all have the same vrot unit vector
                # for cmp in self.mass_components:
                #     if self.mass_components[cmp]:
                #         mcomp = self.components[cmp]
                #         break
                #
                # vrot_LOS = geom.project_velocity_along_LOS(mcomp, vrot, xgal, ygal, zgal)
                # vobs_mass = v_sys + vrot_LOS
                # #########################

                #########################
                # Avoid extra calculations to save memory:
                # Use direct calculation for mass components: simple cylindrical LOS projection
                LOS_hat = geom.LOS_direction_emitframe()
                vobs_mass = v_sys + vrot * xgal/rgal * LOS_hat[1]
                # Excise rgal=0 values
                vobs_mass = model_utils.replace_values_by_refarr(vobs_mass, rgal, 0., v_sys)
                #########################

                #######
                # Higher order components: those that have same light distribution
                for cmp_n in self.higher_order_components:
                    comp = self.higher_order_components[cmp_n]
                    cmps_hiord_geoms = list(self.higher_order_geometries.keys())

                    ####
                    if (not comp._separate_light_profile) | \
                        (comp._higher_order_type.lower().strip() == 'perturbation'):
                        if (comp.name not in cmps_hiord_geoms):
                            ## Use general geometry:
                            v_hiord = comp.velocity(xgal*to_kpc, ygal*to_kpc, zgal*to_kpc, self)
                            if comp._spatial_type != 'unresolved':
                                v_hiord_LOS = geom.project_velocity_along_LOS(comp, v_hiord,
                                                                                   xgal, ygal, zgal)
                            else:
                                v_hiord_LOS = v_hiord
                        else:
                            ## Own geometry:
                            hiord_geom = self.higher_order_geometries[comp.name]

                            nz_sky_samp_hi, _, _ = _calculate_max_skyframe_extents(hiord_geom,
                                        nx_sky_samp, ny_sky_samp, transform_method, angle='sin')
                            sh_hi = (nz_sky_samp_hi, ny_sky_samp, nx_sky_samp)

                            # Apply the geometric transformation to get higher order coordinates
                            # Account for oversampling
                            hiord_geom.xshift = hiord_geom.xshift.value * oversample
                            hiord_geom.yshift = hiord_geom.yshift.value * oversample
                            xhiord, yhiord, zhiord, xsky, ysky, zsky = _get_xyz_sky_gal(hiord_geom, sh_hi,
                                            xcenter_samp, ycenter_samp, (nz_sky_samp_hi - 1) / 2.)

                            # Profiles need positions in kpc
                            v_hiord = comp.velocity(xhiord*to_kpc, yhiord*to_kpc, zhiord*to_kpc, self)

                            # LOS projection
                            if comp._spatial_type != 'unresolved':
                                v_hiord_LOS = hiord_geom.project_velocity_along_LOS(comp, v_hiord,
                                                xhiord, yhiord, zhiord)
                            else:
                                v_hiord_LOS = v_hiord

                            # Remove the oversample from the geometry xyshift
                            hiord_geom.xshift = hiord_geom.xshift.value / oversample
                            hiord_geom.yshift = hiord_geom.yshift.value / oversample

                            sh_hi = nz_sky_samp_hi = xhiord = yhiord = zhiord = None

                        #   No systemic velocity here bc this is relative to
                        #    the center of the galaxy at rest already
                        vobs_mass += v_hiord_LOS
                #######

            elif transform_method.lower().strip() == 'rotate':
                ####
                logger.warning("Transform method 'rotate' has not been fully tested after changes!")
                ####
                vcirc_mass = vrot
                vcirc_mass[rgal == 0] = 0.


            # Calculate "flux" for each position
            flux_mass = np.zeros(rgal.shape)

            # self.light_components SHOULD NOT include
            #    higher-order kin comps with own light profiles.

            tracer_lcomps = model_utils.get_light_components_by_tracer(self, obs.tracer)
            for cmp in tracer_lcomps:
                if (self.light_components[cmp]):
                    lcomp = self.components[cmp]
                    zscale = self.zprofile(zgal*to_kpc)
                    # Differentiate between axisymmetric and non-axisymmetric light components:
                    if lcomp._axisymmetric:
                        # Axisymmetric cases:
                        flux_mass += lcomp.light_profile(rgal*to_kpc) * zscale
                    else:
                        # Non-axisymmetric cases:
                        ## ASSUME IT'S ALL IN THE MIDPLANE, so also apply zscale
                        flux_mass +=  lcomp.light_profile(xgal*to_kpc, ygal*to_kpc, zgal*to_kpc) * zscale


            # Apply extinction if a component exists
            if self.extinction is not None:
                flux_mass *= self.extinction(xsky, ysky, zsky)

            # Apply dimming if a component exists
            if self.dimming is not None:
                flux_mass *= self.dimming(xsky, ysky, zsky)

            if transform_method.lower().strip() == 'direct':
                sigmar = self.dispersions[obs.tracer](rgal*to_kpc)

                # The final spectrum will be a flux weighted sum of Gaussians at each
                # velocity along the line of sight.
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if zcalc_truncate:
                    # Truncate in the z direction by flagging what pixels to include in propogation
                    ai = _make_cube_ai(self, xgal, ygal, zgal, n_wholepix_z_min=n_wholepix_z_min,
                        pixscale=pixscale_samp, oversample=oversample,
                        dscale=dscale, maxr=maxr/2., maxr_y=maxr_y/2.)
                    cube_final += cutils.populate_cube_ais(flux_mass, vobs_mass, sigmar, vx, ai)
                else:
                    # Do complete cube propogation calculation
                    cube_final += cutils.populate_cube(flux_mass, vobs_mass, sigmar, vx)
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            elif transform_method.lower().strip() == 'rotate':
                ###################################
                xgal_final, ygal_final, zgal_final, xsky_final, ysky_final, zsky_final = \
                    _get_xyz_sky_gal_inverse(geom, sh, xcenter_samp, ycenter_samp,
                                             (nz_sky_samp - 1) / 2.)

                #rgal_final = np.sqrt(xgal_final ** 2 + ygal_final ** 2) * pixscale_samp / dscale
                rgal_final = np.sqrt(xgal_final ** 2 + ygal_final ** 2)
                #rgal_final_kpc = rgal_final * pixscale_samp / dscale

                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Simpler to just directly sample sigmar -- not as prone to sampling problems / often constant.
                sigmar_transf = self.dispersion_profile(rgal_final*to_kpc)


                if zcalc_truncate:
                    # cos_inc = np.cos(geom.inc*np.pi/180.)
                    # maxr_y_final = np.max(np.array([maxr*1.5, np.min(
                    #     np.hstack([maxr*1.5/ cos_inc, maxr * 5.]))]))

                    # Use the cos term, and the normal 'direct' maxr_y calculation
                    _, _, maxr_y_final = _calculate_max_skyframe_extents(geom,
                            nx_sky_samp, ny_sky_samp, 'direct', angle='cos')

                    # ---------------------
                    # GET TRIMMING FOR TRANSFORM:
                    thick = self.zprofile.z_scalelength.value
                    if not np.isfinite(thick):
                        thick = 0.
                    # Sample += 2 * scale length thickness
                    # Modify: make sure there are at least 3 *whole* pixels sampled:
                    zsize = np.max([  3.*oversample, int(np.floor( 4.*thick/pixscale_samp*dscale + 0.5 )) ])
                    if ( (zsize%2) < 0.5 ): zsize += 1
                    zarr = np.arange(nz_sky_samp) - (nz_sky_samp - 1) / 2.
                    origpos_z = zarr - np.mean(zarr) + zsize/2.
                    validz = np.where((origpos_z >= -0.5) & (origpos_z < zsize-0.5) )[0]
                    # ---------------------

                    # Rotate + transform cube from inclined to sky coordinates
                    outsh = flux_mass.shape
                    # Cube: z, y, x -- this is in GALAXY coords, so z trim is just in z coord.
                    flux_mass_transf  = geom.transform_cube_affine(flux_mass[validz,:,:], output_shape=outsh)
                    vcirc_mass_transf = geom.transform_cube_affine(vcirc_mass[validz,:,:], output_shape=outsh)

                    # -----------------------
                    # Perform LOS projection
                    # #########################
                    # vobs_mass_transf_LOS = geom.project_velocity_along_LOS(mcomp, vcirc_mass_transf,
                    #                         xgal_final, ygal_final, zgal_final)
                    # vobs_mass_transf = v_sys + vobs_mass_transf_LOS
                    # #########################

                    #########################
                    # Avoid extra calculations to save memory:
                    # Use direct calculation for mass components: simple cylindrical LOS projection
                    LOS_hat = geom.LOS_direction_emitframe()
                    vobs_mass_transf = v_sys + vcirc_mass_transf * xgal_final/rgal_final * LOS_hat[1]
                    # Excise rgal=0 values
                    vobs_mass_transf = model_utils.replace_values_by_refarr(vobs_mass_transf, rgal_final, 0., v_sys)
                    #########################
                    # -----------------------

                    #######
                    # Higher order components: those that have same light distribution
                    for cmp_n in self.higher_order_components:
                        comp = self.higher_order_components[cmp_n]
                        cmps_hiord_geoms = list(self.higher_order_geometries.keys())

                        ####
                        if (not comp._separate_light_profile) | \
                            (comp._higher_order_type.lower().strip() == 'perturbation'):
                            if (comp.name not in cmps_hiord_geoms):
                                ## Use general geometry:
                                v_hiord = comp.velocity(xgal_final*to_kpc, ygal_final*to_kpc,
                                                    zgal_final*to_kpc, self)
                                if comp._spatial_type != 'unresolved':
                                    v_hiord_LOS = geom.project_velocity_along_LOS(comp, v_hiord,
                                                                xgal_final, ygal_final, zgal_final)
                                else:
                                    v_hiord_LOS = v_hiord
                            else:
                                ## Own geometry, not perturbation:
                                hiord_geom = self.higher_order_geometries[comp.name]

                                nz_sky_samp_hi, _, _ = _calculate_max_skyframe_extents(hiord_geom,
                                            nx_sky_samp, ny_sky_samp, 'direct', angle='sin')
                                sh_hi = (nz_sky_samp_hi, ny_sky_samp, nx_sky_samp)

                                # Apply the geometric transformation to get higher order coordinates
                                # Account for oversampling
                                hiord_geom.xshift = hiord_geom.xshift.value * oversample
                                hiord_geom.yshift = hiord_geom.yshift.value * oversample
                                xhiord, yhiord, zhiord, xsky, ysky, zsky = _get_xyz_sky_gal(hiord_geom, sh_hi,
                                                xcenter_samp, ycenter_samp, (nz_sky_samp_hi - 1) / 2.)

                                # Profiles need positions in kpc
                                v_hiord = comp.velocity(xhiord*to_kpc, yhiord*to_kpc, zhiord*to_kpc)

                                # LOS projection
                                if comp._spatial_type != 'unresolved':
                                    v_hiord_LOS = hiord_geom.project_velocity_along_LOS(comp, v_hiord,
                                                    xhiord, yhiord, zhiord)
                                else:
                                    v_hiord_LOS = v_hiord

                                # Remove the oversample from the geometry xyshift
                                hiord_geom.xshift = hiord_geom.xshift.value / oversample
                                hiord_geom.yshift = hiord_geom.yshift.value / oversample

                                sh_hi = nz_sky_samp_hi = xhiord = yhiord = zhiord = None

                            #   No systemic velocity here bc this is relative to
                            #    the center of the galaxy at rest already
                            vobs_mass_transf += v_hiord_LOS
                    #######

                    #######
                    # Truncate in the z direction by flagging what pixels to include in propogation
                    ai_sky = _make_cube_ai(self, xgal_final, ygal_final, zgal_final,
                            n_wholepix_z_min=n_wholepix_z_min,
                            pixscale=pixscale_samp, oversample=oversample,
                            dscale=dscale, maxr=maxr/2., maxr_y=maxr_y_final/2.)
                    cube_final += cutils.populate_cube_ais(flux_mass_transf, vobs_mass_transf,
                                sigmar_transf, vx, ai_sky)

                else:
                    # Rotate + transform cube from inclined to sky coordinates
                    flux_mass_transf =  geom.transform_cube_affine(flux_mass)
                    vcirc_mass_transf = geom.transform_cube_affine(vcirc_mass)

                    # -----------------------
                    # Perform LOS projection
                    # #########################
                    # vobs_mass_transf_LOS = geom.project_velocity_along_LOS(mcomp, vcirc_mass_transf,
                    #                         xgal_final, ygal_final, zgal_final)
                    # vobs_mass_transf = v_sys + vobs_mass_transf_LOS
                    # #########################

                    #########################
                    # Avoid extra calculations to save memory:
                    # Use direct calculation for mass components: simple cylindrical LOS projection
                    LOS_hat = geom.LOS_direction_emitframe()
                    vobs_mass_transf = v_sys + vcirc_mass_transf * xgal_final/rgal_final * LOS_hat[1]
                    # Excise rgal=0 values
                    vobs_mass_transf = model_utils.replace_values_by_refarr(vobs_mass_transf, rgal_final, 0., v_sys)
                    #########################
                    # -----------------------

                    #######
                    # Higher order components: those that have same light distribution
                    for cmp_n in self.higher_order_components:
                        comp = self.higher_order_components[cmp_n]
                        cmps_hiord_geoms = list(self.higher_order_geometries.keys())

                        ####
                        if (not comp._separate_light_profile) | \
                            (comp._higher_order_type.lower().strip() == 'perturbation'):
                            if (comp.name not in cmps_hiord_geoms):
                                ## Use general geometry:
                                v_hiord = comp.velocity(xgal_final*to_kpc, ygal_final*to_kpc,
                                                    zgal_final*to_kpc, self)
                                if comp._spatial_type != 'unresolved':
                                    v_hiord_LOS = geom.project_velocity_along_LOS(comp, v_hiord,
                                                                xgal_final, ygal_final, zgal_final)
                                else:
                                    v_hiord_LOS = v_hiord
                            else:
                                ## Own geometry, not perturbation:
                                hiord_geom = self.higher_order_geometries[comp.name]

                                nz_sky_samp_hi, _, _ = _calculate_max_skyframe_extents(hiord_geom,
                                            nx_sky_samp, ny_sky_samp, 'direct', angle='sin')
                                sh_hi = (nz_sky_samp_hi, ny_sky_samp, nx_sky_samp)

                                # Apply the geometric transformation to get higher order coordinates
                                # Account for oversampling
                                hiord_geom.xshift = hiord_geom.xshift.value * oversample
                                hiord_geom.yshift = hiord_geom.yshift.value * oversample
                                xhiord, yhiord, zhiord, xsky, ysky, zsky = _get_xyz_sky_gal(hiord_geom, sh_hi,
                                                xcenter_samp, ycenter_samp, (nz_sky_samp_hi - 1) / 2.)

                                # Profiles need positions in kpc
                                v_hiord = comp.velocity(xhiord*to_kpc, yhiord*to_kpc, zhiord*to_kpc)

                                # LOS projection
                                if comp._spatial_type != 'unresolved':
                                    v_hiord_LOS = hiord_geom.project_velocity_along_LOS(comp, v_hiord,
                                                    xhiord, yhiord, zhiord)
                                else:
                                    v_hiord_LOS = v_hiord

                                # Remove the oversample from the geometry xyshift
                                hiord_geom.xshift = hiord_geom.xshift.value / oversample
                                hiord_geom.yshift = hiord_geom.yshift.value / oversample

                                sh_hi = nz_sky_samp_hi = xhiord = yhiord = zhiord = None

                            #   No systemic velocity here bc this is relative to
                            #    the center of the galaxy at rest already
                            vobs_mass_transf += v_hiord_LOS
                    #######

                    # Do complete cube propogation calculation
                    cube_final += cutils.populate_cube(flux_mass_transf, vobs_mass_transf, sigmar_transf, vx)
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # Remove the oversample from the geometry xyshift
            geom.xshift = geom.xshift.value / oversample
            geom.yshift = geom.yshift.value / oversample


        #######
        # Higher order components: those that have OWN light distribution, aren't perturbations
        for cmp_n in self.higher_order_components:
            comp = self.higher_order_components[cmp_n]
            cmps_hiord_geoms = list(self.higher_order_geometries.keys())
            cmps_hiord_disps = list(self.higher_order_dispersions.keys())

            _do_comp = False
            if (comp._separate_light_profile) & (comp._higher_order_type.lower().strip() != 'perturbation'):
                if (comp.name in cmps_hiord_geoms):
                    # Own geometry + light distribution
                    _do_comp = True
                    hiord_geom = self.higher_order_geometries[comp.name]
                elif (comp.name not in cmps_hiord_geoms):
                    # Own light distribution, uses galaxy geometry
                    _do_comp = True
                    hiord_geom = geom
                    logger.warning("The case of higher order component using galaxy geometry "
                                   "but own light profile has not been tested")

            ######
            # Catch failure condition: _higher_order_type = 'perturbation' must NOT be included here,
            #       but rather above as a direct perturbation to the mass compoonent velocities.
            if _do_comp & (comp._higher_order_type.lower().strip() == 'perturbation'):
                msg = "Component with comp._higher_order_type = 'perturbation' "
                msg += "must NOT have separate light profile!\n"
                msg += "If comp has own geometry, calculations still must happen "
                msg += "during the mass model cube creation (as this is a perturbation)."
                raise ValueError(msg)

            if _do_comp:
                #######
                # Just create extra cube using the DIRECT calculation method

                nz_sky_samp, maxr, maxr_y = _calculate_max_skyframe_extents(hiord_geom,
                            nx_sky_samp, ny_sky_samp, 'direct', angle='sin')

                sh = (nz_sky_samp, ny_sky_samp, nx_sky_samp)

                # Apply the geometric transformation to get higher order coordinates
                # Account for oversampling
                hiord_geom.xshift = hiord_geom.xshift.value * oversample
                hiord_geom.yshift = hiord_geom.yshift.value * oversample
                xhiord, yhiord, zhiord, xsky, ysky, zsky = _get_xyz_sky_gal(hiord_geom, sh,
                                xcenter_samp, ycenter_samp, (nz_sky_samp - 1) / 2.)

                # Profiles need positions in kpc
                v_hiord = comp.velocity(xhiord*to_kpc, yhiord*to_kpc, zhiord*to_kpc)
                f_hiord = comp.light_profile(xhiord*to_kpc, yhiord*to_kpc, zhiord*to_kpc)

                # Apply extinction if it exists
                if self.extinction is not None:
                    f_hiord *= self.extinction(xsky, ysky, zsky)

                # Apply dimming if a component exists
                if self.dimming is not None:
                    f_hiord *= self.dimming(xsky, ysky, zsky)

                # LOS projection
                if comp._spatial_type != 'unresolved':
                    v_hiord_LOS = hiord_geom.project_velocity_along_LOS(comp, v_hiord, xhiord, yhiord, zhiord)
                else:
                    v_hiord_LOS = v_hiord

                v_hiord_LOS += hiord_geom.vel_shift.value  # galaxy systemic velocity

                if (comp.name in cmps_hiord_disps):
                    sigma_hiord = self.higher_order_dispersions[comp.name](np.sqrt(xhiord**2 + yhiord**2 + zhiord**2)) # r_hiord
                else:
                    # The higher-order term MUST have its own defined dispersion profile:
                    sigma_hiord = comp.dispersion_profile(xhiord*to_kpc, yhiord*to_kpc, zhiord*to_kpc)

                cube_final += cutils.populate_cube(f_hiord, v_hiord_LOS, sigma_hiord, vx)

                # Remove the oversample from the geometry xyshift
                hiord_geom.xshift = hiord_geom.xshift.value / oversample
                hiord_geom.yshift = hiord_geom.yshift.value / oversample


        return cube_final, spec
