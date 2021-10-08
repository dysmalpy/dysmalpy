# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Handling of DysmalPy ModelSets (and Models) to use build the galaxy model

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import os
import logging
import six
from collections import OrderedDict

# Local imports
from .base import _DysmalModel, menc_from_vcirc
from .kinematic_options import KinematicOptions

# Third party imports
import numpy as np
import scipy.ndimage as scp_ndi
import astropy.constants as apy_con
import astropy.units as u
import pyximport; pyximport.install()
from . import cutils


__all__ = ['ModelSet',
           'calc_1dprofile', 'calc_1dprofile_circap_pv']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')

np.warnings.filterwarnings('ignore')


def _make_cube_ai(model, xgal, ygal, zgal, n_wholepix_z_min = 3,
            rstep=None, oversample=None, dscale=None,
            maxr=None, maxr_y=None):

    oversize = 1.5  # Padding factor for x trimming

    thick = model.zprofile.z_scalelength.value
    if not np.isfinite(thick):
        thick = 0.

    # # maxr, maxr_y are already in pixel units
    xsize = np.int(np.floor(2.*(maxr * oversize) +0.5))
    ysize = np.int(np.floor( 2.*maxr_y + 0.5))

    # Sample += 2 * scale length thickness
    # Modify: make sure there are at least 3 *whole* pixels sampled:
    zsize = np.max([ n_wholepix_z_min*oversample, np.int(np.floor(4.*thick/rstep*dscale + 0.5 )) ])

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

def area_segm(rr, dd):

    return (rr**2 * np.arccos(dd/rr) -
            dd * np.sqrt(2. * rr * (rr-dd) - (rr-dd)**2))



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

    #nz_sky_samp = np.int(np.max([nx_sky_samp, ny_sky_samp, maxr_y]))
    if angle.lower().strip() == 'cos':
        nz_sky_samp = np.int(np.max([nx_sky_samp, ny_sky_samp]))
    elif angle.lower().strip() == 'sin':
        nz_sky_samp = np.int(np.max([nx_sky_samp, ny_sky_samp, maxr_y]))
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
        self.geometry = None
        self.dispersion_profile = None
        self.zprofile = None

        # Keep higher-order kinematic components, and their geom/disp/fluxes in OrderedDicts:
        #       (biconical) outflow / uniform flow / etc
        self.higher_order_components = OrderedDict()
        self.higher_order_geometries = OrderedDict()
        self.higher_order_dispersions = OrderedDict()

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

    def __setstate__(self, state):
        # Compatibility hack, to handle the change to generalized
        #    higher order components in ModelSet.simulate_cube().
        self.__dict__ = state

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

            # Cleanup old names:
            del_keys = ['outflow', 'outflow_geometry', 'outflow_dispersion', 'outflow_flux',
                        'inflow', 'inflow_geometry', 'inflow_dispersion', 'inflow_flux',
                        'flow', 'flow_geometry', 'flow_dispersion', 'flow_flux']
            for dkey in del_keys:
                del self.__dict__[dkey]



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
                    if (self.geometry is not None):
                        logger.warning('Current Geometry model is being '
                                    'overwritten!')
                    self.geometry = model

                else:
                    self.higher_order_geometries[geom_type] = model

                self.mass_components[model.name] = False

            elif model._type == 'dispersion':
                if disp_type == 'galaxy':
                    if self.dispersion_profile is not None:
                        logger.warning('Current Dispersion model is being '
                                       'overwritten!')
                    self.dispersion_profile = model

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

        vc = self.circular_velocity(r)
        menc = menc_from_vcirc(vc, r_eff)

        return menc

    def enclosed_mass(self, r,  compute_baryon=False, compute_dm=False,
                        model_key_re=['disk+bulge', 'r_eff_disk'], step1d=0.2):
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

        model_key_re : list, optional
            Two element list which contains the name of the model component
            and parameter to use for the effective radius. Only necessary
            if adiabatic contraction is used. Default is ['disk+bulge', 'r_eff_disk'].

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
                vcirc, vhalo_adi = self.circular_velocity(r, compute_dm=True,
                                    model_key_re=model_key_re, step1d=step1d)
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



    def circular_velocity(self, r, compute_baryon=False, compute_dm=False,
                            model_key_re=['disk+bulge', 'r_eff_disk'], step1d=0.2):
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
            vdm = r*0.
            vbaryon = r*0.

            for cmp in self.mass_components:

                if self.mass_components[cmp]:
                    mcomp = self.components[cmp]

                    # if isinstance(mcomp, DiskBulge) | isinstance(mcomp, LinearDiskBulge):
                    #     cmpnt_v = mcomp.circular_velocity(r)
                    # else:
                    #     cmpnt_v = mcomp.circular_velocity(r)

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

            if (compute_baryon and compute_dm):
                return vel, vbaryon, vdm
            elif (compute_dm and (not compute_baryon)):
                return vel, vdm
            elif (compute_baryon and (not compute_dm)):
                return vel, vbaryon
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
            for i in six.moves.xrange(len(r)):
                datstr = '    '.join(["{0:0.3f}".format(p) for p in profiles[i,:]])
                f.write(datstr+'\n')


    def simulate_cube(self, nx_sky, ny_sky, dscale, rstep,
                      spec_type, spec_step, spec_start, nspec,
                      spec_unit=u.km/u.s, oversample=1, oversize=1,
                      xcenter=None, ycenter=None,
                      transform_method='direct',
                      zcalc_truncate=True,
                      n_wholepix_z_min=3):
        r"""
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

        transform_method: str
            Method for transforming from galaxy to sky coordinates.
            Options are:
                'direct' (calculate (xyz)sky before evaluating) or
                'rotate' (calculate in (xyz)gal, then rotate when creating final cube).
            Default: 'direct'.

        zcalc_truncate: bool
            Setting the default behavior of filling the model cube. If True,
            then the cube is only filled with flux
            to within +- 2 * scale length thickness above and below
            the galaxy midplane (minimum: n_wholepix_z_min [3] whole pixels; to speed up the calculation).
            If False, then no truncation is applied and the cube is filled over the full range of zgal.
            Default: True

        n_wholepix_z_min: int
            Minimum number of whole pixels to include in the z direction when trunctating.
            Default: 3

        Returns
        -------
        cube_final : 3D array
            Line emission cube that incorporates all of the kinematics due to the components
            of the current `ModelSet`

        spec : 1D array
            Values of the spectral channels as determined by `spec_type`, `spec_start`,
            `spec_step`, `nspec`, and `spec_unit`

        """

        if transform_method.lower().strip() not in ['direct', 'rotate']:
            raise ValueError("Transform method {} unknown! "
                    "Must be 'direct' or 'rotate'!".format(transform_method))


        # Start with a 3D array in the sky coordinate system
        # x and y sizes are user provided so we just need
        # the z size where z is in the direction of the L.O.S.
        # We'll just use the maximum of the given x and y

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
            nz_sky_samp, maxr, maxr_y = _calculate_max_skyframe_extents(self.geometry,
                    nx_sky_samp, ny_sky_samp, transform_method)

            # Apply the geometric transformation to get galactic coordinates
            # Need to account for oversampling in the x and y shift parameters
            self.geometry.xshift = self.geometry.xshift.value * oversample
            self.geometry.yshift = self.geometry.yshift.value * oversample
            sh = (nz_sky_samp, ny_sky_samp, nx_sky_samp)

            # Regularly gridded in galaxy space
            #   -- just use the number values from sky space for simplicity
            if transform_method.lower().strip() == 'direct':
                xgal, ygal, zgal, xsky, ysky, zsky = _get_xyz_sky_gal(self.geometry, sh,
                        xcenter_samp, ycenter_samp, (nz_sky_samp - 1) / 2.)

            # Regularly gridded in sky space, will be rotated later
            elif transform_method.lower().strip() == 'rotate':
                xgal, ygal, zgal, xsky, ysky, zsky = _get_xyz_sky_gal_inverse(self.geometry, sh,
                        xcenter_samp, ycenter_samp, (nz_sky_samp - 1) / 2.)


            # The circular velocity at each position only depends on the radius
            # Convert to kpc
            rgal = np.sqrt(xgal ** 2 + ygal ** 2)
            rgal_kpc = rgal * rstep_samp / dscale
            xgal_kpc = xgal * rstep_samp / dscale
            ygal_kpc = ygal * rstep_samp / dscale
            zgal_kpc = zgal * rstep_samp / dscale

            vrot = self.velocity_profile(rgal_kpc)
            # L.O.S. velocity is then just vrot*sin(i)*cos(theta) where theta
            # is the position angle in the plane of the disk
            # cos(theta) is just xgal/rgal
            v_sys = self.geometry.vel_shift.value  # systemic velocity
            if transform_method.lower().strip() == 'direct':
                # Get one of the mass components: all have the same vrot unit vector
                for cmp in self.mass_components:
                    if self.mass_components[cmp]:
                        mcomp = self.components[cmp]
                        break
                vrot_LOS = self.geometry.project_velocity_along_LOS(mcomp, vrot, xgal, ygal, zgal)
                # Already performed in geom.project_velocity_along_LOS()
                #vrot_LOS[rgal == 0] = 0.
                vobs_mass = v_sys + vrot_LOS

                #######
                # Higher order components: those that follow general geometry
                #   (and have same light distribution)
                for cmp_n in self.higher_order_components:
                    comp = self.higher_order_components[cmp_n]
                    cmps_hiord_geoms = list(self.higher_order_geometries.keys())

                    if (comp.name not in cmps_hiord_geoms) & (not comp._separate_light_profile):
                        # Use general geometry:
                        rgal3D = np.sqrt(xgal ** 2 + ygal ** 2 + zgal **2)

                        v_hiord = comp.velocity(xgal_kpc, ygal_kpc, zgal_kpc)
                        if comp._spatial_type != 'unresolved':
                            v_hiord_LOS = self.geometry.project_velocity_along_LOS(comp, v_hiord,
                                                                               xgal, ygal, zgal)
                        else:
                            v_hiord_LOS = v_hiord

                        ## Must handle r=0 excising internally, because v_hiord is a 3-tuple sometimes
                        # v_hiord_LOS[rgal3D == 0] = v_hiord[rgal3D == 0]

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
            for cmp in self.light_components:
                if self.light_components[cmp]:
                    lcomp = self.components[cmp]
                    zscale = self.zprofile(zgal_kpc)
                    # Differentiate between axisymmetric and non-axisymmetric light components:
                    if lcomp._axisymmetric:
                        # Axisymmetric cases:
                        flux_mass += lcomp.light_profile(rgal_kpc) * zscale
                    else:
                        # Non-axisymmetric cases:

                        ## ASSUME IT'S ALL IN THE MIDPLANE:
                        flux_midplane = lcomp.light_profile(xgal_kpc, ygal_kpc, zgal_kpc)
                        flux_mass +=  flux_midplane * zscale

                        ## Later option: directly 3D calculate ????
                        #flux_mass +=  flux_3D


            # Apply extinction if a component exists
            if self.extinction is not None:
                flux_mass *= self.extinction(xsky, ysky, zsky)

            if transform_method.lower().strip() == 'direct':
                sigmar = self.dispersion_profile(rgal_kpc)

                # The final spectrum will be a flux weighted sum of Gaussians at each
                # velocity along the line of sight.
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if zcalc_truncate:
                    # Truncate in the z direction by flagging what pixels to include in propogation
                    ai = _make_cube_ai(self, xgal, ygal, zgal, n_wholepix_z_min=n_wholepix_z_min,
                        rstep=rstep_samp, oversample=oversample,
                        dscale=dscale, maxr=maxr/2., maxr_y=maxr_y/2.)
                    cube_final += cutils.populate_cube_ais(flux_mass, vobs_mass, sigmar, vx, ai)
                else:
                    # Do complete cube propogation calculation
                    cube_final += cutils.populate_cube(flux_mass, vobs_mass, sigmar, vx)
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            elif transform_method.lower().strip() == 'rotate':
                ###################################

                xgal_final, ygal_final, zgal_final, xsky_final, ysky_final, zsky_final = \
                    _get_xyz_sky_gal_inverse(self.geometry, sh, xcenter_samp, ycenter_samp,
                                             (nz_sky_samp - 1) / 2.)

                #rgal_final = np.sqrt(xgal_final ** 2 + ygal_final ** 2) * rstep_samp / dscale
                rgal_final = np.sqrt(xgal_final ** 2 + ygal_final ** 2)
                rgal_final_kpc = rgal_final * rstep_samp / dscale

                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Simpler to just directly sample sigmar -- not as prone to sampling problems / often constant.
                sigmar_transf = self.dispersion_profile(rgal_final_kpc)

                if zcalc_truncate:
                    # cos_inc = np.cos(self.geometry.inc*np.pi/180.)
                    # maxr_y_final = np.max(np.array([maxr*1.5, np.min(
                    #     np.hstack([maxr*1.5/ cos_inc, maxr * 5.]))]))

                    # Use the cos term, and the normal 'direct' maxr_y calculation
                    _, _, maxr_y_final = _calculate_max_skyframe_extents(self.geometry,
                            nx_sky_samp, ny_sky_samp, 'direct', angle='cos')

                    # ---------------------
                    # GET TRIMMING FOR TRANSFORM:
                    thick = self.zprofile.z_scalelength.value
                    if not np.isfinite(thick):
                        thick = 0.
                    # Sample += 2 * scale length thickness
                    # Modify: make sure there are at least 3 *whole* pixels sampled:
                    zsize = np.max([  3.*oversample, np.int(np.floor( 4.*thick/rstep_samp*dscale + 0.5 )) ])
                    if ( (zsize%2) < 0.5 ): zsize += 1
                    zarr = np.arange(nz_sky_samp) - (nz_sky_samp - 1) / 2.
                    origpos_z = zarr - np.mean(zarr) + zsize/2.
                    validz = np.where((origpos_z >= -0.5) & (origpos_z < zsize-0.5) )[0]
                    # ---------------------

                    # Rotate + transform cube from inclined to sky coordinates
                    outsh = flux_mass.shape
                    # Cube: z, y, x -- this is in GALAXY coords, so z trim is just in z coord.
                    flux_mass_transf  = self.geometry.transform_cube_affine(flux_mass[validz,:,:], output_shape=outsh)
                    vcirc_mass_transf = self.geometry.transform_cube_affine(vcirc_mass[validz,:,:], output_shape=outsh)

                    # -----------------------
                    # Perform LOS projection
                    vobs_mass_transf_LOS = self.geometry.project_velocity_along_LOS(mcomp, vcirc_mass_transf,
                                            xgal_final, ygal_final, zgal_final)
                    # Already performed in geom.project_velocity_along_LOS()
                    #vobs_mass_transf_LOS[rgal_final == 0] = 0.
                    vobs_mass_transf = v_sys + vobs_mass_transf_LOS
                    # -----------------------

                    #######
                    # Higher order components: those that follow general geometry
                    #   (and have same light distribution)
                    for cmp_n in self.higher_order_components:
                        comp = self.higher_order_components[cmp_n]
                        cmps_hiord_geoms = list(self.higher_order_geometries.keys())

                        if (comp.name not in cmps_hiord_geoms) & (not comp._separate_light_profile):
                            # Use general geometry:
                            rgal3D = np.sqrt(xgal_final ** 2 + ygal_final ** 2 + zgal_final **2)
                            xgal_kpc = xgal_final * rstep_samp / dscale
                            ygal_kpc = ygal_final * rstep_samp / dscale
                            zgal_kpc = zgal_final * rstep_samp / dscale

                            # Get velocity of higher-order component
                            v_hiord = comp.velocity(xgal_kpc, ygal_kpc, zgal_kpc)
                            # Project along LOS
                            if comp._spatial_type != 'unresolved':
                                v_hiord_LOS = self.geometry.project_velocity_along_LOS(comp, v_hiord,
                                                                xgal_final, ygal_final, zgal_final)
                            else:
                                v_hiord_LOS = v_hiord

                            ## Must handle r=0 excising internally, because v_hiord is a 3-tuple sometimes
                            # v_hiord_LOS[rgal3D == 0] = v_hiord[rgal3D == 0]

                            #   No systemic velocity here bc this is relative to
                            #    the center of the galaxy at rest already
                            vobs_mass += v_hiord_LOS
                    #######


                    #######
                    # Truncate in the z direction by flagging what pixels to include in propogation
                    ai_sky = _make_cube_ai(self, xgal_final, ygal_final, zgal_final,
                            n_wholepix_z_min=n_wholepix_z_min,
                            rstep=rstep_samp, oversample=oversample,
                            dscale=dscale, maxr=maxr/2., maxr_y=maxr_y_final/2.)
                    cube_final += cutils.populate_cube_ais(flux_mass_transf, vobs_mass_transf,
                                sigmar_transf, vx, ai_sky)

                else:
                    # Rotate + transform cube from inclined to sky coordinates
                    flux_mass_transf =  self.geometry.transform_cube_affine(flux_mass)
                    vcirc_mass_transf = self.geometry.transform_cube_affine(vcirc_mass)

                    # -----------------------
                    # Perform LOS projection
                    vobs_mass_transf_LOS = self.geometry.project_velocity_along_LOS(mcomp, vcirc_mass_transf,
                                            xgal_final, ygal_final, zgal_final)
                    # Already performed in geom.project_velocity_along_LOS()
                    #vobs_mass_transf_LOS[rgal_final == 0] = 0.
                    vobs_mass_transf = v_sys + vobs_mass_transf_LOS
                    # -----------------------

                    #######
                    # Higher order components: those that follow general geometry
                    #   (and have same light distribution)
                    for cmp_n in self.higher_order_components:
                        comp = self.higher_order_components[cmp_n]
                        cmps_hiord_geoms = list(self.higher_order_geometries.keys())

                        if (comp.name not in cmps_hiord_geoms) & (not comp._separate_light_profile):
                            # Use general geometry:
                            rgal3D = np.sqrt(xgal_final ** 2 + ygal_final ** 2 + zgal_final **2)
                            xgal_kpc = xgal_final * rstep_samp / dscale
                            ygal_kpc = ygal_final * rstep_samp / dscale
                            zgal_kpc = zgal_final * rstep_samp / dscale

                            v_hiord = comp.velocity(xgal_kpc, ygal_kpc, zgal_kpc)
                            if comp._spatial_type != 'unresolved':
                                v_hiord_LOS = self.geometry.project_velocity_along_LOS(comp, v_hiord,
                                                                xgal_final, ygal_final, zgal_final)
                            else:
                                v_hiord_LOS = v_hiord


                            ## Must handle r=0 excising internally, because v_hiord is a 3-tuple sometimes
                            # v_hiord_LOS[rgal3D == 0] = v_hiord[rgal3D == 0]

                            #   No systemic velocity here bc this is relative to
                            #    the center of the galaxy at rest already
                            vobs_mass += v_hiord_LOS
                    #######

                    # Do complete cube propogation calculation
                    cube_final += cutils.populate_cube(flux_mass_transf, vobs_mass_transf, sigmar_transf, vx)
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # Remove the oversample from the geometry xyshift
            self.geometry.xshift = self.geometry.xshift.value / oversample
            self.geometry.yshift = self.geometry.yshift.value / oversample


        #######
        # Higher order components: those that DON'T follow general geometry,
        #                          or have OWN light distribution
        for cmp_n in self.higher_order_components:
            comp = self.higher_order_components[cmp_n]
            cmps_hiord_geoms = list(self.higher_order_geometries.keys())
            cmps_hiord_disps = list(self.higher_order_dispersions.keys())

            _do_comp = False
            if (comp.name in cmps_hiord_geoms):
                # Own geometry + light distribution
                _do_comp = True
                geom = self.higher_order_geometries[comp.name]
            elif (comp.name not in cmps_hiord_geoms) & (comp._separate_light_profile):
                # Own light distribution, uses galaxy geometry
                _do_comp = True
                geom = self.geometry
                logger.warning("The case of higher order component using galaxy geometry "
                               "but own light profile has not been tested")

            ######
            # Catch failure condition: _higher_order_type = 'perturbation' must NOT be included here,
            #       but rather above as a direct perturbation to the mass compoonent velocities.
            if _do_comp & (comp._higher_order_type.lower().strip() == 'perturbation'):
                msg = "Component with comp._higher_order_type = 'perturbation' "
                msg += "must NOT have own geometry or separate light profile!"
                raise ValueError(msg)

            if _do_comp:
                #######
                # Just create extra cube using the DIRECT calculation method

                nz_sky_samp, maxr, maxr_y = _calculate_max_skyframe_extents(geom,
                            nx_sky_samp, ny_sky_samp, transform_method, angle='sin')

                sh = (nz_sky_samp, ny_sky_samp, nx_sky_samp)

                # Apply the geometric transformation to get higher order coordinates
                # Account for oversampling
                geom.xshift = geom.xshift.value * oversample
                geom.yshift = geom.yshift.value * oversample
                xhiord, yhiord, zhiord, xsky, ysky, zsky = _get_xyz_sky_gal(geom, sh,
                                xcenter_samp, ycenter_samp, (nz_sky_samp - 1) / 2.)

                # Convert to kpc
                xhiord_kpc = xhiord * rstep_samp / dscale
                yhiord_kpc = yhiord * rstep_samp / dscale
                zhiord_kpc = zhiord * rstep_samp / dscale

                r_hiord = np.sqrt(xhiord**2 + yhiord**2 + zhiord**2)
                v_hiord = comp.velocity(xhiord_kpc, yhiord_kpc, zhiord_kpc)
                f_hiord = comp.light_profile(xhiord_kpc, yhiord_kpc, zhiord_kpc)

                # Apply extinction if it exists
                if self.extinction is not None:
                    f_hiord *= self.extinction(xsky, ysky, zsky)

                # LOS projection
                if comp._spatial_type != 'unresolved':
                    v_hiord_LOS = geom.project_velocity_along_LOS(comp, v_hiord, xhiord, yhiord, zhiord)
                else:
                    v_hiord_LOS = v_hiord

                ## Must handle r=0 excising internally, because v_hiord is a 3-tuple sometimes
                # v_hiord_LOS[r_hiord == 0] = v_hiord[r_hiord == 0]

                v_hiord_LOS += geom.vel_shift.value  # galaxy systemic velocity

                if (comp.name in cmps_hiord_disps):
                    sigma_hiord = self.higher_order_dispersions[comp.name](r_hiord)
                else:
                    # The higher-order term MUST have its own defined dispersion profile:
                    sigma_hiord = comp.dispersion_profile(xhiord_kpc, yhiord_kpc, zhiord_kpc)

                cube_final += cutils.populate_cube(f_hiord, v_hiord_LOS, sigma_hiord, vx)

                # Remove the oversample from the geometry xyshift
                geom.xshift = geom.xshift.value / oversample
                geom.yshift = geom.yshift.value / oversample


        return cube_final, spec
