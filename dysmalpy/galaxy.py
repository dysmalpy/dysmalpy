# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Main classes and functions for DYSMALPY for simulating the kinematics of
# a model galaxy and fitting it to observed data.


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging
import copy

import os
import datetime

from collections import OrderedDict

# Third party imports
import numpy as np
import astropy.cosmology as apy_cosmo
import astropy.units as u
import dill as _pickle

# dill py<=3.7 -> py>=3.8 + higher hack:
# See https://github.com/uqfoundation/dill/pull/406
_pickle._dill._reverse_typemap['CodeType'] = _pickle._dill._create_code

# Local imports
# Package imports
from dysmalpy.models import ModelSet
from dysmalpy.data_classes import Data0D, Data1D, Data2D, Data3D
from dysmalpy.instrument import Instrument
from dysmalpy.utils import apply_smoothing_3D, rebin, gaus_fit_sp_opt_leastsq
from dysmalpy import aperture_classes
from dysmalpy.utils_io import write_model_obs_file
from dysmalpy import config
from dysmalpy.observation import Observation

try:
    from dysmalpy.lensing import setup_lensing_transformer_from_params
    _loaded_lensing = True
except:
    _loaded_lensing = False

try:
    from dysmalpy.utils_least_chi_squares_1d_fitter import LeastChiSquares1D
    _loaded_LeastChiSquares1D = True
except:
    _loaded_LeastChiSquares1D = False

__all__ = ['Galaxy']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


# Default cosmology
_default_cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)


class Galaxy:
    r"""
    Container for simulating or modelling a galaxy

    `Galaxy` holds the observed data, model, observing instrument, and
    general information for a galaxy. This can be a simulated or real
    galaxy.

    Parameters
    ----------
    z : float
        Redshift of the galaxy
    cosmo : `~astropy.cosmology` object
            The cosmology to use for modelling. Default is
            astropy.cosmology.FlatLambdaCDM with H0=70., and Om0=0.3.
    model : `~dysmalpy.models.ModelSet` object
            A dysmalpy model to use for simulating and/or fitting data.
            This generates the intrinsic observables of the galaxy based
            on the components included in the ModelSet.
    obs_list : list
            List of `~dysmalpy.observation.Observation` objects,
            which hold `~dysmalpy.instrument.Instrument`, `~dysmalpy.observation.ObsOptions`,
            and  `~dysmalpy.data_classes.Data` instances.
            For each `obs`, `obs.instrument` and `obs.obs_options`
            describes how `model` is converted to observed space data.
    name : str, optional
           Name of the galaxy. Default is "galaxy."

    """

    def __init__(self, z=0, cosmo=_default_cosmo, obs_list=None,
                 model=None, name='galaxy'):

        self._z = z
        self.name = name
        if model is None:
            self.model = ModelSet()
        else:
            self.model = model

        self._cosmo = cosmo
        self._dscale = self._cosmo.arcsec_per_kpc_proper(self._z).value

        # v2.0: everything lives inside Observation instances, inside
        #       OrderedDict self.observations
        self.observations = OrderedDict()
        if obs_list is not None:
            for obs in obs_list:
                self.add_observation(obs)


    def __setstate__(self, state):
        # Compatibility hack, to handle the changed galaxy structure
        #    (properties, not attributes for data[*], instrument)

        self.__dict__ = state
        # quick test if necessary to migrate:
        if '_data' in state.keys():
            pass
        else:
            migrate_keys = ['data', 'data1d', 'data2d', 'data3d', 'instrument', 'dscale']
            for mkey in migrate_keys:
                if (mkey in state.keys()) and ('_{}'.format(mkey) not in state.keys()):
                    self.__dict__['_{}'.format(mkey)] = state[mkey]
                    del self.__dict__[mkey]

    def copy(self):
        return copy.deepcopy(self)

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        if value < 0:
            raise ValueError("Redshift can't be negative!")
        self._z = value

        # Reset dscale:
        self._set_dscale()

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

        # Reset dscale:
        self._set_dscale()

    @property
    def dscale(self):
        return self._dscale

    def _set_dscale(self):
        self._dscale = self._cosmo.arcsec_per_kpc_proper(self._z).value

    def add_observation(self, obs):
        """
        Add an observation to the galaxy.observations ordered dict.
        """
        obs_name = obs.name
        if obs_name in self.observations:
            logger.warning('Overwriting observation {}!'.format(obs_name))
        self.observations[obs_name] = obs


    def get_observation(self, obs_name):
        """
        Retrieve an observation from the galaxy.observations ordered dict.
        """
        try:
            return self.observations[obs_name]
        except KeyError:
            raise KeyError('{} not in self.observations !'.format(obs_name))


    def create_model_data(self, obs_list=None, **kwargs):
        r"""
        Function to simulate data for the galaxy

        The function will initially generate a data cube that will then be optionally
        reduced to 2D, 1D, or single spectrum data if specified. The generated cube
        can be accessed via `Galaxy.model_cube` and the generated final data products
        via `Galaxy.model_data`. Both of these attributes are `data_classes.Data` instances.

        Parameters
        ----------
        obs_list : list, optional
            List of observations to make models for.
            If omitted, will default to making models for all observations in the galaxy.

        """
        if obs_list is None:
            obs_list = self.observations.keys()

        for obs_name in obs_list:

            # Get the individual observation
            obs = self.observations[obs_name]

            ndim_final = obs.instrument.ndim
            line_center = obs.instrument.line_center
            nx_sky = obs.instrument.fov[0]
            ny_sky = obs.instrument.fov[1]
            spec_type = obs.instrument.spec_type
            spec_start = obs.instrument.spec_start.value
            spec_step = obs.instrument.spec_step.value
            spec_unit = obs.instrument.spec_step.unit
            nspec = obs.instrument.nspec
            pixscale = obs.instrument.pixscale.value
            oversample = obs.obs_options.oversample
            oversize = obs.obs_options.oversize

            # Apply lensing transformation if necessary
            this_lensing_transformer = None

            if _loaded_lensing:
                # Only check to get lensing transformer if the lensing modules were successfully loaded.
                if 'lensing_transformer' in kwargs:
                    if kwargs['lensing_transformer'] is not None:
                        this_lensing_transformer = kwargs['lensing_transformer']['0']

                this_lensing_transformer = setup_lensing_transformer_from_params(\
                        params = kwargs,
                        source_plane_nchan = obs.instrument.nspec,
                        image_plane_sizex = nx_sky * oversample * oversize,
                        image_plane_sizey = ny_sky * oversample * oversize,
                        image_plane_pixsc = pixscale / oversample,
                        reuse_lensing_transformer = this_lensing_transformer,
                        cache_lensing_transformer = True,
                        reuse_cached_lensing_transformer = True,
                        verbose = (logger.level >= logging.DEBUG),
                    )

                if this_lensing_transformer is not None:
                    orig_inst = copy.deepcopy(obs.instrument)
                    lens_inst = copy.deepcopy(obs.instrument)

                    lens_inst.fov = (this_lensing_transformer.source_plane_nx, this_lensing_transformer.source_plane_ny)
                    lens_inst.pixscale.value = this_lensing_transformer.source_plane_pixsc

                    obs.instrument = lens_inst

            else:
                # Check if the key lensing params ARE set -- passed in kwargs here to the call to
                #   `setup_lensing_transformer_from_params`.
                #   In this case, if the lensing loading failed, issue & raise an error.
                mesh_file = mesh_ra = mesh_dec = None
                if 'lensing_mesh' in kwargs:
                    mesh_file = kwargs['lensing_mesh']
                if 'lensing_ra' in kwargs:
                    mesh_ra = kwargs['lensing_ra']
                if 'lensing_dec' in kwargs:
                    mesh_dec = kwargs['lensing_dec']

                if ((mesh_file is not None) & (mesh_ra is not None) & (mesh_dec is not None)):
                    wmsg =  "dysmalpy.Galaxy.create_model_data:\n"
                    wmsg += "*******************************************\n"
                    wmsg += "*** ERROR ***\n"
                    wmsg += " dysmalpy.lensing could not be loaded.\n"
                    wmsg += " Unable to perform lensing transformation.\n"
                    wmsg += "*******************************************\n"
                    logger.error(wmsg)
                    raise ValueError(wmsg)


            # Run simulation for the specific observatoin
            sim_cube, spec = self.model.simulate_cube(obs, dscale=self.dscale)

            if this_lensing_transformer is not None:

                logger.debug('Applying lensing transformation '+str(datetime.datetime.now()))
                if this_lensing_transformer.source_plane_data_cube is None:
                    this_lensing_transformer.setSourcePlaneDataCube(sim_cube, verbose=False)
                else:
                    this_lensing_transformer.updateSourcePlaneDataCube(sim_cube, verbose=False)
                sim_cube = this_lensing_transformer.performLensingTransformation(verbose=False)
                sim_cube[np.isnan(sim_cube)] = 0.0

                # store back
                if 'lensing_transformer' in kwargs:
                    if kwargs['lensing_transformer'] is None:
                        kwargs['lensing_transformer'] = {'0': None}
                    kwargs['lensing_transformer']['0'] = this_lensing_transformer

                # mask by data mask if available
                if obs.data is not None:
                    if hasattr(obs.data, 'mask'):
                        if hasattr(obs.data.mask, 'shape'):
                            this_lensing_mask = None
                            if len(obs.data.mask.shape) == 2:
                                this_lensing_mask = obs.data.mask.astype(bool)
                                this_lensing_mask = np.repeat(this_lensing_mask[np.newaxis, :, :], nspec, axis=0)
                            elif len(obs.data.mask.shape) == 3:
                                this_lensing_mask = obs.data.mask.astype(bool)
                            if this_lensing_mask is not None:
                                if this_lensing_mask.shape == sim_cube.shape:
                                    sim_cube[~this_lensing_mask] = 0.0
                # oversample oversize
                logger.debug('Applied lensing transformation '+str(datetime.datetime.now()))

                # Reset the observation instrument back to the original one
                obs.instrument = orig_inst

            # Correct for any oversampling
            if (oversample > 1):
                sim_cube_nooversamp = rebin(sim_cube, (ny_sky*oversize,
                                    nx_sky*oversize))

            # Apply beam smearing and/or instrumental spreading
            if obs.instrument is not None:
                sim_cube_obs = obs.instrument.convolve(cube=sim_cube_nooversamp,
                                                        spec_center=obs.instrument.line_center)
            else:
                sim_cube_obs = sim_cube_nooversamp


            # Re-size the cube back down
            if oversize > 1:
                nx_oversize = sim_cube_obs.shape[2]
                ny_oversize = sim_cube_obs.shape[1]
                sim_cube_final = sim_cube_obs[:,
                    np.int(ny_oversize/2 - ny_sky/2):np.int(ny_oversize/2+ny_sky/2),
                    np.int(nx_oversize/2 - nx_sky/2):np.int(nx_oversize/2+nx_sky/2)]

            else:
                sim_cube_final = sim_cube_obs

            obs.model_cube = Data3D(cube=sim_cube_final, pixscale=pixscale,
                                    spec_type=obs.instrument.spec_type,
                                    spec_arr=spec,
                                    spec_unit=obs.instrument.spec_step.unit)

            if ndim_final == 3:

                if obs.instrument.smoothing_type is not None:
                    obs.model_cube.data = apply_smoothing_3D(obs.model_cube.data,
                            smoothing_type=obs.instrument.smoothing_type,
                            smoothing_npix=obs.instrument.smoothing_npix)

                sim_cube_final_scale = obs.model_cube.data._data.copy()
                if obs.data is not None:
                    if obs.data.flux_map is None:
                        #mask_flat = np.sum(self.data.mask, axis=0)
                        num = np.sum(obs.data.mask * (obs.data.data.unmasked_data[:].value *
                                     obs.model_cube.data / (obs.data.error.unmasked_data[:].value**2)), axis=0)
                        den = np.sum(obs.data.mask*
                                        (obs.model_cube.data**2/(obs.data.error.unmasked_data[:].value**2)), axis=0)

                        scale = num / den
                        ## Handle zeros:
                        scale[den == 0.] = 0.
                        scale3D = np.zeros(shape=(1, scale.shape[0], scale.shape[1],))
                        scale3D[0, :, :] = scale
                        sim_cube_final_scale *= scale3D

                    else:
                        model_peak = np.nanmax(obs.model_cube.data, axis=0)
                        scale = obs.data.flux_map/model_peak
                        scale3D = np.zeros((1, scale.shape[0], scale.shape[1]))
                        scale3D[0, :, :] = scale
                        sim_cube_final_scale *= scale3D

                    mask_cube = obs.data.mask.copy()

                else:

                    mask_cube = None

                obs.model_data = Data3D(cube=sim_cube_final_scale, pixscale=pixscale,
                                         mask_cube=mask_cube,
                                         spec_type=obs.instrument.spec_type,
                                         spec_arr=spec,
                                         spec_unit=obs.instrument.spec_step.spec_unit)

            elif ndim_final == 2:

                if obs.instrument.smoothing_type is not None:
                    obs.model_cube.data = apply_smoothing_3D(obs.model_cube.data,
                                smoothing_type=obs.instrument.smoothing_type,
                                smoothing_npix=obs.instrument.smoothing_npix)

                if 'moment' in obs.instrument.__dict__.keys():
                    if obs.instrument.moment:
                        extrac_type = 'moment'
                    else:
                        extrac_type = 'gauss'
                else:
                    extrac_type = 'gauss'

                if spec_type == "velocity":
                    if extrac_type == 'moment':
                        flux = obs.model_cube.data.moment0().to(u.km/u.s).value
                        vel = obs.model_cube.data.moment1().to(u.km/u.s).value
                        disp = obs.model_cube.data.linewidth_sigma().to(u.km/u.s).value
                    elif extrac_type == 'gauss':
                        mom0 = obs.model_cube.data.moment0().to(u.km/u.s).value
                        mom1 = obs.model_cube.data.moment1().to(u.km/u.s).value
                        mom2 = obs.model_cube.data.linewidth_sigma().to(u.km/u.s).value
                        flux = np.zeros(mom0.shape)
                        vel = np.zeros(mom0.shape)
                        disp = np.zeros(mom0.shape)
                        # <DZLIU><20210805> ++++++++++
                        my_least_chi_squares_1d_fitter = None
                        if ('gauss_extract_with_c' in kwargs) & (_loaded_LeastChiSquares1D):
                            if kwargs['gauss_extract_with_c'] is not None and \
                               kwargs['gauss_extract_with_c'] is not False:
                                # we will use the C++ LeastChiSquares1D to run the 1d spectral fitting
                                # but note that if a spectrum has data all too close to zero, it will fail.
                                # try to prevent this by excluding too low data
                                if obs.data.mask is not None:
                                    this_fitting_mask = copy.copy(obs.data.mask)
                                else:
                                    this_fitting_mask = 'auto'
                                if logger.level > logging.DEBUG:
                                    this_fitting_verbose = True
                                else:
                                    this_fitting_verbose = False
                                # do the least chisquares fitting
                                my_least_chi_squares_1d_fitter = LeastChiSquares1D(\
                                        x = obs.model_cube.data.spectral_axis.to(u.km/u.s).value,
                                        data = obs.model_cube.data.unmasked_data[:,:,:].value,
                                        dataerr = None,
                                        datamask = this_fitting_mask,
                                        initparams = np.array([mom0 / np.sqrt(2 * np.pi) / np.abs(mom2), mom1, mom2]),
                                        nthread = 4,
                                        verbose = this_fitting_verbose)
                        if my_least_chi_squares_1d_fitter is not None:
                            logger.debug('my_least_chi_squares_1d_fitter '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
                            my_least_chi_squares_1d_fitter.runFitting()
                            flux = my_least_chi_squares_1d_fitter.outparams[0,:,:] * np.sqrt(2 * np.pi) * my_least_chi_squares_1d_fitter.outparams[2,:,:]
                            vel = my_least_chi_squares_1d_fitter.outparams[1,:,:]
                            disp = my_least_chi_squares_1d_fitter.outparams[2,:,:]
                            flux[np.isnan(flux)] = 0.0 #<DZLIU><DEBUG># 20210809 fixing this bug
                            logger.debug('my_least_chi_squares_1d_fitter '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
                        else:
                            for i in range(mom0.shape[0]):
                                for j in range(mom0.shape[1]):
                                    if i==0 and j==0:
                                        logger.debug('gaus_fit_sp_opt_leastsq '+str(mom0.shape[0])+'x'+str(mom0.shape[1])+' '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
                                    best_fit = gaus_fit_sp_opt_leastsq(obs.model_cube.data.spectral_axis.to(u.km/u.s).value,
                                                        obs.model_cube.data.unmasked_data[:,i,j].value,
                                                        mom0[i,j], mom1[i,j], mom2[i,j])
                                    flux[i,j] = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
                                    vel[i,j] = best_fit[1]
                                    disp[i,j] = best_fit[2]
                                    if i==mom0.shape[0]-1 and j==mom0.shape[1]-1:
                                        logger.debug('gaus_fit_sp_opt_leastsq '+str(mom0.shape[0])+'x'+str(mom0.shape[1])+' '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
                        # <DZLIU><20210805> ----------

                elif spec_type == "wavelength":

                    cube_with_vel = obs.model_cube.data.with_spectral_unit(u.km/u.s,
                        velocity_convention='optical',
                        rest_value=obs.instrument.line_center)

                    if extrac_type == 'moment':
                        flux = cube_with_vel.moment0().value
                        vel = cube_with_vel.moment1().value
                        disp = cube_with_vel.linewidth_sigma().value
                    elif extrac_type == 'gauss':
                        raise ValueError("Not yet supported!")

                    disp[np.isnan(disp)] = 0.

                else:
                    raise ValueError("spec_type can only be 'velocity' or "
                                     "'wavelength.'")


                if obs.data is not None:
                    if obs.data.mask is not None:
                        # Copy data mask:
                        mask = copy.deepcopy(self.data.mask)

                        # Normalize flux:
                        if (obs.data.data['flux'] is not None) & (obs.data.error['flux'] is not None):
                            num = np.nansum(obs.data.mask*(obs.data.data['flux']*flux)/(obs.data.error['flux']**2))
                            den = np.nansum(obs.data.mask*(flux**2)/(obs.data.error['flux']**2))

                            scale = num / den
                            flux *= scale
                        elif (obs.data.data['flux'] is not None):
                            num = np.nansum(obs.data.mask*(obs.data.data['flux']*flux))
                            den = np.nansum(obs.data.mask*(flux**2))
                            scale = num / den
                            flux *= scale
                    else:
                        mask = None
                else:
                    mask = None

                obs.model_data = Data2D(pixscale=pixscale, velocity=vel,
                                        vel_disp=disp, flux=flux, mask=mask)

            elif ndim_final == 1:

                if spec_type == 'wavelength':

                    cube_with_vel = obs.model_cube.data.with_spectral_unit(
                        u.km / u.s, velocity_convention='optical',
                        rest_value=obs.instrument.line_center)

                    cube_data = cube_with_vel.unmasked_data[:]
                    vel_arr = cube_with_vel.spectral_axis.to(u.km/u.s).value

                elif spec_type == 'velocity':

                    cube_data = sim_cube_obs
                    vel_arr = spec

                else:
                    raise ValueError("spec_type can only be 'velocity' or "
                                     "'wavelength.'")

                try:
                    # Catch case where center_pixel is (None, None)
                    if (obs.obs_options.xcenter is not None) & (obs.obs_options.ycenter is not None):
                        center_pixel = (obs.obs_options.xcenter, obs.obs_options.ycenter)
                    else:
                        center_pixel = None
                except:
                    center_pixel = None

                aper_centers, flux1d, vel1d, disp1d = obs.instrument.apertures.extract_1d_kinematics(spec_arr=vel_arr,
                        cube=cube_data, center_pixel = center_pixel, pixscale=pixscale)

                if obs.data is not None:
                    # Get mask:
                    mask1d = copy.deepcopy(obs.data.mask)

                    # Normalize flux:
                    if (obs.data.data['flux'] is not None) & (obs.data.error['flux'] is not None):
                        if (flux1d.shape[0] == obs.data.data['flux'].shape[0]):
                            num = np.sum(obs.data.mask*(obs.data.data['flux']*flux1d)/(obs.data.error['flux']**2))
                            den = np.sum(obs.data.mask*(flux1d**2)/(obs.data.error['flux']**2))

                            scale = num / den
                            flux1d *= scale
                    elif (obs.data.data['flux'] is not None):
                        if (flux1d.shape[0] == obs.data.data['flux'].shape[0]):
                            num = np.sum(obs.data.mask*(obs.data.data['flux']*flux1d))
                            den = np.sum(obs.data.mask*(flux1d**2))
                            scale = num / den
                            flux1d *= scale
                else:
                    mask1d = None

                # Gather results:
                obs.model_data = Data1D(r=aper_centers, velocity=vel1d,
                                        vel_disp=disp1d, flux=flux1d, mask=mask1d)

            elif ndim_final == 0:

                if obs.instrument.integrate_cube:

                    # Integrate over the spatial dimensions of the cube
                    flux = np.nansum(np.nansum(obs.model_cube.data.unmasked_data[:], axis=2), axis=1)

                    # Normalize to the maximum of the spectrum
                    flux /= np.nanmax(flux)
                    flux = flux.value

                else:

                    # Place slit down on cube
                    raise NotImplementedError('Using slits to create spectrum not implemented yet!')

                obs.model_data = Data0D(x=spec, flux=flux)

            ####

            # Reset observation within the observations ordered dict:
            self.observations[obs.name] = obs


    def preserve_self(self, filename=None, save_data=True, overwrite=False):
        # Check for existing file:
        if (not overwrite) and (filename is not None):
            if os.path.isfile(filename):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
                return None

        if filename is not None:
            galtmp = copy.deepcopy(self)

            galtmp.filename_velocity = copy.deepcopy(galtmp.data.filename_velocity)
            galtmp.filename_dispersion = copy.deepcopy(galtmp.data.filename_dispersion)

            if not save_data:
                galtmp.data = None
                galtmp.model_data = None
                galtmp.model_cube = None

            _pickle.dump(galtmp, open(filename, "wb") )

    def load_self(self, filename=None):
        """
        Load a saved Galaxy from a pickle file

        Parameters
        ----------
        filename : str
            Name of the file with saved Galaxy

        Returns
        -------

        """
        if filename is not None:
            galtmp = _pickle.load(open(filename, "rb"))
            return galtmp


    def save_model_data(self, filename=None, overwrite=False):
        # Check for existing file:
        if (not overwrite) and (filename is not None):
            if os.path.isfile(filename):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
                return None

        if filename is not None:
            write_model_obs_file(gal=self, fname=filename,
                            ndim=self.model_data.ndim, overwrite=overwrite)



def load_galaxy_object(filename=None):
    """
    Load a saved Galaxy from a pickle file

    Parameters
    ----------
    filename : str
               Name of the file with saved Galaxy

    Returns
    -------
    gal: Galaxy object
         The saved dysmalpy Galaxy object

    """
    gal = Galaxy()
    gal = gal.load_self(filename=filename)
    return gal
