# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# File containing Observation and ObservationSet classes which define the individual and set
# of observations of a galaxy.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard imports
import logging
import copy

import datetime

# Third party imports
import numpy as np
import astropy.units as u

# Package imports
from dysmalpy.data_classes import Data0D, Data1D, Data2D, Data3D
from dysmalpy.instrument import Instrument
from dysmalpy.utils import apply_smoothing_3D, rebin, gaus_fit_sp_opt_leastsq

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


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


__all__ = ["Observation", "ObsModOptions", "ObsFitOptions"]


class Observation:
    """
    Class defining an individual observation.

    Each observation consists of three component: instrument defining the instrument
    setup for the observation, the observed data, and any generated dysmalpy model data.
    """

    def __init__(self, name, tracer, weight=1.0, instrument=None, data=None):

        self.name = name
        self.tracer = tracer
        self.weight = weight
        self._instrument = None
        self._data = None
        self.model_cube = None
        self.model_data = None
        self.mod_options = ObsModOptions()
        self.fit_options = ObsFitOptions()
        self.lensing_options = ObsLensingOptions()


        if instrument is not None:
            self.add_instrument(instrument)

        if data is not None:
            self.add_data(data)

    def __deepcopy__(self, memo):
        self2 = type(self)(name=self.name, tracer=self.tracer, weight=self.weight,
                           instrument=self._instrument, data=self._data)
        self2.__dict__.update(self.__dict__)
        return self2

    def __copy__(self):
        self2 = type(self)(name=self.name, tracer=self.tracer, weight=self.weight,
                           instrument=self._instrument, data=self._data)
        self2.__dict__.update(self.__dict__)
        return self2

    def add_instrument(self, instrument):
        self.instrument = instrument

    def add_data(self, data):
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        if not np.any(isinstance(new_data, Data1D) | \
                      isinstance(new_data, Data2D) | \
                      isinstance(new_data, Data3D) | \
                      isinstance(new_data, Data0D) | \
                      isinstance(new_data, type(None))):
            raise TypeError("Data must be one of the following instances: "
                            "   dysmalpy.Data0D, dysmalpy.Data1D, "
                            "   dysmalpy.Data2D, dysmalpy.Data3D, or None")
        self._data = new_data
        self._setup_checks()

    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, new_instrument):
        if not (isinstance(new_instrument, Instrument)) | \
                isinstance(new_instrument, type(None)):
            raise TypeError("Instrument must be a dysmalpy.Instrument instance.")
        self._instrument = new_instrument
        self._setup_checks()

    def _setup_checks(self):
        self._check_1d_datasize()
        self._check_2d_instrument()
        self._check_3d_instrument()

    def _check_1d_datasize(self):
        if (self.data is not None) & (self.instrument is not None):
            if (isinstance(self.data, Data1D)):
                # --------------------------------------------------
                # Check FOV and issue warning if too small:
                maxr = np.max(np.abs(self.data.rarr))
                rstep = self.instrument.pixscale.value
                if ((self.instrument.fov[0] < maxr/rstep) | (self.instrument.fov[1] < maxr/rstep)):
                    wmsg =  "dysmalpy.Galaxy:\n"
                    wmsg += "********************************************************************\n"
                    wmsg += "*** WARNING ***\n"
                    wmsg += "instrument.fov[0,1]="
                    wmsg += "({},{}) is too small".format(self.instrument.fov[0], self.instrument.fov[1])
                    wmsg += " for max data extent ({} pix)\n".format(maxr/rstep)
                    wmsg += "********************************************************************\n"
                    logger.warning(wmsg)
                    raise ValueError(wmsg)
                # --------------------------------------------------

    def _check_2d_instrument(self):
        if (self.data is not None) & (self.instrument is not None):
            if (isinstance(self.data, Data2D)):
                # --------------------------------------------------
                # Check FOV on instrument and reset if not matching:
                if ((self.instrument.fov[0] != self.data.shape[1]) | \
                   (self.instrument.fov[1] != self.data.shape[0])):
                    wmsg =  "dysmalpy.Galaxy:\n"
                    wmsg += "********************************************************************\n"
                    wmsg += "*** INFO ***\n"
                    wmsg += "instrument.fov[0,1]="
                    wmsg += "({},{}) is being reset".format(self.instrument.fov[0], self.instrument.fov[1])
                    wmsg += " to match 2D maps ({}, {})\n".format(self.data.shape[1], self.data.shape[0])
                    wmsg += "********************************************************************\n"
                    logger.info(wmsg)
                    self.instrument.fov = [self.data.shape[1], self.data.shape[0]]
                    # Reset kernel
                    self.instrument._beam_kernel = None
                # --------------------------------------------------

    def _check_3d_instrument(self):
        if (self.data is not None) & (self.instrument is not None):
            if (isinstance(self.data, Data3D)):
                # --------------------------------------------------
                # Check FOV on instrument and reset if not matching:
                if ((self.instrument.fov[0] != self.data.shape[2]) | \
                   (self.instrument.fov[1] != self.data.shape[1])):
                    wmsg =  "dysmalpy.Galaxy:\n"
                    wmsg += "********************************************************************\n"
                    wmsg += "*** INFO ***\n"
                    wmsg += "instrument.fov[0,1]="
                    wmsg += "({},{}) is being reset".format(self.instrument.fov[0], self.instrument.fov[1])
                    wmsg += " to match 3D cube ({}, {})\n".format(self.data.shape[2], self.data.shape[1])
                    wmsg += "********************************************************************\n"
                    logger.info(wmsg)
                    self.instrument.fov = [self.data.shape[2], self.data.shape[1]]
                    # Reset kernel
                    self.instrument._beam_kernel = None


                # --------------------------------------------------
                # Check instrument pixel scale and reset if not matching:
                pixdifftol = 1.e-10 * self.instrument.pixscale.unit
                convunit = self.data.data.wcs.wcs.cunit[0].to(self.instrument.pixscale.unit) * \
                            self.instrument.pixscale.unit
                if np.abs(self.instrument.pixscale -  self.data.data.wcs.wcs.cdelt[0]*convunit) > pixdifftol:
                    wmsg =  "dysmalpy.Galaxy:\n"
                    wmsg += "********************************************************************\n"
                    wmsg += "*** INFO ***\n"
                    wmsg += "instrument.pixscale="
                    wmsg += "{} is being reset".format(self.instrument.pixscale)
                    wmsg += "   to match 3D cube ({})\n".format(self.data.data.wcs.wcs.cdelt[0]*convunit)
                    wmsg += "********************************************************************\n"
                    logger.info(wmsg)
                    self.instrument.pixscale = self.data.data.wcs.wcs.cdelt[0]*convunit
                    # Reset kernel
                    self.instrument._beam_kernel = None
                # --------------------------------------------------



                # --------------------------------------------------
                # Check instrument spectral array and reset if not matching:
                spec_ctype = self.data.data.wcs.wcs.ctype[-1]
                nspec = self.data.shape[0]
                if spec_ctype == 'WAVE':
                    spec_type = 'wavelength'
                elif (spec_ctype == 'VOPT'):
                    spec_type = 'velocity'
                spec_start = self.data.data.spectral_axis[0]
                spec_step = (self.data.data.spectral_axis[1]-
                             self.data.data.spectral_axis[0])
                specdifftol = 1.e-10 * spec_step.unit
                if ((self.instrument.spec_type != spec_type) | \
                   (self.instrument.nspec != nspec) | \
                   (np.abs(self.instrument.spec_start.to(spec_start.unit) - spec_start)>specdifftol) | \
                   (np.abs(self.instrument.spec_step.to(spec_step.unit) - spec_step)>specdifftol) ):
                    wmsg =  "dysmalpy.Galaxy:\n"
                    wmsg += "********************************************************************\n"
                    wmsg += "*** INFO ***\n"
                    wmsg += "instrument spectral settings are being reset\n"
                    wmsg += "   (spec_type={}, spec_start={:0.2f}, spec_step={:0.2f}, nspec={})\n".format(self.instrument.spec_type,
                                    self.instrument.spec_start, self.instrument.spec_step, self.instrument.nspec)
                    wmsg += "   to match 3D cube\n"
                    wmsg += "   (spec_type={}, spec_start={:0.2f}, spec_step={:0.2f}, nspec={})\n".format(spec_type,
                                 spec_start, spec_step, nspec)
                    wmsg += "********************************************************************\n"
                    logger.info(wmsg)
                    self.instrument.spec_type = spec_type
                    self.instrument.spec_step = spec_step
                    self.instrument.spec_start = spec_start
                    self.instrument.nspec = nspec
                    # Reset kernel
                    self.instrument._lsf_kernel = None
                # --------------------------------------------------

    def create_single_obs_model_data(self, model, dscale):
        r"""
        Function to simulate data for the galaxy

        The function will initially generate a data cube that will then be optionally
        reduced to 2D, 1D, or single spectrum data if specified. The generated cube
        can be accessed via `Galaxy.model_cube` and the generated final data products
        via `Galaxy.model_data`. Both of these attributes are `data_classes.Data` instances.

        Parameters
        ----------
        model : ModelSet instance
            Galaxy model set

        dscale : Galaxy dscale
        """

        ndim_final = self.instrument.ndim
        # line_center = self.instrument.line_center
        nx_sky = self.instrument.fov[0]
        ny_sky = self.instrument.fov[1]
        spec_type = self.instrument.spec_type
        # spec_start = self.instrument.spec_start.value
        # spec_step = self.instrument.spec_step.value
        spec_unit = self.instrument.spec_step.unit
        nspec = self.instrument.nspec
        pixscale = self.instrument.pixscale.value
        oversample = self.mod_options.oversample
        oversize = self.mod_options.oversize

        # Apply lensing transformation if necessary
        this_lensing_transformer = None

        # Check if self.lensing_options IS set and valid -- passed to the call to
        #   `setup_lensing_transformer_from_params`.
        #   In this case, if the lensing loading failed, issue & raise an error.
        
        if self.lensing_options is not None and self.lensing_options.is_valid():
            
            # Make sure lensing modules were successfully loaded.
            if not _loaded_lensing:
                wmsg =  "dysmalpy.Galaxy.create_model_data:\n"
                wmsg += "*******************************************\n"
                wmsg += "*** ERROR ***\n"
                wmsg += " dysmalpy.lensing could not be loaded.\n"
                wmsg += " Unable to perform lensing transformation.\n"
                wmsg += "*******************************************\n"
                logger.error(wmsg)
                raise Exception(wmsg)

            # Call lensing.setup_lensing_transformer_from_params
            this_lensing_transformer = setup_lensing_transformer_from_params(\
                    **self.lensing_options.get_lensing_kwargs(oversample=oversample, oversize=oversize),
                )

            # Temporarily use a lens_inst
            orig_inst = copy.deepcopy(self.instrument)
            lens_inst = copy.deepcopy(self.instrument)
            #orig_data = copy.deepcopy(self.data)
            #lens_data = np.zeros((this_lensing_transformer.source_plane_nchan, 
                                  #this_lensing_transformer.source_plane_ny, 
                                  #this_lensing_transformer.source_plane_nx))

            lens_inst.fov = (this_lensing_transformer.source_plane_nx,
                             this_lensing_transformer.source_plane_ny)
            lens_inst.pixscale = this_lensing_transformer.source_plane_pixsc * u.arcsec

            self._instrument = lens_inst
            #self._data = lens_data


        # Run simulation for the specific observation
        sim_cube, spec = model.simulate_cube(obs=self, dscale=dscale)


        # Apply lensing transformation if necessary
        if this_lensing_transformer is not None:
            logger.debug('Applying lensing transformation '+str(datetime.datetime.now()))
            if this_lensing_transformer.source_plane_data_cube is None:
                this_lensing_transformer.setSourcePlaneDataCube(sim_cube, verbose=False)
            else:
                this_lensing_transformer.updateSourcePlaneDataCube(sim_cube, verbose=False)
            sim_cube = this_lensing_transformer.performLensingTransformation(verbose=False)
            sim_cube[np.isnan(sim_cube)] = 0.0
            # mask by data mask if available
            if self.data is not None:
                if hasattr(self.data, 'mask'):
                    if hasattr(self.data.mask, 'shape'):
                        this_lensing_mask = None
                        if len(self.data.mask.shape) == 2:
                            this_lensing_mask = self.data.mask.astype(bool)
                            this_lensing_mask = np.repeat(this_lensing_mask[np.newaxis, :, :], nspec, axis=0)
                        elif len(self.data.mask.shape) == 3:
                            this_lensing_mask = self.data.mask.astype(bool)
                        if this_lensing_mask is not None:
                            if this_lensing_mask.shape == sim_cube.shape:
                                sim_cube[~this_lensing_mask] = 0.0
            logger.debug('Applied lensing transformation '+str(datetime.datetime.now()))

            # Reset the observation instrument back to the original one
            self._instrument = orig_inst

        
        # Correct for any oversampling
        if (oversample > 1):
            sim_cube_nooversamp = rebin(sim_cube, (ny_sky*oversize,
                                nx_sky*oversize))
        else:
            sim_cube_nooversamp = sim_cube

        # Apply beam smearing and/or instrumental spreading
        if self.instrument is not None:
            sim_cube_obs = self.instrument.convolve(cube=sim_cube_nooversamp,
                                                    spec_center=self.instrument.line_center)
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

        self.model_cube = Data3D(cube=sim_cube_final, pixscale=pixscale,
                                 spec_arr=spec, spec_type=spec_type,
                                 spec_unit=spec_unit)

        if (ndim_final == 3):

            if self.instrument.smoothing_type is not None:
                self.model_cube.data = apply_smoothing_3D(self.model_cube.data,
                        smoothing_type=self.instrument.smoothing_type,
                        smoothing_npix=self.instrument.smoothing_npix)

            sim_cube_final_scale = self.model_cube.data._data.copy()
            if self.data is not None:
                if self.data.flux_map is None:
                    #mask_flat = np.sum(self.data.mask, axis=0)
                    num = np.sum(self.data.mask * (self.data.data.unmasked_data[:].value *
                                 self.model_cube.data / (self.data.error.unmasked_data[:].value**2)), axis=0)
                    den = np.sum(self.data.mask*
                                    (self.model_cube.data**2/(self.data.error.unmasked_data[:].value**2)), axis=0)

                    scale = num / den
                    ## Handle zeros:
                    scale[den == 0.] = 0.
                    scale3D = np.zeros(shape=(1, scale.shape[0], scale.shape[1],))
                    scale3D[0, :, :] = scale
                    sim_cube_final_scale *= scale3D

                else:
                    model_peak = np.nanmax(self.model_cube.data, axis=0)
                    scale = self.data.flux_map/model_peak
                    scale3D = np.zeros((1, scale.shape[0], scale.shape[1]))
                    scale3D[0, :, :] = scale
                    sim_cube_final_scale *= scale3D

                mask_cube = self.data.mask.copy()

            else:

                mask_cube = None

            self.model_data = Data3D(cube=sim_cube_final_scale, pixscale=pixscale,
                                     mask_cube=mask_cube, spec_arr=spec,
                                     spec_type=spec_type, spec_unit=spec_unit)

        elif (ndim_final == 2):
            if self.instrument.smoothing_type is not None:
                self.model_cube.data = apply_smoothing_3D(self.model_cube.data,
                            smoothing_type=self.instrument.smoothing_type,
                            smoothing_npix=self.instrument.smoothing_npix)

            if 'moment' in self.instrument.__dict__.keys():
                if self.instrument.moment:
                    extrac_type = 'moment'
                else:
                    extrac_type = 'gauss'
            else:
                extrac_type = 'gauss'

            if spec_type == "velocity":
                if extrac_type == 'moment':
                    flux = self.model_cube.data.moment0().to(u.km/u.s).value
                    vel = self.model_cube.data.moment1().to(u.km/u.s).value
                    disp = self.model_cube.data.linewidth_sigma().to(u.km/u.s).value
                elif extrac_type == 'gauss':
                    mom0 = self.model_cube.data.moment0().to(u.km/u.s).value
                    mom1 = self.model_cube.data.moment1().to(u.km/u.s).value
                    mom2 = self.model_cube.data.linewidth_sigma().to(u.km/u.s).value
                    flux = np.zeros(mom0.shape)
                    vel = np.zeros(mom0.shape)
                    disp = np.zeros(mom0.shape)
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++
                    my_least_chi_squares_1d_fitter = None
                    if (_loaded_LeastChiSquares1D):
                        if self.mod_options.gauss_extract_with_c:
                            # # we will use the C++ LeastChiSquares1D to run the 1d spectral fitting
                            # # but note that if a spectrum has data all too close to zero, it will fail.
                            # # try to prevent this by excluding too low data
                            # this_fitting_mask = 'auto'
                            this_fitting_mask = None
                            if self.data is not None:
                                if self.data.mask is not None:
                                    this_fitting_mask = copy.copy(self.data.mask)

                            # # Only do verbose if logging level is DEBUG or lower
                            # if logger.level <= logging.DEBUG:
                            #     this_fitting_verbose = True
                            # else:
                            #     this_fitting_verbose = False
                            ## Force non-verbose, because multiprocessing pool
                            ##    resets logging to logger.level = 0....
                            this_fitting_verbose = False

                            # do the least chisquares fitting
                            my_least_chi_squares_1d_fitter = LeastChiSquares1D(\
                                    x = self.model_cube.data.spectral_axis.to(u.km/u.s).value,
                                    data = self.model_cube.data.unmasked_data[:,:,:].value,
                                    dataerr = None,
                                    datamask = this_fitting_mask,
                                    initparams = np.array([mom0 / np.sqrt(2 * np.pi) / np.abs(mom2), mom1, mom2]),
                                    nthread = 4,
                                    verbose = this_fitting_verbose)
                    if my_least_chi_squares_1d_fitter is not None:
                        logger.debug('my_least_chi_squares_1d_fitter '+str(datetime.datetime.now()))
                        my_least_chi_squares_1d_fitter.runFitting()
                        flux = my_least_chi_squares_1d_fitter.outparams[0,:,:] * np.sqrt(2 * np.pi) * my_least_chi_squares_1d_fitter.outparams[2,:,:]
                        vel = my_least_chi_squares_1d_fitter.outparams[1,:,:]
                        disp = my_least_chi_squares_1d_fitter.outparams[2,:,:]
                        flux[np.isnan(flux)] = 0.0
                        logger.debug('my_least_chi_squares_1d_fitter '+str(datetime.datetime.now()))
                    else:
                        for i in range(mom0.shape[0]):
                            for j in range(mom0.shape[1]):
                                if i==0 and j==0:
                                    logger.debug('gaus_fit_sp_opt_leastsq '+str(mom0.shape[0])+'x'+str(mom0.shape[1])+' '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
                                best_fit = gaus_fit_sp_opt_leastsq(self.model_cube.data.spectral_axis.to(u.km/u.s).value,
                                                    self.model_cube.data.unmasked_data[:,i,j].value,
                                                    mom0[i,j], mom1[i,j], mom2[i,j])
                                flux[i,j] = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
                                vel[i,j] = best_fit[1]
                                disp[i,j] = best_fit[2]
                                if i==(mom0.shape[0]-1) and j==(mom0.shape[1]-1):
                                    logger.debug('gaus_fit_sp_opt_leastsq '+str(mom0.shape[0])+'x'+str(mom0.shape[1])+' '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++

            elif spec_type == "wavelength":
                cube_with_vel = self.model_cube.data.with_spectral_unit(u.km/u.s,
                    velocity_convention='optical',
                    rest_value=self.instrument.line_center)

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

            if self.data is not None:
                if self.data.mask is not None:
                    # Copy data mask:
                    mask = copy.deepcopy(self.data.mask)

                    # Normalize flux:
                    if (self.data.data['flux'] is not None) & (self.data.error['flux'] is not None):
                        num = np.nansum(self.data.mask*(self.data.data['flux']*flux)/(self.data.error['flux']**2))
                        den = np.nansum(self.data.mask*(flux**2)/(self.data.error['flux']**2))

                        scale = num / den
                        flux *= scale
                    elif (self.data.data['flux'] is not None):
                        num = np.nansum(self.data.mask*(self.data.data['flux']*flux))
                        den = np.nansum(self.data.mask*(flux**2))
                        scale = num / den
                        flux *= scale
                else:
                    mask = None
            else:
                mask = None

            self.model_data = Data2D(pixscale=pixscale, velocity=vel,
                                     vel_disp=disp, flux=flux, mask=mask)

        elif (ndim_final == 1):
            if spec_type == 'wavelength':

                cube_with_vel = self.model_cube.data.with_spectral_unit(
                    u.km / u.s, velocity_convention='optical',
                    rest_value=self.instrument.line_center)

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
                if (self.mod_options.xcenter is not None) & (self.mod_options.ycenter is not None):
                    center_pixel = (self.mod_options.xcenter, self.mod_options.ycenter)
                else:
                    center_pixel = None
            except:
                center_pixel = None

            aper_centers, flux1d, vel1d, disp1d = self.instrument.apertures.extract_1d_kinematics(spec_arr=vel_arr,
                    cube=cube_data, center_pixel = center_pixel, pixscale=pixscale)

            if self.data is not None:
                # Get mask:
                mask1d = copy.deepcopy(self.data.mask)

                # Normalize flux:
                if (self.data.data['flux'] is not None) & (self.data.error['flux'] is not None):
                    if (flux1d.shape[0] == self.data.data['flux'].shape[0]):
                        num = np.sum(self.data.mask*(self.data.data['flux']*flux1d)/(self.data.error['flux']**2))
                        den = np.sum(self.data.mask*(flux1d**2)/(self.data.error['flux']**2))

                        scale = num / den
                        flux1d *= scale
                elif (self.data.data['flux'] is not None):
                    if (flux1d.shape[0] == self.data.data['flux'].shape[0]):
                        num = np.sum(self.data.mask*(self.data.data['flux']*flux1d))
                        den = np.sum(self.data.mask*(flux1d**2))
                        scale = num / den
                        flux1d *= scale
            else:
                mask1d = None

            # Gather results:
            self.model_data = Data1D(r=aper_centers, velocity=vel1d,
                                     vel_disp=disp1d, flux=flux1d, mask=mask1d)

        elif (ndim_final == 0):
            if self.instrument.integrate_cube:

                # Integrate over the spatial dimensions of the cube
                flux = np.nansum(np.nansum(self.model_cube.data.unmasked_data[:], axis=2), axis=1)

                # Normalize to the maximum of the spectrum
                flux /= np.nanmax(flux)
                flux = flux.value

            else:

                # Place slit down on cube
                raise NotImplementedError('Using slits to create spectrum not implemented yet!')

            self.model_data = Data0D(x=spec, flux=flux)

        ####



class ObsModOptions:
    """
    Class to hold options for creating the observed model
    """
    def __init__(self, xcenter=None, ycenter=None, oversample=1, oversize=1,
                 transform_method='direct', zcalc_truncate=None, n_wholepix_z_min=3,
                 gauss_extract_with_c=True):

        self.xcenter = xcenter
        self.ycenter = ycenter
        self.oversample = oversample
        self.oversize = oversize
        self.transform_method = transform_method
        self.zcalc_truncate = zcalc_truncate
        self.n_wholepix_z_min = n_wholepix_z_min
        self.gauss_extract_with_c = gauss_extract_with_c
        # Default always try to use the C++ gaussian fitter


class ObsLensingOptions:
    """
    Class to hold options for lensing the observed model from source to image plane
    """
    def __init__(self, **kwargs):
        self._MandatoryKeys = [
            (['lensing_datadir', 'datadir', None], 'mesh_dir'),             # datadir for the lensing model mesh.dat, fallback to datadir
            ('lensing_mesh',                       'mesh_file'),            # lensing model mesh.dat
            ('lensing_ra',                         'mesh_ra'),              # lensing model ref ra
            ('lensing_dec',                        'mesh_dec'),             # lensing model ref dec
            ('lensing_sra',                        'source_plane_cenra'),   # lensing source plane image center ra
            ('lensing_sdec',                       'source_plane_cendec'),  # lensing source plane image center dec
            ('lensing_ssizex',                     'source_plane_nx'),      # lensing source plane image size in x
            ('lensing_ssizey',                     'source_plane_ny'),      # lensing source plane image size in y
            ('lensing_spixsc',                     'source_plane_pixsc'),   # lensing source plane image pixel size in arcsec unit
            ('nspec',                              'source_plane_nchan'),   # lensing source plane channel number
            ('lensing_imra',                       'image_plane_cenra'),    # lensing image plane image center ra
            ('lensing_imdec',                      'image_plane_cendec'),   # lensing image plane image center dec
            (['nx_sky', 'fov_npix'],               'image_plane_sizex'),    # lensing image plane image size in x
            (['ny_sky', 'fov_npix'],               'image_plane_sizey'),    # lensing image plane image size in y
            ('pixscale',                           'image_plane_pixsc'),    # lensing image plane image pixel size in arcsec unit
        ]
        self._OptionalKeys = [
        ]
        self.valid = False
        if len(kwargs) > 0:
            self.load_keys(**kwargs)

    def load(self, extra = None, **kwargs):
        """Load mandatory keys from the input kwargs. 
        
        If there is no lensing key at all, we will not raise an exception. 
        Only if there are some lensing keys but are not complete, we will raise an exception.
        """
        if extra is None:
            extra = ''
        has_lensing_key = False
        for keytuple in self._MandatoryKeys:
            keyset, keyname = keytuple # key name in kwargs, and key name internally for lensing.py
            if not isinstance(keyset, (list, tuple)):
                keyset = [keyset]
            for key in keyset:
                if key is None:
                    setattr(self, keyname, None)
                    break
                elif key+extra in kwargs:
                    if kwargs[key+extra] is not None:
                        setattr(self, keyname, kwargs[key+extra])
                        if key.startswith('lensing_'):
                            has_lensing_key = True
                        break
                elif key in kwargs:
                    setattr(self, keyname, kwargs[key])
                    if key.startswith('lensing_'):
                        has_lensing_key = True
                    break
                #else: #missing mandatory key
        for keyname in self._OptionalKeys:
            if not hasattr(self, keyname):
                setattr(self, keyname, None)
        if has_lensing_key:
            self.validate(raise_exception = True)
    
    # @classmethod
    # def create_lensing_options(cls, **kwargs):
    #     this_object = cls(**kwargs)
    #     if not this_object.validate():
    #         return None
    #     return this_object
    
    def validate(self, raise_exception = False):
        missing_mendatory_keys = []
        for keytuple in self._MandatoryKeys:
            keyset, keyname = keytuple # key name in kwargs, and key name internally for lensing.py
            if not hasattr(self, keyname):
                if not isinstance(keyset, (list, tuple)):
                    keyset = [keyset]
                missing_mendatory_keys.append(' or '.join(keyset))
        if len(missing_mendatory_keys) > 0:
            if raise_exception:
                raise ValueError('Error! Missing keys to construct a ObsLensingOptions: {}'.format(', '.join(missing_mendatory_keys)))
            self.valid = False
        self.valid = True
        return self.valid
    
    def is_valid(self):
        return self.valid
    
    def get_lensing_kwargs(self, oversample = 1, oversize = 1):
        lensing_kwargs = {}
        for keytuple in self._MandatoryKeys:
            keyset, keyname = keytuple # key name in kwargs, and key name internally for lensing.py
            lensing_kwargs[keyname] = getattr(self, keyname)
            if keyname in ['image_plane_sizex', 'image_plane_sizey']:
                lensing_kwargs[keyname] = int(np.round(lensing_kwargs[keyname] * oversample * oversize))
            elif keyname in ['image_plane_pixsc']:
                lensing_kwargs[keyname] = lensing_kwargs[keyname] / oversample
        lensing_kwargs['cache_lensing_transformer'] = True
        lensing_kwargs['reuse_cached_lensing_transformer'] = True
        lensing_kwargs['verbose'] = False # (logger.level == logging.DEBUG)
        return lensing_kwargs



class ObsFitOptions:
    """
    Class to hold options for creating fitting an observation (or not)

    fit : bool
        whether to fit the Obs at all.
        Default: True


    fit_velocity, fit_dispersion, fit_flux : bool
        1D/2D specific. Whether to include the velocity/dispersion/flux
        profiles / maps in the fitting or not.
        Default: True for velocity/dispersion, False for flux.

    """
    def __init__(self, fit=True,
                 fit_velocity=True, fit_dispersion=True, fit_flux=False):

        self.fit = fit
        self.fit_velocity = fit_velocity
        self.fit_dispersion = fit_dispersion
        self.fit_flux = fit_flux
