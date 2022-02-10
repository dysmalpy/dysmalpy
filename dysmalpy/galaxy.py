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

# Third party imports
import numpy as np
import astropy.cosmology as apy_cosmo
import astropy.units as u
import scipy.interpolate as scp_interp
import dill as _pickle

# Local imports
# Package imports
from dysmalpy.models import ModelSet, calc_1dprofile, calc_1dprofile_circap_pv
from dysmalpy.data_classes import Data0D, Data1D, Data2D, Data3D
from dysmalpy.instrument import Instrument
from dysmalpy.utils import apply_smoothing_3D, rebin, gaus_fit_sp_opt_leastsq
from dysmalpy import aperture_classes
from dysmalpy.utils_io import write_model_obs_file
from dysmalpy import config
import datetime

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
    instrument : `~dysmalpy.instrument.Instrument` object
                 A dysmalpy instrument to use for simulating and/or fitting
                 data. This describes how the observables produced by
                 `model` are converted to observed space data.
    data : `~dysmalpy.data_classes.Data` object
           The observed data for the galaxy that model data can be fit to
    name : str, optional
           Name of the galaxy. Default is "galaxy."
    data1d : `~dysmalpy.data_classes.Data1D` object, optional
             Observed 1D data (i.e rotation curve) for the galaxy
    data2d : `~dysmalpy.data_classes.Data2D` object, optional
             Observed 2D data (i.e. velocity and dispersion maps) for the galaxy
    data3d : `~dysmalpy.data_classes.Data3D` object, optional
             Observed 3D data (i.e. cube) for the galaxy

    """

    def __init__(self, z=0, cosmo=_default_cosmo, model=None, instrument=None,
                 data=None, name='galaxy',
                 data1d=None, data2d=None, data3d=None):

        self._z = z
        self.name = name
        if model is None:
            self.model = ModelSet()
        else:
            self.model = model

        self._data = data

        self._data1d = data1d
        self._data2d = data2d
        self._data3d = data3d
        self._instrument = instrument

        self._cosmo = cosmo

        #self.dscale = self._cosmo.arcsec_per_kpc_proper(self._z).value
        self._dscale = self._cosmo.arcsec_per_kpc_proper(self._z).value
        self.model_data = None
        self.model_cube = None


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

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        if not np.any(isinstance(new_data, Data1D) | isinstance(new_data, Data2D) | \
            isinstance(new_data, Data3D) | isinstance(new_data, Data0D)):
            raise TypeError("Data must be one of the following instances: "
                            "   dysmalpy.Data0D, dysmalpy.Data1D, "
                            "   dysmalpy.Data2D, dysmalpy.Data3D")
        self._data = new_data
        self._setup_checks()

    @property
    def data1d(self):
        return self._data1d

    @data1d.setter
    def data1d(self, new_data1d):
        if not (isinstance(new_data1d, Data1D)):
            raise TypeError("Data1D must be an instance of dysmalpy.Data1D")
        self._data1d = new_data1d

    @property
    def data2d(self):
        return self._data2d

    @data2d.setter
    def data2d(self, new_data2d):
        if not (isinstance(new_data2d, Data2D)):
            raise TypeError("Data2D must be an instance of dysmalpy.Data2D")
        self._data2d = new_data2d

    @property
    def data3d(self):
        return self._data3d

    @data3d.setter
    def data3d(self, new_data3d):
        if not (isinstance(new_data3d, Data3D)):
            raise TypeError("Data3D must be an instance of dysmalpy.Data3D")
        self._data3d = new_data3d


    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, new_instrument):
        if not (isinstance(new_instrument, Instrument)) | (new_instrument is None):
            raise TypeError("Instrument must be a dysmalpy.Instrument instance.")
        self._instrument = new_instrument
        self._setup_checks()

    def _setup_checks(self):
        self._check_1d_datasize()
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
                elif spec_ctype == 'VOPT':
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


    # def create_model_data(self, **kwargs):
    def create_model_data(self, ndim_final=3, nx_sky=None, ny_sky=None,
                          rstep=None, spec_type='velocity', spec_step=10.,
                          spec_start=-1000., nspec=201, line_center=None,
                          spec_unit=(u.km/u.s), aper_centers=None,
                          slit_width=None, slit_pa=None, profile1d_type=None,
                          from_instrument=True, from_data=True,
                          oversample=1, oversize=1,
                          aperture_radius=None, pix_perp=None, pix_parallel=None,
                          pix_length=None,
                          skip_downsample=False, partial_aperture_weight=True,
                          xcenter=None, ycenter=None,
                          transform_method='direct',
                          zcalc_truncate=None,
                          n_wholepix_z_min=3,
                          **kwargs):
        r"""
        Function to simulate data for the galaxy

        The function will initially generate a data cube that will then be optionally
        reduced to 2D, 1D, or single spectrum data if specified. The generated cube
        can be accessed via `Galaxy.model_cube` and the generated final data products
        via `Galaxy.model_data`. Both of these attributes are `data_classes.Data` instances.

        Parameters
        ----------
        ndim_final : {3, 2, 1, 0}
            The dimensionality of the final data products.

            3 = data cube

            2 = velocity and dispersion maps

            1 = velocity and dispersion radial curves

            0 = single spectrum from integrating over the whole model cube

        nx_sky : int
            The number of pixels in the modelled data cube in the x direction

        ny_sky : int
            The number of pixels in the modelled data cube in the y direction

        rstep : float
                Pixel scale of the final model data cube in arcseconds/pixel

        spec_type : {`'velocity'`, `'wavelength'`}
                    Whether the spectral axis of the model data cube should be in
                    velocity units or wavelength units

        spec_step : float
                    The difference between neighboring spectral channels for the model
                    cube

        spec_start : float
                     The value of the first spectral channel for the model cube

        nspec : int
                The number of spectral channels for the model cube

        line_center : float
                      The observed frame wavelength that corresponds to zero velocity.
                      Only necessary if `spec_type` = 'wavelength'

        spec_unit : astropy.units.Unit
                    The units for the spectral axis of the model data cube

        aper_centers : array_like
                       Array of radii in arcseconds for where apertures are
                       centered to create 1D rotation curve.
                       Only necessary if `ndim_final` = 1.

        slit_width : float
                     The width in arcseconds of the pseudoslit used to measure
                     the 1D rotation curve. In practice, circular apertures
                     with radius = `slit_width`/2 are used to create the 1D rotation curve.

        slit_pa : float
                  The position angle of the pseudoslit in degrees. Convention is that negative
                  values of `aper_centers` correspond to East of the center of the cube

        profile1d_type : {`'circ_ap_cube'`, `'rect_ap_cube'`, `'square_ap_cube'`, `'circ_ap_pv'`, `'single_pix_pv'`}
            The extraction method to create the 1D rotation curve.

            "circ_ap_cube" = Extracts the 1D rotation curve through circular
                apertures placed directly on the model cube
            "rect_ap_cube" = Extracts the 1D rotation curve through rectangular
                apertures placed directly on the model cube
            "square_ap_cube" = Extracts the 1D rotation curve through square
                apertures placed directly on the model cube
            "circ_ap_pv" = The model cube is first collapsed to a PV diagram.
                Circular apertures are then placed on the PV diagram
                to construct the 1D rotation curve
            "single_pix_pv" = The model cube is first collapsed to a PV diagram.
                A 1D rotation curve is then extracted for each single
                pixel.

        from_instrument : bool
                          If True, use the settings of the attached Instrument to populate
                          the following parameters: `spec_type`, `spec_start`, `spec_step`,
                          `spec_unit`, `nspec`, `nx_sky`, `ny_sky`, and `rstep`.

        from_data : bool
                    If True, use the observed data to populate the following parameters:
                    `nx_sky` and `ny_sky` if data is 3D or 2D
                    `spec_type`, `spec_start`, `spec_step`, `spec_unit`, `nspec` if data is 3D
                    `slit_width`, `slit_pa`, `profile1D_type`, and `aper_centers` if data is 1D

        oversample : int
                     Oversampling factor for creating the model cube. If `oversample` > 1, then
                     the model cube will first be generated at `rstep`/`oversample` pixel scale.
                     It will then be downsampled to `rstep` pixel scale.

        oversize : int
                   Oversize factor for creating the model cube. If `oversize` > 1, then the model
                   cube will first be generated with `oversize`*`nx_sky` and `oversize`*`ny_sky`
                   number of pixels in the x and y direction respectively before any convolution
                   is performed. After any convolution with the Instrument, the model cube is then
                   cropped to match `nx_sky` and `ny_sky`.

        aperture_radius : float
                          Radius of circular apertures for `ndim_final` = 1 and
                          `profile1d_type` = 'circ_ap_cube' Only used if
                          `from_data` = False, otherwise the apertures attached to the data are
                          used to construct the 1D rotation curve.

        pix_perp : int or array
                   Number of pixels wide each rectangular aperture is for `ndim_final` = 1 and
                   `profile1d_type` = 'rect_ap_cube' in the direction perpendicular to the
                   pseudoslit

        pix_parallel : int or array
                       Number of pixels wide each rectangular aperture is for `ndim_final` = 1 and
                       `profile1d_type` = 'rect_ap_cube' in the direction parallel to the
                       pseudoslit

        pix_length : int or array
                     Number of pixels on each side of a square aperture for `ndim_final` = 1 and
                     `profile1d_type` = 'square_ap_cube'

        skip_downsample : bool
                          If True and `oversample` > 1 then do not downsample back to initial
                          `rstep`. Note the settings of the Instrument will then be changed to
                          match the new pixelscale and FOV size.

        partial_aperture_weight : bool
                                  If True, then use partial pixel weighting when integrating
                                  over an aperture. Only used when `ndim_final` = 1.

        xcenter : float
                  x pixel coordinate of the center of the cube if it should be different than
                  nx_sky/2

        ycenter : float
                  y pixel coordinate of the center of the cube if it should be different than
                  ny_sky/2

        transform_method: str
            Method for transforming from galaxy to sky coordinates.
            Options are:
                'direct' (calculate (xyz)sky before evaluating) or
                'rotate' (calculate in (xyz)gal, then rotate when creating the final cube).
            Default: 'direct'.

        zcalc_truncate: bool
                If True, the cube is only filled with flux to within
                +- 2 * scale length thickness above and below the galaxy midplane
                (minimum: n_wholepix_z_min [3] whole pixels; to speed up the calculation).
                Default: None (will be parsed to 0D/1D/2D/3D: True/True/False/False)

        n_wholepix_z_min: int
            Minimum number of whole pixels to include in the z direction when trunctating.
            Default: 3

        """
        if line_center is None:
            line_center = self.model.line_center



        # Parse default zcalc_truncate, if not already specified:
        if zcalc_truncate is None:
            if (ndim_final == 3) | (ndim_final == 2):
                # Default: no truncation for 2D/3D, because for the smaller spatial extent
                #          of a single spaxel (vs the aggregate in apertures for 1D/0D)
                #          leads to asymmetries when using truncation
                #          -- e.g., dispersion peak not coincident with xycenter
                zcalc_truncate = False
            elif (ndim_final == 1) | (ndim_final == 0):
                # Default: Truncate for 1D/0D, because combining multiple spaxels in apertures
                #          generally avoids the above effects (and gives faster computation times).
                zcalc_truncate = True

        # Pull parameters from the observed data if specified
        if from_data:
            ndim_final = self.data.ndim

            if ndim_final == 3:
                nx_sky = self.data.shape[2]
                ny_sky = self.data.shape[1]
                nspec = self.data.shape[0]
                spec_ctype = self.data.data.wcs.wcs.ctype[-1]
                if spec_ctype == 'WAVE':
                    spec_type = 'wavelength'
                elif spec_ctype == 'VOPT':
                    spec_type = 'velocity'
                spec_start = self.data.data.spectral_axis[0].value
                spec_unit = self.data.data.spectral_axis.unit
                spec_step = (self.data.data.spectral_axis[1].value -
                             self.data.data.spectral_axis[0].value)
                rstep = self.data.data.wcs.wcs.cdelt[0]*3600.

                try:
                    xcenter = self.data.xcenter
                except:
                    pass
                try:
                    ycenter = self.data.ycenter
                except:
                    pass

            elif ndim_final == 2:
                nx_sky = self.data.data['velocity'].shape[1]
                ny_sky = self.data.data['velocity'].shape[0]
                rstep = self.data.pixscale
                try:
                    xcenter = self.data.xcenter
                except:
                    pass
                try:
                    ycenter = self.data.ycenter
                except:
                    pass
                if from_instrument:
                    spec_type = self.instrument.spec_type
                    spec_start = self.instrument.spec_start.value
                    spec_step = self.instrument.spec_step.value
                    spec_unit = self.instrument.spec_start.unit
                    nspec = self.instrument.nspec

            elif ndim_final == 1:

                if from_instrument:
                    nx_sky = self.instrument.fov[0]
                    ny_sky = self.instrument.fov[1]
                    spec_type = self.instrument.spec_type
                    spec_start = self.instrument.spec_start.value
                    spec_step = self.instrument.spec_step.value
                    spec_unit = self.instrument.spec_start.unit
                    nspec = self.instrument.nspec
                    rstep = self.instrument.pixscale.value
                else:

                    maxr = 1.5*np.max(np.abs(self.data.rarr))
                    if rstep is None:
                        # Rough subsampling of ~3 as a "pixel size" = rstep
                        rstep = np.mean(self.data.rarr[1:] - self.data.rarr[0:-1])/3.
                    if nx_sky is None:
                        nx_sky = int(np.ceil(maxr/rstep))
                    if ny_sky is None:
                        ny_sky = int(np.ceil(maxr/rstep))

                    # --------------------------------------------------
                    # Check FOV and issue warning if too small:
                    maxr = np.max(np.abs(self.data.rarr))
                    if ((nx_sky < maxr/rstep) | (ny_sky < maxr/rstep)):
                        wmsg =  "dysmalpy.Galaxy:\n"
                        wmsg += "****************************************************************\n"
                        wmsg += "*** WARNING ***\n"
                        wmsg += "FOV (nx_sky,ny_sky)="
                        wmsg += "({},{}) is too small".format(nx_sky, ny_sky)
                        wmsg += " for max data extent ({} pix)\n".format(maxr/rstep)
                        wmsg += "****************************************************************\n"
                        logger.warning(wmsg)
                        raise ValueError(wmsg)
                    # --------------------------------------------------

                slit_width = self.data.slit_width
                slit_pa = self.data.slit_pa
                aper_centers = self.data.rarr

                try:
                    xcenter = self.data.xcenter
                except:
                    pass
                try:
                    ycenter = self.data.ycenter
                except:
                    pass


                if 'profile1d_type' in self.data.__dict__.keys():
                    if self.data.profile1d_type is not None:
                        profile1d_type = self.data.profile1d_type


            elif ndim_final == 0:

                if from_instrument:
                    nx_sky = self.instrument.fov[0]
                    ny_sky = self.instrument.fov[1]
                    spec_type = self.instrument.spec_type
                    spec_start = self.instrument.spec_start.value
                    spec_step = self.instrument.spec_step.value
                    spec_unit = self.instrument.spec_start.unit
                    nspec = self.instrument.nspec
                    rstep = self.instrument.pixscale.value

                else:

                    if (nx_sky is None) | (ny_sky is None) | \
                                (rstep is None):

                        raise ValueError("At minimum, nx_sky, ny_sky, and rstep must "
                                         "be set if from_instrument and/or from_data"
                                         " is False.")

                slit_width = self.data.slit_width
                slit_pa = self.data.slit_pa
                xarr = self.data.x


            if (ndim_final == 2) | (ndim_final==3):
                # Apply an artificial xycenter / xyshift of -0.5 if ndim = 2,3,
                #   and median smoothing with an EVEN npix is applied
                if self.data.smoothing_type is not None:
                    if self.data.smoothing_type.lower().strip() == 'median':
                        if (self.data.smoothing_npix % 2) == 0:
                            if xcenter is None:
                                xcenter = (nx_sky-1)/2.
                            if ycenter is None:
                                ycenter = (ny_sky-1)/2.
                            xcenter -= 0.5
                            ycenter -= 0.5


        # Pull parameters from the instrument
        elif from_instrument:

            nx_sky = self.instrument.fov[0]
            ny_sky = self.instrument.fov[1]
            spec_type = self.instrument.spec_type
            spec_start = self.instrument.spec_start.value
            spec_step = self.instrument.spec_step.value
            spec_unit = self.instrument.spec_start.unit
            nspec = self.instrument.nspec
            rstep = self.instrument.pixscale.value

            try:
                slit_width = self.instrument.slit_width
            except:
                pass

            if (ndim_final == 1) & (profile1d_type is None):
                raise ValueError("Must set profile1d_type if ndim_final=1, from_data=False!")

        else:

            if (nx_sky is None) | (ny_sky is None) | \
                        (rstep is None):

                raise ValueError("At minimum, nx_sky, ny_sky, and rstep must "
                                 "be set if from_instrument and/or from_data"
                                 " is False.")
            #
            if (ndim_final == 1) & (profile1d_type is None):
                raise ValueError("Must set profile1d_type if ndim_final=1, from_data=False!")

        # sim_cube, spec = self.model.simulate_cube(dscale=self.dscale,
        #                                          **sim_cube_kwargs.dict)


        # Apply lensing transformation if necessary
        this_lensing_transformer = None
        if _loaded_lensing:
            # Only check to get lensing transformer if the lensing modules were successfully loaded.
            if 'lensing_transformer' in kwargs:
                if kwargs['lensing_transformer'] is not None:
                    this_lensing_transformer = kwargs['lensing_transformer']['0']

            this_lensing_transformer = setup_lensing_transformer_from_params(\
                    params = kwargs,
                    source_plane_nchan = nspec,
                    image_plane_sizex = nx_sky * oversample * oversize,
                    image_plane_sizey = ny_sky * oversample * oversize,
                    image_plane_pixsc = rstep / oversample,
                    reuse_lensing_transformer = this_lensing_transformer,
                    cache_lensing_transformer = True,
                    reuse_cached_lensing_transformer = True,
                    verbose = (logger.level >= logging.DEBUG),
                )
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


        if this_lensing_transformer is not None:
            sim_cube, spec = self.model.simulate_cube(nx_sky=this_lensing_transformer.source_plane_nx,
                                                      ny_sky=this_lensing_transformer.source_plane_ny,
                                                      dscale=self.dscale,
                                                      rstep=this_lensing_transformer.source_plane_pixsc,
                                                      spec_type=spec_type,
                                                      spec_step=spec_step,
                                                      nspec=nspec,
                                                      spec_start=spec_start,
                                                      spec_unit=spec_unit,
                                                      oversample=1,
                                                      oversize=1,
                                                      xcenter=None,
                                                      ycenter=None,
                                                      transform_method=transform_method,
                                                      zcalc_truncate=zcalc_truncate,
                                                      n_wholepix_z_min=n_wholepix_z_min)

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
            # oversample oversize
            logger.debug('Applied lensing transformation '+str(datetime.datetime.now()))

        else:

            sim_cube, spec = self.model.simulate_cube(nx_sky=nx_sky,
                                                      ny_sky=ny_sky,
                                                      dscale=self.dscale,
                                                      rstep=rstep,
                                                      spec_type=spec_type,
                                                      spec_step=spec_step,
                                                      nspec=nspec,
                                                      spec_start=spec_start,
                                                      spec_unit=spec_unit,
                                                      oversample=oversample,
                                                      oversize=oversize,
                                                      xcenter=xcenter,
                                                      ycenter=ycenter,
                                                      transform_method=transform_method,
                                                      zcalc_truncate=zcalc_truncate,
                                                      n_wholepix_z_min=n_wholepix_z_min)


        # Correct for any oversampling
        if (oversample > 1) & (not skip_downsample):
            sim_cube_nooversamp = rebin(sim_cube, (ny_sky*oversize,
                                nx_sky*oversize))
        else:
            sim_cube_nooversamp = sim_cube

        if skip_downsample:
            rstep /= (1.*oversample)
            nx_sky *= oversample
            ny_sky *= oversample
            # Fix instrument:
            self.instrument.pixscale = rstep * u.arcsec
            self.instrument.fov = [nx_sky, ny_sky]
            self.instrument.set_beam_kernel()


        # Apply beam smearing and/or instrumental spreading
        if self.instrument is not None:
            sim_cube_obs = self.instrument.convolve(cube=sim_cube_nooversamp,
                                                    spec_center=line_center)
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

        self.model_cube = Data3D(cube=sim_cube_final, pixscale=rstep,
                                 spec_type=spec_type,
                                 spec_arr=spec,
                                 spec_unit=spec_unit)

        if ndim_final == 3:
            if from_data:
                if self.data.smoothing_type is not None:
                    self.model_cube.data = apply_smoothing_3D(self.model_cube.data,
                            smoothing_type=self.data.smoothing_type,
                            smoothing_npix=self.data.smoothing_npix)

                sim_cube_final_scale = self.model_cube.data._data.copy()
                if self.data.flux_map is None:
                    #mask_flat = np.sum(self.data.mask, axis=0)
                    num = np.sum(self.data.mask*(self.data.data.unmasked_data[:].value*
                                     self.model_cube.data/(self.data.error.unmasked_data[:].value**2)), axis=0)
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
                sim_cube_final_scale = self.model_cube.data._data.copy()
                mask_cube = None

            self.model_data = Data3D(cube=sim_cube_final_scale, pixscale=rstep,
                                     mask_cube=mask_cube,
                                     spec_type=spec_type,
                                     spec_arr=spec,
                                     spec_unit=spec_unit)

        elif ndim_final == 2:

            if from_data:
                if self.data.smoothing_type is not None:
                    self.model_cube.data = apply_smoothing_3D(self.model_cube.data,
                                smoothing_type=self.data.smoothing_type,
                                smoothing_npix=self.data.smoothing_npix)

                # How data was extracted:
                if self.data.moment:
                    extrac_type = 'moment'
                else:
                    extrac_type = 'gauss'
            elif from_instrument:
                if 'moment' in self.instrument.__dict__.keys():
                    if self.instrument.moment:
                        extrac_type = 'moment'
                    else:
                        extrac_type = 'gauss'
                else:
                    extrac_type = 'moment'
            else:
                extrac_type = 'moment'

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
                    # <DZLIU><20210805> ++++++++++
                    my_least_chi_squares_1d_fitter = None
                    if ('gauss_extract_with_c' in kwargs) & (_loaded_LeastChiSquares1D):
                        if kwargs['gauss_extract_with_c'] is not None and \
                           kwargs['gauss_extract_with_c'] is not False:
                            # we will use the C++ LeastChiSquares1D to run the 1d spectral fitting
                            # but note that if a spectrum has data all too close to zero, it will fail.
                            # try to prevent this by excluding too low data
                            if from_data:
                                this_fitting_mask = copy.copy(self.data.mask)
                            else:
                                this_fitting_mask = 'auto'
                            if logger.level > logging.DEBUG:
                                this_fitting_verbose = True
                            else:
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
                                best_fit = gaus_fit_sp_opt_leastsq(self.model_cube.data.spectral_axis.to(u.km/u.s).value,
                                                    self.model_cube.data.unmasked_data[:,i,j].value,
                                                    mom0[i,j], mom1[i,j], mom2[i,j])
                                flux[i,j] = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
                                vel[i,j] = best_fit[1]
                                disp[i,j] = best_fit[2]
                                if i==mom0.shape[0]-1 and j==mom0.shape[1]-1:
                                    logger.debug('gaus_fit_sp_opt_leastsq '+str(mom0.shape[0])+'x'+str(mom0.shape[1])+' '+str(datetime.datetime.now())) #<DZLIU><DEBUG>#
                    # <DZLIU><20210805> ----------

            elif spec_type == "wavelength":

                cube_with_vel = self.model_cube.data.with_spectral_unit(u.km/u.s,
                    velocity_convention='optical',
                    rest_value=line_center)

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


            if from_data:
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

            self.model_data = Data2D(pixscale=rstep, velocity=vel,
                                     vel_disp=disp, flux=flux, mask=mask)

        elif ndim_final == 1:

            if spec_type == 'wavelength':

                cube_with_vel = self.model_cube.data.with_spectral_unit(
                    u.km / u.s, velocity_convention='optical',
                    rest_value=line_center)

                cube_data = cube_with_vel.unmasked_data[:]
                vel_arr = cube_with_vel.spectral_axis.to(u.km/u.s).value

            elif spec_type == 'velocity':

                cube_data = sim_cube_obs
                vel_arr = spec

            else:
                raise ValueError("spec_type can only be 'velocity' or "
                                 "'wavelength.'")

            if profile1d_type == 'circ_ap_pv':
                r1d, flux1d, vel1d, disp1d = calc_1dprofile_circap_pv(cube_data,
                                slit_width,slit_pa-180.,
                                rstep, vel_arr)
                vinterp = scp_interp.interp1d(r1d, vel1d,
                                              fill_value='extrapolate')
                disp_interp = scp_interp.interp1d(r1d, disp1d,
                                                  fill_value='extrapolate')
                vel1d = vinterp(aper_centers)
                disp1d = disp_interp(aper_centers)
                flux_interp = scp_interp.interp1d(r1d, flux1d,
                                                  fill_value='extrapolate')
                flux1d = flux_interp(aper_centers)
                aper_model = None

            elif profile1d_type == 'single_pix_pv':
                r1d, flux1d, vel1d, disp1d = calc_1dprofile(cube_data, slit_width,
                            slit_pa-180., rstep, vel_arr)
                vinterp = scp_interp.interp1d(r1d, vel1d,
                                              fill_value='extrapolate')
                disp_interp = scp_interp.interp1d(r1d, disp1d,
                                                  fill_value='extrapolate')
                vel1d = vinterp(aper_centers)
                disp1d = disp_interp(aper_centers)

                flux_interp = scp_interp.interp1d(r1d, flux1d,
                                                  fill_value='extrapolate')
                flux1d = flux_interp(aper_centers)

                aper_model = None
            else:

                if from_data:
                    if (self.data.aper_center_pix_shift is not None):
                        try:
                            center_pixel = (self.data.xcenter + self.data.aper_center_pix_shift[0],
                                            self.data.ycenter + self.data.aper_center_pix_shift[1])
                        except:
                            center_pixel = (np.int(nx_sky / 2) + self.data.aper_center_pix_shift[0],
                                            np.int(ny_sky / 2) + self.data.aper_center_pix_shift[1])
                    else:
                        try:
                            # Catch case where center_pixel is (None, None)
                            if (self.data.xcenter is not None) & (self.data.ycenter is not None):
                                center_pixel = (self.data.xcenter, self.data.ycenter)
                            else:
                                center_pixel = None
                        except:
                            center_pixel = None
                else:
                    center_pixel = None



                #----------------------------------------------------------
                #try:
                if from_data:
                    aper_centers, flux1d, vel1d, disp1d = self.data.apertures.extract_1d_kinematics(spec_arr=vel_arr,
                            cube=cube_data, center_pixel = center_pixel, pixscale=rstep)
                    aper_model = None

                # except:
                #     raise TypeError('Unknown method for measuring the 1D profiles.')

                #----------------------------------------------------------
                else:

                    aper_model = aperture_classes.setup_aperture_types(gal=self,
                                profile1d_type=profile1d_type,
                                slit_width = slit_width,
                                aper_centers=aper_centers,
                                slit_pa=slit_pa,
                                aperture_radius=aperture_radius,
                                pix_perp=pix_perp,
                                pix_parallel=pix_parallel,
                                pix_length=pix_length,
                                partial_weight=partial_aperture_weight,
                                from_data=False)


                    aper_centers, flux1d, vel1d, disp1d = aper_model.extract_1d_kinematics(spec_arr=vel_arr,
                            cube=cube_data, center_pixel = center_pixel,
                            pixscale=rstep)


            if from_data:
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
                                     vel_disp=disp1d, flux=flux1d, mask=mask1d,
                                     slit_width=slit_width, slit_pa=slit_pa)
            self.model_data.apertures = aper_model
            self.model_data.profile1d_type = profile1d_type

        elif ndim_final == 0:

            if self.data.integrate_cube:

                # Integrate over the spatial dimensions of the cube
                flux = np.nansum(np.nansum(self.model_cube.data.unmasked_data[:], axis=2), axis=1)

                # Normalize to the maximum of the spectrum
                flux /= np.nanmax(flux)
                flux = flux.value

            else:

                # Place slit down on cube
                raise NotImplementedError('Using slits to create spectrum not implemented yet!')

            self.model_data = Data0D(x=spec, flux=flux, slit_pa=self.data.slit_pa,
                                     slit_width=self.data.slit_width,
                                     integrate_cube=self.data.integrate_cube)

        ####
        # Reset instrument to orig value
        if skip_downsample:
            rstep *= oversample
            nx_sky /= (1.*oversample)
            ny_sky /= (1.*oversample)
            # Fix instrument:
            self.instrument.pixscale = rstep * u.arcsec
            self.instrument.fov = [nx_sky, ny_sky]
            self.instrument.set_beam_kernel()

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
