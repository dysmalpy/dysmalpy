# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# File containing Observation and ObservationSet classes which define the individual and set
# of observations of a galaxy.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard imports
import logging
import copy
from collections import OrderedDict

# Third party imports

# Package imports
from dysmalpy.instrument import Instrument
from dysmalpy.data_classes import Data

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


__all__ = ["ObservationSet", "Observation"]


class ObservationSet:
    "Class holding all of the individual observations of a galaxy"

    def __init__(self, obs_list=None):

        self.observations = OrderedDict()

        if obs_list is not None:

            for obs in obs_list:

                self.add_observation(obs)

    def add_observation(self, obs):
        """
        Add an observation to the ObservationSet.
        """

        obs_name = obs.name
        self.observations[obs_name] = obs

    def get_observation(self, obs_name):
        """
        Retrieve an observation from the set
        """

        try:

            return self.observations[obs_name]

        except KeyError:

            raise KeyError('{} not in ObservationSet'.format(obs_name))


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
        self.obs_options = ObsOptions()

        if instrument is not None:
            self.add_instrument(instrument)

        if data is not None:
            self.add_data(data)

    def add_instrument(self, instrument):
        self.instrument = instrument

    def add_data(self, data):
        self.data = data

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


class ObsOptions:

    def __init__(self, xcenter=None, ycenter=None, oversample=None, oversize=None,
                 transform_method=None, zcalc_truncate=None, n_wholepix_z_min=None
                 ):

        self.xcenter = xcenter
        self.ycenter = ycenter
        self.oversample = oversample
        self.oversize = oversize
        self.transform_method = transform_method
        self.zcalc_truncate = zcalc_truncate
        self.n_wholepix_z_min = n_wholepix_z_min
