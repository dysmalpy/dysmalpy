# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# File containing all of the available classes to hold observed data for a
# galaxy.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard imports
import logging

# Third party imports
import numpy as np
from spectral_cube import SpectralCube, BooleanArrayMask


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')

# Base Class for a data container
class Data:

    def __init__(self, data=None, error=None, ndim=None, shape=None):

        self.data = data
        self.error = error
        self.ndim = ndim
        self.shape = shape



class Data1D(Data):

    def __init__(self, r, velocity, vel_err=None, vel_disp=None,
                 vel_disp_err=None, mask=None, aper_size=None,
                 aper_pa=None, error_frac=0.2):


        if r.shape != velocity.shape:
            raise ValueError("r and velocity are not the same size.")

        if mask is not None:
            if mask.shape != velocity.shape:
                raise ValueError("mask and velocity are not the same size.")

        data = {'velocity': np.ma.masked_array(velocity, mask=mask)}
        if vel_disp is None:
            data['dispersion'] = None
        else:
            if vel_disp.shape != velocity.shape:
                raise ValueError("vel_disp and velocity are not the same size.")
            data['dispersion'] = np.ma.masked_array(vel_disp, mask=mask)

        # Default to error_frac of the data for errors if an
        # error array isn't given.
        if vel_err is None:
            logger.info(r"No error array found for velocity, "
                        r"using {:.0f}% of the data.".format(error_frac*100))
            vel_err = error_frac*velocity
        else:
            if vel_err.shape != velocity.shape:
                raise ValueError("vel_err and velocity are not the same size.")


        error = {'velocity': np.ma.masked_array(vel_err, mask=mask)}

        if (vel_disp is not None) and (vel_disp_err is None):
            logger.info(r"No error array found for dispersion, "
                        r"using {:.0f}\% of the data.".format(error_frac * 100))
            vel_disp_err = error_frac*vel_disp
            error['dispersion'] = np.ma.masked_array(vel_disp_err, mask=mask)
        elif (vel_disp is not None) and (vel_disp_err is not None):
            if vel_disp_err.shape != velocity.shape:
                raise ValueError("vel_disp_err and velocity are not the"
                                 " same size.")
            else:
                error['dispersion'] = np.ma.masked_array(vel_disp_err,
                                                         mask=mask)
        else:
            error['dispersion'] = None

        shape = velocity.shape
        self.aper_size = aper_size
        self.aper_pa = aper_pa
        self.rarr = r
        super(Data1D, self).__init__(data=data, error=error, ndim=1,
                                     shape=shape)

class Data2D(Data):

    def __init__(self, pixscale, velocity, vel_err=None, vel_disp=None,
                 vel_disp_err=None, mask=None, error_frac=0.2):


        if mask is not None:
            if mask.shape != velocity.shape:
                raise ValueError("mask and velocity are not the same size.")

        data = {'velocity': np.ma.masked_array(velocity, mask=mask)}
        if vel_disp is None:
            data['dispersion'] = None
        else:
            if vel_disp.shape != velocity.shape:
                raise ValueError("vel_disp and velocity are not the same size.")
            data['dispersion'] = np.ma.masked_array(vel_disp, mask=mask)

        # Default to error_frac of the data for errors if an
        # error array isn't given.
        if vel_err is None:
            logger.info(r"No error array found for velocity, "
                        r"using {:.0f}% of the data.".format(error_frac * 100))
            vel_err = error_frac * velocity
        else:
            if vel_err.shape != velocity.shape:
                raise ValueError("vel_err and velocity are not the same size.")

        error = {'velocity': np.ma.masked_array(vel_err, mask=mask)}

        if (vel_disp is not None) and (vel_disp_err is None):
            logger.info(r"No error array found for dispersion, "
                        r"using {:.0f}\% of the data.".format(error_frac * 100))
            vel_disp_err = error_frac * vel_disp
            error['dispersion'] = np.ma.masked_array(vel_disp_err, mask=mask)
        elif (vel_disp is not None) and (vel_disp_err is not None):
            if vel_disp_err.shape != velocity.shape:
                raise ValueError("vel_disp_err and velocity are not the"
                                 " same size.")
            else:
                error['dispersion'] = np.ma.masked_array(vel_disp_err,
                                                         mask=mask)
        else:
            error['dispersion'] = None

        shape = velocity.shape
        self.pixscale = pixscale
        super(Data2D, self).__init__(data=data, error=error, ndim=2,
                                     shape=shape)


class Data3D(Data):

    def __init__(self, cube, pixscale, spec_type, spec_arr,
                 err_cube=None, mask=None, estimate_err=False, error_frac=0.2):

        if mask is not None:
            if mask.shape != cube.shape:
                raise ValueError("mask and velocity are not the same size.")

        if err_cube is not None:
            if err_cube.shape != cube.shape:
                raise ValueError("err_cube and cube are not the same size.")

        if len(spec_arr) != cube.shape[0]:
            raise ValueError("First dimension of cube not the same size as "
                             "spec_arr.")

        # Estimate the error cube if requested by taking error_frac*cube
        if estimate_err:
            err_cube = error_frac*cube

        # Get the spectral step by just taking the difference between the
        # first and second elements. Assumes uniform spacing.
        spec_step = spec_arr[1] - spec_arr[0]

        if (spec_type != 'velocity') | (spec_type != 'wavelength'):
            raise ValueError("spec_type must be one of 'velocity' or "
                             "'wavelength.'")

        data = np.ma.masked_array(cube, mask=mask)
        if err_cube is not None:
            error = np.ma.masked_array(err_cube, mask=mask)
        else:
            error = None

        shape = cube.shape
        self.pixscale = pixscale
        self.spec_type = spec_type
        self.spec_arr = spec_arr
        self.spec_step = spec_step
        super(Data3D, self).__init__(data=data, error=error, ndim=3,
                                     shape=shape)

