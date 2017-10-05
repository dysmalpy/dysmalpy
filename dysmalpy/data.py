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
                 error_frac=0.2):


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
        self.rarr = r
        super(Data1D, self).__init__(data=data, error=error, ndim=1,
                                     shape=shape)

