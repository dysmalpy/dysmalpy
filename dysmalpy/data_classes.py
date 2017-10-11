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
import astropy.units as u
from astropy.wcs import WCS
from spectral_cube import SpectralCube, BooleanArrayMask

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


__all__ = ["Data", "Data1D", "Data2D", "Data3D"]

# TODO: Pull out mask as a separate attribute for data

# Base Class for a data container
class Data:

    def __init__(self, data=None, error=None, ndim=None, mask=None,
                 shape=None):

        self.data = data
        self.error = error
        self.ndim = ndim
        self.shape = shape
        self.mask = mask


class Data1D(Data):

    def __init__(self, r, velocity, vel_err=None, vel_disp=None,
                 vel_disp_err=None, mask=None, slit_width=None,
                 slit_pa=None, estimate_err=False, error_frac=0.2):

        if r.shape != velocity.shape:
            raise ValueError("r and velocity are not the same size.")

        if mask is not None:
            if mask.shape != velocity.shape:
                raise ValueError("mask and velocity are not the same size.")

        data = {'velocity': velocity}

        if vel_disp is None:

            data['dispersion'] = None

        else:

            if vel_disp.shape != velocity.shape:
                raise ValueError("vel_disp and velocity are not the same size.")

            data['dispersion'] = np.ma.masked_array(vel_disp, mask=mask)

        # Override any array given to vel_err if estimate_err is True
        if estimate_err:
            vel_err = error_frac*velocity

        if vel_err is None:

            error = {'velocity': None}

        else:

            if vel_err.shape != velocity.shape:
                raise ValueError("vel_err and velocity are not the "
                                 "same size.")

            error = {'velocity': vel_err}

        if (vel_disp is not None) and (vel_disp_err is None):

            if estimate_err:

                vel_disp_err = error_frac * vel_disp
                error['dispersion'] = vel_disp_err
            else:

                error['dispersion'] = None

        elif (vel_disp is not None) and (vel_disp_err is not None):

            if vel_disp_err.shape != velocity.shape:

                raise ValueError("vel_disp_err and velocity are not the"
                                 " same size.")

            error['dispersion'] = vel_disp_err
        else:

            error['dispersion'] = None

        shape = velocity.shape
        self.slit_width = slit_width
        self.slit_pa = slit_pa
        self.rarr = r
        super(Data1D, self).__init__(data=data, error=error, ndim=1,
                                     shape=shape, mask=mask)


class Data2D(Data):

    def __init__(self, pixscale, velocity, vel_err=None, vel_disp=None,
                 vel_disp_err=None, mask=None, estimate_err=False,
                 error_frac=0.2, ra=None, dec=None, ref_pixel=None):

        if mask is not None:
            if mask.shape != velocity.shape:
                raise ValueError("mask and velocity are not the same size.")

        data = {'velocity': velocity}

        if vel_disp is None:

            data['dispersion'] = None

        else:

            if vel_disp.shape != velocity.shape:
                raise ValueError("vel_disp and velocity are not the same size.")

            data['dispersion'] = vel_disp

        # Override any array given to vel_err if estimate_err is True
        if estimate_err:
            vel_err = error_frac*velocity

        if vel_err is None:

            error = {'velocity': None}

        else:

            if vel_err.shape != velocity.shape:
                raise ValueError("vel_err and velocity are not the "
                                 "same size.")

            error = {'velocity': vel_err}

        if (vel_disp is not None) and (vel_disp_err is None):

            if estimate_err:

                vel_disp_err = error_frac * vel_disp
                error['dispersion'] = vel_disp_err
            else:

                error['dispersion'] = None

        elif (vel_disp is not None) and (vel_disp_err is not None):

            if vel_disp_err.shape != velocity.shape:

                raise ValueError("vel_disp_err and velocity are not the"
                                 " same size.")

            error['dispersion'] = vel_disp_err
        else:

            error['dispersion'] = None

        shape = velocity.shape
        self.pixscale = pixscale
        self.ra = ra
        self.dec = dec
        self.ref_pixel = ref_pixel
        super(Data2D, self).__init__(data=data, error=error, ndim=2,
                                     shape=shape, mask=mask)


class Data3D(Data):

    def __init__(self, cube, pixscale, spec_type, spec_arr,
                 err_cube=None, mask_sky=None, mask_spec=None,
                 estimate_err=False, error_frac=0.2, ra=None, dec=None,
                 ref_pixel=None, spec_unit=None):

        if mask_sky is not None:
            if mask_sky.shape != cube.shape[1:]:
                raise ValueError("mask_sky and last two dimensions of cube do "
                                 "not match.")
        else:

            mask_sky = np.ones((cube_shape[1:]))


        if mask_spec is not None:
            if mask_spec.shape != spec_arr.shape:
                raise ValueError("The length of mask_spec and spec_arr do not "
                                 "match.")
        else:
            mask_spec = np.ones(len(spec_arr))

        mask = _create_cube_mask(mask_sky=mask_sky, mask_spec=mask_spec)

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

        if (spec_type != 'velocity') & (spec_type != 'wavelength'):
            raise ValueError("spec_type must be one of 'velocity' or "
                             "'wavelength.'")

        if (spec_type == 'velocity') and (spec_unit is None):
            spec_unit = u.km/u.s
        elif (spec_type == 'wavelength') and (spec_unit is None):
            spec_unit = u.Angstrom

        if spec_type == 'velocity':
            spec_ctype = 'VOPT'
        else:
            spec_ctype = 'WAVE'

        if (ra is None) | (dec is None):
            xref = 0
            yref = 0
            ra = 0.
            dec = 0.
        elif ref_pixel is not None:
            xref = ref_pixel[1]
            yref = ref_pixel[0]

        else:
            xref = np.int(cube.shape[2] / 2.)
            yref = np.int(cube.shape[1] / 2.)

        # Create a simple header for the cube
        w = WCS(naxis=3)
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN', spec_ctype]
        w.wcs.cdelt = [pixscale / 3600., pixscale / 3600., spec_step]
        w.wcs.crpix = [xref, yref, 1]
        w.wcs.cunit = ['deg', 'deg', spec_unit.to_string()]
        w.wcs.crval = [ra, dec, spec_arr[0]]
        data = SpectralCube(data=cube, wcs=w)
        if err_cube is not None:
            error = SpectralCube(data=err_cube, wcs=w)
        else:
            error = None
        shape = cube.shape

        super(Data3D, self).__init__(data=data, error=error, ndim=3,
                                     shape=shape, mask=mask)

# TODO: Dimension order for Data2D and Data3D?

def _create_cube_mask(mask_sky=None, mask_spec=None):
    """Create a 3D mask from a 2D sky mask and 1D spectral mask"""

    mask_spec = mask_spec.reshape((len(mask_spec, 1, 1)))
    mask_spec_3d = np.tile(mask_spec, (1, mask_sky.shape[0], mask_sky.shape[1]))
    mask_sky_3d = np.tile(mask_sky, (mask_spec.shape[0], 1, 1))

    mask_total_3d = mask_spec_3d*mask_sky_3d

    return mask_total_3d
