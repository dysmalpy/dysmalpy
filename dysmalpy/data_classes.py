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


# Base Class for a data container
class Data(object):

    def __init__(self, data=None, error=None, ndim=None, mask=None,
                 shape=None, 
                 filename_velocity=None, 
                 filename_dispersion=None, 
                 smoothing_type=None, 
                 smoothing_npix=1):
                 
        self.data = data
        self.error = error
        self.ndim = ndim
        self.shape = shape
        self.mask = np.array(mask, dtype=np.bool)
        
        self.smoothing_type = smoothing_type
        self.smoothing_npix = smoothing_npix
        
        self.filename_velocity = filename_velocity
        self.filename_dispersion = filename_dispersion
        


class Data1D(Data):
    """
    Convention:
        slit_pa is angle of slit to left side of major axis (eg, neg r is E)
    """

    def __init__(self, r, velocity, vel_err=None, vel_disp=None,
                 vel_disp_err=None, mask=None, slit_width=None,
                 slit_pa=None, aper_center_pix_shift=None, 
                 estimate_err=False, error_frac=0.2,
                 inst_corr=False, 
                 filename_velocity=None, 
                 filename_dispersion=None):
        # Default assume 1D dispersion is **NOT** instrument corrected
        
        if r.shape != velocity.shape:
            raise ValueError("r and velocity are not the same size.")

        if mask is not None:
            if mask.shape != velocity.shape:
                raise ValueError("mask and velocity are not the same size.")

        else:
            mask = np.ones(len(velocity))

        data = {'velocity': velocity}
        
        # Information about *dispersion* instrument correction
        data['inst_corr'] = inst_corr

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
            
        ############
        self.aper_center_pix_shift = aper_center_pix_shift

        shape = velocity.shape
        self.slit_width = slit_width
        self.slit_pa = slit_pa
        self.rarr = r
        
        self.apertures = None
        
        super(Data1D, self).__init__(data=data, error=error, ndim=1,
                                     shape=shape, mask=mask,
                                     filename_velocity=filename_velocity, 
                                     filename_dispersion=filename_dispersion)


class Data2D(Data):

    def __init__(self, pixscale, velocity, vel_err=None, vel_disp=None,
                 vel_disp_err=None, mask=None, estimate_err=False,
                 error_frac=0.2, ra=None, dec=None, ref_pixel=None,
                 inst_corr=False, 
                 filename_velocity=None, 
                 filename_dispersion=None, 
                 smoothing_type=None, 
                 smoothing_npix=1):

        if mask is not None:
            if mask.shape != velocity.shape:
                raise ValueError("mask and velocity are not the same size.")

        else:
            mask = np.ones(velocity.shape)

        data = {'velocity': velocity}
        
        # Information about *dispersion* instrument correction
        data['inst_corr'] = inst_corr

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
        # Catch a case where Astropy unit might be passed:
        if (type(pixscale) == u.quantity.Quantity):
            self.pixscale = pixscale.to(u.arcsec).value
        else:
            # If constant, should be implicitly arcsec:
            self.pixscale = pixscale
        self.ra = ra
        self.dec = dec
        self.ref_pixel = ref_pixel
        super(Data2D, self).__init__(data=data, error=error, ndim=2,
                                     shape=shape, mask=mask,
                                     filename_velocity=filename_velocity, 
                                     filename_dispersion=filename_dispersion,
                                     smoothing_type=smoothing_type, 
                                     smoothing_npix=smoothing_npix)


class Data3D(Data):

    def __init__(self, cube, pixscale, spec_type, spec_arr,
                 err_cube=None, mask_sky=None, mask_spec=None,
                 estimate_err=False, error_frac=0.2, ra=None, dec=None,
                 ref_pixel=None, spec_unit=None, flux_map=None):

        if mask_sky is not None:
            if mask_sky.shape != cube.shape[1:]:
                raise ValueError("mask_sky and last two dimensions of cube do "
                                 "not match.")
        else:

            mask_sky = np.ones((cube.shape[1:]))

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
            xref = 1
            yref = 1
            ra = -pixscale / 3600. * cube.shape[2]/2
            dec = -pixscale / 3600. * cube.shape[2]/2
            ctype1 = 'RA---CAR'
            ctype2 = 'DEC--CAR'

        elif ref_pixel is not None:
            xref = ref_pixel[1]
            yref = ref_pixel[0]
            ctype1 = 'RA---TAN'
            ctype2 = 'DEC--TAN'

        else:
            xref = np.int(cube.shape[2] / 2.)
            yref = np.int(cube.shape[1] / 2.)
            ctype1 = 'RA---TAN'
            ctype2 = 'DEC--TAN'

        # Create a simple header for the cube
        w = WCS(naxis=3)

        w.wcs.ctype = [ctype1, ctype2, spec_ctype]
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
        self.flux_map = flux_map

        super(Data3D, self).__init__(data=data, error=error, ndim=3,
                                     shape=shape, mask=mask)


class Data0D(Data):
    """
    Data class for storing a single spectrum
    """

    def __init__(self, x, flux, flux_err=None,
                 mask=None, integrate_cube=True, slit_width=None,
                 slit_pa=None, estimate_err=False, error_frac=0.2):

        if x.shape != flux.shape:
            raise ValueError("r and velocity are not the same size.")

        if mask is not None:
            if mask.shape != x.shape:
                raise ValueError("mask and velocity are not the same size.")

        else:
            mask = np.ones(len(x))

        data = flux

        # Override any array given to vel_err if estimate_err is True
        if estimate_err:
            flux_err = error_frac * flux

        if flux_err is None:

            error = None

        else:

            if flux_err.shape != x.shape:
                raise ValueError("vel_err and velocity are not the "
                                 "same size.")

            error = flux_err

        ############
        self.integrate_cube = integrate_cube
        self.slit_width = slit_width
        self.slit_pa = slit_pa
        self.x = x
        shape = x.shape

        super(Data0D, self).__init__(data=data, error=error, ndim=0,
                                     shape=shape, mask=mask)



def _create_cube_mask(mask_sky=None, mask_spec=None):
    """Create a 3D mask from a 2D sky mask and 1D spectral mask"""

    mask_spec = mask_spec.reshape((len(mask_spec), 1, 1))
    mask_spec_3d = np.tile(mask_spec, (1, mask_sky.shape[0], mask_sky.shape[1]))
    mask_sky_3d = np.tile(mask_sky, (mask_spec.shape[0], 1, 1))

    mask_total_3d = mask_spec_3d*mask_sky_3d

    return mask_total_3d
