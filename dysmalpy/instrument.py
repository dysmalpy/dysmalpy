# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# File containing classes for defining different instruments
# to "observe" a model galaxy.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np
import astropy.convolution as apy_conv
from scipy.signal import fftconvolve
import astropy.units as u
import astropy.constants as c
from radio_beam import Beam

__all__ = ["Instrument", "Beam", "LSF", "DoubleBeam"]

# CONSTANTS
sig_to_fwhm = 2.*np.sqrt(2.*np.log(2.))

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


class Instrument:
    """Base Class to define an instrument to observe a model galaxy with."""

    def __init__(self, beam=None, beam_type=None, lsf=None, pixscale=None,
                 spec_type='velocity', spec_start=-1000*u.km/u.s,
                 spec_step=10*u.km/u.s, nspec=201,
                 fov=None, name='Instrument'):

        self.name = name
        self.pixscale = pixscale
        
        # Case of two beams: analytic and empirical: if beam_type==None, assume analytic
        self.beam = beam
        self.beam_type = beam_type
            
        self._beam_kernel = None
        self.lsf = lsf
        self._lsf_kernel = None
        self.fov = fov
        self.spec_type = spec_type
        self.spec_start = spec_start
        self.spec_step = spec_step
        self.nspec = nspec

    def convolve(self, cube, spec_center=None):
        """
        Method to perform the convolutions in both the spatial and
        spectral space. Cube is assumed to be 3D with the first dimension
        corresponding to the spectral dimension.
        First convolve with the instrument PSF, then with the LSF.
        """

        if self.beam is None and self.lsf is None:
            # Nothing to do if a PSF and LSF aren't set for the instrument
            logger.warning("No convolution being performed since PSF and LSF "
                           "haven't been set!")
            return cube

        elif self.lsf is None:

            cube = self.convolve_with_beam(cube)

        elif self.beam is None:

            cube = self.convolve_with_lsf(cube, spec_center=spec_center)

        else:

            cube_conv_beam = self.convolve_with_beam(cube)
            cube = self.convolve_with_lsf(cube_conv_beam,
                                          spec_center=spec_center)

        return cube

    def convolve_with_lsf(self, cube, spec_center=None):
        """Convolve cube with the LSF"""

        if self._lsf_kernel is None:
            self.set_lsf_kernel(spec_center=spec_center)
            
        cube = fftconvolve(cube, self._lsf_kernel, mode='same')
        
        # shp = self._lsf_kernel.shape
        # if shp[0] % 2 == 1:
        #     padf = (shp[0]-1)/2
        # else:
        #     padf = shp[0]/2
        # cube = fftconvolve(cube, self._lsf_kernel, mode='full')
        # cube = cube[padf:-padf, :, :]
        
        return cube

    def convolve_with_beam(self, cube):

        if self.pixscale is None:
            raise ValueError("Pixelscale for this instrument has not "
                             "been set yet. Can't convolve with beam.")

        if self._beam_kernel is None:
            self.set_beam_kernel()
            
        cube = fftconvolve(cube, self._beam_kernel, mode='same')
        
        # shp = self._beam_kernel.shape
        # if shp[1] % 2 == 1:
        #     padf = (shp[1]-1)/2
        # else:
        #     padf = shp[1]/2
        # cube = fftconvolve(cube, self._beam_kernel, mode='full')
        # cube = cube[:, padf:-padf, padf:-padf]
        
        return cube

    def set_beam_kernel(self, support_scaling=8.):

        if (self.beam_type == 'analytic') | (self.beam_type == None):
            

            if isinstance(self.beam, Beam):
                kernel = self.beam.as_kernel(self.pixscale, support_scaling=support_scaling)
                kern2D = kernel.array
            else:
                kernel = self.beam.as_kernel(self.pixscale, support_scaling=support_scaling)
                
            if isinstance(self.beam, DoubleBeam):
                kern2D = kernel
            elif isinstance(self.beam, Moffat):
                kern2D = kernel

        elif self.beam_type == 'empirical':
            if len(self.beam.shape) == 1:
                raise ValueError("1D beam/PSF not currently supported")

            kern2D = self.beam.copy()

            # Replace NaNs/non-finite with zero:
            kern2D[~np.isfinite(kern2D)] = 0.

            # Replace < 0 with zero:
            kern2D[kern2D<0.] = 0.
            kern2D /= np.sum(kern2D[np.isfinite(kern2D)])  # need to normalize

        kern3D = np.zeros(shape=(1, kern2D.shape[0], kern2D.shape[1],))
        kern3D[0, :, :] = kern2D

        self._beam_kernel = kern3D

    def set_lsf_kernel(self, spec_center=None):

        if (self.spec_step is None):
            raise ValueError("Spectral step not defined.")

        elif (self.spec_step is not None) and (self.spec_type == 'velocity'):

            kernel = self.lsf.as_velocity_kernel(self.spec_step)

        elif (self.spec_step is not None) and (self.spec_type == 'wavelength'):

            if (self.spec_center is None) and (spec_center is None):
                raise ValueError("Center wavelength not defined in either "
                                 "the instrument or call to convolve.")

            elif (spec_center is not None):
                #logger.info("Overriding the instrument central wavelength "
                #            "with {}.".format(spec_center))

                kernel = self.lsf.as_wave_kernel(self.spec_step, spec_center)

            else:

                kernel = self.lsf.as_wave_kernel(self.spec_step,
                                                 self.spec_center)

        kern1D = kernel.array
        kern3D = np.zeros(shape=(kern1D.shape[0], 1, 1,))
        kern3D[:, 0, 0] = kern1D

        self._lsf_kernel = kern3D


    @property
    def beam(self):
        return self._beam

    @beam.setter
    def beam(self, new_beam):
        if isinstance(new_beam, Beam):
            self._beam = new_beam
        elif new_beam is None:
            self._beam = None
        else:
            raise TypeError("Beam must be an instance of "
                            "radio_beam.beam.Beam")

    @property
    def lsf(self):
        return self._lsf

    @lsf.setter
    def lsf(self, new_lsf):
        self._lsf = new_lsf

    @property
    def pixscale(self):
        return self._pixscale

    @pixscale.setter
    def pixscale(self, value):
        if value is None:
            self._pixscale = value
        elif not isinstance(value, u.Quantity):
            logger.warning("No units on pixscale. Assuming arcseconds.")
            self._pixscale = value*u.arcsec
        else:
            if (u.arcsec).is_equivalent(value):
                self._pixscale = value
            else:
                raise u.UnitsError("pixscale not in equivalent units to "
                                   "arcseconds.")

    @property
    def spec_step(self):
        return self._spec_step

    @spec_step.setter
    def spec_step(self, value):
        if value is None:
            self._spec_step = value
        elif not isinstance(value, u.Quantity):
            logger.warning("No units on spec_step. Assuming km/s.")
            self._spec_step = value * u.km/u.s
        else:
            self._spec_step = value


    @property
    def spec_center(self):
        if (self.spec_start is None) | (self.nspec is None):
            return None
        else:
            return (self.spec_start + np.round(self.nspec/2)*self.spec_step)



class LSF(u.Quantity):
    """
    An object to handle line spread functions.
    """

    def __new__(cls, dispersion=None, default_unit=u.km/u.s, meta=None):
        """
        Create a new Gaussian Line Spread Function
        Parameters
        ----------
        dispersion : :class:`~astropy.units.Quantity` with speed equivalency
        default_unit : :class:`~astropy.units.Unit`
            The unit to impose on dispersion if they are specified as floats
        """

        # TODO: Allow for wavelength dispersion to be specified

        # error checking

        # give specified values priority
        if dispersion is not None:
            if (u.km/u.s).is_equivalent(dispersion):
                dispersion = dispersion
            else:
                logger.warning("Assuming dispersion has been specified in "
                               "km/s.")
                dispersion = dispersion * default_unit

        self = super(LSF, cls).__new__(cls, dispersion.value, u.km/u.s)
        self._dispersion = dispersion
        self.default_unit = default_unit

        if meta is None:
            self.meta = {}
        elif isinstance(meta, dict):
            self.meta = meta
        else:
            raise TypeError("metadata must be a dictionary")

        return self

    def __repr__(self):
        return "LSF: Vel. Disp. = {0}".format(
            self.dispersion.to(self.default_unit))

    def __str__(self):
        return self.__repr__()

    @property
    def dispersion(self):
        return self._dispersion

    def vel_to_lambda(self, wave):
        """
        Convert from velocity dispersion to wavelength dispersion for
        a given central wavelength.
        """

        if not isinstance(wave, u.Quantity):
            raise TypeError("wave must be a Quantity object. "
                            "Try 'wave*u.Angstrom' or another equivalent unit.")
        return (self.dispersion/c.c.to(self.dispersion.unit))*wave

    def as_velocity_kernel(self, velstep, **kwargs):
        """
        Return a Gaussian convolution kernel in velocity space
        """

        sigma_pixel = (self.dispersion.value /
                       velstep.to(self.dispersion.unit).value)

        return apy_conv.Gaussian1DKernel(sigma_pixel, **kwargs)

    def as_wave_kernel(self, wavestep, wavecenter, **kwargs):
        """
        Return a Gaussian convolution kernel in wavelength space
        """

        sigma_pixel = self.vel_to_lambda(wavecenter).value/wavestep.value

        return apy_conv.Gaussian1DKernel(sigma_pixel, **kwargs)


class DoubleBeam:

    def __init__(self, major1=None, minor1=None, pa1=None, scale1=None,
                 major2=None, minor2=None, pa2=None, scale2=None):


        if (major1 is None) or (major2 is None):
            raise ValueError('Need to specify at least the major axis FWHM of each beam component.')

        if minor1 is None:
            minor1 = major1

        if minor2 is None:
            minor2 = major2

        if pa1 is None:
            pa1 = 0.*u.deg

        if pa2 is None:
            pa2 = 0.*u.deg

        if scale1 is None:
            scale1 = 1.0

        if scale2 is None:
            scale2 = scale1

        self.beam1 = Beam(major=major1, minor=minor1, pa=pa1)
        self.beam2 = Beam(major=major2, minor=minor2, pa=pa2)
        self._scale1 = scale1
        self._scale2 = scale2


    def as_kernel(self, pixscale, support_scaling=None):

        kernel1 = self.beam1.as_kernel(pixscale, support_scaling=support_scaling)
        kernel2 = self.beam2.as_kernel(pixscale, support_scaling=support_scaling)

        if kernel1.shape[0] > kernel2.shape[1]:

            xsize = kernel1.shape[0]
            ysize= kernel1.shape[1]
            kernel2 = self.beam2.as_kernel(pixscale, x_size=xsize, y_size=ysize)

        elif kernel1.shape[0] < kernel2.shape[1]:

            xsize = kernel2.shape[0]
            ysize = kernel2.shape[1]
            kernel1 = self.beam1.as_kernel(pixscale, x_size=xsize, y_size=ysize)

        # Combine the kernels
        kernel_total = 10. * (kernel1.array * self._scale1 / np.sum(kernel1.array) +
                              kernel2.array * self._scale2 / np.sum(kernel2.array))

        return kernel_total
        
class Moffat(object):

    def __init__(self, major_fwhm=None, minor_fwhm=None, pa=None, beta=None, padfac=16.):
        
        if (major_fwhm is None) | (beta is None):
            raise ValueError('Need to specify at least the major axis FWHM + beta of beam.')
        
        if minor_fwhm is None:
            minor_fwhm = major_fwhm
        if (major_fwhm != minor_fwhm) & (pa is None):
            raise ValueError("Need to specifiy 'pa' to have elliptical PSF!")
            
        
        if pa is None:
            pa = 0.*u.deg
        
        #
        self.major_fwhm = major_fwhm
        self.minor_fwhm = minor_fwhm
        self.pa = pa
        self.beta = beta
        
        self.alpha = self.major_fwhm/(2.*np.sqrt(np.power(2., 1./np.float(self.beta)) - 1 ))
        
        self.padfac = padfac

    def as_kernel(self, pixscale, support_scaling=None):
        
        try:
            pixscale = pixscale.to(self.major_fwhm.unit)
            pixscale = pixscale.value
            
            major_fwhm = self.major_fwhm.value
            minor_fwhm = self.minor_fwhm.value
            pa = self.pa.value
            
            alpha = self.alpha.value/pixscale
        except:
            pixscale = pixscale.to(self.major_fwhm.unit)
            pixscale = pixscale.value
            
            major_fwhm = self.major_fwhm
            minor_fwhm = self.minor_fwhm
            pa = self.pa
            alpha = self.alpha/pixscale
        
        
        #padfac = 16. #8. # from Beam
        if support_scaling is not None:
            padfac = support_scaling
        else:
            padfac = self.padfac
        
        npix = np.int(np.ceil(major_fwhm/pixscale/2.35 * padfac))
        if npix % 2 == 0:
            npix += 1
        
        
        print("alpha={}, beta={}, fwhm={}, pixscale={}, npix={}".format(alpha*pixscale, self.beta, 
                    major_fwhm, pixscale, npix))
        
        
        # Arrays
        y, x = np.indices((npix, npix), dtype=float)
        x -= (npix-1)/2.
        y -= (npix-1)/2.
        
        
        
        cost = np.cos(pa*np.pi/180.)
        sint = np.sin(pa*np.pi/180.)
        
        xp = cost*x + sint*y
        yp = -sint*x + cost*y
        
        # print("x={}, y={}".format(x,y))
        # print("xp={}, yp={}".format(xp,yp))
        
        qtmp = minor_fwhm / major_fwhm
        
        
        r = np.sqrt(xp**2 + (yp/qtmp)**2)
        
        kernel = (self.beta-1.)/(np.pi * alpha**2) * np.power( (1. + (r/alpha)**2 ), -1.*self.beta )
        
        return kernel
    