# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# File containing classes for defining different instruments
# to "observe" a model galaxy.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np
# import astropy.convolution as apy_conv
from astropy.convolution.utils import discretize_model as _discretize_model
from astropy.modeling import models as apy_models
from scipy.signal import fftconvolve
import astropy.units as u
import astropy.constants as c
from radio_beam import Beam as _RBeam

__all__ = ["Instrument", "GaussianBeam", "DoubleBeam", "Moffat", "LSF"]

# CONSTANTS
sig_to_fwhm = 2.*np.sqrt(2.*np.log(2.))

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)


def _normalized_gaussian1D_kern(sigma_pixel):
    x_size = int(np.ceil(8*sigma_pixel))
    if x_size % 2 == 0:
        x_size += 1
    x_range = (-(int(x_size) - 1) // 2, (int(x_size) - 1) // 2 + 1)
    
    return _discretize_model(apy_models.Gaussian1D(1.0 / (np.sqrt(2 * np.pi) * sigma_pixel), 
                0, sigma_pixel), x_range)


class Instrument:
    """
    Base class to define an instrument to observe a model galaxy.

    Parameters
    ----------
    beam : 2D array, `~dysmalpy.instrument.GaussianBeam`, `~dysmalpy.instrument.DoubleBeam`, or `~dysmalpy.instrument.Moffat`
           Object describing the PSF of the instrument

    beam_type : {`'analytic'`, `'empirical'`}

        * `'analytic'` implies the beam is one of the provided beams in `dysmalpy`.

        * `'empirical'` implies the provided beam is a 2D array that describes
            the convolution kernel.

    lsf : LSF object
          Object describing the line spread function of the instrument

    pixscale : float or `~astropy.units.Quantity`
            Size of one pixel on the sky. If no units are used, arcseconds are assumed.

    spec_type : {`'velocity'`, `'wavelength'`}
            Whether the spectral axis is in velocity or wavelength space

    spec_start : `~astropy.units.Quantity`
            The value and unit of the first spectral channel

    spec_step : `~astropy.units.Quantity`
            The spacing of the spectral channels

    nspec : int
            Number of spectral channels

    fov : tuple
          x and y size of the FOV of the instrument in pixels

    name : str
           Name of the instrument
           
    """

    def __init__(self, beam=None, beam_type=None, lsf=None, pixscale=None,
                 spec_type='velocity', spec_start=-1000*u.km/u.s,
                 spec_step=10*u.km/u.s, nspec=201,
                 line_center=None,
                 smoothing_type=None, smoothing_npix=None,
                 moment=False,
                 apertures=None,
                 integrate_cube=True, slit_width=None, slit_pa=None,
                 fov=None,
                 ndim=None,
                 name='Instrument'):

        self.name = name
        self.ndim = ndim

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


        # Wave spec options:
        self.line_center = line_center

        # 3D / 2D options:
        self.smoothing_type = smoothing_type
        self.smoothing_npix = smoothing_npix

        # 2D / 1D options:
        self.moment = moment

        # 1D options:
        self.apertures = apertures

        # 0D options
        self.integrate_cube = integrate_cube
        self.slit_width = slit_width
        self.slit_pa = slit_pa



    def convolve(self, cube, spec_center=None):
        """
        Perform convolutions that are associated with this Instrument

        This method convolves the input cube such that the output cube has the correct
        PSF and/or LSF as given by the Instrument's beam and lsf. The cube is first convolved to
        the PSF, then to the LSF.

        Parameters
        ----------
        cube : 3D array
               Input model cube to convolve with the instrument PSF and/or LSF
               The first dimension is assumed to correspond to the spectral dimension.
        spec_center : `~astropy.units.Quantity`
                      Central wavelength to define the conversion from velocity to wavelength
                      Only necessary if the spectral type is 'wavelength' and using an LSF

        Returns
        -------
        cube : 3D array
               New model cube after applying the PSF and LSF convolution
        """

        if self.beam is None and self.lsf is None:
            # Nothing to do if a PSF and LSF aren't set for the instrument
            logger.warning("No convolution being performed since PSF and LSF "
                           "haven't been set!")
            return cube

        elif self.lsf is None:
            cube_conv = self.convolve_with_beam(cube)

        elif self.beam is None:
            cube_conv = self.convolve_with_lsf(cube, spec_center=spec_center)

        else:
            # Separate convolution steps: note this is FASTER than doing a composite convolution
            cube_conv_beam = self.convolve_with_beam(cube)
            cube_conv = self.convolve_with_lsf(cube_conv_beam, spec_center=spec_center)

        return cube_conv

    def convolve_with_lsf(self, cube, spec_center=None):
        """
        Performs line broadening due to the line spread function

        Parameters
        ----------
        cube : 3D array
               Input model cube to convolve with the instrument LSF
               The first dimension is assumed to correspond to the spectral dimension.
        spec_center : `~astropy.units.Quantity`
                      Central wavelength to define the conversion from velocity to wavelength
                      Only necessary if the instrument spectral type is 'wavelength'

        Returns
        -------
        cube : 3D array
               New model cube after applying the LSF convolution
        """

        if self._lsf_kernel is None:
            self.set_lsf_kernel(spec_center=spec_center)

        cube_conv = fftconvolve(cube.copy(), self._lsf_kernel.copy(), mode='same')

        return cube_conv

    def convolve_with_beam(self, cube):
        """
        Performs spatial broadening due to the point spread function

        Parameters
        ----------
        cube : 3D array
               Input model cube to convolve with the instrument PSF
               The first dimension is assumed to correspond to the spectral dimension.

        Returns
        -------
        cube : 3D array
               New model cube after applying the PSF convolution
        """

        if self.pixscale is None:
            raise ValueError("Pixelscale for this instrument has not "
                             "been set yet. Can't convolve with beam.")

        if self._beam_kernel is None:
            self.set_beam_kernel()

        cube_conv = fftconvolve(cube.copy(), self._beam_kernel.copy(), mode='same')

        return cube_conv

    def _clear_kernels(self):
        """
        Delete pre-computed kernels
        """
        self._beam_kernel = None
        self._lsf_kernel = None


    def set_beam_kernel(self, support_scaling=12.):
        """
        Calculate and store the PSF convolution kernel

        Parameters
        ----------
        support_scaling : int
                          The amount to scale the stddev to determine the
                          size of the kernel

        """
        if (self.beam_type == 'analytic') | (self.beam_type == None):

            if isinstance(self.beam, GaussianBeam):
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

            kern2D[~np.isfinite(kern2D)] = 0.               # Replace NaNs/non-finite with zero
            kern2D[kern2D<0.] = 0.                          # Replace < 0 with zero:

            kern2D /= np.sum(kern2D[np.isfinite(kern2D)])   # need to normalize

        kern3D = np.zeros(shape=(1, kern2D.shape[0], kern2D.shape[1],))
        kern3D[0, :, :] = kern2D

        self._beam_kernel = kern3D

    def set_lsf_kernel(self, spec_center=None):
        """
        Calculate and store the LSF convolution kernel

        Parameters
        ----------
        spec_center : `~astropy.units.Quantity`, optional
                      Central wavelength that corresponds to 0 velocity
                      Only necessary if `Instrument.spec_type = 'wavelength'`
                      and `Instrument.spec_center` hasn't been set.

        """
        if (self.spec_step is None):
            raise ValueError("Spectral step not defined.")

        elif (self.spec_step is not None) and (self.spec_type == 'velocity'):
            kernel = self.lsf.as_velocity_kernel(self.spec_step)

        elif (self.spec_step is not None) and (self.spec_type == 'wavelength'):
            if (self.line_center is None) and (spec_center is None):
                raise ValueError("Center wavelength not defined in either "
                                 "the instrument or call to convolve.")

            elif (spec_center is not None):
                kernel = self.lsf.as_wave_kernel(self.spec_step, spec_center)

            else:
                kernel = self.lsf.as_wave_kernel(self.spec_step, self.line_center)

        #kern1D = kernel.array
        kern1D = kernel
        kern3D = np.zeros(shape=(kern1D.shape[0], 1, 1,))
        kern3D[:, 0, 0] = kern1D

        self._lsf_kernel = kern3D

    @property
    def beam(self):
        return self._beam

    @beam.setter
    def beam(self, new_beam):
        if isinstance(new_beam, GaussianBeam) | isinstance(new_beam, Moffat) | isinstance(new_beam, DoubleBeam):
            self._beam = new_beam
        elif new_beam is None:
            self._beam = None
        else:
            raise TypeError("Beam must be an instance of "
                            "instrument.GaussianBeam, instrument.Moffat, or instrument.DoubleBeam")

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


class GaussianBeam(_RBeam):
    """
    Re-definition of radio_beam.Beam to allow it to work with copy.deepcopy and copy.copy.

    PA: angle to beam major axis, in deg E of N (0 is N, +90 is E).
    """

    def __deepcopy__(self, memo):
        self2 = type(self)(major=self._major, minor=self._minor, pa=self._pa, area=None,
                           default_unit=self.default_unit, meta=self.meta)
        self2.__dict__.update(self.__dict__)
        return self2

    def __copy__(self):
        self2 = type(self)(major=self._major, minor=self._minor, pa=self._pa, area=None,
                           default_unit=self.default_unit, meta=self.meta)
        self2.__dict__.update(self.__dict__)
        return self2


class DoubleBeam:
    """
    Beam object that is the superposition of two Gaussian Beams

    Parameters
    ----------
    major1 : `~astropy.units.Quantity`
             FWHM along the major axis of the first Gaussian beam
    minor1 : `~astropy.units.Quantity`
             FWHM along the minor axis of the first Gaussian beam
    pa1 : `~astropy.units.Quantity`
          Position angle of the first Gaussian beam:
            angle to beam major axis, in deg E of N (0 is N, +90 is E).
    scale1 : float
             Flux scaling for the first Gaussian beam
    major2 : `~astropy.units.Quantity`
             FWHM along the major axis of the second Gaussian beam
    minor2 : `~astropy.units.Quantity`
             FWHM along the minor axis of the second Gaussian beam
    pa2 : `~astropy.units.Quantity`
          Position angle of the second Gaussian beam:
            angle to beam major axis, in deg E of N (0 is N, +90 is E).
    scale2 : float
             Flux scaling for the second Gaussian beam
    """

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

        self.beam1 = GaussianBeam(major=major1, minor=minor1, pa=pa1)
        self.beam2 = GaussianBeam(major=major2, minor=minor2, pa=pa2)
        self._scale1 = scale1
        self._scale2 = scale2


    def as_kernel(self, pixscale, support_scaling=None):
        """
        Calculate the convolution kernel for the DoubleBeam

        Parameters
        ----------
        pixscale : `~astropy.units.Quantity`
                   Pixel scale of image that will be convolved
        support_scaling : int
                          The amount to scale the stddev to determine the
                          size of the kernel

        Returns
        -------
        kernel_total : 2D array
                       Convolution kernel for the DoubleBeam
        """

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

    def __deepcopy__(self, memo):
        self2 = type(self)(major1=self.beam1._major, major2=self.beam2._major, minor1=self.beam1._minor,
                           minor2=self.beam2._minor, pa1=self.beam1._pa, pa2=self.beam2._pa,
                           scale1=self._scale1, scale2=self._scale2)

        self2.__dict__.update(self.__dict__)
        return self2

    def __copy__(self):
        self2 = type(self)(major1=self.beam1._major, major2=self.beam2._major, minor1=self.beam1._minor,
                           minor2=self.beam2._minor, pa1=self.beam1._pa, pa2=self.beam2._pa,
                           scale1=self._scale1, scale2=self._scale2)
        self2.__dict__.update(self.__dict__)
        return self2


class Moffat(object):
    """
    Object describing a Moffat PSF

    Parameters
    ----------
    major_fwhm : `~astropy.units.Quantity`
                 FWHM of the Moffat PSF along the major axis
    minor_fwhm : `~astropy.units.Quantity`
                 FWHM of the Moffat PSF along the minor axis
    pa : `~astropy.units.Quantity`
         Position angle of major axis of the Moffat PSF, in deg E of N (0 is N, +90 is E).
    beta : float
           beta parameter of the Moffat PSF
    padfac : int
             The amount to scale the stddev to determine the
             size of the kernel
    """
    def __init__(self, major_fwhm=None, minor_fwhm=None, pa=None, beta=None, padfac=12.):

        if (major_fwhm is None) | (beta is None):
            raise ValueError('Need to specify at least the major axis FWHM + beta of beam.')

        if minor_fwhm is None:
            minor_fwhm = major_fwhm
        if (major_fwhm != minor_fwhm) & (pa is None):
            raise ValueError("Need to specifiy 'pa' to have elliptical PSF!")


        if pa is None:
            pa = 0.*u.deg

        self.major_fwhm = major_fwhm
        self.minor_fwhm = minor_fwhm
        self.pa = pa
        self.beta = beta

        self.alpha = self.major_fwhm/(2.*np.sqrt(np.power(2., 1./np.float(self.beta)) - 1 ))

        self.padfac = padfac

    def as_kernel(self, pixscale, support_scaling=None):
        """
        Calculate the convolution kernel for the Moffat PSF

        Parameters
        ----------
        pixscale : `~astropy.units.Quantity`
                   Pixel scale of image that will be convolved
        support_scaling : int
                          The amount to scale the stddev to determine the
                          size of the kernel

        Returns
        -------
        kernel : 2D array
                 Convolution kernel for the Moffat PSF
        """
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


        if support_scaling is not None:
            padfac = support_scaling
        else:
            padfac = self.padfac

        # Npix: rounded std dev[ in pix] * padfac * 2 -- working in DIAMETER
        # Factor of 1./0.7: For beta~2.5, Moffat FWHM ~ 0.7*Gaus FWHM
        #    -> add extra padding so the Moffat window
        #       isn't much smaller than similar Gaussian PSF.

        npix = np.int(np.ceil(major_fwhm/pixscale/2.35 * 2 * 1./0.7 * padfac))
        if npix % 2 == 0:
            npix += 1

        # Arrays
        y, x = np.indices((npix, npix), dtype=float)
        x -= (npix-1)/2.
        y -= (npix-1)/2.


        cost = np.cos((90.+pa)*np.pi/180.)
        sint = np.sin((90.+pa)*np.pi/180.)

        xp = cost*x + sint*y
        yp = -sint*x + cost*y

        qtmp = minor_fwhm / major_fwhm

        r = np.sqrt(xp**2 + (yp/qtmp)**2)

        kernel = (self.beta-1.)/(np.pi * alpha**2) * np.power( (1. + (r/alpha)**2 ), -1.*self.beta )

        return kernel

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
            The standard deviation of the Gaussian LSF.

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
                               "{}.".format(default_unit))
                dispersion = dispersion * default_unit

        self = super(LSF, cls).__new__(cls, dispersion.value, default_unit)
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

        Parameters
        ----------
        wave : `~astropy.units.Quantity`
               Central wavelength to use in conversion from velocity to wavelength

        Returns
        -------
        wdisp : `~astropy.units.Quantity`
                Dispersion of LSF in wavelength units
        """

        if not isinstance(wave, u.Quantity):
            raise TypeError("wave must be a Quantity object. "
                            "Try 'wave*u.Angstrom' or another equivalent unit.")
        return (self.dispersion/c.c.to(self.dispersion.unit))*wave

    def as_velocity_kernel(self, velstep, **kwargs):
        """
        Return a Gaussian convolution kernel in velocity space

        Parameters
        ----------
        velstep : `~astropy.units.Quantity`
                  Step size in velocity of one spectral channel

        Returns
        -------
        vel_kern : 1D array
                   Convolution kernel for the LSF in velocity space
        """

        sigma_pixel = (self.dispersion.value /
                       velstep.to(self.dispersion.unit).value)

        #return apy_conv.Gaussian1DKernel(sigma_pixel, **kwargs)

        # Astropy kernel DOES NOT TRUNCATE AS DESIRED:
        # Instead, use similar size span, but keep as a normalized Gaussian:
        return _normalized_gaussian1D_kern(sigma_pixel)

    def as_wave_kernel(self, wavestep, wavecenter, **kwargs):
        """
        Return a Gaussian convolution kernel in wavelength space

        Parameters
        ----------
        wavestep : `~astropy.units.Quantity`
                   Step size in wavelength of one spectral channel
        wavecenter : `~astropy.units.Quantity`
                     Central wavelength used to convert from velocity to wavelength

        Returns
        -------
        wave_kern : 1D array
                    Convolution kernal for the LSF in wavelength space
        """

        sigma_pixel = self.vel_to_lambda(wavecenter).value/wavestep.value

        #return apy_conv.Gaussian1DKernel(sigma_pixel, **kwargs)

        # Astropy kernel DOES NOT TRUNCATE AS DESIRED:
        # Instead, use similar size span, but keep as a normalized Gaussian:
        return _normalized_gaussian1D_kern(sigma_pixel)

    def __deepcopy__(self, memo):
        self2 = type(self)(dispersion=self._dispersion, default_unit=self.default_unit,
                           meta=self.meta)
        self2.__dict__.update(self.__dict__)
        return self2

    def __copy__(self):
        self2 = type(self)(dispersion=self._dispersion, default_unit=self.default_unit,
                           meta=self.meta)
        self2.__dict__.update(self.__dict__)
        return self2
