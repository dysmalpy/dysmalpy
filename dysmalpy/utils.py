# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Module containing some useful utility functions
# 
# The `_rotate_points` and `symmetrize_velfield` functions are adopted from `cap_symmetrize_velfield.py` within `display_pixels` created by Michele Cappellari.
#######################################################################
#
# Copyright (C) 2004-2014, Michele Cappellari
# E-mail: cappellari_at_astro.ox.ac.uk
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
#######################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import warnings
import logging

# Third party imports
import numpy as np
import astropy.modeling as apy_mod
import astropy.stats as apy_stats
# import astropy.units as u
import matplotlib.pyplot as plt
import scipy.signal as sp_sig
import scipy.ndimage as sp_ndi
from scipy import interpolate
from scipy.optimize import minimize, leastsq
from scipy.stats import norm
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve_fft as apy_convolve_fft

std2fwhm = (2. *np.sqrt(2.*np.log(2.)))

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


__all__ = [ "rebin", "calc_pixel_distance", "create_aperture_mask", 
            "determine_aperture_centers", "calc_pix_position", 
            "measure_1d_profile_apertures", "apply_smoothing_2D", "apply_smoothing_3D", 
            "symmetrize_velfield", "symmetrize_1D_profile", 
            "fit_truncated_gaussian", "lnlike_truncnorm", "fit_uncertainty_ellipse", 
            "gaus_fit_sp_opt_leastsq", "gaus_fit_apy_mod_fitter", "get_cin_cout",
            "citations"]

# Function to rebin a cube in the spatial dimension
def rebin(arr, new_2dshape):
    shape = (arr.shape[0],
             new_2dshape[0], arr.shape[1] // new_2dshape[0],
             new_2dshape[1], arr.shape[2] // new_2dshape[1])
    return arr.reshape(shape).sum(-1).sum(-2)
    

def calc_pixel_distance(nx, ny, center_coord):
    """
    Function to calculate the distance of each pixel
    from a specific pixel as well as the position angle

    Parameters
    ----------
    nx: float
    ny: float
    center_coord: tuple

    Returns
    -------
    seps: float
    pa: float
    """

    xx, yy = np.meshgrid(range(nx), range(ny))

    dx = xx - center_coord[0]
    dy = yy - center_coord[1]

    # Calculate the separation
    seps = np.sqrt(dx**2 + dy**2)

    pa = np.arctan2(-dx, dy)*180./np.pi

    return seps, pa


def create_aperture_mask(nx, ny, center_coord, dr):
    """
    Function to create an aperture mask. NOT USED.

    Parameters
    ----------
    nx: float
    ny: float
    center_coord: tuple
    dr: float

    """

    seps, pa = calc_pixel_distance(nx, ny, center_coord)

    # Return boolian if center w/in the aperture area
    return seps <= dr

    # ##
    # seps_ul, pa = calc_pixel_distance(nx, ny, [center_coord[0]+0.5, center_coord[1]-0.5])
    # seps_ll, pa = calc_pixel_distance(nx, ny, [center_coord[0]+0.5, center_coord[1]+0.5])
    # seps_ur, pa = calc_pixel_distance(nx, ny, [center_coord[0]-0.5, center_coord[1]-0.5])
    # seps_lr, pa = calc_pixel_distance(nx, ny, [center_coord[0]-0.5, center_coord[1]+0.5])
    #
    #
    # mask_ap = np.zeros(seps_ul.shape)
    # mask_ap[(seps_ul <= dr) | (seps_ll <= dr) | (seps_ur <= dr) | (seps_lr <= dr)] = 1.
    #
    # return mask_ap


def determine_aperture_centers(nx, ny, center_coord, pa, dr):
    """
    Determine the centers of the apertures that span an image/cube along a line with position
    angle pa and goes through center_coord. Each aperture is dr away from each other.
    
    Parameters
    ----------
    nx: float
    ny: float
    center_coord: tuple
    pa: float
    dr: float
    
    """

    pa_rad = -np.pi/180. * pa

    # Calculate the intersection of the line defined by PA and center_coord with the edges
    # of the image/cube
    xcenter = center_coord[0]
    ycenter = center_coord[1]
    count = 0
    edge_points = []

    # Check the x = 0 border
    y_x0 = -xcenter/np.tan(pa_rad) + ycenter
    if (y_x0 >= 0) & (y_x0 <= ny):
        count += 1
        edge_points.append([0, y_x0])

    # Check the y = 0 border
    x_y0 = -ycenter*np.tan(pa_rad) + xcenter
    if (x_y0 >= 0) & (x_y0 <= nx):
        count += 1
        edge_points.append([x_y0, 0])

    # Check the x = nx border
    y_nx = (nx - xcenter)/np.tan(pa_rad) + ycenter
    if (y_nx >= 0) & (y_nx <= ny):
        count += 1
        edge_points.append([nx, y_nx])

    # Check the y = ny border
    x_ny = (ny - ycenter) * np.tan(pa_rad) + xcenter
    if (x_ny >= 0) & (x_ny <= nx):
        count += 1
        edge_points.append([x_ny, ny])

    # Make sure there are only two intersections
    if count != 2:
        raise(ValueError, 'Number of intersections is not 2, something wrong happened!')

    # Calculate the start and end radii for the aperture centers based on the border
    # intersections. If the intersection is below ycenter the radius is negative
    r1 = np.sqrt((edge_points[0][0] - xcenter)**2 + (edge_points[0][1] - ycenter)**2)
    if edge_points[0][1] < ycenter:
        r1 = -r1

    r2 = np.sqrt((edge_points[1][0] - xcenter) ** 2 + (edge_points[1][1] - ycenter) ** 2)
    if edge_points[1][1] < ycenter:
        r2 = -r2

    # Setup the radii for the aperture centers with r = 0 always occurring
    if r1 > r2:
        r_neg = np.sort(-np.arange(0, -r2, dr))
        r_pos = np.arange(0, r1, dr)
        r_centers = np.concatenate((r_neg, r_pos[1:]))

    else:
        r_neg = np.sort(-np.arange(0, -r1, dr))
        r_pos = np.arange(0, r2, dr)
        r_centers = np.concatenate((r_neg, r_pos[1:]))

    # Get the pixel positions for each radii
    xaps, yaps = calc_pix_position(r_centers, pa, xcenter, ycenter)

    return xaps, yaps, r_centers


def calc_pix_position(r, pa, xcenter, ycenter):
    """
    Simple function to determine the pixel that is r away from (xcenter, ycenter) along
    a line with position angle (pa)

    Parameters
    ----------
    r: float
        distance from (xcenter,ycenter) in pixel
    pa: float
        position angle counter-clockwise from North
    xcenter: float
        x center in pixel
    ycenter: float
        y center in pixel

    """

    pa_rad = np.pi/180. * pa
    #signfac = np.sign(np.cos(pa_rad))
    #xnew = -r*np.sin(pa_rad)*signfac + xcenter
    #signfac = np.sign(np.sin(pa_rad))
    signfac = -1
    xnew = -r*np.sin(pa_rad)*signfac + xcenter
    ynew = r*np.cos(pa_rad)*signfac + ycenter

    return xnew, ynew


def measure_1d_profile_apertures(cube, rap, pa, spec_arr, dr=None, center_pixel=None,
                                 ap_centers=None, spec_mask=None, estimate_err=False, nmc=100,
                                 profile_direction='positive', debug=False):
    """
    Measure the 1D rotation curve using equally spaced apertures along a defined axis
    
    Parameters
    ----------
    cube: ndarray
        Cube to measure the 1D profile on
    rap: float
        Radius of the circular apertures in pixels
    dr: float
        Distance between the circular apertures in pixels
    center_pixel: tuple of int
        Central pixel that defines r = 0
    pa: float
        Position angle of the line that the circular apertures lie on
    spec_arr: ndarray
        The spectral array (i.e. wavelengths, frequencies, velocities, etc)
    spec_mask: ndarray
        Boolean mask to apply to the spectrum to exclude from fitting
    estimate_err: bool, optional
        True or False to use Monte Carlo to estimate the errors on the fits
    nmc: int, optional
        The number of trials in the Monte Carlo analysis to use.

    Returns
    ----------
        centers: ndarray
            The radial offset from the central pixel in pixels
        flux: ndarray
            The integrated best fit "flux" for each aperture
        mean: ndarray
            The best fit mean of the Gaussian profile in the same units as spec_arr
        disp: ndarray
            The best fit dispersion of the Gaussian profile in the same units as spec_arr

    Note: flux, mean, and disp will be Nap x 2 arrays if `estimate_err = True` where Nap is the number of apertures that are fit. The first row will be best fit values and the second row will contain the errors on those parameters.
    """
    # profile_direction = 'negative'

    raise ValueError("This function is depreciated, and we should switch any other calls to it.\n Please let codewriters know.")


    ny = cube.shape[1]
    nx = cube.shape[2]

    # Assume the default central pixel is the center of the cube
    if center_pixel is None:
        center_pixel = ((nx - 1) / 2., (ny - 1) / 2.)

    # Assume equal distance between successive apertures equal to diameter of aperture
    if dr is None:
        dr = 2*rap

    # First determine the centers of all the apertures that fit within the cube
    if ap_centers is None:
        xaps, yaps, ap_centers = determine_aperture_centers(nx, ny, center_pixel, pa, dr)
    else:
        xaps, yaps = calc_pix_position(ap_centers, pa, center_pixel[0], center_pixel[1])

    ### SHOULD SORT THIS OUT FROM DATA by correctly setting model geom PA + slit_pa
    # if (profile_direction == 'negative') & (np.abs(pa) < 90.0):
    #     ap_centers = -ap_centers

    # Setup up the arrays to hold the results
    naps = len(xaps)
    flux = np.zeros(naps)
    mean = np.zeros(naps)
    disp = np.zeros(naps)

    if estimate_err:
        flux_err = np.zeros(naps)
        mean_err = np.zeros(naps)
        disp_err = np.zeros(naps)

    # For each aperture, sum up the spaxels within the aperture and fit a Gaussian line profile
    for i in range(naps):

        mask_ap = create_aperture_mask(nx, ny, (xaps[i], yaps[i]), rap)
        mask_cube = np.tile(mask_ap, (cube.shape[0], 1, 1))
        spec = np.nansum(np.nansum(cube*mask_cube, axis=1), axis=1)

        if spec_mask is not None:
            spec_fit = spec[spec_mask]
            spec_arr_fit = spec_arr[spec_mask]
        else:
            spec_fit = spec
            spec_arr_fit = spec_arr

        # Use the first and second moment as a guess of the line parameters
        mom0 = np.sum(spec_fit)
        mom1 = np.sum(spec_fit * spec_arr_fit) / mom0
        mom2 = np.sum(spec_fit * (spec_arr_fit - mom1) ** 2) / mom0

        ## OLD: astropy fitter
        # mod = apy_mod.models.Gaussian1D(amplitude=mom0 / np.sqrt(2 * np.pi * np.abs(mom2)),
        #                                 mean=mom1,
        #                                 stddev=np.sqrt(np.abs(mom2)))
        # mod.amplitude.bounds = (0, None)
        # mod.stddev.bounds = (0, None)
        # fitter = apy_mod.fitting.LevMarLSQFitter()
        # best_fit = fitter(mod, spec_arr_fit, spec_fit)
        #
        # mean[i] = best_fit.mean.value
        # disp[i] = best_fit.stddev.value
        # flux[i] = best_fit.amplitude.value * np.sqrt(2 * np.pi) * disp[i]

        ## NEW: Direct to scipy.optimize.leastsq fitting:
        best_fit = gaus_fit_sp_opt_leastsq(spec_fit, spec_arr_fit, mom0, mom1, mom2)
        flux[i] = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
        mean[i] = best_fit[1]
        disp[i] = best_fit[2]

        if debug:
            print(ap_centers[i], xaps[i], yaps[i])
            plt.figure()
            plt.imshow(mask_ap)
            plt.plot(xaps[i], yaps[i], 'ro', ms=4)
            theta = np.arange(0, 2*np.pi, 0.01)
            xcircle = rap*np.cos(theta) + xaps[i]
            ycircle = rap*np.sin(theta) + yaps[i]
            plt.plot(xcircle, ycircle, 'g--')
            plt.title('r = {0}'.format(ap_centers[i]))

            #plt.figure()
            #plt.plot(spec_arr_fit, spec_fit)
            #plt.plot(spec_arr_fit, best_fit(spec_arr_fit))
            #plt.title('r = {0}'.format(ap_centers[i]))

        if estimate_err:
            residual = spec - best_fit(spec)
            rms = np.std(residual)
            flux_mc = np.zeros(nmc)
            mean_mc = np.zeros(nmc)
            disp_mc = np.zeros(nmc)

            for j in range(nmc):
                rand_spec = np.random.randn(len(spec_fit)) * rms + spec_fit
                best_fit_mc = fitter(mod, spec_arr_fit, rand_spec)
                mean_mc[j] = best_fit_mc.mean.value
                disp_mc[j] = best_fit_mc.stddev.value
                flux_mc[j] = best_fit_mc.amplitude.value * np.sqrt(2 * np.pi) * disp_mc[j]

            flux_err[i] = np.nanstd(flux_mc)
            mean_err[i] = np.nanstd(mean_mc)
            disp_err[i] = np.nanstd(disp_mc)

    if estimate_err:
        return ap_centers, np.vstack([flux, flux_err]), np.vstack([mean, mean_err]), np.vstack([disp, disp_err])

    else:
        return ap_centers, flux, mean, disp



def apply_smoothing_2D(vel, disp, smoothing_type=None, smoothing_npix=1):
    if smoothing_type is None:
        return vel, disp
    else:
        if (smoothing_type.lower() == 'median'):
            if (smoothing_npix % 2) == 1:
                vel = sp_sig.medfilt(vel, kernel_size=smoothing_npix)
                disp = sp_sig.medfilt(disp, kernel_size=smoothing_npix)
            else:
                vel = sp_ndi.median_filter(vel, size=smoothing_npix, mode='constant', cval=0.)  # zero-padding edges
                disp = sp_ndi.median_filter(disp, size=smoothing_npix, mode='constant', cval=0.)  # zero-padding edges
        else:
            print("Smoothing type={} not supported".format(smoothing_type))

        return vel, disp


def apply_smoothing_3D(cube, smoothing_type=None, smoothing_npix=1, quiet=True):
    if smoothing_type is None:
        return cube
    else:
        # Parse smoothing type / npix: could be single values, or arrays:
        #       move them into arrays first
        l_st = len(np.shape(smoothing_type))
        l_sp = len(np.shape(smoothing_npix))
        if ((l_st == 0) & (l_sp == 0)):
            # Both single:
            sm_type_arr = [smoothing_type]
            sm_npix_arr = [smoothing_npix]
        elif (l_st == 0):
            # One type, multiple smooth steps:
            sm_npix_arr = smoothing_npix
            sm_type_arr = np.repeat(smoothing_type, len(smoothing_npix))
        elif (l_sp == 0):
            # Mult type, same smooth size for multiple steps:
            sm_type_arr = smoothing_type
            sm_npix_arr = np.repeat(smoothing_npix, len(smoothing_type))
        else:
            if (len(smoothing_type) != len(smoothing_npix)):
                raise ValueError("'smoothing_type' and 'smoothing_npix' are not the same length!")
            sm_type_arr = smoothing_type
            sm_npix_arr = smoothing_npix

        for sm_type, sm_npix in zip(sm_type_arr, sm_npix_arr):
            cb = cube.filled_data[:].value
            if not quiet:
                print("Applying smoothing: {}, {}".format(sm_type, sm_npix))
            if (sm_type.lower() == 'median'):
                # General for both even and odd:
                cb = sp_ndi.median_filter(cb, size=(1,sm_npix, sm_npix), mode='constant', cval=0.)
                ####
                # if (sm_npix % 2) == 1:
                #     cb = sp_sig.medfilt(cb, kernel_size=(1, sm_npix, sm_npix))
                # else:
                #     cb = sp_ndi.median_filter(cb, size=(1,sm_npix, sm_npix), mode='constant', cval=0.)

            elif (sm_type.lower() == 'gaussian'):
                kernel2d = Gaussian2DKernel(x_stddev=sm_npix / std2fwhm)._array
                kernel3d = kernel2d.reshape((1,kernel2d.shape[0],kernel2d.shape[1]))

                cb = apy_convolve_fft(cb, kernel3d)
            else:
                print("Smoothing type={} not supported".format(sm_type))

            # Make new cube:
            cube = cube._new_cube_with(data=cb, wcs=cube.wcs, mask=cube.mask, meta=cube.meta,
                                                              fill_value=cube.fill_value)

        return cube



# The `_rotate_points` and `symmetrize_velfield` functions below are adopted from `cap_symmetrize_velfield.py` within `display_pixels` created by Michele Cappellari (see license information at the top of this file).

def _rotate_points(x, y, ang):
    """
    Rotates points counter-clockwise by an angle ANG-90 in degrees.

    """
    theta = np.radians(ang - 90.)
    xNew = x * np.cos(theta) - y * np.sin(theta)
    yNew = x * np.sin(theta) + y * np.cos(theta)

    return xNew, yNew


# ----------------------------------------------------------------------

def symmetrize_velfield(xbin, ybin, velBin, errBin, sym=2, pa=90.):
    """
    This routine generates a bi-symmetric ('axisymmetric')
    version of a given set of kinematical measurements.
    PA: is the angle in degrees, measured counter-clockwise, from the vertical axis (Y axis) to the galaxy major axis.
    SYM: bi-symmetry: is 1 for (V,h3,h5) and 2 for (sigma,h4,h6)

    """
    xbin, ybin, velBin = map(np.asarray, [xbin, ybin, velBin])

    assert xbin.size == ybin.size == velBin.size, 'The vectors (xbin, ybin, velBin) must have the same size'

    x, y = _rotate_points(xbin, ybin, -pa)  # Negative PA for counter-clockwise

    xyIn = np.column_stack([x, y])
    xout = np.hstack([x, -x, x, -x])
    yout = np.hstack([y, y, -y, -y])
    xyOut = np.column_stack([xout, yout])
    velOut = interpolate.griddata(xyIn, velBin, xyOut)
    velOut = velOut.reshape(4, xbin.size)
    velOut[0, :] = velBin  # see V3.0.1
    if sym == 1:
        velOut[[1, 3], :] *= -1.
    velSym = np.nanmean(velOut, axis=0)

    errOut = interpolate.griddata(xyIn, errBin, xyOut)
    errOut = errOut.reshape(4, xbin.size)
    errOut[0, :] = errBin
    err_count = np.sum(np.isfinite(errOut), axis=0)
    #err_count[err_count == 0] = 1
    errSym = np.sqrt(np.nansum(errOut**2, axis=0))/err_count

    return velSym, errSym

# ----------------------------------------------------------------------

def symmetrize_1D_profile(rin, vin, errin, sym=1):

    whz = np.where(np.abs(rin) == np.abs(rin).min())[0]
    maxval = np.max([np.abs(rin[0]), np.abs(rin[-1])])
    rnum = np.max([2.*(len(rin)-whz[0]-1)+1, 2.*whz[0]+1])
    rout = np.linspace(-maxval, maxval, num=rnum, endpoint=True)


    vinterp = interpolate.interp1d(rin, vin, kind='cubic', bounds_error=False, fill_value=np.NaN)
    errinterp = interpolate.interp1d(rin, errin, kind='cubic', bounds_error=False, fill_value=np.NaN)

    if sym == 1:
        symm_fac = -1.
    elif sym == 2:
        symm_fac = 1.


    vint = vinterp(rout)
    errint = errinterp(rout)


    velOut = np.vstack([vint, symm_fac*vint[::-1]])
    errOut = np.vstack([errint, errint[::-1]])

    vsym = np.nanmean(velOut, axis=0)

    err_count = np.sum(np.isfinite(errOut), axis=0)
    err_count[err_count == 0] = 1
    errsym = np.sqrt(np.nansum(errOut**2, axis=0))/err_count

    return rout, vsym, errsym


def fit_truncated_gaussian(trace, truncate_value):
    """
    Find the best fitting mean and standard deviation of a sample assuming the distribution
    is a truncated Gaussian with the truncation occurring at low values.

    :param trace: Sample to fit
    :param truncate_value: The value for where the truncation occurs
    :return: mean, standard deviation of the best fitting truncated Gaussian
    """

    # First adjust so that the truncation occurs at 0
    trace_adjust = trace - truncate_value

    mean_guess = np.mean(trace_adjust)
    std_guess = np.std(trace_adjust)
    p = minimize(lnlike_truncnorm, np.array([mean_guess, std_guess]),
                 method='Nelder-Mead', args=(trace_adjust))
    mean_fit = p.x[0] + truncate_value
    std_fit = p.x[1]

    return mean_fit, std_fit


def lnlike_truncnorm(params, x):
    # Function for the negative log-likelihood of a truncated Gaussian
    return -np.sum(np.log(norm.pdf(x, loc=params[0], scale=params[1])) - np.log(1.0 - norm.cdf(0, loc=params[0], scale=params[1])))


def fit_uncertainty_ellipse(chain_x, chain_y, bins=50):
    r"""
    Get the uncertainty ellipse of the sample for each photocenter.
    (Modified from code from Jinyi Shangguan to do photocenter uncertainty ellipses)

    Parameters
    ----------
    pos_list: 2D array
        Sampler chains for photocenter positions
    bins: integer
        The number of bins to use in the 2D histogram

    Returns
    -------
    PA: float
        angle of ellipse, in degrees
    stddev_x: float
        stddev of the "x" axis of the 2D gaussian;
        double to get the full "width" of a 1sig ellipse for matplotlib.Ellipse
    stddev_y: float
        stddev of the "y" axis of the 2D gaussian

    """

    nSamp = len(chain_x)

    chainvals = np.stack([chain_x.T,chain_y.T], axis=1)
    # shape: nSamp, 2

    pmean, pmed, pstd = apy_stats.sigma_clipped_stats(chainvals, axis=0)

    # -> Shift the position to center at zero.
    valshift = chainvals - pmed

    # -> Get the 2D histogram and fit with a 2D Gaussian
    # use +- 2 FWHM in either direction
    range = [[-4.7*pstd[0], 4.7*pstd[0]], [-4.7*pstd[1], 4.7*pstd[1]]]
    p_2dh, px_edge, py_edge = np.histogram2d(valshift[:, 0], valshift[:, 1], bins=bins, range=range)

    px_cnt = (px_edge[1:] + px_edge[:-1]) / 2.
    py_cnt = (py_edge[1:] + py_edge[:-1]) / 2.
    pxx_cnt, pyy_cnt = np.meshgrid(px_cnt, py_cnt)
    m_init = apy_mod.models.Gaussian2D(amplitude=np.max(p_2dh), x_mean=0, y_mean=0,
                               x_stddev=pstd[0], y_stddev=pstd[1], theta=0.)
    fit_m = apy_mod.fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        m = fit_m(m_init, pxx_cnt, pyy_cnt, p_2dh)
    # pars = m.parameters

    PA =        m.theta.value * 180./np.pi
    stddev_x =  m.x_stddev.value
    stddev_y =  m.y_stddev.value

    # Map values onto a "common" frame:
    if stddev_x > stddev_y:
        PA += 90.
        stddev_y =  m.x_stddev.value
        stddev_x =  m.y_stddev.value

    # Map to 0, 180:
    PA = PA % 360

    if PA > 180.:
        PA -= 180.
    # if PA < 0:
    #     PA += 180.

    return PA, stddev_x, stddev_y





###########################################################################################
# Faster gaussian fitting to extract from 3D model cube:
#   directly go to scipy.optimize.leastsq,
#   instead of using AstroPy fitting.LevMarLSQFitter and models.Gaussian1D

def _gaus_resid(coeffs, x, y):
    return coeffs[0]*np.exp(-((x-coeffs[1])**2/(2.*coeffs[2]**2))) - y

def _gaus_dfunc(coeffs, x, *args):
    d_amp = np.exp(-0.5 * (x-coeffs[1])**2 / coeffs[2]**2 )
    d_mean = coeffs[0] * d_amp * (x-coeffs[1]) / coeffs[2]**2
    d_stddev = coeffs[0] * d_amp * (x-coeffs[1])**2 / coeffs[2]**3

    return np.array([d_amp,d_mean,d_stddev])

def gaus_fit_sp_opt_leastsq(x, y, flux_guess, mean_guess, stddev_guess):
    # fitparams = gaus_fit_sp_opt_leastsq(x, y, flux_guess, mean_guess, stddev_guess)
    # x, y: arrays, same length (dep/indep variables)
    # [amp/mean/stddev]_guess: floats containing pre-computed moments from y.

    init_values = np.array([flux_guess/np.sqrt(2*np.pi*(stddev_guess**2)), mean_guess, np.abs(stddev_guess)])

    fitparams, flags = leastsq(_gaus_resid, init_values, args=(x, y), Dfun=_gaus_dfunc, col_deriv=True)

    # fitparams: amp, mean, stddev
    return fitparams


def gaus_fit_apy_mod_fitter(x, y, flux_guess, mean_guess, stddev_guess, yerr=None):
    # fitparams = gaus_fit_apy_mod_fitter(x, y, amp_guess, mean_guess, stddev_guess)
    # x, y: arrays, same length (dep/indep variables)
    # [amp/mean/stddev]_guess: floats containing pre-computed moments from y.

    # OLD: astropy fitter
    mod = apy_mod.models.Gaussian1D(amplitude=flux_guess / (np.sqrt(2 * np.pi) * np.abs(stddev_guess)),
                                    mean=mean_guess,
                                    stddev=np.abs(stddev_guess))
    mod.amplitude.bounds = (0, None)
    mod.stddev.bounds = (0, None)
    fitter = apy_mod.fitting.LevMarLSQFitter()
    if yerr is not None:
        wgt = 1./yerr
        wgt[~np.isfinite(wgt)] = 0.
        #wgt /= wgt.max()
    else:
        wgt = None
    best_fit = fitter(mod, x, y, weights=wgt)

    # fitparams: amp, mean, stddev
    #return [best_fit.amplitude.value, best_fit.mean.value, best_fit.stddev.value]
    return [best_fit.amplitude.value, best_fit.mean.value, best_fit.stddev.value, best_fit]

###########################################################################################

def get_cin_cout(shape, asint=False):

    if asint:
        carr = np.zeros(len(shape), dtype=np.int)
    else:
        carr = np.zeros(len(shape))

    for j,sh in enumerate(shape):
        if sh % 2 == 1:
            ca = 0.5*(sh-1)
        else:
            ca = 0.5*sh
        if asint:
            carr[j] = np.int(np.round(ca))
        else:
            carr[j] = ca

    return tuple(carr)



#########################
def _check_data_inst_FOV_compatibility(gal):
    logger_msg = None
    for obs_name in gal.observations:
        obs = gal.observations[obs_name]
        if obs.fit_options.fit & (obs.data.ndim == 1):
                if min(obs.instrument.fov)/2. <  np.abs(obs.data.rarr).max() / obs.instrument.pixscale.value:
                    if logger_msg is None:
                        logger_msg = ""
                    else:
                        logger_msg += "\n"
                    logger_msg += "obs={}: FOV smaller than the maximum data extent!\n".format(obs.name)
                    logger_msg += "                FOV=[{},{}] pix; max(abs(data.rarr))={} pix".format(obs.instrument.fov[0],
                                    obs.instrument.fov[1], np.abs(obs.data.rarr).max()/ obs.instrument.pixscale)

    if logger_msg is not None:
        logger.warning(logger_msg)

    return None

def _set_instrument_kernels(gal):
    # Pre-calculate instrument kernels:
    for obs_name in gal.observations:
        obs = gal.observations[obs_name]
        if obs.instrument._beam_kernel is None:
            obs.instrument.set_beam_kernel()
        if obs.instrument._lsf_kernel is None and obs.instrument.lsf is not None:
            obs.instrument.set_lsf_kernel(spec_center=obs.instrument.line_center)

    return gal


def citations():
    """
    Return the papers that should be cited when using DYSMALPY
    """

    str = "Please cite the following papers if using DYSMALPY for a publication:\n"
    str += "-----------------------------------------\n"
    str += "Davies et al. (2004a): https://ui.adsabs.harvard.edu/abs/2004ApJ...602..148D\n"
    str += "Davies et al. (2004b): https://ui.adsabs.harvard.edu/abs/2004ApJ...613..781D\n"
    str += "Cresci et al. (2009): https://ui.adsabs.harvard.edu/abs/2009ApJ...697..115C\n"
    str += "Davies et al. (2011): https://ui.adsabs.harvard.edu/abs/2011ApJ...741...69D\n"
    str += "Wuyts et al. (2016): https://ui.adsabs.harvard.edu/abs/2016ApJ...831..149W\n"
    str += "Lang et al. (2017): https://ui.adsabs.harvard.edu/abs/2017ApJ...840...92L\n"
    str += "Price et al. (2021): https://ui.adsabs.harvard.edu/abs/2021ApJ...922..143P\n"
    str += "Lee et al. (2024): in preparation"

    return str
