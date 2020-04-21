# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Module containing some useful utility functions

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library

# Third party imports
import numpy as np
import astropy.modeling as apy_mod
import astropy.units as u
import matplotlib.pyplot as plt
import scipy.signal as sp_sig
import scipy.ndimage as sp_ndi
from scipy import interpolate
from scipy.optimize import minimize
from scipy.stats import norm

from .data_classes import Data1D, Data2D
from spectral_cube import SpectralCube, BooleanArrayMask


def calc_pixel_distance(nx, ny, center_coord):
    """
    Function to calculate the distance of each pixel
    from a specific pixel as well as the position angle
    :param nx:
    :param ny:
    :param center_coord:
    :return:
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

    :param nx:
    :param ny:
    :param center_coord:
    :param dr:
    :return:
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
    :param nx:
    :param ny:
    :param center_coord:
    :param pa:
    :param dr:
    :return:
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
    a line with position angle, pa
    :param r:
    :param pa:
    :param xcenter:
    :param ycenter:
    :return:
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
    :param cube: Cube to measure the 1D profile on
    :param rap: Radius of the circular apertures in pixels
    :param dr: Distance between the circular apertures in pixels
    :param center_pixel: Central pixel that defines r = 0
    :param pa: Position angle of the line that the circular apertures lie on
    :param spec_arr: The spectral array (i.e. wavelengths, frequencies, velocities, etc)
    :param spec_mask: Boolean mask to apply to the spectrum to exclude from fitting
    :param estimate_err: True or False to use Monte Carlo to estimate the errors on the fits
    :param nmc: The number of trials in the Monte Carlo analysis to use.
    :returns: centers: The radial offset from the central pixel in pixels
    :returns: flux: The integrated best fit "flux" for each aperture
    :returns: mean: The best fit mean of the Gaussian profile in the same units as spec_arr
    :returns: disp: The best fit dispersion of the Gaussian profile in the same units as spec_arr

    Note: flux, mean, and disp will be Nap x 2 arrays if estimate_err = True where Nap is the number
          of apertures that are fit. The first row will be best fit values and the second row will
          contain the errors on those parameters.
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

        mod = apy_mod.models.Gaussian1D(amplitude=mom0 / np.sqrt(2 * np.pi * np.abs(mom2)),
                                        mean=mom1,
                                        stddev=np.sqrt(np.abs(mom2)))
        mod.amplitude.bounds = (0, None)
        mod.stddev.bounds = (0, None)
        fitter = apy_mod.fitting.LevMarLSQFitter()
        best_fit = fitter(mod, spec_arr_fit, spec_fit)

        mean[i] = best_fit.mean.value
        disp[i] = best_fit.stddev.value
        flux[i] = best_fit.amplitude.value * np.sqrt(2 * np.pi) * disp[i]

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


def apply_smoothing_3D(cube, smoothing_type=None, smoothing_npix=1):
    if smoothing_type is None:
        return cube
    else:
        if (smoothing_type.lower() == 'median'):
            #cube = sp_sig.medfilt(cube, kernel_size=(1, smoothing_npix, smoothing_npix))
            cb = cube.filled_data[:].value
            if (smoothing_npix % 2) == 1:
                cb = sp_sig.medfilt(cb, kernel_size=(1, smoothing_npix, smoothing_npix))
            else:
                cb = sp_ndi.median_filter(cb, size=(1,smoothing_npix, smoothing_npix), mode='constant', cval=0.)
                
            cube = cube._new_cube_with(data=cb, wcs=cube.wcs,
                                              mask=cube.mask, meta=cube.meta,
                                              fill_value=cube.fill_value)
            #cube = cube.spatial_smooth_median(smoothing_npix)
            
        else:
            print("Smoothing type={} not supported".format(smoothing_type))

        return cube



# _rotate_points and symmetrize_velfield from cap_symmetrize_velfield.py within display_pixels
# package created by Michele Cappelari.

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
    PA: is the angle in degrees, measured counter-clockwise,
      from the vertical axis (Y axis) to the galaxy major axis.
    SYM: by-simmetry: is 1 for (V,h3,h5) and 2 for (sigma,h4,h6)

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

