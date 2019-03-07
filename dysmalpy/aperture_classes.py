# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# File containing all of the available classes to define aperture for 
# extracting 1D spectra for a galaxy/model. 

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard imports
import logging

# Third party imports
import numpy as np
import astropy.modeling as apy_mod
import astropy.units as u

from .utils import calc_pixel_distance

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


__all__ = [ "Aperture", "EllipAperture", "RectAperture", "Apertures", 
            "EllipApertures", "CircApertures", "RectApertures", "SquareApertures" ]

deg2rad = np.pi / 180.


# Base Class for a data container
class Aperture(object):
    """
    Generic case.
    aper_center, center_pixel should be in *Pixel* units, and in x,y coordinates 
    ####Shape center_pixel: 2 ([x, y])
    Shape aper_center: 2 ( [x0, y0] )
    """

    def __init__(self, aper_center=None, nx=None, ny=None):
                 
        self.aper_center = aper_center
        self.nx = nx
        self.ny = ny
        
    def define_aperture_mask(self):
        mask = np.ones((self.ny, self.nx), dtype=np.bool)
        mask[np.int(self.aper_center[1]), np.int(self.aper_center[0])] = True
        return mask
        
    def extract_aper_kin(self, spec_arr=None, 
            cube=None, err=None, mask=None, spec_mask=None):
        """
        spec_array: the spectral direction array -- eg, vel array or wave array.
        """
        
        mask_ap = self.define_aperture_mask()
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

        vel1d = best_fit.mean.value
        disp1d = best_fit.stddev.value
        flux1d = best_fit.amplitude.value * np.sqrt(2 * np.pi) * disp1d
        
        return flux1d, vel1d, disp1d
        

class EllipAperture(Aperture):
    """
    Note: slit_PA is CCW from north / up (sky "y" direction). In Degrees!
    """

    def __init__(self, slit_PA=None, pix_perp=None, pix_parallel=None,
            aper_center=None, nx=None, ny=None):

        # set things here
        self.pix_perp = pix_perp
        self.pix_parallel = pix_parallel
        self.slit_PA = slit_PA


        super(EllipAperture, self).__init__(aper_center=aper_center, 
                                            nx=nx, ny=ny)


    def define_aperture_mask(self):
        seps_sky, pa_sky = calc_pixel_distance(self.nx, self.ny, self.aper_center)

        xskys = seps_sky * -1 * np.sin(pa_sky*deg2rad)
        yskys = seps_sky * np.cos(pa_sky*deg2rad)

        xslits = xskys * np.cos(self.slit_PA*deg2rad)       + yskys * np.sin(self.slit_PA*deg2rad)
        yslits = xskys * -1. * np.sin(self.slit_PA*deg2rad) + yskys * np.cos(self.slit_PA*deg2rad)

        apmask = ( (yslits/self.pix_parallel)**2 + (xslits/self.pix_perp)**2 <= 1. )

        return apmask
        
#
class RectAperture(Aperture):
    """
    Note: slit_PA is CCW from north / up (sky "y" direction). In Degrees!
    """

    def __init__(self, slit_PA=None, pix_perp=None, pix_parallel=None,
            aper_center=None, nx=None, ny=None):

        # set things here
        self.pix_perp = pix_perp
        self.pix_parallel = pix_parallel
        self.slit_PA = slit_PA


        super(RectAperture, self).__init__(aper_center=aper_center, 
                                            nx=nx, ny=ny)


    def define_aperture_mask(self):
        seps_sky, pa_sky = calc_pixel_distance(self.nx, self.ny, self.aper_center)

        xskys = seps_sky * -1 * np.sin(pa_sky*deg2rad)
        yskys = seps_sky * np.cos(pa_sky*deg2rad)

        xslits = xskys * np.cos(self.slit_PA*deg2rad)       + yskys * np.sin(self.slit_PA*deg2rad)
        yslits = xskys * -1. * np.sin(self.slit_PA*deg2rad) + yskys * np.cos(self.slit_PA*deg2rad)

        apmask = ( (xslits <= self.pix_perp/2.) & (yslits <= self.pix_parallel/2.) )

        return apmask


        
class Apertures(object):
    """
    Generic case. Should be array of Aperture objects. Needs the loop.
    """
    def __init__(self, apertures=None, slit_PA=None):
        self.apertures = apertures
        self.slit_PA = slit_PA
        
    
    def extract_1d_kinematics(self, spec_arr=None, 
                cube=None, err=None, mask=None, spec_mask=None, 
                center_pixel = None, pixscale=None):
        """
        aper_centers_pixout: the radial direction positions, relative to kin center, in pixels
        """
        ny = cube.shape[1]
        nx = cube.shape[2]
        
        # Assume the default central pixel is the center of the cube
        if center_pixel is None:
            center_pixel = ((nx - 1) / 2., (ny - 1) / 2.)
        
        naps = len(self.apertures)
        aper_centers_pixout = np.zeros(naps)
        flux1d = np.zeros(naps)
        vel1d = np.zeros(naps)
        disp1d = np.zeros(naps)
        
        for i in range(naps):
            flux1d[i], vel1d[i], disp1d[i] = self.apertures[i].extract_aper_kin(spec_arr=spec_arr, 
                    cube=cube, err=err, mask=mask, spec_mask=spec_mask)
            aper_centers_pixout[i] = (np.sqrt((self.apertures[i].aper_center[0]-center_pixel[0])**2 +
                       (self.apertures[i].aper_center[1]-center_pixel[1])**2 ) *
                       np.sign(-np.sin((self.slit_PA+90.)*deg2rad)*(self.apertures[i].aper_center[1]-center_pixel[1])))

        
        return aper_centers_pixout*pixscale, flux1d, vel1d, disp1d


#
class EllipApertures(Apertures):
    """
    Generic case. Should be array of Aperture objects. Needs the loop.
    Uses same generic extract_1d_kinematics as Apertures.
    Sizes can vary. -- depending on if pix_perp and pix_parallel are arrays or scalar.
    
    FOR THIS CASE: aper_centers are along the slit.
    
    rarr should be in *** ARCSEC ***
    
    """
    def __init__(self, rarr=None, slit_PA=None, pix_perp=None, pix_parallel=None,
             nx=None, ny=None, center_pixel=None, pixscale=None):
        
        #aper_center_pix_shift = None
        
        # Assume the default central pixel is the center of the cube
        if center_pixel is None:
            center_pixel = ((nx - 1) / 2., (ny - 1) / 2.)
        
        try: 
            if len(pix_perp) > 1:
                pix_perp = np.array(pix_perp)
            else:
                pix_perp = np.repeat(pix_perp[0], len(rarr))
        except:
            pix_perp = np.repeat(pix_perp, len(rarr))
            
        try: 
            if len(pix_parallel) > 1:
                pix_parallel = np.array(pix_parallel)
            else:
                pix_parallel = np.repeat(pix_parallel[0], len(rarr))
        except:
            pix_parallel = np.repeat(pix_parallel, len(rarr))
            
        self.pix_perp = pix_perp
        self.pix_parallel = pix_parallel
        self.rarr = rarr
        self.nx = nx
        self.ny = ny
        self.center_pixel = center_pixel
        self.pixscale = pixscale
        
        aper_centers_pix = np.zeros((2,len(self.rarr)))
        apertures = []
        for i in range(len(rarr)):
            aper_cent_pix = [rarr[i]*np.sin(slit_PA*deg2rad)/self.pixscale + self.center_pixel[0],
                            rarr[i]*-1.*np.cos(slit_PA*deg2rad)/self.pixscale + self.center_pixel[1]]
            aper_centers_pix[:,i] = aper_cent_pix
            apertures.append(EllipAperture(slit_PA=slit_PA,
                        pix_perp=self.pix_perp[i], pix_parallel=self.pix_parallel[i],
                        aper_center=aper_cent_pix, nx=self.nx, ny=self.ny))
        
        self.aper_centers_pix = aper_centers_pix
        
        super(EllipApertures, self).__init__(apertures=apertures, slit_PA=slit_PA)
    
    
class CircApertures(EllipApertures):
    def __init__(self, rarr=None, slit_PA=None, rpix=None, 
             nx=None, ny=None, center_pixel=None, pixscale=None):
             
        super(CircApertures, self).__init__(rarr=rarr, slit_PA=slit_PA, 
                pix_perp=rpix, pix_parallel=rpix, nx=nx, ny=ny, 
                center_pixel=center_pixel, pixscale=pixscale)
    
#
class RectApertures(Apertures):
    """
    Generic case. Should be array of Aperture objects. Needs the loop.
    Uses same generic extract_1d_kinematics as Apertures.
    Sizes can vary. -- depending on if pix_perp and pix_parallel are arrays or scalar.
    
    FOR THIS CASE: aper_centers are along the slit.
    
    rarr should be in *** ARCSEC ***
    
    Note here that pix_perp and pix_parallel are the *WIDTHS* of the rectangular apertures
    
    """
    def __init__(self, rarr=None, slit_PA=None, pix_perp=None, pix_parallel=None,
             nx=None, ny=None, center_pixel=None, pixscale=None):
        
        #aper_center_pix_shift = None
        
        # Assume the default central pixel is the center of the cube
        if center_pixel is None:
            center_pixel = ((nx - 1) / 2., (ny - 1) / 2.)
        
        try: 
            if len(pix_perp) > 1:
                pix_perp = np.array(pix_perp)
            else:
                pix_perp = np.repeat(pix_perp[0], len(rarr))
        except:
            pix_perp = np.repeat(pix_perp, len(rarr))
            
        try: 
            if len(pix_parallel) > 1:
                pix_parallel = np.array(pix_parallel)
            else:
                pix_parallel = np.repeat(pix_parallel[0], len(rarr))
        except:
            pix_parallel = np.repeat(pix_parallel, len(rarr))
            
        self.pix_perp = pix_perp
        self.pix_parallel = pix_parallel
        self.rarr = rarr
        self.nx = nx
        self.ny = ny
        self.center_pixel = center_pixel
        self.pixscale = pixscale
        
        aper_centers_pix = np.zeros((2,len(self.rarr)))
        apertures = []
        for i in range(len(rarr)):
            aper_cent_pix = [rarr[i]*np.sin(slit_PA*deg2rad)/self.pixscale + self.center_pixel[0],
                            rarr[i]*-1.*np.cos(slit_PA*deg2rad)/self.pixscale + self.center_pixel[1]]
            aper_centers_pix[:,i] = aper_cent_pix
            apertures.append(RectAperture(slit_PA=slit_PA,
                        pix_perp=self.pix_perp[i], pix_parallel=self.pix_parallel[i],
                        aper_center=aper_cent_pix, nx=self.nx, ny=self.ny))
        
        self.aper_centers_pix = aper_centers_pix
        
        super(RectApertures, self).__init__(apertures=apertures, slit_PA=slit_PA)

class SquareApertures(RectApertures):
    """
    Note here that pix_perp and pix_parallel are the *WIDTHS* of the rectangular apertures
    
    """
    def __init__(self, rarr=None, slit_PA=None, pix_length=None, 
             nx=None, ny=None, center_pixel=None, pixscale=None):
             
        super(SquareApertures, self).__init__(rarr=rarr, slit_PA=slit_PA, 
                pix_perp=pix_length, pix_parallel=pix_length, nx=nx, ny=ny, 
                center_pixel=center_pixel, pixscale=pixscale)
                
                
                
                
                