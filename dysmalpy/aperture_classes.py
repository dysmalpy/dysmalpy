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

try:
    from shapely.geometry import Polygon, Point
    from shapely import affinity as shply_affinity
    shapely_installed = True
except:
    # print("*******************************************")
    # print("* Python package 'shapely' not installed. *")
    # print("*******************************************")
    shapely_installed = False


# Base Class for a data container
class Aperture(object):
    """
    Generic case.
    aper_center, center_pixel should be in *Pixel* units, and in x,y coordinates 
    ####Shape center_pixel: 2 ([x, y])
    Shape aper_center: 2 ( [x0, y0] )
    """

    def __init__(self, aper_center=None, nx=None, ny=None, partial_weight=True,
                moment=False):
                 
        self.aper_center = aper_center
        self.nx = nx
        self.ny = ny
        self.partial_weight = partial_weight
        self.moment = moment
        
        # Setup mask if using partial_weight (partial pixels)
        if self.partial_weight:
            self._mask_ap = self.define_aperture_mask()
        else:
            self._mask_ap = None
        
    def define_aperture_mask(self):
        mask = np.ones((self.ny, self.nx), dtype=np.bool)
        mask[np.int(self.aper_center[1]), np.int(self.aper_center[0])] = True
        return mask
        
    def extract_aper_kin(self, spec_arr=None, 
            cube=None, err=None, mask=None, spec_mask=None):
        """
        spec_array: the spectral direction array -- eg, vel array or wave array.
        """
        # try:
        #     if self.partial_weight:
        #         if self._mask_ap is None:
        #             mask_ap = self.define_aperture_mask()
        #             self._mask_ap = mask_ap
        #         else:
        #             mask_ap = self._mask_ap
        #     else:
        #         mask_ap = self.define_aperture_mask()
        # except:
        #     mask_ap = self.define_aperture_mask()
        if hasattr(self, 'partial_weight'):
            if self.partial_weight:
                mask_ap = self._mask_ap
            else:
                mask_ap = self.define_aperture_mask()
        else:
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
        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Catch case where mixing old + new w/o moment defined
        if hasattr(self, 'moment'):
            moment_calc = self.moment
        else:
            moment_calc = False
        #if self.moment:
        if moment_calc:
            flux1d = mom0
            vel1d = mom1
            disp1d = np.sqrt(np.abs(mom2))
        else:
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
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        return flux1d, vel1d, disp1d
         
class EllipAperture(Aperture):
    """
    Note: slit_PA is CCW from north / up (sky "y" direction). In Degrees!
    """

    def __init__(self, slit_PA=None, pix_perp=None, pix_parallel=None,
            aper_center=None, nx=None, ny=None, partial_weight=True,
            moment=False):

        # set things here
        self.pix_perp = pix_perp
        self.pix_parallel = pix_parallel
        self.slit_PA = slit_PA


        super(EllipAperture, self).__init__(aper_center=aper_center, 
                                            nx=nx, ny=ny, partial_weight=partial_weight,
                                            moment=moment)


    def define_aperture_mask(self):
        seps_sky, pa_sky = calc_pixel_distance(self.nx, self.ny, self.aper_center)

        xskys = seps_sky * -1 * np.sin(pa_sky*deg2rad)
        yskys = seps_sky * np.cos(pa_sky*deg2rad)

        xslits = xskys * np.cos(self.slit_PA*deg2rad)       + yskys * np.sin(self.slit_PA*deg2rad)
        yslits = xskys * -1. * np.sin(self.slit_PA*deg2rad) + yskys * np.cos(self.slit_PA*deg2rad)
        
        try:
            if self.partial_weight:
                do_partial_weight = True
            else:
                do_partial_weight = False
        except:
            do_partial_weight = False
            
        if do_partial_weight:
            if shapely_installed:
                apmask = np.ones(xskys.shape) * -99.
                
                wh_allin = np.where( (np.abs(yslits)/self.pix_parallel + 1./np.sqrt(2.))**2 + \
                                     (np.abs(xslits)/self.pix_perp + 1./np.sqrt(2.))**2 <= 1. )
                wh_allout = np.where( (np.abs(yslits)/self.pix_parallel - 1./np.sqrt(2.))**2 + \
                                      (np.abs(xslits)/self.pix_perp - 1./np.sqrt(2.))**2 > 1. )
                
                apmask[wh_allin] = 1.
                apmask[wh_allout] = 0.
                
                wh_calc = np.where(apmask < 0)
                
                # define shapely object:
                circtmp = Point((0,0)).buffer(1)
                aper_ell = shply_affinity.scale(circtmp, int(self.pix_perp), int(self.pix_parallel))
                
                
                for (whc_y, whc_x) in zip(*wh_calc):
                    x_sky_pix_corners = np.array([xskys[whc_y,whc_x]+0.5, xskys[whc_y,whc_x]+0.5, 
                                                  xskys[whc_y,whc_x]-0.5, xskys[whc_y,whc_x]-0.5] )
                    y_sky_pix_corners = np.array([yskys[whc_y,whc_x]+0.5, yskys[whc_y,whc_x]-0.5, 
                                                  yskys[whc_y,whc_x]-0.5, yskys[whc_y,whc_x]+0.5] )
                    x_slit_pix_corners = x_sky_pix_corners * np.cos(self.slit_PA*deg2rad)       + y_sky_pix_corners * np.sin(self.slit_PA*deg2rad)
                    y_slit_pix_corners = x_sky_pix_corners * -1. * np.sin(self.slit_PA*deg2rad) + y_sky_pix_corners * np.cos(self.slit_PA*deg2rad)
                    
                    cornerspix = np.array([x_slit_pix_corners, y_slit_pix_corners])
                    pix_poly = Polygon(cornerspix.T)
                
                    overlap = aper_ell.intersection(pix_poly)
                    apmask[whc_y,whc_x] = overlap.area
            else:
                raise ValueError("Currently cannot do partial weights if python package shapely is not installed")
                apmask = None # fractional pixels
        else:
            apmask = ( (yslits/self.pix_parallel)**2 + (xslits/self.pix_perp)**2 <= 1. )

        return apmask
        
#
class RectAperture(Aperture):
    """
    Note: slit_PA is CCW from north / up (sky "y" direction). In Degrees!
    """

    def __init__(self, slit_PA=None, pix_perp=None, pix_parallel=None,
            aper_center=None, nx=None, ny=None, partial_weight=True,
            moment=False):

        # set things here
        self.pix_perp = pix_perp
        self.pix_parallel = pix_parallel
        self.slit_PA = slit_PA


        super(RectAperture, self).__init__(aper_center=aper_center, 
                                            nx=nx, ny=ny, partial_weight=partial_weight,
                                            moment=moment)


    def define_aperture_mask(self):
        seps_sky, pa_sky = calc_pixel_distance(self.nx, self.ny, self.aper_center)

        xskys = seps_sky * -1 * np.sin(pa_sky*deg2rad)
        yskys = seps_sky * np.cos(pa_sky*deg2rad)

        xslits = xskys * np.cos(self.slit_PA*deg2rad)       + yskys * np.sin(self.slit_PA*deg2rad)
        yslits = xskys * -1. * np.sin(self.slit_PA*deg2rad) + yskys * np.cos(self.slit_PA*deg2rad)
        
        try:
            if self.partial_weight:
                do_partial_weight = True
            else:
                do_partial_weight = False
        except:
            do_partial_weight = False
            
        if do_partial_weight:
            if shapely_installed:
                apmask = np.ones(xskys.shape) * -99.
                
                wh_allin = np.where(( (np.abs(xslits) <= self.pix_perp/2. - 1./np.sqrt(2.)) & \
                                (np.abs(yslits) <= self.pix_parallel/2.- 1./np.sqrt(2.)) ))
                #
                wh_allout = np.where(( (np.abs(xslits) > self.pix_perp/2. + 1./np.sqrt(2.)) | \
                                (np.abs(yslits) > self.pix_parallel/2.+ 1./np.sqrt(2.)) ))
                
                apmask[wh_allin] = 1.
                apmask[wh_allout] = 0.
                
                # define shapely object:
                corners = np.array([[self.pix_perp/2., self.pix_perp/2., 
                                    -self.pix_perp/2., -self.pix_perp/2.], 
                                    [self.pix_parallel/2., -self.pix_parallel/2., 
                                    -self.pix_parallel/2., self.pix_parallel/2.]])
                aper_poly = Polygon(corners.T)
                
                wh_calc = np.where(apmask < 0)
                
                for (whc_y, whc_x) in zip(*wh_calc):
                    x_sky_pix_corners = np.array([xskys[whc_y,whc_x]+0.5, xskys[whc_y,whc_x]+0.5, 
                                                  xskys[whc_y,whc_x]-0.5, xskys[whc_y,whc_x]-0.5] )
                    y_sky_pix_corners = np.array([yskys[whc_y,whc_x]+0.5, yskys[whc_y,whc_x]-0.5, 
                                                  yskys[whc_y,whc_x]-0.5, yskys[whc_y,whc_x]+0.5] )
                    x_slit_pix_corners = x_sky_pix_corners * np.cos(self.slit_PA*deg2rad)       + y_sky_pix_corners * np.sin(self.slit_PA*deg2rad)
                    y_slit_pix_corners = x_sky_pix_corners * -1. * np.sin(self.slit_PA*deg2rad) + y_sky_pix_corners * np.cos(self.slit_PA*deg2rad)
                    
                    cornerspix = np.array([x_slit_pix_corners, y_slit_pix_corners])
                    pix_poly = Polygon(cornerspix.T)
                    
                    overlap = aper_poly.intersection(pix_poly)
                    apmask[whc_y,whc_x] = overlap.area
                    
            else:
                raise ValueError("Currently cannot do partial weights if python package shapely is not installed")
                apmask = None # fractional pixels
        else:
            apmask = ( (np.abs(xslits) <= self.pix_perp/2.) & (np.abs(yslits) <= self.pix_parallel/2.) )
            
        return apmask


        
class Apertures(object):
    """
    Generic case. Should be array of Aperture objects. Needs the loop.
    """
    def __init__(self, apertures=None, slit_PA=None, rotate_cube=False):
        self.apertures = apertures
        self.slit_PA = slit_PA
        self.slit_PA_unrotated = slit_PA
        self.rotate_cube = rotate_cube
    
    def extract_1d_kinematics(self, spec_arr=None, 
                cube=None, err=None, mask=None, spec_mask=None, 
                center_pixel = None, pixscale=None):
        """
        aper_centers_pixout: the radial direction positions, relative to kin center, in pixels
        """
        # +++++++++++++++++++++++++++++++++++++++++++++
        ## If rotate cube, implement here:
        # self.slit_PA = 0.
        #
        # rotate by self.slit_PA_unrotated
        # +++++++++++++++++++++++++++++++++++++++++++++
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
                    
            if (self.apertures[i].aper_center[1] != center_pixel[1]):
                aper_centers_pixout[i] = (np.sqrt((self.apertures[i].aper_center[0]-center_pixel[0])**2 +
                           (self.apertures[i].aper_center[1]-center_pixel[1])**2 ) *
                           np.sign(-np.sin((self.slit_PA+90.)*deg2rad)*(self.apertures[i].aper_center[1]-center_pixel[1])))
            else:
                aper_centers_pixout[i] = (np.sqrt((self.apertures[i].aper_center[0]-center_pixel[0])**2 +
                           (self.apertures[i].aper_center[1]-center_pixel[1])**2 ) *
                           np.sign(-np.cos((self.slit_PA+90.)*deg2rad)*(self.apertures[i].aper_center[0]-center_pixel[0])))
                           
                           
        
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
             nx=None, ny=None, center_pixel=None, pixscale=None, partial_weight=True, rotate_cube=False,
             moment=False):
        
        #
        if rotate_cube:
            slit_PA_use = 0.
        else:
            slit_PA_use = slit_PA
        
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
            aper_cent_pix = [rarr[i]*np.sin(slit_PA_use*deg2rad)/self.pixscale + self.center_pixel[0],
                            rarr[i]*-1.*np.cos(slit_PA_use*deg2rad)/self.pixscale + self.center_pixel[1]]
            aper_centers_pix[:,i] = aper_cent_pix
            apertures.append(EllipAperture(slit_PA=slit_PA_use,
                        pix_perp=self.pix_perp[i], pix_parallel=self.pix_parallel[i],
                        aper_center=aper_cent_pix, nx=self.nx, ny=self.ny, partial_weight=partial_weight,
                        moment=moment))
        
        self.aper_centers_pix = aper_centers_pix
        
        super(EllipApertures, self).__init__(apertures=apertures, slit_PA=slit_PA, rotate_cube=rotate_cube)
    
    
class CircApertures(EllipApertures):
    def __init__(self, rarr=None, slit_PA=None, rpix=None, 
             nx=None, ny=None, center_pixel=None, pixscale=None, partial_weight=True, rotate_cube=False,
             moment=False):
             
        super(CircApertures, self).__init__(rarr=rarr, slit_PA=slit_PA, 
                pix_perp=rpix, pix_parallel=rpix, nx=nx, ny=ny, 
                center_pixel=center_pixel, pixscale=pixscale, partial_weight=partial_weight, rotate_cube=rotate_cube,
                moment=moment)
    
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
             nx=None, ny=None, center_pixel=None, pixscale=None, partial_weight=True, rotate_cube=False,
             moment=False):
        #
        if rotate_cube:
            slit_PA_use = 0.
        else:
            slit_PA_use = slit_PA
            
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
            aper_cent_pix = [rarr[i]*np.sin(slit_PA_use*deg2rad)/self.pixscale + self.center_pixel[0],
                            rarr[i]*-1.*np.cos(slit_PA_use*deg2rad)/self.pixscale + self.center_pixel[1]]
            aper_centers_pix[:,i] = aper_cent_pix
            apertures.append(RectAperture(slit_PA=slit_PA_use,
                        pix_perp=self.pix_perp[i], pix_parallel=self.pix_parallel[i],
                        aper_center=aper_cent_pix, nx=self.nx, ny=self.ny, partial_weight=partial_weight,
                        moment=moment))
        
        self.aper_centers_pix = aper_centers_pix
        
        super(RectApertures, self).__init__(apertures=apertures, slit_PA=slit_PA, rotate_cube=rotate_cube)

class SquareApertures(RectApertures):
    """
    Note here that pix_perp and pix_parallel are the *WIDTHS* of the rectangular apertures
    
    """
    def __init__(self, rarr=None, slit_PA=None, pix_length=None, 
             nx=None, ny=None, center_pixel=None, pixscale=None, partial_weight=True, rotate_cube=False,
             moment=False):
             
        super(SquareApertures, self).__init__(rarr=rarr, slit_PA=slit_PA, 
                pix_perp=pix_length, pix_parallel=pix_length, nx=nx, ny=ny, 
                center_pixel=center_pixel, pixscale=pixscale, partial_weight=partial_weight, 
                rotate_cube=rotate_cube,
                moment=moment)
                
                
                
                
                
def setup_aperture_types(gal=None, profile1d_type=None, 
            slit_width = None, aper_centers=None, slit_pa=None, 
            aperture_radius=None, pix_perp=None, pix_parallel=None,
            pix_length=None, from_data=True, 
            partial_weight=True, rotate_cube=False, 
            moment=False, 
            oversample=1):
            
    # partial_weight:
    #           are partial pixels weighted in apertures?
    #
            
    if from_data:
        slit_width = gal.data.slit_width
        aper_centers = gal.data.rarr
        slit_pa = gal.data.slit_pa

    rstep = gal.instrument.pixscale.value
    nx = gal.instrument.fov[0]
    ny = gal.instrument.fov[1]
    
    if (oversample > 1):
        nx *= oversample
        ny *= oversample
        rstep /= (1.* oversample)
        aper_centers *= oversample
        
        
    try:
        xcenter_samp = (gal.data.xcenter + 0.5)*oversample - 0.5
        ycenter_samp = (gal.data.ycenter + 0.5)*oversample - 0.5
        center_pixel = [xcenter_samp, ycenter_samp]
    except:
        center_pixel = None
        
    
    if (gal.data.aper_center_pix_shift is not None):
        if center_pixel is not None:
            center_pixel = [center_pixel[0] + gal.data.aper_center_pix_shift[0]*oversample,
                            center_pixel[1] + gal.data.aper_center_pix_shift[1]*oversample]
        else:
            center_pixel = [np.int(nx / 2) + gal.data.aper_center_pix_shift[0]*oversample,
                            np.int(ny / 2) + gal.data.aper_center_pix_shift[1]*oversample]

        
    print("aperture_class: center_pixel={}".format(center_pixel))

    if (profile1d_type.lower() == 'circ_ap_cube'):
        
        if (aperture_radius is not None):
            rpix = aperture_radius/rstep
        else:
            rpix = slit_width/rstep/2.

        apertures = CircApertures(rarr=aper_centers, slit_PA=slit_pa, rpix=rpix,
                 nx=nx, ny=ny, center_pixel=center_pixel, pixscale=rstep,
                 partial_weight=partial_weight, rotate_cube=rotate_cube,
                 moment=moment)

    elif (profile1d_type.lower() == 'rect_ap_cube'):

        if (pix_perp is None):
            pix_perp = slit_width/rstep
        else:
            pix_perp *= oversample
            
        if (pix_parallel is None):
            pix_parallel = slit_width/rstep
        else:
            pix_parallel *= oversample

        aper_centers_pix = aper_centers/rstep
        
        apertures = RectApertures(rarr=aper_centers, slit_PA=slit_pa,
                pix_perp=pix_perp, pix_parallel=pix_parallel, 
                nx=nx, ny=ny, center_pixel=center_pixel, pixscale=rstep,
                partial_weight=partial_weight, rotate_cube=rotate_cube,
                moment=moment)

    elif (profile1d_type.lower() == 'square_ap_cube'):

        if ('pix_length' is None):
            pix_length = slit_width/rstep
        else:
            pix_length *= oversample
        
        aper_centers_pix = aper_centers/rstep

        apertures = SquareApertures(rarr=aper_centers, slit_PA=slit_pa, pix_length = pix_length,
                 nx=nx, ny=ny, center_pixel=center_pixel, pixscale=rstep,
                 partial_weight=partial_weight, rotate_cube=rotate_cube,
                 moment=moment)

    else:
        raise TypeError('Unknown method for measuring the 1D profiles.')

    return apertures


                
                
                