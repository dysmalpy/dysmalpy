# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# File containing all of the available classes to define aperture for
# extracting 1D spectra for a galaxy/model.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard imports
import logging

# Third party imports
import numpy as np
# import scipy.interpolate as scp_interp
# import scipy.ndimage as scp_ndi

from dysmalpy.utils import calc_pixel_distance, gaus_fit_sp_opt_leastsq, gaus_fit_apy_mod_fitter

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

__all__ = [ "Aperture", "EllipAperture", "RectAperture", "Apertures",
            "EllipApertures", "CircApertures", "RectApertures", 
            # "SinglePixelPVApertures", 
            #"CircularPVApertures",
            "setup_aperture_types"] 
            #, 
            #"calc_1dprofile", "calc_1dprofile_circap_pv"]

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
    Base Aperture class, containting a single aperture from which to extract 
    a 1D spectrum and flux, velocity, and dispersion.

    aper_center, center_pixel should be in *Pixel* units, and in x,y coordinates

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
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        mask[np.int(self.aper_center[1]), np.int(self.aper_center[0])] = True
        return mask

    def extract_aper_spec(self, spec_arr=None,
            cube=None, err=None, mask=None, spec_mask=None, skip_specmask=False):
        """
        Extract the raw LOS spectral distribution for the cube (data or model).
        If setting mask, this must be the CUBE mask.
        """
        if hasattr(self, 'partial_weight'):
            if self.partial_weight:
                mask_ap = self._mask_ap
            else:
                mask_ap = self.define_aperture_mask()
        else:
            mask_ap = self.define_aperture_mask()

        mask_cube = np.tile(mask_ap, (cube.shape[0], 1, 1))

        if mask is not None:
            mask_cube *= mask
            spec_mask2 = np.sum(np.sum(mask_cube, axis=2), axis=1)
            spec_mask2[spec_mask2>0] = 1.
            spec_mask2 = np.array(spec_mask2, dtype=bool)
            if spec_mask is None:
                spec_mask = spec_mask2

        spec = np.nansum(np.nansum(cube*mask_cube, axis=1), axis=1)
        if err is not None:
            espec = np.sqrt(np.nansum(np.nansum((err**2)*mask_cube, axis=1),axis=1))

        if (spec_mask is not None) & (not skip_specmask):
            spec_fit = spec[spec_mask]
            spec_arr_fit = spec_arr[spec_mask]
            if err is not None:
                espec_fit = espec[spec_mask]
        else:
            spec_fit = spec
            spec_arr_fit = spec_arr
            if err is not None:
                espec_fit = espec
        if err is None:
            espec_fit = None

        return spec_arr_fit, spec_fit, espec_fit


    def extract_aper_kin(self, spec_arr=None,
            cube=None, err=None, mask=None, spec_mask=None):
        """
        Extract the kinematic information from the aperture LOS spectral distribution:
                flux, velocity, dispersion.

        spec_arr: the spectral direction array -- eg, vel array or wave array.
        """

        spec_arr_fit, spec_fit, espec_fit = self.extract_aper_spec(spec_arr=spec_arr,
                cube=cube, err=err, mask=mask, spec_mask=spec_mask)

        # Use the first and second moment as a guess of the line parameters
        delspec = np.average(spec_arr[1:]-spec_arr[:-1])  # need del spec
        mom0 = np.sum(spec_fit) * delspec
        mom1 = np.sum(spec_fit * spec_arr_fit) * delspec / mom0
        mom2 = np.sum(spec_fit * (spec_arr_fit - mom1) ** 2) * delspec / mom0

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Catch case where mixing old + new w/o moment defined
        if hasattr(self, 'moment'):
            moment_calc = self.moment
        else:
            moment_calc = False
        if moment_calc:
            flux1d = mom0
            vel1d = mom1
            disp1d = np.sqrt(np.abs(mom2))
        else:
            try:
                if err is not None:
                    # Use astropy model fitter:
                    best_fit = gaus_fit_apy_mod_fitter(spec_arr_fit, spec_fit,
                                    mom0, mom1, np.sqrt(np.abs(mom2)), yerr=espec_fit)
                else:
                    # Use unweighted -- FASTER
                    best_fit = gaus_fit_sp_opt_leastsq(spec_arr_fit, spec_fit, mom0, mom1, np.sqrt(np.abs(mom2)))

                flux1d = best_fit[0] * np.sqrt(2 * np.pi) * best_fit[2]
                vel1d = best_fit[1]
                disp1d = best_fit[2]
            except:
                flux1d = np.NaN
                vel1d = np.NaN
                disp1d = np.NaN
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++

        return flux1d, vel1d, disp1d


class EllipAperture(Aperture):
    """
    Elliptical Aperture

    pix_perp and pix_parallel are number of pixels of the elliptical semi axes lengths.

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
            do_partial_weight = True

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
                aper_ell = shply_affinity.scale(circtmp, self.pix_perp, self.pix_parallel)


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
                # apmask = None # fractional pixels
        else:
            apmask = ( (yslits/self.pix_parallel)**2 + (xslits/self.pix_perp)**2 <= 1. )

        return apmask

#
class RectAperture(Aperture):
    """
    Rectangular aperture. 

    pix_perp and pix_parallel are number of pixels of rectangle width/height. 

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
            do_partial_weight = True

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
                # apmask = None # fractional pixels
        else:
            apmask = ( (np.abs(xslits) <= self.pix_perp/2.) & (np.abs(yslits) <= self.pix_parallel/2.) )

        return apmask



class Apertures(object):
    """ 
    Base Aperture class, continaing a list/array of Aperture objects 
    defining the full set of apertures for a kinematic observation. 

    Generic case. Should be array of Aperture objects. Needs the loop.
    """
    def __init__(self, apertures=None, slit_PA=None, rotate_cube=False):
        self.apertures = apertures
        self.slit_PA = slit_PA
        self.slit_PA_unrotated = slit_PA
        self.rotate_cube = rotate_cube

    def extract_1d_kinematics(self, spec_arr=None,
                cube=None, err=None, mask=None, spec_mask=None,
                center_pixel=None, pixscale=None):
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



class EllipApertures(Apertures):
    """
    Set of elliptical apertures. 
    
    Uses same generic extract_1d_kinematics as Apertures.
    Sizes can vary. -- depending on if pix_perp and pix_parallel are arrays or scalar.

    FOR THIS CASE: aper_centers are along the slit.

    rarr should be in *** ARCSEC ***

    Note here that pix_perp and pix_parallel are the semi axes lengths!


    """
    def __init__(self, rarr=None, slit_PA=None, pix_perp=None, pix_parallel=None,
             nx=None, ny=None, center_pixel=None, pixscale=None, partial_weight=True, rotate_cube=False,
             moment=False):

        #
        if rotate_cube:
            slit_PA_use = 0.
        else:
            slit_PA_use = slit_PA

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
    """
    Set of circular apertures. 
    
    Extends EllipApertures.
    """
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
    Set of rectangular apertures. 
    
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



# class SinglePixelPVApertures(Apertures):
#     """
#     Wrapper around the original IDL DYSMAL "single pixel PV" extraction
#     calculations.

#     Preserved as an option for checks, etc.

#     rarr should be in *** ARCSEC ***

#     slit_width: arcsec
#     slit_PA: degrees

#     """
#     def __init__(self, slit_width=None, slit_PA=None,
#                  pixscale=None, rarr=None,
#                  moment=True):

#         self.slit_width = slit_width
#         self.slit_PA = slit_PA
#         self.pixscale = pixscale
#         self.rarr = rarr
        
#         super(SinglePixelPVApertures, self).__init__(apertures=None, slit_PA=slit_PA, rotate_cube=False)

#     def extract_1d_kinematics(self, spec_arr=None,
#                 cube=None, err=None, mask=None, spec_mask=None,
#                 center_pixel = None, pixscale=None):
#         """
#         aper_centers_pixout: the radial direction positions, relative to kin center, in pixels
#         """

#         r1d, flux1d, vel1d, disp1d = calc_1dprofile(cube,
#                         self.slit_width,self.slit_PA-180.,
#                         self.pixscale, spec_arr)
#         vinterp = scp_interp.interp1d(r1d, vel1d, fill_value='extrapolate')
#         disp_interp = scp_interp.interp1d(r1d, disp1d, fill_value='extrapolate')
#         vel1d = vinterp(self.rarr)
#         disp1d = disp_interp(self.rarr)
#         flux_interp = scp_interp.interp1d(r1d, flux1d, fill_value='extrapolate')
#         flux1d = flux_interp(self.rarr)

#         #return self.aper_centers, flux1d, vel1d, disp1d
#         return self.rarr, flux1d, vel1d, disp1d

# class CircularPVApertures(Apertures):
#     """
#     Wrapper around the original IDL DYSMAL "circular aperture PV" extraction
#     calculations.

#     Should be mathemtically equivalent to the cube-applied CircApertures,
#     but preserved as an option for checks, etc.

#     rarr should be in *** ARCSEC ***

#     slit_width: arcsec
#     slit_PA: degrees

#     """
#     def __init__(self, slit_width=None, slit_PA=None,
#                  pixscale=None, rarr=None,
#                  moment=True):

#         self.slit_width = slit_width
#         self.slit_PA = slit_PA
#         self.pixscale = pixscale
#         self.rarr = rarr
        
#         #<TODO><20220618># no self.apertures??
        
#         super(CircularPVApertures, self).__init__(apertures=None, slit_PA=slit_PA, rotate_cube=False)

#     def extract_1d_kinematics(self, spec_arr=None,
#                 cube=None, err=None, mask=None, spec_mask=None,
#                 center_pixel = None, pixscale=None):
#         """
#         aper_centers_pixout: the radial direction positions, relative to kin center, in pixels
#         """

#         r1d, flux1d, vel1d, disp1d = calc_1dprofile_circap_pv(cube,
#                         self.slit_width,self.slit_PA-180.,
#                         self.pixscale, spec_arr)
#         vinterp = scp_interp.interp1d(r1d, vel1d, fill_value='extrapolate')
#         disp_interp = scp_interp.interp1d(r1d, disp1d, fill_value='extrapolate')
#         vel1d = vinterp(self.rarr)
#         disp1d = disp_interp(self.rarr)
#         flux_interp = scp_interp.interp1d(r1d, flux1d, fill_value='extrapolate')
#         flux1d = flux_interp(self.rarr)

#         #return self.aper_centers, flux1d, vel1d, disp1d
#         return self.rarr, flux1d, vel1d, disp1d




# def area_segm(rr, dd):

#     return (rr**2 * np.arccos(dd/rr) -
#             dd * np.sqrt(2. * rr * (rr-dd) - (rr-dd)**2))



# def calc_1dprofile(cube, slit_width, slit_angle, pxs, vx, soff=0.):
#     """
#     Measure the 1D rotation curve from a cube using a pseudoslit.

#     This function measures the 1D rotation curve by first creating a PV diagram based on the
#     input slit properties. Fluxes, velocities, and dispersions are then measured from the spectra
#     at each single position in the PV diagram by calculating the 0th, 1st, and 2nd moments
#     of each spectrum.

#     Parameters
#     ----------
#     cube : 3D array
#         Data cube from which to measure the rotation curve. First dimension is assumed to
#         be spectral direction.

#     slit_width : float
#         Slit width of the pseudoslit in arcseconds

#     slit_angle : float
#         Position angle of the pseudoslit

#     pxs : float
#         Pixelscale of the data cube in arcseconds/pixel

#     vx : 1D array
#         Values of the spectral axis. This array must have the same length as the
#         first dimension of `cube`.

#     soff : float, optional
#         Offset of the slit from center in arcseconds. Default is 0.

#     Returns
#     -------
#     xvec : 1D array
#         Position along slit in arcseconds

#     flux : 1D array
#         Relative flux of the line at each position. Calculated as the sum of the spectrum.

#     vel : 1D array
#         Velocity at each position in same units as given by `vx`. Calculated as the first moment
#         of the spectrum.

#     disp : 1D array
#         Velocity dispersion at each position in the same units as given by `vx`. Calculated as the
#         second moment of the spectrum.

#     """
#     cube_shape = cube.shape
#     psize = cube_shape[1]
#     vsize = cube_shape[0]
#     lin = np.arange(psize) - np.fix(psize/2.)
#     veldata = scp_ndi.interpolation.rotate(cube, slit_angle, axes=(2, 1),
#                                            reshape=False)
#     tmpn = (((lin*pxs) <= (soff+slit_width/2.)) &
#             ((lin*pxs) >= (soff-slit_width/2.)))
#     data = np.zeros((psize, vsize))

#     flux = np.zeros(psize)

#     yvec = vx
#     xvec = lin*pxs

#     for i in range(psize):
#         for j in range(vsize):
#             data[i, j] = np.mean(veldata[j, i, tmpn])
#         flux[i] = np.sum(data[i,:])

#     flux = flux / np.max(flux) * 10.
#     pvec = (flux < 0.)

#     vel = np.zeros(psize)
#     disp = np.zeros(psize)
#     for i in range(psize):
#         vel[i] = np.sum(data[i,:]*yvec)/np.sum(data[i,:])
#         disp[i] = np.sqrt( np.sum( ((yvec-vel[i])**2) * data[i,:]) / np.sum(data[i,:]) )

#     if np.sum(pvec) > 0.:
#         vel[pvec] = -1.e3
#         disp[pvec] = 0.

#     return xvec, flux, vel, disp


# def calc_1dprofile_circap_pv(cube, slit_width, slit_angle, pxs, vx, soff=0.):
#     """
#     Measure the 1D rotation curve from a cube using a pseudoslit

#     This function measures the 1D rotation curve by first creating a PV diagram based on the
#     input slit properties. Fluxes, velocities, and dispersions are then measured from spectra
#     produced by integrating over circular apertures placed on the PV diagram with radii equal
#     to 0.5*`slit_width`. The 0th, 1st, and 2nd moments of the integrated spectra are then calculated
#     to determine the flux, velocity, and dispersion.

#     Parameters
#     ----------
#     cube : 3D array
#         Data cube from which to measure the rotation curve. First dimension is assumed to
#         be spectral direction.

#     slit_width : float
#         Slit width of the pseudoslit in arcseconds

#     slit_angle : float
#         Position angle of the pseudoslit

#     pxs : float
#         Pixelscale of the data cube in arcseconds/pixel

#     vx : 1D array
#         Values of the spectral axis. This array must have the same length as the
#         first dimension of `cube`.

#     soff : float, optional
#         Offset of the slit from center in arcseconds. Default is 0.

#     Returns
#     -------
#     xvec : 1D array
#         Position along slit in arcseconds

#     flux : 1D array
#         Relative flux of the line at each position. Calculated as the sum of the spectrum.

#     vel : 1D array
#         Velocity at each position in same units as given by `vx`. Calculated as the first moment
#         of the spectrum.

#     disp : 1D array
#         Velocity dispersion at each position in the same units as given by `vx`. Calculated as the
#         second moment of the spectrum.

#     """
#     cube_shape = cube.shape
#     psize = cube_shape[1]
#     vsize = cube_shape[0]
#     lin = np.arange(psize) - np.fix(psize/2.)
#     veldata = scp_ndi.interpolation.rotate(cube, slit_angle, axes=(2, 1),
#                                            reshape=False)
#     tmpn = (((lin*pxs) <= (soff+slit_width/2.)) &
#             ((lin*pxs) >= (soff-slit_width/2.)))
#     data = np.zeros((psize, vsize))
#     flux = np.zeros(psize)

#     yvec = vx
#     xvec = lin*pxs

#     for i in range(psize):
#         for j in range(vsize):
#             data[i, j] = np.mean(veldata[j, i, tmpn])
#         tmp = data[i]
#         flux[i] = np.sum(tmp)

#     flux = flux / np.max(flux) * 10.
#     pvec = (flux < 0.)

#     # Calculate circular segments
#     rr = 0.5 * slit_width
#     pp = pxs

#     nslice = int(1 + 2 * np.ceil((rr - 0.5 * pp) / pp))

#     circaper_idx = np.arange(nslice) - 0.5 * (nslice - 1)
#     circaper_sc = np.zeros(nslice)

#     circaper_sc[int(0.5*nslice - 0.5)] = (np.pi*rr**2 -
#                                           2.*area_segm(rr, 0.5*pp))

#     if nslice > 1:
#         circaper_sc[0] = area_segm(rr, (0.5*nslice - 1)*pp)
#         circaper_sc[nslice-1] = circaper_sc[0]

#     if nslice > 3:
#         for cnt in range(1, int(0.5*(nslice-3))+1):
#             circaper_sc[cnt] = (area_segm(rr, (0.5*nslice - 1. - cnt)*pp) -
#                                 area_segm(rr, (0.5*nslice - cnt)*pp))
#             circaper_sc[nslice-1-cnt] = circaper_sc[cnt]

#     circaper_vel = np.zeros(psize)
#     circaper_disp = np.zeros(psize)
#     circaper_flux = np.zeros(psize)

#     nidx = len(circaper_idx)
#     for i in range(psize):
#         tot_vnum = 0.
#         tot_denom = 0.
#         cnt_idx = 0
#         cnt_start = int(i + circaper_idx[0]) if (i + circaper_idx[0]) > 0 else 0
#         cnt_end = (int(i + circaper_idx[nidx-1]) if (i + circaper_idx[nidx-1]) <
#                                                     (psize-1) else (psize-1))
#         for cnt in range(cnt_start, cnt_end+1):
#             tmp = data[cnt]
#             tot_vnum += circaper_sc[cnt_idx] * np.sum(tmp*yvec)
#             tot_denom += circaper_sc[cnt_idx] * np.sum(tmp)
#             cnt_idx = cnt_idx + 1

#         circaper_vel[i] = tot_vnum / tot_denom
#         circaper_flux[i] = tot_denom

#         tot_dnum = 0.
#         cnt_idx = 0
#         for cnt in range(cnt_start, cnt_end+1):
#             tmp = data[cnt]
#             tot_dnum = (tot_dnum + circaper_sc[cnt_idx] *
#                         np.sum(tmp*(yvec-circaper_vel[i])**2))
#             cnt_idx = cnt_idx + 1

#         circaper_disp[i] = np.sqrt(tot_dnum / tot_denom)

#     if np.sum(pvec) > 0.:
#         circaper_vel[pvec] = -1.e3
#         circaper_disp[pvec] = 0.
#         circaper_flux[pvec] = 0.

#     return xvec, circaper_flux, circaper_vel, circaper_disp


def setup_aperture_types(obs=None, profile1d_type=None,
            slit_width = None, aper_centers=None, slit_pa=None,
            aperture_radius=None, pix_perp=None, pix_parallel=None,
            partial_weight=True,
            rotate_cube=False):

    # partial_weight:
    #      are partial pixels weighted in apertures?

    _valid_1dprofs = ['circ_ap_cube', 'rect_ap_cube', 'circ_ap_pv', 'single_pix_pv']
    _print_valid_1dprofs = ['circ_ap_cube', 'rect_ap_cube', 'single_pix_pv']
    if profile1d_type.lower() not in _valid_1dprofs:
        raise ValueError("profile1d_type={} not in the valid list: {}".format(
            profile1d_type.lower(), 
            _print_valid_1dprofs
            ))

    if aper_centers is None:
        raise ValueError("Must set 'aper_centers'!")
    if slit_pa is None:
        raise ValueError("Must set 'slit_pa'!")
    if (slit_width is None) & (profile1d_type not in ['circ_ap_cube', 'circ_ap_pv']):
        if (profile1d_type.lower() == 'rect_ap_cube') & (pix_perp is not None) & (pix_parallel is not None):
            pass
        else:
            msg = "profile1d_type: {}\n".format(profile1d_type)
            if (profile1d_type.lower() in ['rect_ap_cube']) & (
                (pix_perp is None) or (pix_parallel is None)
            ):
                msg += " & pix_perp={}, pix_parallel={}\n".format(pix_perp, pix_parallel)
            msg += "Must set 'slit_width'!"
            raise ValueError(msg)

    # if False:
    #     slit_width = obs.instrument.slit_width
    #     aper_centers = obs.instrument.rarr
    #     slit_pa = obs.instrument.slit_pa

    pixscale = obs.instrument.pixscale.value
    nx = obs.instrument.fov[0]
    ny = obs.instrument.fov[1]


    try:
        xcenter_samp = (obs.obs_options.xcenter + 0.5) - 0.5
        ycenter_samp = (obs.obs_options.ycenter + 0.5) - 0.5
        center_pixel = [xcenter_samp, ycenter_samp]
    except:
        center_pixel = None


    # ----------------------------------------
    # SET UP SPECIFIC TYPES:

    if (profile1d_type.lower() == 'circ_ap_cube'):
        if (aperture_radius is not None):
            rpix = aperture_radius/pixscale
        else:
            if slit_width is None:
                raise ValueError("If not setting 'aperture_radius', must set 'slit_width'.")
            rpix = slit_width/pixscale/2.

        apertures = CircApertures(rarr=aper_centers, slit_PA=slit_pa, rpix=rpix,
                 nx=nx, ny=ny, center_pixel=center_pixel, pixscale=pixscale,
                 partial_weight=partial_weight, rotate_cube=rotate_cube,
                 moment=obs.instrument.moment)

    elif (profile1d_type.lower() == 'rect_ap_cube'):
        if (pix_perp is None):
            pix_perp = slit_width/pixscale

        if (pix_parallel is None):
            pix_parallel = slit_width/pixscale

        apertures = RectApertures(rarr=aper_centers, slit_PA=slit_pa,
                pix_perp=pix_perp, pix_parallel=pix_parallel,
                nx=nx, ny=ny, center_pixel=center_pixel, pixscale=pixscale,
                partial_weight=partial_weight, rotate_cube=rotate_cube,
                moment=obs.instrument.moment)


    elif (profile1d_type.lower() == 'circ_ap_pv'):
        # apertures = CircularPVApertures(rarr=aper_centers, slit_PA=slit_pa,
        #                                 slit_width=slit_width, pixscale=pixscale,
        #                                 moment=obs.instrument.moment)

        # Just do the direct cube circular aperture extraction:
        wmsg = "profile1d_type = 'circ_ap_pv' is depreciated! Use 'circ_ap_cube' instead!"
        logger.warning(wmsg)
        if (aperture_radius is not None):
            rpix = aperture_radius/pixscale
        else:
            if slit_width is None:
                raise ValueError("If not setting 'aperture_radius', must set 'slit_width'.")
            rpix = slit_width/pixscale/2.
        apertures = CircApertures(rarr=aper_centers, slit_PA=slit_pa, rpix=rpix,
                        nx=nx, ny=ny, center_pixel=center_pixel, pixscale=pixscale,
                        partial_weight=partial_weight, rotate_cube=rotate_cube,
                        moment=obs.instrument.moment)

    elif (profile1d_type.lower() == 'single_pix_pv'):
        # apertures = SinglePixelPVApertures(rarr=aper_centers, slit_PA=slit_pa,
        #                                    slit_width=slit_width, pixscale=pixscale,
        #                                    moment=obs.instrument.moment)
        
        # Just use rectangular apertures:
        pix_perp = slit_width/pixscale
        pix_parallel = 1

        apertures = RectApertures(rarr=aper_centers, slit_PA=slit_pa,
                pix_perp=pix_perp, pix_parallel=pix_parallel,
                nx=nx, ny=ny, center_pixel=center_pixel, pixscale=pixscale,
                partial_weight=partial_weight, rotate_cube=rotate_cube,
                moment=obs.instrument.moment)
        
    else:
        raise TypeError('Unknown method for measuring the 1D profiles.')

    return apertures
