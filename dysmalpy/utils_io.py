# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Module containing some useful utility functions

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library

# Third party imports
import numpy as np
from scipy import interpolate

import copy

try:
    import aperture_classes
except:
    from . import aperture_classes

#
# Class for intrinsic rot curve 
class RotCurveInt(object):
    def __init__(self, r=None, vcirc_tot=None, vcirc_bar=None, vcirc_dm=None):

        self.rarr = r
        
        data = {'vcirc_tot': vcirc_tot, 
                'vcirc_bar': vcirc_bar, 
                'vcirc_dm': vcirc_dm}
                
        
        self.data = data

    
def read_model_intrinsic_profile(filename=None):
    # Load the data set to be fit
    dat_arr =   np.loadtxt(filename) #datadir+'{}.obs_prof.txt'.format(galID))
    gal_r      = dat_arr[:,0]
    vcirc_tot  = dat_arr[:,1]
    vcirc_bar  = dat_arr[:,2]
    vcirc_dm   = dat_arr[:,3]
    
    model_int = RotCurveInt(r=gal_r, vcirc_tot=vcirc_tot, vcirc_bar=vcirc_bar, vcirc_dm=vcirc_dm)
    
    return model_int
    
def read_bestfit_1d_obs_file(filename=None):
    """
    Short function to save load space 1D obs profile for a galaxy (eg, for plotting, etc)
    Follows form of H.Ü. example.
    """
    
    # Load the model file
    dat_arr =   np.loadtxt(filename)
    gal_r =     dat_arr[:,0]
    gal_flux =  dat_arr[:,1]
    gal_vel =   dat_arr[:,2]
    gal_disp =  dat_arr[:,3]
    
    slit_width = None
    slit_pa = None
    
        
    #
    model_data = data_classes.Data1D(r=gal_r, velocity=gal_vel,
                             vel_disp=gal_disp, flux=gal_flux, 
                             slit_width=slit_width,
                             slit_pa=slit_pa)
    model_data.apertures = None
                                      
    return model_data

def write_bestfit_1d_obs_file(gal=None, fname=None):
    """
    Short function to save *observed* space 1D obs profile for a galaxy (eg, for plotting, etc)
    Follows form of H.Ü. example.
    """
    model_r = gal.model_data.rarr
    model_flux = gal.model_data.data['flux']
    model_vel = gal.model_data.data['velocity']
    model_disp = gal.model_data.data['dispersion']
    
    # Write 1D circular aperture plots to text file
    np.savetxt(fname, np.transpose([model_r, model_flux, model_vel, model_disp]),
               fmt='%2.4f\t%2.4f\t%5.4f\t%5.4f',
               header='r [arcsec], flux [...], vel [km/s], disp [km/s]')
    
    return None
    
#
def create_vel_profile_files(gal=None, outpath=None, oversample=3, oversize=1, 
            profile1d_type=None, aperture_radius=None, 
            fname_model_matchdata=None, 
            fname_finer=None, 
            fname_intrinsic=None, 
            fname_intrinsic_m = None):
    #
    if outpath is None:
        raise ValueError
    
    if fname_model_matchdata is None:
        #fname_model_matchdata = outpath + '{}_out-1dplots_{}.txt'.format(gal.name, monthday)
        fname_model_matchdata = "{}{}_out-1dplots.txt".format(outpath, gal.name)
    if fname_finer is None:
        #fname_finer = outpath + '{}_out-1dplots_{}_finer_sampling.txt'.format(gal.name, monthday)
        fname_finer = "{}{}_out-1dplots_finer_sampling.txt".format(outpath, gal.name)
        
    if fname_intrinsic is None:
        fname_intrinsic = '{}{}_vcirc_tot_bary_dm.dat'.format(outpath, gal.name)
    if fname_intrinsic_m is None:
        fname_intrinsic_m = '{}{}_menc_tot_bary_dm.dat'.format(outpath, gal.name)
        
    ###
    galin = copy.deepcopy(gal)
    
    # ---------------------------------------------------------------------------
    gal.create_model_data(oversample=oversample, oversize=oversize,
                          line_center=gal.model.line_center,
                          profile1d_type=profile1d_type)
                          
    # -------------------
    # Save Bary/DM vcirc:
    write_vcirc_tot_bar_dm(gal=gal, fname=fname_intrinsic, fname_m=fname_intrinsic_m)
    
    # --------------------------------------------------------------------------
    write_bestfit_1d_obs_file(gal=gal, fname=fname_model_matchdata)
    
    
    # Reload galaxy object: reset things
    gal = copy.deepcopy(galin)
    
    # Try finer scale:
    
    write_1d_obs_finer_scale(gal=gal, fname=fname_finer, oversample=oversample, oversize=oversize,
                profile1d_type=profile1d_type, aperture_radius=aperture_radius)
    
    
    return None
    
#
def write_1d_obs_finer_scale(gal=None, fname=None, 
            profile1d_type=None, aperture_radius=None, 
            oversample=3, oversize=1):
    # Try finer scale:
    rmax_abs = np.max([2.5, np.max(np.abs(gal.model_data.rarr))])
    r_step = 0.025 #0.05
    if rmax_abs > 4.:
        r_step = 0.05
    aper_centers_interp = np.arange(0, rmax_abs+r_step, r_step)
    
    if profile1d_type == 'rect_ap_cube':
        f_par = interpolate.interp1d(gal.data.rarr, gal.data.apertures.pix_parallel, 
                        kind='slinear', fill_value='extrapolate')
        f_perp = interpolate.interp1d(gal.data.rarr, gal.data.apertures.pix_perp, 
                        kind='slinear', fill_value='extrapolate')
        
        pix_parallel_interp = f_par(aper_centers_interp)
        pix_perp_interp = f_perp(aper_centers_interp)
        
        gal.data.apertures = aperture_classes.setup_aperture_types(gal=gal, 
                    profile1d_type=profile1d_type, 
                    aperture_radius=1., 
                    slit_width=gal.data.slit_width, 
                    aper_centers = aper_centers_interp, 
                    slit_pa = gal.data.slit_pa, 
                    pix_perp=pix_perp_interp, pix_parallel=pix_parallel_interp,
                    pix_length=None, from_data=False)
    elif profile1d_type == 'circ_ap_cube':
        gal.data.apertures = aperture_classes.setup_aperture_types(gal=gal, 
                    profile1d_type=profile1d_type, 
                    aperture_radius=aperture_radius, 
                    slit_width=gal.data.slit_width, 
                    aper_centers = aper_centers_interp, 
                    slit_pa = gal.data.slit_pa, 
                    pix_perp=None, pix_parallel=None,
                    pix_length=None, from_data=False)
                    
    gal.create_model_data(oversample=oversample, oversize=oversize, 
                          line_center=gal.model.line_center,
                          profile1d_type=profile1d_type)
                          
                          
    write_bestfit_1d_obs_file(gal=gal, fname=fname)
    
    return None
    
def write_vcirc_tot_bar_dm(gal=None, fname=None, fname_m=None):
    # -------------------
    # Save Bary/DM vcirc:
    
    rstep = 0.1
    rmax = 40.   #17.2   # kpc
    rarr = np.arange(0, rmax+rstep, rstep)
    
    vcirc_bar = gal.model.components['disk+bulge'].circular_velocity(rarr)
    vcirc_dm  = gal.model.components['halo'].circular_velocity(rarr)
    vcirc_tot = gal.model.circular_velocity(rarr)
    
    menc_tot, menc_bar, menc_dm = gal.model.enclosed_mass(rarr)
    
    vcirc_bar[~np.isfinite(vcirc_bar)] = 0.
    vcirc_dm[~np.isfinite(vcirc_dm)] = 0.
    
    vcirc_tot[~np.isfinite(vcirc_tot)] = 0.
    
    profiles = np.zeros((len(rarr), 4))
    profiles[:,0] = rarr
    profiles[:,1] = vcirc_tot
    profiles[:,2] = vcirc_bar
    profiles[:,3] = vcirc_dm
    
    profiles_m = np.zeros((len(rarr), 4))
    profiles_m[:,0] = rarr
    profiles_m[:,1] = np.log10(menc_tot)
    profiles_m[:,2] = np.log10(menc_bar)
    profiles_m[:,3] = np.log10(menc_dm)
    profiles_m[~np.isfinite(profiles_m)] = 0.
    
    save_vcirc_tot_bar_dm_files(gal=gal, fname=fname, fname_m=fname_m, 
                    profiles=profiles, profiles_m=profiles_m)
    
    return None


#
def save_vcirc_tot_bar_dm_files(gal=None, fname=None, fname_m=None, profiles=None, profiles_m=None):
    with open(fname, 'w') as f:
        namestr = '#   r   vcirc_tot vcirc_bar   vcirc_dm'
        f.write(namestr+'\n')
        unitstr = '#   [kpc]   [km/s]   [km/s]   [km/s]' 
        f.write(unitstr+'\n')
        for i in range(profiles.shape[0]):
            datstr = '    '.join(["{0:0.3f}".format(p) for p in profiles[i,:]])
            f.write(datstr+'\n')
            
    with open(fname_m, 'w') as f:
        namestr = '#   r   lmenc_tot   lmenc_bar   lmenc_dm'
        f.write(namestr+'\n')
        unitstr = '#   [kpc]   [log10Msun]   [log10Msun]   [log10Msun]' 
        f.write(unitstr+'\n')
        for i in range(profiles.shape[0]):
            datstr = '    '.join(["{0:0.3f}".format(p) for p in profiles_m[i,:]])
            f.write(datstr+'\n')
            
    return None

