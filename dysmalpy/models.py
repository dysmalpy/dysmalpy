# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# File containing all of the available models to use build the
# galaxy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import os
import abc
import logging
import time
from collections import OrderedDict

# Third party imports
import numpy as np
import scipy.special as scp_spec
import scipy.io as scp_io
import scipy.interpolate as scp_interp
import scipy.optimize as scp_opt
import scipy.ndimage as scp_ndi
import astropy.constants as apy_con
import astropy.units as u
from astropy.modeling import Model
import astropy.cosmology as apy_cosmo
import pyximport; pyximport.install(setup_args={'include_dirs':['/Users/ttshimiz/Github/dysmalpy/dysmalpy/']})
from . import cutils

# Local imports
from .parameters import DysmalParameter

__all__ = ['ModelSet', 'MassModel', 'Sersic', 'NFW',
           'DispersionConst', 'Geometry', 'BiconicalOutflow',
           'KinematicOptions', 'ZHeightGauss', 'DiskBulge']

# NOORDERMEER DIRECTORY
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
_dir_noordermeer = dir_path+"/data/noordermeer/"

# CONSTANTS
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc

# DEFAULT COSMOLOGY
_default_cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')

np.warnings.filterwarnings('ignore')

# TODO: Tied parameters are NOT automatically updated when variables change!!
# TODO: Need to keep track during the fitting!


def sersic_mr(r, mass, n, r_eff):
    """Radial mass function for a generic sersic model"""

    bn = scp_spec.gammaincinv(2. * n, 0.5)
    alpha = r_eff / (bn ** n)
    amp = (mass / (2 * np.pi) / alpha ** 2 / n /
           scp_spec.gamma(2. * n))
    mr = amp * np.exp(-bn * (r / r_eff) ** (1. / n))

    return mr


def sersic_menc(r, mass, n, r_eff):
    """Enclosed mass as a function of r for a sersic model"""

    bn = scp_spec.gammaincinv(2. * n, 0.5)
    integ = scp_spec.gammainc(2 * n, bn * (r / r_eff) ** (1. / n))
    norm = mass

    return norm*integ
    
def v_circular(mass_enc, r):
    """
    Default method to evaluate the circular velocity
    as a function of radius using the standard equation:
    v(r) = SQRT(GM(r)/r)
    """
    vcirc = np.sqrt(G.cgs.value * mass_enc * Msun.cgs.value /
                    (r * 1000. * pc.cgs.value))
    vcirc = vcirc/1e5

    return vcirc

def menc_from_vcirc(vcirc, r):
    menc = ((vcirc*1e5)**2.*(r*1000.*pc.cgs.value) /
                  (G.cgs.value * Msun.cgs.value))
    return menc

def apply_noord_flat(r, r_eff, mass, n, invq):

    noordermeer_n = np.arange(0.5, 8.1, 0.1)  # Sersic indices
    noordermeer_invq = np.array([1, 2, 3, 4, 5, 6, 8, 10, 20,
                                 100])  # 1:1, 1:2, 1:3, ...flattening

    nearest_n = noordermeer_n[
        np.argmin(np.abs(noordermeer_n - n))]
    nearest_q = noordermeer_invq[
        np.argmin(np.abs(noordermeer_invq - invq))]

    # Need to do this internally instead of relying on IDL save files!!
    file_noord = _dir_noordermeer + 'VC_n{0:3.1f}_invq{1}.save'.format(
        nearest_n, nearest_q)
    restNVC = scp_io.readsav(file_noord)
    N2008_vcirc = restNVC.N2008_vcirc
    N2008_rad = restNVC.N2008_rad
    N2008_Re = restNVC.N2008_Re
    N2008_mass = restNVC.N2008_mass

    v_interp = scp_interp.interp1d(N2008_rad, N2008_vcirc,
                                   fill_value="extrapolate")
    vcirc = (v_interp(r / r_eff * N2008_Re) * np.sqrt(
             mass / N2008_mass) * np.sqrt(N2008_Re / r_eff))

    return vcirc

def area_segm(rr, dd):

    return (rr**2 * np.arccos(dd/rr) -
            dd * np.sqrt(2. * rr * (rr-dd) - (rr-dd)**2))


def calc_1dprofile(cube, slit_width, slit_angle, pxs, vx, soff=0.):

    # Get the data for the observed velocity profile
    cube_shape = cube.shape
    psize = cube_shape[1]
    vsize = cube_shape[0]
    lin = np.arange(psize) - np.fix(psize/2.)
    veldata = scp_ndi.interpolation.rotate(cube, slit_angle, axes=(2, 1),
                                           reshape=False)
    tmpn = (((lin*pxs) <= (soff+slit_width/2.)) &
            ((lin*pxs) >= (soff-slit_width/2.)))
    data = np.zeros((psize, vsize))
    flux = np.zeros(psize)

    yvec = vx
    xvec = lin*pxs

    for i in range(psize):
        for j in range(vsize):
            data[i, j] = np.mean(veldata[j, i, tmpn])
        tmp = data[i]
        flux[i] = np.sum(tmp)

    flux = flux / np.max(flux) * 10.
    pvec = (flux < 0.)

    # Calculate circular segments
    rr = 0.5 * slit_width
    pp = pxs

    nslice = np.int(1 + 2 * np.ceil((rr - 0.5 * pp) / pp))

    circaper_idx = np.arange(nslice) - 0.5 * (nslice - 1)
    circaper_sc = np.zeros(nslice)

    circaper_sc[int(0.5*nslice - 0.5)] = (np.pi*rr**2 -
                                          2.*area_segm(rr, 0.5*pp))

    if nslice > 1:
        circaper_sc[0] = area_segm(rr, (0.5*nslice - 1)*pp)
        circaper_sc[nslice-1] = circaper_sc[0]

    if nslice > 3:
        for cnt in range(1, int(0.5*(nslice-3))+1):
            circaper_sc[cnt] = (area_segm(rr, (0.5*nslice - 1. - cnt)*pp) -
                                area_segm(rr, (0.5*nslice - cnt)*pp))
            circaper_sc[nslice-1-cnt] = circaper_sc[cnt]

    circaper_vel = np.zeros(psize)
    circaper_disp = np.zeros(psize)

    nidx = len(circaper_idx)
    for i in range(psize):
        tot_vnum = 0.
        tot_denom = 0.
        cnt_idx = 0
        cnt_start = int(i + circaper_idx[0]) if (i + circaper_idx[0]) > 0 else 0
        cnt_end = (int(i + circaper_idx[nidx-1]) if (i + circaper_idx[nidx-1]) <
                                                    (psize-1) else (psize-1))
        for cnt in range(cnt_start, cnt_end+1):
            tmp = data[cnt]
            tot_vnum += circaper_sc[cnt_idx] * np.sum(tmp*yvec)
            tot_denom += circaper_sc[cnt_idx] * np.sum(tmp)
            cnt_idx = cnt_idx + 1

        circaper_vel[i] = tot_vnum / tot_denom

        tot_dnum = 0.
        cnt_idx = 0
        for cnt in range(cnt_start, cnt_end+1):
            tmp = data[cnt]
            tot_dnum = (tot_dnum + circaper_sc[cnt_idx] *
                        np.sum(tmp*(yvec-circaper_vel[i])**2))
            cnt_idx = cnt_idx + 1

        circaper_disp[i] = np.sqrt(tot_dnum / tot_denom)

    if np.sum(pvec) > 0.:
        circaper_vel[pvec] = -1.e3
        circaper_disp[pvec] = 0.

    return xvec, circaper_vel, circaper_disp


# Generic model container which tracks all components, parameters,
# parameter settings, model settings, etc.
class ModelSet:

    def __init__(self):

        self.mass_components = OrderedDict()
        self.components = OrderedDict()
        self.light_components = OrderedDict()
        self.geometry = None
        self.dispersion_profile = None
        self.zprofile = None
        self.outflow = None
        self.outflow_geometry = None
        self.outflow_dispersion = None
        self.outflow_flux = None
        self.parameters = None
        self.fixed = OrderedDict()
        self.tied = OrderedDict()
        self.param_names = OrderedDict()
        self._param_keys = OrderedDict()
        self.nparams = 0
        self.nparams_free = 0
        self.nparams_tied = 0
        self.kinematic_options = KinematicOptions()
        self.line_center = None
        
        # Option for dealing with 3D data:
        self.per_spaxel_norm_3D = False

    def add_component(self, model, name=None, light=False, geom_type='galaxy',
                      disp_type='galaxy'):
        """Add a model component to the set"""

        # Check to make sure its the correct class
        if isinstance(model, _DysmalModel):

            # Check to make sure it has a name
            if (name is None) & (model.name is None):
                raise ValueError('Please give this component a name!')

            elif name is not None:
                model = model.rename(name)

            if model._type == 'mass':

                # Make sure there isn't a mass component already named this
                if list(self.mass_components.keys()).count(model.name) > 0:
                    raise ValueError('Component already exists. Please give'
                                     'it a unique name.')
                else:
                    self.mass_components[model.name] = True

            elif model._type == 'geometry':

                if geom_type == 'galaxy':
                    if (self.geometry is not None):
                        logger.warning('Current Geometry model is being '
                                    'overwritten!')
                    self.geometry = model

                elif geom_type == 'outflow':

                    self.outflow_geometry = model

                else:
                    logger.error("geom_type can only be either 'galaxy' or "
                                 "'outflow'.")

                self.mass_components[model.name] = False

            elif model._type == 'dispersion':

                if disp_type == 'galaxy':
                    if self.dispersion_profile is not None:
                        logger.warning('Current Dispersion model is being '
                                       'overwritten!')
                    self.dispersion_profile = model

                elif disp_type == 'outflow':

                    self.outflow_dispersion = model

                self.mass_components[model.name] = False

            elif model._type == 'zheight':
                if self.zprofile is not None:
                    logger.warning('Current z-height model is being '
                                   'overwritten!')
                self.zprofile = model
                self.mass_components[model.name] = False

            elif model._type == 'outflow':
                if self.outflow is not None:
                    logger.warning('Current outflow model is being '
                                   'overwritten!')
                self.outflow = model
                self.mass_components[model.name] = False

            else:
                raise TypeError("This model type is not known. Must be one of"
                                "'mass', 'geometry', 'dispersion', 'zheight',"
                                " or 'outflow'.")

            if light:
                self.light_components[model.name] = True
            else:
                self.light_components[model.name] = False

            self._add_comp(model)

        else:

            raise TypeError('Model component must be a '
                            'dysmalpy.models.DysmalModel instance!')

    def _add_comp(self, model):

        # Update the components list
        self.components[model.name] = model

        # Update the parameters and parameters_free arrays
        if self.parameters is None:
            self.parameters = model.parameters
        else:

            self.parameters = np.concatenate([self.parameters,
                                              model.parameters])
        self.param_names[model.name] = model.param_names
        self.fixed[model.name] = model.fixed
        self.tied[model.name] = model.tied

        # Update the dictionaries containing the locations of the parameters
        # in the parameters array. Also count number of tied parameters
        key_dict = OrderedDict()
        ntied = 0
        for i, p in enumerate(model.param_names):
            key_dict[p] = i + self.nparams
            if model.tied[p]:
                ntied += 1

        self._param_keys[model.name] = key_dict
        self.nparams += len(model.param_names)
        self.nparams_free += (len(model.param_names) - sum(model.fixed.values())
                              - ntied)
        self.nparams_tied += ntied

    def set_parameter_value(self, model_name, param_name, value):
        """Method to set a specific parameter value"""

        try:
            comp = self.components[model_name]
        except KeyError:
            raise KeyError('Model not part of the set.')

        try:
            param_i = comp.param_names.index(param_name)
        except ValueError:
            raise ValueError('Parameter is not part of model.')

        self.components[model_name].parameters[param_i] = value
        self.parameters[self._param_keys[model_name][param_name]] = value

    def set_parameter_fixed(self, model_name, param_name, fix):
        """Method to set a specific parameter fixed or not"""

        try:
            comp = self.components[model_name]
        except KeyError:
            raise KeyError('Model not part of the set.')

        try:
            param_i = comp.param_names.index(param_name)
        except ValueError:
            raise ValueError('Parameter is not part of model.')

        self.components[model_name].fixed[param_name] = fix
        self.fixed[model_name][param_name] = fix
        if fix:
            self.nparams_free -= 1
        else:
            self.nparams_free += 1

    def update_parameters(self, theta):
        """Update all of the free parameters of the model
           Then update all of the tied parameters.
        """

        # Sanity check to make sure the array given is the right length
        if len(theta) != self.nparams_free:
            raise ValueError('Length of theta is not equal to number '
                             'of free parameters, {}'.format(self.nparams_free))

        # Loop over all of the parameters
        pfree, pfree_keys = self._get_free_parameters()
        for cmp in pfree_keys:
            for pp in pfree_keys[cmp]:
                ind = pfree_keys[cmp][pp]
                if ind != -99:
                    self.set_parameter_value(cmp, pp, theta[ind])

        # Now update all of the tied parameters if there are any
        if self.nparams_tied > 0:
            for cmp in self.tied:
                for pp in self.tied[cmp]:
                    if self.tied[cmp][pp]:
                        new_value = self.tied[cmp][pp](self)
                        self.set_parameter_value(cmp, pp, new_value)

    # Methods to grab the free parameters and keys
    def _get_free_parameters(self):
        p = np.zeros(self.nparams_free)
        pkeys = OrderedDict()
        j = 0
        for cmp in self.fixed:
            pkeys[cmp] = OrderedDict()
            for pm in self.fixed[cmp]:
                if self.fixed[cmp][pm] | np.bool(self.tied[cmp][pm]):
                    pkeys[cmp][pm] = -99
                else:
                    pkeys[cmp][pm] = j
                    p[j] = self.parameters[self._param_keys[cmp][pm]]
                    j += 1
        return p, pkeys

    def get_free_parameters_values(self):
        pfree, pfree_keys = self._get_free_parameters()
        return pfree

    def get_free_parameter_keys(self):
        pfree, pfree_keys = self._get_free_parameters()
        return pfree_keys

    def get_log_prior(self):
        log_prior_model = 0.
        pfree_dict = self.get_free_parameter_keys()
        comps_names = pfree_dict.keys()
        for compn in comps_names:
            comp = self.components.__getitem__(compn)
            params_names = pfree_dict[compn].keys()
            for paramn in params_names:
                if pfree_dict[compn][paramn] >= 0:
                    # Free parameter: add to total prior
                    log_prior_model += comp.prior[paramn].log_prior(comp.__getattribute__(paramn))
        return log_prior_model
        
    def get_dm_aper(self, r, rstep=0.2):
        lnin = 0
        try:
            lnin = len(r)
            if lnin == 1:
                r = r[0]
                makearr = True
            else:
                rgal = r
                makearr = False
        except:
            makearr = True
            
        if makearr:
            nstep = np.floor_divide(r,rstep) 
            rgal = np.linspace(0.,nstep*rstep,num=nstep+1)
            rgal = np.append(rgal, r_eff)
            
            
        ## Get DM frac:
        vel, vdm = self.velocity_profile(rgal, compute_dm=True)
        
        if self.kinematic_options.pressure_support:
            # Correct for pressure support to get circular velocity:
            vc = self.kinematic_options.correct_for_pressure_support(rgal, self, vel)
        else:
            vc = vel.copy()
        
        # Not generally true if a term is oblate; to be updated
        # r_eff is the last (or only) entry:
        if (lnin <= 1):
            dm_frac = vdm[-1]**2/vc[-1]**2
            if lnin == 1:
                dm_frac = np.array([df_frac])
        else:
            dm_frac = vdm**2/vc**2
        
        return dm_frac
            
    def get_dm_frac_effrad(self, rstep=0.2, model_key_re=None):
        # RE needs to be in kpc
        comp = self.components.__getitem__(model_key_re[0])
        param_i = comp.param_names.index(model_key_re[1])
        r_eff = comp.parameters[param_i]
        
        
        dm_frac = self.get_dm_aper(self, r_eff, rstep=rstep)
        
        
        # # Get DM frac:
        # if self.kinematic_options.adiabatic_contract:
        #     nstep = np.floor_divide(r_eff,rstep) 
        #     rgal = np.linspace(0.,nstep*rstep,num=nstep+1)
        #     rgal = np.append(rgal, r_eff)
        # else:
        #     rgal = np.array([r_eff])
        # 
        # vel, vdm = self.velocity_profile(rgal, compute_dm=True)
        # 
        # if self.kinematic_options.pressure_support:
        #     # Correct for pressure support to get circular velocity:
        #     vc = self.kinematic_options.correct_for_pressure_support(rgal, self, vel)
        # else:
        #     vc = vel.copy()
        # 
        # # Not generally true if a term is oblate; to be updated
        # # r_eff is the last (or only) entry:
        # dm_frac = vdm[-1]**2/vc[-1]**2
        
        return dm_frac
        

    def enclosed_mass(self, r):
        """
        Method to calculate the total enclosed mass for the whole model
        as a function of radius
        :param r: Radius in kpc
        :return: Mass enclosed within each radius in Msun
        """

        # First check to make sure there is at least one mass component in the
        # model set.
        if len(self.mass_components) == 0:
            raise AttributeError("There are no mass components so an enclosed "
                                 "mass can't be calculated.")
        else:

            enc_mass = r*0.
            enc_dm = r*0.
            enc_bary = r*0.

            for cmp in self.mass_components:
                if self.mass_components[cmp]:
                    mcomp = self.components[cmp]
                    enc_mass_cmp = mcomp.enclosed_mass(r)
                    enc_mass += enc_mass_cmp

                    if mcomp._subtype == 'dark_matter':

                        enc_dm += enc_mass_cmp

                    elif mcomp._subtype == 'baryonic':

                        enc_bary += enc_mass_cmp

            if (np.sum(enc_dm) > 0) & self.kinematic_options.adiabatic_contract:

                vcirc, vhalo_adi = self.velocity_profile(r, compute_dm=True)
                # enc_dm_adi = ((vhalo_adi*1e5)**2.*(r*1000.*pc.cgs.value) /
                #               (G.cgs.value * Msun.cgs.value))
                enc_dm_adi = menc_from_vcirc(vhalo_adi, r)
                enc_mass = enc_mass - enc_dm + enc_dm_adi
                enc_dm = enc_dm_adi

        return enc_mass, enc_bary, enc_dm



    def velocity_profile(self, r, compute_dm=False, skip_bulge=False):
        """
        Method to calculate the 1D velocity profile
        as a function of radius
        """

        # First check to make sure there is at least one mass component in the
        # model set.
        if len(self.mass_components) == 0:
            raise AttributeError("There are no mass components so a velocity "
                                 "can't be calculated.")
        else:

            vdm = r*0.
            vbaryon = r*0.

            for cmp in self.mass_components:

                if self.mass_components[cmp]:
                    mcomp = self.components[cmp]
                    if isinstance(mcomp, DiskBulge):
                        cmpnt_v = mcomp.circular_velocity(r, skip_bulge=skip_bulge)
                    else:
                        cmpnt_v = mcomp.circular_velocity(r)
                    if mcomp._subtype == 'dark_matter':

                        vdm = np.sqrt(vdm ** 2 + cmpnt_v ** 2)

                    elif mcomp._subtype == 'baryonic':

                        vbaryon = np.sqrt(vbaryon ** 2 + cmpnt_v ** 2)

                    else:
                        raise TypeError("{} mass model subtype not recognized"
                                        " for {} component. Only 'dark_matter'"
                                        " or 'baryonic' accepted.".format(
                                        mcomp._subtype, cmp))
            vels = self.kinematic_options.apply_adiabatic_contract(r,
                                            vbaryon, vdm,compute_dm=compute_dm)

            if compute_dm:
                vel = vels[0]
                vdm = vels[1]

            else:

                vel = vels

            vel = self.kinematic_options.apply_pressure_support(r, self, vel)
            
            if compute_dm:
                return vel, vdm
            else:
                return vel
                
    def circular_velocity(self, r):
        vel = self.velocity_profile(r, compute_dm=False)
        vcirc = self.kinematic_options.correct_for_pressure_support(r, self, vel)
        return vcirc
    
    def get_vmax(self, r=None):
        if r is None:
            r = np.linspace(0., 25., num=251, endpoint=True)
        
        vel = self.velocity_profile(r, compute_dm=False, skip_bulge=True)
        
        vmax = vel.max()
        return vmax
        
    def write_vrot_vcirc_file(self, r=None, filename=None):
        self.write_profile_file(r=r, filename=filename, 
                cols=['velocity_profile', 'circular_velocity'], 
                colnames=['vrot', 'vcirc'], colunits=['[km/s]', '[km/s]'])
                
        # if r is None:     r = np.arange(0., 10.+0.1, 0.1)  # stepsize 0.1 kpc
        #     
        # vrot = self.velocity_profile(r)
        # vcirc = self.circular_velocity(r)
        #     
        # with open(filename, 'w') as f:
        #     namestr = '#    r      vrot      vcirc'
        #     f.write(namestr+'\n')
        #     unitstr = '#   [kpc]    [km/s]      [km/s]'
        #     f.write(unitstr+'\n')
        #     
        #     for i in six.moves.xrange(len(r)):
        #         datastr = '{:0.1f}  {:0.3f}   {:0.3f}'.format(r[i], vrot[i], vcirc[i])
        #         f.write(datstr+'\n')
                
        
    def write_profile_file(self, r=None, filename=None, 
            cols=None, prettycolnames=None, colunits=None):
        """
            Input:
                filename:        output filename to write to. Will be written as ascii, w/ space delimiter.
                
                cols:            the names of ModelSet methods that will be called as function of r, 
                                 and to be  saved as a column in the output file
                prettycolnames:  alternate column names for output in file header (eg, 'vrot' not 'velocity_profile')
                
            Optional:
                colunits:        units of each column. r is added by hand, and will always be in kpc.
        """
        if cols is None:        cols = ['velocity_profile', 'circular_velocity', 'get_dm_aper']
        if prettycolnames is None:    prettycolnames = cols
        if r is None:           r = np.arange(0., 10.+0.1, 0.1)  # stepsize 0.1 kpc
            
        profiles = np.zeros((len(r), len(cols)+1))
        profiles[:,0] = r
        for j in six.moves.xrange(len(cols)):
            try:
                fnc = getattr(self, cols[j])
                arr = fnc(r)
            except:
                arr = np.ones(len(r))*-99.
            profiles[:, j+1] = arr
        
        colsout = ['r']
        colsout.extend(prettycolnames)
        if colunits is not None:
            unitsout = ['kpc']
            unitsout.extend(colunits)
        
        with open(filename, 'w') as f:
            namestr = '#   ' + '   '.join(colsout)
            f.write(namestr+'\n')
            if colunits is not None:
                unitstr = '#   ' + '   '.join(unitsout)
                f.write(unitstr+'\n')
            for i in six.moves.xrange(len(r)):
                datstr = '    '.join(["{0:0.3f}".format(p) for p in profiles[i,:]])
                f.write(datstr+'\n')
            
    
    def simulate_cube(self, nx_sky, ny_sky, dscale, rstep,
                      spec_type, spec_step, spec_start, nspec,
                      spec_unit=u.km/u.s, line_center=None, oversample=1, oversize=1,
                      debug=False):

        """Simulate an IFU cube of this model set"""

        # Start with a 3D array in the sky coordinate system
        # x and y sizes are user provided so we just need
        # the z size where z is in the direction of the L.O.S.
        # We'll just use the maximum of the given x and y

        nx_sky_samp = nx_sky*oversample*oversize
        ny_sky_samp = ny_sky*oversample*oversize
        rstep_samp = rstep/oversample

        if (np.mod(nx_sky, 2) == 1) & (np.mod(oversize, 2) == 0) & (oversize > 1):
            nx_sky_samp = nx_sky_samp + 1

        if (np.mod(ny_sky, 2) == 1) & (np.mod(oversize, 2) == 0) & (oversize > 1):
            ny_sky_samp = ny_sky_samp + 1

        #nz_sky_samp = np.max([nx_sky_samp, ny_sky_samp])

        # Setup the final IFU cube
        spec = np.arange(nspec) * spec_step + spec_start
        if spec_type == 'velocity':
            vx = (spec * spec_unit).to(u.km / u.s).value
        elif spec_type == 'wavelength':
            if line_center is None:
                raise ValueError("line_center must be provided if spec_type is "
                                 "'wavelength.'")
            line_center_conv = line_center.to(spec_unit).value
            vx = (spec - line_center_conv) / line_center_conv * apy_con.c.to(
                u.km / u.s).value

        cube_final = np.zeros((nspec, ny_sky_samp, nx_sky_samp))
        
        v_sys = self.geometry.vel_shift.value  # systemic velocity

        # First construct the cube based on mass components
        if sum(self.mass_components.values()) > 0:

            # Create 3D arrays of the sky pixel coordinates
            cos_inc = np.cos(self.geometry.inc*np.pi/180.)
            maxr = np.sqrt(nx_sky_samp**2 + ny_sky_samp**2)
            maxr_y = np.max(np.array([maxr*1.5, np.min(
                np.hstack([maxr*1.5/ cos_inc, maxr * 5.]))]))
            #nz_sky_samp = np.int(np.max([nx_sky_samp, ny_sky_samp, maxr_y]))
            nz_sky_samp = np.int(np.max([nx_sky_samp, ny_sky_samp]))
            if np.mod(nz_sky_samp, 2) < 0.5:
                nz_sky_samp += 1

            sh = (nz_sky_samp, ny_sky_samp, nx_sky_samp)
            zsky, ysky, xsky = np.indices(sh)
            zsky = zsky - (nz_sky_samp - 1) / 2.
            ysky = ysky - (ny_sky_samp - 1) / 2.
            xsky = xsky - (nx_sky_samp - 1) / 2.

            # Apply the geometric transformation to get galactic coordinates
            # Need to account for oversampling in the x and y shift parameters
            self.geometry.xshift = self.geometry.xshift.value * oversample
            self.geometry.yshift = self.geometry.yshift.value * oversample
            xgal, ygal, zgal = self.geometry(xsky, ysky, zsky)

            # The circular velocity at each position only depends on the radius
            # Convert to kpc
            rgal = np.sqrt(xgal ** 2 + ygal ** 2) * rstep_samp / dscale
            vcirc = self.velocity_profile(rgal)
            # L.O.S. velocity is then just vcirc*sin(i)*cos(theta) where theta
            # is the position angle in the plane of the disk
            # cos(theta) is just xgal/rgal
            vobs_mass = v_sys + (vcirc * np.sin(np.radians(self.geometry.inc.value)) *
                    xgal / (rgal / rstep_samp * dscale))
            vobs_mass[rgal == 0] = 0.

            # Calculate "flux" for each position

            flux_mass = np.zeros(vobs_mass.shape)

            for cmp in self.light_components:
                if self.light_components[cmp]:
                    zscale = self.zprofile(zgal * rstep_samp / dscale)
                    flux_mass += self.components[cmp].mass_to_light(rgal) * zscale

            # The final spectrum will be a flux weighted sum of Gaussians at each
            # velocity along the line of sight.
            sigmar = self.dispersion_profile(rgal)
            cube_final += cutils.populate_cube(flux_mass, vobs_mass, sigmar, vx)
            
            self.geometry.xshift = self.geometry.xshift.value / oversample
            self.geometry.yshift = self.geometry.yshift.value / oversample

        if self.outflow is not None:

            if self.outflow._spatial_type == 'resolved':
                # Create 3D arrays of the sky pixel coordinates
                sin_inc = np.sin(self.outflow_geometry.inc * np.pi / 180.)
                maxr = np.sqrt(nx_sky_samp ** 2 + ny_sky_samp ** 2)
                maxr_y = np.max(np.array([maxr * 1.5, np.min(
                    np.hstack([maxr * 1.5 / sin_inc, maxr * 5.]))]))
                nz_sky_samp = np.int(np.max([nx_sky_samp, ny_sky_samp, maxr_y]))
                if np.mod(nz_sky_samp, 2) < 0.5:
                    nz_sky_samp += 1

                sh = (nz_sky_samp, ny_sky_samp, nx_sky_samp)
                zsky, ysky, xsky = np.indices(sh)
                zsky = zsky - (nz_sky_samp - 1) / 2.
                ysky = ysky - (ny_sky_samp - 1) / 2.
                xsky = xsky - (nx_sky_samp - 1) / 2.
                # Apply the geometric transformation to get outflow coordinates
                # Account for oversampling
                self.outflow_geometry.xshift = self.outflow_geometry.xshift.value * oversample
                self.outflow_geometry.yshift = self.outflow_geometry.yshift.value * oversample
                xout, yout, zout = self.outflow_geometry(xsky, ysky, zsky)

                # Convert to kpc
                xout_kpc = xout * rstep_samp / dscale
                yout_kpc = yout * rstep_samp / dscale
                zout_kpc = zout * rstep_samp / dscale

                rout = np.sqrt(xout**2 + yout**2 + zout**2)
                vout = self.outflow(xout_kpc, yout_kpc, zout_kpc)
                fout = self.outflow.light_profile(xout_kpc, yout_kpc, zout_kpc)

                # L.O.S. velocity is v*cos(alpha) = -v*zsky/rsky
                vobs = v_sys -vout * zsky/rout
                vobs[rout == 0] = vout[rout == 0]

                sigma_out = self.outflow_dispersion(rout)
                #for zz in range(nz_sky_samp):
                #    f_cube = np.tile(fout[zz, :, :], (nspec, 1, 1))
                #    vobs_cube = np.tile(vobs[zz, :, :], (nspec, 1, 1))
                #    sig_cube = np.tile(sigma_out[zz, :, :], (nspec, 1, 1))
                #    tmp_cube = np.exp(
                #        -0.5 * ((velcube - vobs_cube) / sig_cube) ** 2)
                #    cube_sum = np.nansum(tmp_cube, 0)
                #    cube_sum[cube_sum == 0] = 1
                #    cube_final += tmp_cube / cube_sum * f_cube
                cube_final += cutils.populate_cube(fout, vobs, sigma_out, vx)
                
                self.outflow_geometry.xshift = self.outflow_geometry.xshift.value / oversample
                self.outflow_geometry.yshift = self.outflow_geometry.yshift.value / oversample

            elif self.outflow._spatial_type == 'unresolved':

                # Set where the unresolved will be located and account for oversampling
                xshift = self.outflow_geometry.xshift.value * oversample
                yshift = self.outflow_geometry.yshift.value * oversample

                # The coordinates where the unresolved outflow is placed needs to be
                # an integer pixel so for now we round to nearest integer.
                xpix = np.int(np.round(xshift)) + nx_sky_samp/2
                ypix = np.int(np.round(yshift)) + ny_sky_samp/2

                voutflow = v_sys + self.outflow(vx)
                cube_final[:, ypix, xpix] += voutflow
                
                xshift = self.outflow_geometry.xshift.value / oversample
                yshift = self.outflow_geometry.yshift.value / oversample

        if debug:
            return cube_final, spec, flux_mass, vobs_mass

        return cube_final, spec


# ***** Mass Component Model Classes ******
# Base abstract mass model component class
class _DysmalModel(Model):

    parameter_constraints = DysmalParameter.constraints

    @property
    def prior(self):
        return self._constraints['prior']


class _DysmalFittable1DModel(_DysmalModel):

    linear = False
    fit_deriv = None
    col_fit_deriv = True
    fittable = True

    inputs = ('x',)
    outputs = ('y',)


class MassModel(_DysmalFittable1DModel):

    _type = 'mass'

    @abc.abstractmethod
    def enclosed_mass(self, *args, **kwargs):
        """Evaluate the enclosed mass as a function of radius"""

    def circular_velocity(self, r):
        """
        Default method to evaluate the circular velocity
        as a function of radius using the standard equation:
        v(r) = SQRT(GM(r)/r)
        """

        mass_enc = self.enclosed_mass(r)
        # vcirc = np.sqrt(G.cgs.value * mass_enc * Msun.cgs.value /
        #                 (r * 1000. * pc.cgs.value))
        # vcirc = vcirc/1e5
        
        vcirc = v_circular(mass_enc, r)

        return vcirc


class Sersic(MassModel):
    """
    1D Sersic mass model with parameters defined by the total mass,
    Sersic index, and effective radius.
    """

    total_mass = DysmalParameter(default=1, bounds=(5, 14))
    r_eff = DysmalParameter(default=1, bounds=(0, 50))
    n = DysmalParameter(default=1, bounds=(0, 8))

    _subtype = 'baryonic'

    def __init__(self, total_mass, r_eff, n, invq=1.0, noord_flat=False,
                 **kwargs):
        self.invq = invq
        self.noord_flat = noord_flat
        super(Sersic, self).__init__(total_mass, r_eff, n, **kwargs)

    @staticmethod
    def evaluate(r, total_mass, r_eff, n):
        """
        1D Sersic profile parameterized by the total mass and
        effective radius
        """

        return sersic_mr(r, 10**total_mass, n, r_eff)

    def enclosed_mass(self, r):
        """
        Calculate the enclosed mass as a function of radius
        :param r: Radii at which to calculate the enclosed mass
        :return: 1D enclosed mass profile
        """

        return sersic_menc(r, 10**self.total_mass, self.n, self.r_eff)

    def circular_velocity(self, r):

        if self.noord_flat:

            vcirc = apply_noord_flat(r, self.r_eff, 10**self.total_mass,
                                     self.n, self.invq)

        else:

            vcirc = super(Sersic, self).circular_velocity(r)

        return vcirc

    def mass_to_light(self, r):

        return sersic_mr(r, 1.0, self.n, self.r_eff)


class DiskBulge(MassModel):
    """Class with a combined disk and bulge to allow varying of B/T and the
    total baryonic mass"""

    total_mass = DysmalParameter(default=10, bounds=(5, 14))
    r_eff_disk = DysmalParameter(default=1, bounds=(0, 50))
    n_disk = DysmalParameter(default=1, fixed=True, bounds=(0, 8))
    r_eff_bulge = DysmalParameter(default=1, bounds=(0, 50))
    n_bulge = DysmalParameter(default=4., fixed=True, bounds=(0, 8))
    bt = DysmalParameter(default=0.2, bounds=(0, 1))

    _subtype = 'baryonic'

    def __init__(self, total_mass, r_eff_disk, n_disk, r_eff_bulge,
                 n_bulge, bt, invq_disk=5, invq_bulge=1, noord_flat=False,
                 light_component='disk', **kwargs):

        self.invq_disk = invq_disk
        self.invq_bulge = invq_bulge
        self.noord_flat = noord_flat
        self.light_component = light_component

        super(DiskBulge, self).__init__(total_mass, r_eff_disk, n_disk,
                                        r_eff_bulge, n_bulge, bt, **kwargs)

    @staticmethod
    def evaluate(r, total_mass, r_eff_disk, n_disk, r_eff_bulge, n_bulge, bt):

        mbulge_total = 10**total_mass*bt
        mdisk_total = 10**total_mass*(1 - bt)

        mr_bulge = sersic_mr(r, mbulge_total, n_bulge, r_eff_bulge)
        mr_disk = sersic_mr(r, mdisk_total, n_disk, r_eff_disk)

        return mr_bulge+mr_disk

    def enclosed_mass(self, r):

        mbulge_total = 10 ** self.total_mass * self.bt
        mdisk_total = 10 ** self.total_mass * (1 - self.bt)

        menc_bulge = sersic_menc(r, mbulge_total, self.n_bulge,
                                 self.r_eff_bulge)
        menc_disk = sersic_menc(r, mdisk_total, self.n_disk,
                                self.r_eff_disk)

        return menc_disk+menc_bulge
        
    def enclosed_mass_disk(self, r):
        mdisk_total = 10 ** self.total_mass * (1 - self.bt)
        
        menc_disk = sersic_menc(r, mdisk_total, self.n_disk,
                                self.r_eff_disk)
        return menc_disk
        
    def enclosed_mass_bulge(self, r):
        mbulge_total = 10 ** self.total_mass * self.bt
    
        menc_bulge= sersic_menc(r, mbulge_total, self.n_bulge,
                                 self.r_eff_bulge)
        return menc_bulge

    def circular_velocity_disk(self, r):
        if self.noord_flat:
            mdisk_total = 10**self.total_mass*(1-self.bt)
            vcirc = apply_noord_flat(r, self.r_eff_disk, mdisk_total,
                                     self.n_disk, self.invq_disk)

        else:
            mass_enc = self.enclosed_mass_disk(r)
            vcirc = v_circular(mass_enc, r)

        return vcirc
        
    def circular_velocity_bulge(self, r):
        if self.noord_flat:
            mbulge_total = 10**self.total_mass*self.bt
            vcirc = apply_noord_flat(r, self.r_eff_bulge, mbulge_total,
                                     self.n_bulge, self.invq_bulge)
        else:
            mass_enc = self.enclosed_mass_bulge(r)
            vcirc = v_circular(mass_enc, r)

        return vcirc
        
    def circular_velocity(self, r, skip_bulge=False):

        #if self.noord_flat:
        # mbulge_total = 10**self.total_mass*self.bt
        # mdisk_total = 10**self.total_mass*(1-self.bt)
        # 
        # vbulge = apply_noord_flat(r, self.r_eff_bulge, mbulge_total,
        #                          self.n_bulge, self.invq_bulge)
        # vdisk = apply_noord_flat(r, self.r_eff_disk, mdisk_total,
        #                          self.n_disk, self.invq_disk)
        # 
        
        if skip_bulge:
            vdisk = self.circular_velocity_disk(r)
        
            vcirc = vdisk
        else:
            vbulge = self.circular_velocity_bulge(r)
            vdisk = self.circular_velocity_disk(r)
        
            vcirc = np.sqrt(vbulge**2 + vdisk**2)

        # else:
        # 
        #     vcirc = super(DiskBulge, self).circular_velocity(r)

        return vcirc
        
    def velocity_profile(self, r, modelset):
        vcirc = self.circular_velocity(r)
        vrot = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc)
        return vrot
        
    def velocity_profile_disk(self, r, modelset):
        vcirc = self.circular_velocity_disk(r)
        vrot = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc)
        return vrot
        
    def velocity_profile_bulge(self, r, modelset):
        vcirc = self.circular_velocity_bulge(r)
        vrot = modelset.kinematic_options.apply_pressure_support(r, modelset, vcirc)
        return vrot
        

    def mass_to_light(self, r):

        if self.light_component == 'disk':

            flux = sersic_mr(r, 1.0, self.n_disk, self.r_eff_disk)

        elif self.light_component == 'bulge':

            flux = sersic_mr(r, 1.0, self.n_bulge, self.r_eff_bulge)

        elif self.light_component == 'total':

            flux_disk = sersic_mr(r, 1.0-self.bt,
                                  self.n_disk, self.r_eff_disk)
            flux_bulge = sersic_mr(r, self.bt,
                                   self.n_bulge, self.r_eff_bulge)
            flux = flux_disk + flux_bulge

        else:

            raise ValueError("light_component can only be 'disk', 'bulge', "
                             "or 'total.'")

        return flux


class DarkMatterHalo(MassModel):
    """
    Generic class for dark matter halo profiles
    """

    # Standard parameters for a dark matter halo profile
    mvirial = DysmalParameter(default=1.0, bounds=(5, 20))
    conc = DysmalParameter(default=5.0, bounds=(6, 20))

    _subtype = 'dark_matter'

    @abc.abstractmethod
    def calc_rvir(self, *args, **kwargs):
        """
        Method to calculate the virial radius
        """

    @abc.abstractmethod
    def calc_rho0(self, *args, **kwargs):
        """
        Method to calculate the scale density
        """

    def velocity_profile(self, r, model):
        """
        Calculate velocity profile, including any adiabatic contraction
        """

        if model.kinematic_options.adiabatic_contract:
            raise NotImplementedError("Adiabatic contraction not currently supported!")
        else:
            return self.circular_velocity(r)


class TwoPowerHalo(MassModel):
    """
    Class for a generic two power law density model for a dark matter halo
    See Equation 2.64 of Binney & Tremaine 'Galactic Dynamics'
    """

    # Powerlaw slopes for the density model
    mvirial = DysmalParameter(default=1.0, bounds=(5, 20))
    conc = DysmalParameter(default=5.0, bounds=(6, 20))
    alpha = DysmalParameter(default=1.0)
    beta = DysmalParameter(default=3.0)

    _subtype = 'dark_matter'

    def __init__(self, mvirial, conc, alpha, beta, z=0, cosmo=_default_cosmo,
                 **kwargs):
        self.z = z
        #self.alpha = alpha
        #self.beta = beta
        self.cosmo = cosmo
        super(TwoPowerHalo, self).__init__(mvirial, conc, alpha, beta, **kwargs)

    def evaluate(self, r, mvirial, conc, alpha, beta):

        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0()
        rs = rvirial / self.conc

        return rho0 / ((r/rs)**alpha * (1 + r/rs)**(beta - alpha))

    def enclosed_mass(self, r):

        rvirial = self.calc_rvir()
        rs = rvirial/self.conc
        aa = 10**self.mvirial*(r/rvirial)**(3 - self.alpha)
        bb = (scp_spec.hyp2f1(3-self.alpha, self.beta-self.alpha, 4-self.alpha, -r/rs) /
              scp_spec.hyp2f1(3 - self.alpha, self.beta - self.alpha, 4 - self.alpha, -self.conc))

        return aa*bb

    def calc_rho0(self):

        rvir = self.calc_rvir()
        rs = rvir/self.conc
        aa = -10**self.mvirial/(4*np.pi*self.conc**(3-self.alpha)*rs**3)
        bb = (self.alpha - 3) / scp_spec.hyp2f1(3-self.alpha, self.beta-self.alpha, 4-self.alpha, -self.conc)

        return aa*bb

    def calc_rvir(self):
        """
        Calculate the virial radius based on virial mass and redshift
        M_vir = 100*H(z)^2/G * R_vir^3
        """

        g_new_unit = G.to(u.pc / u.Msun * (u.km / u.s) ** 2).value
        hz = self.cosmo.H(self.z).value
        rvir = ((10 ** self.mvirial * (g_new_unit * 1e-3) /
                 (10 * hz * 1e-3) ** 2) ** (1. / 3.))

        return rvir


class NFW(DarkMatterHalo):
    """
    1D NFW mass model parameterized by the virial radius, virial mass, and
    concentration.
    """

    def __init__(self, mvirial, conc, z=0, cosmo=_default_cosmo,
                 **kwargs):
        self.z = z
        self.cosmo = cosmo
        super(NFW, self).__init__(mvirial, conc, **kwargs)

    def evaluate(self, r, mvirial, conc):
        """3D mass density profile"""

        rvirial = self.calc_rvir()
        rho0 = self.calc_rho0()
        rs = rvirial / self.conc

        return rho0 / (r / rs * (1 + r / rs) ** 2)

    def enclosed_mass(self, r):
        """
        Calculate the enclosed mass as a function of radius
        :param r: Radii at which to calculate the enclosed mass
        :return: 1D enclosed mass profile
        """

        rho0 = self.calc_rho0()
        rvirial = self.calc_rvir()
        rs = rvirial/self.conc
        aa = 4.*np.pi*rho0*rvirial**3/self.conc**3

        # For very small r, bb can be negative.
        bb = np.abs(np.log((rs + r)/rs) - r/(rs + r))

        return aa*bb

    def calc_rho0(self):

        rvirial = self.calc_rvir()
        aa = 10**self.mvirial/(4.*np.pi*rvirial**3)*self.conc**3
        bb = 1./(np.log(1.+self.conc) - (self.conc/(1.+self.conc)))

        return aa * bb

    def calc_rvir(self):
        """
        Calculate the virial radius based on virial mass and redshift
        M_vir = 100*H(z)^2/G * R_vir^3
        """
        g_new_unit = G.to(u.pc / u.Msun * (u.km / u.s) ** 2).value
        hz = self.cosmo.H(self.z).value
        rvir = ((10 ** self.mvirial * (g_new_unit * 1e-3) /
                (10 * hz * 1e-3) ** 2) ** (1. / 3.))

        return rvir



# ****** Geometric Model ********
class _DysmalFittable3DModel(_DysmalModel):

    linear = False
    fit_deriv = None
    col_fit_deriv = True
    fittable = True

    inputs = ('x', 'y', 'z')

    @property
    def prior(self):
        return self._constraints['prior']


class Geometry(_DysmalFittable3DModel):
    """
    Class to hold the geometric parameters that can be fit.
    Also takes as input the sky coordinates and returns the
    corresponding galaxy plane coordinates.
    
    Convention:
        PA is angle of blue side, CCW from North
    """

    inc = DysmalParameter(default=45.0, bounds=(0, 90))
    pa = DysmalParameter(default=0.0, bounds=(-180, 180))
    xshift = DysmalParameter(default=0.0)
    yshift = DysmalParameter(default=0.0)
    
    vel_shift = DysmalParameter(default=0.0, fixed=True)  # default: none

    _type = 'geometry'
    outputs = ('xp', 'yp', 'zp')

    @staticmethod
    def evaluate(x, y, z, inc, pa, xshift, yshift, vel_shift):

        inc = np.pi / 180. * inc
        pa = np.pi / 180. * (pa - 90.)

        # Apply the shifts in the sky system
        xsky = x - xshift
        ysky = y - yshift
        zsky = z

        xtmp = xsky * np.cos(pa) + ysky * np.sin(pa)
        ytmp = -xsky * np.sin(pa) + ysky * np.cos(pa)
        ztmp = zsky

        xgal = xtmp
        ygal = ytmp * np.cos(inc) - ztmp * np.sin(inc)
        zgal = ytmp * np.sin(inc) + ztmp * np.cos(inc)

        return xgal, ygal, zgal


# ******* Dispersion Profiles **************
class DispersionProfile(_DysmalFittable1DModel):
    """Base dispersion profile model class"""
    _type = 'dispersion'


class DispersionConst(DispersionProfile):

    sigma0 = DysmalParameter(default=10., bounds=(0, None), fixed=True)

    @staticmethod
    def evaluate(r, sigma0):

        return np.ones(r.shape)*sigma0


# ******* Z-Height Profiles ***************
class ZHeightProfile(_DysmalFittable1DModel):
    """Base z-height profile model class"""
    _type = 'zheight'


class ZHeightGauss(ZHeightProfile):

    sigmaz = DysmalParameter(default=1.0, fixed=True, bounds=(0, 10))

    def __init__(self, sigmaz, **kwargs):
        super(ZHeightGauss, self).__init__(sigmaz, **kwargs)

    @staticmethod
    def evaluate(z, sigmaz):
        return np.exp(-0.5*(z/sigmaz)**2)


# ****** Kinematic Options Class **********
class KinematicOptions:
    """
    A simple container for holding the settings for different
    options for calculating the kinematics of a galaxy. Also
    has methods for applying these options.
    """

    def __init__(self, adiabatic_contract=False, pressure_support=False,
                 pressure_support_re=None):
        self.adiabatic_contract = adiabatic_contract
        self.pressure_support = pressure_support
        self.pressure_support_re = pressure_support_re

    def apply_adiabatic_contract(self, r, vbaryon, vhalo, compute_dm=False):

        if self.adiabatic_contract:
            logger.info("Applying adiabatic contraction.")
            rprime_all = np.zeros(len(r))
            for i in range(len(r)):
                result = scp_opt.newton(_adiabatic, r[i] + 1.,
                                        args=(r[i], vhalo, r, vbaryon[i]))
                rprime_all[i] = result

            vhalo_adi_interp = scp_interp.interp1d(r, vhalo,
                                                   fill_value='extrapolate')
            vhalo_adi = vhalo_adi_interp(rprime_all)
            vel = np.sqrt(vhalo_adi ** 2 + vbaryon ** 2)

        else:

            vel = np.sqrt(vhalo ** 2 + vbaryon ** 2)
            
        if compute_dm:
            if self.adiabatic_contract:
                return vel, vhalo_adi
            else:
                return vel, vhalo
        else:
            return vel

    def apply_pressure_support(self, r, model, vel):

        if self.pressure_support:

            if self.pressure_support_re is None:
                pre = None
                for cmp in model.mass_components:
                    if model.mass_components[cmp]:
                        mcomp = model.components[cmp]
                        if mcomp._subtype == 'baryonic':
                            if isinstance(mcomp, DiskBulge):
                                pre = mcomp.r_eff_disk.value
                            else:
                                pre = mcomp.r_eff.value
                            break

                if pre is None:
                    logger.warning("No baryonic mass component found. Using "
                                   "1 kpc as the pressure support effective"
                                   " radius")
                    pre = 1.0

            else:
                pre = self.pressure_support_re

            if model.dispersion_profile is None:
                raise AttributeError("Can't apply pressure support without "
                                     "a dispersion profile!")

            #logger.info("Applying pressure support with effective radius of {} "
            #            "kpc.".format(pre))
            sigma = model.dispersion_profile(r)
            vel_squared = (
                vel ** 2 - 3.36 * (r / pre) * sigma ** 2)
            # if array:
            try:
                vel_squared[vel_squared < 0] = 0.
            else:
                # if float single value:
                if (vel_squared < 0):
                    vel_squared = 0.
            vel = np.sqrt(vel_squared)

        return vel
    
    def correct_for_pressure_support(self, r, model, vel):
        if self.pressure_support:
            if self.pressure_support_re is None:
                pre = None
                for cmp in model.mass_components:
                    if model.mass_components[cmp]:
                        mcomp = model.components[cmp]
                        if mcomp._subtype == 'baryonic':
                            if isinstance(mcomp, DiskBulge):
                                pre = mcomp.r_eff_disk.value
                            else:
                                pre = mcomp.r_eff.value
                            break

                if pre is None:
                    logger.warning("No baryonic mass component found. Using "
                                   "1 kpc as the pressure support effective"
                                   " radius")
                    pre = 1.0

            else:
                pre = self.pressure_support_re

            if model.dispersion_profile is None:
                raise AttributeError("Can't apply pressure support without "
                                     "a dispersion profile!")

            #logger.info("Correcting for pressure support with effective radius of {} "
            #            "kpc.".format(pre))
            sigma = model.dispersion_profile(r)
            vel_squared = (
                vel ** 2 + 3.36 * (r / pre) * sigma ** 2)
            # if array:
            try:
                vel_squared[vel_squared < 0] = 0.
            else:
                # if float single value:
                if (vel_squared < 0):
                    vel_squared = 0.
            vel = np.sqrt(vel_squared)
        return vel

class BiconicalOutflow(_DysmalFittable3DModel):
    """Model for a biconical outflow. Assumption is symmetry above and below
       the vertex and around the outflow axis."""

    n = DysmalParameter(default=0.5, fixed=True)
    vmax = DysmalParameter(min=0)
    rturn = DysmalParameter(default=0.5, min=0)
    thetain = DysmalParameter(bounds=(0, 90))
    dtheta = DysmalParameter(default=20.0, bounds=(0, 90))
    rend = DysmalParameter(default=1.0, min=0)
    norm_flux = DysmalParameter(default=0.0)

    _type = 'outflow'
    _spatial_type = 'resolved'
    outputs = ('vout',)

    def __init__(self, n, vmax, rturn, thetain, dtheta, rend, norm_flux,
                 profile_type='both', tau_flux=5.0, **kwargs):

        valid_profiles = ['increase', 'decrease', 'both', 'constant']

        if profile_type in valid_profiles:
            self.profile_type = profile_type
        else:
            logger.error("Invalid profile type. Must be one of 'increase',"
                         "'decrease', 'constant', or 'both.'")

        self.tau_flux = tau_flux
        #self.norm_flux = norm_flux

        super(BiconicalOutflow, self).__init__(n, vmax, rturn, thetain,
                                               dtheta, rend, norm_flux, **kwargs)

    def evaluate(self, x, y, z, n, vmax, rturn, thetain, dtheta, rend, norm_flux):
        """Evaluate the outflow velocity as a function of position x, y, z"""

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.abs(z)/r)*180./np.pi
        theta[r == 0] = 0.
        vel = np.zeros(r.shape)

        if self.profile_type == 'increase':

            amp = vmax/rend**n
            vel[r <= rend] = amp*r[r <= rend]**n

        elif self.profile_type == 'decrease':

            amp = -vmax/rend**n
            vel[r <= rend] = vmax + amp*r[r <= rend]** n
            vel[r == 0] = 0

        elif self.profile_type == 'both':

            vel[r <= rturn] = vmax*(r[r <= rturn]/rturn)**n
            ind = (r > rturn) & (r <= 2*rturn)
            vel[ind] = vmax*(2 - r[ind]/rturn)**n
            
        elif self.profile_type == 'constant':
            
            vel[r <= rend] = vmax

        thetaout = np.min([thetain+dtheta, 90.])
        ind_zero = (theta < thetain) | (theta > thetaout) | (vel < 0)
        vel[ind_zero] = 0.

        return vel

    def light_profile(self, x, y, z):

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.abs(z) / r) * 180. / np.pi
        theta[r == 0] = 0.
        flux = 10**self.norm_flux*np.exp(-self.tau_flux*(r/self.rend))
        thetaout = np.min([self.thetain + self.dtheta, 90.])
        ind_zero = ((theta < self.thetain) |
                    (theta > thetaout) |
                    (r > self.rend))
        flux[ind_zero] = 0.

        return flux


class UnresolvedOutflow(_DysmalFittable1DModel):
    """
    Model for an unresolved outflow component with a specific flux and width.
    """

    vcenter = DysmalParameter(default=0)
    fwhm = DysmalParameter(default=1000.0, bounds=(0, None))
    amplitude = DysmalParameter(default=1.0, bounds=(0, None))

    _type = 'outflow'
    _spatial_type = 'unresolved'
    outputs = ('vout',)

    @staticmethod
    def evaluate(v, vcenter, fwhm, amplitude):

        return amplitude*np.exp(-(v - vcenter)**2/(fwhm/2.35482)**2)


def _adiabatic(rprime, r_adi, adia_v_dm, adia_x_dm, adia_v_disk):

    if rprime < 0.:
        rprime = 0.1
    if rprime < adia_x_dm[1]:
        rprime = adia_x_dm[1]
    rprime_interp = scp_interp.interp1d(adia_x_dm, adia_v_dm,
                                        fill_value="extrapolate")
    result = (r_adi + r_adi * ((r_adi*adia_v_disk**2) /
                               (rprime*(rprime_interp(rprime))**2)) - rprime)
    return result
