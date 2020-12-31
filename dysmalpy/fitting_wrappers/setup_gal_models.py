# Methods for setting up galaxies / models for fitting fitting_wrappers

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys

import numpy as np
import astropy.units as u
import astropy.constants as apy_con

from dysmalpy import parameters
from dysmalpy import fitting
from dysmalpy import galaxy, instrument, models
from dysmalpy import aperture_classes

try:
    import tied_functions, data_io
except:
    from . import tied_functions, data_io

import emcee


# ------------------------------------------------------------
def setup_single_object_1D(params=None, data=None):
    # ------------------------------------------------------------
    # Setup galaxy, instrument, model:
    gal = setup_gal_model_base(params=params)

    # ------------------------------------------------------------
    # Load data:
    if data is None:
        # Setup datadir, if set. If not set (so datadir=None), fdata must be the full path.
        if 'datadir' in params.keys():
            datadir = params['datadir']
        else:
            datadir = None
        if datadir is None:
            datadir = ''

        if 'fdata_mask' in params.keys():
            fdata_mask = params['fdata_mask']
        else:
            fdata_mask = None
        gal.data = data_io.load_single_object_1D_data(fdata=params['fdata'], fdata_mask=fdata_mask, params=params, datadir=datadir)
        gal.data.filename_velocity = datadir+params['fdata']

        if (params['profile1d_type'] != 'circ_ap_pv') & (params['profile1d_type'] != 'single_pix_pv'):
            gal.data.apertures = setup_basic_aperture_types(gal=gal, params=params)
    else:
        gal.data = data
        if gal.data.apertures is None:
            gal.data.apertures = setup_basic_aperture_types(gal=gal, params=params)

    #
    gal.data.profile1d_type = params['profile1d_type']

    # --------------------------------------------------
    # Check FOV and issue warning if too small:
    maxr = np.max(np.abs(gal.data.rarr))
    if (params['fov_npix'] < maxr/params['pixscale']):
        wmsg = "Input FOV 'fov_npix'={}".format(params['fov_npix'])
        wmsg += " is too small for max data extent ({} pix)".format(maxr/params['pixscale'])
        print("WARNING: dysmalpy_fit_single_1D: {}".format(wmsg))
    # --------------------------------------------------

    # ------------------------------------------------------------
    # Setup fitting dict:
    fit_dict = setup_fit_dict(params=params, ndim_data=1)

    return gal, fit_dict


# ------------------------------------------------------------
def setup_single_object_2D(params=None, data=None):
    # ------------------------------------------------------------
    # Setup galaxy, instrument, model:

    gal = setup_gal_model_base(params=params)

    # ------------------------------------------------------------
    # Load data:
    if data is None:
        gal.data = data_io.load_single_object_2D_data(params=params)
    else:
        gal.data = data

    # ------------------------------------------------------------
    # Setup fitting dict:
    fit_dict = setup_fit_dict(params=params, ndim_data=2)

    return gal, fit_dict


# ------------------------------------------------------------
def setup_single_object_3D(params=None, data=None):
    # ------------------------------------------------------------
    # Load data:
    if data is None:
        data = data_io.load_single_object_3D_data(params=params)


    # ------------------------------------------------------------
    # Setup galaxy, instrument, model:

    gal = setup_gal_model_base(params=params)

    # Override FOV from the cube shape:
    gal.instrument.fov = [data.shape[2], data.shape[1]]

    # ------------------------------------------------------------

    gal.data = data

    # ------------------------------------------------------------
    # Setup fitting dict:
    fit_dict = setup_fit_dict(params=params, ndim_data=3)

    return gal, fit_dict

# ------------------------------------------------------------
def setup_gal_model_base(params=None,
        tied_fdm_func=None,
        tied_mvirial_func_NFW=None,
        tied_mvirial_func_SMHM=None,
        tied_alpha_TPH_func=None,
        tied_rB_Burk_func=None,
        tied_alpha_Ein_func=None,
        tied_n_Ein_func=None,
        tied_sigmaz_func=None):

    if 'components_list' in params.keys():
        components_list = params['components_list']
    else:
        components_list = ['disk+bulge', 'const_disp_prof', 'geometry', 'zheight_gaus']
        if params['include_halo']:
            components_list.append('halo')
            # Or explicitly do NFW / TPH / etc?

    if 'light_components_list' in params.keys():
        light_components_list = params['light_components_list']
    else:
        light_components_list = ['disk']

    # ------------------------------------------------------------
    # Initialize the Galaxy, Instrument, and Model Set
    gal = galaxy.Galaxy(z=params['z'], name=params['galID'])
    mod_set = models.ModelSet()
    inst = instrument.Instrument()

    # ------------------------------------------------------------
    # Baryonic Component: Combined Disk+Bulge
    if 'disk+bulge' in components_list:
        comp_bary = 'disk+bulge'
        mod_set = add_disk_bulge_comp(gal=gal, mod_set=mod_set, params=params,
                            light_components_list=light_components_list)

    # ------------------------------------------------------------
    # Halo Component: (if added)
    # ------------------------------------------------------------
    if 'halo' in components_list:
        mod_set = add_halo_comp(gal=gal, mod_set=mod_set, params=params,
                    tied_fdm_func=tied_fdm_func,
                    tied_mvirial_func_NFW=tied_mvirial_func_NFW,
                    tied_mvirial_func_SMHM=tied_mvirial_func_SMHM,
                    tied_alpha_TPH_func=tied_alpha_TPH_func,
                    tied_rB_Burk_func=tied_rB_Burk_func,
                    tied_alpha_Ein_func=tied_alpha_Ein_func,
                    tied_n_Ein_func=tied_n_Ein_func,
                    comp_bary=comp_bary)

    # ------------------------------------------------------------
    # Dispersion profile
    if 'const_disp_prof' in components_list:
        mod_set = add_const_disp_prof_comp(gal=gal, mod_set=mod_set, params=params)

    # ------------------------------------------------------------
    # z-height profile
    if 'zheight_gaus' in components_list:
        mod_set = add_zheight_gaus_comp(gal=gal, mod_set=mod_set, params=params,
                    tied_sigmaz_func=tied_sigmaz_func)

    # --------------------------------------
    # Geometry
    if 'geometry' in components_list:
        mod_set = add_geometry_comp(gal=gal, mod_set=mod_set, params=params)

    # --------------------------------------
    # Set some kinematic options for calculating the velocity profile
    mod_set.kinematic_options.adiabatic_contract = params['adiabatic_contract']
    mod_set.kinematic_options.pressure_support = params['pressure_support']

    # --------------------------------------
    # Set up the instrument
    inst = setup_instrument_params(inst=inst, params=params)

    # Add the model set and instrument to the Galaxy
    gal.model = mod_set
    gal.instrument = inst

    return gal

def add_disk_bulge_comp(gal=None, mod_set=None, params=None, light_components_list=None):
    total_mass =  params['total_mass']        # log M_sun
    bt =          params['bt']                # Bulge-Total ratio
    r_eff_disk =  params['r_eff_disk']        # kpc
    n_disk =      params['n_disk']
    invq_disk =   params['invq_disk']         # 1/q0, disk
    r_eff_bulge = params['r_eff_bulge']       # kpc
    n_bulge =     params['n_bulge']
    invq_bulge =  params['invq_bulge']
    noord_flat =  params['noord_flat']        # Switch for applying Noordermeer flattening

    # Fix components
    bary_fixed = {'total_mass': params['total_mass_fixed'],
                  'r_eff_disk': params['r_eff_disk_fixed'],
                  'n_disk': params['n_disk_fixed'],
                  'r_eff_bulge': params['r_eff_bulge_fixed'],
                  'n_bulge': params['n_bulge_fixed'],
                  'bt': params['bt_fixed']}

    # Set bounds
    bary_bounds = {'total_mass': (params['total_mass_bounds'][0], params['total_mass_bounds'][1]),
                   'r_eff_disk': (params['r_eff_disk_bounds'][0], params['r_eff_disk_bounds'][1]),
                   'n_disk':     (params['n_disk_bounds'][0], params['n_disk_bounds'][1]),
                   'r_eff_bulge': (params['r_eff_bulge_bounds'][0], params['r_eff_bulge_bounds'][1]),
                   'n_bulge': (params['n_bulge_bounds'][0], params['n_bulge_bounds'][1]),
                   'bt':         (params['bt_bounds'][0], params['bt_bounds'][1])}


    no_baryons = False
    if 'disk' in light_components_list:
        light_component_bary = 'disk'
    elif 'bulge' in light_components_list:
        light_component_bary = 'bulge'
    elif 'disk+bulge' in light_components_list:
        light_component_bary = 'total'
    else:
        no_baryons = True

    bary = models.DiskBulge(total_mass=total_mass, bt=bt,
                            r_eff_disk=r_eff_disk, n_disk=n_disk,
                            invq_disk=invq_disk,
                            r_eff_bulge=r_eff_bulge, n_bulge=n_bulge,
                            invq_bulge=invq_bulge,
                            noord_flat=noord_flat,
                            name='disk+bulge',
                            fixed=bary_fixed, bounds=bary_bounds,
                            light_component=light_component_bary)

    if 'linear_masses' in params.keys():
        if params['linear_masses']:
            bary = models.LinearDiskBulge(total_mass=total_mass, bt=bt,
                                    r_eff_disk=r_eff_disk, n_disk=n_disk,
                                    invq_disk=invq_disk,
                                    r_eff_bulge=r_eff_bulge, n_bulge=n_bulge,
                                    invq_bulge=invq_bulge,
                                    noord_flat=noord_flat,
                                    name='disk+bulge',
                                    fixed=bary_fixed, bounds=bary_bounds)

    bary = set_comp_param_prior(comp=bary, param_name='total_mass', params=params)
    bary = set_comp_param_prior(comp=bary, param_name='bt', params=params)
    bary = set_comp_param_prior(comp=bary, param_name='r_eff_disk', params=params)
    bary = set_comp_param_prior(comp=bary, param_name='n_disk', params=params)
    bary = set_comp_param_prior(comp=bary, param_name='r_eff_bulge', params=params)
    bary = set_comp_param_prior(comp=bary, param_name='n_bulge', params=params)

    # --------------------------------------
    # Add the model component to the ModelSet
    if no_baryons:
        raise ValueError("You must include baryons, they are the light tracers.")
    else:
        mod_set.add_component(bary, light=True)

    return mod_set

def add_halo_comp(gal=None, mod_set=None, params=None,
        tied_fdm_func=None,
        tied_mvirial_func_NFW=None,
        tied_mvirial_func_SMHM=None,
        tied_alpha_TPH_func=None,
        tied_rB_Burk_func=None,
        tied_alpha_Ein_func=None,
        tied_n_Ein_func=None,
        comp_bary='disk+bulge'):

    if tied_fdm_func is None:
        tied_fdm_func = tied_functions.tie_fdm
    if tied_mvirial_func_NFW is None:
        tied_mvirial_func_NFW = tied_functions.tie_lmvirial_NFW
    if tied_mvirial_func_SMHM is None:
        tied_mvirial_func_SMHM = tied_functions.tied_mhalo_mstar
    if tied_alpha_TPH_func is None:
        tied_alpha_TPH_func = tied_functions.tie_alpha_TwoPower
    if tied_rB_Burk_func is None:
        tied_rB_Burk_func = tied_functions.tie_rB_Burkert
    if tied_alpha_Ein_func is None:
        tied_alpha_Ein_func = tied_functions.tie_alphaEinasto_Einasto
    if tied_n_Ein_func is None:
        tied_n_Ein_func = tied_functions.tie_nEinasto_Einasto

    # Halo component
    if (params['halo_profile_type'].strip().upper() == 'NFW'):

        # NFW halo fit:
        mvirial =                   params['mvirial']
        conc =                      params['halo_conc']
        fdm =                       params['fdm']

        halo_fixed = {'mvirial':    params['mvirial_fixed'],
                      'conc':       params['halo_conc_fixed'],
                      'fdm':        params['fdm_fixed']}

        halo_bounds = {'mvirial':   (params['mvirial_bounds'][0], params['mvirial_bounds'][1]),
                       'conc':      (params['halo_conc_bounds'][0], params['halo_conc_bounds'][1]),
                       'fdm':       (params['fdm_bounds'][0], params['fdm_bounds'][1])}

        halo = models.NFW(mvirial=mvirial, conc=conc, fdm=fdm, z=gal.z,
                          fixed=halo_fixed, bounds=halo_bounds, name='halo')
        #
        if 'linear_masses' in params.keys():
            if params['linear_masses']:
                halo = models.LinearNFW(mvirial=mvirial, conc=conc, fdm=fdm, z=gal.z,
                                  fixed=halo_fixed, bounds=halo_bounds, name='halo')

        halo = set_comp_param_prior(comp=halo, param_name='mvirial', params=params)
        halo = set_comp_param_prior(comp=halo, param_name='halo_conc', params=params)
        halo = set_comp_param_prior(comp=halo, param_name='fdm', params=params)

        if (params['fdm_fixed'] is False) and (params['mvirial_fixed'] is False):
            fdm_tied = mvir_tied = False
            if 'mvirial_tied' in params.keys():
                if params['mvirial_tied']:
                    mvir_tied = True
                else:
                    mvir_tied = False
            if 'fdm_tied' in params.keys():
                if params['fdm_tied']:
                    fdm_tied = True
                else:
                    fdm_tied = False

            if (not fdm_tied) and (not mvir_tied):
                msg = "For the NFW halo, 'fdm' and 'mvirial' cannot both be free,\n"
                msg += "if one is not tied to the other. Setting 'mvirial=Tied'.\n"
                msg += "Alternatively, specify 'mvirial_tied, True' (to fit fdm)\n"
                msg += " or 'fdm_tied, True' (to fit mvirial)."
                print(msg)

                params['mvirial_tied'] = True

        elif ((not params['fdm_fixed']) & (params['mvirial_fixed'])):
            if 'mvirial_tied' not in params.keys():
                params['mvirial_tied'] = True
            else:
                if (not params['mvirial_tied']):
                    # Override setting and make tied, as it can't be truly fixed
                    params['mvirial_tied'] = True
        elif ((not params['mvirial_fixed']) & (params['fdm_fixed'])):
            if 'fdm_tied' not in params.keys():
                params['fdm_tied'] = True
            else:
                if (not params['fdm_tied']):
                    # Override setting and make tied, as it can't be truly fixed
                    params['fdm_tied'] = True

        if 'fdm_tied' in params.keys():
            if params['fdm_tied']:
                # Tie fDM to the virial mass
                halo.fdm.tied = tied_fdm_func
                halo.fdm.fixed = False


        if 'mvirial_tied' in params.keys():
            if params['mvirial_tied']:
                # Tie the virial mass to fDM
                halo.mvirial.tied = tied_mvirial_func_NFW
                halo.mvirial.fixed = False

    elif (params['halo_profile_type'].strip().upper() == 'TWOPOWERHALO'):
        # Two-power halo fit:

        # Add values needed:
        mod_set.components[comp_bary].lmstar = params['lmstar']
        mod_set.components[comp_bary].fgas =  params['fgas']
        mod_set.components[comp_bary].mhalo_relation = params['mhalo_relation']
        mod_set.components[comp_bary].truncate_lmstar_halo = params['truncate_lmstar_halo']

        # Setup parameters:
        mvirial =  params['mvirial']
        conc =     params['halo_conc']
        alpha =    params['alpha']
        beta =     params['beta']
        fdm =      params['fdm']

        halo_fixed = {'mvirial':    params['mvirial_fixed'],
                      'conc':       params['halo_conc_fixed'],
                      'alpha':      params['alpha_fixed'],
                      'beta':       params['beta_fixed'],
                      'fdm':        params['fdm_fixed']}

        halo_bounds = {'mvirial':   (params['mvirial_bounds'][0], params['mvirial_bounds'][1]),
                       'conc':      (params['halo_conc_bounds'][0], params['halo_conc_bounds'][1]),
                       'alpha':     (params['alpha_bounds'][0], params['alpha_bounds'][1]),
                       'beta':      (params['beta_bounds'][0], params['beta_bounds'][1]),
                       'fdm':       (params['fdm_bounds'][0], params['fdm_bounds'][1]) }

        halo = models.TwoPowerHalo(mvirial=mvirial, conc=conc,
                            alpha=alpha, beta=beta, fdm=fdm, z=gal.z,
                            fixed=halo_fixed, bounds=halo_bounds, name='halo')

        # Tie the virial mass to Mstar
        if params['mvirial_tied']:
            halo.mvirial.tied = tied_mvirial_func_SMHM

        if 'fdm_tied' in params.keys():
            if params['fdm_tied']:
                # Tie fDM to the virial mass
                halo.fdm.tied = tied_fdm_func
                halo.fdm.fixed = False
        else:
            params['fdm_tied'] = False

        #if (params['fdm_fixed'] is False) & (not params['fdm_tied']):
        if 'alpha_tied' in params.keys():
            if params['alpha_tied']:
                # Tie the alpha mass to fDM
                halo.alpha.tied = tied_alpha_TPH_func
                halo.alpha.fixed = False


        halo = set_comp_param_prior(comp=halo, param_name='mvirial', params=params)
        halo = set_comp_param_prior(comp=halo, param_name='halo_conc', params=params)
        halo = set_comp_param_prior(comp=halo, param_name='alpha', params=params)
        halo = set_comp_param_prior(comp=halo, param_name='beta', params=params)
        halo = set_comp_param_prior(comp=halo, param_name='fdm', params=params)



    elif (params['halo_profile_type'].strip().upper() == 'BURKERT'):
        # Burkert halo profile:

        # Add values needed:
        mod_set.components[comp_bary].lmstar = params['lmstar']
        mod_set.components[comp_bary].fgas =  params['fgas']
        mod_set.components[comp_bary].mhalo_relation = params['mhalo_relation']
        mod_set.components[comp_bary].truncate_lmstar_halo = params['truncate_lmstar_halo']

        # Setup parameters:
        mvirial =  params['mvirial']
        rB =       params['rB']
        fdm =      params['fdm']

        halo_fixed = {'mvirial':    params['mvirial_fixed'],
                      'rB':         params['rB_fixed'],
                      'fdm':        params['fdm_fixed']}

        halo_bounds = {'mvirial':   (params['mvirial_bounds'][0], params['mvirial_bounds'][1]),
                       'rB':        (params['rB_bounds'][0], params['rB_bounds'][1]),
                       'fdm':       (params['fdm_bounds'][0], params['fdm_bounds'][1]) }

        halo = models.Burkert(mvirial=mvirial, rB=rB, fdm=fdm, z=gal.z,
                          fixed=halo_fixed, bounds=halo_bounds, name='halo')

        # Tie the virial mass to Mstar
        if params['mvirial_tied']:
            halo.mvirial.tied = tied_mvirial_func_SMHM


        #if params['fdm_fixed'] is False:
        if 'rB_tied' in params.keys():
            if params['rB_tied']:
                # Tie the rB to fDM
                halo.rB.tied = tied_rB_Burk_func

        halo = set_comp_param_prior(comp=halo, param_name='mvirial', params=params)
        halo = set_comp_param_prior(comp=halo, param_name='rB', params=params)
        halo = set_comp_param_prior(comp=halo, param_name='fdm', params=params)

    elif (params['halo_profile_type'].strip().upper() == 'EINASTO'):
        # Einastro halo profile:
        # Add values needed:
        mod_set.components[comp_bary].lmstar = params['lmstar']
        mod_set.components[comp_bary].fgas =  params['fgas']
        mod_set.components[comp_bary].mhalo_relation = params['mhalo_relation']
        mod_set.components[comp_bary].truncate_lmstar_halo = params['truncate_lmstar_halo']

        # Setup parameters:
        mvirial =           params['mvirial']
        fdm =               params['fdm']
        conc =              params['halo_conc']

        halo_fixed = {'mvirial':        params['mvirial_fixed'],
                      'conc':           params['halo_conc_fixed'],
                      'fdm':            params['fdm_fixed']}

        halo_bounds = {'mvirial':       (params['mvirial_bounds'][0], params['mvirial_bounds'][1]),
                       'conc':          (params['halo_conc_bounds'][0], params['halo_conc_bounds'][1]),
                       'fdm':           (params['fdm_bounds'][0], params['fdm_bounds'][1]) }

        if 'alphaEinasto' in params.keys():
            alphaEinasto =                  params['alphaEinasto']
            halo_fixed['alphaEinasto'] =    params['alphaEinasto_fixed']
            halo_bounds['alphaEinasto'] =   (params['alphaEinasto_bounds'][0],
                                             params['alphaEinasto_bounds'][1])
            halo = models.Einasto(mvirial=mvirial, alphaEinasto=alphaEinasto, conc=conc, fdm=fdm, z=gal.z,
                          fixed=halo_fixed, bounds=halo_bounds, name='halo')
            halo = set_comp_param_prior(comp=halo, param_name='alphaEinasto', params=params)

        elif 'nEinasto' in params.keys():
            nEinasto =                  params['nEinasto']
            halo_fixed['nEinasto'] =    params['nEinasto_fixed']
            halo_bounds['nEinasto'] =   (params['nEinasto_bounds'][0], params['nEinasto_bounds'][1])
            halo = models.Einasto(mvirial=mvirial, nEinasto=nEinasto, conc=conc, fdm=fdm, z=gal.z,
                          fixed=halo_fixed, bounds=halo_bounds, name='halo')

            halo = set_comp_param_prior(comp=halo, param_name='nEinasto', params=params)

        # Tie the virial mass to Mstar
        if params['mvirial_tied']:
            halo.mvirial.tied = tied_mvirial_func_SMHM


        #if params['fdm_fixed'] is False:
        if 'alphaEinasto_tied' in params.keys():
            if params['alphaEinasto_tied']:
                # Tie the Einasto param to fDM
                halo.alphaEinasto.tied = tied_alpha_Ein_func
        if 'nEinasto_tied' in params.keys():
            if params['nEinasto_tied']:
                # Tie the Einasto param to fDM
                halo.nEinasto.tied = tied_n_Ein_func

        halo = set_comp_param_prior(comp=halo, param_name='mvirial', params=params)
        halo = set_comp_param_prior(comp=halo, param_name='fdm', params=params)

    else:
        raise ValueError("{} halo profile type not recognized!".format(params['halo_profile_type']))

    # --------------------------------------
    # Add the model component to the ModelSet
    mod_set.add_component(halo)

    return mod_set

def add_const_disp_prof_comp(gal=None, mod_set=None, params=None):

    sigma0 = params['sigma0']       # km/s
    disp_fixed = {'sigma0': params['sigma0_fixed']}
    disp_bounds = {'sigma0': (params['sigma0_bounds'][0], params['sigma0_bounds'][1])}

    disp_prof = models.DispersionConst(sigma0=sigma0, fixed=disp_fixed,
                        bounds=disp_bounds, name='dispprof')

    disp_prof = set_comp_param_prior(comp=disp_prof, param_name='sigma0', params=params)

    # --------------------------------------
    # Add the model component to the ModelSet
    mod_set.add_component(disp_prof)

    return mod_set

def add_zheight_gaus_comp(gal=None, mod_set=None, params=None,
            tied_sigmaz_func=None):

    if tied_sigmaz_func is None:
        tied_sigmaz_func = tied_functions.tie_sigz_reff

    sigmaz = params['sigmaz']      # kpc
    zheight_fixed = {'sigmaz': params['sigmaz_fixed']}
    zheight_bounds = {'sigmaz': (params['sigmaz_bounds'][0], params['sigmaz_bounds'][1])}

    zheight_prof = models.ZHeightGauss(sigmaz=sigmaz, name='zheightgaus',
                                       fixed=zheight_fixed, bounds=zheight_bounds)
    if params['zheight_tied']:
        zheight_prof.sigmaz.tied = tied_sigmaz_func
    else:
        # Do prior changes away from default flat prior, if so specified:
        zheight_prof = set_comp_param_prior(comp=zheight_prof, param_name='sigmaz', params=params)

    # --------------------------------------
    # Add the model component to the ModelSet
    mod_set.add_component(zheight_prof)

    return mod_set

def add_geometry_comp(gal=None, mod_set=None, params=None):
    params = _preprocess_geom_parameters(params=params)

    inc = params['inc']                # degrees
    pa =  params['pa']                 # default convention; neg r is blue side

    xshift = params['xshift']          # pixels from center
    yshift = params['yshift']          # pixels from center
    vel_shift = params['vel_shift']    # km/s ; systemic vel

    geom_fixed = {'inc': params['inc_fixed'],
                  'pa': params['pa_fixed'],
                  'xshift': params['xshift_fixed'],
                  'yshift': params['yshift_fixed'],
                  'vel_shift': params['vel_shift_fixed']}

    geom_bounds = {'inc':  (params['inc_bounds'][0], params['inc_bounds'][1]),
                   'pa':  (params['pa_bounds'][0], params['pa_bounds'][1]),
                   'xshift':  (params['xshift_bounds'][0], params['xshift_bounds'][1]),
                   'yshift':  (params['yshift_bounds'][0], params['yshift_bounds'][1]),
                   'vel_shift': (params['vel_shift_bounds'][0], params['vel_shift_bounds'][1])}

    geom = models.Geometry(inc=inc, pa=pa, xshift=xshift, yshift=yshift, vel_shift=vel_shift,
                           fixed=geom_fixed, bounds=geom_bounds, name='geom')
    geom = set_comp_param_prior(comp=geom, param_name='inc', params=params)
    geom = set_comp_param_prior(comp=geom, param_name='pa', params=params)
    geom = set_comp_param_prior(comp=geom, param_name='xshift', params=params)
    geom = set_comp_param_prior(comp=geom, param_name='yshift', params=params)
    geom = set_comp_param_prior(comp=geom, param_name='vel_shift', params=params)

    # --------------------------------------
    # Add the model component to the ModelSet

    mod_set.add_component(geom)

    return mod_set

def _preprocess_geom_parameters(params=None):
    if 'pa' not in list(params.keys()):
        params['pa'] = params['slit_pa']  # default convention; neg r is blue side

    for key in ['xshift', 'yshift', 'vel_shift']:
        if key not in list(params.keys()):
            params[key] = 0.

    for key in ['xshift', 'yshift', 'vel_shift', 'inc', 'pa']:
        if '{}_fixed'.format(key) not in list(params.keys()):
            params['{}_fixed'.format(key)] = True

    for key in ['xshift', 'yshift']:
        if '{}_bounds'.format(key) not in list(params.keys()):
            params['{}_bounds'.format(key)] = (-1.,1.)

    if 'vel_shift_bounds' not in list(params.keys()):
        params['vel_shift_bounds'] = (-100., 100.)
    if 'pa_bounds' not in list(params.keys()):
        params['pa_bounds'] = (-180., 180.)
    if 'inc' not in list(params.keys()):
        params['inc_bounds'] = (0., 90.)

    return params

def setup_instrument_params(inst=None, params=None):
    pixscale = params['pixscale']*u.arcsec                # arcsec/pixel
    fov = [params['fov_npix'], params['fov_npix']]        # (nx, ny) pixels
    spec_type = params['spec_type']                       # 'velocity' or 'wavelength'
    if spec_type.strip().lower() == 'velocity':
        spec_start = params['spec_start']*u.km/u.s        # Starting value of spectrum
        spec_step = params['spec_step']*u.km/u.s          # Spectral step
    else:
        raise ValueError("not implemented for wavelength yet!")
    nspec = params['nspec']                               # Number of spectral pixels



    if params['psf_type'].lower().strip() == 'gaussian':
        beamsize = params['psf_fwhm']*u.arcsec              # FWHM of beam, Gaussian
        beam = instrument.GaussianBeam(major=beamsize)
    elif params['psf_type'].lower().strip() == 'moffat':
        beamsize = params['psf_fwhm']*u.arcsec              # FWHM of beam, Moffat
        beta = params['psf_beta']
        beam = instrument.Moffat(major_fwhm=beamsize, beta=beta)
    elif params['psf_type'].lower().strip() == 'doublegaussian':
        # Kernel of both components multipled by: self._scaleN / np.sum(kernelN.array)
        #    -- eg, scaleN controls the relative amount of flux in each component.

        beamsize1 = params['psf_fwhm1']*u.arcsec              # FWHM of beam, Gaussian
        beamsize2 = params['psf_fwhm2']*u.arcsec              # FWHM of beam, Gaussian

        try:
            scale1 = params['psf_scale1']                     # Flux scaling of component 1
        except:
            scale1 = 1.                                       # If ommitted, assume scale2 is rel to scale1=1.
        scale2 = params['psf_scale2']                         # Flux scaling of component 2

        beam = instrument.DoubleBeam(major1=beamsize1, major2=beamsize2,
                        scale1=scale1, scale2=scale2)

    else:
        raise ValueError("PSF type {} not recognized!".format(params['psf_type']))

    if params['use_lsf']:
        sig_inst = params['sig_inst_res'] * u.km / u.s  # Instrumental spectral resolution  [km/s]
        lsf = instrument.LSF(sig_inst)
        inst.lsf = lsf
        inst.set_lsf_kernel()

    inst.beam = beam
    inst.pixscale = pixscale
    inst.fov = fov
    inst.spec_type = spec_type
    inst.spec_step = spec_step
    inst.spec_start = spec_start
    inst.nspec = nspec

    # Set the beam kernel so it doesn't have to be calculated every step
    inst.set_beam_kernel(support_scaling=12.)   # ORIGINAL: support_scaling=8.

    return inst



def setup_fit_dict(params=None, ndim_data=None):

    if params['fit_method'] == 'mcmc':

        fit_dict = setup_mcmc_dict(params=params, ndim_data=ndim_data)

    elif params['fit_method'] == 'mpfit':

        fit_dict = setup_mpfit_dict(params=params, ndim_data=ndim_data)

    return fit_dict


def setup_mcmc_dict(params=None, ndim_data=None):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Parameters for the MCMC fitting + output filenames

    fitting.ensure_dir(params['outdir'])

    outdir = params['outdir']
    galID = params['galID']

    if 'plot_type' in params.keys():
        plot_type = params['plot_type']
    else:
        plot_type = 'pdf'

    # All in one directory:
    f_plot_trace_burnin = outdir+'{}_mcmc_burnin_trace.{}'.format(galID, plot_type)
    f_plot_trace = outdir+'{}_mcmc_trace.{}'.format(galID, plot_type)
    f_model = outdir+'{}_galaxy_model.pickle'.format(galID)

    if ndim_data == 1:
        f_model_bestfit = outdir+'{}_out-1dplots.txt'.format(galID)
    elif ndim_data == 2:
        f_model_bestfit = outdir+'{}_out-velmaps.fits'.format(galID)
    elif ndim_data == 3:
        f_model_bestfit = outdir+'{}_out-cube.fits'.format(galID)
    elif ndim_data == 0:
        f_model_bestfit = outdir+'{}_out-0d.txt'.format(galID)
    else:
        f_model_bestfit = None

    f_cube = outdir+'{}_mcmc_bestfit_model_cube.fits'.format(galID)

    if np.int(emcee.__version__[0]) >= 3:
        ftype_sampler = 'h5'
    else:
        ftype_sampler = 'pickle'
    f_sampler = outdir+'{}_mcmc_sampler.{}'.format(galID, ftype_sampler)
    f_burn_sampler = outdir+'{}_mcmc_burn_sampler.{}'.format(galID, ftype_sampler)

    f_plot_param_corner = outdir+'{}_mcmc_param_corner.{}'.format(galID, plot_type)
    f_plot_bestfit = outdir+'{}_mcmc_best_fit.{}'.format(galID, plot_type)
    f_plot_bestfit_multid = outdir+'{}_mcmc_best_fit_multid.{}'.format(galID, plot_type)
    f_results = outdir+'{}_mcmc_results.pickle'.format(galID)
    f_chain_ascii = outdir+'{}_mcmc_chain_blobs.dat'.format(galID)
    f_vel_ascii = outdir+'{}_galaxy_bestfit_vel_profile.dat'.format(galID)
    f_log = outdir+'{}_info.log'.format(galID)

    mcmc_dict = {'outdir': outdir,
                'f_plot_trace_burnin':  f_plot_trace_burnin,
                'f_plot_trace':  f_plot_trace,
                'f_model': f_model,
                'f_model_bestfit': f_model_bestfit,
                'f_cube': f_cube,
                'f_sampler':  f_sampler,
                'f_burn_sampler':  f_burn_sampler,
                'f_plot_param_corner':  f_plot_param_corner,
                'f_plot_bestfit':  f_plot_bestfit,
                'f_plot_bestfit_multid': f_plot_bestfit_multid,
                'f_results':  f_results,
                'f_mcmc_results': f_results, 
                'f_chain_ascii': f_chain_ascii,
                'f_vel_ascii': f_vel_ascii,
                'f_log': f_log,
                'do_plotting': True}

    for key in params.keys():
        if key not in mcmc_dict.keys():
            # Copy over all various fitting options
            mcmc_dict[key] = params[key]

    # #
    if 'linked_posteriors' in mcmc_dict.keys():
        if mcmc_dict['linked_posteriors'] is not None:
            linked_post_arr = []
            for lpost in mcmc_dict['linked_posteriors']:
                if lpost.strip().lower() == 'total_mass':
                    linked_post_arr.append(['disk+bulge', 'total_mass'])
                elif lpost.strip().lower() == 'mvirial':
                    linked_post_arr.append(['halo', 'mvirial'])
                elif lpost.strip().lower() == 'fdm':
                    linked_post_arr.append(['halo', 'fdm'])
                elif lpost.strip().lower() == 'alpha':
                    linked_post_arr.append(['halo', 'alpha'])
                elif lpost.strip().lower() == 'rb':
                    linked_post_arr.append(['halo', 'rB'])
                elif lpost.strip().lower() == 'r_eff_disk':
                    linked_post_arr.append(['disk+bulge', 'r_eff_disk'])
                elif lpost.strip().lower() == 'bt':
                    linked_post_arr.append(['disk+bulge', 'bt'])
                elif lpost.strip().lower() == 'sigma0':
                    linked_post_arr.append(['dispprof', 'sigma0'])
                elif lpost.strip().lower() == 'inc':
                    linked_post_arr.append(['geom', 'inc'])
                elif lpost.strip().lower() == 'pa':
                    linked_post_arr.append(['geom', 'pa'])
                elif lpost.strip().lower() == 'xshift':
                    linked_post_arr.append(['geom', 'xshift'])
                elif lpost.strip().lower() == 'yshift':
                    linked_post_arr.append(['geom', 'yshift'])
                elif lpost.strip().lower() == 'vel_shift':
                    linked_post_arr.append(['geom', 'vel_shift'])
                else:
                    raise ValueError("linked posterior for {} not currently implemented!".format(lpost))

            # "Bundle of linked posteriors"
            linked_posterior_names = [ linked_post_arr ]
            mcmc_dict['linked_posterior_names'] = linked_posterior_names
        else:
            mcmc_dict['linked_posterior_names'] = None
    else:
        mcmc_dict['linked_posterior_names'] = None


    #
    mcmc_dict['model_key_re'] = ['disk+bulge', 'r_eff_disk']
    mcmc_dict['model_key_halo'] = ['halo']


    if 'continue_steps' not in mcmc_dict.keys():
        mcmc_dict['continue_steps'] = False

    return mcmc_dict


def setup_mpfit_dict(params=None, ndim_data=None):
    if 'plot_type' in params.keys():
        plot_type = params['plot_type']
    else:
        plot_type = 'pdf'

    fitting.ensure_dir(params['outdir'])
    outdir = params['outdir']
    galID = params['galID']
    f_model = outdir+'{}_galaxy_model.pickle'.format(galID)
    f_cube = outdir+'{}_mpfit_bestfit_model_cube.fits'.format(galID)
    f_plot_bestfit = outdir+'{}_mpfit_best_fit.{}'.format(galID, plot_type)
    f_results = outdir+'{}_mpfit_results.pickle'.format(galID)
    f_plot_bestfit_multid = outdir+'{}_mpfit_best_fit_multid.{}'.format(galID, plot_type)
    f_vel_ascii = outdir+'{}_galaxy_bestfit_vel_profile.dat'.format(galID)
    f_log = outdir+'{}_info.log'.format(galID)


    if ndim_data == 1:
        f_model_bestfit = outdir+'{}_out-1dplots.txt'.format(galID)
    elif ndim_data == 2:
        f_model_bestfit = outdir+'{}_out-velmaps.fits'.format(galID)
    elif ndim_data == 3:
        f_model_bestfit = outdir+'{}_out-cube.fits'.format(galID)
    elif ndim_data == 0:
        f_model_bestfit = outdir+'{}_out-0d.txt'.format(galID)
    else:
        f_model_bestfit = None

    mpfit_dict = {'outdir': outdir,
                  'f_model': f_model,
                  'f_model_bestfit': f_model_bestfit,
                  'f_cube': f_cube,
                  'f_plot_bestfit':  f_plot_bestfit,
                  'f_plot_bestfit_multid': f_plot_bestfit_multid,
                  'f_results':  f_results,
                  'f_vel_ascii': f_vel_ascii,
                  'f_log': f_log,
                  'do_plotting': True}

    for key in params.keys():
        if key not in mpfit_dict.keys():
            # Copy over all various fitting options
            mpfit_dict[key] = params[key]

    return mpfit_dict


def setup_basic_aperture_types(gal=None, params=None):

    if ('aperture_radius' in params.keys()):
        aperture_radius=params['aperture_radius']
    else:
        aperture_radius = None

    if ('pix_perp' in params.keys()):
        pix_perp=params['pix_perp']
    else:
        pix_perp = None

    if ('pix_parallel' in params.keys()):
        pix_parallel=params['pix_parallel']
    else:
        pix_parallel = None

    if ('pix_length' in params.keys()):
        pix_length=params['pix_length']
    else:
        pix_length = None


    if ('partial_weight' in params.keys()):
        partial_weight = params['partial_weight']
    else:
        # # Preserve previous default behavior
        # partial_weight = False

        ## NEW default behavior: always use partial_weight:
        partial_weight = True

    if ('moment_calc' in params.keys()):
        moment_calc = params['moment_calc']
    else:
        moment_calc = False

    apertures = aperture_classes.setup_aperture_types(gal=gal,
                profile1d_type=params['profile1d_type'],
                aperture_radius=aperture_radius,
                pix_perp=pix_perp, pix_parallel=pix_parallel,
                pix_length=pix_length, from_data=True,
                partial_weight=partial_weight,
                moment=moment_calc)


    return apertures

def setup_data_weighting_method(method='UNSET', r=None):
    if r is not None:
        rmax = np.abs(np.max(r))
    else:
        rmax = None

    if method == 'UNSET':
        raise ValueError("Must set method if setting data point weighting!")
    elif (method is None):
        weight = None
    elif ((method.strip().lower() == 'none') | (method.strip().lower() == 'uniform')):
        weight = None
        #weight = np.ones(len(r), dtype=np.float)
    # exp[ A * (r/rmax) ]  // exponential or more general power-law
    elif method.strip().lower() == 'radius_rmax':
        # value at 0: 1 // value at rmax: 2.718
        weight = np.exp( np.abs(r)/ rmax )
    elif method.strip().lower() == 'radius_rmax_a5':
        # value at 0: 1 // value at rmax: 5.
        weight = np.exp( np.log(5.) * (np.abs(r)/ rmax)  )
    elif method.strip().lower() == 'radius_rmax_a10':
        # value at 0: 1 // value at rmax: 10.
        weight = np.exp( np.log(10.) * (np.abs(r)/ rmax)  )
    # exp[ A * (r/rmax)^2 ]  // exponential or more general power-law
    elif method.strip().lower() == 'radius_rmax2':
        # value at 0: 1 // value at rmax: 2.718
        weight = np.exp((np.abs(r)/ rmax)**2 )
    elif method.strip().lower() == 'radius_2rmax2':
        # value at 0: 1 // value at rmax: 7.389
        weight = np.exp( 2. * (np.abs(r)/ rmax)**2 )
    elif method.strip().lower() == 'radius_rmax2_a5':
        # value at 0: 1 // value at rmax: 5.
        weight = np.exp( np.log(5.) * (np.abs(r)/ rmax)**2 )
    elif method.strip().lower() == 'radius_rmax2_a10':
        # value at 0: 1 // value at rmax: 10.
        weight = np.exp( np.log(10.) * (np.abs(r)/ rmax)**2 )
    else:
        raise ValueError("Weighting method not implmented yet!: {}".format(method))

    return weight


def set_comp_param_prior(comp=None, param_name=None, params=None):
    if params['{}_fixed'.format(param_name)] is False:
        if '{}_prior'.format(param_name) in list(params.keys()):
            # Default to using pre-set value!
            try:
                try:
                    center = comp.prior[param_name].center
                except:
                    center = params[param_name]
            except:
                # eg, UniformPrior
                center = None

            # Default to using pre-set value, if already specified!!!
            try:
                try:
                    stddev = comp.prior[param_name].stddev
                except:
                    stddev = params['{}_stddev'.format(param_name)]
            except:
                stddev = None

            if params['{}_prior'.format(param_name)].lower() == 'flat':
                comp.__getattribute__(param_name).prior = parameters.UniformPrior()
            elif params['{}_prior'.format(param_name)].lower() == 'flat_linear':
                comp.__getattribute__(param_name).prior = parameters.UniformLinearPrior()
            elif params['{}_prior'.format(param_name)].lower() == 'gaussian':
                comp.__getattribute__(param_name).prior = parameters.BoundedGaussianPrior(center=center, stddev=stddev)
            elif params['{}_prior'.format(param_name)].lower() == 'sine_gaussian':
                comp.__getattribute__(param_name).prior = parameters.BoundedSineGaussianPrior(center=center, stddev=stddev)
            elif params['{}_prior'.format(param_name)].lower() == 'gaussian_linear':
                comp.__getattribute__(param_name).prior = parameters.BoundedGaussianLinearPrior(center=center, stddev=stddev)
            elif params['{}_prior'.format(param_name)].lower() == 'tied_flat_lowtrunc':
                comp.__getattribute__(param_name).prior = TiedUniformPriorLowerTrunc(compn='disk+bulge', paramn='total_mass')
            elif params['{}_prior'.format(param_name)].lower() == 'tied_gaussian_lowtrunc':
                comp.__getattribute__(param_name).prior = TiedBoundedGaussianPriorLowerTrunc(center=center, stddev=stddev,
                                                            compn='disk+bulge', paramn='total_mass')
            else:
                print(" CAUTION: {}: {} prior is not currently supported. Defaulting to 'flat'".format(param_name,
                                    params['{}_prior'.format(param_name)]))
                pass

    return comp
