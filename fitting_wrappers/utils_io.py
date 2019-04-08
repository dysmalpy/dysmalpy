# Script to fit KMOS3D kinematics

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys
from contextlib import contextmanager


import datetime

import numpy as np
import pandas as pd
import astropy.units as u

from dysmalpy import data_classes
from dysmalpy import parameters
from dysmalpy import fitting
from dysmalpy import utils as dysmalpy_utils

from dysmalpy import aperture_classes

from astropy.table import Table

import astropy.io.fits as fits

def read_fitting_params_input(fname=None):
    params = {}
    
    columns = ['keys', 'values']
    df = pd.read_csv(fname, sep=',', comment='#', names=columns, skipinitialspace=True).copy() 
    
    for j, key in enumerate(df['keys'].values):
        if key is np.NaN:
            pass
        else:
            valset = False
            try:
                tmpval = df['values'][j].split('#')[0].strip()
            except:
                tmpval = df['values'][j].strip()
            try:
                tmparr = tmpval.split(' ')
                tmparrnew = []
                if len(tmparr) > 1:
                    tmparrnew = []
                    for ta in tmparr:
                        if len(ta) > 0:
                            tv = ta.strip()
                            try:
                                tvn = np.float(tv)
                            except:
                                tvn = tv
                            tmparrnew.append(tvn)
                    tmpval = tmparrnew
                    valset = True
            
            except:
                pass
            
            if not valset:
                strtmpval = str(tmpval).strip()
                if strtmpval == 'True':
                    tmpval = True
                elif strtmpval == 'False':
                    tmpval = False
                elif strtmpval == 'None':
                    tmpval = None
                else:
                    try:
                        fltval = np.float(tmpval)
                        if (fltval % 1) == 0.:
                            tmpval = np.int(fltval)
                        else:
                            tmpval = fltval
                    except:
                        tmpval = strtmpval.strip()
        
            params[key] = tmpval
        
    return params
    #

def read_fitting_params(fname=None):
    if fname is None:
        raise ValueError("parameter filename {} not found!".format(fname))

    # READ FILE HERE!
    param_input = read_fitting_params_input(fname=fname)
    

    # Set some defaults if not otherwise specified
    params = {'nWalkers': 20, 
              'nCPUs': 4, 
              'scale_param_a': 2,
              'nBurn': 10,
              'nSteps': 10,
              'minAF': None, 
              'maxAF': None, 
              'nEff': 10,
              'do_plotting': True,
              'oversample': 1,
              'fitdispersion': True,
              'include_halo': False, 
              'red_chisq': True }

    # param_filename
    fname_split = fname.split('/')
    params['param_filename'] = fname_split[-1]

    for key in param_input.keys():
        params[key] = param_input[key]

    return params
    
def save_results_ascii_files(fit_results=None, gal=None, params=None):
    
    outdir = params['outdir']
    galID = params['galID']

    if params['fit_method'] == 'mcmc':
        f_ascii_machine = outdir+'{}_mcmc_best_fit_results.dat'.format(galID)

        f_ascii_pretty = outdir+'{}_mcmc_best_fit_results.info'.format(galID)
        
        
        # --------------------------------------------
        # get fdm_best, lfdm, ufdm
        if params['include_halo']:
            flatblobs = np.hstack(np.stack(fit_results.sampler['blobs'], axis=1))
            
            lfdm, ufdm = np.percentile(flatblobs, [15.865, 84.135])
            
            nPostBins=40
            yb, xb = np.histogram(flatblobs, bins=nPostBins)
            wh_pk = np.where(yb == yb.max())[0][0]
            peak_hist = np.average([xb[wh_pk], xb[wh_pk+1]])
            
            ## Use max prob as guess to get peakKDE value,
            ##      the peak of the marginalized posterior distributions (following M. George's speclens)
            fdm_best = fitting.find_peak_gaussian_KDE(flatblobs, peak_hist)[0]
        # --------------------------------------------
        
        
        with open(f_ascii_machine, 'w') as f:
            namestr = '# component    param_name    fixed    best_value   l68_err   u68_err'
            f.write(namestr+'\n')
            
            for cmp_n in gal.model.param_names.keys():
                for param_n in gal.model.param_names[cmp_n]:
                    
                    if '{}:{}'.format(cmp_n,param_n) in fit_results.chain_param_names:
                        whparam = np.where(fit_results.chain_param_names == '{}:{}'.format(cmp_n, param_n))[0][0]
                        best = fit_results.bestfit_parameters[whparam]
                        l68 = fit_results.bestfit_parameters_l68_err[whparam]
                        u68 = fit_results.bestfit_parameters_u68_err[whparam]
                    else:
                        best = getattr(gal.model.components[cmp_n], param_n).value
                        l68 = -99.
                        u68 = -99.
                        
                    datstr = '{: <12}   {: <11}   {: <5}   {:9.4f}   {:9.4f}   {:9.4f}'.format(cmp_n, param_n,
                                "{}".format(gal.model.fixed[cmp_n][param_n]), best, l68, u68)
                    f.write(datstr+'\n')
                    
            ###
            
            #fdm_best, lfdm, ufdm
            if params['include_halo']:
                datstr = '{: <12}   {: <11}   {: <5}   {:9.4f}   {:9.4f}   {:9.4f}'.format('f_DM_RE', '-----',
                            '-----', fdm_best, lfdm, ufdm)
                f.write(datstr+'\n')
            
            
            ###
            datstr = '{: <12}   {: <11}   {: <5}   {:9.4f}   {:9.4f}   {:9.4f}'.format('redchisq', '-----',
                        '-----', fit_results.bestfit_redchisq, -99, -99)
            f.write(datstr+'\n')
            
            #
        #
        with open(f_ascii_pretty, 'w') as f:
            f.write('###############################'+'\n')
            f.write(' Fitting for {}'.format(params['galID'])+'\n')
            f.write('\n')

            f.write("Date: {}".format(datetime.datetime.now())+'\n')
            f.write('\n')

            try:
                f.write('Datafile: {}'.format(params['fdata'])+'\n')
            except:
                try:
                    f.write('Datafiles:\n')
                    f.write(' vel:  {}'.format(params['fdata_vel'])+'\n')
                    f.write(' verr: {}'.format(params['fdata_verr'])+'\n')
                    f.write(' disp: {}'.format(params['fdata_disp'])+'\n')
                    f.write(' derr: {}'.format(params['fdata_derr'])+'\n')
                    try:
                        f.write(' mask: {}'.format(params['fdata_mask'])+'\n')
                    except:
                        pass
                except:
                    pass
            f.write('Paramfile: {}'.format(params['param_filename'])+'\n')

            f.write('\n')
            f.write('Fitting method: {}'.format(params['fit_method'].upper()))
            f.write('\n')
            f.write('###############################'+'\n')
            f.write(' Fitting results'+'\n')

            for cmp_n in gal.model.param_names.keys():
                f.write('-----------'+'\n')
                f.write(' {}'.format(cmp_n)+'\n')

                for param_n in gal.model.param_names[cmp_n]:

                    if '{}:{}'.format(cmp_n,param_n) in fit_results.chain_param_names:
                        whparam = np.where(fit_results.chain_param_names == '{}:{}'.format(cmp_n, param_n))[0][0]
                        best = fit_results.bestfit_parameters[whparam]
                        l68 = fit_results.bestfit_parameters_l68_err[whparam]
                        u68 = fit_results.bestfit_parameters_u68_err[whparam]


                        datstr = '    {: <11}    {:9.4f}  -{:9.4f} +{:9.4f}'.format(param_n, best, l68, u68)
                        f.write(datstr+'\n')
                #
                f.write('\n')
                #
                for param_n in gal.model.param_names[cmp_n]:

                    if '{}:{}'.format(cmp_n,param_n) not in fit_results.chain_param_names:
                        best = getattr(gal.model.components[cmp_n], param_n).value

                        datstr = '    {: <11}    {:9.4f}  [FIXED]'.format(param_n, best)
                        f.write(datstr+'\n')

            
            ####
            if params['include_halo']:
                f.write('\n')
                f.write('-----------'+'\n')
                datstr = '    {: <11}    {:9.4f}  -{:9.4f} +{:9.4f}'.format('f_DM(R_E)',
                            fdm_best, lfdm, ufdm)
                f.write(datstr+'\n')

            ####
            f.write('\n')
            f.write('-----------'+'\n')
            datstr = 'Red. chisq: {:0.4f}'.format(fit_results.bestfit_redchisq)
            f.write(datstr+'\n')

            f.write('\n')

    elif params['fit_method'] == 'mpfit':

        f_ascii_machine = outdir + '{}_mpfit_best_fit_results.dat'.format(galID)

        f_ascii_pretty = outdir + '{}_mpfit_best_fit_results.info'.format(galID)

        with open(f_ascii_machine, 'w') as f:
            namestr = '# component    param_name    fixed    best_value   error'
            f.write(namestr + '\n')

            for cmp_n in gal.model.param_names.keys():
                for param_n in gal.model.param_names[cmp_n]:

                    if '{}:{}'.format(cmp_n, param_n) in fit_results.chain_param_names:
                        whparam = \
                        np.where(fit_results.chain_param_names == '{}:{}'.format(cmp_n, param_n))[
                            0][0]
                        best = fit_results.bestfit_parameters[whparam]
                        err = fit_results.bestfit_parameters_err[whparam]
                    else:
                        best = getattr(gal.model.components[cmp_n], param_n).value
                        err = -99

                    datstr = '{: <12}   {: <11}   {: <5}   {:9.4f}   {:9.4f}'.format(
                        cmp_n, param_n,
                        "{}".format(gal.model.fixed[cmp_n][param_n]), best, err)
                    f.write(datstr + '\n')

            #
            datstr = '{: <12}   {: <11}   {: <5}   {:9.4f}   {:9.4f}'.format('redchisq',
                                                                                       '-----',
                                                                                       '-----',
                                                                                       fit_results.bestfit_redchisq,
                                                                                       -99)
            f.write(datstr + '\n')

        #
        with open(f_ascii_pretty, 'w') as f:
            f.write('###############################' + '\n')
            f.write(' Fitting for {}'.format(params['galID']) + '\n')
            f.write('\n')

            f.write("Date: {}".format(datetime.datetime.now()) + '\n')
            f.write('\n')

            try:
                f.write('Datafile: {}'.format(params['fdata']) + '\n')
            except:
                try:
                    f.write('Datafiles:\n')
                    f.write(' vel:  {}'.format(params['fdata_vel']) + '\n')
                    f.write(' verr: {}'.format(params['fdata_verr']) + '\n')
                    f.write(' disp: {}'.format(params['fdata_disp']) + '\n')
                    f.write(' derr: {}'.format(params['fdata_derr']) + '\n')
                    try:
                        f.write(' mask: {}'.format(params['fdata_mask']) + '\n')
                    except:
                        pass
                except:
                    pass
            f.write('Paramfile: {}'.format(params['param_filename']) + '\n')

            f.write('\n')
            f.write('Fitting method: MPFIT')
            f.write('\n')
            f.write('###############################' + '\n')
            f.write(' Fitting results' + '\n')

            for cmp_n in gal.model.param_names.keys():
                f.write('-----------' + '\n')
                f.write(' {}'.format(cmp_n) + '\n')

                for param_n in gal.model.param_names[cmp_n]:

                    if '{}:{}'.format(cmp_n, param_n) in fit_results.chain_param_names:
                        whparam = \
                        np.where(fit_results.chain_param_names == '{}:{}'.format(cmp_n, param_n))[
                            0][0]
                        best = fit_results.bestfit_parameters[whparam]
                        err = fit_results.bestfit_parameters_err[whparam]

                        datstr = '    {: <11}    {:9.4f}  +/-{:9.4f}'.format(param_n, best,
                                                                                    err)
                        f.write(datstr + '\n')
                #
                f.write('\n')
                #
                for param_n in gal.model.param_names[cmp_n]:

                    if '{}:{}'.format(cmp_n, param_n) not in fit_results.chain_param_names:
                        best = getattr(gal.model.components[cmp_n], param_n).value

                        datstr = '    {: <11}    {:9.4f}  [FIXED]'.format(param_n, best)
                        f.write(datstr + '\n')

            #
            f.write('\n')
            f.write('-----------' + '\n')
            datstr = 'Red. chisq: {:0.4f}'.format(fit_results.bestfit_redchisq)
            f.write(datstr + '\n')

            f.write('\n')
    
    
    return None

    
def setup_fit_dict(params=None):

    if params['fit_method'] == 'mcmc':

        fit_dict = setup_mcmc_dict(params=params)

    elif params['fit_method'] == 'mpfit':

        fit_dict = setup_mpfit_dict(params=params)

    return fit_dict


def setup_mcmc_dict(params=None):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Parameters for the MCMC fitting + output filenames
    
    fitting.ensure_dir(params['outdir'])
    
    outdir = params['outdir']
    galID = params['galID']
    
    # All in one directory:
    f_plot_trace_burnin = outdir+'{}_mcmc_burnin_trace.pdf'.format(galID)
    f_plot_trace = outdir+'{}_mcmc_trace.pdf'.format(galID)
    f_model = outdir+'{}_galaxy_model.pickle'.format(galID)
    f_cube = outdir+'{}_mcmc_bestfit_model_cube.fits'.format(galID)
    f_sampler = outdir+'{}_mcmc_sampler.pickle'.format(galID)
    f_burn_sampler = outdir+'{}_mcmc_burn_sampler.pickle'.format(galID)
    f_plot_param_corner = outdir+'{}_mcmc_param_corner.pdf'.format(galID)
    f_plot_bestfit = outdir+'{}_mcmc_best_fit.pdf'.format(galID)
    f_plot_bestfit_multid = outdir+'{}_mcmc_best_fit_multid.pdf'.format(galID)
    f_mcmc_results = outdir+'{}_mcmc_results.pickle'.format(galID)
    f_chain_ascii = outdir+'{}_mcmc_chain_blobs.dat'.format(galID)
    f_vel_ascii = outdir+'{}_galaxy_bestfit_vel_profile.dat'.format(galID)
    f_log = outdir+'{}_info.log'.format(galID)
    
    mcmc_dict = {'outdir': outdir, 
                'f_plot_trace_burnin':  f_plot_trace_burnin,
                'f_plot_trace':  f_plot_trace,
                'f_model': f_model,
                'f_cube': f_cube,
                'f_sampler':  f_sampler,
                'f_burn_sampler':  f_burn_sampler,
                'f_plot_param_corner':  f_plot_param_corner,
                'f_plot_bestfit':  f_plot_bestfit,
                'f_plot_bestfit_multid': f_plot_bestfit_multid, 
                'f_mcmc_results':  f_mcmc_results, 
                'f_chain_ascii': f_chain_ascii,
                'f_vel_ascii': f_vel_ascii, 
                'f_log': f_log, 
                'do_plotting': True}
                
    for key in params.keys():
        # Copy over all various fitting options
        mcmc_dict[key] = params[key]

    if mcmc_dict['include_halo']:

        mcmc_dict['compute_dm'] = True

    else:

        mcmc_dict['compute_dm'] = False
        
    return mcmc_dict


def setup_mpfit_dict(params=None):

    fitting.ensure_dir(params['outdir'])
    outdir = params['outdir']
    galID = params['galID']
    f_model = outdir+'{}_galaxy_model.pickle'.format(galID)
    f_cube = outdir+'{}_mpfit_bestfit_model_cube.fits'.format(galID)
    f_plot_bestfit = outdir+'{}_mpfit_best_fit.pdf'.format(galID)
    f_results = outdir+'{}_mpfit_results.pickle'.format(galID)
    f_plot_bestfit_multid = outdir+'{}_mpfit_best_fit_multid.pdf'.format(galID)
    f_vel_ascii = outdir+'{}_galaxy_bestfit_vel_profile.dat'.format(galID)
    f_log = outdir+'{}_info.log'.format(galID)

    mpfit_dict = {'outdir': outdir,
                  'f_model': f_model,
                  'f_cube': f_cube,
                  'f_plot_bestfit':  f_plot_bestfit,
                  'f_plot_bestfit_multid': f_plot_bestfit_multid,
                  'f_results':  f_results,
                  'f_vel_ascii': f_vel_ascii,
                  'f_log': f_log,
                  'do_plotting': True}

    for key in params.keys():
        # Copy over all various fitting options
        mpfit_dict[key] = params[key]

    return mpfit_dict
    
    
    

    
    
def setup_basic_aperture_types(gal=None, params=None):
    
    if ('aperture_radius' in params.keys()):
        aperture_radius=params['aperture_radius']
    else:
        aperture_radius = None
    #
    if ('pix_perp' in params.keys()):
        pix_perp=params['pix_perp']
    else:
        pix_perp = None
    #
    if ('pix_parallel' in params.keys()):
        pix_parallel=params['pix_parallel']
    else:
        pix_parallel = None
    #
    if ('pix_length' in params.keys()):
        pix_length=params['pix_length']
    else:
        pix_length = None
    
    apertures = aperture_classes.setup_aperture_types(gal=gal, 
                profile1d_type=params['profile1d_type'], 
                aperture_radius=aperture_radius, 
                pix_perp=pix_perp, pix_parallel=pix_parallel,
                pix_length=pix_length, from_data=True)
    
    return apertures

    
def load_single_object_1D_data(fdata=None, params=None):
    
    # Load the data set to be fit
    dat_arr =   np.loadtxt(fdata)
    gal_r =     dat_arr[:,0]
    gal_vel =   dat_arr[:,1]
    gal_disp =  dat_arr[:,3]
    err_vel =   dat_arr[:,2]
    err_disp =  dat_arr[:,4]
    
    #####
    # Apply symmetrization if wanted:
    try:
        if params['symmetrize_data']:
            gal_r_new, gal_vel, err_vel = dysmalpy_utils.symmetrize_1D_profile(gal_r, gal_vel, err_vel, sym=1)
            gal_r, gal_disp, err_disp = dysmalpy_utils.symmetrize_1D_profile(gal_r, gal_disp, err_disp, sym=2)
    except:
        pass
    
    
    data1d = data_classes.Data1D(r=gal_r, velocity=gal_vel,
                                      vel_disp=gal_disp, vel_err=err_vel,
                                      vel_disp_err=err_disp, slit_width=params['slit_width'],
                                      slit_pa=params['slit_pa'], inst_corr=params['data_inst_corr'] )
                                      
    return data1d
    
def load_single_object_2D_data(params=None, adjust_error=True, 
            automask=True, vmax=500., dispmax=600.):
    
    # +++++++++++++++++++++++++++++++++++++++++++
    # Upload the data set to be fit
    gal_vel = fits.getdata(params['fdata_vel'])
    err_vel = fits.getdata(params['fdata_verr'])
    gal_disp = fits.getdata(params['fdata_disp'])
    err_disp = fits.getdata(params['fdata_derr'])
    mask = fits.getdata(params['fdata_mask'])
    
    
    # Auto mask som bad data
    if automask: 
        indtmp = (gal_disp > dispmax) | (np.abs(gal_vel) > vmax)
        mask[indtmp] = 0
        
        
        
    if adjust_error:
        # Mask > 1sig lower error outliers:
        errv_l68 = np.percentile(err_vel, 15.865)
        errd_l68 = np.percentile(err_disp, 15.865)
        
        indv = (err_vel < errv_l68)
        err_vel[indv] = errv_l68
        
        indd = (err_disp < errd_l68)
        err_disp[indd] = errd_l68
        
        
    
    # Mask pixels with zero error for vel/disp:
    mask[(err_vel == 0)] = 0
    mask[(err_disp == 0)] = 0
    
    
    #####
    # Apply symmetrization if wanted:
    try:
        if params['symmetrize_data']:
            ybin, xbin = np.indices(gal_vel.shape, dtype=np.float64)
            ybin = ybin.flatten()
            xbin = xbin.flatten()
            xbin -= (gal_vel.shape[1]-1.)/2.
            ybin -= (gal_vel.shape[0]-1.)/2.
            xbin -= params['xshift']  
            ybin -= params['yshift']
            
            bool_mask = np.array(mask.copy(), dtype=bool)
            bool_mask_flat = np.array(mask.copy(), dtype=bool).flatten()
            
            gal_vel_flat_in = gal_vel.flatten()
            err_vel_flat_in = err_vel.flatten()
            gal_vel_flat_in[~bool_mask_flat] = np.NaN
            err_vel_flat_in[~bool_mask_flat] = np.NaN
            
            gal_vel_flat, err_vel_flat = dysmalpy_utils.symmetrize_velfield(xbin, ybin, 
                                gal_vel_flat_in, err_vel_flat_in, 
                                sym=1, pa=params['pa'])
            
            gal_disp_flat_in = gal_disp.flatten()
            err_disp_flat_in = err_disp.flatten()
            gal_disp_flat_in[~bool_mask_flat] = np.NaN
            err_disp_flat_in[~bool_mask_flat] = np.NaN
            gal_disp_flat, err_disp_flat = dysmalpy_utils.symmetrize_velfield(xbin, ybin, 
                                gal_disp_flat_in, err_disp_flat_in, 
                                sym=2, pa=params['pa'])
                                
            #
            gal_vel[bool_mask] = gal_vel_flat[bool_mask_flat]
            err_vel[bool_mask] = err_vel_flat[bool_mask_flat]
            gal_disp[bool_mask] = gal_disp_flat[bool_mask_flat]
            err_disp[bool_mask] = err_disp_flat[bool_mask_flat]
    except:
        pass
    
    
    data2d = data_classes.Data2D(pixscale=params['pixscale'], velocity=gal_vel,
                                      vel_disp=gal_disp, vel_err=err_vel,
                                      vel_disp_err=err_disp, mask=mask,
                                      filename_velocity=params['fdata_vel'],
                                      filename_dispersion=params['fdata_disp'],
                                      smoothing_type=params['smoothing_type'],
                                      smoothing_npix=params['smoothing_npix'],
                                      inst_corr=params['data_inst_corr'])
                                      
            
    return data2d
    
#
def load_single_object_3D_data(params=None):
    
    raise ValueError("Not generically supported for now: will need to write your own wrapper to load cubes.")
    
    
    data3d = None
    
    return data3d
    
#
def set_comp_param_prior(comp=None, param_name=None, params=None):
    
    if params['{}_fixed'.format(param_name)] is False:
        if '{}_prior'.format(param_name) in list(params.keys()):
            if params['{}_prior'.format(param_name)].lower() == 'flat':
                
                comp.prior[param_name] = parameters.UniformPrior()

            elif params['{}_prior'.format(param_name)].lower() == 'gaussian':
                comp.prior[param_name] = parameters.BoundedGaussianPrior(center=params[param_name],
                                                                        stddev=params['{}_stddev'.format(param_name)])
                                                                        
            elif params['{}_prior'.format(param_name)].lower() == 'sine_gaussian':
                comp.prior[param_name] = parameters.BoundedSineGaussianPrior(center=params[param_name],
                                                                        stddev=params['{}_stddev'.format(param_name)])

            elif params['{}_prior'.format(param_name)].lower() == 'tied_flat':
                comp.prior[param_name] = TiedUniformPrior()

            elif params['{}_prior'.format(param_name)].lower() == 'tied_gaussian':
                comp.prior[param_name] = TiedBoundedGaussianPrior(center=params[param_name],
                                                                  stddev=params['{}_stddev'.format(
                                                                                 param_name)])

            else:
                print(" CAUTION: {}: {} prior is not currently supported. Defaulting to 'flat'".format(param_name, 
                                    params['{}_prior'.format(param_name)]))
                pass
    
    
    return comp

    
#
def tie_sigz_reff(model_set):
 
    reff = model_set.components['disk+bulge'].r_eff_disk.value
    invq = model_set.components['disk+bulge'].invq_disk
    sigz = 2.0*reff/invq/2.35482

    return sigz


class TiedUniformPrior(parameters.UniformPrior):

    def log_prior(self, param, modelset):

        pmin = modelset.components['disk+bulge'].total_mass.value

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (param.value >= pmin) & (param.value <= pmax):
            return 0.
        else:
            return -np.inf


class TiedBoundedGaussianPrior(parameters.BoundedGaussianPrior):

    def log_prior(self, param, modelset):

        pmin = modelset.components['disk+bulge'].total_mass.value

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (param.value >= pmin) & (param.value <= pmax):
            return parameters.norm.pdf(param.value, loc=self.center, scale=self.stddev)
        else:
            return -np.inf


def ensure_path_trailing_slash(path):
    if (path[-1] != '/'):
        path += '/'
    return path
