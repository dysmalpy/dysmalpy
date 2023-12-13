# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Module containing some useful utility functions

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import copy
from collections import OrderedDict

# Local imports
from dysmalpy.data_io import ensure_dir
from dysmalpy.fitting import MCMCFitter, MPFITFitter, NestedFitter

# Third party import

__all__ = [ "OutputOptions" ]

class OutputOptions:
    """
    Class to hold all options for file output during and after fitting
    """

    def __init__(self, outdir='./fit/',
                 save_model=True,
                 save_model_bestfit=True,
                 save_bestfit_cube=True,
                 save_data=True,
                 save_vel_ascii=True,
                 save_results=True,
                 save_reports=True,
                 file_base=None,
                 do_plotting=True,
                 plot_type='pdf',
                 overwrite=False
                 ):

        self.outdir = outdir
        self.save_model = save_model
        self.save_model_bestfit = save_model_bestfit
        self.save_bestfit_cube = save_bestfit_cube
        self.save_data = save_data
        self.save_vel_ascii = save_vel_ascii
        self.save_results = save_results
        self.save_reports = save_reports
        self.file_base = file_base
        self.do_plotting = do_plotting
        self.plot_type = plot_type
        self.overwrite = overwrite

        self.f_params = None

        # Galaxy specific filenames
        self.f_model = None
        self.f_vcirc_ascii = None
        self.f_mass_ascii = None
        self.f_results = None
        self.f_plot_bestfit = None      # Plotting function inserts obs name before filetype later
        self.f_log = None
        self.f_report_pretty = None
        self.f_report_machine = None

        # Observation and tracer specific filenames
        self.f_model_bestfit = None
        self.f_bestfit_cube = None
        self.f_vel_ascii = None

        # Bayesian fitting specific filenames
        self.f_sampler_results_continue = None
        self.f_sampler_results = None
        self.f_sampler_results_tmp = None
        self.f_burn_sampler = None
        self.f_plot_trace_burnin = None
        self.f_plot_trace = None
        self.f_plot_param_corner = None
        self.f_plot_run = None
        self.f_chain_ascii = None


    def __deepcopy__(self, memo):
        self2 = type(self)(outdir=self.outdir,
                           save_model=self.save_model,
                           save_model_bestfit=self.save_model_bestfit,
                           save_bestfit_cube=self.save_bestfit_cube,
                           save_data=self.save_data,
                           save_vel_ascii=self.save_vel_ascii,
                           save_results=self.save_results,
                           save_reports=self.save_reports,
                           file_base=self.file_base,
                           do_plotting=self.do_plotting,
                           plot_type=self.plot_type,
                           overwrite=self.overwrite)
        self2.__dict__.update(self.__dict__)
        return self2

    def __copy__(self):
        self2 = type(self)(outdir=self.outdir,
                           save_model=self.save_model,
                           save_model_bestfit=self.save_model_bestfit,
                           save_bestfit_cube=self.save_bestfit_cube,
                           save_data=self.save_data,
                           save_vel_ascii=self.save_vel_ascii,
                           save_results=self.save_results,
                           save_reports=self.save_reports,
                           file_base=self.file_base,
                           do_plotting=self.do_plotting,
                           plot_type=self.plot_type,
                           overwrite=self.overwrite)
        self2.__dict__.update(self.__dict__)
        return self2

    def as_dict(self):
        return self.__dict__

    def copy(self):
        return copy.deepcopy(self)

    def set_output_options(self, gal, fitter):

        ensure_dir(self.outdir)

        # Initialize per-obs/per-tracer filenames as OrderedDict:
        self.f_model_bestfit = OrderedDict()
        self.f_bestfit_cube = OrderedDict()
        self.f_vel_ascii = OrderedDict()

        if isinstance(fitter, MCMCFitter):
            fit_type = 'mcmc'
        elif isinstance(fitter, MPFITFitter):
            fit_type = 'mpfit'
        elif isinstance(fitter, NestedFitter):
            fit_type = 'nested'
        else:
            raise ValueError("Unrecognized Fitter: {}!".format(type(fitter)))

        if self.file_base is None:
            self.file_base = gal.name

        if self.file_base[-1] == '_':
            self.file_base = self.file_base[0:]

        self.f_log = "{}{}_{}.log".format(self.outdir,self.file_base, fit_type)

        if self.save_model:
            self.f_model = "{}{}_model.pickle".format(self.outdir,self.file_base)

        if self.save_model_bestfit:
            for obs_name in gal.observations:
                obs = gal.observations[obs_name]

                if obs.data.ndim == 1:
                    self.f_model_bestfit[obs_name] = "{}{}_{}_{}".format(self.outdir, self.file_base, obs_name, 'out-1dplots.txt')
                elif obs.data.ndim == 2:
                    self.f_model_bestfit[obs_name] = "{}{}_{}_{}".format(self.outdir, self.file_base, obs_name, 'out-velmaps.fits')
                elif obs.data.ndim == 3:
                    self.f_model_bestfit[obs_name] = "{}{}_{}_{}".format(self.outdir, self.file_base, obs_name, 'out-cube.fits')
                elif obs.data.ndim == 0:
                    self.f_model_bestfit[obs_name] = "{}{}_{}_{}".format(self.outdir, self.file_base, obs_name, 'out-0d.txt')

        if self.save_bestfit_cube:

            for obs_name in gal.observations:

                obs = gal.observations[obs_name]
                self.f_bestfit_cube[obs_name] = "{}{}_{}_bestfit_cube.fits".format(self.outdir, self.file_base, obs_name)

        if self.save_vel_ascii:

            self.f_vcirc_ascii = "{}{}_bestfit_vcirc.dat".format(self.outdir,self.file_base)
            self.f_mass_ascii = "{}{}_bestfit_menc.dat".format(self.outdir,self.file_base)


            for tracer in gal.model.dispersions:
                self.f_vel_ascii[tracer] = "{}{}_{}_bestfit_velprofile.dat".format(self.outdir, self.file_base, tracer)

        if self.save_results:

            self.f_results = "{}{}_{}_results.pickle".format(self.outdir, self.file_base, fit_type)

        if self.save_reports:
            self.f_report_pretty = "{}{}_{}_bestfit_results_report.info".format(self.outdir, self.file_base, fit_type)
            self.f_report_machine = "{}{}_{}_bestfit_results.dat".format(self.outdir, self.file_base, fit_type)

        if self.do_plotting:

            self.f_plot_bestfit = "{}{}_{}_bestfit.{}".format(self.outdir, self.file_base, fit_type, self.plot_type)

        if fit_type == 'mcmc':
            if fitter._emcee_version < 3:
                self._set_mcmc_filenames_221()
            else:
                self._set_mcmc_filenames_3()

        elif fit_type == 'nested':
            self._set_nested_filenames()



    def clear_filenames(self):

        keys = self.__dict__.keys()
        for key in keys:
            if key[0:2] == 'f_':
                self.__dict__[key] = None



    def _set_mcmc_filenames_221(self):

        self.f_sampler_continue = self.outdir+self.file_base+'_mcmc_sampler_continue.pickle'
        self.f_sampler_results = self.outdir+self.file_base+'_mcmc_sampler.pickle'
        self.f_sampler_results_tmp = self.outdir+self.file_base+'_mcmc_sampler_INPROGRESS.pickle'
        self.f_burn_sampler = self.outdir+self.file_base+'_mcmc_burn_sampler.pickle'
        self.f_plot_trace_burnin = self.outdir+self.file_base+'_mcmc_burnin_trace.{}'.format(self.plot_type)
        self.f_plot_trace = self.outdir+self.file_base+'_mcmc_trace.{}'.format(self.plot_type)
        self.f_plot_param_corner = self.outdir+self.file_base+'_mcmc_param_corner.{}'.format(self.plot_type)
        self.f_chain_ascii = self.outdir+self.file_base+'_mcmc_chain_blobs.dat'

    def _set_mcmc_filenames_3(self):

        ftype_sampler = 'h5'
        self.f_sampler_results = self.outdir+self.file_base+'_mcmc_sampler.{}'.format(ftype_sampler)
        self.f_plot_trace_burnin = self.outdir+self.file_base+'_mcmc_burnin_trace.{}'.format(self.plot_type)
        self.f_plot_trace = self.outdir+self.file_base+'_mcmc_trace.{}'.format(self.plot_type)
        self.f_plot_param_corner = self.outdir+self.file_base+'_mcmc_param_corner.{}'.format(self.plot_type)
        self.f_chain_ascii = self.outdir+self.file_base+'_mcmc_chain_blobs.dat'



    def _set_nested_filenames(self):

        ftype_results = 'pickle'
        # ftype_checkpoint = ftype_results
        ftype_checkpoint = 'save'
        self.f_checkpoint = self.outdir+self.file_base+'_nested_checkpoints.{}'.format(ftype_checkpoint)
        self.f_sampler_results = self.outdir+self.file_base+'_nested_sampler_results.{}'.format(ftype_results)
        self.f_plot_trace = self.outdir+self.file_base+'_nested_trace.{}'.format(self.plot_type)
        self.f_plot_run = self.outdir+self.file_base+'_nested_run.{}'.format(self.plot_type)
        self.f_plot_param_corner = self.outdir+self.file_base+'_nested_param_corner.{}'.format(self.plot_type)
        self.f_chain_ascii = self.outdir+self.file_base+'_nested_chain_blobs.dat'
