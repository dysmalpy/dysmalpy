# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Module containing some useful utility functions

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import warnings

# Third party imports
import numpy as np
import astropy.units as u


class ConfigBase:
    """
    Base class to handle settings for different functions.
    """

    def __init__(self, **kwargs):
        self.set_defaults()
        self.fill_values(**kwargs)

    def set_defaults(self):
        raise ValueError("Must be set for each inheriting class!")

    def fill_values(self, **kwargs):
        for key in self.__dict__.keys():
            if key in kwargs.keys():
                self.__dict__[key] = kwargs[key]

    @property
    def dict(self):
        return self.to_dict()

    def to_dict(self):
        kwarg_dict = {}
        for key in self.__dict__.keys():
            kwarg_dict[key] = self.__dict__[key]
        return kwarg_dict


class Config_create_model_data(ConfigBase):
    """
    Class to handle settings for Galaxy.create_model_data.
    """
    def __init__(self, **kwargs):

        super(Config_create_model_data, self).__init__(**kwargs)

    def set_defaults(self):
        self.ndim_final = 3
        self.line_center = None
        self.aper_centers = None
        self.slit_width = None
        self.slit_pa = None
        self.profile1d_type = None
        self.from_instrument = True
        self.from_data = True
        self.aperture_radius = None
        self.pix_perp = None
        self.pix_parallel = None
        self.pix_length = None
        self.skip_downsample = False
        self.partial_aperture_weight = False
        #self.partial_weight = False   ## used for rot curve plotting -- but only passed to aperture
        self.gauss_extract_with_c = None # True or False or None, whether to use faster C++ 1d gaussian spectral fitting


class Config_simulate_cube(ConfigBase):
    """
    Class to handle settings for model_set.simulate_cube
    """
    def __init__(self, **kwargs):
        super(Config_simulate_cube, self).__init__(**kwargs)

    def set_defaults(self):
        self.nx_sky = None
        self.ny_sky = None
        self.rstep = None
        self.spec_type = 'velocity'
        self.spec_step = 10.
        self.spec_start = -1000.
        self.nspec = 201
        self.spec_unit = (u.km/u.s)
        self.xcenter = None
        self.ycenter = None
        self.oversample = 1
        self.oversize = 1
        self.transform_method = 'direct'
        self.zcalc_truncate = None     # Default will be set in galaxy.create_model_data:
                                       #    0D/1D/2D/3D: True/True/False/False
                                       #    because for the smaller spatial extent of a single spaxel
                                       #    for 2D/3D leads to asymmetries from truncation,
                                       #    while this is less important for 0D/1D (combo in apertures).
                                       #    Previous: True
        self.n_wholepix_z_min = 3
        self.lensing_datadir = None # datadir for the lensing model mesh.dat
        self.lensing_mesh = None # lensing model mesh.dat
        self.lensing_ra = None # lensing model ref ra
        self.lensing_dec = None # lensing model ref dec
        self.lensing_sra = None # lensing source plane image center ra
        self.lensing_sdec = None # lensing source plane image center dec
        self.lensing_ssizex = None # lensing source plane image size in x
        self.lensing_ssizey = None # lensing source plane image size in y
        self.lensing_spixsc = None # lensing source plane image pixel size in arcsec unit
        self.lensing_imra = None # lensing image plane image center ra
        self.lensing_imdec = None # lensing image plane image center dec
        self.lensing_transformer = None # a placeholder for the object pointer


class ConfigFitBase(ConfigBase):
    """
    Class to handle settings for fitting.fit_mcmc
    """
    def __init__(self, **kwargs):
        super(ConfigFitBase, self).__init__(**kwargs)

    def set_defaults(self):
        # Fitting defaults that are shared between all fitting methods

        self.fitdispersion = True
        self.fitflux = False

        self.blob_name = None

        self.model_key_re = ['disk+bulge','r_eff_disk']
        self.model_key_halo=['halo']

        self.save_model = True
        self.save_model_bestfit = True
        self.save_bestfit_cube=True
        self.save_data = True
        self.save_vel_ascii = True
        self.save_results = True

        self.overwrite = False

        self.f_model = None
        self.f_model_bestfit = None
        self.f_cube = None
        self.f_plot_bestfit = None

        # Specific to 3D: 'f_plot_spaxel', 'f_plot_aperture', 'f_plot_channel'
        self.f_plot_spaxel = None
        self.f_plot_aperture = None
        self.f_plot_channel = None

        self.f_results = None

        self.f_vel_ascii = None
        self.f_vcirc_ascii = None
        self.f_mass_ascii = None
        self.f_log = None

        self.do_plotting = True
        self.plot_type = 'pdf'


class Config_fit_mcmc(ConfigFitBase):
    """
    Class to handle settings for fitting.fit_mcmc
    """
    def __init__(self, **kwargs):
        self.set_mcmc_defaults()
        super(Config_fit_mcmc, self).__init__(**kwargs)

    def set_mcmc_defaults(self):
        # MCMC specific defaults
        self.nWalkers = 10
        self.nCPUs = 1
        self.cpuFrac = None
        self.scale_param_a = 3.
        self.nBurn = 2.
        self.nSteps = 10.
        self.minAF = 0.2
        self.maxAF = 0.5
        self.nEff = 10
        self.oversampled_chisq = True
        self.red_chisq = False

        self.save_burn = False

        self.outdir = 'mcmc_fit_results/'
        self.save_intermediate_sampler_chain = True
        self.nStep_intermediate_save = 5
        self.continue_steps = False

        self.nPostBins = 50
        self.linked_posterior_names = None

        self.input_sampler = None

        self.f_sampler = None
        self.f_sampler_tmp = None
        self.f_burn_sampler = None
        self.f_chain_ascii = None

        self.f_plot_trace_burnin = None
        self.f_plot_trace = None
        self.f_plot_param_corner = None

class Config_fit_mpfit(ConfigFitBase):
    """
    Class to handle settings for fitting.fit_mcmc
    """
    def __init__(self, **kwargs):
        self.set_mpfit_defaults()
        super(Config_fit_mpfit, self).__init__(**kwargs)

    def set_mpfit_defaults(self):
        # MPFIT specific defaults

        self.use_weights=False
        self.maxiter=200

        self.outdir='mpfit_fit_results/'
