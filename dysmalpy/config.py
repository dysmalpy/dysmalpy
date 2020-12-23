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
        self.zcalc_truncate = True



class ConfigFitBase(ConfigBase):
    """
    Class to handle settings for fitting.fit_mcmc
    """
    def __init__(self, **kwargs):
        super(ConfigFitBase, self).__init__(**kwargs)
    def set_defaults(self):
        # Fitting defaults that are shared between all fitting methods
        raise NotImplementedError

class Config_fit_mcmc(ConfigFitBase):
    """
    Class to handle settings for fitting.fit_mcmc
    """
    def __init__(self, **kwargs):
        self.set_mcmc_defaults()
        super(Config_fit_mcmc, self).__init__(**kwargs)

    def set_mcmc_defaults(self):
        # MCMC specific defaults
        raise NotImplementedError

class Config_fit_mpfit(ConfigFitBase):
    """
    Class to handle settings for fitting.fit_mcmc
    """
    def __init__(self, **kwargs):
        self.set_mpfit_defaults()
        super(Config_fit_mpfit, self).__init__(**kwargs)

    def set_mpfit_defaults(self):
        # MPFIT specific defaults
        raise NotImplementedError



# def default_create_gal_model_kwargs():
#     create_gal_model_kwargs = {'ndim_final':      3,
#                        'line_center':             None,
#                        'aper_centers':            None,
#                        'slit_width':              None,
#                        'slit_pa':                 None,
#                        'profile1d_type':          None,
#                        'from_instrument':         True,
#                        'from_data':               True,
#                        'aperture_radius':         None,
#                        'pix_perp':                None,
#                        'pix_parallel':            None,
#                        'pix_length':              None,
#                        'skip_downsample':         False,
#                        'partial_aperture_weight': False }
#
#     return create_gal_model_kwargs
#
# def default_sim_cube_kwargs():
#     sim_cube_kwargs = {'nx_sky':         None,
#                        'ny_sky':         None,
#                        'rstep':          None,
#                        'spec_type':     'velocity',
#                        'spec_step':      10.,
#                        'spec_start':     -1000.,
#                        'nspec':          201,
#                        'spec_unit':      (u.km/u.s),
#                        'xcenter':        None,
#                        'ycenter':        None,
#                        'oversample':     1,
#                        'oversize':       1,
#                        'zcalc_truncate': True }
#
#     return sim_cube_kwargs
#
#
# def config_create_model_data(**kwargs_in):
#     create_gal_model_kwargs = default_create_gal_model_kwargs()
#     sim_cube_kwargs = default_sim_cube_kwargs()
#
#     for key in kwargs_in.keys():
#         if key in create_gal_model_kwargs.keys():
#             create_gal_model_kwargs[key] = kwargs_in[key]
#         if key in sim_cube_kwargs.keys():
#             create_gal_model_kwargs[key] = kwargs_in[key]
#
#     return create_gal_model_kwargs, sim_cube_kwargs
