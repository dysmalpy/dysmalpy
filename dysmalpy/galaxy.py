# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Main classes and functions for DYSMALPY for simulating the kinematics of
# a model galaxy and fitting it to observed data.


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging
import copy
from collections import OrderedDict
import os

# Third party imports
import astropy.cosmology as apy_cosmo
import dill as _pickle

# dill py<=3.7 -> py>=3.8 + higher hack:
# See https://github.com/uqfoundation/dill/pull/406
_pickle._dill._reverse_typemap['CodeType'] = _pickle._dill._create_code

# Local imports
# Package imports
from dysmalpy.models import ModelSet
from dysmalpy.utils_io import write_model_obs_file


__all__ = ['Galaxy']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')


# Default cosmology
_default_cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)


class Galaxy:
    r"""
    Container for simulating or modelling a galaxy

    `Galaxy` holds the observed data, model, observing instrument, and
    general information for a galaxy. This can be a simulated or real
    galaxy.

    Parameters
    ----------
    z : float
        Redshift of the galaxy
    cosmo : `~astropy.cosmology` object
            The cosmology to use for modelling. Default is
            astropy.cosmology.FlatLambdaCDM with H0=70., and Om0=0.3.
    model : `~dysmalpy.models.ModelSet` object
            A dysmalpy model to use for simulating and/or fitting data.
            This generates the intrinsic observables of the galaxy based
            on the components included in the ModelSet.
    obs_list : list
            List of `~dysmalpy.observation.Observation` objects,
            which hold `~dysmalpy.instrument.Instrument`, `~dysmalpy.observation.ObsModOptions`,
            and  `~dysmalpy.data_classes.Data` instances.
            For each `obs`, `obs.instrument` and `obs.mod_options`
            describes how `model` is converted to observed space data.
    name : str, optional
           Name of the galaxy. Default is "galaxy."

    """

    def __init__(self, z=0, cosmo=_default_cosmo, obs_list=None,
                 model=None, name='galaxy'):

        self._z = z
        self.name = name
        if model is None:
            self.model = ModelSet()
        else:
            self.model = model

        self._cosmo = cosmo
        self._dscale = self._cosmo.arcsec_per_kpc_proper(self._z).value

        # v2.0: everything lives inside Observation instances, inside
        #       OrderedDict self.observations
        self.observations = OrderedDict()
        if obs_list is not None:
            for obs in obs_list:
                self.add_observation(obs)


    def __setstate__(self, state):
        # Compatibility hack, to handle the changed galaxy structure
        #    (properties, not attributes for data[*], instrument)

        self.__dict__ = state
        # quick test if necessary to migrate:
        if '_data' in state.keys():
            pass
        else:
            migrate_keys = ['data', 'data1d', 'data2d', 'data3d', 'instrument', 'dscale']
            for mkey in migrate_keys:
                if (mkey in state.keys()) and ('_{}'.format(mkey) not in state.keys()):
                    self.__dict__['_{}'.format(mkey)] = state[mkey]
                    del self.__dict__[mkey]

    def copy(self):
        return copy.deepcopy(self)

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        if value < 0:
            raise ValueError("Redshift can't be negative!")
        self._z = value

        # Reset dscale:
        self._set_dscale()

    @property
    def cosmo(self):
        return self._cosmo

    @cosmo.setter
    def cosmo(self, new_cosmo):
        if not isinstance(new_cosmo, apy_cosmo.FLRW):
            raise TypeError("Cosmology must be an astropy.cosmology.FLRW "
                            "instance.")
        if new_cosmo is None:
            self._cosmo = _default_cosmo
        self._cosmo = new_cosmo

        # Reset dscale:
        self._set_dscale()

    @property
    def dscale(self):
        return self._dscale

    def _set_dscale(self):
        self._dscale = self._cosmo.arcsec_per_kpc_proper(self._z).value

    def add_observation(self, obs):
        """
        Add an observation to the galaxy.observations ordered dict.
        """
        obs_name = obs.name
        if obs_name in self.observations:
            logger.warning('Overwriting observation {}!'.format(obs_name))
        self.observations[obs_name] = obs


    def get_observation(self, obs_name):
        """
        Retrieve an observation from the galaxy.observations ordered dict.
        """
        try:
            return self.observations[obs_name]
        except KeyError:
            raise KeyError('{} not in self.observations !'.format(obs_name))


    def create_model_data(self, obs_list=None):
        r"""
        Function to simulate data for the galaxy

        The function will initially generate a data cube that will then be optionally
        reduced to 2D, 1D, or single spectrum data if specified. The generated cube
        can be accessed via `Galaxy.model_cube` and the generated final data products
        via `Galaxy.model_data`. Both of these attributes are `data_classes.Data` instances.

        Calls the per-observation `Observation.create_single_obs_model_data` method

        Parameters
        ----------
        obs_list : list, optional
            List of observations to make models for.
            If omitted, will default to making models for all observations in the galaxy.

        """
        if obs_list is None:
            obs_list = self.observations.keys()

        for obs_name in obs_list:
            # Get the individual observation
            obs = self.observations[obs_name]
            obs.create_single_obs_model_data(self.model, self.dscale)

            #####################


    def preserve_self(self, filename=None, save_data=True, overwrite=False):
        # Check for existing file:
        if (not overwrite) and (filename is not None):
            if os.path.isfile(filename):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
                return None

        if filename is not None:
            galtmp = copy.deepcopy(self)

            if not save_data:
                for obs_name in galtmp.observations:
                    galtmp.observations[obs_name].data = None
                    galtmp.observations[obs_name].model_data = None
                    galtmp.observations[obs_name].model_cube = None

            _pickle.dump(galtmp, open(filename, "wb") )

    def load_self(self, filename=None):
        """
        Load a saved Galaxy from a pickle file

        Parameters
        ----------
        filename : str
            Name of the file with saved Galaxy

        Returns
        -------

        """
        if filename is not None:
            galtmp = _pickle.load(open(filename, "rb"))
            return galtmp


    def save_model_data(self, filename=None, overwrite=False):

        logger.warning("Only writing 1 observation! FIX ME!!!!")
        # Check for existing file:
        if (not overwrite) and (filename is not None):
            if os.path.isfile(filename):
                logger.warning("overwrite={} & File already exists! Will not save file. \n {}".format(overwrite, filename))
                return None

        if filename is not None:
            for obs_name in self.observations:
                obs = self.observations[obs_name]
                write_model_obs_file(obs=obs, model=self.model, fname=filename, overwrite=overwrite)



def load_galaxy_object(filename=None):
    """
    Load a saved Galaxy from a pickle file

    Parameters
    ----------
    filename : str
               Name of the file with saved Galaxy

    Returns
    -------
    gal: Galaxy object
         The saved dysmalpy Galaxy object

    """
    gal = Galaxy()
    gal = gal.load_self(filename=filename)
    return gal
