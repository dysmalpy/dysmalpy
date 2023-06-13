# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Dispersion models for DysmalPy

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

# Third party imports
import numpy as np

# Local imports
from .base import _DysmalFittable1DModel
from dysmalpy.parameters import DysmalParameter

__all__ = ['DispersionConst']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")


# ******* Dispersion Profiles **************
class DispersionProfile(_DysmalFittable1DModel):
    """Base object for dispersion profile models"""
    _type = 'dispersion'

    tracer = 'LINE'

    def __init__(self, tracer=None, **kwargs):
        if tracer is None:
            raise ValueError("Dispersion profiles must have a 'tracer' specified!")

        self.tracer = tracer

        super(DispersionProfile, self).__init__(**kwargs)


class DispersionConst(DispersionProfile):
    """
    Model for a constant dispersion

    Parameters
    ----------
    sigma0 : float
        Value of the dispersion at all radii

    tracer : string
        (Attribute): Name of the dynamical tracer
    """
    sigma0 = DysmalParameter(default=10., bounds=(0, None), fixed=True)

    @staticmethod
    def evaluate(r, sigma0):
        """Dispersion as a function of radius"""
        return np.ones(r.shape)*sigma0
