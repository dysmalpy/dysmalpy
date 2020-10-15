# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# This module contains our new Parameter class which applies a prior function
# to a parameter

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

# Standard library
import abc
import operator
import functools

# Third party
import numpy as np
from scipy.stats import norm
from astropy.modeling import Parameter
from astropy.units import Quantity
import six

__all__ = ['DysmalParameter', 'Prior', 'UniformPrior', 'GaussianPrior',
           'BoundedGaussianPrior']


def _binary_comparison_operation(op):
    @functools.wraps(op)
    def wrapper(self, val):

        if self._model is None:
            if op is operator.lt:
                # Because OrderedDescriptor uses __lt__ to work, we need to
                # call the super method, but only when not bound to an instance
                # anyways
                return super(Parameter, self).__lt__(val)
            else:
                return NotImplemented

        if self.unit is not None:
            self_value = Quantity(self.value, self.unit)
        else:
            self_value = self.value

        return op(self_value, val)

    return wrapper


# ******* PRIORS ************
# Base class for priors
@six.add_metaclass(abc.ABCMeta)
class Prior:
    """
    Base class for priors
    """

    @abc.abstractmethod
    def log_prior(self, *args, **kwargs):
        """Returns the log value of the prior given the parameter value"""

    @abc.abstractmethod
    def sample_prior(self, *args, **kwargs):
        """Returns a random sample of parameter values distributed according to the prior"""


class UniformPrior(Prior):
    """
    Object for flat priors
    """

    @staticmethod
    def log_prior(param, **kwargs):
        """
        Returns the log value of the prior given the parameter value

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object for which the prior is associated to
        modelset :

        Returns
        -------
        lprior : `0` or `-np.inf`
            Log prior value. 0 if the parameter value is within the bounds otherwise `-np.inf`
        """

        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (param.value >= pmin) & (param.value <= pmax):
            return 0.
        else:
            return -np.inf

    @staticmethod
    def sample_prior(param, N=1):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object for which the prior is associated to
        N : int, optional
            Size of random sample. Default is 1.

        Returns
        -------
        rsamp : float or array
            Random sample of parameter values

        """
        if param.bounds[0] is None:
            pmin = -1.e5  # Need to default to a finite value for the rand dist.
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = 1.e5  # Need to default to a finite value for the rand dist.
        else:
            pmax = param.bounds[1]

        return np.random.rand(N)*(pmax-pmin) + pmin


# CAN THIS? BC NEED TO USE LinearDiskBulge / etc, bc of walker jumps ?????
class UniformLinearPrior(Prior):
    # Note: must bounds input as LINEAR BOUNDS
    @staticmethod
    def log_prior(param, **kwargs):

        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (np.power(10., param.value) >= pmin) & (np.power(10., param.value) <= pmax):
            return 0.
        else:
            return -np.inf

    @staticmethod
    def sample_prior(param, N=1):
        if param.bounds[0] is None:
            pmin = -1.e13  # Need to default to a finite value for the rand dist.
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = 1.e13  # Need to default to a finite value for the rand dist.
        else:
            pmax = param.bounds[1]

        rvs = []
        while len(rvs) < N:
            test_v = np.random.rand(1)[0] * (pmax-pmin) + pmin

            if (test_v >= pmin) & (test_v <= pmax):
                rvs.append(np.log10(test_v))

        return rvs


class GaussianPrior(Prior):
    """
    Object for gaussian priors

    Parameters
    ----------
    center : float
        Mean of the Gaussian prior

    stddev : float
        Standard deviation of the Gaussian prior
    """
    def __init__(self, center=0, stddev=1.0):

        self.center = center
        self.stddev = stddev

    def log_prior(self, param, **kwargs):
        """
        Returns the log value of the prior given the parameter value

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object for which the prior is associated to
        modelset :

        Returns
        -------
        lprior : float
            Log prior value calculated using `~scipy.stats.norm.pdf`
        """
        return np.log(norm.pdf(param.value, loc=self.center,
                        scale=self.stddev))

    def sample_prior(self, param, N=1):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object for which the prior is associated to
        N : int, optional
            Size of random sample. Default is 1.

        Returns
        -------
        rsamp : float or array
            Random sample of parameter values

        """
        return np.random.normal(loc=self.center,
                                scale=self.stddev, size=N)


class BoundedGaussianPrior(Prior):
    """
    Object for Gaussian priors that only extend to a minimum and maximum value

    Parameters
    ----------
    center : float
        Mean of the Gaussian prior

    stddev : float
        Standard deviation of the Gaussian prior
    """
    def __init__(self, center=0, stddev=1.0):

        self.center = center
        self.stddev = stddev

    def log_prior(self, param, **kwargs):
        """
        Returns the log value of the prior given the parameter value

        The parameter value is first checked to see if its within `param.bounds`.
        If so then the standard Gaussian distribution is used to calculate the prior.

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object for which the prior is associated to
        modelset :

        Returns
        -------
        lprior : float
            Log prior value calculated using `~scipy.stats.norm.pdf` if `param.value` is within
            `param.bounds`
        """

        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (param.value >= pmin) & (param.value <= pmax):
            return np.log(norm.pdf(param.value, loc=self.center, scale=self.stddev))
        else:
            return -np.inf

    def sample_prior(self, param, N=1):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object for which the prior is associated to
        N : int, optional
            Size of random sample. Default is 1.

        Returns
        -------
        rsamp : float or array
            Random sample of parameter values

        """
        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        rvs = []
        while len(rvs) < N:

            test_v = np.random.normal(loc=self.center, scale=self.stddev,
                                      size=1)[0]
            if (test_v >= pmin) & (test_v <= pmax):

                rvs.append(test_v)

        return rvs


class BoundedGaussianLinearPrior(Prior):

    def __init__(self, center=0, stddev=1.0):

        self.center = center
        self.stddev = stddev

    def log_prior(self, param, **kwargs):

        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (np.power(10., param.value) >= pmin) & (np.power(10., param.value) <= pmax):
            return np.log(norm.pdf(np.power(10., param.value), loc=self.center, scale=self.stddev))
        else:
            return -np.inf

    def sample_prior(self, param, N=1):

        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        rvs = []
        while len(rvs) < N:

            test_v = np.random.normal(loc=self.center, scale=self.stddev,
                                      size=1)[0]
            if (test_v >= pmin) & (test_v <= pmax):

                rvs.append(np.log10(test_v))

        return rvs

class BoundedSineGaussianPrior(Prior):
    """
    Object for priors described by a bounded sine Gaussian distribution

    Parameters
    ----------
    center : float
        Central value of the Gaussian prior BEFORE applying sine function

    stddev : float
        Standard deviation of the Gaussian prior AFTER applying sine function
    """

    def __init__(self, center=0, stddev=1.0):
        # Center, bounds in INC
        # Stddev in SINE INC

        self.center = center
        self.stddev = stddev

        self.center_sine = np.sin(self.center*np.pi/180.)

    def log_prior(self, param, **kwargs):
        """
        Returns the log value of the prior given the parameter value

        The parameter value is first checked to see if its within `param.bounds`.
        If so then a Gaussian distribution on the sine of the parameter is used to
        calculate the prior.

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object for which the prior is associated to
        modelset :

        Returns
        -------
        lprior : float
            Log prior value
        """
        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        if (param.value >= pmin) & (param.value <= pmax):
            return np.log(norm.pdf(np.sin(param.value*np.pi/180.), loc=self.center_sine, scale=self.stddev))
        else:
            return -np.inf

    def sample_prior(self, param, N=1):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object for which the prior is associated to
        N : int, optional
            Size of random sample. Default is 1.

        Returns
        -------
        rsamp : float or array
            Random sample of parameter values

        """
        if param.bounds[0] is None:
            pmin = -np.inf
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:
            pmax = np.inf
        else:
            pmax = param.bounds[1]

        rvs = []
        while len(rvs) < N:

            test_v_sine = np.random.normal(loc=self.center_sine, scale=self.stddev,
                                      size=1)[0]
            test_v = np.abs(np.arcsin(test_v_sine))*180./np.pi
            if (test_v >= pmin) & (test_v <= pmax):

                rvs.append(test_v)

        return rvs


class DysmalParameter(Parameter):
    """
    New parameter class for dysmalpy based on `~astropy.modeling.Parameter`

    The main change is adding a prior as part of the constraints
    """
    constraints = ('fixed', 'tied', 'bounds', 'prior')

    def __init__(self, name='', description='', default=None, unit=None,
                 getter=None, setter=None, fixed=False, tied=False, min=None,
                 max=None, bounds=None, prior=None):

        if prior is None:
            prior = UniformPrior

        super(DysmalParameter, self).__init__(name=name,
                                              description=description,
                                              default=default,
                                              unit=unit,
                                              getter=getter,
                                              setter=setter,
                                              fixed=fixed,
                                              tied=tied,
                                              min=min,
                                              max=max,
                                              bounds=bounds)

        # Set prior:
        self.prior = prior



    __eq__ = _binary_comparison_operation(operator.eq)
    __ne__ = _binary_comparison_operation(operator.ne)
    __lt__ = _binary_comparison_operation(operator.lt)
    __gt__ = _binary_comparison_operation(operator.gt)
    __le__ = _binary_comparison_operation(operator.le)
    __ge__ = _binary_comparison_operation(operator.ge)
