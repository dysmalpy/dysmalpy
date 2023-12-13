# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# This module contains our new Parameter class which applies a prior function
# to a parameter

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

# Standard library
import abc
import operator
import functools
import copy

# Third party
import numpy as np
from scipy.stats import norm, truncnorm
from astropy.modeling import Parameter
from astropy.units import Quantity
import six

__all__ = ['DysmalParameter', 'Prior', 'UniformPrior', 'GaussianPrior',
           'BoundedGaussianPrior', 'BoundedGaussianLinearPrior',
           'BoundedSineGaussianPrior', 'UniformLinearPrior',
           'ConditionalUniformPrior', 'ConditionalEmpiricalUniformPrior']


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



def _f_cond_boundaries(param, modelset, f_bounds):
    """
    Default function f_cond(param, modelset, f_bounds) that takes the parameter and model set as input,
        as well as the boundary function for the conditions.
        It must return True/False if the value does / does not satisfy the conditional requirements.
    """

    # Get boundaries for current param / model values:
    pmin, pmax = f_bounds(param, modelset)

    if (param.value >= pmin) & (param.value <= pmax):
        return True
    else:
        return False

def _f_boundaries_from_cond(param, modelset, f_cond):
    """
    Default function f_bounds(param, modelset, f_cond) that takes the parameter and model set as input,
        as well as the conditional function.
        It will return pmin, pmax that are empirically determined based on conditions
            (up to some rough lower/upper limits).
    This should only be called to initialize walkers, so hopefully it should be robust against
        slight limit inaccuracies.
    """
    if (param.bounds[0] is None) or (param.bounds[0] == -np.inf):
        pmin_pb = -1.e2     # Need to default to a finite value
    else:
        pmin_pb = param.bounds[0]

    if (param.bounds[1] is None) or (param.bounds[1] == np.inf):
        pmax_pb = 1.e2     # Need to default to a finite value
    else:
        pmax_pb = param.bounds[1]


    origstepsize = 0.01
    Nsteps_max = 1001
    Nsteps = np.min([np.int(np.round((pmax_pb-pmin_pb)/origstepsize)), Nsteps_max])
    stepsize = (pmax_pb-pmin_pb)/(1.*Nsteps)
    parr = np.arange(pmin_pb, pmax_pb+stepsize, stepsize)
    condarr = np.zeros(len(parr), dtype=bool)

    rarr = np.arange(0., 15.1, 0.1)

    mod = copy.deepcopy(modelset)

    for i, p in enumerate(parr):
        # Update the param values:
        mod.set_parameter_value(param._model._name, param._name, p)
        condarr[i] = np.all(np.isfinite(mod.circular_velocity(rarr)))

    # Defaults:
    pmin = pmin_pb
    pmax = pmax_pb
    if not np.all(condarr):
        condarr_int = np.array(condarr, dtype=int)
        condarr_delts = condarr_int[1:]-condarr_int[:-1]
        wh_pos = np.where(condarr_delts > 0)[0]
        wh_neg = np.where(condarr_delts < 0)[0]
        if len(wh_pos) > 0:
            ind = wh_pos[-1]+1
            pmin = parr[ind]
        if len(wh_neg) > 0:
            ind = wh_neg[-1]-1
            if ind >= 0:
                pmax = parr[ind]
            else:
                # Probably suspect.... but default to minimum....
                pmax = parr[0]
                pmin -= 1.

    return pmin, pmax

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
    def prior_unit_transform(self, *args, **kwargs):
        """Map a uniform random variable drawn from [0.,1.] to the prior of interest"""

    @abc.abstractmethod
    def sample_prior(self, *args, **kwargs):
        """Returns a random sample of parameter values distributed according to the prior"""


class UniformPrior(Prior):
    """
    Object for flat priors
    """
    def __init__(self):
        pass

    @staticmethod
    def log_prior(param, **kwargs):
        """
        Returns the log value of the prior given the parameter value

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

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


    def prior_unit_transform(self, param, u, **kwargs):
        """
        Transforms a uniform random variable Uniform[0.,1.] to the prior distribution

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        u : float or list-like
            Random uniform variable(s) drawn from Uniform[0.,1.]

        Returns
        -------
        v : float or list-like
            Transformation of the random uniform variable u to random value(s) 
            drawn from the prior distribution.

        """
        if param.bounds[0] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmin = param.bounds[0]
        if param.bounds[1] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmax = param.bounds[1]

        # Scale and shift the unit [0., 1.] to the bounds:
        # v = range * u + min
        v = (pmax-pmin) * u + pmin

        return v

    @staticmethod
    def sample_prior(param, N=1, **kwargs):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
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

    def __init__(self):
        pass

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

    def prior_unit_transform(self, param, u, **kwargs):
        """
        Transforms a uniform random variable Uniform[0.,1.] to the prior distribution

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        u : float or list-like
            Random uniform variable(s) drawn from Uniform[0.,1.]

        Returns
        -------
        v : float or list-like
            Transformation of the random uniform variable u to random value(s) 
            drawn from the prior distribution.

        """

        if param.bounds[0] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmin = param.bounds[0]
        if param.bounds[1] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmax = param.bounds[1]

        # Scale and shift the unit [0., 1.] to the bounds:
        # v = range * u + min
        v = (pmax-pmin) * u + pmin

        return v

    @staticmethod
    def sample_prior(param, N=1, **kwargs):
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
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        Returns
        -------
        lprior : float
            Log prior value calculated using `~scipy.stats.norm.pdf`
        """
        return np.log(norm.pdf(param.value, loc=self.center,
                        scale=self.stddev))


    def prior_unit_transform(self, param, u, **kwargs):
        """
        Transforms a uniform random variable Uniform[0.,1.] to the prior distribution

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        u : float or list-like
            Random uniform variable(s) drawn from Uniform[0.,1.]

        Returns
        -------
        v : float or list-like
            Transformation of the random uniform variable u to random value(s) 
            drawn from the prior distribution.

        """

        v  = norm.ppf(u, loc=self.center, scale=self.stddev)

        return v

    def sample_prior(self, param, N=1, **kwargs):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
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
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

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

    def prior_unit_transform(self, param, u, **kwargs):
        """
        Transforms a uniform random variable Uniform[0.,1.] to the prior distribution

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        u : float or list-like
            Random uniform variable(s) drawn from Uniform[0.,1.]

        Returns
        -------
        v : float or list-like
            Transformation of the random uniform variable u to random value(s) 
            drawn from the prior distribution.

        """

        if param.bounds[0] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:            
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmax = param.bounds[1]

        a = (pmin - self.center) / self.stddev
        b = (pmax - self.center) / self.stddev
        v = truncnorm.ppf(u, a, b, loc=self.center, scale=self.stddev)

        return v

    def sample_prior(self, param, N=1, **kwargs):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
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


    def prior_unit_transform(self, param, u, **kwargs):

        if param.bounds[0] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:            
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmax = param.bounds[1]

        a = (pmin - self.center) / self.stddev
        b = (pmax - self.center) / self.stddev
        v = truncnorm.ppf(u, a, b, loc=self.center, scale=self.stddev)

        return v

    def sample_prior(self, param, N=1, **kwargs):

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
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

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

    def prior_unit_transform(self, param, u, **kwargs):
        """
        Transforms a uniform random variable Uniform[0.,1.] to the prior distribution

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated

        u : float or list-like
            Random uniform variable(s) drawn from Uniform[0.,1.]

        Returns
        -------
        v : float or list-like
            Transformation of the random uniform variable u to random value(s) 
            drawn from the prior distribution.

        """

        if param.bounds[0] is None:
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmin = param.bounds[0]

        if param.bounds[1] is None:            
            raise ValueError("Parameter must have well-defined bounds! bounds: {}".format(param.bounds))
        else:
            pmax = param.bounds[1]

        a = (pmin - self.center) / self.stddev
        b = (pmax - self.center) / self.stddev
        v = truncnorm.ppf(np.sin(u*np.pi/180.), a, b, loc=self.center, scale=self.stddev)

        return v

    def sample_prior(self, param, N=1, **kwargs):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
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


class ConditionalUniformPrior(Prior):
    """
    Object for flat priors, but with boundaries that are conditional on other model parameters.
    """

    def __init__(self, f_bounds=None, f_cond=_f_cond_boundaries):
        """
        Initialize `ConditionalUniformPrior` instance.

        Parameters
        ----------
        f_bounds : function
            Function `f_bounds(param, modelset)` that takes the parameter and model set as input.
            It must return a 2-element array with the lower, upper bounds for the parameter,
            for the given other model parameter settings.
            These will then be used to uniformly sample the parameter within these bounds.

            Note this will not be perfect, given the other parameters will be perturbed within their priors (and thus some of the sampled value tuples may end up in the bad region), but hopefully the MCMC walkers will still be able to work with this.

        f_cond : function, optional
            Function `f_cond(param, modelset, self.f_bounds)` that takes the parameter and model set as input.
            It must return True/False if the value does / does not satisfy the conditional requirements.
            If `True`, then the log prior will be 0., if `False`, then it will be `-np.inf`

            If not set, it will default to a conditional based on the boundary function.

        """
        self.f_bounds = f_bounds
        self.f_cond = f_cond

    def log_prior(self, param, modelset=None, **kwargs):
        """
        Returns the log value of the prior given the parameter value

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
        
        modelset : `~dysmalpy.models.ModelSet`
            Current `~dysmalpy.models.ModelSet`, of which param is a part

        Returns
        -------
        lprior : `0` or `-np.inf`
            Log prior value. 0 if the parameter value is within the bounds otherwise `-np.inf`
        """
        if modelset is None:
            raise ValueError("Must pass `modelset` when calling log_prior() for ConditionalUniformPrior!")

        if (self.f_cond(param, modelset, self.f_bounds)):
            return 0.
        else:
            return -np.inf

    
    def prior_unit_transform(self, param, u, modelset=None, **kwargs):

        raise NotImplementedError("Need to implement in a way that uses something similar to self.f_cond()")

        return v

    def sample_prior(self, param, N=1, modelset=None, **kwargs):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
        N : int, optional
            Size of random sample. Default is 1.

        Returns
        -------
        rsamp : float or array
            Random sample of parameter values

        """
        if modelset is None:
            raise ValueError("Must pass `modelset` when calling sample_prior() for ConditionalUniformPrior!")

        pmin, pmax = self.f_bounds(param, modelset)

        # Catch infs -- move to small/large, but finite, values:
        if (pmin is None) or (pmin == -np.inf):
            pmin = -1.e20  # Need to default to a finite value for the rand dist.

        if (pmax is None) or (pmax == np.inf):
            pmax = 1.e20  # Need to default to a finite value for the rand dist.

        return np.random.rand(N)*(pmax-pmin) + pmin



class ConditionalEmpiricalUniformPrior(Prior):
    """
    Object for flat priors, but with boundaries that are conditional on other model parameters.
    Determined through empirical testing of f_cond, and bounds are then inferred based on
    iterating f_cond.
    """

    def __init__(self, f_cond=None, f_bounds=_f_boundaries_from_cond):
        """
        Initialize ConditionalEmpiricalUniformPrior instance.

        Parameters
        ----------

        f_cond : function
            Function `f_cond(param, modelset, self.f_bounds)` that takes the parameter and model set as input.
            It must return True/False if the value does / does not satisfy the conditional requirements.
            If `True`, then the log prior will be 0., if `False`, then it will be `-np.inf`

        f_bounds : function, optional
            Function `f_bounds(param, modelset, self.f_cond)` that takes the parameter and model set as input.
            It must return a 2-element array with the lower, upper bounds for the parameter,
            for the given other model parameter settings.
            These will then be used to uniformly sample the parameter within these bounds.

            Note this will not be perfect, given the other parameters will be perturbed within their priors (and thus some of the sampled value tuples may end up in the bad region), but hopefully the MCMC walkers will still be able to work with this.

            If not set, it will default to a function that iterates to find boundaries based on `self.f_cond`.

        """
        self.f_cond = f_cond
        self.f_bounds = f_bounds

    def log_prior(self, param, modelset=None, **kwargs):
        """
        Returns the log value of the prior given the parameter value

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
        
        modelset : `~dysmalpy.models.ModelSet`
            Current `~dysmalpy.models.ModelSet`, of which param is a part

        Returns
        -------
        lprior : `0` or `-np.inf`
            Log prior value. 0 if the parameter value is within the bounds otherwise `-np.inf`
        """
        if modelset is None:
            raise ValueError("Must pass `modelset` when calling log_prior() for ConditionalUniformPrior!")

        if (self.f_cond(param, modelset)):
            return 0.
        else:
            return -np.inf


    def prior_unit_transform(self, param, u, modelset=None, **kwargs):

        raise NotImplementedError("Need to implement in a way that uses something similar to self.f_cond()")

        return v
        
    def sample_prior(self, param, N=1, modelset=None, **kwargs):
        """
        Returns a random sample of parameter values distributed according to the prior

        Parameters
        ----------
        param : `~dysmalpy.parameters.DysmalParameter`
            `~dysmalpy.parameters.DysmalParameter` object with which the prior is associated
        N : int, optional
            Size of random sample. Default is 1.

        Returns
        -------
        rsamp : float or array
            Random sample of parameter values

        """
        if modelset is None:
            raise ValueError("Must pass `modelset` when calling sample_prior() for ConditionalUniformPrior!")

        pmin, pmax = self.f_bounds(param, modelset, self.f_cond)

        # Catch infs -- move to small/large, but finite, values:
        if (pmin is None) or (pmin == -np.inf):
            pmin = -1.e20  # Need to default to a finite value for the rand dist.

        if (pmax is None) or (pmax == np.inf):
            pmax = 1.e20  # Need to default to a finite value for the rand dist.

        return np.random.rand(N)*(pmax-pmin) + pmin



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
            prior = UniformPrior()

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

        try:
            # Set prior:
            self.prior = prior
        except:
            # Quick backwards compatibility for AstroPy v3:
            self._prior = prior


    __eq__ = _binary_comparison_operation(operator.eq)
    __ne__ = _binary_comparison_operation(operator.ne)
    __lt__ = _binary_comparison_operation(operator.lt)
    __gt__ = _binary_comparison_operation(operator.gt)
    __le__ = _binary_comparison_operation(operator.le)
    __ge__ = _binary_comparison_operation(operator.ge)
