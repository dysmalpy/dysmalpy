.. dysmalpy documentation master file, created by
   sphinx-quickstart on Fri Oct  9 09:33:11 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. the "raw" directive below is used to hide the title in favor of
   just the logo being visible

=========
dysmalpy
=========

Dysmalpy is a Python-based forward modeling code designed for analyzing galaxy kinematics. It has been developed and maintained at the Max Planck Institute for Extraterrestrial Physics (MPE) and it extends the DYSMAL fitting models introduced in previous thoroughly-tested works (Cresci et al. 2009 and Davies et al. 2011, as well as subsequent improvements described in Wuyts et al. 2016, Genzel et al. 2017, and Übler et al. 2018).

The code employs a set of models that describe the mass distribution and various kinematic components to describe and fit the kinematics of galaxies. Dysmalpy includes many features, including support for multiple halo profiles, flexibility in modeling baryon components such as non-circular higher-order kinematic features, multi-observation fitting, the ability to tie model component parameters together, and options for fitting using either least-squares minimization (with `MPFIT`_) or Markov chain Monte Carlo (MCMC) posterior sampling (with `emcee`_) or dynamic nested sampling (with `Dynesty`_). 

Dysmalpy is parametric in nature, allowing direct fitting of intrinsic galaxy properties, exploration of mass decomposition, dark matter fractions, and assessment of parameter degeneracies and associated uncertainties. This stands in contrast to a non-parametric kinematic fitting approach, which requires additional steps for interpreting recovered intrinsic galaxy kinematic properties.

The forward modeling process involves simulating the mass distribution of a galaxy, generating a 3D mock cube capturing composite kinematics, and accounting for observational effects such as beam smearing and instrumental line broadening. The model cube can be directly compared to the datacube in 3D, but it can also be compared to 1D or 2D kinematic observations by extracting the corresponding one or two dimensional profiles following the same procedure that was used on the observed data. For detailed information, refer to the Appendix in Price et al. (2021).

The overall basic usage of DYSMALPY can be summarized as follows:

1) **Setup steps:** Import modules, set paths, define global constants and variables
2) **Initialize:** Create a galaxy object with its corresponding parameters, add the model set (disk, bulge, DM halo, etc), set up the observation and instrument information.
3) **Fitting:** Perform fitting/bayesian sampling in either 1D, 2D, or 3D using either MPFIT, MCMC or dynamic nested sampling.
4) **Assess:** Visualise, assess fit, and fine tune the fitting


.. _MPFIT: https://code.google.com/archive/p/astrolibpy/issues
.. _emcee: https://emcee.readthedocs.io
.. _Dynesty: https://dynesty.readthedocs.io



.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation.rst
   installation_anaconda.rst
   examples_downloads.rst
   overview_code_structure.rst
   api.rst
   acknowledging_and_referencing.rst
   bugfix_instructions.rst
   mailing_list.rst

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials/dysmalpy_quickstart_example.rst
   tutorials/models/index.rst
   tutorials/fitting/index.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
