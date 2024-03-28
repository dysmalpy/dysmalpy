.. dysmalpy documentation master file, created by
   sphinx-quickstart on Fri Oct  9 09:33:11 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. the "raw" directive below is used to hide the title in favor of
   just the logo being visible

=========
dysmalpy
=========


DysmalPy (DYnamical Simulation and Modelling ALgorithm in PYthon) is a 
Python-based forward modeling code designed for analyzing galaxy kinematics. 
It was originally inspired by Reinhard Genzel's DISDYN program 
(e.g., `Tacconi et al. 1994`_), has been developed and is maintained at 
the Max Planck Institute for Extraterrestrial Physics (MPE). 
It extends the IDL-based DYSMAL fitting models introduced and thoroughly 
tested in previous works (`Davies et al. 2004a`_; `Davies et al. 2004b`_; 
`Cresci et al. 2009`_; `Davies et al. 2011`_) as well as subsequent 
improvements described by `Wuyts et al. 2016`_; `Lang et al. 2017`_; 
`Übler et al. 2018`_. Its Python incarnation and latest developments and 
testing are presented by `Price et al. 2021`_ and Lee et al. 2024, in prep. 

Dysmalpy is a Python-based forward modeling code designed for analyzing galaxy 
kinematics. It has been developed and maintained at the Max Planck Institute 
for Extraterrestrial Physics (MPE), and it extends the DYSMAL fitting models 
introduced in previous thoroughly tested works (`Cresci et al. 2009`_ and 
`Davies et al. 2011`_, as well as subsequent improvements described in 
`Wuyts et al. 2016`_, `Genzel et al. 2017`_, and `Übler et al. 2018`_).

The code employs a set of models that describe the mass distribution and 
various kinematic components to describe and fit the kinematics of galaxies. 
Dysmalpy includes many features, including support for multiple halo profiles,
flexibility in modeling baryon components such as non-circular higher-order 
kinematic features, multi-observation fitting, the ability to tie model 
component parameters together, and options for fitting using either 
least-squares minimization (with `MPFIT`_) or Markov chain Monte Carlo (MCMC) 
posterior sampling (with `emcee`_) or dynamic nested sampling (with `Dynesty`_). 

Dysmalpy is parametric in nature, allowing the direct fitting of the intrinsic galaxy 
properties, exploration of mass decomposition, dark matter fractions, and 
assessment of parameter degeneracies and associated uncertainties. This stands 
in contrast to a non-parametric kinematic fitting approach, which requires 
additional steps for interpreting recovered intrinsic galaxy kinematic 
properties.

The forward modeling process involves simulating the mass distribution of a 
galaxy, generating a 3D mock cube capturing composite kinematics, and 
accounting for observational effects such as beam smearing and instrumental 
line broadening. The model cube can be directly compared to the datacube in 3D, 
but it can also be compared to 1D or 2D kinematic observations by extracting 
the corresponding one or two-dimensional profiles following the same procedure 
that was used on the observed data. For detailed information, refer to the 
Appendix in `Price et al. 2021`_ as well as Lee et al. 2024, in prep.

The overall basic usage of DYSMALPY can be summarised as follows:

1.  **Setup steps:** Import modules, set paths, define global constants and variables
2.  **Initialise:** Create a galaxy object with its corresponding parameters, add the model set (disk, bulge, DM halo, etc), set up the observation and instrument information.
3.  **Fitting:** Perform fitting/bayesian sampling in either 1D, 2D, or 3D using either MPFIT, MCMC or dynamic nested sampling.
4.  **Assess:** Visualise, assess fit, and fine tune the fitting

We strongly recommend to follow and understand the `tutorials`_ section of the 
main website. Alternatively, you can run and familiarize yourself with the '
jupyter notebooks in the `notebooks`_ folder (these will be included in your 
installation of dysmalpy under examples/notebooks).


.. _MPFIT: https://code.google.com/archive/p/astrolibpy
.. _emcee: https://emcee.readthedocs.io
.. _Dynesty: https://dynesty.readthedocs.io
.. _installation instructions: https://github.com/dysmalpy/dysmalpy/blob/add_dynesty/docs/installation.rst
.. _notebooks: https://github.com/dysmalpy/dysmalpy/tree/juan_edits/examples/notebooks
.. _tutorials: https://www.mpe.mpg.de/resources/IR/DYSMALPY/
.. _Tacconi et al. 1994: https://ui.adsabs.harvard.edu/abs/1994ApJ...426L..77T/abstract
.. _Davies et al. 2004a: https://ui.adsabs.harvard.edu/abs/2004ApJ...602..148D/abstract
.. _Davies et al. 2004b: https://ui.adsabs.harvard.edu/abs/2004ApJ...613..781D/abstract
.. _Cresci et al. 2009: https://ui.adsabs.harvard.edu/abs/2009ApJ...697..115C/abstract
.. _Davies et al. 2011: https://ui.adsabs.harvard.edu/abs/2011ApJ...741...69D/abstract
.. _Wuyts et al. 2016: https://ui.adsabs.harvard.edu/abs/2016ApJ...831..149W/abstract
.. _Lang et al. 2017: https://ui.adsabs.harvard.edu/abs/2017ApJ...840...92L/abstract
.. _Genzel et al. 2017: https://ui.adsabs.harvard.edu/abs/2017Natur.543..397G/abstract
.. _Übler et al. 2018: https://ui.adsabs.harvard.edu/abs/2018ApJ...854L..24U/abstract
.. _Price et al. 2021: https://ui.adsabs.harvard.edu/abs/2021ApJ...922..143P/abstract


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
   workflow_on_how_to_contribute.rst
   how_to_build_and_deploy_documentation.rst

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials/dysmalpy_quickstart_example.rst
   tutorials/aperture_drawing.rst
   tutorials/models/index.rst
   tutorials/fitting/index.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
