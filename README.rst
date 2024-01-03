Dynamical Simulation and Modeling Algorithm
-------------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

.. image:: docs/_static/dpy_logo_spiral/DPy_h_blk_wh.png
   :alt: DPy Logo

DYSMALPY website: https://www.mpe.mpg.de/resources/IR/DYSMALPY/

Dysmalpy is a Python-based forward modeling code designed for analyzing galaxy 
kinematics. It has been developed and maintained at the Max Planck Institute 
for Extraterrestrial Physics (MPE) and it extends the DYSMAL fitting models 
introduced in previous thoroughly tested works (Cresci et al. 2009 and Davies 
et al. 2011, as well as subsequent improvements described in Wuyts et al. 2016, 
Genzel et al. 2017, and Übler et al. 2018).

The code employs a set of models that describe the mass distribution and 
various kinematic components to describe and fit the kinematics of galaxies. 
Dysmalpy includes many features, including support for multiple halo profiles,
flexibility in modeling baryon components such as non-circular higher-order 
kinematic features, multi-observation fitting, the ability to tie model 
component parameters together, and options for fitting using either 
least-squares minimization (with `MPFIT`_) or Markov chain Monte Carlo (MCMC) 
posterior sampling (with `emcee`_) or dynamic nested sampling (with `Dynesty`_). 

Dysmalpy is parametric in nature, allowing direct fitting of intrinsic galaxy 
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
Appendix in Price et al. (2021).


.. _MPFIT: https://code.google.com/archive/p/astrolibpy
.. _emcee: https://emcee.readthedocs.io
.. _Dynesty: https://dynesty.readthedocs.io
.. _installation: https://github.com/dysmalpy/dysmalpy/blob/add_dynesty/docs/installation.rst
.. _notebooks: https://github.com/dysmalpy/dysmalpy/tree/juan_edits/examples/notebooks
.. _tutorials: https://www.mpe.mpg.de/resources/IR/DYSMALPY/

Dependencies
------------
* python (version >= 3.10)
* numpy (version >= 1.24.3)
* scipy (version >=1.9.3)
* matplotlib
* pandas
* astropy (version >= 5.3)
* multiprocess
* emcee (version >= 3)
* dynesty (version >= 2.0.0)
* corner (version >= 2.2.2)
* dill (version >= 0.3.7)
* photutils (version >= 1.8.0)
* shapely (version >= 2)
* spectral-cube (version >= 0.6.0)
* radio-beam (version >= 0.3.3)
* h5py (version >= 3.8.0)
* six


Installation
------------

To install DYSMALPY, please follow the instructions in the `installation`_ file.

Usage
-----

The overall basic usage of DYSMALPY can be summarized as follows:

**1) Setup steps:** Import modules, set paths, define global constants and 
variables.

**2) Initialize:** Create a galaxy object with its corresponding parameters, 
add the model set (disk, bulge, DM halo, etc), set up the observation and 
instrument information.

**3) Fitting:** Perform fitting/bayesian sampling in either 1D, 2D, or 3D using 
either MPFIT, MCMC or dynamic nested sampling.

**4) Assess:** Visualise, assess fit, and fine tune the fitting. 

We strongly recommend to follow and understand the `tutorials`_ section of the main website. 
Alternatively, you can run and familiarize yourself with the jupyter notebooks in the `examples/notebooks`_ folder (these will be included in your installation of dysmalpy).

Contact
-------

If you have any questions or suggestions, please contact the developers at dysmalpy@mpe.mpg.de.


License
-------

This project is Copyright (c) MPE/IR-Submm Group. See the licenses folder for 
license information. 
