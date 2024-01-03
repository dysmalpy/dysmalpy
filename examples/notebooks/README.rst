.. _readme_examples:
.. highlight:: shell

==================================
Description of the notebook examples
==================================

This directory contains multiple examples of the various capabilities of ``DysmalPy``. 
It is adviced that the user gets familiar with the code by running and understanding these
examples. 

With a working installation of ``DysmalPy``, the examples can be run simply by changing the
corresponding paths in the examples to the paths on your system. The examples are divided into the following categories:

1D examples:
------------

- **dysmalpy_example_fit_1D.ipynb:** *Find the best fit models for the 1D velocity and velocity dispersion profiles of galaxy GS4_43501 (z=1.613) using both MPFIT and MCMC as well as nested sampling*
- **dysmalpy_example_fit_1D_old_no_dynesty.ipynb:** *Same as above but without the nested sampling part*
- **dysmalpy_example_fitting_wrapper_1D.ipynb** *Same as the first notebook but using the wrapper functions*

2D examples:
------------

- **dysmalpy_example_fit_2D.ipynb:** *Find the best fit models for the 2D velocity and velocity dispersion maps of galaxy GS4_43501 (z=1.613) using both MPFIT and MCMC as well as nested sampling*
- **dysmalpy_example_fitting_wrapper_2D.ipynb:** *Same as the first notebook but using the wrapper functions*


3D examples:
------------

- **dysmalpy_example_fit_3D.ipynb:** *Find the best fit models for the 3D velocity and velocity dispersion maps of galaxy GS4_43501 (z=1.613) using MPFIT and generating a mask for the IFU cube*

Outflow examples:
-----------------

- **dysmalpy_example_fit_outflow.ipynb:** *Fit a biconical outflow model of galaxy NGC5728 using MPFIT and MCMC*

Masking examples:
-----------------

- **dysmalpy_example_fit_masking.ipynb:** *This example expands on dysmalpy_example_fitting_wrapper_3D.ipynb to demonstrate Dysmalpy's built-in masking function*

Multi-observation examples:
---------------------------

- **dysmalpy_example_multiobs.ipynb:** *This notebook shows the user how to do multi-observation fitting (either from data at different spatial resolution or at different wavelength ranges).*

Model examples:
---------------

- **dysmalpy_example_model.ipynb:** *This notebook shows the user how to use DysmalPy to generate a model flux, velocity and velocity dispersion profiles for a given galaxy.*
- **dysmalpy_example_model_wrapper.ipynb:** *Same as above but using the wrapper functions*
- **dysmalpy_example_model_hiord.ipynb:** *Same as above but using the higher order kinematic components*
- **dysmalpy_example_model_hiord_wrapper.ipynb:** *Same as above but using the wrapper functions*


Python script examples:
-----------------------

- **dysmalpy_quickstart_example.ipynb:** *A quickstart example of how to use DysmalPy (the contents of the pyhon script are embedded into the notebook format)*
- **dysmalpy_quickstart_example.py:** *Same as above but in python script format*
- **dysmalpy_example_fit_mcmc_full_1D.py:** *Similar as dysmalpy_example_fit_1D.ipynb but in python script format*
- **dysmalpy_example_fit_dynesty_full_1D.py:** *Similar as dysmalpy_example_fit_1D.ipynb but using dynesty instead of emcee*
- **dysmalpy_example_fit_mcmc_full_2D.py:** *Similar as dysmalpy_example_fit_2D.ipynb but in python script format*

Feedback
-----------

If you find any issues in these examples, please contact the maintainers at 
dysmalpy@mpe.mpg.de.
