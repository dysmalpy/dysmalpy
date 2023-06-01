.. _overview_code_structure:
.. highlight:: shell

=======================
Code Structure Overview
=======================


Most DysmalPy functionality is built around using ``galaxy`` objects,
which are instances of the ``Galaxy`` class.
The key attributes of ``galaxy`` are ``model`` and ``observations``, 
an ordered list which contains one or more ``Observation`` instances that 
each have their own ``instrument`` attribute, 
as well as optional attributes of ``data``, ``mod_options``, ``fit_options``.

The ``data`` object contains information about the observed kinematic data,
and any specifics of how it was extracted
(e.g., aperture settings, gaussian or moment extraction).
The ``instrument`` object contains ``beam`` and ``lsf``, encoding the
information to convolve an intrinsic model cube to match the observational data
(i.e., the PSF and LSF).
First, ``model`` itself contains a number of model objects in ``model.components``.
These components are taken together to generate the full intrinsic 3D model cube
using ``model.simulate_cube()``.

To generate a model matching the observed data, and including the instrumental effects,
``galaxy.create_model_data()`` is used (which begins by calling
``model.simulate_cube()`` for each ``observation`` instance within ``galaxy`` 
before convolving with that ``observation``'s ``instrument`` instance 
and performing any extraction).

To perform fitting with DysmalPy, a ``galaxy`` must be
constructed containing the appropriate observational data, instrumental settings,
and the model to be fit (including specifying which of the model parameters are
free or fixed, and their constraints).
The ``galaxy`` object is then passed as input to
``fitter.fit(galaxy, **kwargs)``, where ``fitter`` is an instance of one of the 
possible ``Fitter`` classes (including ``MPFITFitter``, ``MCMCFitter``, and ``NestedFitter``). 
These fitting routines then return a ``results`` object,
which is an instance of the  ``MPFITResults``, ``MCMCResults``, or ``NestedResults`` 
class, as appropriate.

Below is an overview schematic of the key classes and functionality of
DysmalPy.

.. image:: _static/dpy_code_schematic/dysmalpy_code_structure.svg
