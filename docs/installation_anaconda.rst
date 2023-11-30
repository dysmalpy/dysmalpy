.. _install-conda:
.. highlight:: shell

========================
Python Environment Setup
========================

This guide walks you through the steps needed to
set up a DysmalPy-ready python installation
with the required dependencies.

If you already have an Anaconda python installation on your computer,
we recommend making a new environment (see :ref:`'Creating an Anaconda environment'<conda_env_create>`)
for installing the DysmalPy dependencies.

.. note::
    If you install the DysmalPy dependencies in an Anaconda environment,
    and not the root installation, then you will need to activate this
    environment before running DysmalPy.
    (See :ref:`'Activating an Anaconda environment'<conda_env_activate>`).


.. _conda_install:

Install Anaconda
----------------

Currently, Anaconda has installers available for Windows, MacOS, and Linux,
which can be downloaded from their website: `Anaconda downloads`_.

.. _Anaconda downloads: https://www.anaconda.com/products/individual#Downloads

This installer (GUI for Windows and MacOS; shell script for Linux) will
guide you through the installation process. However, for further instructions or
in case problems arise, we refer you to the
`Anaconda installation guide`_.

.. _Anaconda installation guide: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

The full Anaconda installation requires at least 3 GB of disk space initially.
(Alternatively, you can install `Miniconda`_, which requires only 400 MB of disk space initially)

.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html



.. _conda_env_create:

Creating an Anaconda environment
--------------------------------

.. note::
    If you already had an installation of Anaconda,
    you should ensure it's up to date before proceeding by running
    ``$ conda update conda``

If you already have an Anaconda python installation, we suggest creating a new environment where
the DysmalPy dependencies can be installed. This ensures that any package dependency requirements
do not conflict with dependencies of any other packages you have installed.

To create a new environment with python installed, from the terminal or an Anaconda Prompt,
use the following command:

.. code-block:: console

    $ conda create --name my-env python


or, if you want to select a specific version of python, use for example:

.. code-block:: console

    $ conda create --name my-env python=3.10

.. warning::
    DysmalPy requires python version ``>=3.10``.


Then follow the prompts to finish creating the new environment ``my-env``.

(Further information about Anaconda environments can be found `here`_).

.. _here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


Now that you have a dedicated environment for DysmalPy, you will either need to
activate the environment before you run DysmalPy, or activate it by default by
modifying your shell login script (e.g., your .bashrc, .bash_profile, .profile, .tcshrc, ... file).



.. _conda_env_activate:

Activating an Anaconda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have installed the DysmalPy dependencies in the non-root Anaconda environment,
you activate this environment with the following command:

.. code-block:: console

    $ conda activate my-env

or for MacOS and Linux

.. code-block:: console

    $ source activate my-env

or for Windows:

.. code-block:: console

    $ activate my-env


To deactivate the environment, you can then use the command:

.. code-block:: console

    $ conda deactivate

or for MacOS and Linux

.. code-block:: console

    $ source deactivate

or for Windows:

.. code-block:: console

    $ deactivate

.. tip::
    Check that your environment is activated by running

    .. code-block:: console

        $ which python

    This should return a path ending with ``envs/my-env/bin/python``.

    If this is **not** the version of python returned, you may have to
    run ``$ conda deactivate`` twice (first to deactivate ``my-env``, then to
    deactivate ``base``).
    Then reactivate ``my-env`` by running ``$ conda activate my-env``.



.. _add_channels:

Adding channels to ``conda``
----------------------------

A number of the DysmalPy dependencies are not available in the default Anaconda channels,
but are instead available in the ``astroconda`` channel.
To ensure this channel is installed, from the terminal or the Anaconda Prompt, run:

.. code-block:: console

    $ conda config --add channels http://ssb.stsci.edu/astroconda

To verify the channel has been added, check that the ``astroconda`` url shows up in
the list of channels returned by the following command:

.. code-block:: console

    $ conda config --show channels




.. _install_deps:

Installing DysmalPy dependencies with ``conda`` and ``pip``
-----------------------------------------------------------

The benefit of using an Anaconda python distribution is the easy management of
packages, and all of their dependencies.

Most of the dependencies of DysmalPy can be installed with ``conda``. The remainder
will be installed using ``pip`` at the end.

.. attention::
    If you are using an environment, activate it before proceeding with the
    installation of the dependencies.
    See :ref:`Activating an Anaconda environment <conda_env_activate>`.


.. attention::
    Before beginning with dependency installation, make sure the
    ``astroconda`` channel has been added to conda.
    See :ref:`Adding channels to conda <add_channels>`.


#. Install dependencies with ``conda``:

    We will use ``conda`` to install `AstroPy`_, `emcee`_, `corner`_, `shapely`_,
    and `photutils`_.
    We will also ensure that `ipython`_, `NumPy`_, `SciPy`_, `matplotlib`_, `cython`_,
    and ``dill`` are installed, as well as a number of other ``astropy`` dependencies.

    .. _ipython: https://ipython.org/
    .. _NumPy: https://numpy.org/
    .. _SciPy: https://scipy.org
    .. _matplotlib: https://matplotlib.org
    .. _AstroPy: https://astropy.org
    .. _emcee: https://emcee.readthedocs.io
    .. _corner: https://corner.readthedocs.io
    .. _shapely: https://github.com/Toblerity/Shapely
    .. _cython: https://cython.org
    .. _photutils: https://photutils.readthedocs.io

    From the terminal or an Anaconda prompt, run the following:

    .. code-block:: console

        $ conda install astropy ipython numpy scipy matplotlib cython dill 
        $ conda install -c astropy -c defaults h5py pandas
        $ conda install -c conda-forge -c astropy photutils emcee shapely corner  
        $ conda install -c conda-forge dynesty


#. Install remaining dependencies with ``pip``:

    We will then use ``pip`` to install `spectral-cube`_ and `radio-beam`_.

        .. _spectral-cube: https://spectral-cube.readthedocs.io
        .. _radio-beam: https://radio-beam.readthedocs.io

    Again from the terminal or an Anaconda prompt, run:

    .. code-block:: console

        $ pip install spectral-cube radio-beam


#. *OPTIONAL*: Install libraries for C++ extensions

    To compile the Dysmalpy C++ Gaussian least-squares fitter and the lensing modules,
    the ``gsl``, ``cfitsio``, and ``libcblas`` libraries are needed.

    These can be installed separately (using your normal means), or can be installed
    as follows:

    .. code-block:: console

        $ conda install gsl cfitsio
        $ conda install -c conda-forge libcblas


    Note that the installation directory will be needed later when compiling the
    extensions. This is either `/PATH/TO/ANACONDA` if using anaconda as above
    (where the base `/PATH/TO/ANACONDA` should be listed under the "active env location"
    from the output of `$ conda info`), or whatever directory was specified
    for the separate install.


.. note::
    If AstroPy is already installed, it can be updated to the
    most recent version by running ``$ conda update astropy``.
    (See also the `AstroPy installation documentation`_.)

.. _AstroPy installation documentation: https://docs.astropy.org/en/stable/install.html#using-conda


.. tip::
    If for some reason the package can't be found, try running the installation by
    specifying the ``astropy`` or ``conda-forge`` channels:
    ``$ conda install -c astropy PACKAGE``
    or
    ``$ conda install -c conda-forge PACKAGE``

    If this still fails, as a last resort try to use ``pip`` to install the package by running:
    ``$ pip install PACKAGE``
