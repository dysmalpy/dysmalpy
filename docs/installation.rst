.. _install:
.. highlight:: shell

============
Installation
============

.. _install_windows:

Windows `.bat` File
-------------------

A `.bat` file, for running DysmalPy with a parameters file (e.g., ``fitting.params``),
should be available next week.

Prior to using DysmalPy with this `.bat` file, the DysmalPy source code
and dependencies will need to be installed (see :ref:`'From Source'<install_source>`).


.. _install_afs:

AFS Machine
-----------

If you are on an AFS machine, ``dysmalpy`` is located at
`/afs/mpe.mpg.de/astrosoft/dysmalpy`. We have further setup
an Anaconda environment the contains all of the necessary
Python packages to run ``dysmalpy``. To activate this environment
as well as set environment variables, run this command in your
terminal:

.. code-block:: console

    $ source /afs/mpe/astrosoft/dysmalpy/dysmalpy_setup.sh

To check whether the setup ran successfully run:

.. code-block:: console

    $ which python

This should return `/afs/mpe.mpg.de/astrosoft/dysmalpy/anaconda3/bin//python`.
Keep in mind that using this environment will override any environment
you have setup locally and only Python packages installed in the
``dysmalpy`` environment will be available. If there is a package you
would like installed in the environment, please contact either `Taro`_
or `Sedona`_.

.. _Taro: shimizu@mpe.mpg.de
.. _Sedona: sedona@mpe.mpg.de

For those who are more familiar with Python, you can simply add
'/afs/mpe.mpg.de/astrosoft/dysmalpy/dysmalpy/' to your ``PYTHONPATH``
environment variable. Be sure to have all of the required packages
installed.

.. _install_source:

From Source
-----------

``dysmalpy`` can also be installed from source. You will need to setup
Python 3 on your machine and install all of the dependent packages. Please
follow the instructions in :ref:`Python Environment Setup <install-conda>` .


After this is completed, you can download the latest DysmalPy package `here`_
(current version: 1.7.1).

.. _here: releases/dysmalpy-1.7.1.tar.gz


Basic Installation
******************

From a terminal, change directories to where the package was downloaded.

To install the basic DysmalPy functionality (without any of the C++ extensions),
type (where N.N.N is the current version):

.. code-block:: console

    $ tar zxvf dysmalpy-N.N.N.tar.gz
    $ cd dysmalpy-N.N.N
    $ python setup.py install



Installation with extensions
****************************

In order to install DysmalPy with the C++ extensions, we will need to also
build the extensions.

If the `gsl` and `cfitsio` are installed in non-standard locations
(e.g., if they were installed using conda during the dependency setups),
then we will need specify those directories as below.

Typically, if `BASEDIR` is the relevant absolute directory path (e.g., `/PATH/TO/ANACONDA`
if installed with conda, as explained in the :ref:`dependencies setup<install_deps>`),
then `LIBDIR` and `INCLUDEDIR` are `BASEDIR/lib` and `BASEDIR/include`, respectively.

(If they are installed in so the headers are in `/usr/include` or `/usr/local/include`
and the libraries are in `/usr/lib` or `/usr/local/lib`,
the `--include-dirs` and `--lib-dirs` flags can be omitted.)


From a terminal, change directories to where the package was downloaded,
then install the package and build the extensions by running:

.. code-block:: console

    $ tar zxvf dysmalpy-N.N.N.tar.gz
    $ cd dysmalpy-N.N.N
    $ python setup.py build_ext --include-dirs=INCLUDEDIR --lib-dirs=LIBDIR install



After the installation is complete, you should
be able to run ``import dysmalpy`` within IPython or your Jupyter notebook.
