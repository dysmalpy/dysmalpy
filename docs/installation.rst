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


After this is completed, you can download the latest DysmalPy package `here`_.

.. _here: releases/dysmalpy-1.0.0.tar.gz

From a terminal, change directories to where the package was downloaded
and type:

.. code-block:: console

    $ tar zxvf dysmalpy-N.N.N.tar.gz
    $ cd dysmalpy-N.N.N
    $ python setup.py install

N.N.N is the current version. After the installation is complete, you should
be able to run ``import dysmalpy`` within IPython or your Jupyter notebook.
