.. _install:
.. highlight:: shell

============
Installation
============

Dependencies
------------

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

From Source
-----------

``dysmalpy`` can also be installed from source. You can download the
package `here`_.

.. _here: releases/dysmalpy-1.0.tar.gz

From a terminal, change directories to where the package was downloaded
and type:

.. code-block:: console

    $ tar zxvf dysmalpy-N.N.tar.gz
    $ cd dysmalpy-N.N
    $ python setup.py install

