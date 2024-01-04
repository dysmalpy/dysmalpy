.. _install:
.. highlight:: shell

============
Installation
============

You can install ``dysmalpy`` in a number of ways:

1. If you have limited experience with Python, we recommend installing ``dysmalpy`` using the Anaconda Python distribution.

    1.1 Using pip to install the dependencies (default)

    1.2 Using conda to install the dependencies

2. If you are more familiar with Python, you can install ``dysmalpy`` using the development version. 

3. If you are a member of the MPE group, you can install ``dysmalpy`` using the provided ``.bat`` file or by using the Anaconda environment we have setup.

.. _install_with_anaconda:

1. Using Anaconda
-----------------

To install Anaconda and all the relevant packages and dependencies, please follow the instructions at :ref:`Python Environment Setup<conda_install>`.

After your anaconda installation is complete. You can download the latest ``dysmalpy`` package here: `tar.gz`_ | `zip`_ 
(current version: |release|).

.. _tar.gz: https://github.com/dysmalpy/dysmalpy/archive/refs/tags/v|release|.tar.gz

.. _zip: https://github.com/dysmalpy/dysmalpy/archive/refs/tags/v|release|.zip

From a terminal, change directories to where the package was downloaded

To install Dysmalpy run:

(Where N.N.N is the current version)

.. code-block::

        tar zxvf dysmalpy-N.N.N.tar.gz
        cd dysmalpy-N.N.N

.. note::
    You can also clone the repository from GitHub with (make sure you have git installed with e.g. ``conda install git``):

    .. code-block::

        git clone https://github.com/dysmalpy/dysmalpy.git

**1.1 Using pip to install the dependencies**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use pip to install the package and dependencies:

.. code-block::

        python3 -m pip install . && python3 setup.py check_build


Alternatively, you can use the following command:

.. code-block::

        python3 setup.py install --single-version-externally-managed --root=/ && python3 setup.py check_build


By default, the first part of the command will try to install dysmalpy with the additional C++ extensions that you may 
have installed :ref:`here<conda_optional_install>`. These are not necessary but it is adviced for you to have them. 
The second part of the command will check if the extensions were compiled succesfully, you will see in the terminal 
which extensions were compiled and which were not.


**1.2 Using conda to install the dependencies**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have chosen to install the python dependencies with conda, please use the following command: 

.. code-block::

        python3 -m pip install . --no-deps && python3 setup.py check_build

.. note:: 
    The ``--no-deps`` flag will prevent pip from installing the dependencies that you have already installed with conda in (:ref:`Python Environment Setup<conda_install>`).

.. _install_clone:

2. Development version
----------------------

To get the most up-to-date development version of ``dysmalpy``, clone the repository from GitHub.

Within your desired parent directory, clone the repository from GitHub:

.. code-block::

    git clone https://github.com/dysmalpy/dysmalpy.git 


Then add the path to this repository to your python path (e.g., 
`PATH/TO/PARENT/DIRECTORY/dysmalpy` to `$PYTHONPATH` as defined in 
.bashrc or .bash_profile for bash, or the equivalent for your shell). 


For examples on using `git fetch` or `git pull` to get updates, 
or how to check out other branches, please see e.g. the tutorial here: 
`https://git-scm.com/docs/gittutorial`_

.. warning::
    This way of installing ``dysmalpy`` will give you a basic installation with most of the functionality, but your ``dysmalpy`` installation will not contain the modules that need to be compiled.

.. _https://git-scm.com/docs/gittutorial: https://git-scm.com/docs/gittutorial


After the installation is complete, you should
be able to run ``import dysmalpy`` within IPython or your Jupyter notebook.


.. tip::
    Especially if working with the development version of ``dysmalpy``, you can 
    confirm the location of the package that is imported by checking 
    the output of 
    
    .. code-block::

        import dysmalpy
        print (dysmalpy.__file__)




--------------------------------------------------------------------



.. 2. Development version
.. ----------------------

.. You will need to setup Python 3 on your machine and install all of the dependent packages. Please
.. follow the instructions in `Python Environment Setup <installation-anaconda>`_ 
.. (it is strongly adviced that you follow those instructions before running the commands here).


.. After this is completed, you can download the latest DysmalPy package here: `tar.gz`_ | `zip`_ 
.. (current version: |release|).

.. .. _tar.gz: https://github.com/ttshimiz/dysmalpy/archive/refs/tags/v|release|.tar.gz

.. .. _zip: https://github.com/ttshimiz/dysmalpy/archive/refs/tags/v|release|.zip

.. Default installation
.. ^^^^^^^^^^^^^^^^^^^^^^

.. From a terminal, change directories to where the package was downloaded

.. To install Dysmalpy run:

.. (Where N.N.N is the current version)

.. .. code-block:: console

..     $ tar zxvf dysmalpy-N.N.N.tar.gz
..     $ cd dysmalpy-N.N.N
..     $ # You can use pip to install the package:
..     $ python -m pip install .
..     $ # Alternatively, you can use the following command:
..     $ python setup.py install --single-version-externally-managed --root=/


.. By default, this will try to install dysmalpy with the optional C++ extensions that you may 
.. have installed `here <installation-anaconda>`_. If setup.py is not able to find those extensions dysmalpy will be installed 
.. with its basic functionality. 


.. Basic Installation
.. ^^^^^^^^^^^^^^^^^^

.. From a terminal, change directories to where the package was downloaded.

.. To install the basic DysmalPy functionality (without any of the C++ extensions) from the command line, 
.. run:

.. .. 
..     (where N.N.N is the current version):

..     $ tar zxvf dysmalpy-N.N.N.tar.gz
..     $ cd dysmalpy-N.N.N
..     $ python setup.py install


.. .. code-block:: console

..     $ tar zxvf dysmalpy-|release|.tar.gz
..     $ cd dysmalpy-|release|
..     $ # You can use pip to install the package:
..     $ python -m pip install .
..     $ # Alternatively, you can use the following command:
..     $ python setup.py install --single-version-externally-managed --root=/


.. Installation with extensions
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. In order to install DysmalPy with the C++ extensions, we will need to also
.. build the extensions.

.. If the `gsl` and `cfitsio` are installed in non-standard locations
.. (e.g., if they were installed using conda during the dependency setups),
.. then we will need specify those directories as below.

.. Typically, if `BASEDIR` is the relevant absolute directory path (e.g., `/PATH/TO/ANACONDA`
.. if installed with conda, as explained in the :ref:`dependencies setup<install_deps>`),
.. then `LIBDIR` and `INCLUDEDIR` are `BASEDIR/lib` and `BASEDIR/include`, respectively.

.. (If they are installed in so the headers are in `/usr/include` or `/usr/local/include`
.. and the libraries are in `/usr/lib` or `/usr/local/lib`,
.. the `--include-dirs` and `--library_dirs` flags can be omitted.)


.. From a terminal, change directories to where the package was downloaded,
.. then install the package and build the extensions by running:

.. .. code-block:: console

..     $ tar zxvf dysmalpy-|release|.tar.gz
..     $ cd dysmalpy-|release|
..     $ python setup.py build_ext --include-dirs=INCLUDEDIR --library_dirs=LIBDIR install --single-version-externally-managed --root=/




.. _install_mpe:

3. MPE group installations
----------------------------


.. _install_windows:

Windows `.bat` File
~~~~~~~~~~~~~~~~~~~

A `.bat` file, for running DysmalPy with a parameters file (e.g., ``fitting.params``) 
is available for MPE-group specific architecture. 

Prior to using DysmalPy with this `.bat` file, the DysmalPy source code
and dependencies will need to be installed (see :ref:`Using Anaconda<conda_install>`).


.. _install_afs:

AFS Machine
~~~~~~~~~~~

If you are on an AFS machine, ``dysmalpy`` is located at
`/afs/mpe.mpg.de/astrosoft/dysmalpy`. We have further setup
an Anaconda environment the contains all of the necessary
Python packages to run ``dysmalpy``. To activate this environment
as well as set environment variables, run this command in your
terminal:

.. code-block::

    source /afs/mpe/astrosoft/dysmalpy/dysmalpy_setup.sh

To check whether the setup ran successfully run:

.. code-block::

    which python

This should return `/afs/mpe.mpg.de/astrosoft/dysmalpy/anaconda3/bin/python`.
Keep in mind that using this environment will override any environment
you have setup locally and only Python packages installed in the
``dysmalpy`` environment will be available. If there is a package you
would like installed in the environment, please contact `Taro`_.

.. _Taro: shimizu@mpe.mpg.de

For those who are more familiar with Python, you can simply add
`/afs/mpe.mpg.de/astrosoft/dysmalpy/dysmalpy/` to your ``PYTHONPATH``
environment variable. Be sure to have all of the required packages
installed.