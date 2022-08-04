#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

import logging

try:
    from setuptools import setup, Extension
except:
    from distutils.core import setup, Extension

from Cython.Build import cythonize

dir_path = os.path.dirname(os.path.realpath(__file__))

init_string = open(os.path.join(dir_path, 'dysmalpy', '__init__.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, init_string, re.M)
__version__ = mo.group(1)


logging.basicConfig()
log = logging.getLogger(__file__)


with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ['numpy', 'scipy', 'matplotlib', 'astropy',
                'emcee', 'corner', 'cython', 'dill',
                'shapely', 'photutils', 'spectral-cube', 'radio-beam',
                'h5py', 'pandas', 'six']

setup_requirements = ['Cython', 'numpy']


setup_args = {'name': 'dysmalpy',
        'author': "Taro Shimizu & Sedona Price",
        'author_email': 'shimizu@mpe.mpg.de',
        'python_requires': '>=3.7',
        'classifiers': [
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: 3-clause BSD',
            'Natural Language :: English',
            "Topic :: Scientific/Engineering",
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
        ],
        'description': "A modelling and fitting package for galaxy kinematics.",
        'install_requires': requirements,
        'setup_requires': setup_requirements,
        'license': "3-clause BSD",
        'long_description': readme,
        'include_package_data': True,
        'packages': ['dysmalpy', 'dysmalpy.extern', 'dysmalpy.models',
                     'dysmalpy.fitting', 'dysmalpy.fitting_wrappers'],
        'package_data': {'dysmalpy': ['data/noordermeer/*.save', 'data/deprojected_sersic_models_tables/*.fits']},
        'version': __version__ }


# Add CONDA include and lib paths if necessary
conda_include_path = "."
conda_lib_path = "."
if 'CONDA_PREFIX' in os.environ:
    conda_include_path = os.path.join(os.getenv('CONDA_PREFIX'), 'include')
    conda_lib_path = os.path.join(os.getenv('CONDA_PREFIX'), 'lib')
    log.debug('conda_include_path: {!r}'.format(conda_include_path))
    log.debug('conda_lib_path: {!r}'.format(conda_lib_path))


# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")


ext_modules = cythonize([Extension("dysmalpy.models.cutils",
                                   sources=["dysmalpy/models/cutils.pyx"],
                                   include_dirs=[conda_include_path, "/usr/include", "/usr/local/include", "/opt/local/include"],
                                   library_dirs=[conda_lib_path, "/usr/lib", "/usr/lib/x86_64-linux-gnu", "/usr/local/lib", "/opt/local/lib"],
                        )])
ext_modules_optional = cythonize([
                Extension("dysmalpy.models.cutils",
                          sources=["dysmalpy/models/cutils.pyx"],
                          include_dirs=[conda_include_path, "/usr/include", "/usr/local/include", "/opt/local/include"],
                           library_dirs=[conda_lib_path, "/usr/lib", "/usr/lib/x86_64-linux-gnu", "/usr/local/lib", "/opt/local/lib"],
                          ),
                Extension("dysmalpy.lensingTransformer",
                    sources=["dysmalpy/lensing_transformer/lensingTransformer.cpp"],
                    language="c++",
                    include_dirs=[conda_include_path, "lensing_transformer", "/usr/include", "/usr/local/include", "/opt/local/include"],
                    libraries=['gsl', 'gslcblas', 'cfitsio'],
                    library_dirs=[conda_lib_path, "/usr/lib", "/usr/lib/x86_64-linux-gnu", "/usr/local/lib", "/opt/local/lib"],
                    depends=["dysmalpy/lensing_transformer/lensingTransformer.hpp"],
                    extra_compile_args=['-std=c++11'], optional=True
                ),
                Extension("dysmalpy.leastChiSquares1D",
                    sources=["dysmalpy/utils_least_chi_squares_1d_fitter/leastChiSquares1D.cpp"],
                    language="c++",
                    include_dirs=[conda_include_path, "utils_least_chi_squares_1d_fitter", "/usr/include", "/usr/local/include", "/opt/local/include"],
                    libraries=['gsl', 'gslcblas', 'pthread'],
                    library_dirs=[conda_lib_path, "/usr/lib", "/usr/lib/x86_64-linux-gnu", "/usr/local/lib", "/opt/local/lib"],
                    depends=["dysmalpy/utils_least_chi_squares_1d_fitter/leastChiSquares1D.hpp",
                            "dysmalpy/utils_least_chi_squares_1d_fitter/leastChiSquaresFunctions1D.hpp"],
                    extra_compile_args=['-std=c++11'], optional=True
                )
             ])


try:
    # try building with optional modules :
    setup( **setup_args, ext_modules=ext_modules_optional )

    log.info("Installation with optional modules successful!")

except:
    log.warning("The optional modules could not be compiled")

    # If this new 'setup' call don't fail, the module
    # will be successfully installed, without the optional modules:
    setup( **setup_args, ext_modules=ext_modules )


    log.info("Installation without optional modules successful!")
