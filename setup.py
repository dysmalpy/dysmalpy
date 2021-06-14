#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

try:
    from setuptools import setup
except:
    from distutils.core import setup

from Cython.Build import cythonize

dir_path = os.path.dirname(os.path.realpath(__file__))

init_string = open(os.path.join(dir_path, 'dysmalpy', '__init__.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, init_string, re.M)
__version__ = mo.group(1)


with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ['numpy', 'scipy', 'matplotlib', 'astropy',
                'emcee', 'corner', 'cython', 'dill',
                'shapely', 'spectral-cube', 'radio-beam',
                'h5py', 'pandas', 'six', 'tkinter']

setup_requirements = ['Cython', 'numpy']

setup(
    author="Taro Shimizu & Sedona Price",
    author_email='shimizu@mpe.mpg.de',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: 3-clause BSD',
        'Natural Language :: English',
        "Topic :: Scientific/Engineering",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="A modelling and fitting package for galaxy kinematics.",
    install_requires=requirements,
    setup_requires=setup_requirements,
    license="3-clause BSD",
    long_description=readme,
    include_package_data=True,
    name='dysmalpy',
    packages=['dysmalpy', 'dysmalpy.extern', 'dysmalpy.fitting_wrappers'],
    package_data={'dysmalpy': ['data/noordermeer/*.save']},
    version=__version__,
    ext_modules=cythonize("dysmalpy/cutils.pyx")
)
