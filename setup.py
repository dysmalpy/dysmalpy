#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports #
import os
import re
import site
import copy
import tempfile
import shutil

from setuptools import setup, Extension, find_packages, Command
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

import logging

# Find version #
dir_path = os.path.dirname(os.path.realpath(__file__))
init_string = open(os.path.join(dir_path, 'dysmalpy', '__init__.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, init_string, re.M)
__version__ = mo.group(1)


logging.basicConfig()
log = logging.getLogger(__file__)


with open('README.rst') as readme_file:
    readme = readme_file.read()

setup_requirements = ['Cython', 'numpy']
import_list = ["dysmalpy.lensing", "dysmalpy.utils_least_chi_squares_1d_fitter"]
name_list = ["LensingTransformer", "LeastChiSquares1D"]

# Get a list of site-packages directories
site_packages = site.getsitepackages()

# Search for the installation path
install_path = None
for site_package in site_packages:
    potential_path = os.path.abspath(site_package)
    if os.path.exists(potential_path):
        install_path = potential_path
        print(f"Installation path: {install_path}")
        break

# Initialize the directory paths:
include_dirs=["/usr/include", "/usr/include/x86_64-linux-gnu", "/usr/include/aarch64-linux-gnu", "/usr/local/include", "/opt/local/include"]
library_dirs=["/usr/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/local/lib", "/opt/local/lib"]

# Add CONDA include and lib paths if necessary
conda_include_path = "."
conda_lib_path = "."
if 'CONDA_PREFIX' in os.environ:
    conda_include_path = os.path.join(os.getenv('CONDA_PREFIX'), 'include')
    conda_lib_path = os.path.join(os.getenv('CONDA_PREFIX'), 'lib')
    log.debug('conda_include_path: {!r}'.format(conda_include_path))
    log.debug('conda_lib_path: {!r}'.format(conda_lib_path))
    include_dirs.append(conda_include_path)
    library_dirs.append(conda_lib_path)
    
c_include_path = "."
if 'C_INCLUDE_PATH' in os.environ:
    c_include_path = os.getenv('C_INCLUDE_PATH').split(":")
    include_dirs.extend(c_include_path)

cplus_include_path = "."
if 'CPLUS_INCLUDE_PATH' in os.environ:
    cplus_include_path = os.getenv('CPLUS_INCLUDE_PATH').split(":")
    include_dirs.extend(cplus_include_path)

ld_library_path = "."
if 'LD_LIBRARY_PATH' in os.environ:
    ld_library_path = os.getenv('LD_LIBRARY_PATH').split(":")
    library_dirs.extend(ld_library_path)


class CheckBuildCommand(Command):
    """Custom command to check for compiled .so files after build."""
    description = 'Check for compiled .so files'
    user_options = []

    def initialize_options(self):
        self.build_lib = None

    def finalize_options(self):
        self.set_undefined_options('build_ext', ('build_lib', 'build_lib'))

    def run(self):
        if not getattr(self, '_has_run', False):
            setattr(self, '_has_run', True)

            print("\nLet's check if the advanced C++ extensions can be compiled successfully.\n")
            
            successes = []
            failures = []

            # Loop to build each extension separately
            for i, ext_module in enumerate(original_ext_modules):
                # Create a temporary directory for the current extension
                temp_dir = tempfile.mkdtemp()

                # Create a deep copy of the original setup_args
                setup_args_copy = copy.deepcopy(setup_args)

                # Add the current extension to ext_modules dynamically
                setup_args_copy['ext_modules'] = [ext_module]
                # Make sure the extension is not optional for this check
                ext_module.optional = False

                # Set the build directory to the temporary directory
                setup_args_copy['script_args'] = ['build_ext', '-b', temp_dir]

                # Print information about the extension being compiled
                print(f"\nCompiling extension: {ext_module.name}")
                
                try:
                    # Run the setup function
                    result = setup(**setup_args_copy)
                    print(f"[✓] {ext_module.name} compiled successfully.")
                    successes.append(ext_module.name)
                    
                except SystemExit as e:
                    print(f"[x] Compilation failed for {ext_module.name}. Error: {str(e)}")
                    failures.append(ext_module.name)
                    pass
                except Exception as e:
                    print(f"[x] An unexpected error occurred during compilation of {ext_module.name}. Error: {str(e)}")
                    failures.append(ext_module.name)
                    pass

                finally:
                    # Clean up the temporary directory
                    shutil.rmtree(temp_dir)
                
            # Print a summary of the results
            print("\nSummary:")
            print(f"Advanced C++ extensions compiled successfully: {successes}")
            print(f"Advanced C++ extensions that failed to compile: {failures}\n")


# Only the mandatory modules
original_ext_modules = [
        # Basic modules
        Extension("dysmalpy.models.cutils",
                sources=["dysmalpy/models/cutils.pyx"],   
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                ),
        # Lensing transformer
        Extension("dysmalpy.lensingTransformer",
                    sources=["dysmalpy/lensing_transformer/lensingTransformer.cpp"],
                    language="c++",
                    include_dirs=include_dirs+["lensing_transformer"],
                    libraries=['gsl', 'gslcblas', 'cfitsio'],
                    library_dirs=library_dirs,
                    depends=["dysmalpy/lensing_transformer/lensingTransformer.hpp"],
                    extra_compile_args=['-std=c++11'], 
                    optional=True
                ),
        # Chi squared fitter
        Extension("dysmalpy.leastChiSquares1D",
                    sources=["dysmalpy/utils_least_chi_squares_1d_fitter/leastChiSquares1D.cpp"],
                    language="c++",
                    include_dirs=include_dirs+["utils_least_chi_squares_1d_fitter"],
                    libraries=['gsl', 'gslcblas', 'pthread'],
                    library_dirs=library_dirs,
                    depends=["dysmalpy/utils_least_chi_squares_1d_fitter/leastChiSquares1D.hpp",
                            "dysmalpy/utils_least_chi_squares_1d_fitter/leastChiSquaresFunctions1D.hpp"],
                    extra_compile_args=['-std=c++11'], 
                    optional=True
                )
            ]

# Cythonize the extensions (default to the site-packages directory)
ext_modules = cythonize(original_ext_modules, annotate=True)

# Create a modified version of the extension modules to build in the local directory
local_ext_modules = [
    Extension(
        module.name,
        sources=module.sources,
        include_dirs=module.include_dirs,
        library_dirs=module.library_dirs,
        depends=module.depends,
        extra_compile_args=module.extra_compile_args,
        optional=module.optional,
    )
    for module in original_ext_modules
]


# Cythonize the extensions for the local directory
local_ext_modules = cythonize(local_ext_modules, build_dir=".", annotate=True)

# Combine the two sets of extensions
all_ext_modules = ext_modules + local_ext_modules

class BuildExtCommand(build_ext):
    def run(self):
        # Run the original build_ext command with --inplace for the local directory
        self.inplace = True
        build_ext.run(self)


# Setup #
setup_args = {
        'name': 'dysmalpy',
        'author': "MPE IR/Sub-mm Group",
        'author_email': 'dysmalpy@mpe.mpg.de',
        'description': "A modelling and fitting package for galaxy kinematics.",
        'long_description': readme,
        'url': "https://github.com/dysmalpy/dysmalpy",
        'include_package_data': True,
        'packages': find_packages(), 
        'setup_requires': setup_requirements,
        'version': __version__,
        'ext_modules': all_ext_modules,
        'cmdclass': {'check_build': CheckBuildCommand, 'build_ext': BuildExtCommand},
        'package_data': {
            'dysmalpy': [
                'data/deprojected_sersic_models_tables/*.fits',
                'tests/test_data/*', 
                'tests/test_data_lensing/*', 
                'tests/test_data_masking/*',
                'examples/notebooks/*',
                'examples/examples_param_files/*',
                'examples/examples_param_files/model_examples/*',
            ]},
        }
                    

# Perform the setup
result = setup(**setup_args)

