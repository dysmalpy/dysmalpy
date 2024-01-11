
"""
    This is a simple script to check if the C++ libraries were built succesfully during installation from pypi.
    
    It is meant to be run from the command line as:
        python -m dysmalpy.check_build
        
    Created: 05/01/2024 by Juan Espejo
"""

import os, sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # here we do not setLevel so that it inherits its caller logging level.
if '__main__' in logging.Logger.manager.loggerDict: # pragma: no cover
    logger.setLevel(logging.getLogger('__main__').level)
from ctypes import cdll
from distutils.sysconfig import get_config_var
# import site

print("\nChecking if the C++ libraries were built succesfully during installation.")

# # Get the path to the dysmalpy package from the local version
# site_packages = site.getsitepackages()[0]
# sys.path.insert(0, site_packages)
# dysmalpy_path_env = f'{sys.path[0]}/dysmalpy'
# print("Dysmalpy installation path: {}\n".format(dysmalpy_path_env))

# Get the path to the dysmalpy package from the installed version
import dysmalpy
dysmalpy_path_local = os.path.abspath(os.path.dirname(dysmalpy.__file__))
print("Dysmalpy local path: {}".format(dysmalpy_path_local))

#########
# cutils
#########

# Check if the cutils C++ library was compiled succesfully in your local directory
try:
    mylib = cdll.LoadLibrary(dysmalpy_path_local+os.sep+"models"+os.sep+"cutils"+get_config_var('EXT_SUFFIX'))
    print("The cutils C++ library was compiled succesfully in your local directory.\n\
# Compiled file: {}\n".format(mylib))
except OSError as e:
    #print("The cutils C++ library was not compiled succesfully")
    logger.error("The cutils C++ library was not compiled succesfully in your local directory")
    raise e

# # Check if the cutils C++ library was compiled succesfully in your envirnoment
# try:
#     mylib = cdll.LoadLibrary(dysmalpy_path_env+os.sep+"models"+os.sep+"cutils"+get_config_var('EXT_SUFFIX'))
#     print("The cutils C++ library was compiled succesfully in your envirnoment.\n\
# # Compiled file: {}\n".format(mylib))
# except OSError as e:
#     #print("The cutils C++ library was not compiled succesfully")
#     logger.error("The cutils C++ library was not compiled succesfully in your envirnoment.")
#     raise e

######################
# lensing transformer
######################

# Check if the lensingTransformer C++ library was compiled succesfully in your local directory
try:
    mylib = cdll.LoadLibrary(dysmalpy_path_local+os.sep+"lensingTransformer"+get_config_var('EXT_SUFFIX'))
    print("The lensingTransformer C++ library was compiled succesfully in your local directory.\n\
# Compiled file: {}\n".format(mylib))
except OSError as e:
    logger.error("The lensingTransformer C++ library was not compiled succesfully in your local directory.")
    raise e

# # Check if the lensingTransformer C++ library was compiled succesfully in your envirnoment
# try:
#     mylib = cdll.LoadLibrary(dysmalpy_path_env+os.sep+"lensingTransformer"+get_config_var('EXT_SUFFIX'))
#     print("The lensingTransformer C++ library was compiled succesfully in your envirnoment.\n\
# # Compiled file: {}\n".format(mylib))
# except OSError as e:
#     logger.error("The lensingTransformer C++ library was not compiled succesfully in your envirnoment.")
#     raise e

####################
# least chi-squares 
####################

# Check if the leastChiSquares1D C++ library was compiled succesfully in your local directory
try:
    mylib = cdll.LoadLibrary(dysmalpy_path_local+os.sep+"leastChiSquares1D"+get_config_var('EXT_SUFFIX'))
    print("The leastChiSquares1D C++ library was compiled succesfully in your local directory.\n\
# Compiled file: {}\n".format(mylib))
except OSError as e:
    logger.error("The leastChiSquares1D C++ library was not compiled succesfully in your local directory.")
    raise e

# # Check if the leastChiSquares1D C++ library was compiled succesfully in your envirnoment
# try:
#     mylib = cdll.LoadLibrary(dysmalpy_path_env+os.sep+"leastChiSquares1D"+get_config_var('EXT_SUFFIX'))
#     print("The leastChiSquares1D C++ library was compiled succesfully in your envirnoment.\n\
# # Compiled file: {}\n".format(mylib))
# except OSError as e:
#     logger.error("The leastChiSquares1D C++ library was not compiled succesfully in your envirnoment.")
#     raise e
