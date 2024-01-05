
"""
    This is a simple script to check if the C++ libraries were built succesfully during installation from pypi.
    
    It is meant to be run from the command line as:
        python3 -m dysmalpy.check_build
        
    Created: 05/01/2024 by Juan Espejo
"""

import os
import logging
import dysmalpy
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # here we do not setLevel so that it inherits its caller logging level.
if '__main__' in logging.Logger.manager.loggerDict: # pragma: no cover
    logger.setLevel(logging.getLogger('__main__').level)
from ctypes import cdll
from distutils.sysconfig import get_config_var

print("\nChecking if the C++ libraries were built succesfully during installation.")

dysmalpy_path = os.path.abspath(os.path.dirname(dysmalpy.__file__))    
print("Dysmalpy installation path: {}\n".format(dysmalpy_path))

#########
# cutils
#########

try:
    mylib = cdll.LoadLibrary(dysmalpy_path+os.sep+"models"+os.sep+"cutils"+get_config_var('EXT_SUFFIX'))
    print("The cutils C++ library was compiled succesfully.\n\
# Compiled file: {}\n".format(mylib))
except OSError as e:
    #print("The cutils C++ library was not compiled succesfully")
    logger.error("The cutils C++ library was not compiled succesfully")
    raise e

######################
# lensing transformer
######################

try:
    mylib = cdll.LoadLibrary(dysmalpy_path+os.sep+"lensingTransformer"+get_config_var('EXT_SUFFIX'))
    print("The lensingTransformer C++ library was compiled succesfully.\n\
# Compiled file: {}\n".format(mylib))
except OSError as e:
    logger.error("The lensingTransformer C++ library was not compiled succesfully")
    raise e

####################
# least chi-squares 
####################

try:
    mylib = cdll.LoadLibrary(dysmalpy_path+os.sep+"leastChiSquares1D"+get_config_var('EXT_SUFFIX'))
    print("The lensingTransformer C++ library was compiled succesfully.\n\
# Compiled file: {}\n".format(mylib))
except OSError as e:
    logger.error("The leastChiSquares1D C++ library was not compiled succesfully")
    raise e