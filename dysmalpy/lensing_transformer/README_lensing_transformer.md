### About

This is a C++ library to efficiently do lensing transformation for three-dimensional data cubes. It replies on the lensing model mesh grid file, name like "mesh.dat", which is an output of the [Glafic software](https://www.slac.stanford.edu/~oguri/glafic/). By using a pointers' array as the image plane data array, and mapping each pointer to the memory address of source plane data array's elements, the lensing transformation can be done instatenously as long as the pointers' mapping is done once and for all. 

The library has two source code files: 

- `lensingTransformer.cpp` 
- `lensingTransformer.hpp`

It has dependencies on the following third-party libraries:

- `gsl`
- `cfitsio`
- `fftw3` (for some functionalities to be implemented in the future)
- `python`

This library will be used by the `dysmalpy.lensing` module, i.e., the `"lensing.py"` code. You may find the usage of functions in this library there. In principle the functions in this library are not to be used elsewhere, because the input arguments are rather complex. 


### Compiling the Library using Python setup.py

The library can be compiled automatically when you install the DysmalPy package with 

```
python setup.py build_ext --inplace
```

But below you can also find how to compile it by yourself.


### Compiling the Library Manually

On a Mac OS we probably need to compile those third-party libraries by ourselves. Once the third-party libraires are ready, for example, headers are put into a sub-directory `3rd/include/` and libraries are put into `3rd/lib/`, then the dynamic library `*.so` file can be compiled as follows:

```
PYTHON_INC="/usr/local/Cellar/python@3.9/3.9.5/Frameworks/Python.framework/Versions/Current/Headers"
PYTHON_LIB="/usr/local/Cellar/python@3.9/3.9.5/Frameworks/Python.framework/Versions/Current/lib"
PYTHON_DLL=python3.9

clang++ --std=c++11 --shared \
        -I3rd/include \
        -I${PYTHON_INC} \
        -L3rd/lib -lcfitsio -lgsl -lcblas -lfftw3 -lfftw3_threads -lfftw3_mpi -lpthread \
        -L${PYTHON_LIB} -l${PYTHON_DLL} \
        -o libLensingTransformer.so \
        lensingTransformer.cpp
```

Note that the python version, header and library locations in the above compilation command are also user-dependent. 

On a Linux OS, usually these third-party libraries exist. Assuming `"fitsio.h"` and `"gsl/gsl_matrix.h"` can be found in `/usr/include`,  `"Python.h"` can be found in `"/usr/include/python3.6m"`, and `"libcfitsio.so"`, `"libgsl.so"`, `"libgslcblas.so"` and `"libpython3.6m.so*"` in `/usr/lib/x86_64-linux-gnu`, then we can easily compile our dynamic library `*.so` file as follows:

```
PYTHON_INC="/usr/include/python3.6m"
PYTHON_LIB="/usr/lib/x86_64-linux-gnu"
PYTHON_DLL=python3.6m

g++ --std=c++11 -shared -fPIC \
        -I/usr/include \
        -I${PYTHON_INC} \
        -L/usr/lib/x86_64-linux-gnu -L/usr/lib -lcfitsio -lgsl -lgslcblas -lpthread \
        -L${PYTHON_LIB} -l${PYTHON_DLL} \
        -o libLensingTransformer.so \
        lensingTransformer.cpp
```


### Compiling the Testing Program Manually

Here we also have a testing main program in C++, `main.cpp`, and in Python, `main.py`, to show how the functions are called. 

To compile the `main.cpp` into an excutable on a Mac OS:

```
clang++ --std=c++11 \
        -I. -I3rd/include \
        -I${PYTHON_INC} \
        -L. -L3rd/lib -lcfitsio -lgsl -lcblas -lfftw3 -lfftw3_threads -lfftw3_mpi -lpthread \
        -L${PYTHON_LIB} -l${PYTHON_DLL} \
        -lLensingTransformer \
        -o main.exe \
        main.cpp
```

Or on a Linux OS: 

```
g++ --std=c++11 \
        -I. \
        -I/usr/include \
        -I${PYTHON_INC} \
        -o main.exe \
        main.cpp \
        -L. -lLensingTransformer \
        -L/usr/lib/x86_64-linux-gnu -L/usr/lib -lcfitsio -lgsl -lgslcblas -lpthread \
        -L${PYTHON_LIB} -l${PYTHON_DLL}
        # -l must be after main.cpp
```

Again those third-party libraries are needed for the compilation. 

Once you have prepared a `"data/mesh.dat"` and a `"data/model_cube.fits"` file, executing the `main.exe` in the command line will give you "data/image_plane_cube.fits". 

For the python application `main.py`, it does not need compiling. Assuming you already have a `"data/mesh.dat"` and a `"data/model_cube.fits"` file, and the "libLensingTransformer.so". Directly executing `python3 main.py` in the command line will give you the output data cube `"data/image_plane_cube_from_python.fits"`. 


### Update History

Last update: 2021-08-03, Daizhong Liu, MPE Garching. 

