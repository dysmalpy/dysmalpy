### About

This is a C++ library to efficiently do lensing transformation for three-dimensional data cubes. It replies on the lensing model mesh grid file, name like "mesh.dat", which is an output of the [Glafic software](https://www.slac.stanford.edu/~oguri/glafic/). By using a pointers' array as the image plane data array, and mapping each pointer to the memory address of source plane data array's elements, the lensing transformation can be done instatenously as long as the pointers' mapping is done once and for all. 

The library has two source code files: 

- `lensingTransformer.cpp` 
- `lensingTransformer.hpp`

It has dependencies on the following third-party libraries:

- `cfitsio`
- `gsl`

This library will be used by the `dysmalpy.lensing` module, i.e., the `"lensing.py"` code. You may find the usage of functions in this library there. In principle the functions in this library are not to be used elsewhere, because the input arguments are rather complex. 


### Compiling the Library using Python setup.py

The library can be compiled automatically when you install the DysmalPy package with 

```
python setup.py build_ext --inplace
```

But below you can also find how to compile it by yourself.


### Compiling the Library Manually

Assuming that the third-party libraires exist in our operation system, for example, headers exist as `/usr/local/include/gsl/gsl*.h` and libraries exist as `/usr/local/lib/libgsl*`, then our dynamic library `*.so` file can be compiled as follows. 

On Mac OS:

```
clang++ --std=c++11 --shared \
        -I/usr/local/include \
        -L/usr/local/lib -lcfitsio -lgsl -lgslcblas \
        -o libLensingTransformer.so \
        lensingTransformer.cpp
```

On Linux OS:

```
g++ --std=c++11 -shared -fPIC \
        -I/usr/include \
        -L/usr/lib/x86_64-linux-gnu -L/usr/lib -lcfitsio -lgsl -lgslcblas -lpthread \
        -o libLensingTransformer.so \
        lensingTransformer.cpp
```


### Compiling the Testing Program Manually

Here we also have a testing main program in C++, `main.cpp`, and in Python, `main.py`, to show how the functions are called. 

To compile the `main.cpp` into an excutable on a Mac OS:

```
clang++ --std=c++11 \
        -I/usr/local/include \
        -L. -L/usr/local/lib -lcfitsio -lgsl -lcblas \
        -lLensingTransformer \
        -o main.exe \
        main.cpp
```

Or on a Linux OS: 

```
g++ --std=c++11 \
        -I. \
        -I/usr/include \
        -o main.exe \
        main.cpp \
        -L. -lLensingTransformer \
        -L/usr/lib/x86_64-linux-gnu -L/usr/lib -lcfitsio -lgsl -lgslcblas
        
        # note that -l must be after main.cpp
```

Again those third-party libraries are needed for the compilation. 

Once you have prepared a `"data/mesh.dat"` and a `"data/model_cube.fits"` file, executing the `main.exe` in the command line will give you "data/image_plane_cube.fits". 

For the python application `main.py`, it does not need compiling. Assuming you already have a `"data/mesh.dat"` and a `"data/model_cube.fits"` file, and the "libLensingTransformer.so". Directly executing `python3 main.py` in the command line will give you the output data cube `"data/image_plane_cube_from_python.fits"`. 


### Update History

Last updates: 

- 2021-08-11, Daizhong Liu, MPE Garching. 
- 2021-08-03, Daizhong Liu, MPE Garching. 

