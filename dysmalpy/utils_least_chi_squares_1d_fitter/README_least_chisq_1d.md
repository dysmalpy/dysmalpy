### About

This is a C++ library to efficiently do 1D Gaussian spectral line fitting for all pixels in a data cube.

The library has four source code files:

- `leastChiSquares1D.cpp`
- `leastChiSquares1D.hpp`
- `leastChiSquaresFunctions1D.hpp`

It has dependencies on the following third-party libraries:

- `gsl`

This library will be used by the `dysmalpy.galaxy` module, i.e., the `"galaxy.py"` code, especially replacing `gaus_fit_sp_opt_leastsq`. You may find the usage of functions in this library there. In principle the functions in this library are not to be used elsewhere, because the input arguments are rather complex.


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
clang++ --std=c++11 --shared -Wall \
        -I/usr/local/include \
        -L/usr/local/lib -lgsl -lgslcblas -lpthread \
        -o libLeastChiSquares1D.so \
        leastChiSquares1D.cpp
```


On Linux OS:

```
g++ -std=c++11 -pthread -shared -fPIC \
        -I/usr/include \
        -Wl,--no-as-needed \
        -L/usr/lib/x86_64-linux-gnu -L/usr/lib -lgsl -lgslcblas -lpthread \
        -o libLeastChiSquares1D.so \
        leastChiSquares1D.cpp
```


### Compiling the Testing Program Manually

Here we also have a testing main program in C++, `main.cpp`, and in Python, `main.py`, to show how the functions are called.

To compile the `main.cpp` into an excutable on a Mac OS:

```
clang++ --std=c++11 -Wall \
        -I/usr/local/include \
        -L/usr/local/lib -lgsl -lgslcblas -lpthread \
        -o main.exe \
        leastChiSquares1D.cpp \
        main.cpp
```


### Update History

Last update: 2021-08-05, Daizhong Liu, MPE Garching.
