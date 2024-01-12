/*
    This is a library to perform Nonlinear Least-Squares Fitting with the GSL (GNU Scientific Library).
    
    See some related GSL document here: https://www.gnu.org/software/gsl/doc/html/nls.html
    
    fdf source code: https://github.com/ampl/gsl/blob/master/multilarge_nlinear/fdf.c
    
    Last updates: 
        2021-08-04, first version, Daizhong Liu, MPE.  
    
 */
#ifndef LEASTCHISQUARES1D_H
#define LEASTCHISQUARES1D_H

// Detect windows
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(WIN64) || defined(_WIN64) || defined(__WIN64__) || defined(__NT__)
    #define __WINDOWS__
    #define WIN32
    #define GSL_DLL
#endif

//#include <Python.h> // This implies inclusion of the following standard headers: <stdio.h>, <string.h>, <errno.h>, <limits.h>, <assert.h> and <stdlib.h> (if available). -- https://docs.python.org/2/c-api/intro.html
#include <stdio.h>
#ifdef __WINDOWS__
    #include <io.h>
#else
    #include <unistd.h> // for access()
#endif
#include <iomanip> // for std::setw std::setfill
#include <iostream> // for std::cout std::cerr
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector> // for std::vector
#include <cmath> // for std::nan
#include <thread>
#include <mutex>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_multifit_nlinear.h>
#include "leastChiSquaresFunctions1D.hpp"


/*
    LeastChiSquares1D Class.
 */
class LeastChiSquares1D {

public: 
    LeastChiSquares1D();
    virtual ~LeastChiSquares1D();
    
#ifndef leastChiSquaresFunctions1D_hpp
    struct DataStruct
    {
        size_t ndata;
        double *x;
        double *y;
        double *yerr;
    };
    struct ParamsStruct
    {
        std::vector<std::vector<double> > values; // number of parameters for each component
    };
#endif
    
    struct DataStruct Data;
    struct ParamsStruct Params;
    
    struct CallBackPayloadStruct
    {
        int verbose;
    };
    
    //virtual int f(const gsl_vector *variables, void *data, gsl_vector *f);
    //virtual int df(const gsl_vector *variables, void *data, gsl_matrix *J);
    //virtual int fvv(const gsl_vector *variables, const gsl_vector *v, void *data, gsl_vector *fvv);
    // see https://www.gnu.org/software/gsl/doc/html/nls.html#sec-providing-function-minimized
    virtual void setfdf() {};
    
    void setParams(double *p, size_t nparams);
    void setParams(std::vector<double> p);
    void setParams(double *p, std::vector<size_t> nparams);
    void setParams(std::vector<std::vector<double> > p);
    
    double *getParamsFlattened(std::vector<std::vector<double> > p);
    std::vector<double> getParamsFlattenedVector(std::vector<std::vector<double> > p);
    size_t getParamsCount(std::vector<std::vector<double> > p);
    
    void setData(double *x, double *y, double *yerr, size_t ndata);
    
    static void callback(const size_t iter, void *callback_payload, const gsl_multifit_nlinear_workspace *workspace); // needed by gsl
    
    void runFitting(\
        double *x, 
        double *y, 
        double *yerr, 
        size_t ndata, 
        double *initparams, 
        size_t nparams, 
        double *outparams, 
        double *outparamerrs, 
        double *outyfitted, 
        double *outyresidual, 
        double *outchisq, 
        size_t maxniter = 1000, 
        int verbose = 1); // for one component nparams
    
    void runFitting(\
        double *x, 
        double *y, 
        double *yerr, 
        size_t ndata, 
        double *initparams, 
        std::vector<size_t> nparams, 
        double *outparams, 
        double *outparamerrs, 
        double *outyfitted, 
        double *outyresidual, 
        double *outchisq, 
        size_t maxniter = 1000, 
        int verbose = 1); // for more than one component nparams
    
    int errorCode();
    int debugCode();
    int debugLevel();
    void setDebugLevel(int DebugLevel);

protected:
    int ErrorCode;
    int DebugCode;
    
    const gsl_multifit_nlinear_type *T;
    gsl_multifit_nlinear_workspace *Workspace;
    gsl_multifit_nlinear_fdf fdf;
    gsl_multifit_nlinear_parameters fdfparams;
};


// prevent name mangling, using extern "C"
#ifdef __cplusplus 
extern "C" {
#endif

// A global variable, debug flag.
extern int GlobalDebug;

// Function to check endian.
bool isLittleEndian();

// Function to set global debug
void setGlobalDebugLevel(int value);

// Struct for multi-threading
struct MultiThreadPayloadStruct
{
    size_t kfirst; // first index of the subloop for this thread
    size_t klast; // last index of the subloop for this thread (included)
    size_t kstep; // step of the subloop for this thread (default 1)
    double *outptr; // a pointer to the output array
    std::mutex *mu; 
    int dryrun;
};

// Function exposed to Python to create a class.
void *createLeastChiSquares1D();

// Function exposed to Python to run the fitting for one 1D data.
double *fitLeastChiSquares1D(\
    void *ptr, 
    double *x,
    double *y,
    double *yerr,
    long ndata,
    double *initparams,
    long nparams,
    int maxniter = 1000,
    int verbose = 1,
    struct MultiThreadPayloadStruct *mthread = NULL);

// Function exposed to Python to run the fitting for each spaxel's 1D data in a 3D data cube.
double *fitLeastChiSquares1DForDataCube(\
    void *ptr,
    double *x,
    double *data,
    double *dataerr,
    long nx,
    long ny,
    long nchan,
    double *initparamsall,
    long nparams,
    int maxniter = 1000,
    int verbose = 1,
    struct MultiThreadPayloadStruct *mthread = NULL);

// Function exposed to Python to run the fitting of 3D data cube with multithreading
double *fitLeastChiSquares1DForDataCubeWithMultiThread(\
    double *x,
    double *data,
    double *dataerr,
    long nx,
    long ny,
    long nchan,
    double *initparamsall,
    long nparams,
    int maxniter = 1000,
    int verbose = 1,
    int nthread = 4);

// Function exposed to Python to destroy a class.
// No need to use it because the above function already did 
// both new and destroy of the class object. 
void destroyLeastChiSquares1D(void *ptr);

// Function exposed to Python to free the memory
// of the data array returned by the function
// "fitLeastChiSquares1DForDataCubeWithMultiThread"
void freeDataArrayMemory(double *arr);



#ifdef __cplusplus
}
#endif


#endif // LEASTCHISQUARES1D_H
