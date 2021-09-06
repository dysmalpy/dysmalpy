/*
    This is a testing main program to call the leastChiSquares1D functions.
    
    Last updates: 
        2021-08-04, first version, Daizhong Liu, MPE.  
    
 */

//#include <Python.h> // This implies inclusion of the following standard headers: <stdio.h>, <string.h>, <errno.h>, <limits.h>, <assert.h> and <stdlib.h> (if available). -- https://docs.python.org/2/c-api/intro.html
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "leastChiSquares1D.hpp"
//#include "leastChiSquaresGaussian1D.hpp"



int main(int argc, char **argv)
{
    //Py_Initialize();
    
    std::cout << "Hello" << std::endl;
    
    //PyRun_SimpleString("print('Hello from PyRun')\n");
    
    
    std::cout << "--------------------------------------------------------------" << std::endl;
    std::cout << "Generating a 100-element data array and 3-element params array" << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
    double x[100]; 
    double y[100]; 
    double yerr[100];
    size_t ndata = 100;
    double initparams[3] = {1.5, 0.30, 0.125}; 
    size_t nparams = 3; 
    double outparams[3] = {0.0, 0.0, 0.0};
    double outparamerrs[3] = {0.0, 0.0, 0.0};
    double outyfitted[100]; 
    double outyresidual[100];
    double outchisq = std::nan("");
    size_t i, j, k, q;
    
    
    const double A = 25.0;  /* amplitude */
    const double mu = 0.45;  /* center */
    const double sigma = 0.15; /* width */
    const gsl_rng_type *T = gsl_rng_default;
    gsl_rng *r; // 

    gsl_rng_env_setup();
    r = gsl_rng_alloc(T);

    /* generate synthetic data with noise */
    for (i=0; i < ndata; i++) {
        double xi = (double)i / (double)ndata; // 0.0 to 0.999
        double y0 = A * std::exp( - (xi - mu) * (xi - mu) / ( 2.0 * sigma * sigma ));
        double dy = gsl_ran_gaussian(r, 0.25); // random noise
        x[i] = xi;
        y[i] = y0 + dy;
        yerr[i] = 0.25;
    }
    
    /* print true values */
    std::cout << "input: ";
    std::cout << A << " ";
    std::cout << mu << " ";
    std::cout << sigma;
    std::cout << std::endl;
    
    
    
    
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Fitting with LeastChiSquares1D::runFitting()" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    
    //LeastChiSquaresGaussian1D *MyLeastSquareFitter = new LeastChiSquaresGaussian1D(); 
    LeastChiSquares1D *MyLeastSquareFitter = new LeastChiSquares1D();
    
    /* print initial values */
    std::cout << "init: ";
    for (i=0; i<3; i++) {
        if (i>0) { std::cout << " "; }
        std::cout << initparams[i];
    }
    std::cout << std::endl;
    
    /* run the fitting */
    MyLeastSquareFitter->runFitting(x, y, yerr, ndata, initparams, nparams, outparams, outparamerrs, outyfitted, outyresidual, &outchisq);
    
    /* clea up */
    delete MyLeastSquareFitter;
    
    
    
    
    
    
    
    
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Generating a 137x137x100 data cube and 137x137x3 params cube" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    /* Prepare a data cube */
    long nx = 137, ny = 137, nchan = 100, maxniter = 1000, verbose = 2;
    verbose = 0;
    double *data = (double *)malloc(nchan*ny*nx*sizeof(double));
    double *dataerr = (double *)malloc(nchan*ny*nx*sizeof(double));
    double *initparamsall = (double *)malloc(nparams*ny*nx*sizeof(double));
    double *outall = NULL;
    int ncheckpoints = 10;
    
    /* prepare timeval */
    struct timeval tv0, tv1;
    
    /* generate synthetic data with noise */
    for (k=0; k < nchan; k++) {
        for (j=0; j < ny; j++) {
            for (i=0; i < nx; i++) {
                data[k*ny*nx + j*nx + i] = y[k];
                dataerr[k*ny*nx + j*nx + i] = yerr[k];
            }
        }
    }
    
    std::cout << "init params: ";
    for (k=0; k < nparams; k++) {
        if (k > 0) { std::cout << ", "; }
        std::cout << initparams[k]; 
    }
    std::cout << std::endl;
    for (j=0; j < ny; j++) {
        for (i=0; i < nx; i++) {
            if ( (j*nx + i) % ((size_t)(nx*ny/ncheckpoints)) == 0 ) {
                std::cout << "init params at pixel (x,y) = (" << i << "," << j << "): ";
            }
            for (k=0; k < nparams; k++) {
                initparamsall[k*ny*nx + j*nx + i] = initparams[k];
                
                if (k == 0) { initparamsall[k*ny*nx + j*nx + i] *= (0.8 + 0.4 * (((double)(i)/(double)(nx-1))-0.5)); } // scale x0.8 to x1.2
                else if (k == 1) { initparamsall[k*ny*nx + j*nx + i] += (0.2 * (((double)(i)/(double)(nx-1))-0.5)); } // scale +-0.1
                else if (k == 2) { initparamsall[k*ny*nx + j*nx + i] += (0.2 * (((double)(i)/(double)(nx-1))-0.5)); } // scale +-0.1
                
                if ( (j*nx + i) % ((size_t)(nx*ny/ncheckpoints)) == 0 ) {
                    if (k > 0) { std::cout << ", "; }
                    std::cout << initparamsall[k*ny*nx + j*nx + i];
                }
            }
            if ( (j*nx + i) % ((size_t)(nx*ny/ncheckpoints)) == 0 ) {
                std::cout << std::endl;
            }
        }
    }
    
    
    
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Fitting with fitLeastChiSquares1D" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    
    std::cout << "fitting ..." << std::endl;
    
    gettimeofday(&tv0, NULL);
    
    //LeastChiSquaresGaussian1D *my_least_chisq_fitter = (LeastChiSquaresGaussian1D *)createLeastChiSquares1D();
    LeastChiSquares1D *my_least_chisq_fitter = (LeastChiSquares1D *)createLeastChiSquares1D();
    
    outall = fitLeastChiSquares1D(my_least_chisq_fitter, x, y, yerr, ndata, initparams, nparams, maxniter, verbose);
    gsl_vector_view outallVectorView = gsl_vector_view_array(outall, (size_t)(nparams*2 + ndata*2 + 1));
    gsl_vector *outallVector = &(outallVectorView.vector); 
    
    gettimeofday(&tv1, NULL);
    
    std::cout << "fitting used " << (tv1.tv_sec - tv0.tv_sec) + 1.0e-6 * (tv1.tv_usec - tv0.tv_usec) << "seconds" << std::endl;
    
    std::cout << "best-fit: ";
    {   
        gsl_vector_view outparams_view = gsl_vector_subvector(outallVector, 0, nparams);
        gsl_vector_view outparamerrs_view = gsl_vector_subvector(outallVector, nparams, nparams);
        //gsl_vector_view outyfitted_view = gsl_vector_subvector(outallVector, nparams*2, ndata);
        //gsl_vector_view outyresidual_view = gsl_vector_subvector(outallVector, nparams*2+ndata, ndata);
        gsl_vector_view outchisq_view = gsl_vector_subvector(outallVector, nparams*2+ndata*2, 1);
        for (q=0; q < nparams; q++) {
            if (q > 0) { std::cout << ", "; }
            std::cout << gsl_vector_get(&outparams_view.vector, q) << " +- " << gsl_vector_get(&outparamerrs_view.vector, q);
        }
        std::cout << ", ";
        std::cout << "chisq: " << gsl_vector_get(&outchisq_view.vector, 0);
        std::cout << ", ";
        std::cout << "rchisq: " << gsl_vector_get(&outchisq_view.vector, 0) / (double)(ndata - nparams);
        std::cout << std::endl;
    }
    
    delete outall; outall=NULL;
    
    
    
    
    
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Fitting with fitLeastChiSquares1DForDataCube" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    
    std::cout << "fitting ..." << std::endl;
    
    gettimeofday(&tv0, NULL);
    
    outall = fitLeastChiSquares1DForDataCube(my_least_chisq_fitter, x, data, dataerr, nx, ny, nchan, initparamsall, nparams, maxniter, verbose);
    gsl_matrix_view outallMatrixView = gsl_matrix_view_array(outall, (size_t)(nparams*2 + nchan*2 + 1), (size_t)(ny*nx)); // nrows (slower-changing), ncols (faster-changing)
    gsl_matrix *outallMatrix = &(outallMatrixView.matrix); 
    
    gettimeofday(&tv1, NULL);
    
    std::cout << "fitting used " << (tv1.tv_sec - tv0.tv_sec) + 1.0e-6 * (tv1.tv_usec - tv0.tv_usec) << "seconds" << std::endl;
    
    for (k=0; k < ny*nx; k += ((size_t)(nx*ny/ncheckpoints))) {
        i = k % (nx);
        j = (size_t) (k / (nx));
        gsl_vector_view outparams_view = gsl_matrix_subcolumn(outallMatrix, k, 0, nparams);
        gsl_vector_view outparamerrs_view = gsl_matrix_subcolumn(outallMatrix, k, nparams, nparams);
        //gsl_vector_view outyfitted_view = gsl_matrix_subcolumn(outallMatrix, k, nparams*2, nchan);
        //gsl_vector_view outyresidual_view = gsl_matrix_subcolumn(outallMatrix, k, nparams*2+nchan, nchan);
        gsl_vector_view outchisq_view = gsl_matrix_subcolumn(outallMatrix, k, nparams*2+nchan*2, 1);
        std::cout << "best-fit at pixel (x,y) = (" << i << "," << j << "): ";
        for (q=0; q < nparams; q++) {
            if (q > 0) { std::cout << ", "; }
            std::cout << gsl_vector_get(&outparams_view.vector, q) << " +- " << gsl_vector_get(&outparamerrs_view.vector, q);
        }
        std::cout << ", ";
        std::cout << "chisq: " << gsl_vector_get(&outchisq_view.vector, 0);
        std::cout << ", ";
        std::cout << "rchisq: " << gsl_vector_get(&outchisq_view.vector, 0) / (double)(nchan - nparams);
        std::cout << std::endl;
    }
    
    destroyLeastChiSquares1D(my_least_chisq_fitter);
    
    delete outall; outall=NULL;
    
    
    
    
    
    
    
    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << "Fitting with fitLeastChiSquares1DForDataCubeWithMultiThread" << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;
    
    std::cout << "fitting ..." << std::endl;
    
    gettimeofday(&tv0, NULL);
    
    verbose = 1;
    setGlobalDebugLevel(1);
    outall = fitLeastChiSquares1DForDataCubeWithMultiThread(x, data, dataerr, nx, ny, nchan, initparamsall, nparams, maxniter, verbose);
    gsl_matrix_view outallMatrixView2 = gsl_matrix_view_array(outall, (size_t)(nparams*2 + nchan*2 + 1), (size_t)(ny*nx)); // nrows (slower-changing), ncols (faster-changing)
    outallMatrix = &(outallMatrixView2.matrix); 
    
    gettimeofday(&tv1, NULL);
    
    std::cout << "fitting used " << (tv1.tv_sec - tv0.tv_sec) + 1.0e-6 * (tv1.tv_usec - tv0.tv_usec) << "seconds" << std::endl;
    
    for (k=0; k < ny*nx; k += ((size_t)(nx*ny/ncheckpoints))) {
        i = k % (nx);
        j = (size_t) (k / (nx));
        gsl_vector_view outparams_view = gsl_matrix_subcolumn(outallMatrix, k, 0, nparams);
        gsl_vector_view outparamerrs_view = gsl_matrix_subcolumn(outallMatrix, k, nparams, nparams);
        gsl_vector_view outchisq_view = gsl_matrix_subcolumn(outallMatrix, k, nparams*2+nchan*2, 1);
        std::cout << "best-fit at pixel (x,y) = (" << i << "," << j << "): ";
        for (q=0; q < nparams; q++) {
            if (q > 0) { std::cout << ", "; }
            std::cout << gsl_vector_get(&outparams_view.vector, q) << " +- " << gsl_vector_get(&outparamerrs_view.vector, q);
        }
        std::cout << ", ";
        std::cout << "chisq: " << gsl_vector_get(&outchisq_view.vector, 0);
        std::cout << ", ";
        std::cout << "rchisq: " << gsl_vector_get(&outchisq_view.vector, 0) / (double)(nchan - nparams);
        std::cout << std::endl;
    }
    
    
    delete outall; outall=NULL;
    
    
    
    
    std::cout << "-------------------------------------------- ALL DONE" << std::endl;
    
     
    
    delete data; data=NULL;
    delete dataerr; dataerr=NULL;
    delete initparamsall; initparamsall=NULL;
    
    
}


