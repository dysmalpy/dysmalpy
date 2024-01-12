/*
    Static functions for "LeastChiSquares1D". 
    
    Created by Daizhong Liu on 2021-08-04.
 */

#ifndef leastChiSquaresFunctions1D_hpp
#define leastChiSquaresFunctions1D_hpp

// Detect windows
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(WIN64) || defined(_WIN64) || defined(__WIN64__) || defined(__NT__)
    #define __WINDOWS__
    #define WIN32
    #define GSL_DLL
#endif

#include <stdio.h>
#ifdef __WINDOWS__
    #include <io.h>
#else
    #include <unistd.h> // for access()
#endif
#include <iomanip> // for std::setw std::setfill
#include <iostream> // for std::cout std::cerr
#include <algorithm>
#include <vector> // for std::vector
#include <cmath> // for std::nan
#include <thread>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>

#ifdef __WINDOWS__
    void PyInit_leastChiSquares1D(void) {};
#endif


struct DataStruct4gsl
{
    size_t n;
    double *t;
    double *y;
};


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


namespace LeastChiSquaresFunctionsGaussian1D 
{
    static int func4gsl_f(const gsl_vector *variables, void *data, gsl_vector *f);// __attribute__ ((unused));
    static int func4gsl_df(const gsl_vector *variables, void *data, gsl_matrix *J);// __attribute__ ((unused));
    static int func4gsl_fvv(const gsl_vector *variables, const gsl_vector *v, void *data, gsl_vector *fvv);// __attribute__ ((unused));
    // see https://www.gnu.org/software/gsl/doc/html/nls.html#sec-providing-function-minimized
    
    /*
    static double gaussian(const double a, const double b, const double c, const double t);
    static int func_f(const gsl_vector * x, void *params, gsl_vector * f);
    static int func_df(const gsl_vector * x, void *params, gsl_matrix * J);
    static int func_fvv(const gsl_vector * x, const gsl_vector * v, void *params, gsl_vector * fvv);
    */

    int func4gsl_f(const gsl_vector *variables, void *data, gsl_vector *f)
    {
        // This should be the function: model minus data. 
        // variables are the free variables in the fitting function to be optimized.
        //std::cout << "func4gsl_f is called" << std::endl; 
        size_t i=0, j=0;
        //size_t ndata = ((struct DataStruct4gsl *)data)->n;
        //double *x = ((struct DataStruct4gsl *)data)->t;
        //double *y = ((struct DataStruct4gsl *)data)->y;
        size_t ndata = ((struct DataStruct *)data)->ndata;
        double *x = ((struct DataStruct *)data)->x;
        double *y = ((struct DataStruct *)data)->y;
        //std::cout << "func4gsl_f ndata " << ndata << std::endl;
        //std::cout << "func4gsl_f variables->size " << variables->size << std::endl;
        for (i=0; i < ndata; i++) {
            double ymodel = 0.0;
            for (j=0; j < (size_t)(variables->size/3); j++) {
                double A = gsl_vector_get(variables, 3*j+0);
                double mu = gsl_vector_get(variables, 3*j+1);
                double sigma = gsl_vector_get(variables, 3*j+2);
                ymodel += A * std::exp( - (x[i]-mu) * (x[i]-mu) / (2.0 * sigma * sigma) ); // 1D Gaussian
            }
            //gsl_vector_set(f, i, ymodel - y[i]); // this will cause no advance error when manually running gsl_multifit_nlinear_iterate
            gsl_vector_set(f, i, y[i] - ymodel);
        }
        return GSL_SUCCESS;
    }


    int func4gsl_df(const gsl_vector *variables, void *data, gsl_matrix *J)
    {
        // This should be the function: partial(model minus data) / partial(param) for each param.
        // If an analytic Jacobian is unavailable, or too expensive to compute, this function pointer may be set to NULL, in which case the Jacobian will be internally computed using finite difference approximations of the function f. -- https://www.gnu.org/software/gsl/doc/html/nls.html#sec-providing-function-minimized 
        //std::cout << "func4gsl_df is called" << std::endl;
        size_t i=0, j=0;
        //size_t ndata = ((struct DataStruct4gsl *)data)->n;
        //double *x = ((struct DataStruct4gsl *)data)->t;
        //double *y = ((struct DataStruct4gsl *)data)->y;
        size_t ndata = ((struct DataStruct *)data)->ndata;
        double *x = ((struct DataStruct *)data)->x;
        //double *y = ((struct DataStruct *)data)->y;
        for (i=0; i < ndata; i++) {
            double partial_A = 0.0;
            double partial_mu = 0.0;
            double partial_sigma = 0.0;
            for (j=0; j < (size_t)(variables->size/3); j++) {
                double A = gsl_vector_get(variables, 3*j+0);
                double mu = gsl_vector_get(variables, 3*j+1);
                double sigma = gsl_vector_get(variables, 3*j+2);
                
                /*
                partial_A = std::exp( - (x[i]-mu) * (x[i]-mu) / (2.0 * sigma * sigma) ); // 1D Gaussian derivative against A
                partial_mu = A * std::exp( - (x[i]-mu) * (x[i]-mu) / (2.0 * sigma * sigma) ) \
                               * ( 2.0 * (x[i]-mu) / (2.0 * sigma * sigma) ); 
                               // 1D Gaussian derivative against mu
                partial_sigma = A * std::exp( - (x[i]-mu) * (x[i]-mu) / (2.0 * sigma * sigma) ) \
                                  * ( (x[i]-mu) * (x[i]-mu) / (sigma * sigma * sigma) ); 
                                  // 1D Gaussian derivative against sigma
                */
                
                // following https://www.gnu.org/software/gsl/doc/html/nls.html "Geodesic Acceleration Example 2"
                double zi = (x[i]-mu) / sigma;
                double ei = std::exp( - 0.5 * zi * zi ); // = std::exp( - (x[i]-mu) * (x[i]-mu) / (2.0 * sigma * sigma) )
                partial_A = -ei; // 1D Gaussian derivative against A
                partial_mu = -A * ei \
                                * ( zi / sigma ); 
                                // 1D Gaussian derivative against mu
                partial_sigma = -A * ei \
                                   * ( zi * zi / sigma ); 
                                   // 1D Gaussian derivative against sigma
                                  
                gsl_matrix_set(J, i, 3*j+0, partial_A);
                gsl_matrix_set(J, i, 3*j+1, partial_mu);
                gsl_matrix_set(J, i, 3*j+2, partial_sigma);
            }
        }
        return GSL_SUCCESS;
    }



    int func4gsl_fvv(const gsl_vector *variables, const gsl_vector *v, void *data, gsl_vector *fvv)
    {
        // TODO: for 3 variable fitting only
        //std::cout << "func4gsl_fvv is called" << std::endl;
        //struct DataStruct4gsl * d = (struct DataStruct4gsl *) data;
        struct DataStruct * d = (struct DataStruct *) data;
        double a = gsl_vector_get(variables, 0);
        double b = gsl_vector_get(variables, 1);
        double c = gsl_vector_get(variables, 2);
        double va = gsl_vector_get(v, 0);
        double vb = gsl_vector_get(v, 1);
        double vc = gsl_vector_get(v, 2);
        size_t i;

        //for (i = 0; i < d->n; i++)
        for (i = 0; i < d->ndata; i++)
        {
            //double ti = d->t[i];
            double ti = d->x[i];
            double zi = (ti - b) / c;
            double ei = exp(-0.5 * zi * zi);
            double Dab = -zi * ei / c;
            double Dac = -zi * zi * ei / c;
            double Dbb = a * ei / (c * c) * (1.0 - zi*zi);
            double Dbc = a * zi * ei / (c * c) * (2.0 - zi*zi);
            double Dcc = a * zi * zi * ei / (c * c) * (3.0 - zi*zi);
            double sum;

            sum = 2.0 * va * vb * Dab +
                  2.0 * va * vc * Dac +
                        vb * vb * Dbb +
                  2.0 * vb * vc * Dbc +
                        vc * vc * Dcc;

            gsl_vector_set(fvv, i, sum);
        }

        return GSL_SUCCESS;
    }
    
    
    /* functions from gsl website
    double gaussian(const double a, const double b, const double c, const double t)
    {
        const double z = (t - b) / c;
        return (a * exp(-0.5 * z * z));
    }
    
    int func_f(const gsl_vector * x, void *params, gsl_vector * f)
    {
        struct DataStruct4gsl *d = (struct DataStruct4gsl *) params;
        double a = gsl_vector_get(x, 0);
        double b = gsl_vector_get(x, 1);
        double c = gsl_vector_get(x, 2);
        size_t i;
        
        for (i = 0; i < d->n; ++i)
        {
            double ti = d->t[i];
            double yi = d->y[i];
            double y = gaussian(a, b, c, ti);
            
            gsl_vector_set(f, i, yi - y);
        }
        
        return GSL_SUCCESS;
    }
    
    int func_df(const gsl_vector * x, void *params, gsl_matrix * J)
    {
        struct DataStruct4gsl *d = (struct DataStruct4gsl *) params;
        double a = gsl_vector_get(x, 0);
        double b = gsl_vector_get(x, 1);
        double c = gsl_vector_get(x, 2);
        size_t i;
        
        for (i = 0; i < d->n; ++i)
        {
            double ti = d->t[i];
            double zi = (ti - b) / c;
            double ei = exp(-0.5 * zi * zi);
            
            gsl_matrix_set(J, i, 0, -ei);
            gsl_matrix_set(J, i, 1, -(a / c) * ei * zi);
            gsl_matrix_set(J, i, 2, -(a / c) * ei * zi * zi);
        }
        
        return GSL_SUCCESS;
    }
    
    int func_fvv(const gsl_vector * x, const gsl_vector * v, void *params, gsl_vector * fvv)
    {
        struct DataStruct4gsl *d = (struct DataStruct4gsl *) params;
        double a = gsl_vector_get(x, 0);
        double b = gsl_vector_get(x, 1);
        double c = gsl_vector_get(x, 2);
        double va = gsl_vector_get(v, 0);
        double vb = gsl_vector_get(v, 1);
        double vc = gsl_vector_get(v, 2);
        size_t i;
        
        for (i = 0; i < d->n; ++i)
        {
            double ti = d->t[i];
            double zi = (ti - b) / c;
            double ei = exp(-0.5 * zi * zi);
            double Dab = -zi * ei / c;
            double Dac = -zi * zi * ei / c;
            double Dbb = a * ei / (c * c) * (1.0 - zi*zi);
            double Dbc = a * zi * ei / (c * c) * (2.0 - zi*zi);
            double Dcc = a * zi * zi * ei / (c * c) * (3.0 - zi*zi);
            double sum;
            
            sum = 2.0 * va * vb * Dab +
            2.0 * va * vc * Dac +
            vb * vb * Dbb +
            2.0 * vb * vc * Dbc +
            vc * vc * Dcc;
            
            gsl_vector_set(fvv, i, sum);
        }
        
        return GSL_SUCCESS;
    }
    */
    
    
} // end namespace LeastChiSquaresFunctionsGaussian1D

#endif /* leastChiSquaresFunctions1D_hpp */
