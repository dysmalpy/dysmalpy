/*
    This is a library to transform a data cube from source plane to image plane given a lensing model, and convolve with certain point spread function and line spread function.
    
    Last updates: 
        2021-07-23, first version finished. Daizhong Liu, MPE.
        2021-08-04, debugging level adjusted. Daizhong Liu, MPE.  
    
 */
#ifndef LENSINGTRANSFORMER_H
#define LENSINGTRANSFORMER_H

// Detect windows
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(WIN64) || defined(_WIN64) || defined(__WIN64__) || defined(__NT__)
    #define __WINDOWS__
    #define GSL_DLL
#endif

//#include <Python.h> // This implies inclusion of the following standard headers: <stdio.h>, <string.h>, <errno.h>, <limits.h>, <assert.h> and <stdlib.h> (if available). -- https://docs.python.org/2/c-api/intro.html
#include <stdio.h>
#ifdef __WINDOWS__
    #include <io.h>
    #define F_OK 0
#else
    #include <unistd.h> // for access()
#endif
#include <iomanip> // for std::setw std::setfill
#include <iostream> // for std::cout std::cerr
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector> // for std::vector
#include <cmath> // for std::nan
#include <cstring> // for std::memcpy
#include <thread>
#include "fitsio.h"
#include "gsl/gsl_matrix.h"
//#include "fftw3/fftw3.h"
//#include "fftw3/fftw3-mpi.h"

#ifdef __WINDOWS__
    void PyInit_lensingTransformer(void) {};
#endif

/*
    LensingTransformer Class.
 */
class LensingTransformer {

public: 
    LensingTransformer();
    ~LensingTransformer();
    void readGlaficMeshFile(std::string FilePath, double ra, double dec, int verbose=1);
    void readGlaficLensFile(std::string FilePath, double ra, double dec, double pixelsize, int verbose=1);
    void readSourcePlaneDataCube(std::string FilePath, double ra, double dec, double pixelsize, double cenx=std::nan(""), double ceny=std::nan(""), int verbose=1);
    void readSourcePlaneDataCube(double *data, long nx, long ny, long nchan, double ra, double dec, double pixelsize, double cenx=std::nan(""), double ceny=std::nan(""), int verbose=1);
    void linkSourcePlaneToImagePlane(double ra, double dec, double pixelsize, long sizex, long sizey, double cenx=std::nan(""), double ceny=std::nan(""), int verbose=1);
    void writeImagePlaneDataCube(std::string FilePath, int overwrite=0, int verbose=1);
    int errorCode();
    int debugCode();
    int debugLevel();
    void setDebugLevel(int DebugLevel);

    gsl_matrix *DeflectionX;
    gsl_matrix *DeflectionY;
    double LensModelCenterRA; // in units of degree
    double LensModelCenterDec; // in units of degree
    double LensModelPixelSize; // in units of arcsec
    std::vector<gsl_matrix *> SourcePlaneDataCube;
    double SourcePlaneCenterX; // in units of pixel, starting from 0 to NAXIS1-1
    double SourcePlaneCenterY; // in units of pixel, starting from 0 to NAXIS2-1
    double SourcePlaneCenterRA; // in units of degree
    double SourcePlaneCenterDec; // in units of degree
    double SourcePlanePixelSize; // in units of arcsec
    //std::string SourcePlaneCTYPE3; // 
    //std::string SourcePlaneCUNIT3; // 
    //double SourcePlaneCRPIX3; //
    //double SourcePlaneCRVAL3; //
    //double SourcePlaneCDELT3; //
    double ImagePlaneCenterX; // in units of pixel, starting from 0 to NAXIS1-1
    double ImagePlaneCenterY; // in units of pixel, starting from 0 to NAXIS2-1
    double ImagePlaneCenterRA; // in units of degree
    double ImagePlaneCenterDec; // in units of degree
    double ImagePlanePixelSize; // in units of arcsec
    double LensModelCenterInSourcePlaneX; 
    double LensModelCenterInSourcePlaneY; 
    double LensModelCenterInImagePlaneX; 
    double LensModelCenterInImagePlaneY; 
    gsl_vector *SourcePlaneMeshGridX1;
    gsl_vector *SourcePlaneMeshGridX2;
    gsl_vector *SourcePlaneMeshGridY1;
    gsl_vector *SourcePlaneMeshGridY2;
    gsl_vector *ImagePlaneMeshGridX1;
    gsl_vector *ImagePlaneMeshGridX2;
    gsl_vector *ImagePlaneMeshGridY1;
    gsl_vector *ImagePlaneMeshGridY2;
    gsl_matrix *SourcePlane2D; // 2d array
    std::vector<std::vector<double *> > ImagePlane2D; // should be a 2d array of pointers, each pixel pointing to a source plane pixel.
    std::string LensFile;

private:
    int ErrorCode;
    int DebugCode;
};


// prevent name mangling, using extern "C"
#ifdef __cplusplus 
extern "C" {
#endif

// A global variable, debug flag.
extern int GlobalDebug;

// A global variable, list of all LensingTransformer instances.
//extern std::vector<LensingTransformer *> AllLensingTransformerInstances;
//extern PyListObject *PyAllLensingTransformerInstances;

// Function to check endian.
bool isLittleEndian();

// struct
struct MeshGridCell
{
    double x11, y11, x12, y12, x22, y22, x21, y21; // 1-based xy coordinate of the four corners, lower left, lower right, upper right, and upper left. (same order as glafic mesh file)
    double xcen, ycen;
    int i11, j11, i12, j12, i22, j22, i21, j21; // 0-based integer index of the four corners, lower left, lower right, upper right, and upper left.
    int icen, jcen;
};

// Function to check if a point is in a polygon
bool checkPointInMeshGridCell(double testx, double testy, MeshGridCell cell);
int checkPointInPolygon(double testx, double testy, long nvert, double *vertx, double *verty);


// Function to set global debug
void setGlobalDebugLevel(int value);


// Function exposed to Python to create a class.
void *createLensingTransformer(\
    const char *mesh_file,
    double mesh_ra,
    double mesh_dec,
    double *source_plane_data_cube,
    long source_plane_data_nx,
    long source_plane_data_ny,
    long source_plane_data_nchan, 
    double source_plane_ra,
    double source_plane_dec,
    double source_plane_pixelsize,
    double source_plane_cenx=std::nan(""),
    double source_plane_ceny=std::nan(""),
    int verbose=1);


void updateSourcePlaneDataCube(\
    void *ptr,
    double *source_plane_data_cube,
    int verbose=1);


double *performLensingTransformation(\
    void *ptr,
    double image_plane_ra,
    double image_plane_dec,
    double image_plane_pixelsize,
    long image_plane_sizex,
    long image_plane_sizey,
    double image_plane_cenx=std::nan(""),
    double image_plane_ceny=std::nan(""),
    int verbose=1);

// Function exposed to Python to destroy a class.
void destroyLensingTransformer(void *ptr);


// Function to save data to FITS file.
void saveDataCubeToFitsFile(std::string FilePath, double *data, long sizex, long sizey, long nchan, double pixelsize, double ra, double dec, double cenx=std::nan(""), double ceny=std::nan(""), int verbose=1);


#ifdef __cplusplus
}
#endif


#endif // LENSINGTRANSFORMER_H
