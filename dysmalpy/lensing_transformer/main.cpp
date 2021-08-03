/*
    This is a testing main program to call the lensingTransformer functions.
    
    Last update: 2021-08-03, Daizhong Liu, MPE.  
    
 */

#include <stdio.h>
#include <stdlib.h>
#include "gsl/gsl_matrix.h"
#include <iostream>
#include "lensingTransformer.hpp"

int main(int argc, char **argv)
{
    Py_Initialize();
    
    std::cout << "Hello" << std::endl;
    
    //PyRun_SimpleString("print('Hello from PyRun')\n");
    
    LensingTransformer *MyLensingTransformer = new LensingTransformer(); 
    //MyLensingTransformer->readGlaficLensFile("data/lens.fits", 135.3434883, 18.2418031, 0.02);
    MyLensingTransformer->readGlaficMeshFile("data/mesh.dat", 135.3434883, 18.2418031);
    MyLensingTransformer->readSourcePlaneDataCube("data/model_cube.fits", 135.3434883, 18.2418031, 0.02);
    MyLensingTransformer->linkSourcePlaneToImagePlane(135.3434883, 18.2418031, 0.04, 600, 600);
    MyLensingTransformer->writeImagePlaneDataCube("data/image_plane_cube.fits", 1);
    
    long nchan = MyLensingTransformer->SourcePlaneDataCube.size();
    long ny = MyLensingTransformer->SourcePlaneDataCube[0]->size1;
    long nx = MyLensingTransformer->SourcePlaneDataCube[0]->size2;
    double *data = (double *)malloc(nchan*ny*nx*sizeof(double));
    long ichan, j, i;
    for (ichan=0; ichan<nchan; ichan++) {
        memcpy(data+ichan*ny*nx, gsl_matrix_ptr(MyLensingTransformer->SourcePlaneDataCube[ichan],0,0), ny*nx*sizeof(double));
        //for (j=0; j<ny; j++) {
        //    for (i=0; i<nx; i++) {
        //        data[ichan*ny*nx+j*nx+i] = gsl_matrix_get(MyLensingTransformer->SourcePlaneDataCube[ichan], j, i);
        //    }
        //}
    }
    
    LensingTransformer *MyLensingTransformer2 = (LensingTransformer *) createLensingTransformer(\
        "data/mesh.dat", 135.3434883, 18.2418031, 
        data, nx, ny, nchan, 135.3434883, 18.2418031, 0.02);
    
    //MyLensingTransformer2->readGlaficLensFile("data/lens.fits", 135.3434883, 18.2418031);
    //MyLensingTransformer2->readGlaficMeshFile("data/mesh.dat", 135.3434883, 18.2418031);
    //MyLensingTransformer2->readSourcePlaneDataCube("data/model_cube.fits", 135.3434883, 18.2418031, 0.02);
    //MyLensingTransformer2->linkSourcePlaneToImagePlane(135.3434883, 18.2418031, 0.04, 600, 600);
    //MyLensingTransformer2->writeImagePlaneDataCube("data/image_plane_cube.fits", 1);
    
    double *outputdata = (double *)performLensingTransformation(\
        MyLensingTransformer2,
        135.3434883, 18.2418031, 0.04, 600, 600);
    
    saveDataCubeToFitsFile("data/image_plane_cube_2.fits", \
        outputdata, 600, 600, nchan, 0.04, 135.3434883, 18.2418031);
    
    // debug output, to check with the pixel values in another output file "data/image_plane_cube.fits"
    //ichan = 48-1;
    //j = 107;
    //std::cout << "outputdata[" << ichan << "," << j << ",207:207+10]"; 
    //for (i=207; i<207+10; i++) { std::cout << " " << outputdata[ichan*ny*nx+j*nx+i]; }
    //std::cout << std::endl; 
    // should be 0.00241317, 0.00244377, 0.00247104, 0.00254584, 0.0025672
    
    
    // clean up 
    destroyLensingTransformer(MyLensingTransformer2);
    
    delete data;
    
    delete MyLensingTransformer;
}


