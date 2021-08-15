/*
    See notes in "lensingTransformer.hpp".
 */
#include "lensingTransformer.hpp"


LensingTransformer::LensingTransformer()
{
    this->ErrorCode = 0;
    this->DebugCode = 1;
    this->DeflectionX = NULL;
    this->DeflectionY = NULL;
    this->LensModelCenterRA = std::nan("");
    this->LensModelCenterDec = std::nan("");
    this->LensModelPixelSize = std::nan("");
    this->SourcePlaneCenterX = std::nan("");
    this->SourcePlaneCenterY = std::nan("");
    this->SourcePlaneCenterRA = std::nan("");
    this->SourcePlaneCenterDec = std::nan("");
    this->SourcePlanePixelSize = std::nan("");
    this->ImagePlaneCenterX = std::nan("");
    this->ImagePlaneCenterY = std::nan("");
    this->ImagePlaneCenterRA = std::nan("");
    this->ImagePlaneCenterDec = std::nan("");
    this->ImagePlanePixelSize = std::nan("");
    this->SourcePlaneDataCube.clear();
    this->SourcePlane2D = NULL; 
    this->ImagePlane2D.clear();
    this->SourcePlaneMeshGridX1 = NULL;
    this->SourcePlaneMeshGridX2 = NULL;
    this->SourcePlaneMeshGridY1 = NULL;
    this->SourcePlaneMeshGridY2 = NULL;
    this->ImagePlaneMeshGridX1 = NULL;
    this->ImagePlaneMeshGridX2 = NULL;
    this->ImagePlaneMeshGridY1 = NULL;
    this->ImagePlaneMeshGridY2 = NULL;
    //if (this->DebugCode > 0) {
    //    std::cout << "LensingTransformer is created." << std::endl;
    //}
}


LensingTransformer::~LensingTransformer()
{
    if (this->DebugCode > 0) 
        std::cout << "LensingTransformer::~LensingTransformer() is called." << std::endl;
    if (this->DeflectionX) { 
        gsl_matrix_free(this->DeflectionX); 
        this->DeflectionX = NULL;
    }
    if (this->DeflectionY) {
        gsl_matrix_free(this->DeflectionY);
        this->DeflectionY = NULL;
    }
    if (this->SourcePlaneDataCube.size()>0) {
        this->SourcePlaneDataCube.clear();
    }
    if (this->ImagePlane2D.size()>0) {
        this->ImagePlane2D.clear();
    }
    if (this->SourcePlane2D) {
        //delete this->SourcePlane2D;
        gsl_matrix_free(this->SourcePlane2D);
    }
    if (this->SourcePlaneMeshGridX1) {
        gsl_vector_free(this->SourcePlaneMeshGridX1);
        this->SourcePlaneMeshGridX1 = NULL;
    }
    if (this->SourcePlaneMeshGridY1) {
        gsl_vector_free(this->SourcePlaneMeshGridY1);
        this->SourcePlaneMeshGridY1 = NULL;
    }
    if (this->SourcePlaneMeshGridX2) {
        gsl_vector_free(this->SourcePlaneMeshGridX2);
        this->SourcePlaneMeshGridX2 = NULL;
    }
    if (this->SourcePlaneMeshGridY2) {
        gsl_vector_free(this->SourcePlaneMeshGridY2);
        this->SourcePlaneMeshGridY2 = NULL;
    }
    if (this->ImagePlaneMeshGridX1) {
        gsl_vector_free(this->ImagePlaneMeshGridX1);
        this->ImagePlaneMeshGridX1 = NULL;
    }
    if (this->ImagePlaneMeshGridY1) {
        gsl_vector_free(this->ImagePlaneMeshGridY1);
        this->ImagePlaneMeshGridY1 = NULL;
    }
    if (this->ImagePlaneMeshGridX2) {
        gsl_vector_free(this->ImagePlaneMeshGridX2);
        this->ImagePlaneMeshGridX2 = NULL;
    }
    if (this->ImagePlaneMeshGridY2) {
        gsl_vector_free(this->ImagePlaneMeshGridY2);
        this->ImagePlaneMeshGridY2 = NULL;
    }
    if (this->DebugCode > 0) {
        std::cout << "LensingTransformer is destroyed." << std::endl;
    }
    if (this->DebugCode > 0) 
        std::cout << "LensingTransformer::~LensingTransformer() finished." << std::endl;
}


void LensingTransformer::readGlaficMeshFile(std::string FilePath, double ra, double dec, int verbose)
{
    /* Open glafic "lens.fits" file and read the X Y deflection arrays.
    */
    if (this->DebugCode > 0) 
        std::cout << "readGlaficMeshFile is called." << std::endl;
    // Clean up previous loads
    if(this->SourcePlaneMeshGridX1) { gsl_vector_free(this->SourcePlaneMeshGridX1); this->SourcePlaneMeshGridX1 = NULL; }
    if(this->SourcePlaneMeshGridX2) { gsl_vector_free(this->SourcePlaneMeshGridX2); this->SourcePlaneMeshGridX2 = NULL; }
    if(this->SourcePlaneMeshGridY1) { gsl_vector_free(this->SourcePlaneMeshGridY1); this->SourcePlaneMeshGridY1 = NULL; }
    if(this->SourcePlaneMeshGridY2) { gsl_vector_free(this->SourcePlaneMeshGridY2); this->SourcePlaneMeshGridY2 = NULL; }
    if(this->ImagePlaneMeshGridX1) { gsl_vector_free(this->ImagePlaneMeshGridX1); this->ImagePlaneMeshGridX1 = NULL; }
    if(this->ImagePlaneMeshGridX2) { gsl_vector_free(this->ImagePlaneMeshGridX2); this->ImagePlaneMeshGridX2 = NULL; }
    if(this->ImagePlaneMeshGridY1) { gsl_vector_free(this->ImagePlaneMeshGridY1); this->ImagePlaneMeshGridY1 = NULL; }
    if(this->ImagePlaneMeshGridY2) { gsl_vector_free(this->ImagePlaneMeshGridY2); this->ImagePlaneMeshGridY2 = NULL; }
    // 
    this->LensModelCenterRA = ra;
    this->LensModelCenterDec = dec;
    //
    std::ifstream fp(FilePath);
    if (!fp.is_open()) { this->ErrorCode=255; std::cerr << "Error! Failed to open the mesh file \"" << FilePath << "\"!" << std::endl; return; }
    double xi1, yi1, xs1, ys1, xi2, yi2, xs2, ys2;
    std::istream::streampos ipos = fp.tellg(); 
    long nrow = std::count(std::istreambuf_iterator<char>(fp), std::istreambuf_iterator<char>(), '\n');
    fp.seekg(ipos);
    if (verbose > 0) 
        std::cout << "readGlaficMeshFile() " << nrow << " rows are found in the mesh file." << std::endl;
    this->SourcePlaneMeshGridX1 = gsl_vector_alloc(nrow);
    this->SourcePlaneMeshGridY1 = gsl_vector_alloc(nrow);
    this->SourcePlaneMeshGridX2 = gsl_vector_alloc(nrow);
    this->SourcePlaneMeshGridY2 = gsl_vector_alloc(nrow);
    this->ImagePlaneMeshGridX1 = gsl_vector_alloc(nrow);
    this->ImagePlaneMeshGridY1 = gsl_vector_alloc(nrow);
    this->ImagePlaneMeshGridX2 = gsl_vector_alloc(nrow);
    this->ImagePlaneMeshGridY2 = gsl_vector_alloc(nrow);
    long irow = 0;
    while(fp >> xi1 >> yi1 >> xs1 >> ys1 >> xi2 >> yi2 >> xs2 >> ys2) {
        gsl_vector_set(this->SourcePlaneMeshGridX1, irow, xs1);
        gsl_vector_set(this->SourcePlaneMeshGridY1, irow, ys1);
        gsl_vector_set(this->SourcePlaneMeshGridX2, irow, xs2);
        gsl_vector_set(this->SourcePlaneMeshGridY2, irow, ys2);
        gsl_vector_set(this->ImagePlaneMeshGridX1, irow, xi1);
        gsl_vector_set(this->ImagePlaneMeshGridY1, irow, yi1);
        gsl_vector_set(this->ImagePlaneMeshGridX2, irow, xi2);
        gsl_vector_set(this->ImagePlaneMeshGridY2, irow, yi2);
        irow++;
        //std::cout << "irow " << irow << " xi1 " << xi1 << " yi1 " << yi1 << " xs1 " << xs1 << " ys1 " << ys1 << std::endl;
    }
    if (irow < (long)(this->SourcePlaneMeshGridX1->size)) {
        // trim size if there are empty lines in mesh.dat
        gsl_vector *NewSourcePlaneMeshGridX1 = gsl_vector_alloc(irow);
        gsl_vector *NewSourcePlaneMeshGridY1 = gsl_vector_alloc(irow);
        gsl_vector *NewSourcePlaneMeshGridX2 = gsl_vector_alloc(irow);
        gsl_vector *NewSourcePlaneMeshGridY2 = gsl_vector_alloc(irow);
        gsl_vector *NewImagePlaneMeshGridX1 = gsl_vector_alloc(irow);
        gsl_vector *NewImagePlaneMeshGridY1 = gsl_vector_alloc(irow);
        gsl_vector *NewImagePlaneMeshGridX2 = gsl_vector_alloc(irow);
        gsl_vector *NewImagePlaneMeshGridY2 = gsl_vector_alloc(irow);
        gsl_vector_memcpy(NewSourcePlaneMeshGridX1, this->SourcePlaneMeshGridX1);
        gsl_vector_memcpy(NewSourcePlaneMeshGridY1, this->SourcePlaneMeshGridY1);
        gsl_vector_memcpy(NewSourcePlaneMeshGridX2, this->SourcePlaneMeshGridX2);
        gsl_vector_memcpy(NewSourcePlaneMeshGridY2, this->SourcePlaneMeshGridY2);
        gsl_vector_memcpy(NewImagePlaneMeshGridX1, this->ImagePlaneMeshGridX1);
        gsl_vector_memcpy(NewImagePlaneMeshGridY1, this->ImagePlaneMeshGridY1);
        gsl_vector_memcpy(NewImagePlaneMeshGridX2, this->ImagePlaneMeshGridX2);
        gsl_vector_memcpy(NewImagePlaneMeshGridY2, this->ImagePlaneMeshGridY2);
        gsl_vector_free(this->SourcePlaneMeshGridX1); 
        gsl_vector_free(this->SourcePlaneMeshGridY1); 
        gsl_vector_free(this->SourcePlaneMeshGridX2); 
        gsl_vector_free(this->SourcePlaneMeshGridY2); 
        gsl_vector_free(this->ImagePlaneMeshGridX1); 
        gsl_vector_free(this->ImagePlaneMeshGridY1); 
        gsl_vector_free(this->ImagePlaneMeshGridX2); 
        gsl_vector_free(this->ImagePlaneMeshGridY2); 
        this->SourcePlaneMeshGridX1 = NewSourcePlaneMeshGridX1;
        this->SourcePlaneMeshGridY1 = NewSourcePlaneMeshGridY1;
        this->SourcePlaneMeshGridX2 = NewSourcePlaneMeshGridX2;
        this->SourcePlaneMeshGridY2 = NewSourcePlaneMeshGridY2;
        this->ImagePlaneMeshGridX1 = NewImagePlaneMeshGridX1;
        this->ImagePlaneMeshGridY1 = NewImagePlaneMeshGridY1;
        this->ImagePlaneMeshGridX2 = NewImagePlaneMeshGridX2;
        this->ImagePlaneMeshGridY2 = NewImagePlaneMeshGridY2;
    } else if (irow < 4) {
        std::cerr << "Error! No enough rows are found in the mesh file: " << FilePath << std::endl;
        gsl_vector_free(this->SourcePlaneMeshGridX1); 
        gsl_vector_free(this->SourcePlaneMeshGridY1); 
        gsl_vector_free(this->SourcePlaneMeshGridX2); 
        gsl_vector_free(this->SourcePlaneMeshGridY2); 
        gsl_vector_free(this->ImagePlaneMeshGridX1); 
        gsl_vector_free(this->ImagePlaneMeshGridY1); 
        gsl_vector_free(this->ImagePlaneMeshGridX2); 
        gsl_vector_free(this->ImagePlaneMeshGridY2); 
        this->SourcePlaneMeshGridX1 = NULL;
        this->SourcePlaneMeshGridY1 = NULL;
        this->SourcePlaneMeshGridX2 = NULL;
        this->SourcePlaneMeshGridY2 = NULL;
        this->ImagePlaneMeshGridX1 = NULL;
        this->ImagePlaneMeshGridY1 = NULL;
        this->ImagePlaneMeshGridX2 = NULL;
        this->ImagePlaneMeshGridY2 = NULL;
        this->ErrorCode = 255; // bad
        return; 
    }
    if (verbose > 0) 
        std::cout << "readGlaficMeshFile() " << this->SourcePlaneMeshGridX1->size << " mesh grid are read." << std::endl;
    this->ErrorCode = 0; // good
    if (this->DebugCode > 0) 
        std::cout << "readGlaficMeshFile finished." << std::endl;
}


void LensingTransformer::readGlaficLensFile(std::string FilePath, double ra, double dec, double pixelsize, int verbose)
{
    /* Open glafic "lens.fits" file and read the X Y deflection arrays.
    */
    if (this->DebugCode > 0) 
        std::cout << "readGlaficLensFile is called." << std::endl;
    // Clean up previous loads
    if (this->DeflectionX) {
        gsl_matrix_free(this->DeflectionX); 
        this->DeflectionX = NULL;
    }
    if (this->DeflectionY) {
        gsl_matrix_free(this->DeflectionY); 
        this->DeflectionY = NULL;
    }
    // Store input values
    this->LensModelCenterRA = ra;
    this->LensModelCenterDec = dec;
    this->LensModelPixelSize = pixelsize;
    // Open FITS file.
    // See some programming instructions at https://heasarc.gsfc.nasa.gov/docs/software/fitsio/cexamples/cookbook.c
    // and at  https://heasarc.gsfc.nasa.gov/docs/software/fitsio/quick/node9.html
    fitsfile *fptr = NULL;
    long i = 0;
    int status = 0;
    int iomode = 0; // READONLY (0) or READWRITE (1)
    //char *keyvalue = NULL;
    //char *keycomment = NULL;
    int bitpix = 0;
    int datatype = 0;
    //int nfound = 0;
    int maxdim = 3;
    int naxis = 0;
    long *naxes = NULL;
    long fpixel = 0; // first pixel's index, (not coordinate, [1-NAXIS1, 1-NAXIS2, ...]) 
    float nullval = 0; // don't check for null values in the image
    long nbuffer = 0;
    double *buffer = NULL;
    int anynull;
    // open file
    if (this->DebugCode > 0) 
        std::cout << "calling fits_open_file" << std::endl;
    if (verbose > 0)
        std::cout << "readGlaficLensFile() " << FilePath << std::endl;
    this->ErrorCode = fits_open_file(&fptr, FilePath.c_str(), iomode, &status);
    if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    // get image dimension
    if (this->DebugCode > 0) 
        std::cout << "calling fits_get_img_param" << std::endl;
    naxes = new long[maxdim];
    this->ErrorCode = fits_get_img_param(fptr, maxdim, &bitpix, &naxis, naxes, &status);
    if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    if (verbose > 0) {
        std::cout << "readGlaficLensFile() bitpix: " << bitpix << ", naxis: " << naxis << ", naxes: ";
        for (i=0; i<naxis; i++) {
            std::cout << naxes[i];
            if (naxis-1!=i) std::cout << " x "; 
        }
        std::cout << ", status: " << status << std::endl;
    }
    if (naxis!=3) { this->ErrorCode=255; std::cerr << "Error! The input data \"" << FilePath << "\" is not of 3-dimension!" << std::endl; }
    // prepare output data arrays
    // note that they are C-order, for gsl_matrix_get(m,i,j), it's accessing m->data[i * m->tda + j]
    // see https://www.gnu.org/software/gsl/doc/html/vectors.html
    this->DeflectionX = gsl_matrix_alloc(naxes[1], naxes[0]);
    this->DeflectionY = gsl_matrix_alloc(naxes[1], naxes[0]);
    //gsl_matrix_set_zero(this->DeflectionX); // seems already done
    //gsl_matrix_set_zero(this->DeflectionY); // seems already done
    // read image data array
    nbuffer = naxes[1]*naxes[0];
    datatype = TDOUBLE;
    fpixel = 1; // set to the first pixel (lower left corner in DS9)
    // get some pixels for checking
    if (this->DebugCode > 2) {
        buffer = new double[10];
        for (i=0; i<10; i++) buffer[i] = 0.0;
        std::cout << "calling fits_read_img" << std::endl;
        status = 0;
        this->ErrorCode = fits_read_img(fptr, datatype, fpixel, 10, &nullval,
                  buffer, &anynull, &status);
        if(this->ErrorCode != 0)
            fits_report_error(stderr, status);
        std::cout << "checking buffer";
        for (i=0; i<10; i++) 
            std::cout << " " << buffer[i];
        std::cout << std::endl;
        delete buffer;
        buffer = NULL;
    }
    // get the first channel, DeflectionX
    fpixel = 1; 
    buffer = gsl_matrix_ptr(this->DeflectionX, 0, 0); // directly accessing this->DeflectionX->data
    if (this->DebugCode > 0) 
        std::cout << "calling fits_read_img" << std::endl;
    this->ErrorCode = fits_read_img(fptr, datatype, fpixel, nbuffer, &nullval, buffer, &anynull, &status);
    if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    // get the second channel, DeflectionY
    fpixel += nbuffer; 
    buffer = gsl_matrix_ptr(this->DeflectionY, 0, 0); // directly accessing this->DeflectionX->data
    if (this->DebugCode > 0) 
        std::cout << "calling fits_read_img" << std::endl;
    this->ErrorCode = fits_read_img(fptr, datatype, fpixel, nbuffer, &nullval, buffer, &anynull, &status);
    if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    // check
    if (this->DebugCode > 2) {
        std::cout << "checking gsl_matrix_get(this->DeflectionX, 0, 0:10)"; 
        for (i=0; i<10; i++) 
            std::cout << " " << gsl_matrix_get(this->DeflectionY, 0, i);
        std::cout << std::endl;
        std::cout << "checking gsl_matrix_get(this->DeflectionX, 1, 0:10)";
        for (i=0; i<10; i++) 
            std::cout << " " << gsl_matrix_get(this->DeflectionY, 1, i);
        std::cout << std::endl;
    }
    // clean up
    if (this->DebugCode > 0) 
        std::cout << "cleaning up" << std::endl;
    delete naxes;
    buffer = NULL;
    // close file
    this->ErrorCode = fits_close_file(fptr, &status);
    if (this->ErrorCode != 0) {
        fits_report_error(stderr, status);
        std::cerr << "Error! Failed to open the FITS file \"" << FilePath << "\"! Please check error messages above." << std::endl;
    }
    if (this->DebugCode > 0) 
        std::cout << "readGlaficLensFile finished." << std::endl;
}


void LensingTransformer::readSourcePlaneDataCube(std::string FilePath, double ra, double dec, double pixelsize, double cenx, double ceny, int verbose)
{
    /* Read source plane data cube from FITS file.
     */
    if (this->DebugCode > 0) 
        std::cout << "readSourcePlaneDataCube is called." << std::endl;
    // Check input values
    if (pixelsize<=0.0) {
        this->ErrorCode = 255; 
        std::cerr << "Error! pixelsize " << pixelsize << " is not a positive float number for readSourcePlaneDataCube()." << std::endl;
        return;
    }
    // Clean up previous loads
    long ichan = 0;
    if (this->SourcePlaneDataCube.size()>0) {
        for (ichan=0; ichan<(long)(this->SourcePlaneDataCube.size()); ichan++) {
            gsl_matrix_free(this->SourcePlaneDataCube[ichan]);
        }
    }
    this->SourcePlaneDataCube.clear();
    // Store input values
    this->SourcePlaneCenterRA = ra;
    this->SourcePlaneCenterDec = dec;
    this->SourcePlanePixelSize = pixelsize;
    // Open FITS file with cfitsio.
    fitsfile *fptr = NULL;
    long i = 0;
    int status = 0, iomode = 0, bitpix = 0, datatype = 0, maxdim = 3, naxis = 0, anynull = 0;
    long *naxes = NULL;
    long nbuffer = 0, fpixel = 0; // first pixel's index 
    float nullval = 0; // don't check for null values in the image
    double *buffer = NULL;
    // open file
    this->ErrorCode = fits_open_file(&fptr, FilePath.c_str(), iomode, &status);
    if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    // get image dimension
    naxes = new long[maxdim];
    this->ErrorCode = fits_get_img_param(fptr, maxdim, &bitpix, &naxis, naxes, &status);
    if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    if (verbose > 0) {
        std::cout << "readSourcePlaneDataCube() bitpix: " << bitpix << ", naxis: " << naxis << ", naxes: ";
        for (i=0; i<naxis; i++) {
            std::cout << naxes[i];
            if (naxis-1!=i) std::cout << " x "; 
        }
        std::cout << ", status: " << status << std::endl;
    }
    if (naxis!=3) { this->ErrorCode=255; std::cerr << "Error! The input data \"" << FilePath << "\" is not of 3-dimension!" << std::endl; }
    // Store the so-called image center, which is actually the reference pixel
    if (std::isnan(cenx)) {
        this->SourcePlaneCenterX = (double(naxes[0])+1.0)/2.0;
    } else {
        this->SourcePlaneCenterX = cenx;
    }
    if (std::isnan(ceny)) {
        this->SourcePlaneCenterY = (double(naxes[1])+1.0)/2.0;
    } else {
        this->SourcePlaneCenterY = ceny;
    }
    // prepare output data arrays
    fpixel = 1; // set to the very first pixel, cfitsio pixel coordinate is 1-based
    nbuffer = naxes[1]*naxes[0];
    datatype = TDOUBLE;
    for (ichan=0; ichan<naxes[2]; ichan++) {
        this->SourcePlaneDataCube.push_back(gsl_matrix_alloc(naxes[1], naxes[0]));
        // read image data array
        if (ichan>0) fpixel += nbuffer; // next channel
        buffer = gsl_matrix_ptr(this->SourcePlaneDataCube[ichan], 0, 0); // directly accessing matrix data, gsl index is 0-based, row, col
        this->ErrorCode = fits_read_img(fptr, datatype, fpixel, nbuffer, &nullval, buffer, &anynull, &status);
        if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    }
    // check
    if (this->DebugCode > 2) {
        std::cout << "checking gsl_matrix_get(this->SourcePlaneDataCube[0], 0, 0:10)"; 
        for (i=0; i<10; i++) 
            std::cout << " " << gsl_matrix_get(this->SourcePlaneDataCube[0], 0, i);
        std::cout << std::endl;
        std::cout << "checking gsl_matrix_get(this->SourcePlaneDataCube[1], 0, 0:10)";
        for (i=0; i<10; i++) 
            std::cout << " " << gsl_matrix_get(this->SourcePlaneDataCube[1], 0, i);
        std::cout << std::endl;
        std::cout << "checking gsl_matrix_get(this->SourcePlaneDataCube[1], 1, 0:10)";
        for (i=0; i<10; i++) 
            std::cout << " " << gsl_matrix_get(this->SourcePlaneDataCube[1], 1, i);
        std::cout << std::endl;
    }
    // clean up
    if (this->DebugCode > 0) 
        std::cout << "cleaning up" << std::endl;
    delete naxes;
    buffer = NULL;
    // close file
    this->ErrorCode = fits_close_file(fptr, &status);
    if (this->ErrorCode != 0) {
        fits_report_error(stderr, status);
        std::cerr << "Error! Failed to open the FITS file \"" << FilePath << "\"! Please check error messages above." << std::endl;
    }
    if (this->DebugCode > 0) 
        std::cout << "readSourcePlaneDataCube finished." << std::endl;
}


void LensingTransformer::readSourcePlaneDataCube(double *data, long nx, long ny, long nchan, double ra, double dec, double pixelsize, double cenx, double ceny, int verbose)
{
    /* Read source plane data cube from data array.
     */
    if (this->DebugCode > 0) 
        std::cout << "readSourcePlaneDataCube is called." << std::endl;
    // Check input values
    if (pixelsize<=0.0) {
        this->ErrorCode = 255; 
        std::cerr << "Error! pixelsize " << pixelsize << " is not a positive float number for readSourcePlaneDataCube()." << std::endl;
        return;
    }
    // Clean up previous loads
    long ichan = 0;
    if (this->SourcePlaneDataCube.size()>0) {
        for (ichan=0; ichan<(long)(this->SourcePlaneDataCube.size()); ichan++) {
            gsl_matrix_free(this->SourcePlaneDataCube[ichan]);
        }
    }
    this->SourcePlaneDataCube.clear();
    // Store input values
    this->SourcePlaneCenterRA = ra;
    this->SourcePlaneCenterDec = dec;
    this->SourcePlanePixelSize = pixelsize;
    // Store the so-called image center, which is actually the reference pixel
    if (std::isnan(cenx)) {
        this->SourcePlaneCenterX = (double(nx)+1.0)/2.0;
    } else {
        this->SourcePlaneCenterX = cenx;
    }
    if (std::isnan(ceny)) {
        this->SourcePlaneCenterY = (double(ny)+1.0)/2.0;
    } else {
        this->SourcePlaneCenterY = ceny;
    }
    // Load data array
    long i=0, j=0;
    for (ichan=0; ichan<nchan; ichan++) {
        this->SourcePlaneDataCube.push_back(gsl_matrix_alloc(ny, nx));
        for (j=0; j<ny; j++) {
            for (i=0; i<nx; i++) {
                gsl_matrix_set(this->SourcePlaneDataCube[ichan], j, i, *(data+ichan*ny*nx+j*nx+i));
            }
        }
    }
    if (this->DebugCode > 0) 
        std::cout << "readSourcePlaneDataCube finished." << std::endl;
}


void LensingTransformer::linkSourcePlaneToImagePlane(double ra, double dec, double pixelsize, long sizex, long sizey, double cenx, double ceny, int verbose)
{
    /* Link source plane to image plane.
    
    Inputs are for the image plane.
     */
    if (this->DebugCode > 0) 
        std::cout << "linkSourcePlaneToImagePlane is called." << std::endl;
    // Check input values
    if (this->SourcePlaneDataCube.size()==0) {
        this->ErrorCode = 255; 
        std::cerr << "Error! SourcePlaneDataCube is not initialized before calling linkSourcePlaneToImagePlane()." << std::endl;
        return;
    }
    if (std::isnan(this->SourcePlanePixelSize)) {
        this->ErrorCode = 255; 
        std::cerr << "Error! SourcePlanePixelSize is not initialized before calling linkSourcePlaneToImagePlane()." << std::endl;
        return;
    }
    if (sizex<4) {
        this->ErrorCode = 255; 
        std::cerr << "Error! sizex " << sizex << " is not a positive long integer or too small for linkSourcePlaneToImagePlane()." << std::endl;
        return;
    }
    if (sizey<4) {
        this->ErrorCode = 255; 
        std::cerr << "Error! sizey " << sizey << " is not a positive long integer or too small for linkSourcePlaneToImagePlane()." << std::endl;
        return;
    }
    if (pixelsize<=0.0) {
        this->ErrorCode = 255; 
        std::cerr << "Error! pixelsize " << pixelsize << " is not a positive float number for linkSourcePlaneToImagePlane()." << std::endl;
        return;
    }
    /*
    if ((!this->DeflectionX)||(!this->DeflectionY)) {
        this->ErrorCode = 255; 
        std::cerr << "Error! DeflectionX and DeflectionY are not initialized before calling linkSourcePlaneToImagePlane()." << std::endl;
        return;
    }
    */
    if ((!this->SourcePlaneMeshGridX1) || (!this->SourcePlaneMeshGridY1) || (!this->SourcePlaneMeshGridX2) || (!this->SourcePlaneMeshGridY2) || (!this->ImagePlaneMeshGridX1) || (!this->ImagePlaneMeshGridY1) || (!this->ImagePlaneMeshGridX2) || (!this->ImagePlaneMeshGridY2)) {
        this->ErrorCode = 255; 
        std::cerr << "Error! SourcePlaneMeshGrid and ImagePlaneMeshGrid are not initialized before calling linkSourcePlaneToImagePlane()." << std::endl;
        return;
    }
    unsigned long nmesh = this->SourcePlaneMeshGridX1->size; 
    if (nmesh < 4) {
        this->ErrorCode = 255; 
        std::cerr << "Error! SourcePlaneMeshGrid and ImagePlaneMeshGrid do not have enough cells for linkSourcePlaneToImagePlane()." << std::endl;
        return;
    }
    if ((this->SourcePlaneMeshGridX1->size != nmesh) || (this->SourcePlaneMeshGridY1->size != nmesh) || (this->SourcePlaneMeshGridX2->size != nmesh) || (this->SourcePlaneMeshGridY2->size != nmesh) || (this->ImagePlaneMeshGridX1->size != nmesh) || (this->ImagePlaneMeshGridY1->size != nmesh) || (this->ImagePlaneMeshGridX2->size != nmesh) || (this->ImagePlaneMeshGridY2->size != nmesh)) {
        this->ErrorCode = 255; 
        std::cerr << "Error! SourcePlaneMeshGrid and ImagePlaneMeshGrid do not have the same number of cells for linkSourcePlaneToImagePlane()." << std::endl;
        return;
    }
    // 
    // Store input values
    this->ImagePlaneCenterRA = ra;
    this->ImagePlaneCenterDec = dec;
    if (std::isnan(cenx)) {
        this->ImagePlaneCenterX = (double(sizex)+1.0)/2.0;
    } else {
        this->ImagePlaneCenterX = cenx;
    }
    if (std::isnan(ceny)) {
        this->ImagePlaneCenterY = (double(sizey)+1.0)/2.0;
    } else {
        this->ImagePlaneCenterY = ceny;
    }
    this->ImagePlanePixelSize = pixelsize;
    // 
    // Initialize the SourcePlane2D array, where each pixel's memory address will be linked to the ImagePlane2D pointer array's elements
    long i=0, j=0;
    const long nx = this->SourcePlaneDataCube[0]->size2; // source plane sizex
    const long ny = this->SourcePlaneDataCube[0]->size1; // source plane sizey. The number of rows is size1. See https://www.gnu.org/software/gsl/doc/html/vectors.html
    //this->SourcePlane2D = new double*[ny];
    //long k = long(this->SourcePlaneDataCube.size()-1);
    //for(j=0; j<ny; j++) {
    //    this->SourcePlane2D[j] = new double[nx];
    //    for (i=0; i<nx; i++) {
    //        this->SourcePlane2D[j][i] = gsl_matrix_get(this->SourcePlaneDataCube[k], j, i); // use the central frame k
    //    }
    //}
    this->SourcePlane2D = gsl_matrix_alloc(ny, nx);
    // 
    // Find min max and resolution of the mesh grid
    double xsmin = std::min(gsl_vector_min(this->SourcePlaneMeshGridX1), gsl_vector_min(this->SourcePlaneMeshGridX2));
    double xsmax = std::max(gsl_vector_max(this->SourcePlaneMeshGridX1), gsl_vector_max(this->SourcePlaneMeshGridX2));
    double ysmin = std::min(gsl_vector_min(this->SourcePlaneMeshGridY1), gsl_vector_min(this->SourcePlaneMeshGridY2));
    double ysmax = std::max(gsl_vector_max(this->SourcePlaneMeshGridY1), gsl_vector_max(this->SourcePlaneMeshGridY2));
    double ximin = std::min(gsl_vector_min(this->ImagePlaneMeshGridX1), gsl_vector_min(this->ImagePlaneMeshGridX2));
    double ximax = std::max(gsl_vector_max(this->ImagePlaneMeshGridX1), gsl_vector_max(this->ImagePlaneMeshGridX2));
    double yimin = std::min(gsl_vector_min(this->ImagePlaneMeshGridY1), gsl_vector_min(this->ImagePlaneMeshGridY2));
    double yimax = std::max(gsl_vector_max(this->ImagePlaneMeshGridY1), gsl_vector_max(this->ImagePlaneMeshGridY2));
    double xires = std::min(\
        std::min(\
            std::fabs(this->ImagePlaneMeshGridX1->data[0] - this->ImagePlaneMeshGridX2->data[0]),\
            std::fabs(this->ImagePlaneMeshGridX1->data[nmesh-1] - this->ImagePlaneMeshGridX2->data[nmesh-1])\
            ),\
        std::min(\
            std::fabs(this->ImagePlaneMeshGridX1->data[1] - this->ImagePlaneMeshGridX2->data[1]),\
            std::fabs(this->ImagePlaneMeshGridX1->data[nmesh-2] - this->ImagePlaneMeshGridX2->data[nmesh-2])\
            )\
        );
    double yires = std::min(\
        std::min(\
            std::fabs(this->ImagePlaneMeshGridY1->data[0] - this->ImagePlaneMeshGridY2->data[0]),\
            std::fabs(this->ImagePlaneMeshGridY1->data[nmesh-1] - this->ImagePlaneMeshGridY2->data[nmesh-1])\
            ),\
        std::min(\
            std::fabs(this->ImagePlaneMeshGridY1->data[1] - this->ImagePlaneMeshGridY2->data[1]),\
            std::fabs(this->ImagePlaneMeshGridY1->data[nmesh-2] - this->ImagePlaneMeshGridY2->data[nmesh-2])\
            )\
        );
    double xsres = std::min(\
        std::min(\
            std::fabs(this->SourcePlaneMeshGridX1->data[0] - this->SourcePlaneMeshGridX2->data[0]),\
            std::fabs(this->SourcePlaneMeshGridX1->data[nmesh-1] - this->SourcePlaneMeshGridX2->data[nmesh-1])\
            ),\
        std::min(\
            std::fabs(this->SourcePlaneMeshGridX1->data[1] - this->SourcePlaneMeshGridX2->data[1]),\
            std::fabs(this->SourcePlaneMeshGridX1->data[nmesh-2] - this->SourcePlaneMeshGridX2->data[nmesh-2])\
            )\
        );
    double ysres = std::min(\
        std::min(\
            std::fabs(this->SourcePlaneMeshGridY1->data[0] - this->SourcePlaneMeshGridY2->data[0]),\
            std::fabs(this->SourcePlaneMeshGridY1->data[nmesh-1] - this->SourcePlaneMeshGridY2->data[nmesh-1])\
            ),\
        std::min(\
            std::fabs(this->SourcePlaneMeshGridY1->data[1] - this->SourcePlaneMeshGridY2->data[1]),\
            std::fabs(this->SourcePlaneMeshGridY1->data[nmesh-2] - this->SourcePlaneMeshGridY2->data[nmesh-2])\
            )\
        );
    if (this->DebugCode > 0) {
        std::cout << "source plane data cube pixel size " << this->SourcePlanePixelSize << std::endl;
        std::cout << "source plane mesh grid field of view: xmin " << xsmin << " xmax " << xsmax << " ymin " << ysmin << " ymax " << ysmax << std::endl;
        std::cout << "image plane mesh grid field of view: xmin " << ximin << " xmax " << ximax << " ymin " << yimin << " ymax " << yimax << std::endl;
        std::cout << "source plane mesh grid resolution in arcsec unit: x " << xsres << " y " << ysres << std::endl;
        std::cout << "image plane mesh grid resolution in arcsec unit: x " << xires << " y " << yires << std::endl;
    }
    // 
    // Initialize ImagePlane2D
    for (j=0; j<sizey; j++) {
        std::vector<double *> this_row;
        for (i=0; i<sizex; i++) {
            // find the image plane mesh grid that contains this pixel
            this_row.push_back(NULL);
        }
        this->ImagePlane2D.push_back(this_row); 
    }
    // 
    // Find the corresponding reference pixel for the len model RA Dec
    double dRA_in_arcsec, dDec_in_arcsec, dRA_in_pixel, dDec_in_pixel;
    dRA_in_arcsec = -(this->LensModelCenterRA - this->ImagePlaneCenterRA) * std::cos(this->ImagePlaneCenterDec * 4.0 * std::atan(1.0) / 180.0) * 3600.0; // deg2rad(degrees) == degrees * 4.0 * atan(1.0) / 180.0
    dDec_in_arcsec = (this->LensModelCenterDec - this->ImagePlaneCenterDec) * 3600.0;
    dRA_in_pixel = dRA_in_arcsec / this->ImagePlanePixelSize;
    dDec_in_pixel = dDec_in_arcsec / this->ImagePlanePixelSize;
    this->LensModelCenterInImagePlaneX = dRA_in_pixel + this->ImagePlaneCenterX;
    this->LensModelCenterInImagePlaneY = dDec_in_pixel + this->ImagePlaneCenterY;
    // 
    dRA_in_arcsec = -(this->LensModelCenterRA - this->SourcePlaneCenterRA) * std::cos(this->SourcePlaneCenterDec * 4.0 * std::atan(1.0) / 180.0) * 3600.0; // deg2rad(degrees) == degrees * 4.0 * atan(1.0) / 180.0
    dDec_in_arcsec = (this->LensModelCenterDec - this->SourcePlaneCenterDec) * 3600.0;
    dRA_in_pixel = dRA_in_arcsec / this->SourcePlanePixelSize;
    dDec_in_pixel = dDec_in_arcsec / this->SourcePlanePixelSize;
    this->LensModelCenterInSourcePlaneX = dRA_in_pixel + this->SourcePlaneCenterX;
    this->LensModelCenterInSourcePlaneY = dDec_in_pixel + this->SourcePlaneCenterY;
    if (this->DebugCode > 0) {
        std::cout << "lens model center RA Dec: " << this->LensModelCenterRA << " " << this->LensModelCenterDec << ", image plane center RA Dec: " << this->ImagePlaneCenterRA << " " << this->ImagePlaneCenterDec << std::endl;
        std::cout << "lens model center in image plane: x " << this->LensModelCenterInImagePlaneX << " y " << this->LensModelCenterInImagePlaneY << std::endl;
        std::cout << "lens model center in source plane: x " << this->LensModelCenterInSourcePlaneX << " y " << this->LensModelCenterInSourcePlaneY << std::endl;
    }
    //
    // Link each pixel in the ImagePlane2D
    long imesh = 0, NumMeshUsed = 0; 
    MeshGridCell cellim, cellsr; // image plane and source plane mesh grid cell 
    for (imesh=0; imesh<(long)(nmesh); imesh+=4) {
        cellim.x11 = (this->ImagePlaneMeshGridX1->data[imesh+0] / this->ImagePlanePixelSize) + this->LensModelCenterInImagePlaneX; 
        cellim.y11 = (this->ImagePlaneMeshGridY1->data[imesh+0] / this->ImagePlanePixelSize) + this->LensModelCenterInImagePlaneY; 
        cellim.x12 = (this->ImagePlaneMeshGridX1->data[imesh+1] / this->ImagePlanePixelSize) + this->LensModelCenterInImagePlaneX; 
        cellim.y12 = (this->ImagePlaneMeshGridY1->data[imesh+1] / this->ImagePlanePixelSize) + this->LensModelCenterInImagePlaneY; 
        cellim.x22 = (this->ImagePlaneMeshGridX1->data[imesh+2] / this->ImagePlanePixelSize) + this->LensModelCenterInImagePlaneX; 
        cellim.y22 = (this->ImagePlaneMeshGridY1->data[imesh+2] / this->ImagePlanePixelSize) + this->LensModelCenterInImagePlaneY; 
        cellim.x21 = (this->ImagePlaneMeshGridX1->data[imesh+3] / this->ImagePlanePixelSize) + this->LensModelCenterInImagePlaneX; 
        cellim.y21 = (this->ImagePlaneMeshGridY1->data[imesh+3] / this->ImagePlanePixelSize) + this->LensModelCenterInImagePlaneY;
        cellim.xcen = (cellim.x11 + cellim.x12 + cellim.x22 + cellim.x21) / 4.0;
        cellim.ycen = (cellim.y11 + cellim.y12 + cellim.y22 + cellim.y21) / 4.0;
        cellim.i11 = long(cellim.x11-1.0); // lower left x index, 0-based, floor round
        cellim.j11 = long(cellim.y11-1.0); // lower left y index, 0-based, floor round
        cellim.i12 = long(cellim.x12-0.0); // lower right x index, 0-based, ceil round
        cellim.j12 = long(cellim.y12-1.0); // lower right y index, 0-based, floor round
        cellim.i22 = long(cellim.x22-0.0); // upper right x index, 0-based, ceil round
        cellim.j22 = long(cellim.y22-0.0); // upper right y index, 0-based, ceil round
        cellim.i21 = long(cellim.x21-1.0); // upper left x index, 0-based, floor round
        cellim.j21 = long(cellim.y21-0.0); // upper left y index, 0-based, ceil round
        cellim.icen = long(cellim.xcen-0.5); // center x index, 0-based, round
        cellim.jcen = long(cellim.ycen-0.5); // center y index, 0-based, round
        if (this->DebugCode > 2) {
            std::cout << "imesh " << imesh << " image plane ";
            std::cout << this->ImagePlaneMeshGridX1->data[imesh+0] << " " << this->ImagePlaneMeshGridY1->data[imesh+0] << " ";
            std::cout << this->ImagePlaneMeshGridX1->data[imesh+1] << " " << this->ImagePlaneMeshGridY1->data[imesh+1] << " ";
            std::cout << this->ImagePlaneMeshGridX1->data[imesh+2] << " " << this->ImagePlaneMeshGridY1->data[imesh+2] << " ";
            std::cout << this->ImagePlaneMeshGridX1->data[imesh+3] << " " << this->ImagePlaneMeshGridY1->data[imesh+3] << " ";
            std::cout << " lower left " << cellim.i11 << " " << cellim.j11 << " lower right " << cellim.i12 << " " << cellim.j12;
            std::cout << " upper right " << cellim.i22 << " " << cellim.j22 << " upper left " << cellim.i21 << " " << cellim.j21 << std::endl;
        }
        // if grid mesh cell is not in the image plane as defined by the input arguments, skip this cell
        if ((cellim.i11 < 0) || (cellim.i11 > sizex-1) || (cellim.j11 < 0) || (cellim.j11 > sizey-1) || (cellim.i12 < 0) || (cellim.i12 > sizex-1) || (cellim.j12 < 0) || (cellim.j12 > sizey-1) || (cellim.i22 < 0) || (cellim.i22 > sizex-1) || (cellim.j22 < 0) || (cellim.j22 > sizey-1) || (cellim.i21 < 0) || (cellim.i21 > sizex-1) || (cellim.j21 < 0) || (cellim.j21 > sizey-1)) {
            if (this->DebugCode > 2) {
                if (imesh > 10) {
                    std::cout << "returning after 10 mesh processed." << std::endl;
                    return;
                }
            }
            continue;
        }
        // convert mesh grid xy offset (in arcsec unit) to pixel
        cellsr.x11 = (this->SourcePlaneMeshGridX1->data[imesh+0] / this->SourcePlanePixelSize) + this->LensModelCenterInSourcePlaneX; 
        cellsr.y11 = (this->SourcePlaneMeshGridY1->data[imesh+0] / this->SourcePlanePixelSize) + this->LensModelCenterInSourcePlaneY;
        cellsr.x12 = (this->SourcePlaneMeshGridX1->data[imesh+1] / this->SourcePlanePixelSize) + this->LensModelCenterInSourcePlaneX;
        cellsr.y12 = (this->SourcePlaneMeshGridY1->data[imesh+1] / this->SourcePlanePixelSize) + this->LensModelCenterInSourcePlaneY;
        cellsr.x22 = (this->SourcePlaneMeshGridX1->data[imesh+2] / this->SourcePlanePixelSize) + this->LensModelCenterInSourcePlaneX;
        cellsr.y22 = (this->SourcePlaneMeshGridY1->data[imesh+2] / this->SourcePlanePixelSize) + this->LensModelCenterInSourcePlaneY;
        cellsr.x21 = (this->SourcePlaneMeshGridX1->data[imesh+3] / this->SourcePlanePixelSize) + this->LensModelCenterInSourcePlaneX;
        cellsr.y21 = (this->SourcePlaneMeshGridY1->data[imesh+3] / this->SourcePlanePixelSize) + this->LensModelCenterInSourcePlaneY;
        cellsr.xcen = (cellsr.x11 + cellsr.x12 + cellsr.x22 + cellsr.x21) / 4.0;
        cellsr.ycen = (cellsr.y11 + cellsr.y12 + cellsr.y22 + cellsr.y21) / 4.0;
        cellsr.i11 = long(cellsr.x11-1.0); // lower left x index, 0-based, floor round
        cellsr.j11 = long(cellsr.y11-1.0); // lower left y index, 0-based, floor round
        cellsr.i12 = long(cellsr.x12-0.0); // lower right x index, 0-based, ceil round
        cellsr.j12 = long(cellsr.y12-1.0); // lower right y index, 0-based, floor round
        cellsr.i22 = long(cellsr.x22-0.0); // upper right x index, 0-based, ceil round
        cellsr.j22 = long(cellsr.y22-0.0); // upper right y index, 0-based, ceil round
        cellsr.i21 = long(cellsr.x21-1.0); // upper left x index, 0-based, floor round
        cellsr.j21 = long(cellsr.y21-0.0); // upper left y index, 0-based, ceil round
        cellsr.icen = long(cellsr.xcen-0.5); // center x index, 0-based, round
        cellsr.jcen = long(cellsr.ycen-0.5); // center y index, 0-based, round
        // if grid mesh cell is not in the source plane data cube as loaded previously by the function readSourcePlaneDataCube, skip this cell
        if ((cellsr.icen < 0) || (cellsr.icen > nx-1) || (cellsr.jcen < 0) || (cellsr.jcen > ny-1)) {
            if (this->DebugCode > 2) {
                if (imesh > 10) {
                    std::cout << "returning after 10 mesh processed." << std::endl;
                    return;
                }
            }
            continue;
        }
        if (this->DebugCode > 2) {
            long lenimesh = long(std::floor(std::log10(std::abs((long)(nmesh)))))+1;
            std::cout << "imesh " << imesh << " image plane lower left, lower right, upper right, upper left:" << std::endl;
            std::cout << "      " << std::setw(lenimesh) << " " << "[arcsec] (";
            std::cout << this->ImagePlaneMeshGridX1->data[imesh+0] << "," << this->ImagePlaneMeshGridY1->data[imesh+0] << "), (";
            std::cout << this->ImagePlaneMeshGridX1->data[imesh+1] << "," << this->ImagePlaneMeshGridY1->data[imesh+1] << "), (";
            std::cout << this->ImagePlaneMeshGridX1->data[imesh+2] << "," << this->ImagePlaneMeshGridY1->data[imesh+2] << "), (";
            std::cout << this->ImagePlaneMeshGridX1->data[imesh+3] << "," << this->ImagePlaneMeshGridY1->data[imesh+3] << ")" << std::endl;
            std::cout << "      " << std::setw(lenimesh) << " " << "[pixel]  (";
            std::cout << cellim.x11 << "," << cellim.y11 << "), (";
            std::cout << cellim.x12 << "," << cellim.y12 << "), (";
            std::cout << cellim.x22 << "," << cellim.y22 << "), (";
            std::cout << cellim.x21 << "," << cellim.y21 << ")" << std::endl;
            std::cout << "      " << std::setw(lenimesh) << " " << "[index]  (";
            std::cout << cellim.i11 << "," << cellim.j11 << "), (";
            std::cout << cellim.i12 << "," << cellim.j12 << "), (";
            std::cout << cellim.i22 << "," << cellim.j22 << "), (";
            std::cout << cellim.i21 << "," << cellim.j21 << ")" << std::endl;
            //
            std::cout << "imesh " << imesh << " source plane lower left, lower right, upper right, upper left:" << std::endl;
            std::cout << "      " << std::setw(lenimesh) << " " << "[arcsec] (";
            std::cout << this->SourcePlaneMeshGridX1->data[imesh+0] << "," << this->SourcePlaneMeshGridY1->data[imesh+0] << "), (";
            std::cout << this->SourcePlaneMeshGridX1->data[imesh+1] << "," << this->SourcePlaneMeshGridY1->data[imesh+1] << "), (";
            std::cout << this->SourcePlaneMeshGridX1->data[imesh+2] << "," << this->SourcePlaneMeshGridY1->data[imesh+2] << "), (";
            std::cout << this->SourcePlaneMeshGridX1->data[imesh+3] << "," << this->SourcePlaneMeshGridY1->data[imesh+3] << ")" << std::endl;
            std::cout << "      " << std::setw(lenimesh) << " " << "[pixel]  (";
            std::cout << cellsr.x11 << "," << cellsr.y11 << "), (";
            std::cout << cellsr.x12 << "," << cellsr.y12 << "), (";
            std::cout << cellsr.x22 << "," << cellsr.y22 << "), (";
            std::cout << cellsr.x21 << "," << cellsr.y21 << ")" << std::endl;
            std::cout << "      " << std::setw(lenimesh) << " " << "[index]  (";
            std::cout << cellsr.i11 << "," << cellsr.j11 << "), (";
            std::cout << cellsr.i12 << "," << cellsr.j12 << "), (";
            std::cout << cellsr.i22 << "," << cellsr.j22 << "), (";
            std::cout << cellsr.i21 << "," << cellsr.j21 << ")" << std::endl;
        }
        // map the pixel from source plane to image plane
        for (j = std::min(cellim.j11, cellim.j12); j <= std::max(cellim.j21, cellim.j22); j++) {
            for (i = std::min(cellim.i11, cellim.i21); i <= std::max(cellim.i12, cellim.i22); i++) {
                // in case the mesh grid is not rectange, we check if this pixel is exactly in the mesh grid 
                // TODO: this can be speed up if we are sure that the image plane mesh grid is rectangle
                //if (checkPointInMeshGridCell((double)i, (double)j, cellim)) {
                    if (this->DebugCode > 2) {
                        std::cout << "imesh " << imesh << " mapping source plane pixel " << cellsr.icen << " " << cellsr.jcen << " to image plane pixel " << i << " " << j << std::endl;
                    }
                    this->ImagePlane2D[j][i] = &(this->SourcePlane2D->data[cellsr.jcen * nx + cellsr.icen]);
                    // if the source plane mesh grid cell has more than one source plane pixels
                    if ((cellsr.i11 != cellsr.icen) || (cellsr.i22 != cellsr.icen) || (cellsr.j11 != cellsr.jcen) || (cellsr.j22 != cellsr.jcen)) {
                        // subgrid-level pixel mapping 
                        double cellim_subgrid_fy1 = (double(j+1) - cellim.y11) / (cellim.y21 - cellim.y11); // y fraction at left side
                        double cellim_subgrid_fy2 = (double(j+1) - cellim.y12) / (cellim.y22 - cellim.y12); // y fraction at right side
                        double cellim_subgrid_fx1 = (double(i+1) - cellim.x11) / (cellim.x12 - cellim.x11); // x fraction at bottom side
                        double cellim_subgrid_fx2 = (double(i+1) - cellim.x21) / (cellim.x22 - cellim.x21); // x fraction at top side
                        double cellim_subgrid_fy = (cellim_subgrid_fy1 + cellim_subgrid_fy2) / 2.0;
                        double cellim_subgrid_fx = (cellim_subgrid_fx1 + cellim_subgrid_fx2) / 2.0;
                        //double cellsr_subgrid_fy1 = (cellsr.ycen - cellsr.y11) / (cellsr.y21 - cellsr.y11); // y fraction at left side
                        //double cellsr_subgrid_fy2 = (cellsr.ycen - cellsr.y12) / (cellsr.y22 - cellsr.y12); // y fraction at right side
                        //double cellsr_subgrid_fx1 = (cellsr.xcen - cellsr.x11) / (cellsr.x12 - cellsr.x11); // x fraction at bottom side
                        //double cellsr_subgrid_fx2 = (cellsr.xcen - cellsr.x21) / (cellsr.x22 - cellsr.x21); // x fraction at top side
                        //double cellsr_subgrid_fy = (cellsr_subgrid_fy1 + cellsr_subgrid_fy2) / 2.0;
                        //double cellsr_subgrid_fx = (cellsr_subgrid_fx1 + cellsr_subgrid_fx2) / 2.0;
                        /*
                           Equation for subgrid fractional y in source plane: 
                             (cellsr_subgrid_y - cellsr.y11)  / (cellsr.y21 - cellsr.y11) + (cellsr_subgrid_y - cellsr.y12) / (cellsr.y22 - cellsr.y12) = 2.0 * cellim_subgrid_fy
                           
                           and that for subgrid fractional y in source plane:
                             (cellsr_subgrid_x - cellsr.x11)  / (cellsr.x12 - cellsr.x11) + (cellsr_subgrid_x - cellsr.x21) / (cellsr.x22 - cellsr.x21) = 2.0 * cellim_subgrid_fx
                           
                           Sovling cellsr_subgrid_y:
                             (cellsr_subgrid_y - cellsr.y11) * (cellsr.y22 - cellsr.y12) + (cellsr_subgrid_y - cellsr.y12) * (cellsr.y21 - cellsr.y11) 
                               = 2.0 * cellim_subgrid_fy * (cellsr.y21 - cellsr.y11) * (cellsr.y22 - cellsr.y12)
                             
                             cellsr_subgrid_y * ( (cellsr.y22 - cellsr.y12) + (cellsr.y21 - cellsr.y11) ) 
                               - cellsr.y11 * (cellsr.y22 - cellsr.y12) - cellsr.y12 * (cellsr.y21 - cellsr.y11) 
                                 = 2.0 * cellim_subgrid_fy * (cellsr.y21 - cellsr.y11) * (cellsr.y22 - cellsr.y12)
                           
                             cellsr_subgrid_y \
                               = ( 2.0 * cellim_subgrid_fy * (cellsr.y21 - cellsr.y11) * (cellsr.y22 - cellsr.y12) \
                                   + cellsr.y11 * (cellsr.y22 - cellsr.y12) + cellsr.y12 * (cellsr.y21 - cellsr.y11) \
                                 ) / ( (cellsr.y22 - cellsr.y12) + (cellsr.y21 - cellsr.y11) );
                           
                           Sovling cellsr_subgrid_x:
                             (cellsr_subgrid_x - cellsr.x11) * (cellsr.x22 - cellsr.x21) + (cellsr_subgrid_x - cellsr.x21) * (cellsr.x12 - cellsr.x11) = 2.0 * cellim_subgrid_fx * (cellsr.x12 - cellsr.x11) * (cellsr.x22 - cellsr.x21)
                             
                             cellsr_subgrid_x 
                               = ( 2.0 * cellim_subgrid_fx * (cellsr.x12 - cellsr.x11) * (cellsr.x22 - cellsr.x21) \
                                   + cellsr.x11 * (cellsr.x22 - cellsr.x21) + cellsr.x21 * (cellsr.x12 - cellsr.x11) \
                                 ) / ( (cellsr.x22 - cellsr.x21) + (cellsr.x12 - cellsr.x11) );
                        */
                        double cellsr_subgrid_y \
                               = ( 2.0 * cellim_subgrid_fy * (cellsr.y21 - cellsr.y11) * (cellsr.y22 - cellsr.y12) \
                                   + cellsr.y11 * (cellsr.y22 - cellsr.y12) + cellsr.y12 * (cellsr.y21 - cellsr.y11) \
                                 ) / ( (cellsr.y22 - cellsr.y12) + (cellsr.y21 - cellsr.y11) );
                        
                        double cellsr_subgrid_x \
                               = ( 2.0 * cellim_subgrid_fx * (cellsr.x12 - cellsr.x11) * (cellsr.x22 - cellsr.x21) \
                                   + cellsr.x11 * (cellsr.x22 - cellsr.x21) + cellsr.x21 * (cellsr.x12 - cellsr.x11) \
                                 ) / ( (cellsr.x22 - cellsr.x21) + (cellsr.x12 - cellsr.x11) );
                        // 
                        if (this->DebugCode > 2) {
                            std::cout << "imesh " << imesh << " subgrid mapping source plane pixel " << cellsr_subgrid_x << " " << cellsr_subgrid_y << " to image plane fractional " << cellim_subgrid_fx << " " << cellim_subgrid_fy << std::endl;
                        }
                        // 
                        long cellsr_subgrid_i = long(std::round(cellsr_subgrid_x)-1);
                        long cellsr_subgrid_j = long(std::round(cellsr_subgrid_y)-1);
                        while (cellsr_subgrid_i < 0) { cellsr_subgrid_i++; }
                        while (cellsr_subgrid_i > nx-1) { cellsr_subgrid_i--; }
                        while (cellsr_subgrid_j < 0) { cellsr_subgrid_j++; }
                        while (cellsr_subgrid_j > ny-1) { cellsr_subgrid_j--; }
                        this->ImagePlane2D[j][i] = &(this->SourcePlane2D->data[cellsr_subgrid_j * nx + cellsr_subgrid_i]);
                        // 
                        if (this->DebugCode > 2) {
                            std::cout << "imesh " << imesh << " subgrid mapping source plane pixel " << cellsr_subgrid_i << " " << cellsr_subgrid_j << " to image plane pixel " << i << " " << j << std::endl;
                        }
                    }
                //}
            } // end for i
        } // end for j
        // 
        NumMeshUsed++;
        //
        if (this->DebugCode > 2) {
            if (NumMeshUsed > 10) {
                std::cout << "returning after 10 mesh calculated." << std::endl;
                return;
            }
        }
    }
    //
    if (this->DebugCode > 0) 
        std::cout << "linkSourcePlaneToImagePlane finished." << std::endl;
}



void LensingTransformer::writeImagePlaneDataCube(std::string FilePath, int overwrite, int verbose)
{
    /* Dereference ImagePlane2D and repeat for each channel then save the ImagePlane 3D data cube.
     */
    if (this->DebugCode > 0) 
        std::cout << "writeImagePlaneDataCube is called." << std::endl;
    // Check file existence.
    if ((access(FilePath.c_str(), F_OK) == 0) && (overwrite <= 0) ) {
        this->ErrorCode = 0;
        std::cout << "Found existing file \"" << FilePath << "\" and overwrite is not set. Doing nothing." << std::endl;
        return;
    }
    // Check necessary data
    if (this->SourcePlaneDataCube.size()==0) {
        this->ErrorCode = 255; 
        std::cerr << "Error! SourcePlaneDataCube is not initialized before calling writeImagePlaneDataCube()." << std::endl;
        return;
    }
    if (!this->SourcePlane2D) {
        this->ErrorCode = 255; 
        std::cerr << "Error! SourcePlane2D is not initialized before calling writeImagePlaneDataCube(). Please call linkSourcePlaneToImagePlane() first." << std::endl;
        return;
    }
    if (this->ImagePlane2D.size()==0) {
        this->ErrorCode = 255; 
        std::cerr << "Error! ImagePlane2D is not initialized before calling writeImagePlaneDataCube(). Please call linkSourcePlaneToImagePlane() first." << std::endl;
        return;
    }
    if (this->ImagePlane2D[0].size()==0) {
        this->ErrorCode = 255; 
        std::cerr << "Error! ImagePlane2D is not initialized before calling writeImagePlaneDataCube(). Please call linkSourcePlaneToImagePlane() first." << std::endl;
        return;
    }
    // Get source plane data cube 3D dimension
    const long nchan = this->SourcePlaneDataCube.size(); // cube channel number
    //const long nx = this->SourcePlaneDataCube[0]->size2; // source plane sizex
    //const long ny = this->SourcePlaneDataCube[0]->size1; // source plane sizey. The number of rows is size1. See https://www.gnu.org/software/gsl/doc/html/vectors.html
    // Get image plane data 2D dimension, which is set by the input arguments when calling linkSourcePlaneToImagePlane()
    const long sizey = this->ImagePlane2D.size();
    const long sizex = this->ImagePlane2D[0].size();
    // Open FITS file with cfitsio.
    fitsfile *fptr = NULL;
    long i = 0, j = 0, ichan = 0;
    int status = 0, datatype = 0, bitpix = -64, naxis = 3, decimals = 12;
    long naxes[3] = {sizex, sizey, nchan};
    long nbuffer = 0, fpixel = 0; // first pixel's index 
    double *buffer = NULL;
    // create file
    if (verbose > 0) {
        std::cout << "writeImagePlaneDataCube() opening file for writing: \"" << FilePath << "\"" << std::endl;
    }
    if (access(FilePath.c_str(), F_OK) == 0) {
        this->ErrorCode = fits_create_file(&fptr, ("!"+FilePath).c_str(), &status); // prepending "!" means overwriting
    } else {
        this->ErrorCode = fits_create_file(&fptr, (FilePath).c_str(), &status); // 
    }
    if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    // create image header
    if (this->DebugCode > 0) { std::cout << "fits_create_img" << std::endl; }
    this->ErrorCode = fits_create_img(fptr, bitpix, naxis, naxes, &status);
    if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    // write wcs header
    if (this->DebugCode > 0) { std::cout << "fits_write_key_str" << std::endl; }
    this->ErrorCode = fits_write_key_str(fptr, "RADESYS", "FK5", "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_str(fptr, "SPECSYS", "TOPOCENT", "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_flt(fptr, "EQUINOX", 2000.000, 3, "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_str(fptr, "CTYPE1", "RA---TAN", "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_str(fptr, "CTYPE2", "DEC--TAN", "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_str(fptr, "CTYPE3", "", "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_str(fptr, "CUNIT1", "deg", "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_str(fptr, "CUNIT2", "deg", "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_str(fptr, "CUNIT3", "", "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_dbl(fptr, "CRPIX1", this->ImagePlaneCenterX, decimals, "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_dbl(fptr, "CRPIX2", this->ImagePlaneCenterY, decimals, "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_dbl(fptr, "CRPIX3", 1.0, decimals, "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_dbl(fptr, "CRVAL1", this->ImagePlaneCenterRA, decimals, "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_dbl(fptr, "CRVAL2", this->ImagePlaneCenterDec, decimals, "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_dbl(fptr, "CRVAL3", 1.0, decimals, "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_dbl(fptr, "CDELT1", -this->ImagePlanePixelSize/3600.0, decimals, "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_dbl(fptr, "CDELT2", this->ImagePlanePixelSize/3600.0, decimals, "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    this->ErrorCode = fits_write_key_dbl(fptr, "CDELT3", 1.0, decimals, "", &status); if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    if (this->DebugCode > 0) { std::cout << "fits_write_key_*" << std::endl; }
    // write buffer
    datatype = TDOUBLE;
    nbuffer = sizey*sizex;
    gsl_matrix *image = gsl_matrix_calloc(sizey, sizex); // gsl_matrix_calloc sets all zero. gsl_matrix_alloc
    for (ichan=0; ichan<nchan; ichan++) {
        if (this->DebugCode > 0) { std::cout << "gsl_matrix_memcpy at ichan " << ichan << std::endl; }
        gsl_matrix_memcpy(this->SourcePlane2D, this->SourcePlaneDataCube[ichan]);
        for (j=0; j<sizey; j++) {
            for (i=0; i<sizex; i++) {
                if (this->DebugCode > 99) { 
                    std::cout << "gsl_matrix_set at ichan " << ichan << " pixel " << i << " " << j << std::flush;
                    std::cout << " ptr " << this->ImagePlane2D[j][i] << std::endl;
                }
                if (this->ImagePlane2D[j][i]) {
                    gsl_matrix_set(image, j, i, *(this->ImagePlane2D[j][i]));
                }
            }
        }
        fpixel = ichan * sizey * sizex + 1; // cfitsio pixel coordinate starts from 1.
        buffer = image->data;
        this->ErrorCode = fits_write_img(fptr, datatype, fpixel, nbuffer, buffer, &status);
        if (this->ErrorCode != 0) { fits_report_error(stderr, status); return; }
    }
    // clean up
    if (this->DebugCode > 0) 
        std::cout << "cleaning up" << std::endl;
    gsl_matrix_free(image);
    buffer = NULL;
    // close file
    this->ErrorCode = fits_close_file(fptr, &status);
    if (this->ErrorCode != 0) {
        fits_report_error(stderr, status);
        std::cerr << "Error! Failed to write the FITS file \"" << FilePath << "\"! Please check error messages above." << std::endl;
    }
    if (this->DebugCode > 0) 
        std::cout << "writeImagePlaneDataCube finished." << std::endl;
}


int LensingTransformer::errorCode()
{
    return this->ErrorCode;
}


int LensingTransformer::debugCode()
{
    return this->DebugCode;
}


int LensingTransformer::debugLevel()
{
    return this->DebugCode;
}

void LensingTransformer::setDebugLevel(int DebugLevel)
{
    this->DebugCode = DebugLevel;
}




// Global DebugCode flag.
//#ifdef DEBUG
int GlobalDebug = 0;
//#else
//int GlobalDebug = 0;
//#endif


// This will store all created and allocated instances.
//std::vector<LensingTransformer *> AllLensingTransformerInstances;


// This will store all created and allocated instances.
// See PyList documentation at https://docs.python.org/3/c-api/list.html
//PyListObject *PyAllLensingTransformerInstances = (PyListObject *) PyList_New(0);


// Function to check endianness.
bool isLittleEndian()
{
    // Nice piece of code from https://stackoverflow.com/questions/12791864/c-program-to-check-little-vs-big-endian
    volatile uint32_t EndianCheckVar = 0x01234567;
    bool IsLittleEndian = ((*((uint8_t*)(&EndianCheckVar))) == 0x67);
    return IsLittleEndian;
}


// Function to check if a point is in a polygon
// from https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
// from https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
bool checkPointInMeshGridCell(double testx, double testy, MeshGridCell cell)
{
    double vertx[5] = {cell.x11, cell.x12, cell.x22, cell.x21, cell.x11};
    double verty[5] = {cell.y11, cell.y12, cell.y22, cell.y21, cell.y11};
    int c = checkPointInPolygon(testx, testy, 5, vertx, verty);
    return (c>0);
}
int checkPointInPolygon(double testx, double testy, long nvert, double *vertx, double *verty)
{
    int i = 0, j = 0, c = 0;
    for (i = 0, j = nvert-1; i < nvert; j = i++) {
        if ( ((verty[i]>testy) != (verty[j]>testy)) &&
            (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) ) {
            c = !c;
            // The variable c is switching from 0 to 1 and 1 to 0 each time the horizontal ray crosses any edge. So basically it's keeping track of whether the number of edges crossed are even or odd. 0 means even and 1 means odd.
        }
    }
    return c;
}


void setGlobalDebugLevel(int value)
{
    GlobalDebug = value;
}


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
    double source_plane_cenx,
    double source_plane_ceny,
    int verbose)
{
    if (GlobalDebug > 0) 
        std::cout << "createLensingTransformer is called." << std::endl;
    
    // 
    LensingTransformer *my_lensing_transformer = new LensingTransformer();
    
    my_lensing_transformer->setDebugLevel(GlobalDebug);
    
    /* 
    AllLensingTransformerInstances.push_back(ptr);
    */
    
    //PyList_Append(PyAllLensingTransformerInstances, ptr);
    
    //
    my_lensing_transformer->readGlaficMeshFile(\
        mesh_file,
        mesh_ra,
        mesh_dec,
        verbose);
    if (my_lensing_transformer->errorCode() != 0) { std::cerr << "Error! Seems something failed." << std::endl; return NULL; }
    
    //
    if (GlobalDebug > 0) { 
        std::cout << "createLensingTransformer source_plane_data_cube addr " << source_plane_data_cube << std::endl;
    }
    if (GlobalDebug > 2) { 
        std::cout << "createLensingTransformer source_plane_data_cube first 8 bytes";
        //std::cout << " " << std::hex << std::setfill('0') << std::setw(2) << source_plane_data_cube[0];
        uint64_t rawBytes;
        std::memcpy(&rawBytes, source_plane_data_cube, sizeof(double));
        std::cout << " " << std::hex << std::setfill('0') << std::setw(2) << rawBytes << std::dec; 
        std::cout << " IsLittleEndian? " << isLittleEndian();
        //for (int k=0; k<8; k++) { std::cout << " " << std::hex << std::setfill('0') << std::setw(2) << rawBytes; }
        std::cout << std::endl;
    }
    
    //
    my_lensing_transformer->readSourcePlaneDataCube(\
        source_plane_data_cube,
        source_plane_data_nx,
        source_plane_data_ny,
        source_plane_data_nchan, 
        source_plane_ra,
        source_plane_dec,
        source_plane_pixelsize,
        source_plane_cenx,
        source_plane_ceny,
        verbose);
    if (my_lensing_transformer->errorCode() != 0) { std::cerr << "Error! Seems something failed." << std::endl; return NULL; }
    
    if (GlobalDebug > 0) 
        std::cout << "createLensingTransformer my_lensing_transformer addr " << my_lensing_transformer << std::endl;
    
    if (GlobalDebug > 0) 
        std::cout << "createLensingTransformer finished." << std::endl;
    
    return my_lensing_transformer;
}


void updateSourcePlaneDataCube(\
    void *ptr,
    double *source_plane_data_cube,
    int verbose)
{
    /* Update the SourcePlaneDataCube with a data array that has the same dimensions.
    */
    if (GlobalDebug > 0) 
        std::cout << "updateSourcePlaneDataCube is called." << std::endl;
    //
    LensingTransformer *my_lensing_transformer = (LensingTransformer *)ptr;
    // Get source plane data cube 3D dimension
    const long nchan = my_lensing_transformer->SourcePlaneDataCube.size(); // cube channel number
    const long nx = my_lensing_transformer->SourcePlaneDataCube[0]->size2; // source plane sizex
    const long ny = my_lensing_transformer->SourcePlaneDataCube[0]->size1; // source plane sizey. 
    // 
    long i=0, j=0, ichan=0, ipixel=0;
    for (ichan=0; ichan<nchan; ichan++) {
        if (GlobalDebug > 1) { std::cout << "update source plane data cube at ichan " << ichan << std::endl; }
        for (j=0; j<ny; j++) {
            for (i=0; i<nx; i++) {
                gsl_matrix_set(my_lensing_transformer->SourcePlaneDataCube[ichan], j, i, source_plane_data_cube[ipixel]);
                ipixel++;
            }
        }
    }
}


double *performLensingTransformation(\
    void *ptr, 
    double image_plane_ra,
    double image_plane_dec,
    double image_plane_pixelsize,
    long image_plane_sizex,
    long image_plane_sizey,
    double image_plane_cenx,
    double image_plane_ceny,
    int verbose)
{
    /* Return lensed image plane data cube.
    */
    if (GlobalDebug > 0) 
        std::cout << "performLensingTransformation is called." << std::endl;
    //
    LensingTransformer *my_lensing_transformer = (LensingTransformer *)ptr;
    // 
    // TODO: check input size and ImagePlane2D size if the latter exists.
    if (my_lensing_transformer->ImagePlane2D.size()>0) {
        if ((long)(my_lensing_transformer->ImagePlane2D.size()) != image_plane_sizey) {
            std::cerr << "Error! The input image sizey (" << image_plane_sizey << ") does not match the ImagePlane2D sizey (" << my_lensing_transformer->ImagePlane2D.size() << ")!" << std::endl;
            return NULL;
            // Instead of returnning NULL, we can do my_lensing_transformer->ImagePlane2D.clear()
        } else {
            if ((long)(my_lensing_transformer->ImagePlane2D[0].size()) != image_plane_sizex) {
                std::cerr << "Error! The input image sizex (" << image_plane_sizex << ") does not match the ImagePlane2D sizex (" << my_lensing_transformer->ImagePlane2D[0].size() << ")!" << std::endl;
                return NULL;
                // Instead of returnning NULL, we can do my_lensing_transformer->ImagePlane2D.clear()
            } else {
                // Everything seems good, we can use my_lensing_transformer->ImagePlane2D and do not need to redo linkSourcePlaneToImagePlane().
            }
        }
    }
    // 
    // linkSourcePlaneToImagePlane
    if (my_lensing_transformer->ImagePlane2D.size()==0) {
        my_lensing_transformer->linkSourcePlaneToImagePlane(\
            image_plane_ra,
            image_plane_dec,
            image_plane_pixelsize,
            image_plane_sizex,
            image_plane_sizey,
            image_plane_cenx,
            image_plane_ceny,
            verbose);
        if (my_lensing_transformer->errorCode() != 0) { std::cerr << "Error! Seems something failed." << std::endl; return NULL; }
    }
    //
    long nchan = my_lensing_transformer->SourcePlaneDataCube.size();
    long sizey = image_plane_sizey;
    long sizex = image_plane_sizex;
    long i=0, j=0, ichan=0, ipixel=0;
    // 
    double *data = (double *)malloc(nchan*sizey*sizex*sizeof(double));
    std::fill(data, data+nchan*sizey*sizex, std::nan(""));
    for (ichan=0; ichan<nchan; ichan++) {
        if (GlobalDebug > 1) { std::cout << "copying data at ichan " << ichan << std::endl; }
        gsl_matrix_memcpy(my_lensing_transformer->SourcePlane2D, my_lensing_transformer->SourcePlaneDataCube[ichan]);
        for (j=0; j<sizey; j++) {
            for (i=0; i<sizex; i++) {
                if (my_lensing_transformer->ImagePlane2D[j][i]) {
                    data[ipixel] = *(my_lensing_transformer->ImagePlane2D[j][i]);
                }
                ipixel++;
            }
        }
    }
    // 
    if (GlobalDebug > 0) {
        //saveDataCubeToFitsFile("data/image_plane_cube_from_c_for_debugging.fits", data, image_plane_sizex, image_plane_sizey, nchan, image_plane_pixelsize, image_plane_ra, image_plane_dec, image_plane_cenx, image_plane_ceny);
        std::cout << "performLensingTransformation data ptr " << data << std::endl;
        std::cout << "performLensingTransformation finished." << std::endl;
    }
    return data;
}



void destroyLensingTransformer(void *ptr)
{
    if (GlobalDebug > 0) 
        std::cout << "destroyLensingTransformer is called." << std::endl;
    
    /* 
    std::vector<LensingTransformer *>::iterator position = std::find(
        AllLensingTransformerInstances.begin(), 
        AllLensingTransformerInstances.end(), 
        (LensingTransformer *)ptr
    );
    if (position != AllLensingTransformerInstances.end())
        AllLensingTransformerInstances.erase(position);
    */
    
    //PyList_GetItem(PyAllLensingTransformerInstances, );
    
    //
    LensingTransformer *my_lensing_transformer = (LensingTransformer *)ptr;
    
    if (GlobalDebug > 0) {
        std::cout << "my_lensing_transformer " << my_lensing_transformer << std::endl;
        std::cout << "my_lensing_transformer->debugLevel() " << my_lensing_transformer->debugCode() << std::endl;
    }
    
    if (my_lensing_transformer) 
        delete my_lensing_transformer;
    
    my_lensing_transformer = NULL;
    ptr = NULL;
    
    if (GlobalDebug > 0) 
        std::cout << "destroyLensingTransformer finished." << std::endl;
}





void saveDataCubeToFitsFile(std::string FilePath, double *data, long sizex, long sizey, long nchan, double pixelsize, double ra, double dec, double cenx, double ceny, int verbose)
{
    // Similar to LensingTransformer::writeImagePlaneDataCube() but is a standalone function.
    
    // if cenx ceny are NaN, set them to image center
    if (std::isnan(cenx)) {
        cenx = (double(sizex)+1.0)/2.0;
    }
    if (std::isnan(ceny)) {
        ceny = (double(sizey)+1.0)/2.0;
    }
    
    // Open FITS file with cfitsio.
    fitsfile *fptr = NULL;
    long ichan=0;
    int status = 0, datatype = 0, bitpix = -64, naxis = 3, decimals = 12, errcode;
    long naxes[3] = {sizex, sizey, nchan};
    long nbuffer = 0, fpixel = 0; // first pixel's index 
    double *buffer = NULL;
    // create file
    if (verbose > 0) {
        std::cout << "saveDataCubeToFitsFile() opening file for writing: \"" << FilePath << "\"" << std::endl;
        std::cout << "saveDataCubeToFitsFile() data dimension: " << sizex << " " << sizey << " " << nchan << " " << " (x, y, channel)" << std::endl;
    }
    if (access(FilePath.c_str(), F_OK) == 0) {
        errcode = fits_create_file(&fptr, ("!"+FilePath).c_str(), &status); // prepending "!" means overwriting
    } else {
        errcode = fits_create_file(&fptr, (FilePath).c_str(), &status); // 
    }
    if (errcode != 0) { fits_report_error(stderr, status); return; }
    // create image header
    errcode = fits_create_img(fptr, bitpix, naxis, naxes, &status);
    if (errcode != 0) { fits_report_error(stderr, status); return; }
    // write wcs header
    errcode = fits_write_key_str(fptr, "RADESYS", "FK5", "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_str(fptr, "SPECSYS", "TOPOCENT", "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_flt(fptr, "EQUINOX", 2000.000, 3, "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_str(fptr, "CTYPE1", "RA---TAN", "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_str(fptr, "CTYPE2", "DEC--TAN", "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_str(fptr, "CTYPE3", "", "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_str(fptr, "CUNIT1", "deg", "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_str(fptr, "CUNIT2", "deg", "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_str(fptr, "CUNIT3", "", "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_dbl(fptr, "CRPIX1", cenx, decimals, "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_dbl(fptr, "CRPIX2", ceny, decimals, "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_dbl(fptr, "CRPIX3", 1.0, decimals, "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_dbl(fptr, "CRVAL1", ra, decimals, "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_dbl(fptr, "CRVAL2", dec, decimals, "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_dbl(fptr, "CRVAL3", 1.0, decimals, "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_dbl(fptr, "CDELT1", -pixelsize/3600.0, decimals, "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_dbl(fptr, "CDELT2", pixelsize/3600.0, decimals, "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    errcode = fits_write_key_dbl(fptr, "CDELT3", 1.0, decimals, "", &status); if (errcode != 0) { fits_report_error(stderr, status); return; }
    // write buffer
    datatype = TDOUBLE;
    nbuffer = sizey*sizex;
    if (verbose > 0) {
        std::cout << "saveDataCubeToFitsFile() writing channel images" << std::flush;
    }
    for (ichan=0; ichan<nchan; ichan++) {
        if (verbose > 0) {
            std::cout << " " << ichan << std::flush;
            if (ichan == nchan-1) {
                std::cout << std::endl;
            }
        }
        fpixel = ichan * sizey * sizex + 1; // cfitsio pixel coordinate starts from 1.
        buffer = data + ichan * sizey * sizex;
        errcode = fits_write_img(fptr, datatype, fpixel, nbuffer, buffer, &status);
        if (errcode != 0) { fits_report_error(stderr, status); return; }
    }
    // clean up
    buffer = NULL;
    // close file
    errcode = fits_close_file(fptr, &status); 
    if (errcode != 0) {
        fits_report_error(stderr, status);
        std::cerr << "Error! Failed to write the FITS file \"" << FilePath << "\"! Please check error messages above." << std::endl;
    }
}




