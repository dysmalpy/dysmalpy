/*
    See notes in "leastChiSquares1D.hpp".
 */
#include "leastChiSquares1D.hpp"

using namespace LeastChiSquaresFunctionsGaussian1D;


LeastChiSquares1D::LeastChiSquares1D()
{
    this->ErrorCode = 0;
    this->DebugCode = 3;
    //
    this->Data.ndata = 0;
    this->Data.x = NULL;
    this->Data.y = NULL;
    this->Data.yerr = NULL;
    //
    this->Params.values.clear();
    //
    this->Workspace = NULL;
    //this->fdf;
    //this->fdfparams = gsl_multifit_nlinear_default_parameters();
}


LeastChiSquares1D::~LeastChiSquares1D()
{
    if (this->Workspace) {
        gsl_multifit_nlinear_free(this->Workspace);
        this->Workspace = NULL;
    }
}


void LeastChiSquares1D::setParams(double *p, size_t nparams)
{
    this->Params.values.clear();
    std::vector<double> params_for_one_component;
    for (size_t i=0; i<nparams; i++) {
        params_for_one_component.push_back(p[i]);
    }
    this->Params.values.push_back(params_for_one_component);
}
void LeastChiSquares1D::setParams(std::vector<double> p)
{
    this->Params.values.clear();
    std::vector<double> params_for_one_component;
    for (size_t i=0; i < p.size(); i++) {
        params_for_one_component.push_back(p[i]);
    }
    this->Params.values.push_back(params_for_one_component);
}
void LeastChiSquares1D::setParams(double *p, std::vector<size_t> nparams)
{
    this->Params.values.clear();
    size_t k=0;
    for (size_t j=0; j < nparams.size(); j++) {
        std::vector<double> params_for_one_component;
        params_for_one_component.clear();
        for (size_t i=0; i < nparams[j]; i++) {
            params_for_one_component.push_back(p[k]);
            k++;
        }
        this->Params.values.push_back(params_for_one_component);
    }
}
void LeastChiSquares1D::setParams(std::vector<std::vector<double> > p)
{
    this->Params.values.clear();
    for (size_t j=0; j < p.size(); j++) {
        std::vector<double> params_for_one_component;
        params_for_one_component.clear();
        for (size_t i=0; i < p[j].size(); i++) {
            params_for_one_component.push_back(p[j][i]);
        }
        this->Params.values.push_back(params_for_one_component);
    }
}


double *LeastChiSquares1D::getParamsFlattened(std::vector<std::vector<double> > p)
{
    if (p.size() <= 0) { return NULL; }
    size_t nparams = 0;
    for (size_t j=0; j < p.size(); j++) {
        for (size_t i=0; i < p[j].size(); i++) {
            nparams++;
        }
    }
    double *outparams = (double *)malloc(nparams*sizeof(double));
    size_t k=0;
    for (size_t j=0; j < p.size(); j++) {
        for (size_t i=0; i < p[j].size(); i++) {
            *(outparams+k) = p[j][i];
            k++;
        }
    }
    // TODO: delete outparams outside
    return outparams;
}


std::vector<double> LeastChiSquares1D::getParamsFlattenedVector(std::vector<std::vector<double> > p)
{
    std::vector<double> outParamsVector;
    outParamsVector.clear();
    if (p.size() <= 0) { return outParamsVector; }
    size_t nparams = 0;
    for (size_t j=0; j < p.size(); j++) {
        for (size_t i=0; i < p[j].size(); i++) {
            nparams++;
        }
    }
    // 
    for (size_t j=0; j < p.size(); j++) {
        for (size_t i=0; i < p[j].size(); i++) {
            outParamsVector.push_back(p[j][i]);
        }
    }
    // 
    return outParamsVector;
}


size_t LeastChiSquares1D::getParamsCount(std::vector<std::vector<double> > p)
{
    if (p.size() <= 0) { return 0; }
    size_t nparams = 0;
    for (size_t j=0; j < p.size(); j++) {
        for (size_t i=0; i < p[j].size(); i++) {
            nparams++;
        }
    }
    return nparams;
}


void LeastChiSquares1D::setData(double *x, double *y, double *yerr, size_t ndata)
{
    //
    //this->Data.x = (double *)malloc(ndata*sizeof(double));
    //std::memcpy(this->Data.x, x, ndata);
    //
    this->Data.x = x;
    this->Data.y = y;
    this->Data.yerr = yerr;
    this->Data.ndata = ndata;
}



void LeastChiSquares1D::callback(const size_t iter, void *callback_payload, const gsl_multifit_nlinear_workspace *workspace)
{
    gsl_vector *x = gsl_multifit_nlinear_position(workspace);
    int verbose = 3;
    if (NULL != callback_payload) {
        verbose = ((LeastChiSquares1D::CallBackPayloadStruct *)callback_payload)->verbose; // callback_payload contains verbose level
    }
    size_t k = 0;
    //
    //gsl_vector *f = gsl_multifit_nlinear_residual(w);
    //gsl_vector *x = gsl_multifit_nlinear_position(w);
    //double avratio = gsl_multifit_nlinear_avratio(w);
    // 
    // compute reciprocal condition number of J(x)
    //double rcond;
    //gsl_multifit_nlinear_rcond(&rcond, w);
    // 
    // compute fresidual
    gsl_vector *fresidual = gsl_multifit_nlinear_residual(workspace);
    //
    // compute chisq
    double chisq;
    gsl_blas_ddot(fresidual, fresidual, &chisq);
    // 
    // compute dof
    double dof = workspace->f->size - workspace->x->size; // degree of freedom, ndata - nparams
    //
    // print
    if (verbose >= 3) {
        std::cout << "iter: " << iter << ", ";
        std::cout << "fit: ";
        for (k=0; k < x->size; k++) {
            if (k > 0) { std::cout << " "; }
            std::cout << gsl_vector_get(x, k);
        }
        //std::cout << ", ";
        //std::cout << "err: ";
        //gsl_matrix *J = gsl_multifit_nlinear_jac(this->Workspace);
        //gsl_matrix *covar = gsl_matrix_alloc(this->Workspace->x->size, this->Workspace->x->size);
        //gsl_multifit_nlinear_covar(J, 0.0, covar);
        //for (size_t k=0; k < x->size; k++) {
        //    if (k>0) std::cout << " ";
        //    std::cout << std::sqrt(gsl_matrix_get(covar, k, k));
        //}
        std::cout << ", ";
        std::cout << "chisq: " << chisq;
        std::cout << ", ";
        std::cout << "rchisq: " << chisq/dof;
        std::cout << std::endl;
    }
}



void LeastChiSquares1D::runFitting(\
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
    size_t maxniter, 
    int verbose)
{
    std::vector<size_t> nparams2;
    nparams2.push_back(nparams);
    this->runFitting(x, y, yerr, ndata, initparams, nparams2, 
                     outparams, outparamerrs, outyfitted, outyresidual, outchisq, 
                     maxniter, verbose);
}


void LeastChiSquares1D::runFitting(\
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
    size_t maxniter, 
    int verbose)
{
    /*
        The function to be called to run the fitting. 
        
    */
    
    if ( (this->DebugCode >= 2) || (verbose >= 2) ) {
        std::cout << "LeastChiSquares1D::runFitting() started" << std::endl;
    }
    
    /*
    if ( (this->Data.ndata <= 0) || (! this->Data.x) || (! this->Data.y) ) {
        if (this->DebugCode >= 2) {
            std::cout << "Error! No data has been set before running LeastChiSquares1D::runFitting()." << std::endl; 
        } 
        return;
    }
    */
    
    /*
    if ( (this->Params.size() <= 0) ) {
        if (this->DebugCode >= 2) {
            std::cout << "Error! No params have been set before running LeastChiSquares1D::runFitting()." << std::endl; 
        } 
        return;
    }
    
    if ( (this->Params[0].size() <= 0) ) {
        if (this->DebugCode >= 2) {
            std::cout << "Error! No params[0] have been set before running LeastChiSquares1D::runFitting()." << std::endl; 
        }
        return;
    }
    */
    
    //
    size_t k=0;
    
    // 
    this->ErrorCode = 0;
    
    // check existing workspace
    if ( NULL != this->Workspace ) {
        if (this->DebugCode >= 2) {
            std::cout << "LeastChiSquares1D::runFitting() this->Workspace " << this->Workspace << std::endl; 
        }
        if (this->Data.ndata != ndata) {
            if (this->DebugCode >= 2) {
                std::cout << "LeastChiSquares1D::runFitting() clearing previous workspace" << std::endl; 
            }
            gsl_multifit_nlinear_free(this->Workspace);
            this->Workspace = NULL;
        }
    }
    
    //
    this->setData(x, y, yerr, ndata);
    this->setParams(initparams, nparams);
    size_t ntotalparams = this->getParamsCount(this->Params.values);
    
    /* prepare fdfparams */
    this->fdfparams = gsl_multifit_nlinear_default_parameters(); // see definition in "gsl/gsl_multifit_nlinear.h
    //this->fdfparams.trs = gsl_multifit_nlinear_trs_lm;
    //this->fdfparams.trs = gsl_multifit_nlinear_trs_lmaccel;
    //this->fdfparams.trs = gsl_multifit_nlinear_trs_dogleg;
    //this->fdfparams.trs = gsl_multifit_nlinear_trs_ddogleg;
    //this->fdfparams.trs = gsl_multifit_nlinear_trs_subspace2D;
    //printf("# %s/%s\n", gsl_multifit_nlinear_name(work), gsl_multifit_nlinear_trs_name(work));
 
    
    /* allocate workspace with default parameters */
    if ( NULL == this->Workspace ) {
        this->T = gsl_multifit_nlinear_trust;
        this->Workspace = gsl_multifit_nlinear_alloc(this->T, &(this->fdfparams), ndata, ntotalparams);
        if ( (this->DebugCode >= 2) || (verbose >= 2) ) {
            std::cout << "LeastChiSquares1D::runFitting() setting workspace with ntotalparams " << ntotalparams << " ndata " << ndata << std::endl; 
        }
    } else {
        if ( (this->DebugCode >= 2) || (verbose >= 2) ) {
            std::cout << "LeastChiSquares1D::runFitting() re-using workspace with ntotalparams " << ntotalparams << " ndata " << ndata << std::endl; 
        }
    }
    if (this->DebugCode >= 3) {
        std::cout << "LeastChiSquares1D::runFitting() this->Workspace = " << std::hex << this->Workspace << std::dec << ", this->Workspace->type = " << this->Workspace->type << ", this->Workspace->x->size = " << this->Workspace->x->size << ", this->Workspace->f->size = " << this->Workspace->f->size << std::endl; 
        //std::cout << "LeastChiSquares1D::runFitting() this->Workspace->state = " << this->Workspace->state << ", this->Workspace->state->params = " << ((trust_state_t *)(this->Workspace->state))->params << std::endl; 
    }
    
    /* fdf */
#ifndef leastChiSquaresFunctions1D_hpp
    this->setfdf();
#else
    this->fdf.f = func4gsl_f; 
    this->fdf.df = func4gsl_df;
    this->fdf.fvv = func4gsl_fvv;
    //this->fdf.f = func_f; // 
    //this->fdf.df = func_df;
    //this->fdf.fvv = func_fvv; 
    //this->fdf.f = LeastChiSquaresFunctionsGaussian1D::func4gsl_f; 
    //this->fdf.df = LeastChiSquaresFunctionsGaussian1D::func4gsl_df;
    this->fdf.df = NULL; // set to NULL for finite-difference Jacobian
    this->fdf.fvv = NULL; // not using geodesic acceleration
#endif
    this->fdf.n = ndata;
    this->fdf.p = ntotalparams;
    this->fdf.params = &(this->Data); // this is the data but called "params" in gsl. here we call params as the variables of the fitting function.
    
    if (this->DebugCode >= 3) {
        std::cout << "LeastChiSquares1D::runFitting() setfdf " << std::endl; 
    }
    if (this->DebugCode >= 3) {
        //std::cout << "LeastChiSquares1D::runFitting() this->fdf.f = 0x" << std::hex << (size_t)this->fdf.f << ", &func4gsl_f = 0x" << std::hex << (size_t)&func4gsl_f << std::dec << std::endl; 
        std::cout << "LeastChiSquares1D::runFitting() this->fdf.f = " << std::hex << this->fdf.f << std::dec << std::endl;
    }
    
    /* initialize solver with starting point and weights */
    std::vector<double> initParamsVector = this->getParamsFlattenedVector(this->Params.values);
    gsl_vector_view initParamsVectorView = gsl_vector_view_array(initParamsVector.data(), ntotalparams);
    gsl_vector_view xVectorView = gsl_vector_view_array(this->Data.x, ndata); // not used
    gsl_vector_view yVectorView = gsl_vector_view_array(this->Data.y, ndata);
    gsl_vector *yerrGslVector = NULL;
    if (NULL != this->Data.yerr) {
        gsl_vector_view yerrVectorView = gsl_vector_view_array(this->Data.yerr, ndata);
        yerrGslVector = &yerrVectorView.vector;
    }
    
    if ( (this->DebugCode >= 2) || (verbose >= 2) ) {
        std::cout << "LeastChiSquares1D::runFitting() set initparams "; 
        std::copy(initParamsVector.begin(), initParamsVector.end(), std::ostream_iterator<double>(std::cout, " ")); 
        std::cout << std::endl; 
    }
    if (this->DebugCode >= 4) {
        std::cout << "LeastChiSquares1D::runFitting() set x "; 
        for (k=0; k < ndata; k++) { std::cout << gsl_vector_get(&xVectorView.vector, k) << " "; }
        std::cout << std::endl; 
    }
    if (this->DebugCode >= 4) {
        std::cout << "LeastChiSquares1D::runFitting() set y "; 
        for (k=0; k < ndata; k++) { std::cout << gsl_vector_get(&yVectorView.vector, k) << " "; }
        std::cout << std::endl; 
    }
    if (NULL != this->Data.yerr) {
        if (this->DebugCode >= 4) {
            std::cout << "LeastChiSquares1D::runFitting() set yerr "; 
            for (k=0; k < ndata; k++) { std::cout << gsl_vector_get(yerrGslVector, k) << " "; }
            std::cout << std::endl; 
        }
    }
    
    //if (this->DebugCode >= 4) {
    //    std::cout << "LeastChiSquares1D::runFitting() gsl_multifit_nlinear_init ..." << std::endl; 
    //    gsl_multifit_nlinear_init(&initParamsVectorView.vector, &(this->fdf), this->Workspace); // init without weights
    //    std::cout << "LeastChiSquares1D::runFitting() gsl_multifit_nlinear_init done" << std::endl; 
    //}
    
    gsl_vector *initParamsGslVector = &(initParamsVectorView.vector);
    gsl_vector *dataWeightsGslVector = NULL;
    
    if (NULL != this->Data.yerr) {
        std::vector<double> dataWeightsVector;
        for (k=0; k < ndata; k++) {
            double tmpval = gsl_vector_get(yerrGslVector, k);
            dataWeightsVector.push_back(1.0/tmpval/tmpval); // (1 / si**2)
        }
        gsl_vector_view dataWeightsVectorView = gsl_vector_view_array(dataWeightsVector.data(), ndata);
        
        if (this->DebugCode >= 4) {
            std::cout << "LeastChiSquares1D::runFitting() set dataweights "; 
            std::copy(dataWeightsVector.begin(), dataWeightsVector.end(), std::ostream_iterator<double>(std::cout, " ")); 
            std::cout << std::endl; 
        }
        
        if (this->DebugCode >= 2) {
            std::cout << "LeastChiSquares1D::runFitting() gsl_multifit_nlinear_winit ..." << std::endl; 
        }
        
        dataWeightsGslVector = &(dataWeightsVectorView.vector);
        
    } else {
        if (this->DebugCode >= 2) {
            std::cout << "LeastChiSquares1D::runFitting() gsl_multifit_nlinear_init ..." << std::endl; 
        }
    }
    
    this->ErrorCode = gsl_multifit_nlinear_winit(initParamsGslVector, dataWeightsGslVector, &(this->fdf), this->Workspace);
    // this will call "gsl/multifit_nlinear/fdf.c" 
    //     return (w->type->init) (w->state, w->sqrt_wts, w->fdf, w->x, w->f, w->J, w->g);
    // then "gsl/multifit_nlinear/trust.c" 
    //     static int trust_init(void *vstate, const gsl_vector *swts, gsl_multifit_nlinear_fdf *fdf,  
    //         const gsl_vector *x, gsl_vector *f, gsl_matrix *J, gsl_vector *g)
    
    
    
    /*
    // 2021-08-04 testing!!!
    gsl_vector *x4gsl = gsl_vector_alloc(ntotalparams);
    gsl_vector_set(x4gsl, 0, 1.0);
    gsl_vector_set(x4gsl, 1, 0.0);
    gsl_vector_set(x4gsl, 2, 1.0);
    gsl_multifit_nlinear_fdf fdf;
    fdf.f = func4gsl_f;
    fdf.df = func4gsl_df;
    fdf.fvv = func4gsl_fvv;
    fdf.n = ndata;
    fdf.p = ntotalparams;
    struct DataStruct4gsl fit_data;
    fit_data.t = (double *)malloc(ndata * sizeof(double));
    fit_data.y = (double *)malloc(ndata * sizeof(double));
    fit_data.n = ndata;
    for (k=0; k<ndata; k++) {
        fit_data.t[k] = this->Data.x[k];
        fit_data.y[k] = this->Data.y[k];
    }
    fdf.params = &fit_data;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
    fdf_params.trs = gsl_multifit_nlinear_trs_lmaccel;
    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_workspace *work = gsl_multifit_nlinear_alloc(T, &(fdf_params), ndata, ntotalparams);
    gsl_vector * ff = gsl_multifit_nlinear_residual(work);
    gsl_vector * yy = gsl_multifit_nlinear_position(work);
    if (this->DebugCode >= 2) {
        std::cout << "LeastChiSquares1D::runFitting() gsl_multifit_nlinear_init ..." << std::endl; 
    }
    this->ErrorCode = gsl_multifit_nlinear_init(x4gsl, &fdf, work);
    if (this->DebugCode >= 2) {
        std::cout << "LeastChiSquares1D::runFitting() gsl_multifit_nlinear_init done" << std::endl; 
    }
    // 2021-08-04 testing!!!
    */
    
    
    
    if (this->DebugCode >= 2) {
        std::cout << "LeastChiSquares1D::runFitting() gsl_multifit_nlinear_winit done" << std::endl; 
    }

    /* compute initial cost function */
    gsl_vector *fresidual = gsl_multifit_nlinear_residual(this->Workspace);
    
    double chisq0;
    gsl_blas_ddot(fresidual, fresidual, &chisq0);

    /* solve the system with a maximum of 'maxniter' iterations */
    const double xtol = 1.0e-8;
    const double gtol = 1.0e-8;
    const double ftol = 1.0e-8; // 0.0 do not limit fractional change of chisq 
    int status = 0, info = 0, niter = 0; 
    if (this->DebugCode >= 2) {
        std::cout << "LeastChiSquares1D::runFitting() gsl_multifit_nlinear_driver ... maxniter = " << maxniter << std::endl; 
    }
    //void *callback_payload = NULL;
    struct LeastChiSquares1D::CallBackPayloadStruct callback_payload;
    callback_payload.verbose = verbose;
    
    /*
    // 2021-08-04 testing 2!!!
    status = gsl_multifit_nlinear_iterate(this->Workspace);
    std::cout << "LeastChiSquares1D::runFitting() gsl_multifit_nlinear_driver status error message: " << gsl_strerror(status) << std::endl;
    this->callback(0, NULL, this->Workspace);
    
    status = gsl_multifit_nlinear_iterate(this->Workspace);
    std::cout << "LeastChiSquares1D::runFitting() gsl_multifit_nlinear_driver status error message: " << gsl_strerror(status) << std::endl;
    this->callback(1, NULL, this->Workspace);
    // 2021-08-04 testing 2!!!
    */
    
    status = gsl_multifit_nlinear_driver(maxniter, xtol, gtol, ftol,
             LeastChiSquares1D::callback, &callback_payload, &info, this->Workspace);
             // reason for stopping: 
             // (info == 1) ? "small step size" : "small gradient"
             // info 27 GSL_ENOPROG "iteration is not making progress towards solution"
    
    niter = gsl_multifit_nlinear_niter(this->Workspace);
    
    if ( (this->DebugCode >= 2) || (verbose >= 2) ) {
        std::cout << "LeastChiSquares1D::runFitting() gsl_multifit_nlinear_driver done, niter = " << niter << ", status = " << status << ", info = " << info << std::endl; 
    }
    if ( ( status != GSL_SUCCESS ) && ( (this->DebugCode >= 2) || (verbose >= 2) ) ) {
        std::cout << "LeastChiSquares1D::runFitting() gsl_multifit_nlinear_driver niter = " << niter << ", status = " << status << ", info = " << info << ", status error message: \"" << gsl_strerror(status) << "\"" << ", info error message: \"" << gsl_strerror(info) << "\"" << std::endl;
    }
    
    
    //double dof = ndata - ntotalparams; // not used
    
    /* compute covariance of best fit parameters */
    gsl_matrix *J = gsl_multifit_nlinear_jac(this->Workspace);
    gsl_matrix *covar = gsl_matrix_alloc(ntotalparams, ntotalparams);
    gsl_multifit_nlinear_covar(J, 0.0, covar);

    /* compute final cost */
    double chisq;
    gsl_blas_ddot(fresidual, fresidual, &chisq);
    *outchisq = chisq;
    //double c = GSL_MAX_DBL(1, sqrt(chisq / dof));

    /* store cond(J(x)) = 1.0 / rcond */
    double rcond;
    gsl_multifit_nlinear_rcond(&rcond, this->Workspace);
    
    /* final bestfit */
    for (k=0; k < ntotalparams; k++) {
        outparams[k] = gsl_vector_get(this->Workspace->x, k);
        outparamerrs[k] = std::sqrt(gsl_matrix_get(covar, k, k));
    }
    for (k=0; k < ndata; k++) {
        outyresidual[k] = gsl_vector_get(fresidual, k);
        outyfitted[k] = this->Data.y[k] - outyresidual[k];
    }
    if ( (this->DebugCode >= 2) || (verbose >= 2) ) {
        std::cout << "LeastChiSquares1D::runFitting() bestfit ";
        for (k=0; k < ntotalparams; k++) {
            if (k>0) { std::cout << ", "; }
            std::cout << outparams[k] << " +- " << outparamerrs[k];
        } 
        std::cout << std::endl;
    }
    // bestfit_A = gsl_vector_get(this->Workspace->x, 0);
    // bestfit_mu = gsl_vector_get(this->Workspace->x, 1);
    // bestfit_sigma = gsl_vector_get(this->Workspace->x, 2);
    // err_A = std::sqrt(gsl_matrix_get(covar, 0, 0));
    // err_mu = std::sqrt(gsl_matrix_get(covar, 1, 1));
    // err_sigma = std::sqrt(gsl_matrix_get(covar, 2, 2));
    
    /* clean up */
    //gsl_matrix_free(J);
    gsl_matrix_free(covar);
    
    if ( (this->DebugCode >= 2) || (verbose >= 2) ) {
        std::cout << "LeastChiSquares1D::runFitting() finished" << std::endl;
    }
    
}


int LeastChiSquares1D::errorCode()
{
    return this->ErrorCode;
}


int LeastChiSquares1D::debugCode()
{
    return this->DebugCode;
}


int LeastChiSquares1D::debugLevel()
{
    return this->DebugCode;
}

void LeastChiSquares1D::setDebugLevel(int DebugLevel)
{
    this->DebugCode = DebugLevel;
}




// Global DebugCode flag.
//#ifdef DEBUG
int GlobalDebug = 0;
//#else
//int GlobalDebug = 0;
//#endif


// Function to check endianness.
bool isLittleEndian()
{
    // Nice piece of code from https://stackoverflow.com/questions/12791864/c-program-to-check-little-vs-big-endian
    volatile uint32_t EndianCheckVar = 0x01234567;
    bool IsLittleEndian = ((*((uint8_t*)(&EndianCheckVar))) == 0x67);
    return IsLittleEndian;
}


void setGlobalDebugLevel(int value)
{
    GlobalDebug = value;
}


void *createLeastChiSquares1D()
{
    if (GlobalDebug > 0) 
        std::cout << "createLeastChiSquares1D is called." << std::endl;
    
    // 
    LeastChiSquares1D *my_least_chisq_fitter = new LeastChiSquares1D();
    
    my_least_chisq_fitter->setDebugLevel(GlobalDebug);
    
    //
    if (my_least_chisq_fitter->errorCode() != 0) { std::cerr << "Error! Seems something failed." << std::endl; return NULL; }
    
    //
    if (GlobalDebug > 0) 
        std::cout << "createLeastChiSquares1D my_least_chisq_fitter addr = " << std::hex << my_least_chisq_fitter << std::dec << std::endl;
    
    if (GlobalDebug > 0) 
        std::cout << "createLeastChiSquares1D finished." << std::endl;
    
    return my_least_chisq_fitter;
}


double *fitLeastChiSquares1D(\
    void *ptr, 
    double *x, 
    double *y, 
    double *yerr, 
    long ndata, 
    double *initparams, 
    long nparams, 
    int maxniter, 
    int verbose, 
    struct MultiThreadPayloadStruct *mthread)
{
    /* Run the fitting, return outparams + outparamerrs + outyfitted + outyresidual.
    */
    if ( (GlobalDebug > 0) && (NULL == mthread) ) { 
        std::cout << "fitLeastChiSquares1D is called." << std::endl;
    }
    //
    // create class
    LeastChiSquares1D *my_least_chisq_fitter = (LeastChiSquares1D *)ptr;
    // 
    // prepare output array pointer
    double *outall = NULL;
    std::mutex *mu = NULL;
    // 
    // load mthread
    if (NULL != mthread) {
        outall = mthread->outptr;
        mu = mthread->mu;
    }
    //
    // allocate memory for the output array if it is NULL
    if (NULL == outall) { 
        if ( (GlobalDebug > 0) && (verbose >= 1) ) {
            std::cout << "fitLeastChiSquares1D allocating memory for \"outall\" output array" << std::endl;
        }
        outall = (double *)malloc((nparams*2 + ndata*2 + 1)*sizeof(double)); // a complex output array in the memory, =  outparams + outparamerrs + outyfitted + outyresidual
        for (size_t k=0; k < (size_t)(nparams*2 + ndata*2 + 1); k++) { outall[k] = std::nan(""); }
    } else {
        if ( (GlobalDebug > 0) && (verbose >= 1) ) {
            if (NULL != mu) { mu->lock(); }
            std::cout << "fitLeastChiSquares1D re-using \"outall\" output array at addr " << std::hex << outall << std::dec << std::endl;
            if (NULL != mu) { mu->unlock(); }
        }
    }
    // 
    double *outparams = &(outall[0]);
    double *outparamerrs = &(outall[nparams]);
    double *outyfitted = &(outall[nparams*2]);
    double *outyresidual = &(outall[nparams*2+ndata]);
    double *outchisq = &(outall[nparams*2+ndata*2]);
    my_least_chisq_fitter->runFitting(x, y, yerr, (size_t)ndata, initparams, (size_t)nparams, outparams, outparamerrs, outyfitted, outyresidual, outchisq, maxniter, verbose);
    // 
    if (GlobalDebug > 0) {
        if (NULL != mu) { 
            mu->lock();
            std::cout << "fitLeastChiSquares1D finished from thread " << std::hex << std::this_thread::get_id() << std::dec << std::endl;
            mu->unlock();
        } else {
            std::cout << "fitLeastChiSquares1D finished." << std::endl;
        }
    }
    return outall;
}



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
    int maxniter, 
    int verbose, 
    struct MultiThreadPayloadStruct *mthread)
{
    /* Run the fitting, return outparams + outparamerrs + outyfitted + outyresidual.
    
    The input data must be a normal FITS 3D data cube, that is, the 1-st (lowest, fatest-changing) dimension is image width, 2-nd dimension image height, and 3-rd (highest, slowest-changing) dimension channels. We will do a transpose to increase the speed.   
    
    Note that outparams 1-st (lowest, fatest-changing) dimension is parameters, then 2-nd dimension image height, then 3-rd dimension image width. Its transpose is the normal FITS 3D data cube.   
    */
    if ( (GlobalDebug > 0) && (NULL == mthread) ) { 
        std::cout << "fitLeastChiSquares1DForDataCube is called." << std::endl;
    }
    //
    // create class
    LeastChiSquares1D *my_least_chisq_fitter = (LeastChiSquares1D *)ptr;
    size_t k=0, kfirst=0, klast=(size_t)(ny*nx)-1, kstep=1;
    std::mutex *mu = NULL;
    // 
    // prepare output array pointer
    double *outall = NULL;
    // 
    // load mthread
    if (NULL != mthread) {
        kfirst = mthread->kfirst;
        klast = mthread->klast;
        kstep = mthread->kstep;
        outall = mthread->outptr;
        mu = mthread->mu;
    }
    //
    // transpose and make data cube (ny*nx)*nchan size
    gsl_matrix_view dataMatrixView = gsl_matrix_view_array(data, (size_t)nchan, (size_t)(ny*nx)); // nrows (slower-changing), ncols (faster-changing)
    gsl_matrix *dataMatrix = &(dataMatrixView.matrix);
    //gsl_matrix *dataMatrixTransposed = gsl_matrix_alloc((size_t)(ny*nx), (size_t)nchan); // nrows, ncols
    //gsl_matrix_transpose_memcpy(dataMatrixTransposed, dataMatrix); // dest, src
    //
    // transpose and make dataerr cube (ny*nx)*nchan size
    gsl_matrix *dataerrMatrix = NULL;
    if (NULL != dataerr) {
        gsl_matrix_view dataerrMatrixView = gsl_matrix_view_array(dataerr, (size_t)nchan, (size_t)(ny*nx)); // nrows (slower-changing), ncols (faster-changing)
        dataerrMatrix = &(dataerrMatrixView.matrix);
        //gsl_matrix *dataerrMatrixTransposed = gsl_matrix_alloc((size_t)(ny*nx), (size_t)nchan); // nrows, ncols
        //gsl_matrix_transpose_memcpy(dataerrMatrixTransposed, dataerrMatrix); // dest, src
    }
    //
    // transpose and make initparams cube (ny*nx)*nparams size
    gsl_matrix_view paramsMatrixView = gsl_matrix_view_array(initparamsall, (size_t)nparams, (size_t)(ny*nx)); // nrows (slower-changing), ncols (faster-changing)
    gsl_matrix *paramsMatrix = &(paramsMatrixView.matrix);
    /* 20210804 testing
    std::cout << "fitLeastChiSquares1DForDataCube paramsMatrix k=0 ";
    for (k=0; k < (size_t)(ny*nx); k++) { std::cout << gsl_matrix_get(paramsMatrix, 0, k) << " "; }
    std::cout << std::endl;
    std::cout << "fitLeastChiSquares1DForDataCube paramsMatrix k=1 ";
    for (k=0; k < (size_t)(ny*nx); k++) { std::cout << gsl_matrix_get(paramsMatrix, 1, k) << " "; }
    std::cout << std::endl;
    std::cout << "fitLeastChiSquares1DForDataCube paramsMatrix k=2 ";
    for (k=0; k < (size_t)(ny*nx); k++) { std::cout << gsl_matrix_get(paramsMatrix, 2, k) << " "; } 
    std::cout << std::endl;
    gsl_vector_view params_view = gsl_matrix_column(paramsMatrix, 0);
    std::cout << "fitLeastChiSquares1DForDataCube paramsMatrix column=0/" << paramsMatrix->tda << " ";
    for (k=0; k < (size_t)(nparams); k++) { std::cout << gsl_vector_get(&params_view.vector, k) << " "; }
    std::cout << std::endl;
    std::cout << "fitLeastChiSquares1DForDataCube paramsMatrix column=0/" << paramsMatrix->tda << " ";
    for (k=0; k < (size_t)(nparams); k++) { std::cout << *(gsl_vector_ptr(&params_view.vector, 0) + k) << " "; }
    std::cout << std::endl;
    */
    //gsl_matrix *paramsMatrixTransposed = gsl_matrix_alloc((size_t)(ny*nx), (size_t)nparams); // nrows, ncols
    //gsl_matrix_transpose_memcpy(paramsMatrixTransposed, paramsMatrix); // dest, src
    // 
    // allocate memory for the output array if it is NULL
    if (NULL == outall) { 
        if ( (GlobalDebug > 0) && (verbose >= 1) ) {
            std::cout << "fitLeastChiSquares1DForDataCube allocating memory for \"outall\" output array" << std::endl;
        }
        outall = (double *)malloc((nparams*ny*nx*2 + nchan*ny*nx*2 + ny*nx)*sizeof(double)); // a complex output array in the memory, =  outparams + outparamerrs + outyfitted + outyresidual
        for (k=0; k < (size_t)(nparams*ny*nx*2 + nchan*ny*nx*2 + ny*nx); k++) { outall[k] = std::nan(""); }
    } else {
        if ( (GlobalDebug > 0) && (verbose >= 1) ) {
            if (NULL != mu) { mu->lock(); }
            std::cout << "fitLeastChiSquares1DForDataCube re-using \"outall\" output array at addr " << std::hex << outall << " from thread " << std::this_thread::get_id() << std::dec << std::endl;
            if (NULL != mu) { mu->unlock(); }
        }
    }
    gsl_matrix_view outallMatrixView = gsl_matrix_view_array(outall, (size_t)(nparams*2 + nchan*2 + 1), (size_t)(ny*nx)); // nrows (slower-changing), ncols (faster-changing)
    gsl_matrix *outallMatrix = &(outallMatrixView.matrix);
    // 
    /*
    size_t i=0, j=0;
    for (j=0; j<(size_t)ny; j++) {
        for (i=0; i<(size_t)nx; i++) {
            double *y = &(data[j*nx+i]);
            double *yerr = &(dataerr[j*nx+i]);
            double *outparams = &(outall[nparams*ny*nx*0 + j*nx + i]);
            double *outparamerrs = &(outall[nparams*ny*nx*1 + j*nx + i]);
            double *outyfitted = &(outall[nparams*ny*nx*2 + j*nx + i]);
            double *outyresidual = &(outall[nparams*ny*nx*2 + nchan*ny*nx + j*nx + i]);
            double *outchisq = &(outall[nparams*ny*nx*2 + nchan*ny*nx*2 + j*nx + i]);
            double *initparams = &(initparamsall[j*nx + i]); 
            my_least_chisq_fitter->runFitting(x, y, yerr, (size_t)nchan, initparams, (size_t)nparams, outparams, outparamerrs, outyfitted, outyresidual, outchisq, maxniter, verbose);
        }
    }
    */
    // 
    // be careful about the dimension order
    //for (k=0; k < (size_t)(ny*nx); k++) {
    for (k = kfirst; k <= klast; k += kstep) {
        // load y array for this pixel
        gsl_vector *y_view_vector = NULL;
        gsl_vector_view y_view = gsl_matrix_column(dataMatrix, k);
        y_view_vector = &y_view.vector;
        if (std::isnan(gsl_vector_max(y_view_vector))) { continue; } /* skips if data includes NaN */ 
        
        // load yerr array for this pixel
        gsl_vector *yerr_view_vector = NULL;
        if (NULL != dataerr) {
            gsl_vector_view yerr_view = gsl_matrix_column(dataerrMatrix, k);
            yerr_view_vector = &yerr_view.vector; 
            if (std::isnan(gsl_vector_max(yerr_view_vector))) { continue; } /* skips if data includes NaN */
        }
        
        // load initparams array for this pixel
        gsl_vector *initparams_view_vector = NULL;
        gsl_vector_view initparams_view = gsl_matrix_column(paramsMatrix, k);
        initparams_view_vector = &initparams_view.vector;
        if (std::isnan(gsl_vector_max(initparams_view_vector))) { continue; } /* skips if data includes NaN */
        
        gsl_vector_view outall_view = gsl_matrix_column(outallMatrix, k);
        gsl_vector *outall_view_vector = &outall_view.vector;
        
        gsl_vector *outall_vector = gsl_vector_calloc(outall_view_vector->size);
        
        //gsl_vector_view outparams_view = gsl_matrix_subcolumn(outallMatrix, k, 0, nparams);
        //gsl_vector_view outparamerrs_view = gsl_matrix_subcolumn(outallMatrix, k, nparams, nparams);
        //gsl_vector_view outyfitted_view = gsl_matrix_subcolumn(outallMatrix, k, nparams*2, nchan);
        //gsl_vector_view outyresidual_view = gsl_matrix_subcolumn(outallMatrix, k, nparams*2+nchan, nchan);
        //gsl_vector_view outchisq_view = gsl_matrix_subcolumn(outallMatrix, k, nparams*2+nchan*2, 1);
        //gsl_vector *outparams_vector = gsl_vector_calloc((size_t)nparams);
        //gsl_vector *outparamerrs_vector = gsl_vector_calloc((size_t)nparams);
        //gsl_vector *outyfitted_vector = gsl_vector_calloc((size_t)nchan);
        //gsl_vector *outyresidual_vector = gsl_vector_calloc((size_t)nchan);
        //gsl_vector *outchisq_vector = gsl_vector_calloc(1);
        //double *y = (&y_view.vector)->data;
        //double *yerr = (&yerr_view.vector)->data;
        //double *initparams = (&initparams_view.vector)->data;
        //double *y = gsl_vector_ptr(&y_view.vector, 0); // this is wrong in memory
        //double *yerr = gsl_vector_ptr(&yerr_view.vector, 0); // this is wrong in memory
        //double *initparams = gsl_vector_ptr(&initparams_view.vector, 0); // this is wrong in memory
        
        gsl_vector *y_vector = gsl_vector_calloc(y_view_vector->size);
        
        gsl_vector *yerr_vector = NULL;
        if (NULL != dataerr) {
            yerr_vector = gsl_vector_calloc(yerr_view_vector->size);
        }
        
        gsl_vector *initparams_vector = gsl_vector_calloc(initparams_view_vector->size);
        
        gsl_vector_memcpy(y_vector, y_view_vector);
        
        if (NULL != dataerr) { 
            gsl_vector_memcpy(yerr_vector, yerr_view_vector);
        }
        
        gsl_vector_memcpy(initparams_vector, initparams_view_vector);
        
        double *y = y_vector->data;
        
        double *yerr = NULL;
        if (NULL != dataerr) {
            yerr = yerr_vector->data;
        }
        
        double *initparams = initparams_vector->data;
        
        //double *outparams = (&outparams_view.vector)->data;
        //double *outparamerrs = (&outparams_view.vector)->data;
        //double *outyfitted = (&outyfitted_view.vector)->data;
        //double *outyresidual = (&outyresidual_view.vector)->data;
        //double *outchisq = (&outchisq_view.vector)->data; 
        //double *outparams = outparams_vector->data;
        //double *outparamerrs = outparams_vector->data;
        //double *outyfitted = outyfitted_vector->data;
        //double *outyresidual = outyresidual_vector->data;
        //double *outchisq = outchisq_vector->data;
        //double *outparams = (&outall_view.vector)->data;
        //double *outparamerrs = (&outall_view.vector)->data + nparams;
        //double *outyfitted = (&outall_view.vector)->data + nparams*2;
        //double *outyresidual = (&outall_view.vector)->data + nparams*2 + nchan;
        //double *outchisq = (&outall_view.vector)->data + nparams*2 + nchan*2;
        
        double *outparams = gsl_vector_ptr(outall_vector, 0);
        double *outparamerrs = gsl_vector_ptr(outall_vector, nparams);
        double *outyfitted = gsl_vector_ptr(outall_vector, nparams*2);
        double *outyresidual = gsl_vector_ptr(outall_vector, nparams*2 + nchan);
        double *outchisq = gsl_vector_ptr(outall_vector, nparams*2 + nchan*2);
        my_least_chisq_fitter->runFitting(x, y, yerr, (size_t)nchan, initparams, (size_t)nparams, outparams, outparamerrs, outyfitted, outyresidual, outchisq, maxniter, verbose);
        //
        //gsl_matrix_set_col(outallMatrix, k, &outall_view.vector);
        if (NULL != mthread) {
            if (mthread->dryrun <= 0) {
                if (NULL != mu) { mu->lock(); }
                gsl_matrix_set_col(outallMatrix, k, outall_vector);
                if (NULL != mu) { mu->unlock(); }
            } else {
                if ( (GlobalDebug > 0) && (verbose >= 1) && (k == kfirst) ) {
                    if (NULL != mu) { mu->lock(); }
                    std::cout << "fitLeastChiSquares1DForDataCube dryrun mode, not writing anything to \"outall\" at " << std::hex << outall << " from thread " << std::this_thread::get_id() << std::dec << std::endl;
                    if (NULL != mu) { mu->unlock(); }
                }
            }
        } else {
            gsl_matrix_set_col(outallMatrix, k, outall_vector);
        }
        //for (size_t q=0; q < (size_t)(nparams*2 + nchan*2 + 1); q++) {
        //    gsl_matrix_set(outallMatrix, q, k, gsl_vector_get(&outall_view.vector, q)); // irow (slow-changing), icol (fast-changing). index = irow * tda + icol.
        //}
        if (NULL != y_vector) gsl_vector_free(y_vector);
        if (NULL != yerr_vector) gsl_vector_free(yerr_vector);
        if (NULL != initparams_vector) gsl_vector_free(initparams_vector);
        if (NULL != outall_vector) gsl_vector_free(outall_vector);
    }
    // 
    if (GlobalDebug > 0) {
        if (NULL != mu) { 
            mu->lock();
            std::cout << "fitLeastChiSquares1DForDataCube finished from thread " << std::hex << std::this_thread::get_id() << std::dec << std::endl;
            mu->unlock();
        } else {
            std::cout << "fitLeastChiSquares1DForDataCube finished." << std::endl;
        }
    }
    return outall;
}



double *fitLeastChiSquares1DForDataCubeWithMultiThread(\
    double *x, 
    double *data, 
    double *dataerr, 
    long nx, 
    long ny, 
    long nchan, 
    double *initparamsall, 
    long nparams, 
    int maxniter, 
    int verbose, 
    int nthread)
{
    /* Run the fitting, return outparams + outparamerrs + outyfitted + outyresidual.
    
    The input data must be a normal FITS 3D data cube, that is, the 1-st (lowest, fatest-changing) dimension is image width, 2-nd dimension image height, and 3-rd (highest, slowest-changing) dimension channels. We will do a transpose to increase the speed.   
    
    Note that outparams 1-st (lowest, fatest-changing) dimension is parameters, then 2-nd dimension image height, then 3-rd dimension image width. Its transpose is the normal FITS 3D data cube.   
    */
    if (GlobalDebug > 0) {
        std::cout << "fitLeastChiSquares1DForDataCubeWithMultiThread is called." << std::endl;
    }
    //
    // create threads
    std::vector<LeastChiSquares1D *> my_least_chisq_fitters;
    std::vector<std::thread> my_threads;
    size_t k = 0;
    size_t nstep = 0;
    double *outall = NULL;
    if ( (GlobalDebug > 0) && (verbose >= 1) ) {
        std::cout << "fitLeastChiSquares1DForDataCubeWithMultiThread allocating memory for \"outall\" output array" << std::endl;
    }
    outall = (double *)malloc((nparams*ny*nx*2 + nchan*ny*nx*2 + ny*nx)*sizeof(double)); // a complex output array in the memory, =  outparams + outparamerrs + outyfitted + outyresidual
    for (k=0; k < (size_t)(nparams*ny*nx*2 + nchan*ny*nx*2 + ny*nx); k++) { outall[k] = std::nan(""); }
    std::mutex mu; 
    if (nthread <= 0) { nthread = 1; }
    if ((size_t)nthread > (size_t)(ny*nx)) { nthread = (ny*nx); }
    nstep = (size_t)((ny*nx)/nthread); // dividing the loop range 0-(ny*nx-1) into strides for 'nthread' threads
    struct MultiThreadPayloadStruct *my_thread_payloads = new struct MultiThreadPayloadStruct[nthread];
    for (k=0; k < (size_t)nthread; k++) {
        // create class
        LeastChiSquares1D *my_least_chisq_fitter = new LeastChiSquares1D();
        my_least_chisq_fitter->setDebugLevel(GlobalDebug);
        my_least_chisq_fitters.push_back(my_least_chisq_fitter);
        
        if ( (GlobalDebug > 0) && (verbose >= 1) ) {
            std::cout << "fitLeastChiSquares1DForDataCubeWithMultiThread my_least_chisq_fitter->setDebugLevel(" << GlobalDebug << ")" << std::endl;
        }
        //std::cout << "fitLeastChiSquares1DForDataCubeWithMultiThread my_least_chisq_fitter->debugLevel() = " << my_least_chisq_fitter->debugLevel() << std::endl;
        
        // prepare mthread payload
        struct MultiThreadPayloadStruct *mthread = &my_thread_payloads[k];
        mthread->kfirst = k*nstep;
        mthread->kstep = 1;
        mthread->outptr = outall;
        mthread->mu = &mu; // TODO
        mthread->dryrun = 0; // for some debugging purpose, setting dryrun=1 will not write data into outptr
        if (k < (size_t)nthread - 1) {
            mthread->klast = (k+1)*nstep - 1;
        } else {
            mthread->klast = (size_t)(ny*nx) - 1;
            //mthread->dryrun = 1; // debugging, letting the last thread not writing anything to outptr
        }
        
        if ( (GlobalDebug > 0) || (verbose >= 1) ) {
            mu.lock();
            std::cout << "fitLeastChiSquares1DForDataCubeWithMultiThread creating thread " << k+1 << " / " << nthread << ", kfirst " << mthread->kfirst << ", klast " << mthread->klast << " / " << (size_t)(ny*nx)-1 << ", mthread " << std::hex << mthread << std::dec << "." << std::endl;
            mu.unlock();
        }
        
        // create thread to run the fitting
        std::thread my_thread(\
            fitLeastChiSquares1DForDataCube, \
            (void *)my_least_chisq_fitter, 
            x, data, dataerr, nx, ny, nchan, 
            initparamsall, nparams, 
            maxniter, verbose, 
            mthread);
        
        my_threads.push_back(std::move(my_thread));
    }
    
    for (std::thread & th : my_threads) {
        if (th.joinable()) {
            if ( (GlobalDebug > 0) || (verbose >= 1) ) {
                mu.lock();
                std::cout << "fitLeastChiSquares1DForDataCubeWithMultiThread joining thread " << std::hex << &th << std::dec << std::endl;
                mu.unlock();
            }
            th.join();
        }
    }
    
    // clean up
    for (k=0; k < my_least_chisq_fitters.size(); k++) {
        delete my_least_chisq_fitters[k];
        my_least_chisq_fitters[k] = NULL;
    }
    
    // 
    if (GlobalDebug > 0) {
        std::cout << "fitLeastChiSquares1DForDataCubeWithMultiThread finished." << std::endl;
    }
    return outall;
}



void destroyLeastChiSquares1D(void *ptr)
{
    /* This function is not used. */
    
    if (GlobalDebug > 0) 
        std::cout << "destroyLeastChiSquares1D is called." << std::endl;
    //
    LeastChiSquares1D *my_least_chisq_fitter = (LeastChiSquares1D *)ptr;
    
    if (GlobalDebug > 0) {
        std::cout << "my_least_chisq_fitter " << my_least_chisq_fitter << std::endl;
        std::cout << "my_least_chisq_fitter->debugLevel() " << my_least_chisq_fitter->debugCode() << std::endl;
    }
    
    if (my_least_chisq_fitter) 
        delete my_least_chisq_fitter;
    
    my_least_chisq_fitter = NULL;
    ptr = NULL;
    
    if (GlobalDebug > 0) 
        std::cout << "destroyLeastChiSquares1D finished." << std::endl;
}



void freeDataArrayMemory(double *arr)
{
    /* This function is exposed to Python to free the memory of 
       the data array "outall" created in the function 
       "fitLeastChiSquares1DForDataCubeWithMultiThread". */
       
    if (GlobalDebug > 0) 
        std::cout << "freeDataArrayMemory is called." << std::endl;
    
    if (arr) {
        free(arr);
        arr = NULL;
    }
        
}





