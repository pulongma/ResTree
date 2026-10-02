#ifndef _USE_RcppEigen
#define _USE_RcppEigen
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp17)]]
#endif

#ifndef _USE_GSL
#define _USE_GSL
#include <gsl/gsl_sf.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_errno.h>
#endif


#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include <mutex>
#include <atomic>
#include <vector>
#include <cfloat>
#include <limits>

#ifndef _USE_GP
#define _USE_GP
#include "GPutils.h"
#endif

namespace {

// GSL's error handler is process-global.  Changing it concurrently from OpenMP
// workers is a data race, so install the non-aborting handler exactly once.
std::once_flag gsl_error_handler_once;
std::atomic<int> covariance_gsl_status(GSL_SUCCESS);
void ensure_gsl_error_handler_off(){
  std::call_once(gsl_error_handler_once, [](){ gsl_set_error_handler_off(); });
}

void record_covariance_error(int status){
  if(status==GSL_SUCCESS){ status=GSL_EFAILED; }
  int expected=GSL_SUCCESS;
  covariance_gsl_status.compare_exchange_strong(expected, status);
}

bool hyperg_u_e10(const double a, const double b, const double x,
                  gsl_sf_result_e10& result){
  ensure_gsl_error_handler_off();
  const int status=gsl_sf_hyperg_U_e10_e(a,b,x,&result);
  if(status!=GSL_SUCCESS || !std::isfinite(result.val) || result.val<=0.0){
    record_covariance_error(status);
    return false;
  }
  return true;
}

}

void record_cholesky_error(){ record_covariance_error(-1001); }

bool checked_cholesky(const Eigen::MatrixXd& V, Eigen::MatrixXd& L){
  Eigen::LLT<Eigen::MatrixXd> llt(V);
  if(llt.info()!=Eigen::Success){
    L.setConstant(V.rows(),V.cols(),std::numeric_limits<double>::quiet_NaN());
    record_cholesky_error(); return false;
  }
  L = llt.matrixL();
  if(!L.allFinite() || (L.diagonal().array()<=0.0).any()){
    record_cholesky_error(); return false;
  }
  return true;
}

void reset_covariance_error(){
  covariance_gsl_status.store(GSL_SUCCESS);
}

bool covariance_error_occurred(){
  return covariance_gsl_status.load()!=GSL_SUCCESS;
}

int covariance_error_status(){
  return covariance_gsl_status.load();
}

void stop_on_covariance_error(const std::string& context){
  const int status=covariance_error_status();
  if(status==-1001){
    Rcpp::stop(context+": Cholesky factorization failed: covariance is not numerically positive definite. Check duplicate inputs, range and nugget; no jitter was added.");
  }
  if(status!=GSL_SUCCESS){
    Rcpp::stop(context+": covariance evaluation failed (status "+
      std::to_string(status)+", "+std::string(gsl_strerror(status))+").");
  }
}

/****************************************************************************************/
/* Distance Matrices  */
/****************************************************************************************/

double gcdistance(const double& lonA, const double& latA, const double& lonB, const double& latB)
{
  
  // Given the latitude and longitude of two places, this function
  // computes great circle distance between them by using haversine
  // formula, which is well-conditioned for small distance
  
  double dist_lat = 0;
  double dist_lon = 0;
  double dTemp = 0;
  double temp_lon = 0;
  double dist = 0;
  double earthRadius = 6371.0087714; //WGS84 mean radius in km 
  double dTemp1 = 0;
  double dTemp2 =0;

  dist_lat = std::fabs(latA - latB);
  temp_lon = std::fabs(lonA - lonB);
  dist_lon = (temp_lon<180.0)?(temp_lon):(360.0-temp_lon);
  dTemp1 = sin(0.5*dist_lat*M_PI/180.0);
  dTemp2 = sin(0.5*dist_lon*M_PI/180.0);
  dTemp = dTemp1*dTemp1 + cos(latA*M_PI/180.0)*cos(latB*M_PI/180.0)*dTemp2*dTemp2;
  
  dist = earthRadius * 2.0 * atan2(sqrt(dTemp), sqrt(1.0 - dTemp));
  
  return dist;
}

// Modified Bessel function K_nu(z) through GSL. The non-aborting handler is
// installed, but this wrapper currently discards the returned status; unlike
// hyperg_u_e10 it does not record special-function failure in the error flag.
double BesselK(const double& nu, const double& z){

  gsl_sf_result result;
  ensure_gsl_error_handler_off();
  gsl_sf_bessel_Knu_e(nu, z, &result);

  return result.val;

}

Eigen::MatrixXd CH(const Eigen::MatrixXd & d, const double & range, const double & tail, const double & nu) {
  ensure_gsl_error_handler_off();
  const double log_con=gsl_sf_lngamma(nu+tail)-gsl_sf_lngamma(nu);
  const double log_dbl_min=std::log(DBL_MIN);
  const double log_ten=std::log(10.0);

  int n1 = d.rows();
  int n2 = d.cols();
  
  Eigen::MatrixXd covmat(n1, n2);
  covmat.setOnes();
  if(range==0.0){ // limiting case with range=0
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        if(d(i,j)!=0.0){
          covmat(i,j)=0.0;
        }
      }
    }
  }else{
    double dtemp;
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        if(d(i,j)!=0.0){
          dtemp = d(i,j) / range;
          gsl_sf_result_e10 result;
          if(!hyperg_u_e10(tail,1.0-nu,dtemp*dtemp,result)){
            covmat(i,j)=NA_REAL;
            continue;
          }
          const double log_cov=log_con+std::log(result.val)+
            static_cast<double>(result.e10)*log_ten;
          if(!std::isfinite(log_cov)){
            record_covariance_error(GSL_EFAILED);
            covmat(i,j)=NA_REAL;
          }else if(log_cov<log_dbl_min){
            // A successful correlation below the double range is numerically zero.
            covmat(i,j)=0.0;
          }else if(log_cov>1e-10){
            // The normalized CH correlation is in [0,1]; a material excess means
            // that the special-function evaluation is not numerically trustworthy.
            record_covariance_error(GSL_EFAILED);
            covmat(i,j)=NA_REAL;
          }else{
            covmat(i,j)=std::exp(std::min(0.0,log_cov));
          }
        }
      }
    }
  }
  
  
  return covmat;
}





Eigen::MatrixXd matern(const Eigen::MatrixXd& d, const double & range, const double & nu){
  // Column-major traversal (j outer, i inner), the order the matrix is laid
  // out, so each element is produced once from contiguous memory.
  const int n1 = d.rows();
  const int n2 = d.cols();

  Eigen::MatrixXd covmat(n1, n2);
  double tau=0.0;

  double dtemp, con1;

  if(range==0.0){ // limiting case with range=0
    covmat.setOnes();
    for(int j=0; j<n2; j++){
      for(int i=0; i<n1; i++){
        if(d(i,j)!=0.0){
          covmat(i,j)=0.0;
        }
      }
    }
  }else{
    if(nu==0.5){
      for(int j=0; j<n2; j++){
        for(int i=0; i<n1; i++){
          covmat(i,j) = exp(-d(i,j)/range);
        }
      }
    }else if(nu==1.5){
      for(int j=0; j<n2; j++){
        for(int i=0; i<n1; i++){
          dtemp =  d(i,j)/range;
          covmat(i,j) = (1.0+dtemp) * exp(-dtemp);
        }
      }
    }else if(nu==2.5){
      for(int j=0; j<n2; j++){
        for(int i=0; i<n1; i++){
          dtemp =  d(i,j)/range;
          covmat(i,j) = (1.0 + dtemp + dtemp*dtemp/3.0) * exp(-dtemp);
        }
      }
    }else{
      con1 = pow(2.0, 1.0-nu);
      for(int j=0; j<n2; j++){
        for(int i=0; i<n1; i++){
          if(d(i,j)==0.0){
            covmat(i,j) = 1.0;
          }else{
            tau = d(i,j)/range;
            covmat(i,j) = (con1 / tgamma(nu)) * pow(tau, nu) * BesselK(nu, tau);
          }
        }
      }
    }
  }

  return covmat;
}



Eigen::MatrixXd powexp(const Eigen::MatrixXd& d, const double& range, const double& nu){
  int n1 = d.rows();
  int n2 = d.cols();
  
  Eigen::MatrixXd covmat(n1, n2);

  if(nu==2.0){
   for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        covmat(i,j) = exp(-(d(i,j)/range)*(d(i,j)/range));
      }
    }
  }else{

    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        covmat(i,j) = exp(-pow(d(i,j)/range, nu));
      }
    }
  }

  return covmat;
}


Eigen::MatrixXd cauchy(const Eigen::MatrixXd& d, const double& range, const double& tail, const double& nu){

  int n1 = d.rows();
  int n2 = d.cols();
  
  Eigen::MatrixXd covmat(n1, n2);
  
  if(nu==2.0){
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        covmat(i, j) = pow(1.0 + (d(i,j)/range)*(d(i,j)/range), -tail/2.0);
      }
    }    
  }else{
    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        covmat(i, j) = pow(1.0 + pow(d(i,j)/range, nu), -tail/nu);
      }
    }    
  }

  
  
  return covmat;
}


/****************************************************************************************/
/****************************************************************************************/

/****************************************************************************************/
/****************************************************************************************/

Eigen::MatrixXd  pdist(const Eigen::MatrixXd& locs1, const Eigen::MatrixXd& locs2, 
  const std::string& dtype){


  int n1 = locs1.rows();
  int n2 = locs2.rows();
  Eigen::MatrixXd distmat(n1, n2); 
  

  if(dtype=="GCD"){

    for(int i=0; i<n1; i++){
      for(int j=0; j<n2; j++){
        distmat(i,j) = gcdistance(locs1(i,0), locs1(i,1), locs2(j,0), locs2(j,1));
      }
    }

  }else if(dtype=="Euclidean"){
    // ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x'y: one matrix product instead of
    // a per-pair row expression.
    const Eigen::VectorXd norm1 = locs1.rowwise().squaredNorm();
    if(locs1.data()==locs2.data() && n1==n2){
      // Self distance: symmetric BY CONSTRUCTION.  Pre-fill n_i + n_j (a
      // commutative sum, so exactly symmetric), subtract 2 X X' as a
      // symmetric rank-k update (syrk) of the LOWER triangle only, then
      // mirror the lower triangle to the upper.  A general GEMM evaluates
      // (i,j) and (j,i) as different dot-product accumulations and leaves
      // the matrix symmetric only to the last bit.
      distmat.colwise() = norm1;
      distmat.rowwise() += norm1.transpose();
      sym_rank_update(distmat, locs1, -2.0);
      distmat = distmat.array().max(0.0).sqrt().matrix(); // set all negatives to 0.0 while preserve positives
      distmat.diagonal().setZero();
    }else{
      const Eigen::VectorXd norm2 = locs2.rowwise().squaredNorm();
      distmat.noalias() = -2.0 * locs1 * locs2.transpose();
      distmat.colwise() += norm1;
      distmat.rowwise() += norm2.transpose();
      distmat = distmat.array().max(0.0).sqrt().matrix();
    }
  }else{
    // Reachable from OpenMP workers: never throw; record the deferred flag,
    // return zeros, and let the serial caller raise the error.
    record_covariance_error(GSL_EINVAL);
    distmat.setZero();
  }

  return distmat;
}

Eigen::MatrixXd iso_kernel(const Eigen::MatrixXd& d, const double& range, const double& tail, 
  const double& nu, const std::string& family){

  // Assigned by the family branch below; no separate allocation (the
  // branches return a matrix that is moved in).
  Eigen::MatrixXd cormat;

  if(family=="CH"){
    cormat = CH(d, range, tail, nu);

  }else if(family=="matern"){

    cormat = matern(d, range, nu);

  }else if(family=="gauss"){
    cormat = powexp(d, range, 2.0);

  }else if(family=="powexp"){
    cormat = powexp(d, range, nu);   
     
  }else if(family=="cauchy"){
    cormat = cauchy(d, range, tail, nu);

  }else{
    // worker-safe deferred error (see pdist): record, return zeros.
    record_covariance_error(GSL_EINVAL);
    cormat = Eigen::MatrixXd::Zero(d.rows(), d.cols());
  }

  return cormat;

}


void adist(Eigen::MatrixXd* dist, const Eigen::MatrixXd& input1, const Eigen::MatrixXd& input2, const std::string dtype){ 
  
  int Dim_x = input1.cols();


  for(int k=0; k<Dim_x; k++){
    dist[k] = pdist(input1.col(k), input2.col(k), dtype);
  }


  return;
}


// anisotropic covariance function
Eigen::MatrixXd ARD_kernel(Eigen::MatrixXd* d, const Eigen::VectorXd& range,  
  const double& tail, const double& nu, const std::string& family){

  int n1, n2, Dim;

  Eigen::MatrixXd dtemp, covmat, covtemp;

  Dim = range.size();
  n1 = d[0].rows();
  n2 = d[0].cols();
  covmat = Eigen::MatrixXd::Ones(n1, n2);
  
  Eigen::MatrixXd scaleddist(n1,n2);
  scaleddist.setZero();
  for(int i=0; i<Dim; i++){
    scaleddist.array() +=  (d[i].array()/range(i)) * (d[i].array()/range(i));
  }
  scaleddist = scaleddist.array().sqrt();

  if(family=="CH"){

    covmat = CH(scaleddist, 1.0, tail, nu);

  }else if(family=="cauchy"){

    covmat = cauchy(scaleddist, 1.0, tail, nu);

  }else if(family=="matern"){

    covmat = matern(scaleddist, 1.0, nu);

  }else if(family=="powexp"){

    covmat = powexp(scaleddist, 1.0, nu);

  }else if(family=="gauss"){

    covmat = powexp(scaleddist, 1.0, 2.0);

  }else{
    // worker-safe deferred error (see pdist): record, return zeros.
    record_covariance_error(GSL_EINVAL);
    covmat.setZero();
  }

  return covmat;
}



Eigen::MatrixXd ikernel(const Eigen::MatrixXd& input1, const Eigen::MatrixXd& input2, const GPM& covpar){
  
  Eigen::MatrixXd mat;
  if(covpar.form=="isotropic"){
    Eigen::MatrixXd dmat = pdist(input1, input2, covpar.dtype);
    mat = iso_kernel(dmat, covpar.range(0), covpar.tail, covpar.nu, covpar.family);
  }else if(covpar.form=="tensor" && covpar.family!="gauss"){
    // Separable product of correlations, NOT of covariance matrices: sig2
    // and the nugget are applied once by the caller. Reuse one distance
    // matrix across dimensions instead of retaining d full matrices.
    mat = Eigen::MatrixXd::Ones(input1.rows(), input2.rows());
    Eigen::MatrixXd distance(input1.rows(), input2.rows());
    for(Eigen::Index k=0; k<input1.cols(); ++k){
      for(Eigen::Index j=0; j<input2.rows(); ++j)
        for(Eigen::Index i=0; i<input1.rows(); ++i)
          distance(i,j) = std::abs(input1(i,k)-input2(j,k));
      mat.array() *= iso_kernel(distance, covpar.range(k), covpar.tail,
                               covpar.nu, covpar.family).array();
    }
  }else if(covpar.form=="ARD" || covpar.form=="tensor"){
    // A Gaussian tensor product equals exp(-sum_j h_j^2): use the same
    // scaled-distance fast path as ARD, including identical roundoff.
    if(covpar.dtype=="Euclidean"){
      // ARD is Euclidean distance after column scaling.  One scaled matrix
      // distance avoids allocating one full distance matrix per dimension.
      Eigen::MatrixXd scaled1 = input1;
      scaled1.array().rowwise() /= covpar.range.transpose().array();
      Eigen::MatrixXd dmat;
      if(input1.data()==input2.data() && input1.rows()==input2.rows()){
        // Self kernel: hand pdist ONE matrix so it takes its symmetric
        // (syrk + mirrored lower triangle) branch.
        dmat = pdist(scaled1, scaled1, "Euclidean");
      }else{
        Eigen::MatrixXd scaled2 = input2;
        scaled2.array().rowwise() /= covpar.range.transpose().array();
        dmat = pdist(scaled1, scaled2, "Euclidean");
      }
      mat = iso_kernel(dmat, 1.0, covpar.tail, covpar.nu, covpar.family);
    }else{
      int dim = input1.cols();
      // std::vector, not raw new[]: exception-safe by construction.
      std::vector<Eigen::MatrixXd> dist(dim);
      adist(dist.data(), input1, input2, covpar.dtype);
      mat = ARD_kernel(dist.data(), covpar.range, covpar.tail, covpar.nu, covpar.family);
    }
  }else{
    // worker-safe deferred error (see pdist): record, return zeros.
    record_covariance_error(GSL_EINVAL);
    mat = Eigen::MatrixXd::Zero(input1.rows(), input2.rows());
  }

  return mat;
}








/**********************************************************************/

void seq_maximin(const Eigen::MatrixXd& input, int r_knot,  
  Eigen::Ref<Eigen::VectorXi> neigh_ind,  Eigen::Ref<Eigen::VectorXi> input_ind){


    int n = input.rows();
    // mean location as the first reference 
    Eigen::RowVectorXd mean_loc = input.colwise().mean(); 


    Eigen::VectorXi ind_all = Eigen::VectorXi::LinSpaced(n,0,n-1);
    Eigen::Array<bool, Eigen::Dynamic, 1> input_flag(n);
    input_flag.setConstant(true);

    if(r_knot==0){
      input_ind = ind_all;
      return;
    }

    (input.rowwise() - mean_loc).rowwise().squaredNorm().minCoeff(&neigh_ind(0)); 
    input_flag.row(neigh_ind(0)) = false;

    if(r_knot>1){
      // Maintain the distance to the nearest selected knot; updating it with
      // only the newest knot costs O(n*r*p) over the whole selection.
      Eigen::VectorXd dmin =
        (input.rowwise() - input.row(neigh_ind(0))).rowwise().squaredNorm();
      dmin(neigh_ind(0)) = -1.0;
      for(int k=1; k<r_knot; k++){
        dmin.maxCoeff(&neigh_ind(k));
        input_flag(neigh_ind(k)) = false;
        const Eigen::VectorXd dnew =
          (input.rowwise() - input.row(neigh_ind(k))).rowwise().squaredNorm();
        for(int i=0; i<n; i++){
          if(input_flag(i) && dnew(i)<dmin(i)){ dmin(i)=dnew(i); }
        }
        // Excluding selected rows also prevents repeated selection when the
        // locations contain exact duplicates and every remaining nearest
        // distance is zero.
        dmin(neigh_ind(k)) = -1.0;
      }
    }

    int count = 0;
    for(int i=0; i<n; i++){

      if(input_flag(i)){
        input_ind(count) = ind_all(i);
        count += 1;
      }
    }

    return;
}

// [[Rcpp::export]]
Rcpp::List find_neigh(const Eigen::MatrixXd& input, int r_knot){
    int n = input.rows();
    if(r_knot<0 || r_knot>n){ Rcpp::stop("r_knot must be between zero and n."); }
    Eigen::VectorXi neigh_ind(r_knot);
    Eigen::VectorXi input_ind(n-r_knot);
    seq_maximin(input, r_knot, neigh_ind, input_ind);
    return Rcpp::List::create(Rcpp::_["neigh_ind"]=neigh_ind,
      Rcpp::_["input_ind"]=input_ind
      );
}


// exact greedy maximin ordering; deterministic ties are broken by the smallest index
Eigen::VectorXi maximin_order(const Eigen::MatrixXd& input){


    int n = input.rows();

    Eigen::VectorXi ord(n);

    // mean location as the first reference
    Eigen::RowVectorXd mean_loc = input.colwise().mean();

    // start from the input closest to the mean location
    (input.rowwise() - mean_loc).rowwise().squaredNorm().minCoeff(&ord(0));

    // squared distance to the current selected set
    Eigen::VectorXd mindist = (input.rowwise() - input.row(ord(0))).rowwise().squaredNorm();
    mindist(ord(0)) = -1.0; // exclude selected inputs

    Eigen::VectorXd dnew(n);

    for(int k=1; k<n; k++){
        mindist.maxCoeff(&ord(k));
        dnew = (input.rowwise() - input.row(ord(k))).rowwise().squaredNorm();
        mindist = mindist.cwiseMin(dnew);
        mindist(ord(k)) = -1.0;
    }

    return ord;
}








/**********************************************************************/
