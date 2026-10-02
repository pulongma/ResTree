// -*- mode: c++; -*-

#ifndef _USE_GPutils
#define _USE_GPutils

struct GPM {
      double sig2=1.0;
      Eigen::VectorXd coef=Eigen::VectorXd::Zero(1);
      Eigen::VectorXd range=Eigen::VectorXd::Ones(1);
      double tail=0.5;
      double nu=0.5;
      double nugget=0.0;
      std::string form="ARD";
      std::string family="matern";
      std::string dtype="Euclidean";
};

/****************************************************************************************/
/* Symmetric-by-construction helpers                                                    */
/****************************************************************************************/
// Every symmetric matrix the engines factorise is formed on its LOWER triangle
// by a symmetric rank-k update (syrk) and then mirrored to the upper triangle,
// so that the matrix is symmetric EXACTLY (not to the last bit) and every code
// path factorises literally the same numbers.

// V(lower) += alpha * A * A'; then V = mirror of its lower triangle.
template<class Derived>
inline void sym_rank_update(Eigen::MatrixXd& V, const Eigen::MatrixBase<Derived>& A, double alpha){
  V.selfadjointView<Eigen::Lower>().rankUpdate(A, alpha);
  V.triangularView<Eigen::StrictlyUpper>() = V.transpose();
}

// V -= Q' Q with Q (q x r), i.e. the symmetric downdate of a knot block by the
// propagated coefficients.  rankUpdate takes the r x q matrix Q'.
inline void sym_downdate(Eigen::MatrixXd& V, const Eigen::MatrixXd& Q){
  sym_rank_update(V, Q.transpose(), -1.0);
}

/****************************************************************************************/
/* Distance Matrices  */
/****************************************************************************************/

double gcdistance(const double& lonA, const double& latA, const double& lonB, const double& latB);
double BesselK(const double& nu, const double& z);
// Hypergeometric-U is evaluated inside OpenMP workers.  A worker must never
// call into R's error machinery, so numerical failures are recorded atomically
// and reported by the exported caller after the parallel region has ended.
void reset_covariance_error();
void record_cholesky_error();
bool checked_cholesky(const Eigen::MatrixXd& V, Eigen::MatrixXd& L);
bool covariance_error_occurred();
int covariance_error_status();
void stop_on_covariance_error(const std::string& context);

// correlation functions
Eigen::MatrixXd CH(const Eigen::MatrixXd& d, const double & range, const double & tail, const double & nu);
Eigen::MatrixXd matern(const Eigen::MatrixXd& d, const double & range, const double & nu);
Eigen::MatrixXd powexp(const Eigen::MatrixXd& d, const double& range, const double& nu);
Eigen::MatrixXd cauchy(const Eigen::MatrixXd& d, const double& range, const double& tail, const double& nu);

Eigen::MatrixXd  pdist(const Eigen::MatrixXd& locs1, const Eigen::MatrixXd& locs2, 
  const std::string& dtype);
Eigen::MatrixXd iso_kernel(const Eigen::MatrixXd& d, const double& range, const double& tail, 
  const double& nu, const std::string& family);

void adist(Eigen::MatrixXd* dist, const Eigen::MatrixXd& input1, const Eigen::MatrixXd& input2, const std::string dtype);
Eigen::MatrixXd ARD_kernel(Eigen::MatrixXd* d, const Eigen::VectorXd& range,  
  const double& tail, const double& nu, const std::string& family);
// Correlation matrix between the rows of input1 and input2 (sig2 and the
// nugget are the caller's). Covariance options, including ARD and tensor,
// share this interface; the parameter reference avoids copying.
Eigen::MatrixXd ikernel(const Eigen::MatrixXd& input1, const Eigen::MatrixXd& input2, const GPM& covpar);

/****************************************************************************************/
/****************************************************************************************/


/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/

void seq_maximin(const Eigen::MatrixXd& input, int r_knot, 
  Eigen::Ref<Eigen::VectorXi> neigh_ind, Eigen::Ref<Eigen::VectorXi> input_ind);
Rcpp::List find_neigh(const Eigen::MatrixXd& input, int r_knot);

Eigen::VectorXi maximin_order(const Eigen::MatrixXd& input);

#endif
