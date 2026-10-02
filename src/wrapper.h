#ifndef RESTREE_WRAPPER_H
#define RESTREE_WRAPPER_H
// Main-thread R/C++ conversions shared by responsibility-specific entry points.
#include "ResTree.h"
#include "prediction.h"

#include <chrono>

class SMC;




#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

// Shared main-thread validation; implementation lives in wrapper.cpp.
void restree_validate_covariance(int dimension, double sig2, const Eigen::VectorXd& range,
    double tail, double nu, double nugget, const std::string& form,
    const std::string& family, const std::string& dtype);
void restree_validate_ncores(int& ncores);

struct PredictiveWeights {
	Eigen::VectorXd probability, log_probability;
};

// Center before squaring: subtracting large second moments loses small
// uncertainty even for one component. Called outside all worker regions.
template<class Mean, class Var>
void predictive_moments(const Mean& pm, const Var& pv, const Eigen::VectorXd& w,
	Eigen::VectorXd& mean, Eigen::VectorXd& variance){
	if(pm.rows()!=pv.rows() || pm.cols()!=pv.cols() || pm.cols()!=w.size())
		Rcpp::stop("component moments and weights must have matching dimensions.");
	Eigen::Index anchor;
	w.maxCoeff(&anchor);
	long double total=0.0L;
	for(Eigen::Index k=0; k<w.size(); ++k) total += w(k);
	mean.resize(pm.rows()); variance.resize(pm.rows());
	for(Eigen::Index i=0; i<pm.rows(); ++i){
		const long double origin=pm(i,anchor);
		long double shift=0.0L, v=0.0L;
		for(Eigen::Index k=0; k<w.size(); ++k){
			if(w(k)==0.0) continue;
			if(!std::isfinite(pm(i,k)) || !std::isfinite(pv(i,k)) || pv(i,k)<0.0)
				Rcpp::stop("a positive-weight component has invalid predictive moments.");
			shift += (long double)w(k)*((long double)pm(i,k)-origin);
		}
		shift /= total;
		mean(i)=(double)(origin+shift);
		for(Eigen::Index k=0; k<w.size(); ++k){
			if(w(k)==0.0) continue;
			const long double delta=((long double)pm(i,k)-origin)-shift;
			v += (long double)w(k)*((long double)pv(i,k)+delta*delta);
		}
		variance(i)=(double)(v/total);
	}
}
// Borrow slot data through protected R handles for this main-thread call.
// New owning engines copy X/y once; a valid Full cache reuses its owned copy.
// Pass maps to Eigen::Ref, templates or move-in parameters: binding to a
// const MatrixXd& would silently construct a temporary.
struct ModelSpec {
	Rcpp::NumericMatrix x_sexp;   // protects the slot (or its one-time REALSXP coercion)
	Rcpp::NumericVector y_sexp;
	Eigen::Map<const Eigen::MatrixXd> X;
	Eigen::Map<const Eigen::VectorXd> y;
	int depth = 1, r = 1;
	std::string leaf_model, design, cut_method;
	double prior_rho = 0.05, prior_beta = 2.0;
	int cut_candidates = 30; // ordinary PP uniform: complementary draws per coordinate
	int seed = 42;          // seed of the fixed Full tree's random cuts (model slot)
	int nested_factor = 2;  // candidates per node under design "nested", as a multiple of r
	Rcpp::List tree;        // the stored fixed tree (Full; empty for PP / WN)

	ModelSpec(Rcpp::NumericMatrix x_, Rcpp::NumericVector y_)
		: x_sexp(x_), y_sexp(y_),
		  X(x_sexp.begin(), x_sexp.nrow(), x_sexp.ncol()),
		  y(y_sexp.begin(), y_sexp.size()) {}
};

PredictiveWeights predictive_weights(const Eigen::VectorXd& w);
double predictive_log_mixture(const Eigen::VectorXd& log_density,const PredictiveWeights& weights);
GPM read_theta(SEXP theta_sexp,int p);
SEXP smc_handle_tag();
SEXP prediction_handle_tag();
SMC& smc_from_handle(SEXP handle);
PredictionState& prediction_from_handle(SEXP handle);
SEXP make_prediction_handle(ResTree&& model,FittedTrees&& trees,SEXP info=R_NilValue);
Rcpp::List prediction_state_info(SEXP handle);
Rcpp::List smc_engine_info_cpp(SEXP handle);
Rcpp::List smc_info(const SMC& engine);
ModelSpec read_model(SEXP model_sexp);
void validate_model_data(const ModelSpec& sp);
int restree_default_particles(const ModelSpec& sp);
void configure_threads(ResTree& m,int ncores);
void reapply_threads(const ResTree& m);
void configure_fixed(ResTree& m,const ModelSpec& sp,const GPM& th,int ncores);
ResTree make_model(const ModelSpec& sp,const GPM& th,int ncores);
void configure_smc(SMC& eng,const ModelSpec& sp,int n_particles,
    const std::string& resampling,double temper_alpha,int seed,int ncores);
Rcpp::List geometry_to_list(const FullTreeGeometry& geom);
std::shared_ptr<FullTreeGeometry> geometry_from_spec(const ModelSpec& sp,const char* who);
StructureEval cached_full_loglik(const ModelSpec& sp,const GPM& th,int ncores);
Rcpp::DataFrame level_diag_frame(const SMC& eng);
Rcpp::DataFrame nodes_frame(const SMC& eng,const std::vector<int>& reps);
int particle_internal_knots(const SMC& eng,int p);
#endif
