// scores.cpp -- proper scoring rules of a mixture predictive (R/scores.R):
// the compiled core of restree_score() / restree_lppd() / restree_crps().
//
// A prediction at m points is a mixture over K distinct trees with weights
// w_k (summing to 1 after zero-weight components are dropped); component k at
// point i is Gaussian N(mu_ik, var_ik) or, for WN leaves scored with
// wn_leaf = "student", Student-t with df_ik degrees of freedom and scale
// s_ik = sqrt(var_ik (df_ik - 2) / df_ik) (df = Inf denotes Gaussian).  Per
// point the entry point below computes, on request,
//   * the CRPS: Gaussian mixture in closed form (Grimit, Gneiting, Berrocal &
//     Johnson 2006, QJRMS 132:2925-2942),
//       CRPS = sum_k w_k A(y - mu_k, s_k^2) - 1/2 sum_k sum_l w_k w_l A(mu_k - mu_l, s_k^2 + s_l^2),
//       A(mu, s^2) = mu (2 Phi(mu/s) - 1) + 2 s phi(mu/s);
//     a single Student-t in closed form (Gneiting & Raftery 2007); a Student-t
//     mixture by adaptive quadrature of the squared mixture cdf,
//       CRPS = int_{-inf}^{y} F(x)^2 dx + int_{y}^{inf} (1 - F(x))^2 dx,
//     on the standardised scale, the line split at the component locations
//     (at most 32 quantiles of them) and at y, every piece integrated with
//     QUADPACK's dqags / dqagi through R's own C entry points Rdqags / Rdqagi
//     (the routines stats::integrate() calls) at rel.tol = rel_tol and
//     abs.tol = rel_tol / (number of pieces on that side), using R's pt();
//   * the logarithmic score -log sum_k w_k f_k(y) by log-sum-exp;
//   * the mixture cdf at y, F(y) = sum_k w_k F_k(y) (coverage: a < F(y) < b);
//   * mixture quantiles for the requested probabilities by bracketed
//     bisection on the monotone cdf over [min_k q_ik, max_k q_ik] (q_ik the
//     component quantiles), to a bracket width of 1e-8 max_k s_ik or the
//     rounding limit.
// Distribution and quadrature helpers call the R API. Keep the scoring loop
// on the main R thread, regardless of the engine's requested worker count.
// Native likelihood/fit/prediction parallelism is independent of scoring.
// Numerical integration failures are reported after each main-thread pass.

#include <Rcpp.h>
#include <R_ext/Applic.h>
#include "wrapper.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

const double kInf = std::numeric_limits<double>::infinity();

// ---- one component's distribution: Gaussian (df = Inf) or Student-t --------
inline double comp_cdf(double x, double mu, double sc, double df, bool lower){
	const double z = (x - mu) / sc;
	return std::isfinite(df) ? R::pt(z, df, lower, 0) : R::pnorm(z, 0.0, 1.0, lower, 0);
}
inline double comp_logpdf(double x, double mu, double sc, double df){
	const double z = (x - mu) / sc;
	if(std::isfinite(df) && !std::isfinite(z) && std::isfinite(x-mu)){
		const double logz=std::log(std::abs(x-mu))-std::log(sc);
		const double logratio=2*logz-std::log(df);
		return -std::log(sc)-0.5*std::log(df)-R::lbeta(0.5,df/2)
			-0.5*(df+1)*(logratio+std::log1p(std::exp(-logratio)));
	}
	return (std::isfinite(df) ? R::dt(z, df, 1) : R::dnorm(z, 0.0, 1.0, 1)) - std::log(sc);
}
inline double comp_quantile(double p, double mu, double sc, double df){
	return mu + sc * (std::isfinite(df) ? R::qt(p, df, 1, 0) : R::qnorm(p, 0.0, 1.0, 1, 0));
}

// A(mu, s^2) = mu (2 Phi(mu/s) - 1) + 2 s phi(mu/s), the Gaussian CRPS kernel
inline double crps_kernel(double mu, double s){
	const double z = mu / s;
	return mu * (2.0 * R::pnorm(z, 0.0, 1.0, 1, 0) - 1.0) + 2.0 * s * R::dnorm(z, 0.0, 1.0, 0);
}

// The components of one prediction point after dropping zero weights.
struct PointMixture {
	std::vector<double> mu, sc, var, df, w;   // sc: Gaussian sd or Student-t scale
	int K() const { return static_cast<int>(w.size()); }
	bool gaussian() const {
		for(double d : df){ if(std::isfinite(d)) return false; }
		return true;
	}
	double cdf(double x, bool lower = true) const {
		double s = 0.0;
		for(int k=0; k<K(); ++k){ s += w[k] * comp_cdf(x, mu[k], sc[k], df[k], lower); }
		return s;
	}
};

// ---- CRPS of a Gaussian mixture, closed form -------------------------------
double crps_gaussian_mixture(double y, const PointMixture& c){
	const int K = c.K();
	double t1 = 0.0;
	for(int k=0; k<K; ++k){ t1 += c.w[k] * crps_kernel(y - c.mu[k], c.sc[k]); }
	// the cross term is symmetric: the diagonal once, each pair twice
	double t2 = 0.0;
	for(int k=0; k<K; ++k){
		t2 += c.w[k] * c.w[k] * crps_kernel(0.0, std::hypot(c.sc[k], c.sc[k]));
		for(int l=k+1; l<K; ++l){
			t2 += 2.0 * c.w[k] * c.w[l] * crps_kernel(c.mu[k] - c.mu[l], std::hypot(c.sc[k], c.sc[l]));
		}
	}
	return t1 - 0.5 * t2;
}

// ---- CRPS of a single location-scale Student-t, closed form ----------------
// (Gneiting & Raftery 2007, eq. for the t; df > 2 for the WN NIG posterior)
double crps_single_t(double y, double mu, double sc, double df){
	if(!std::isfinite(df)){
		const double z = (y - mu) / sc;
		return sc * (z * (2.0 * R::pnorm(z, 0.0, 1.0, 1, 0) - 1.0) + 2.0 * R::dnorm(z, 0.0, 1.0, 0) - 1.0 / std::sqrt(M_PI));
	}
	const double z = (y - mu) / sc;
	const double c0 = 2.0 * std::sqrt(df) / (df - 1.0) * std::exp(R::lbeta(0.5, df - 0.5) - 2.0 * R::lbeta(0.5, df / 2.0));
	const double delta=y-mu;
	if(std::isfinite(z) && std::abs(z)<=std::sqrt(std::numeric_limits<double>::max()/2.0))
		return delta*(2.0*R::pt(z,df,1,0)-1.0)
			+ sc*(2.0*R::dt(z,df,0)*(df+z*z)/(df-1.0)-c0);
	// Compute the vanishing tail term without 0*Inf or an overflowing z^2.
	// log|z| remains representable even when delta/sc itself overflows.
	const double logz=std::log(std::abs(delta))-std::log(sc);
	const double logq=2.0*logz;
	const double logsum=logq+std::log1p(df*std::exp(-logq));
	const double logdens=-0.5*std::log(df)-R::lbeta(0.5,df/2.0)
		-0.5*(df+1.0)*(logsum-std::log(df));
	const double tail=std::exp(std::log(sc)+std::log(2.0)+logdens+logsum-std::log(df-1.0));
	return std::abs(delta)+tail-sc*c0;
}

// ---- CRPS of a Student-t mixture by quadrature of the squared cdf ----------
struct CdfIntegrand { const PointMixture* c; bool lower; };

// QUADPACK integrand: x[j] <- F(x[j])^2 (lower) or (1 - F(x[j]))^2 (upper), in place
void squared_cdf(double* x, int n, void* ex){
	const CdfIntegrand* g = static_cast<const CdfIntegrand*>(ex);
	for(int j=0; j<n; ++j){ const double F = g->c->cdf(x[j], g->lower); x[j] = F * F; }
}

// stats::integrate()'s messages for QUADPACK's ier codes
const char* quadpack_message(int ier){
	switch(ier){
		case 1: return "maximum number of subdivisions reached";
		case 2: return "roundoff error was detected";
		case 3: return "extremely bad integrand behaviour";
		case 4: return "roundoff error is detected in the extrapolation table";
		case 5: return "the integral is probably divergent";
		default: return "the input is invalid";
	}
}

// integrate f over (a, b), either end possibly infinite, as stats::integrate() does
void integrate_piece(integr_fn f, void* ex, double a, double b, double epsabs, double epsrel,
	int limit, std::vector<int>& iwork, std::vector<double>& work,
	double& value, double& abserr, int& ier){
	int neval = 0, last = 0, lenw = 4 * limit;
	ier = 0;
	if(std::isfinite(a) && std::isfinite(b)){
		Rdqags(f, ex, &a, &b, &epsabs, &epsrel, &value, &abserr, &neval, &ier, &limit, &lenw, &last, iwork.data(), work.data());
	}else{
		int inf; double bound;
		if(std::isfinite(a)){ inf = 1; bound = a; }
		else if(std::isfinite(b)){ inf = -1; bound = b; }
		else { inf = 2; bound = 0.0; }
		Rdqagi(f, ex, &bound, &inf, &epsabs, &epsrel, &value, &abserr, &neval, &ier, &limit, &lenw, &last, iwork.data(), work.data());
	}
}

// stats::quantile(x, probs, type = 7)
std::vector<double> quantile_type7(std::vector<double> x, const std::vector<double>& probs){
	std::sort(x.begin(), x.end());
	const int n = static_cast<int>(x.size());
	std::vector<double> q(probs.size());
	for(size_t i=0; i<probs.size(); ++i){
		const double index = 1.0 + (n - 1) * probs[i];
		const int lo = static_cast<int>(std::floor(index)), hi = static_cast<int>(std::ceil(index));
		double qs = x[lo - 1];
		if(index > lo && x[hi - 1] != qs){ const double h = index - lo; qs = (1.0 - h) * qs + h * x[hi - 1]; }
		q[i] = qs;
	}
	return q;
}

// Student-t mixture CRPS at one point: value and summed absolute error.
void crps_t_mixture(double y, const PointMixture& c, double rel_tol, int subdivisions,
	std::vector<int>& iwork, std::vector<double>& work, double& value, double& error, int& ier){
	const int K = c.K();
	value = 0.0; error = 0.0; ier = 0;
	// identical components collapse to one
	bool all_same = K > 1;
	for(int k=1; k<K && all_same; ++k){
		all_same = (c.mu[k]==c.mu[0] && c.sc[k]==c.sc[0] && c.df[k]==c.df[0]);
	}
	if(K == 1 || all_same){ value = crps_single_t(y, c.mu[0], c.sc[0], c.df[0]); return; }
	iwork.resize(subdivisions); work.resize(4*static_cast<size_t>(subdivisions));
	// standardise: unit = mixture sd, so the absolute tolerances are scale-free
	double center = 0.0; for(int k=0; k<K; ++k){ center += c.w[k] * c.mu[k]; }
	double v = 0.0; for(int k=0; k<K; ++k){ v += c.w[k] * (c.var[k] + (c.mu[k] - center) * (c.mu[k] - center)); }
	const double unit = std::sqrt(v);
	PointMixture s;
	s.w = c.w; s.df = c.df; s.mu.resize(K); s.sc.resize(K); s.var.resize(K);
	for(int k=0; k<K; ++k){ s.mu[k] = (c.mu[k] - center) / unit; s.sc[k] = c.sc[k] / unit; s.var[k] = c.var[k] / (unit * unit); }
	const double yy = (y - center) / unit;
	// component locations split the line into pieces (at most 32 quantiles of them)
	std::vector<double> knots = s.mu;
	if(K > 32){
		// seq(0, 1, length.out = 32) as R computes it: i * by with by = 1/31, the last exactly 1
		std::vector<double> pr(32);
		const double by = 1.0 / 31.0;
		for(int i=0; i<32; ++i){ pr[i] = i * by; }
		pr[31] = 1.0;
		knots = quantile_type7(s.mu, pr);
	}
	std::vector<double> left{-kInf, yy}, right{yy, kInf};
	for(double kn : knots){ if(kn < yy) left.push_back(kn); else if(kn > yy) right.push_back(kn); }
	auto sort_unique = [](std::vector<double>& b){ std::sort(b.begin(), b.end()); b.erase(std::unique(b.begin(), b.end()), b.end()); };
	sort_unique(left); sort_unique(right);
	auto accumulate = [&](const std::vector<double>& breaks, bool lower){
		CdfIntegrand g{&s, lower};
		const int npieces = static_cast<int>(breaks.size()) - 1;
		for(int j=0; j<npieces && ier==0; ++j){
			double val = 0.0, err = 0.0; int e = 0;
			integrate_piece(squared_cdf, &g, breaks[j], breaks[j+1], rel_tol / npieces, rel_tol, subdivisions, iwork, work, val, err, e);
			if(e != 0){ ier = e; return; }
			value += val; error += err;
		}
	};
	accumulate(left, true);
	if(ier == 0) accumulate(right, false);
	value *= unit; error *= unit;
}

// ---- mixture quantiles by bracketed bisection -------------------------------
double mixture_quantile(double p, const PointMixture& c, int maxit){
	const int K = c.K();
	double lo = kInf, hi = -kInf, smax = 0.0;
	for(int k=0; k<K; ++k){
		const double q = comp_quantile(p, c.mu[k], c.sc[k], c.df[k]);
		lo = std::min(lo, q); hi = std::max(hi, q); smax = std::max(smax, c.sc[k]);
	}
	const double tol = 1e-8 * smax;
	for(int it=0; it<maxit; ++it){
		const double mid = 0.5 * lo + 0.5 * hi;
		if(!(hi - lo > tol) || mid == lo || mid == hi) break;   // converged, or at rounding width
		if(c.cdf(mid) >= p) hi = mid; else lo = mid;
	}
	return 0.5 * lo + 0.5 * hi;
}

}  // namespace

// Scores of a mixture predictive at m points (rows) with K components
// (columns): see the file header.  df may be NULL (all Gaussian) or an m x K
// matrix with Inf for Gaussian components.  Returns crps and crps_abs_error
// (the latter 0 for closed forms), logscore (pointwise -log density), cdf_y,
// and quantiles (m x length(probs)); unrequested outputs are empty.
// [[Rcpp::export]]
Rcpp::List score_mixture_cpp(Rcpp::NumericVector y, Rcpp::NumericMatrix mean, Rcpp::NumericMatrix var,
	Rcpp::Nullable<Rcpp::NumericMatrix> df, Rcpp::NumericVector w, Rcpp::NumericVector probs,
	bool want_crps, bool want_logscore, bool want_cdf, double rel_tol, int subdivisions, int ncores){

	const int m = mean.nrow(), K = mean.ncol();
	if(var.nrow()!=m || var.ncol()!=K) Rcpp::stop("mean and var must have the same dimensions.\n");
	if(w.size()!=K) Rcpp::stop("w must have one weight per component.\n");
	if(y.size()!=m) Rcpp::stop("y must have one value per prediction point.\n");
	Rcpp::NumericMatrix dfm(0, 0);
	const bool has_df = df.isNotNull();
	if(has_df){ dfm = Rcpp::NumericMatrix(df.get()); if(dfm.nrow()!=m || dfm.ncol()!=K) Rcpp::stop("df must have the dimensions of mean.\n"); }
	if(!(rel_tol > 0.0 && rel_tol < 1.0)) Rcpp::stop("rel_tol must lie in (0, 1).\n");
	if(subdivisions < 1 || subdivisions > std::numeric_limits<int>::max()/4)
		Rcpp::stop("subdivisions must be positive and at most INT_MAX/4.");
	restree_validate_ncores(ncores);
	// ncores is validated for API consistency; R-dependent scoring is serial.
	const int nq = probs.size();
	for(int j=0; j<nq; ++j){ if(!(probs[j] > 0.0 && probs[j] < 1.0)) Rcpp::stop("probs must lie in (0, 1).\n"); }

	// weights: drop zero-weight components, normalise (as .restree_pred_parts)
	std::vector<int> keep; double wsum = 0.0;
	for(int k=0; k<K; ++k){
		if(!std::isfinite(w[k]) || w[k] < 0.0) Rcpp::stop("weights must be finite and nonnegative.\n");
		if(w[k] > 0.0){ keep.push_back(k); wsum += w[k]; }
	}
	if(keep.empty()) Rcpp::stop("at least one component needs positive weight.\n");
    // Validate supported components before entering workers. Zero-weight
    // columns need not define a distribution; positive subnormal weights do.
    for(int k : keep){
        for(int i=0; i<m; ++i){
            if(!std::isfinite(mean(i,k)) || !std::isfinite(var(i,k)) || var(i,k)<=0.0)
                Rcpp::stop("supported component means must be finite and variances finite and positive.\n");
            if(has_df && !(dfm(i,k)>2.0))
                Rcpp::stop("supported Student-t df must exceed 2, or be Inf for Gaussian components.\n");
        }
    }
	// Preserve positive subnormal support in densities even if w / sum(w)
	// underflows. Probability-scale weights remain sufficient for moments/CDFs.
	std::vector<double> probability(keep.size()), log_probability(keep.size());
	double log_total;
	if(std::isfinite(wsum)){
		log_total = std::log(wsum);
		for(size_t a=0; a<keep.size(); ++a) probability[a] = w[keep[a]]/wsum;
	}else{
		double scale = 0.0, scaled_total = 0.0;
		for(int k : keep) scale = std::max(scale, w[k]);
		for(int k : keep) scaled_total += w[k]/scale;
		log_total = std::log(scale) + std::log(scaled_total);
		for(size_t a=0; a<keep.size(); ++a) probability[a] = (w[keep[a]]/scale)/scaled_total;
	}
	for(size_t a=0; a<keep.size(); ++a) log_probability[a] = std::log(w[keep[a]])-log_total;

	Rcpp::NumericVector crps(want_crps ? m : 0), crps_err(want_crps ? m : 0),
		logscore(want_logscore ? m : 0), cdf_y(want_cdf ? m : 0);
	Rcpp::NumericMatrix quant(nq ? m : 0, nq);
	std::vector<int> fail(m, 0);

	ParallelError worker_error;
	worker_error.reset();

	{
		std::vector<int> iwork;
		std::vector<double> work;
		PointMixture c;
		const int Kk = static_cast<int>(keep.size());
		std::vector<double> L;   // log-sum-exp terms of the log score
		worker_error.run("score-setup",[&]{
			// Allocated lazily only for actual mixture quadrature, not closed forms.
			c.mu.resize(Kk); c.sc.resize(Kk); c.var.resize(Kk); c.df.resize(Kk); c.w.resize(Kk);
			L.resize(Kk);
		});
		// Main thread only: Rmath/QUADPACK and interrupt handling are safe here.
		for(int i=0; i<m; ++i){
			if(i%128==0) Rcpp::checkUserInterrupt();
			worker_error.run("score-point",[&]{
			for(int a=0; a<Kk; ++a){
				const int k = keep[a];
				c.w[a] = probability[a];
				c.mu[a] = mean(i, k);
				c.df[a] = has_df ? dfm(i, k) : kInf;
                // Respect the supplied variance: no response-unit-dependent
                // epsilon floor. Take square roots before multiplication to
                // retain positive subnormal variances in Student-t scales.
                c.var[a] = var(i,k);
                c.sc[a] = std::sqrt(c.var[a]);
                if(std::isfinite(c.df[a]))
                    c.sc[a] *= std::sqrt((c.df[a]-2.0)/c.df[a]);
			}
			const double yi = y[i];
			if(want_crps){
				if(c.gaussian()){
					crps[i] = crps_gaussian_mixture(yi, c); crps_err[i] = 0.0;
				}else{
					double val, err; int ier;
					crps_t_mixture(yi, c, rel_tol, subdivisions, iwork, work, val, err, ier);
					crps[i] = val; crps_err[i] = err; fail[i] = ier;
				}
			}
			if(want_logscore){
				// -log sum_k w_k f_k(y) by log-sum-exp
				double mx = -kInf;
				for(int a=0; a<Kk; ++a){ L[a] = log_probability[a] + comp_logpdf(yi, c.mu[a], c.sc[a], c.df[a]); mx = std::max(mx, L[a]); }
				if(std::isinf(mx)){ logscore[i] = -mx; }
				else{
					double s = 0.0; for(int a=0; a<Kk; ++a){ s += std::exp(L[a] - mx); }
					logscore[i] = -(mx + std::log(s));
				}
			}
			if(want_cdf) cdf_y[i] = c.cdf(yi);
			for(int j=0; j<nq; ++j) quant(i, j) = mixture_quantile(probs[j], c, 200);
			});
		}
	}
	worker_error.rethrow();
	for(int i=0; i<m; ++i){
		if(want_crps && !std::isfinite(crps[i]))
			Rcpp::stop("CRPS is not finite at prediction point %d.",i+1);
		if(fail[i] != 0) Rcpp::stop("CRPS integration failed at prediction point %d: %s", i + 1, quadpack_message(fail[i]));
	}
	return Rcpp::List::create(Rcpp::_["crps"] = crps, Rcpp::_["crps_abs_error"] = crps_err,
		Rcpp::_["logscore"] = logscore, Rcpp::_["cdf_y"] = cdf_y, Rcpp::_["quantiles"] = quant);
}
