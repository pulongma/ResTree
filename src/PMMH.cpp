// Particle-marginal Metropolis-Hastings and retained-state prediction.
#include "SMC.h"
#include "wrapper.h"

/*==================== the PMMH chain in the compiled layer ====================*/
/* Keep proposal, acceptance and Robbins-Monro expressions in their current
   order. Chain draws use R's RNG on the main thread, managed by the generated
   wrapper's RNGScope. Per-likelihood SMC seeds use a separate wrapped counter.
   The FP_CONTRACT guard below preserves rounding in the acceptance path. */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif


namespace {

// theta at the chain state u = (log sig2, log range_1..nr, log nugget): the
// restree_theta the R driver rebuilt at u, read as read_theta reads it (a
// scalar range under form ARD/tensor is expanded to one per input column) and
// validated the same way -- a rejected theta was an evidence of -Inf.
GPM theta_at_state(const GPM& base, const double* u, int nr, int d){
	GPM th = base;
	th.sig2 = std::exp(u[0]);
	th.range.resize(nr);
	for(int j=0; j<nr; ++j){ th.range(j) = std::exp(u[1+j]); }
	th.nugget = std::exp(u[1+nr]);
	if((th.form=="ARD" || th.form=="tensor") && nr==1 && d>1)
		th.range = Eigen::VectorXd::Constant(d, th.range(0));
	restree_validate_covariance(d, th.sig2, th.range, th.tail, th.nu, th.nugget,
		th.form, th.family, th.dtype);
	return th;
}

// log prior of u = (log sig2, log range, log nugget): .restree_log_prior_u()
// of R/validation.R term by term, in R's evaluation order -- inverse-gamma
// (shape a, rate b) on sig2, half-Cauchy on every range and on the nugget,
// plus the Jacobian sum(u) of the log parameterisation.  The constant
// a log b - lgamma(a) is evaluated by R (R's own lgamma) and passed in; sums
// accumulate in long double as R's sum() does.  Without a prior the value is
// 0 (the evidence alone; no Jacobian), exactly as in R.
struct LogPriorU {
	bool proper = false;
	double shape = 0.0, rate = 0.0, lp_sig2_const = 0.0;
	Eigen::VectorXd range_scale;
	double nugget_scale = 1.0;

	double operator()(const double* u, int nr) const {
		const double neg_inf = -std::numeric_limits<double>::infinity();
		const double sig2 = std::exp(u[0]), nugget = std::exp(u[1+nr]);
		bool ok = std::isfinite(sig2) && sig2>0.0 && std::isfinite(nugget) && nugget>0.0;
		for(int j=0; j<nr; ++j){ const double r = std::exp(u[1+j]); ok = ok && std::isfinite(r) && r>0.0; }
		if(!ok) return neg_inf;
		if(!proper) return 0.0;
		// R: a * log(b) - lgamma(a) - (a + 1) * log(sig2) - b / sig2, left to right
		const double t_shape = (shape + 1.0) * std::log(sig2);
		const double lp_sig2 = (lp_sig2_const - t_shape) - rate / sig2;
		const double c = std::log(2.0 / M_PI);
		long double s = 0.0L;
		for(int j=0; j<nr; ++j){
			const double sc = range_scale(j), q = std::exp(u[1+j]) / sc;
			const double q2 = q * q;
			s += (c - std::log(sc)) - std::log1p(q2);
		}
		const double lp_range = static_cast<double>(s);
		const double qn = nugget / nugget_scale, qn2 = qn * qn;
		const double lp_nugget = (c - std::log(nugget_scale)) - std::log1p(qn2);
		const double lp = (lp_sig2 + lp_range) + lp_nugget;
		long double su = 0.0L;
		for(int k=0; k<2+nr; ++k){ su += u[k]; }
		return lp + static_cast<double>(su);
	}
};

LogPriorU read_log_prior(Rcpp::List prior, int nr){
	LogPriorU lp;
	lp.proper = Rcpp::as<bool>(prior["proper"]);
	if(!lp.proper) return lp;
	lp.shape = Rcpp::as<double>(prior["shape"]);
	lp.rate = Rcpp::as<double>(prior["rate"]);
	lp.lp_sig2_const = Rcpp::as<double>(prior["lp_sig2_const"]);
	lp.range_scale = Rcpp::as<Eigen::VectorXd>(prior["range_scale"]);
	lp.nugget_scale = Rcpp::as<double>(prior["nugget_scale"]);
	if(lp.range_scale.size()!=nr) Rcpp::stop("prior$range_scale must have one entry per range.\n");
	return lp;
}

// log Z(theta(u)) at a seed, as .restree_log_evidence() computed it: the
// persistent engine's run (smc_engine_logZ_cpp without its attributes), or at
// depth 0 the exact root-stop terminal of restree_loglik(). Rejected theta,
// covariance errors and non-finite values yield -Inf. Interrupts, structural
// invariant failures and WN depth/allocation errors must instead propagate.
struct EvidenceAtState {
	SMC* eng = nullptr;   // depth >= 1
	const ModelSpec* sp = nullptr;     // depth 0
	int ncores = 1;
	const GPM* base = nullptr;
	int nr = 1, d = 1;

	double operator()(const double* u, unsigned int seed) const {
		const double neg_inf = -std::numeric_limits<double>::infinity();
		try{
			const GPM th = theta_at_state(*base, u, nr, d);
			if(eng){
				reapply_threads(*eng);
				reset_covariance_error();
				const bool reuse = eng->sig2_only_change(th, seed);
				eng->covpar = th;
				eng->seed = seed;
				eng->run(reuse,false,false); // PMMH needs the evidence, not ESS history
				stop_on_covariance_error("restree_loglik");
				return std::isfinite(eng->logZ) ? eng->logZ : neg_inf;
			}
			reset_covariance_error();
			Eigen::VectorXd S = Eigen::VectorXd::Constant(1, 1.0);
			Eigen::VectorXd J = Eigen::VectorXd::Constant(1, NA_REAL), cuts = J;
			const ResTree m = make_model(*sp, th, ncores);
			StructureEval ev = m.evaluate_structure(S, J, cuts, sp->leaf_model);
			if(!ev.stats.ok) return neg_inf;
			stop_on_covariance_error("restree_loglik");
			return std::isfinite(ev.loglik) ? ev.loglik : neg_inf;
		}catch(Rcpp::internal::InterruptedException&){
			throw;
		}catch(AdaptiveDepthError&){
			throw;
		}catch(ParticleStateError&){
			throw;
		}catch(WorkerFailure&){
			throw;
		}catch(std::bad_alloc&){
			throw;
		}catch(std::exception&){
			return neg_inf;
		}
	}
};

// (seed + k) mod 2^31, the derived-seed rule of .restree_seed_add().
unsigned int derived_seed(int seed, long long k){
	return static_cast<unsigned int>((static_cast<long long>(seed) + k) % 2147483648LL);
}

// The accepted extended PMMH state contains the entire final SMC ensemble.
// Group identical trees, sum their weights, and detach compact geometry from
// the mutable proposal workspace. Rejections share the immutable snapshot.
struct PmmhEnsemble {
	FittedTrees trees;
	Eigen::VectorXd weights;
};
struct PmmhDraw {
	std::shared_ptr<const PmmhEnsemble> ensemble;
	long long state = 0;
	unsigned int smc_seed = 0;
};

PmmhDraw retain_ensemble(const SMC* eng,unsigned int seed,long long state){
	PmmhDraw draw;
	draw.state=state; draw.smc_seed=seed;
	PmmhEnsemble ensemble;
	if(eng){
		std::vector<int> reps;
		Eigen::VectorXi groups;
		Eigen::VectorXd weights;
		eng->distinct_trees(reps,weights,groups);
		predictive_weights(weights); // validate the completed sampler output
		std::vector<int> keep;
		std::vector<double> positive;
		for(size_t k=0; k<reps.size(); ++k) if(weights(k)>0.0){
			keep.push_back(reps[k]); positive.push_back(weights(k));
		}
		ensemble.trees=eng->fitted_trees(keep);
		ensemble.trees.source_indices.clear(); // compact indices 0,...,K-1
		ensemble.weights=Eigen::Map<Eigen::VectorXd>(positive.data(),positive.size());
		ensemble.weights/=ensemble.weights.sum();
	}else{
		ensemble.trees.nodes.emplace_back(std::vector<PredictionNode>{
			{1,ACT_STOP,-1,std::numeric_limits<double>::quiet_NaN()}});
		ensemble.weights=Eigen::VectorXd::Ones(1);
	}
	draw.ensemble=std::make_shared<const PmmhEnsemble>(std::move(ensemble));
	return draw;
}

// Per-state SMC mixtures, followed by an equal-weight MCMC mixture. Preserve
// component distributions (including WN degrees of freedom) for marginal
// scores/quantiles; moment-matching each state to one component is incorrect.
struct PmmhComponents {
	Eigen::MatrixXd mean,variance,df;
	Eigen::VectorXd lpd,weights;
};
struct PmmhPrediction {
	Eigen::MatrixXd mean,variance;
	Eigen::VectorXd lpd;
	std::vector<std::shared_ptr<const PmmhComponents>> components;
	int used=0;
	bool wn=false,scored=false;
	PmmhPrediction(int np,int capacity,bool wn_,bool scored_)
		: mean(np,capacity),variance(np,capacity),lpd(capacity),
		  wn(wn_),scored(scored_) { components.reserve(capacity); }
	void append(ResTree& model,const GPM& theta,const PmmhDraw& draw,
		const Eigen::MatrixXd& xnew,const Eigen::VectorXd* ynew,bool repeat){
		if(repeat && used>0){
			components.push_back(components.back());
			mean.col(used)=mean.col(used-1); variance.col(used)=variance.col(used-1);
			lpd(used)=lpd(used-1); ++used; return;
		}
		// A rejected proposal may have overwritten the SMC covariance.
		struct RestoreTheta {
			ResTree& model; GPM previous;
			~RestoreTheta(){ model.covpar=std::move(previous); }
		} restore{model,model.covpar};
		model.covpar=theta; reapply_threads(model); reset_covariance_error();
		PmmhComponents block;
		block.weights=draw.ensemble->weights;
		std::vector<int> reps(block.weights.size());
		std::iota(reps.begin(),reps.end(),0);
		Predictor(model,draw.ensemble->trees).predict(xnew,reps,theta.sig2,
			block.mean,block.variance,ynew,scored ? &block.lpd : nullptr,
			wn ? &block.df : nullptr);
		stop_on_covariance_error("PMMH prediction");
		if(!block.mean.allFinite() || !block.variance.allFinite() ||
		   (block.variance.array()<0.0).any())
			throw std::runtime_error("non-finite PMMH predictive moments");
		Eigen::VectorXd mu,va;
		predictive_moments(block.mean,block.variance,block.weights,mu,va);
		const double joint=scored ?
			predictive_log_mixture(block.lpd,predictive_weights(block.weights)) : NA_REAL;
		components.push_back(std::make_shared<const PmmhComponents>(std::move(block)));
		mean.col(used)=mu; variance.col(used)=va; lpd(used)=joint; ++used;
	}
	Rcpp::List output(const std::vector<int>& index,const Eigen::MatrixXd& theta,
		const Eigen::MatrixXd& xnew,const Eigen::VectorXd* ynew,double seconds){
		if(used==0) return Rcpp::List::create();
		mean.conservativeResize(Eigen::NoChange,used);
		variance.conservativeResize(Eigen::NoChange,used); lpd.conservativeResize(used);
		const Eigen::VectorXd iteration_weights=Eigen::VectorXd::Constant(used,1.0/used);
		Eigen::VectorXd mu,va;
		predictive_moments(mean,variance,iteration_weights,mu,va);
		// Nested log mixtures preserve tiny positive SMC weights before the
		// division by the retained-iteration count could round them to zero.
		const double joint=scored ?
			predictive_log_mixture(lpd,predictive_weights(iteration_weights)) : NA_REAL;
		size_t total=0;
		double weight_scale=1.0/used;
		for(const auto& block:components){
			total+=block->weights.size();
			if((block->weights.array()*weight_scale==0.0).any()) weight_scale=1.0;
		}
		if(total>static_cast<size_t>(std::numeric_limits<int>::max()))
			Rcpp::stop("too many PMMH predictive components for an R matrix");
		// Allocate the pooled R matrices once. Repeats share native blocks
		// while fitting, but have their own iteration-labelled output columns.
		const int np=xnew.rows(), K=static_cast<int>(total);
		Rcpp::NumericMatrix pm(np,K),pv(np,K),df(wn ? np : 0,wn ? K : 0);
		Rcpp::NumericVector weights(K),joint_components(scored ? K : 0);
		Rcpp::IntegerVector iteration(K);
		int col=0;
		for(int s=0; s<used; ++s){
			const auto& block=*components[s];
			for(int k=0; k<block.weights.size(); ++k,++col){
				std::copy(block.mean.col(k).data(),block.mean.col(k).data()+np,pm.begin()+col*(R_xlen_t)np);
				std::copy(block.variance.col(k).data(),block.variance.col(k).data()+np,pv.begin()+col*(R_xlen_t)np);
				if(wn) std::copy(block.df.col(k).data(),block.df.col(k).data()+np,df.begin()+col*(R_xlen_t)np);
				weights[col]=block.weights(k)*weight_scale; iteration[col]=index[s];
				if(scored) joint_components[col]=block.lpd(k);
			}
		}
		Rcpp::List history=Rcpp::List::create(
			Rcpp::_["index"]=index,Rcpp::_["theta"]=theta,
			Rcpp::_["mean"]=mean,Rcpp::_["var"]=variance,
			Rcpp::_["joint_lpd"]=lpd,Rcpp::_["component_iteration"]=iteration,
			Rcpp::_["xnew"]=xnew);
		history["y"]=ynew ? Rcpp::wrap(*ynew) : R_NilValue;
		Rcpp::List out=Rcpp::List::create(
			Rcpp::_["mean"]=mu,Rcpp::_["var"]=va,
			Rcpp::_["par_mean"]=pm,Rcpp::_["par_var"]=pv,
			Rcpp::_["w"]=weights,Rcpp::_["lpd"]=joint,
			Rcpp::_["mcmc_prediction"]=history,Rcpp::_["pred_time"]=seconds);
		out["par_df"]=wn ? static_cast<SEXP>(df) : R_NilValue;
		out["lpd_particle"]=scored ? static_cast<SEXP>(joint_components) : R_NilValue;
		return out;
	}
};

// Version 2 stores ensembles, not sampled particles. The iteration mapping is
// one-based; ensemble/tree/node offsets are zero-based. No per-state engines.
Rcpp::List export_pmmh_trees(const std::vector<int>& index,
	const std::vector<PmmhDraw>& draws){
	std::map<const PmmhEnsemble*,int> seen;
	std::vector<int> ensemble,ensemble_offset{0},offset{0},id,act,J;
	std::vector<double> cut,state,seed,weight;
	for(const auto& draw:draws){
		auto found=seen.find(draw.ensemble.get());
		if(found==seen.end()){
			const int slot=static_cast<int>(seen.size());
			seen.emplace(draw.ensemble.get(),slot);
			for(size_t k=0; k<draw.ensemble->trees.nodes.size(); ++k){
				for(const auto& node:draw.ensemble->trees.nodes[k]){
					id.push_back(node.id); act.push_back(node.act);
					J.push_back(node.jdim); cut.push_back(node.cutval);
				}
				offset.push_back(static_cast<int>(id.size()));
				weight.push_back(draw.ensemble->weights(k));
			}
			ensemble_offset.push_back(static_cast<int>(weight.size()));
			ensemble.push_back(slot+1);
		}else ensemble.push_back(found->second+1);
		state.push_back(static_cast<double>(draw.state)); seed.push_back(draw.smc_seed);
	}
	return Rcpp::List::create(Rcpp::_["version"]=2,Rcpp::_["index"]=index,
		Rcpp::_["ensemble"]=ensemble,Rcpp::_["ensemble_offset"]=ensemble_offset,
		Rcpp::_["weight"]=weight,Rcpp::_["offset"]=offset,Rcpp::_["id"]=id,
		Rcpp::_["act"]=act,Rcpp::_["J"]=J,Rcpp::_["cut"]=cut,
		Rcpp::_["state"]=state,Rcpp::_["seed"]=seed);
}

// Coordinatewise summary on the NATURAL parameter scale. Repeated states
// remain in the sample. The effective burn-in matches shortened-chain R policy.
Rcpp::NumericVector posterior_point(const Rcpp::NumericMatrix& chain,
    int completed, int burnin, const std::string& method){
  if(completed<1) return Rcpp::NumericVector();
  const int start=std::min(burnin,completed-1), n=completed-start;
  Rcpp::NumericVector point(chain.ncol());
  std::vector<double> values(n);
  for(int j=0; j<chain.ncol(); ++j){
    for(int i=0; i<n; ++i) values[i]=std::exp(chain(start+i,j));
    if(method=="mean"){
      long double total=0;
      for(double value:values) total+=value;
      point[j]=static_cast<double>(total/n);
    }else{
      const int mid=n/2;
      std::nth_element(values.begin(),values.begin()+mid,values.end());
      const double upper=values[mid];
      point[j]=(n%2) ? upper : static_cast<double>(
        (static_cast<long double>(*std::max_element(values.begin(),values.begin()+mid))+upper)/2);
    }
  }
  return point;
}

// A text progress bar over the iterations (utils::txtProgressBar style 3).
void print_progress(int it, int n_iter){
	const int width = 50, filled = (n_iter>0) ? (width * it) / n_iter : width;
	std::string bar(static_cast<size_t>(filled), '='); bar.append(static_cast<size_t>(width - filled), ' ');
	Rprintf("\r  |%s| %3d%%", bar.c_str(), (n_iter>0) ? (100 * it) / n_iter : 100);
	if(it>=n_iter) Rprintf("\n");
	R_FlushConsole();
}

}  // namespace

// The PMMH chain of .restree_pmmh(): n_iter iterations of sig2_moves
// random-walk moves on log sigma^2 (blocked = TRUE: the engine re-evaluates
// them from its retained block statistics) followed by one joint random-walk
// move of all of u, with the evidence at a fresh derived seed per evaluation
// (seed + 1, seed + 2, ... mod 2^31), the trie saved before the joint proposal
// and restored after a rejection when blocked, and the proposal scale adapted
// by Robbins-Monro during the burn-in when adapt = "burnin".  The chain's own
// random numbers are R's (norm_rand, unif_rand): call set.seed() first.
// Returns the per-iteration record (state, its evidence, proposal, its
// evidence, accept indicator, scale), the counters and, after an interrupt,
// the number of completed iterations (the arrays beyond it are NA).
// [[Rcpp::export]]
Rcpp::List smc_engine_pmmh_cpp(SEXP handle, SEXP model, SEXP theta0,
	Rcpp::NumericVector u0, int n_iter, int burnin, Rcpp::NumericVector prop_sd,
	int sig2_moves, bool blocked, Rcpp::List prior, std::string adapt,
	double target_accept, int seed, int ncores, bool verbose,
	Rcpp::Nullable<Rcpp::NumericMatrix> input_new = R_NilValue,
	Rcpp::Nullable<Rcpp::NumericVector> output_new = R_NilValue,
	std::string point_estimate = "median"){

	const ModelSpec sp = read_model(model);
	const int d = static_cast<int>(sp.X.cols());
	const GPM base = read_theta(theta0, d);
	const int nq = static_cast<int>(u0.size()), nr = nq - 2;
	if(nr<1) Rcpp::stop("the chain state needs at least (log sig2, log range, log nugget).\n");
	if(prop_sd.size()!=nq) Rcpp::stop("prop_sd must have one entry per chain coordinate.\n");
	// Iteration counts, burn-in, move counts and seeds are checked by the R
	// PMMH driver. Proposed theta values are generated here and checked above.
	if(adapt!="burnin" && adapt!="none") Rcpp::stop("adapt must be \"burnin\" or \"none\".\n");
	restree_validate_ncores(ncores);

	EvidenceAtState Z;
	Z.base = &base; Z.nr = nr; Z.d = d; Z.ncores = ncores; Z.sp = &sp;
	if(!Rf_isNull(handle)){
		// the handle stays protected by the R caller for the whole call
		Z.eng = &smc_from_handle(handle);
	}else if(sp.depth!=0){
		Rcpp::stop("the PMMH chain needs a persistent SMC engine at depth >= 1.\n");
	}
	const LogPriorU lp = read_log_prior(prior, nr);
	const int count=n_iter-burnin;
	Eigen::MatrixXd Xnew(0,d);
	Eigen::VectorXd ynew;
	const bool online=input_new.isNotNull(), scored=output_new.isNotNull();
	if(online) Xnew=Rcpp::as<Eigen::MatrixXd>(input_new.get());
	if(scored) ynew=Rcpp::as<Eigen::VectorXd>(output_new.get());
	if(Xnew.cols()!=d || (scored && (!online || ynew.size()!=Xnew.rows())))
		Rcpp::stop("PMMH prediction inputs are not aligned.");
	PmmhPrediction prediction(Xnew.rows(),online ? count : 0,
		sp.leaf_model=="WhiteNoise",scored);
	std::unique_ptr<ResTree> root_model;
	if(online && !Z.eng) root_model.reset(new ResTree(make_model(sp,base,ncores)));
	ResTree* prediction_model=Z.eng ? static_cast<ResTree*>(Z.eng) : root_model.get();
	std::vector<int> retained_index;
	std::vector<PmmhDraw> retained_draws;
	retained_index.reserve(count); retained_draws.reserve(count);
	double prediction_seconds=0.0;

	std::vector<double> pc(u0.begin(), u0.end()), pp(nq);
	double zc = Z(pc.data(), static_cast<unsigned int>(seed));
	if(!std::isfinite(zc)) Rcpp::stop("PMMH could not evaluate the evidence at theta0");
	PmmhDraw current=retain_ensemble(Z.eng,static_cast<unsigned int>(seed),0);
	PmmhDraw completed=current;

	Rcpp::NumericMatrix chain(n_iter, nq), chain_prop(n_iter, nq);
	Rcpp::NumericVector zs(n_iter, NA_REAL), zs_prop(n_iter, NA_REAL), scale_trace(n_iter, NA_REAL);
	Rcpp::LogicalVector accepted(n_iter, false);
	std::fill(chain.begin(), chain.end(), NA_REAL);
	std::fill(chain_prop.begin(), chain_prop.end(), NA_REAL);

	int acc = 0, acc_sig2 = 0, n_sig2 = 0, it_done = 0;
	long long seed_counter = 0;
	double log_scale = 0.0;
	bool interrupted = false;
	std::string error;   // non-empty: the chain stopped on this error (partial chain returned)
	auto next_seed = [&](){ ++seed_counter; return derived_seed(seed, seed_counter); };
	auto step_sd = [&](int j){ return prop_sd[j] * std::exp(log_scale); };
	// the MH test of the R loop: no uniform is drawn for a non-finite evidence
	auto accept_move = [&](double zp, const double* prop){
		if(!std::isfinite(zp)) return false;
		return std::log(unif_rand()) < (zp + lp(prop, nr)) - (zc + lp(pc.data(), nr));
	};
	auto retain=[&](int iteration,const PmmhDraw& draw,const double* u){
		if(online){
			const auto start=std::chrono::steady_clock::now();
			const bool repeat=!retained_draws.empty() &&
				retained_draws.back().state==draw.state;
			prediction.append(*prediction_model,theta_at_state(base,u,nr,d),draw,
				Xnew,scored ? &ynew : nullptr,repeat);
			prediction_seconds+=std::chrono::duration<double>(
				std::chrono::steady_clock::now()-start).count();
		}
		retained_index.push_back(iteration); retained_draws.push_back(draw);
	};

	try{
		for(int it=1; it<=n_iter; ++it){
			Rcpp::checkUserInterrupt();
			for(int k=0; k<sig2_moves; ++k){
				pp = pc;
				const double step0 = norm_rand() * step_sd(0);   // R: pc[1] + rnorm(1) * sd[1]
				pp[0] = pc[0] + step0;
				const double zp = Z(pp.data(), next_seed());
				++n_sig2;
				if(accept_move(zp, pp.data())){
					pc = pp; zc = zp; ++acc_sig2;
					current=retain_ensemble(Z.eng,derived_seed(seed,seed_counter),seed_counter);
				}
			}
			// the joint move is a full evaluation whichever coordinates it
			// touches, so it perturbs all of u
			for(int j=0; j<nq; ++j){                          // R: pc + rnorm(nq) * sd
				const double step = norm_rand() * step_sd(j);
				pp[j] = pc[j] + step;
			}
			if(blocked && Z.eng) Z.eng->save_trie();
			const double zp = Z(pp.data(), next_seed());
			for(int j=0; j<nq; ++j){ chain_prop(it-1, j) = pp[j]; }
			zs_prop[it-1] = zp;
			if(accept_move(zp, pp.data())){
				pc = pp; zc = zp; ++acc; accepted[it-1] = true;
				current=retain_ensemble(Z.eng,derived_seed(seed,seed_counter),seed_counter);
			}else if(blocked && Z.eng){
				Z.eng->restore_trie();
			}
			for(int j=0; j<nq; ++j){ chain(it-1, j) = pc[j]; }
			zs[it-1] = zc;
			scale_trace[it-1] = std::exp(log_scale);
			if(adapt=="burnin" && it<=burnin){
				const double gain = ((accepted[it-1] ? 1.0 : 0.0) - target_accept) / std::pow(static_cast<double>(it), 0.6);
				log_scale += gain;
				log_scale = std::min(std::max(log_scale, std::log(1e-3)), std::log(1e3));
			}
			if(it>burnin) retain(it,current,pc.data());
			it_done = it; completed=current;
			if(verbose) print_progress(it, n_iter);
		}
	}catch(Rcpp::internal::InterruptedException&){
		interrupted = true;
		if(verbose) Rprintf("\n");
	}catch(AdaptiveDepthError& e){
		// The evidence functor rethrows these (never -Inf).  Keep the
		// iterations completed so far: R re-signals the error with the
		// partial chain attached (.restree_pmmh), instead of discarding it.
		error = e.what();
		if(verbose) Rprintf("\n");
	}catch(std::bad_alloc& e){
		error = std::string("allocation failure during the PMMH evidence evaluation (") + e.what() + ")";
		if(verbose) Rprintf("\n");
	}catch(ParticleStateError& e){
		// As for depth failures, preserve completed iterations and let R
		// signal an explicit error rather than reject a corrupted proposal.
		error = e.what();
		if(verbose) Rprintf("\n");
	}catch(WorkerFailure& e){
		error = e.what();
		if(verbose) Rprintf("\n");
	}catch(std::exception& e){
		error = e.what();
		if(verbose) Rprintf("\n");
	}
	if(!error.empty() && blocked && Z.eng){
		// the trie moved aside by save_trie() for the failed joint move would
		// otherwise stay allocated on the handle until it is released
		Z.eng->restore_trie();
	}
	// The R driver reduces burn-in when an early interruption returns a
	// shorter chain. Its last COMPLETED iteration, not a partially evaluated
	// proposal, supplies the single retained state in that case.
	if(interrupted && it_done>=2 && retained_index.empty()){
		std::vector<double> last(nq);
		for(int j=0; j<nq; ++j) last[j]=chain(it_done-1,j);
		try{ retain(it_done,completed,last.data()); }
		catch(std::exception& e){ error=e.what(); }
	}
	Eigen::MatrixXd retained_theta(retained_index.size(),nq);
	for(size_t k=0; k<retained_index.size(); ++k)
		for(int j=0; j<nq; ++j) retained_theta(k,j)=std::exp(chain(retained_index[k]-1,j));
	Rcpp::List predictions=online ? prediction.output(retained_index,retained_theta,
		Xnew,scored ? &ynew : nullptr,prediction_seconds) : Rcpp::List::create();

	return Rcpp::List::create(
		Rcpp::_["theta_hat"] = posterior_point(chain,it_done,burnin,point_estimate),
		Rcpp::_["chain"] = chain, Rcpp::_["logZ_trace"] = zs,
		Rcpp::_["chain_prop"] = chain_prop, Rcpp::_["logZ_prop"] = zs_prop,
		Rcpp::_["accepted"] = accepted, Rcpp::_["scale_trace"] = scale_trace,
		Rcpp::_["acc"] = acc, Rcpp::_["acc_sig2"] = acc_sig2, Rcpp::_["n_sig2"] = n_sig2,
		Rcpp::_["seed_counter"] = static_cast<double>(seed_counter),
		Rcpp::_["log_scale"] = log_scale,
		Rcpp::_["it_done"] = it_done, Rcpp::_["interrupted"] = interrupted,
		Rcpp::_["error"] = error,
		Rcpp::_["trees"] = export_pmmh_trees(retained_index,retained_draws),
		Rcpp::_["prediction"] = predictions);
}
#if defined(__clang__)
#pragma STDC FP_CONTRACT ON
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

// Later prediction from the chain's retained JOINT parameter/tree states.
// No sampler, evidence evaluation, RNG, or per-draw native handle is needed.
// [[Rcpp::export]]
Rcpp::List pmmh_predict_cpp(SEXP model, SEXP theta,
	const Eigen::Map<Eigen::MatrixXd> chain, Rcpp::List history,
	Eigen::MatrixXd input_new,
	Rcpp::Nullable<Rcpp::NumericVector> output_new = R_NilValue, int ncores = 1){
	const ModelSpec sp=read_model(model);
	const GPM base=read_theta(theta,sp.X.cols());
	if(!history.containsElementNamed("version") || Rcpp::as<int>(history["version"])!=2)
		Rcpp::stop("PMMH final SMC ensembles are unavailable; refit with the current package.");
	Rcpp::IntegerVector index=history["index"], which=history["ensemble"], offset=history["offset"];
	Rcpp::IntegerVector ensemble_offset=history["ensemble_offset"];
	Rcpp::NumericVector weight=history["weight"];
	Rcpp::IntegerVector id=history["id"], act=history["act"], J=history["J"];
	Rcpp::NumericVector cut=history["cut"], state=history["state"];
	const int M=index.size(), nr=chain.cols()-2;
	if(M<1 || which.size()!=M || state.size()!=M || offset.size()<2 ||
	   ensemble_offset.size()<2 || ensemble_offset[0]!=0 ||
	   ensemble_offset[ensemble_offset.size()-1]!=weight.size() ||
	   offset.size()!=weight.size()+1 ||
	   offset[0]!=0 || offset[offset.size()-1]!=id.size() ||
	   act.size()!=id.size() || J.size()!=id.size() || cut.size()!=id.size() ||
	   nr!=base.range.size() || input_new.cols()!=sp.X.cols())
		Rcpp::stop("inconsistent retained PMMH prediction state; refit.");
	FittedTrees trees;
	for(int k=0; k<offset.size()-1; ++k){
		if(offset[k]<0 || offset[k+1]<=offset[k] || offset[k+1]>id.size())
			Rcpp::stop("invalid retained PMMH tree offsets; refit.");
		std::vector<PredictionNode> rows;
		rows.reserve(offset[k+1]-offset[k]);
		for(int j=offset[k]; j<offset[k+1]; ++j){
			if(act[j]==ACT_SPLIT && (J[j]<0 || J[j]>=sp.X.cols() ||
			   !std::isfinite(cut[j]) || cut[j]<0.0 || cut[j]>1.0))
				Rcpp::stop("invalid retained PMMH split; refit.");
			rows.push_back({id[j],static_cast<int8_t>(act[j]),static_cast<int8_t>(J[j]),cut[j]});
		}
		trees.nodes.emplace_back(std::move(rows));
	}
	std::vector<std::shared_ptr<const PmmhEnsemble>> ensembles;
	for(int e=0; e<ensemble_offset.size()-1; ++e){
		const int begin=ensemble_offset[e],end=ensemble_offset[e+1];
		if(begin<0 || end<=begin || end>weight.size())
			Rcpp::stop("invalid retained PMMH ensemble offsets; refit.");
		PmmhEnsemble saved;
		saved.weights.resize(end-begin);
		for(int k=begin; k<end; ++k){
			saved.trees.nodes.push_back(std::move(trees.nodes[k]));
			saved.weights(k-begin)=weight[k];
		}
		predictive_weights(saved.weights);
		ensembles.push_back(std::make_shared<const PmmhEnsemble>(std::move(saved)));
	}
	Eigen::VectorXd ynew;
	const bool scored=output_new.isNotNull();
	if(scored){
		ynew=Rcpp::as<Eigen::VectorXd>(output_new.get());
		if(ynew.size()!=input_new.rows()) Rcpp::stop("prediction responses are not aligned.");
	}
	ResTree engine=make_model(sp,base,ncores);
	PmmhPrediction prediction(input_new.rows(),M,sp.leaf_model=="WhiteNoise",scored);
	Eigen::MatrixXd selected(M,chain.cols());
	const auto start=std::chrono::steady_clock::now();
	for(int m=0; m<M; ++m){
		Rcpp::checkUserInterrupt();
		if(index[m]<1 || index[m]>chain.rows() || which[m]<1 || which[m]>(int)ensembles.size())
			Rcpp::stop("invalid retained PMMH iteration or ensemble index; refit.");
		selected.row(m)=chain.row(index[m]-1);
		GPM th=base;
		th.sig2=selected(m,0); th.range=selected.row(m).segment(1,nr).transpose();
		th.nugget=selected(m,nr+1);
		PmmhDraw draw; draw.ensemble=ensembles[which[m]-1];
		prediction.append(engine,th,draw,input_new,scored ? &ynew : nullptr,
			m>0 && which[m]==which[m-1] && selected.row(m)==selected.row(m-1));
	}
	return prediction.output(Rcpp::as<std::vector<int>>(index),selected,input_new,
		scored ? &ynew : nullptr,std::chrono::duration<double>(
			std::chrono::steady_clock::now()-start).count());
}
