// Known-parameter likelihood and SMC evidence evaluation.
#include "SMC.h"
#include "wrapper.h"

/*==================== restree_loglik ====================*/

//' Log-likelihood of a residual-tree model at known covariance parameters
//'
//' Evaluates \eqn{\log p(y \mid \theta)} for the model specified by a
//' \code{\link{restree_model}} object at the covariance parameters
//' \code{theta}, and returns it as a single number:
//' \itemize{
//'   \item \code{leaf_model = "Full"}: one generated fixed tree (split
//'     while \eqn{n > r}, extending \code{depth} as needed, split dimension cycling over the
//'     coordinates, cuts from the model's \code{cut_method}) --
//'     \eqn{\log p(y \mid T, \theta)}. It is the exact dense GP
//'     log-likelihood when the root holds all rows, \code{n <= r}; use
//'     \code{r = nrow(x)} for that reference. Requesting \code{depth = 0}
//'     alone does not prevent Full from splitting.
//'   \item \code{leaf_model = "PP"} or \code{"WN"}: the tree is inferred by
//'     the sequential Monte Carlo sampler and averaged out under the model's
//'     Chipman--George--McCulloch prior, \eqn{\log Z(\theta) = \log \sum_T
//'     p(T)\, p(y \mid T, \theta)}; \eqn{\exp} of the estimate is unbiased
//'     for \eqn{Z(\theta)} at any particle count.
//'   \item any leaf model with \code{tree} supplied (a \code{list(S, J,
//'     cuts)} in the matrix convention of
//'     \code{\link{restree_trees}(fit, form = "matrix")}, e.g. one
//'     column of it, or the MAP tree of a fit): \eqn{\log p(y \mid T,
//'     \theta)} for that fixed tree.
//' }
//' \code{PP} and \code{WN} split adaptively: a node with fewer than
//' \code{r} observations may still split, spending all its rows as knots
//' (for \code{WN} the split is scored as the dense Gaussian knot block, the
//' convention shared by \code{\link{restree_fit}}).  A supplied \code{tree}
//' is rescored with \eqn{r_v = \min(r, n_v)} knots at every split, whatever
//' the leaf model. A supplied Full tree cannot stop with more than
//' \code{r} rows in a terminal; oversized terminals raise an error.
//'
//' Full returns the likelihood of its residual-tree covariance, not in
//' general the dense GP likelihood. PP/WN evidence targets their own
//' tree-prior mixtures, not a posterior-weighted average of tree likelihoods.
//' The log of an unbiased evidence estimate is not itself unbiased. Neither
//' tree averaging nor greater depth guarantees a closer dense-GP approximation.
//'
//' WN depth zero always specifies one WN leaf, even when \code{n > r}.
//' For positive depth, WN has no structural depth boundary: every nonempty node
//' compares stopping with splitting at every level, and a node with \code{n <= r}
//' spends all \code{n} rows as knots (an empty residual leaf) and still competes
//' its split with the stop. The realized WN tree therefore grows past the initial
//' depth as the data need. PP keeps its depth cap; Full uses its separate
//' size-controlled fixed tree without SMC.
//' A WN split beyond the representational limit (realized depth 25) raises an
//' error, also in EB/PMMH; it is never silently stopped.
//'
//' Model-based likelihood computation is compiled; data validation occurs
//' when model and theta objects are built. Fitted root-WN MLE evaluations
//' additionally verify their fitted-state record before reading the fitted mean.
//' The compiled entry still checks its inputs. Repeated Full calls reuse the
//' stored geometry through an internal native cache. Exact comparisons of the
//' data and geometry payload invalidate it after edits; a serialized pointer
//' rebuilds lazily. Covariance factors and likelihoods are always recomputed.
//' Repeated PP/WN calls run a fresh SMC; EB/PMMH use fit-local persistent engines.
//'
//' @inheritSection restree_model Leaf models
//' @param model a \code{\link{restree_model}} (or a fitted \code{restree},
//'   whose model slots are used). For a fitted depth-zero WN MLE,
//'   evaluates the iid Gaussian likelihood using the fitted mean and
//'   \code{theta$sig2} as its variance. An unfitted WN model retains the
//'   conjugate leaf marginal used by the Bayesian/SMC interfaces.
//' @param theta a \code{\link{restree_theta}} object, or a named list with
//'   the same fields.
//' @param tree \code{NULL}, or a \code{list(S, J, cuts)} with one entry per
//'   internal-node position (\code{2^depth} entries; \code{S}: 0 = split,
//'   1 = terminal, \code{NA} = unreachable). For WN with positive model
//'   depth, the array depth may
//'   exceed the model's supplied depth. All three models also accept
//'   \code{list(nodes = tab)}, where \code{tab} is one tree's rows from
//'   \code{restree_trees(fit)$nodes} (id, split, J, cut). This sparse form
//'   avoids allocating depth-sized arrays, including for nested PP. Include
//'   every nonempty node, including terminal nodes at PP's depth cap. PP
//'   cannot split at or beyond its cap; positive-depth WN remains open up to
//'   its representation limit. Matrix input remains supported.
//' @param nparticles SMC particles (PP / WN with \code{tree = NULL});
//'   \code{NULL} (default) uses \eqn{100\,\mathrm{depth}} for both PP and
//'   WN, using the configured model depth, not the realized WN tree depth.
//'   This is the rule of \code{\link{restree_fit}}. Root-only models use one
//'   particle. Explicit \code{nparticles} values override the default for
//'   positive depth. Full and supplied-tree likelihoods do not use SMC.
//' @param resampling \code{"stratified"} or \code{"multinomial"}.
//' @param temper_alpha tempering exponent of the SMC weights; \code{1} is
//'   the plain evidence estimator.
//' @param seed random seed of the SMC (\code{PP} / \code{WN}).  A \code{Full}
//'   model's tree is part of the model (built by \code{restree_model()} from
//'   its own \code{seed}), so this argument does not affect it.
//' @param ncores positive thread request. Uses the allocation-bounded engine/
//'   Eigen policy of \code{\link{restree_fit}}; without OpenMP execution is serial.
//' @param verify if \code{TRUE}, every distinct sampled tree is rescored by
//'   a separate fixed-structure traversal (sharing numerical kernels) and the call errors on any
//'   disagreement above 1e-6. This development validation can be substantially
//'   more expensive than an ordinary sampler run and is off by default.
//'   PP and WN verification use reached-node records, without depth-sized
//'   matrices. Nested PP carries candidate rows along the recursion.
//'   Applies only to sampled PP/WN trees: Full, depth-zero and supplied-tree
//'   calls do not run this check. It does not establish exact integrated
//'   evidence or accuracy relative to the original dense GP.
//' @param diagnostics if \code{TRUE}, the returned number carries the attributes
//'   described under Value. The default \code{FALSE} returns the bare number and
//'   skips optional distinct-tree grouping, weighted knot counts and level-table
//'   export. With both flags false, it also skips realized-tree score reconstruction.
//' @return A single number: the log-likelihood (fixed tree) or log evidence
//'   (SMC); with \code{diagnostics = TRUE} it carries the attributes \code{n_knot} (internal knots spent; the
//'   particle-weighted mean for SMC) and, for SMC, \code{ess} (final
//'   effective sample size), \code{n_distinct_trees}, \code{n_resample}
//'   \code{root_split_weight} (estimated posterior root-split mass),
//'   \code{root_split_fraction} (unweighted particle fraction), and
//'   \code{level_diag}, a data frame with one row per tree level (the
//'   level-end \code{ess} before and after resampling, \code{resampled},
//'   \code{max_weight}, the level's \code{logZ_increment} and the number of
//'   distinct node-paths \code{n_paths}); \code{\link{restree_diagnostics}} collects
//'   it over replicate runs.
//' @examples
//' set.seed(1)
//' n <- 300; X <- matrix(runif(2 * n), n, 2)
//' y <- sin(4 * pi * X[, 1]) + rnorm(n, sd = 0.3)
//' th <- restree_theta(sig2 = 1, range = 0.2, nugget = 0.1)
//'
//' ## exact GP log-likelihood: all training rows fit in the Full root
//' restree_loglik(restree_model(X, y, depth = 0, r = nrow(X), leaf_model = "Full"), th)
//' ## size-constrained Full approximation, initially allocating depth 3
//' restree_loglik(restree_model(X, y, depth = 3, r = 15, leaf_model = "Full"), th)
//' ## PP leaves: tree inferred and averaged out by SMC
//' m <- restree_model(X, y, depth = 3, r = 15, leaf_model = "PP")
//' restree_loglik(m, th, nparticles = 100, seed = 7)
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector restree_loglik(SEXP model, SEXP theta,
	Rcpp::Nullable<Rcpp::List> tree = R_NilValue,
	Rcpp::Nullable<Rcpp::IntegerVector> nparticles = R_NilValue,
	std::string resampling = "stratified",
	double temper_alpha = 1.0, int seed = 42, int ncores = 1,
	bool verify = false, bool diagnostics = false){

	// Benchmark entry point: shape checks only, no pass over the data
	// (read_model); the inputs are trusted as restree_model() built them.
	const ModelSpec sp = read_model(model);
	const GPM th = read_theta(theta, (int)sp.X.cols());
	// Unlike restree_fit, this exported R function calls native code directly.
	// These scalar controls have no R validation. Thread policy is resolved by
	// configure_threads; the early guard also covers fitted root-WN evaluation.
	int ncores_i = ncores;
	if(ncores_i<1) Rcpp::stop("ncores must be a positive integer.");
	// default particle count: the rule of .restree_model_nparticles in
	// R/validation.R (restree_default_particles in wrapper.cpp).
	int n_particles;
	if(nparticles.isNull()){
		n_particles = restree_default_particles(sp);
	}else{
		Rcpp::IntegerVector np(nparticles);
		if(np.size()!=1 || Rcpp::IntegerVector::is_na(np[0]))
			Rcpp::stop("nparticles must be one integer.\n");
		n_particles = np[0];
	}
	if(n_particles<1) Rcpp::stop("nparticles must be at least 1.\n");
	if(sp.depth==0 && sp.leaf_model=="WhiteNoise") n_particles=1;
	if(resampling!="stratified" && resampling!="multinomial")
		Rcpp::stop("resampling must be stratified or multinomial.\n");
	if(!std::isfinite(temper_alpha) || temper_alpha<=0.0 || temper_alpha>1.0)
		Rcpp::stop("temper_alpha must lie in (0, 1].\n");
	if(seed<0) Rcpp::stop("seed must be nonnegative.\n");
	reset_covariance_error();
	// The fixed-structure evaluator is built only on the routes that use it
	// (tree =, Full, depth 0); the SMC route builds its own engine below, and
	// verify = TRUE rescores on that engine (it is-a ResTree with the same
	// data and configuration), so no route copies X and y more than once.
	const int n_internal = 1 << sp.depth;

	auto finish = [&](double value, double n_knot){
		stop_on_covariance_error("restree_loglik");
		if(!std::isfinite(value))
			Rcpp::stop("restree_loglik produced a non-finite value (numerically singular covariance? increase theta$nugget).\n");
		Rcpp::NumericVector out = Rcpp::NumericVector::create(value);
		if(diagnostics) out.attr("n_knot") = n_knot;
		return out;
	};

  // A fitted root-WN MLE represents an iid Gaussian, not the conjugate
  // tree-leaf marginal. Its estimated mean travels in the fit results;
  // theta supplies the variance so evaluation at another variance is valid.
  Rcpp::S4 fitted_model(model);
  if(sp.depth==0 && sp.leaf_model=="WhiteNoise" && fitted_model.is("restree")){
    Rcpp::List results=fitted_model.slot("results");
    if(Rcpp::as<std::string>(fitted_model.slot("method"))=="ebayes"){
      // Model-based benchmark evaluations remain free of R-level fit checks.
      Rcpp::Environment ns=Rcpp::Environment::namespace_env("ResTree");
      Rcpp::Function check=ns[".restree_verify_fitted_state"];
      check(model);
    }
    if(results.containsElementNamed("leaf_mle")){
      Rcpp::List pars=results["leaf_mle"];
      if(pars.containsElementNamed("family") && Rcpp::as<std::string>(pars["family"])=="gaussian_wn"){
        const double mu=Rcpp::as<double>(pars["mean"]);
        if(!std::isfinite(mu)) Rcpp::stop("fitted WN mean must be finite.");
        const double quad=(sp.y.array()-mu).square().sum();
        return finish(-0.5*(sp.y.size()*std::log(2.0*M_PI*th.sig2)+quad/th.sig2),0.0);
      }
    }
  }

	// ---- a supplied fixed structure ----
	if(tree.isNotNull()){
		Rcpp::List tr(tree.get());
		if(tr.containsElementNamed("nodes")){
			Rcpp::DataFrame nd(tr["nodes"]);
			for(const char* nm:{"id","split","J","cut"})
				if(!nd.containsElementNamed(nm)) Rcpp::stop("tree$nodes needs id, split, J, and cut.");
			Rcpp::NumericVector ids=Rcpp::as<Rcpp::NumericVector>(nd["id"]),
				split=Rcpp::as<Rcpp::NumericVector>(nd["split"]),
				axis=Rcpp::as<Rcpp::NumericVector>(nd["J"]),
				cut=Rcpp::as<Rcpp::NumericVector>(nd["cut"]);
			if(split.size()!=ids.size() || axis.size()!=ids.size() || cut.size()!=ids.size())
				Rcpp::stop("tree$nodes columns must have equal lengths.");
			if(nd.containsElementNamed("tree")){
				Rcpp::NumericVector tree_id=Rcpp::as<Rcpp::NumericVector>(nd["tree"]);
				if(tree_id.size()!=ids.size()) Rcpp::stop("tree$nodes columns must have equal lengths.");
				for(R_xlen_t i=0;i<tree_id.size();++i)
					if(!std::isfinite(tree_id[i]) || tree_id[i]!=tree_id[0])
						Rcpp::stop("tree$nodes must describe exactly one tree.");
			}
			ResTree::FixedNodes nodes;
			for(R_xlen_t i=0;i<ids.size();++i){
				if(!std::isfinite(ids[i]) || ids[i]!=std::floor(ids[i]) || ids[i]<1 ||
					ids[i]>=(2<<restree_max_wn_depth) || (split[i]!=0 && split[i]!=1))
					Rcpp::stop("tree$nodes contains an invalid id or split indicator.");
				const bool s=split[i]==1;
				if(s && (!std::isfinite(axis[i]) || axis[i]!=std::floor(axis[i]) ||
					axis[i]<1 || axis[i]>sp.X.cols() || !std::isfinite(cut[i])))
					Rcpp::stop("every sparse split needs a valid dimension and finite cut.");
				if(!nodes.emplace((int)ids[i],ResTree::FixedNode{s,s?(int)axis[i]-1:0,s?cut[i]:0}).second)
					Rcpp::stop("tree$nodes contains duplicate node ids.");
			}
			const ResTree m=make_model(sp,th,ncores_i);
			StructureEval ev=m.evaluate_nodes(nodes,sp.leaf_model);
			if(!ev.stats.ok) Rcpp::stop("sparse fixed-tree evaluation failed: %s",ev.stats.error.c_str());
			return finish(ev.loglik,ev.stats.n_knot);
		}
		if(!tr.containsElementNamed("S") || !tr.containsElementNamed("J") ||
		   !tr.containsElementNamed("cuts"))
			Rcpp::stop("tree must be a list with S, J, and cuts.\n");
		Eigen::VectorXd S = Rcpp::as<Eigen::VectorXd>(tr["S"]);
		Eigen::VectorXd J = Rcpp::as<Eigen::VectorXd>(tr["J"]);
		Eigen::VectorXd cuts = Rcpp::as<Eigen::VectorXd>(tr["cuts"]);
		const bool valid_size = sp.leaf_model=="WhiteNoise" && sp.depth>0 ?
			(S.size()>=1 && S.size()<=(1<<restree_max_wn_depth) && !(S.size() & (S.size()-1))) : S.size()==n_internal;
		if(!valid_size || J.size()!=S.size() || cuts.size()!=S.size())
			Rcpp::stop("tree$S, tree$J, and tree$cuts must each have length 2^depth = %d.\n", n_internal);
		for(int i=0; i<S.size(); ++i){
			if(std::isnan(S(i)) || S(i)>=0.5) continue;
			if(!std::isfinite(J(i)) || J(i)!=std::floor(J(i)) || J(i)<1 || J(i)>sp.X.cols() ||
			   !std::isfinite(cuts(i)))
				Rcpp::stop("every split needs a valid dimension (1..ncol(x)) and a finite cut.\n");
		}
		const ResTree m = make_model(sp, th, ncores_i);
		StructureEval ev = m.evaluate_structure(S, J, cuts, sp.leaf_model);
		if(!ev.stats.ok) Rcpp::stop("fixed-structure evaluation failed: %s", ev.stats.error.c_str());
		return finish(ev.loglik, ev.stats.n_knot);
	}

	// ---- Full leaves: the fixed tree stored in the model ----
	if(sp.leaf_model=="Full"){
		StructureEval ev = cached_full_loglik(sp,th,ncores_i);
		if(!ev.stats.ok) Rcpp::stop("fixed-tree evaluation failed: %s", ev.stats.error.c_str());
		return finish(ev.loglik, ev.stats.n_knot);
	}

	// ---- PP / WN: SMC evidence ----
	if(sp.depth==0){
		// the tree space is the single root-stop structure: exact terminal marginal
		Eigen::VectorXd S = Eigen::VectorXd::Constant(1, 1.0);
		Eigen::VectorXd J = Eigen::VectorXd::Constant(1, NA_REAL), cuts = J;
		const ResTree m = make_model(sp, th, ncores_i);
		StructureEval ev = m.evaluate_structure(S, J, cuts, sp.leaf_model);
		if(!ev.stats.ok) Rcpp::stop("depth-0 evaluation failed: %s", ev.stats.error.c_str());
		Rcpp::NumericVector out = finish(ev.loglik, 0.0);
		if(diagnostics){ out.attr("ess") = (double)n_particles; out.attr("n_distinct_trees") = 1; }
		return out;
	}
	// One fresh engine per call, configured exactly as smc_sample_cpp and the
	// persistent evidence engine (a small-node split spends every row as a
	// knot and is scored as the dense knot block for every leaf model, as
	// the fixed-structure evaluator scores it, so verify = TRUE is a real
	// check).
	SMC eng(sp.y, sp.X);
	configure_smc(eng, sp, n_particles, resampling, temper_alpha, seed, ncores_i);
	eng.covpar = th;
	eng.run(false,false,verify); // final/level ESS diagnostics do not need per-node history
	stop_on_covariance_error("restree_loglik");
	if(!std::isfinite(eng.logZ))
		Rcpp::stop("restree_loglik: the evidence is not finite -- the residual covariance is numerically singular (duplicated inputs with a zero nugget?); increase theta$nugget.\n");

	// A value-only call (the default) returns here: the distinct-tree grouping,
	// the weighted knot count and the level table are diagnostics, and the
	// grouping walks the reached-node records, not N * 2^depth positions.
	if(!verify && !diagnostics) return finish(eng.logZ, 0.0);
	std::vector<int> reps; Eigen::VectorXd w_distinct; Eigen::VectorXi group;
	eng.distinct_trees(reps, w_distinct, group);
	const double wsum = w_distinct.sum();
	double n_knot = 0.0;
	if(diagnostics)
		for(size_t k=0; k<reps.size(); ++k)
			n_knot += (w_distinct(k)/wsum) * particle_internal_knots(eng, reps[k]);

	if(verify){
		// Rescore every DISTINCT realized tree with the independent
		// fixed-structure evaluator (a direct top-down pass, no trie
		// bookkeeping) and refuse to return on any disagreement.
		// Both PP and WN use one representative's reached-node records at a
		// time. PP bottom leaves remain explicit; nested candidate inheritance
		// is local to the independent recursive evaluator, not a dense table.
		double max_err = 0.0;
		// (eng is-a ResTree: same data, depth, r, design, theta and ncores
		// as make_model() would set, so the rescoring is the same arithmetic.)
		for(size_t k=0; k<reps.size(); ++k){
			const int p = reps[k];
			SMC::NodeTable table;
			eng.collect_nodes(std::vector<int>{p},table);
			ResTree::FixedNodes nodes;
			for(size_t j=0;j<table.id.size();++j)
				nodes.emplace(table.id[j],ResTree::FixedNode{
					table.split[j]!=0,table.J[j],table.cut[j]});
			StructureEval ev=eng.evaluate_nodes(nodes,sp.leaf_model);
			if(!ev.stats.ok)
				Rcpp::stop("verification rescoring failed: %s", ev.stats.error.c_str());
			max_err = std::max(max_err, std::abs(ev.loglik - eng.particle_loglik_tree(p)));
		}
		if(max_err > 1e-6)
			Rcpp::stop("realized-tree score disagrees with the fixed-structure rescoring (max error %.6g)", max_err);
	}
	Rcpp::NumericVector out = finish(eng.logZ, n_knot);
	if(!diagnostics) return out;     // verify = TRUE without diagnostics: bare number
	Eigen::VectorXd w = (eng.particle_log_weight.array() - log_sum_exp(eng.particle_log_weight)).exp();
	out.attr("ess") = 1.0 / w.squaredNorm();
	out.attr("n_distinct_trees") = (int)reps.size();
	out.attr("n_resample") = eng.n_resample;
	double root_mass=0.0, root_count=0.0;
	for(int p=0;p<n_particles;++p) if(eng.act_value(p,1)==ACT_SPLIT){
		root_mass+=w(p); root_count+=1.0;
	}
	out.attr("root_split_weight")=root_mass;
	out.attr("root_split_fraction")=root_count/n_particles;
	out.attr("level_diag") = level_diag_frame(eng);
	return out;
}
