// Fixed-tree geometry and Full-model likelihood workspaces.
#include "wrapper.h"

// ==================== R geometry conversion and model likelihood cache ====================

// This integer payload fixes knots, cuts and routing independently of theta.
// Native conversion accepts legacy dense payloads but never creates one.
Rcpp::List geometry_to_list(const FullTreeGeometry& geom){
 const int count=static_cast<int>(geom.nodes.size());
 std::vector<int> order(count);std::iota(order.begin(),order.end(),0);
 std::sort(order.begin(),order.end(),[&](int a,int b){return geom.nodes[a].id<geom.nodes[b].id;});
 Rcpp::IntegerVector ids(count),ns(count),no(count),split(count),ks(count),ko(count);
 Rcpp::NumericVector S(count,NA_REAL),J(count,NA_REAL),cuts(count,NA_REAL);
 std::vector<int> rows,knots;int nsplit=0,nleaf=0;
 for(int i=0;i<count;++i){
  const auto& nd=geom.nodes[order[i]];ids[i]=nd.id;
  no[i]=static_cast<int>(rows.size())+1;ns[i]=nd.idx.size();
  split[i]=nd.idx.empty()?NA_INTEGER:(nd.split?1:0);
  for(int row:nd.idx) rows.push_back(row+1);
  ko[i]=static_cast<int>(knots.size())+1;ks[i]=nd.split?nd.kn.size():0;
  if(nd.split){
   for(int k=0;k<nd.kn.size();++k) knots.push_back(nd.kn(k)+1);
   ++nsplit;S[i]=0;J[i]=nd.J+1;cuts[i]=nd.cut;
  }else if(!nd.idx.empty()){
   ++nleaf;if(nd.level<geom.depth) S[i]=1;
  }
 }
 return Rcpp::List::create(Rcpp::_["format"]=2,Rcpp::_["id"]=ids,
  Rcpp::_["S"]=S,Rcpp::_["J"]=J,Rcpp::_["cuts"]=cuts,
  Rcpp::_["node_offset"]=no,Rcpp::_["node_size"]=ns,Rcpp::_["node_split"]=split,
  Rcpp::_["rows"]=Rcpp::wrap(rows),Rcpp::_["knot_offset"]=ko,
  Rcpp::_["knot_size"]=ks,Rcpp::_["knots"]=Rcpp::wrap(knots),
  Rcpp::_["n_split"]=nsplit,Rcpp::_["n_leaf"]=nleaf,
  Rcpp::_["seed"]=(int)geom.seed,Rcpp::_["cut_method"]=geom.cut_method,
  Rcpp::_["design"]=geom.design,Rcpp::_["depth"]=geom.depth);
}

// Never let Rcpp's REAL-to-INT conversion silently truncate edited geometry.
// Generated payloads are integer; legacy numeric integer vectors remain valid.
Rcpp::IntegerVector full_integer_field(const Rcpp::List& tree, const char* name,
 bool allow_missing=false, bool scalar=false){
 SEXP value=tree[name];
 if(TYPEOF(value)!=INTSXP && TYPEOF(value)!=REALSXP)
  Rcpp::stop("Full geometry %s must contain integers.",name);
 if(scalar && Rf_xlength(value)!=1)
  Rcpp::stop("Full geometry %s must be a scalar.",name);
 if(TYPEOF(value)==INTSXP){
  Rcpp::IntegerVector out(value);
  if(!allow_missing) for(int x:out) if(x==NA_INTEGER)
   Rcpp::stop("Full geometry %s cannot contain missing values.",name);
  return out;
 }
 Rcpp::NumericVector input(value);
 Rcpp::IntegerVector out(input.size());
 for(R_xlen_t i=0;i<input.size();++i){
  const double x=input[i];
  if(allow_missing && std::isnan(x)){out[i]=NA_INTEGER;continue;}
  if(!std::isfinite(x) || x!=std::floor(x) ||
     x < -static_cast<double>(INT_MAX) || x>static_cast<double>(INT_MAX))
   Rcpp::stop("Full geometry %s must contain representable integers.",name);
  out[i]=static_cast<int>(x);
 }
 return out;
}

std::shared_ptr<FullTreeGeometry> geometry_from_spec(const ModelSpec& sp,
	const char* who){
	if(sp.leaf_model!="Full")
		Rcpp::stop("%s: only the Full leaf model carries a fixed tree.\n", who);
	const Rcpp::List& tr = sp.tree;
	const char* need[] = {"S", "J", "cuts", "node_offset", "node_size", "node_split",
		"rows", "knot_offset", "knot_size", "knots", "seed", "design", "cut_method",
		"depth", "n_split", "n_leaf"};
	for(const char* nm : need){
		if(tr.size()==0 || !tr.containsElementNamed(nm))
			Rcpp::stop("%s: the model carries no fixed tree (slot `tree`); it was built by an "
				"older ResTree -- rebuild it with restree_model().\n", who);
	}
	auto geom = std::make_shared<FullTreeGeometry>();
	geom->n = (int)sp.y.size(); geom->dim = (int)sp.X.cols();
	geom->depth = sp.depth; geom->r = sp.r; geom->design = sp.design;
	geom->cut_method = sp.cut_method; geom->seed = (unsigned int)sp.seed;
	Rcpp::NumericVector S=tr["S"],J=tr["J"],cuts=tr["cuts"];
 Rcpp::IntegerVector ns=full_integer_field(tr,"node_size"),
  no=full_integer_field(tr,"node_offset"),split=full_integer_field(tr,"node_split",true),
  ks=full_integer_field(tr,"knot_size"),ko=full_integer_field(tr,"knot_offset"),
  rows=full_integer_field(tr,"rows"),knots=full_integer_field(tr,"knots");
 const bool compact=tr.containsElementNamed("id");
 if(full_integer_field(tr,"depth",false,true)[0]!=sp.depth ||
    (compact && (!tr.containsElementNamed("format") || full_integer_field(tr,"format",false,true)[0]!=2)))
  Rcpp::stop("%s: stored geometry format/depth does not match the model.",who);
 const int count=ns.size(),numI=1<<sp.depth;
 Rcpp::IntegerVector ids=compact ? full_integer_field(tr,"id") :
  Rcpp::seq(1,count);
 if(count<1 || no.size()!=count || split.size()!=count || ks.size()!=count ||
    ko.size()!=count || ids.size()!=count ||
    S.size()!=(compact?count:numI) || J.size()!=S.size() || cuts.size()!=S.size() ||
    (!compact && count!=(2<<sp.depth)-1))
  Rcpp::stop("%s: the stored fixed tree does not match its depth/record count.",who);
 if(full_integer_field(tr,"seed",false,true)[0]!=sp.seed || Rcpp::as<std::string>(tr["design"])!=sp.design ||
    Rcpp::as<std::string>(tr["cut_method"])!=sp.cut_method)
  Rcpp::stop("%s: stored geometry settings differ from the model; rebuild it.",who);
 std::unordered_map<int,int> positions;
 int row_end=0,knot_end=0,nsplit=0,nleaf=0;
 for(int i=0;i<count;++i){
  if(no[i]<1 || ko[i]<1) Rcpp::stop("%s: invalid row/knot offset.",who);
  const int id=ids[i],sz=ns[i],off=no[i]-1,nk=ks[i],koff=ko[i]-1;
  if(id<1 || id>=(2<<sp.depth) || (i>0 && id<=ids[i-1]) ||
     sz<0 || nk<0 || off!=row_end || koff!=knot_end ||
     sz>rows.size()-off || nk>knots.size()-koff)
   Rcpp::stop("%s: corrupt stored tree addresses or noncontiguous row/knot blocks.",who);
  row_end+=sz;knot_end+=nk;
  const bool is_split=split[i]==1;
  if((sz==0 && split[i]!=NA_INTEGER) || (sz>0 && split[i]!=0 && split[i]!=1) ||
     is_split!=(sz>sp.r) || nk!=(is_split?sp.r:0))
   Rcpp::stop("%s: stored geometry violates Full's split iff n > r rule.",who);
  // Legacy dense tables may contain unreached empty positions.
  if(!compact && sz==0 && !(id>1 && split[id/2-1]==1)) continue;
  FullTreeNodeGeometry nd;nd.id=id;nd.split=is_split;
  for(int a=id;a>1;a>>=1) ++nd.level;
  if(is_split && nd.level>=sp.depth) Rcpp::stop("%s: split exceeds stored depth.",who);
  const int si=compact?i:id-1;
  const double expected=is_split?0.0:(sz>0 && nd.level<sp.depth?1.0:NA_REAL);
  if(si<S.size() && !(std::isnan(expected)?std::isnan(S[si]):S[si]==expected))
   Rcpp::stop("%s: stored S disagrees with node splits.",who);
  if(is_split){
   if(J[si]!=(nd.level%geom->dim)+1 || !std::isfinite(cuts[si]) || cuts[si]<0 || cuts[si]>1)
    Rcpp::stop("%s: invalid Full split dimension or cut.",who);
   nd.J=static_cast<int>(J[si])-1;nd.cut=cuts[si];++nsplit;
  }else{
   if(si<J.size() && (!std::isnan(J[si]) || !std::isnan(cuts[si])))
    Rcpp::stop("%s: terminal Full node carries a split.",who);
   if(sz>0) ++nleaf;
  }
  nd.idx.resize(sz);
  for(int j=0;j<sz;++j){
   const int raw_row=rows[off+j];
   if(raw_row<1 || raw_row>geom->n) Rcpp::stop("%s: row id outside training data.",who);
   const int row=raw_row-1;
   nd.idx[j]=row;
  }
  nd.kn.resize(nk);std::vector<char> taken(sz,0);
  for(int j=0;j<nk;++j){
   const int raw_knot=knots[koff+j];
   if(raw_knot<1 || raw_knot>sz) Rcpp::stop("%s: invalid knot position.",who);
   const int k=raw_knot-1;
   if(taken[k]) Rcpp::stop("%s: invalid or duplicated knot position.",who);
   nd.kn(j)=k;taken[k]=1;
  }
  if(is_split){
   nd.res.resize(sz-nk);int j=0;
   for(int k=0;k<sz;++k) if(!taken[k]) nd.res(j++)=k;
   nd.side.resize(sz-nk);
   for(int k=0;k<sz-nk;++k) nd.side[k]=!(sp.X(nd.idx[nd.res(k)],nd.J)<nd.cut);
  }
  positions.emplace(id,static_cast<int>(geom->nodes.size()));
  geom->nodes.push_back(std::move(nd));
 }
 if(row_end!=rows.size() || knot_end!=knots.size() || !positions.count(1) ||
    geom->nodes[0].idx.size()!=static_cast<size_t>(geom->n))
  Rcpp::stop("%s: invalid root or trailing row/knot payload.",who);
 for(int i=0;i<geom->n;++i) if(geom->nodes[0].idx[i]!=i)
  Rcpp::stop("%s: Full root rows must retain the training row order.",who);
 for(auto& nd:geom->nodes){
  if(nd.id>1 && (!positions.count(nd.id/2) || !geom->nodes[positions.at(nd.id/2)].split))
   Rcpp::stop("%s: unreachable stored Full node.",who);
  if(!nd.split) continue;
  if(!positions.count(2*nd.id) || !positions.count(2*nd.id+1))
   Rcpp::stop("%s: a Full split is missing a child record.",who);
  nd.left=positions.at(2*nd.id);nd.right=positions.at(2*nd.id+1);
  std::vector<int> left,right;
  for(int k=0;k<nd.res.size();++k)
   (nd.side[k]?right:left).push_back(nd.idx[nd.res(k)]);
  if(left!=geom->nodes[nd.left].idx || right!=geom->nodes[nd.right].idx)
   Rcpp::stop("%s: child rows disagree with the stored knots/cut and input routing; rebuild model.",who);
 }
 if(!tr.containsElementNamed("n_split") || !tr.containsElementNamed("n_leaf") ||
    full_integer_field(tr,"n_split",false,true)[0]!=nsplit ||
    full_integer_field(tr,"n_leaf",false,true)[0]!=nleaf)
  Rcpp::stop("%s: Full node summary counts disagree with its records.",who);
	return geom;
}

// Full model-local native cache. Only immutable geometry and owned data persist;
// all covariance factors are recomputed. The external-pointer attribute serializes
// without its native address, so restoration lazily rebuilds it. Exact data and
// payload comparisons invalidate it after R slot edits (no hash collision risk).
struct FullLoglikCache {
 std::shared_ptr<const FullTreeGeometry> geometry;
 std::unique_ptr<ResTree> model;
 Rcpp::List key;
 unsigned int preparations=0,evaluations=0;
};
SEXP full_loglik_cache_tag(){return Rf_install("ResTree::FullLoglikCache::v1");}
SEXP full_loglik_cache_attr(){return Rf_install(".restree_native_geometry");}
FullLoglikCache& full_loglik_cache(Rcpp::List tree){
 SEXP value=Rf_getAttrib(tree,full_loglik_cache_attr());
 if(TYPEOF(value)!=EXTPTRSXP || R_ExternalPtrTag(value)!=full_loglik_cache_tag() ||
    !R_ExternalPtrAddr(value)){
  Rcpp::XPtr<FullLoglikCache> ptr(new FullLoglikCache(),true,full_loglik_cache_tag());
  tree.attr(".restree_native_geometry")=ptr;
  value=ptr;
 }
 return *static_cast<FullLoglikCache*>(R_ExternalPtrAddr(value));
}
Rcpp::List full_payload_copy(const Rcpp::List& tree){
 Rcpp::List key(tree.size());
 for(int i=0;i<tree.size();++i) key[i]=Rcpp::clone(Rcpp::RObject(tree[i]));
 key.attr("names")=Rcpp::clone(Rcpp::RObject(tree.names()));
 return key; // deliberately no pointer attributes: no preserved-object cycle
}
bool full_payload_equal(const Rcpp::List& a,const Rcpp::List& b){
 if(a.size()!=b.size() || !R_compute_identical(a.names(),b.names(),0)) return false;
 for(int i=0;i<a.size();++i) if(!R_compute_identical(a[i],b[i],0)) return false;
 return true;
}
StructureEval cached_full_loglik(const ModelSpec& sp,const GPM& th,int ncores){
 auto& cache=full_loglik_cache(sp.tree);
 const auto* model=cache.model.get();
 const bool same=model && cache.geometry && model->X.rows()==sp.X.rows() &&
  model->X.cols()==sp.X.cols() && model->y.size()==sp.y.size() &&
  cache.geometry->depth==sp.depth && cache.geometry->r==sp.r &&
  cache.geometry->seed==static_cast<unsigned int>(sp.seed) &&
  cache.geometry->design==sp.design && cache.geometry->cut_method==sp.cut_method &&
  std::memcmp(model->X.data(),sp.X.data(),sp.X.size()*sizeof(double))==0 &&
  std::memcmp(model->y.data(),sp.y.data(),sp.y.size()*sizeof(double))==0 &&
  full_payload_equal(sp.tree,cache.key);
 if(!same){
  validate_model_data(sp);
  auto geometry=geometry_from_spec(sp,"restree_loglik");
  std::unique_ptr<ResTree> fresh(new ResTree(sp.y,sp.X));
  Rcpp::List key=full_payload_copy(sp.tree);
  cache.model=std::move(fresh);cache.geometry=std::move(geometry);cache.key=key;
  ++cache.preparations;
 }
 configure_fixed(*cache.model,sp,th,ncores);
 auto result=cache.model->evaluate_geometry(*cache.geometry);
 ++cache.evaluations;
 return result;
}

// The fixed Full tree of a model, built once by restree_model(leaf_model =
// "Full") and stored in its `tree` slot (see geometry_to_list): a node splits
// iff it has more than r rows, extending depth as needed, and spends r knots (pick_knots on the
// node rows, `design`), the split dimension cycles level %% d, the cut comes
// from `cut_method` (uniform / balanced draw per node from `seed`).  Nothing
// here reads a covariance parameter.
// [[Rcpp::export]]
Rcpp::List fixed_tree_geometry_cpp(Eigen::MatrixXd x, int depth, int r,
	std::string design, std::string cut_method, int seed=42, int ncores=1){
	// restree_model validates x and every geometry option before this call.
	ResTree model_(Eigen::VectorXd::Zero(x.rows()), x);
	model_.depth = depth; model_.r_knot = r; model_.design = design;
	configure_threads(model_, ncores);
	std::shared_ptr<FullTreeGeometry> geom =
		model_.build_generated_geometry(cut_method, (unsigned int)seed);
	Rcpp::List out=geometry_to_list(*geom);
	full_loglik_cache(out); // empty tagged pointer: stable serialization before/after use
	return out;
}

// Shared validity for compact and legacy Full geometry; main thread only.
// [[Rcpp::export]]
bool full_tree_validity_cpp(SEXP model){
 const auto sp=read_model(model);
 geometry_from_spec(sp,"Full validity");
 return true;
}

// Native cache statistics.
// [[Rcpp::export]]
Rcpp::List full_loglik_cache_info_cpp(SEXP model){
 const auto sp=read_model(model);
 auto& cache=full_loglik_cache(sp.tree);
 return Rcpp::List::create(Rcpp::_["preparations"]=cache.preparations,
  Rcpp::_["evaluations"]=cache.evaluations,
  Rcpp::_["nodes"]=cache.geometry?cache.geometry->nodes.size():0);
}

// ==================== Stateless and prepared Full fitting ====================

namespace {
// The Gaussian fit statistics of a fixed-tree fit.
void add_gaussian_fit_stats(Rcpp::List& out, const StructureEval& ev,
	const std::string& baseline, double sig2, const char* who){
	const bool gauss_ok = (baseline=="Full" || baseline=="PP");
	if(gauss_ok && (!std::isfinite(ev.stats.quad) ||
		!std::isfinite(ev.stats.logdet))){
		Rcpp::stop("%s produced non-finite Gaussian fit statistics.\n", who);
	}
	const double ll_cond = restree_loglik_conditional(ev.stats.logdet,
		ev.stats.quad, ev.stats.n_gauss, sig2);
	if(gauss_ok && !std::isfinite(ll_cond)){
		Rcpp::stop("%s produced a non-finite conditional log-likelihood.\n", who);
	}
	out["sigma2"] = sig2;
	out["quad"] = gauss_ok ? ev.stats.quad : NA_REAL;
	out["logdet"] = gauss_ok ? ev.stats.logdet : NA_REAL;
	out["n_gauss"] = ev.stats.n_gauss;
	out["n_knot"] = ev.stats.n_knot_live;
	out["loglik_conditional"] = gauss_ok ? ll_cond : NA_REAL;
}

}
namespace {

// Fit-local ownership: no borrowed R slots and no theta-dependent factors are
// retained between evaluations. Full owns a concrete ResTree and one compact
// fitted tree; it neither constructs nor depends on an SMC sampler.
struct FullFitEngine {
	ResTree eng;
	const std::shared_ptr<const FullTreeGeometry> geometry;
	FittedTrees trees; // initialized only if predictions are requested
	FullFitEngine(const ModelSpec& sp, const GPM& th, int ncores)
		: eng(sp.y, sp.X), geometry(geometry_from_spec(sp, "full_fit_engine_new_cpp")){
		configure_fixed(eng, sp, th, ncores);
	}
};

SEXP full_fit_engine_tag(){ return Rf_install("ResTree::FullFitEngine::v2"); }

FullFitEngine& full_fit_engine_from_handle(SEXP handle){
	if(TYPEOF(handle)!=EXTPTRSXP || R_ExternalPtrTag(handle)!=full_fit_engine_tag() ||
		R_ExternalPtrAddr(handle)==nullptr){
		Rcpp::stop("invalid Full MLE workspace; create a new workspace for this fit.\n");
	}
	return *static_cast<FullFitEngine*>(R_ExternalPtrAddr(handle));
}

// The stateless reference and prepared MLE use exactly the same arithmetic.
Rcpp::List evaluate_full_fit(ResTree& eng,
	const FullTreeGeometry& geometry, FittedTrees& trees,
	const Eigen::Ref<const Eigen::MatrixXd>& input_new,
	Rcpp::Nullable<Eigen::VectorXd> output_new,
	std::chrono::steady_clock::time_point t0, const char* who){

	// R's .restree_xnew validates values once; only native indexing guards stay.
	if(input_new.cols()!=eng.dim)
		Rcpp::stop("%s: input_new must have the training input dimension.\n", who);
	Eigen::VectorXd yn;
	const bool has_y = output_new.isNotNull();
	if(has_y){
		yn = Rcpp::as<Eigen::VectorXd>(output_new.get());
		if(yn.size()!=input_new.rows())
			Rcpp::stop("%s: output_new must have one value per test row.\n", who);
	}
	reapply_threads(eng);
	reset_covariance_error();
	const double sig2 = eng.covpar.sig2;
	StructureEval ev = eng.evaluate_geometry(geometry);
	stop_on_covariance_error(who);
	if(!ev.stats.ok){
		Rcpp::stop("%s failed: %s", who, ev.stats.error.c_str());
	}
	if(!std::isfinite(ev.loglik)){
		Rcpp::stop("%s produced a non-finite tree log-likelihood.\n", who);
	}
	auto t1 = std::chrono::steady_clock::now();

	Eigen::VectorXd pred_mean(0), pred_var(0);
	Eigen::VectorXd lpd_tree(0);
	if(input_new.rows()>0){
		if(trees.nodes.empty()) trees=full_prediction_trees(geometry);
		Eigen::MatrixXd pm, pv;
  Predictor(eng,trees).predict(input_new,std::vector<int>{0},sig2,pm,pv,
   has_y?&yn:nullptr,has_y?&lpd_tree:nullptr);
		stop_on_covariance_error(who);
		pred_mean = pm.col(0);
		pred_var = pv.col(0);
		if(!pred_mean.allFinite() || !pred_var.allFinite()){
			Rcpp::stop("%s produced non-finite prediction moments.\n", who);
		}
	}
	auto t2 = std::chrono::steady_clock::now();

	Rcpp::List out = Rcpp::List::create(
		Rcpp::_["mean"] = pred_mean,
		Rcpp::_["var"] = pred_var,
		Rcpp::_["tree_loglik"] = ev.loglik,
		Rcpp::_["loglik"] = ev.loglik,
		Rcpp::_["fit_time"] = std::chrono::duration<double>(t1-t0).count(),
		Rcpp::_["pred_time"] = std::chrono::duration<double>(t2-t1).count());
	add_gaussian_fit_stats(out, ev, eng.baseline, sig2, who);
	if(has_y && input_new.rows()>0 && lpd_tree.size()>0){
		if(!std::isfinite(lpd_tree(0))){
			Rcpp::stop("%s produced a non-finite predictive density.\n", who);
		}
		out["lpd"] = lpd_tree(0);
	}

	// Compact geometry records; explicit matrix export is an R API operation.
 out["geometry_nodes"] = static_cast<double>(geometry.nodes.size());
	return out;
}

}  // namespace

// Stateless fixed Full evaluation, retained as a reference and for one-off use.
// Reconstructs the same stored geometry; does not select another tree or knots.
// [[Rcpp::export]]
Rcpp::List fixed_tree_fit_cpp(SEXP model, SEXP theta, const Eigen::Map<Eigen::MatrixXd> input_new,
	Rcpp::Nullable<Eigen::VectorXd> output_new=R_NilValue, int ncores=1){
	const ModelSpec sp = read_model(model);
	const GPM th = read_theta(theta, (int)sp.X.cols());
	const auto t0 = std::chrono::steady_clock::now();
	ResTree eng(sp.y, sp.X);
	configure_fixed(eng, sp, th, ncores);
	const auto geometry = geometry_from_spec(sp, "fixed_tree_fit_cpp");
	FittedTrees trees;
	return evaluate_full_fit(eng, *geometry, trees, input_new, output_new, t0, "fixed_tree_fit_cpp");
}

// Internal Full MLE workspace: prepared once outside the multi-start optimizer.
// Its tagged external pointer owns the training data and immutable geometry;
// the Rcpp finalizer releases them when the fit-local handle is collected.
// [[Rcpp::export]]
SEXP full_fit_engine_new_cpp(SEXP model, SEXP theta, int ncores=1){
	const ModelSpec sp = read_model(model);
	const GPM th = read_theta(theta, (int)sp.X.cols());
	std::unique_ptr<FullFitEngine> engine(new FullFitEngine(sp, th, ncores));
	Rcpp::XPtr<FullFitEngine> ptr(engine.get(), true, full_fit_engine_tag());
	engine.release();
	return ptr;
}

// Update theta only: geometry and owned data survive until this fit finishes.
// [[Rcpp::export]]
Rcpp::List full_fit_engine_eval_cpp(SEXP handle, SEXP theta, const Eigen::Map<Eigen::MatrixXd> input_new,
	Rcpp::Nullable<Eigen::VectorXd> output_new=R_NilValue){
	FullFitEngine& workspace = full_fit_engine_from_handle(handle);
	const GPM th = read_theta(theta, workspace.eng.dim);
	workspace.eng.covpar = th;
	return evaluate_full_fit(workspace.eng, *workspace.geometry, workspace.trees, input_new, output_new,
		std::chrono::steady_clock::now(), "full_fit_engine_eval_cpp");
}

// Free the workspace now (see smc_engine_release_cpp); a NULL, foreign or
// released handle is a no-op.
// [[Rcpp::export]]
void full_fit_engine_release_cpp(SEXP handle){
	if(TYPEOF(handle)!=EXTPTRSXP || R_ExternalPtrTag(handle)!=full_fit_engine_tag()) return;
	FullFitEngine* p = static_cast<FullFitEngine*>(R_ExternalPtrAddr(handle));
	if(p==nullptr) return;
	R_ClearExternalPtr(handle);
	delete p;
}
