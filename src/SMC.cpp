// Sequential Monte Carlo over residual-tree structures.
#include "SMC.h"
#include "wrapper.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <vector>

// ==================== Particle transitions and cut draws ====================

double SMC::update_ancestor_loglik(int p, int id, double value){
	return propagate_particle_loglik(sparse_particles[p],id,value,
		[this](int t){ return trie[t].loglik_knot; });
}

// One particle's move at node id: the draws and O(level) node lookups of
// draw_OneStepAhead_Proposal_Level_Order, with all heavy quantities read
// from the trie.
void SMC::transition(int p, int id, std::mt19937_64& rng,
	double& Phi, double& w_log){

	SparseNode* nd = find_node(p,id);
	if(!nd || nd->act!=ACT_PENDING){ return; } // absent / empty / terminal: no-op

	const int32_t t = nd->trie_at;
	// Copy the sampling quantities out of the trie entry: trie_child() below
	// can reallocate the trie vector, so no reference may be held across it.
	const double rho = trie[t].rho;
	const double rho_prior = prior_stop(trie[t].level);

	// Draw the action from the exact local posterior (rho, lambda) -- no
	// mixture.  Deterministic cuts consume no cut randomness.
	std::bernoulli_distribution rbern(rho);
	const int S = rbern(rng) ? 1 : 0;

	if(S==1){ // stop
		nd->act = ACT_STOP;
		// Phi unchanged; prior/proposal ratio for the realized stop:
		w_log = w_log + std::log(rho_prior) - std::log(rho);
		return;
	}

	// split: draw the dimension from the clamped posterior lambda
	record_split_depth(trie[t].level);
	std::discrete_distribution<int> rcat(trie[t].lambda.data(), trie[t].lambda.data()+dim);
	const int Ji = rcat(rng);
	const double lambda_log_Ji = trie[t].lambda_log(Ji);

	nd->act = ACT_SPLIT;
	nd->jdim = (int8_t)Ji;
	nd->cutval = trie[t].cuts(Ji);

	double child_ll[2];
	for(int side=0; side<2; ++side){
		const int cid = 2*id + side;
		const int cn = trie[t].child_n(Ji, side);
		SparseNode child;
		child_ll[side] = child.Loglik = trie[t].child_loglik_lr(Ji, side);
		if(cn==0){
			child.act = ACT_EMPTY;
		}else if(!compare_node(trie[t].level+1,cn)){
			child.act = ACT_BOTTOM;
		}else{
			child.act = ACT_PENDING;
			child.trie_at = trie.child(t, Ji, side, dim);
		}
		insert_node(p,cid,child);
	}

	// Map node references survive insertion; path-trie references do not.
	// Preserve the original knot + (left + right), then nearest ancestor first.
	nd->Loglik = trie[t].loglik_knot + (child_ll[0] + child_ll[1]);
	const double Phi_new = update_ancestor_loglik(p,id,nd->Loglik);
	const double logBF = Phi_new - Phi;

	// The proposal probability of this split is (1 - rho) * lambda(Ji); the
	// weight divides by exactly the probabilities the action was drawn from.
	const double log1m_rho_post = std::log1p(-rho);
	const double log_proposal_ratio = std::log1p(-rho_prior) - log1m_rho_post
		+ std::log(prior_lambda(Ji)) - lambda_log_Ji;
	w_log = w_log + log_proposal_ratio + logBF;
	Phi = Phi_new;
}


// Select the smaller branch using its probability directly. In particular,
// a rare split is not lost by computing 1-exp(log posterior stop).
static bool draw_log_mass_left(double left, double right, std::mt19937_64& rng) {
 const double neginf=-std::numeric_limits<double>::infinity();
 if(left==neginf) return false;
 if(right==neginf) return true;
 const double log_total=log_sum_exp2(left,right);
 const bool left_smaller=left<right;
 std::bernoulli_distribution rare(std::exp(std::min(left,right)-log_total));
 return rare(rng) ? left_smaller : !left_smaller;
}

void SMC::draw_integrated_wn_cut_move(int p, int id, std::mt19937_64& rng,
 RandomCutMove& out) {
 auto* nd=worker_node(p,id); if(!nd) return;
 const PathNode& node=trie[nd->trie_at];
 out.integrated_wn=true;
 // expand() skips all split work for a WN node with prior stop probability one.
 if(prior_stop(node.level)==1.0) {
  out.active=true; out.S=1; out.rho=1.0; out.log_weight_increment=0.0;
  nd->rhoval=1.0; return;
 }
 if(!node.proposals.wn) {
  int expected=-1;
  internal_error_node.compare_exchange_strong(expected,id);
  return;
 }
 const WNCutProposal& proposal=*node.proposals.wn;
 out.active=true;
 out.rho=node.rho;
 nd->rhoval=out.rho;
 out.log_weight_increment=proposal.log_weight_increment;
 out.S=draw_log_mass_left(proposal.log_stop_mass,proposal.log_split_mass,rng) ? 1 : 0;
 if(out.S==1) return;
 size_t index=1;
 while(index<proposal.leaf_offset) {
  const bool left=draw_log_mass_left(proposal.log_mass_tree[2*index],
   proposal.log_mass_tree[2*index+1],rng);
  index=2*index+(left ? 0 : 1);
 }
 const WNCutInterval& interval=proposal.intervals[index-proposal.leaf_offset];
 out.Ji=interval.coordinate;
 out.cell=interval.rank_index;
 out.cut_rep=interval.cut_upper;   // any cut in (lower, upper] realises the same partition
 out.cut=interval.cut_lower;
 if(interval.cut_lower!=interval.cut_upper) {
  const int m=proposal.n_residual;
  const double a=(double)interval.rank_index/(m-1), b=(double)(interval.rank_index+1)/(m-1);
  std::uniform_real_distribution<double> uniform(0.0,1.0);
  double fraction=uniform(rng);
  if(cut_method=="balanced") {
   // Invert the truncated Beta(2,2) CDF on its own interval. This keeps
   // tail probabilities accurate and uses no R API on an OpenMP worker.
   const double target=fraction*beta22_interval_mass(a,b);
   double lo=0.0,hi=1.0;
   for(int k=0; k<52; ++k) {
    const double mid=lo+0.5*(hi-lo);
    if(beta22_interval_mass(a,a+(b-a)*mid)<target) lo=mid; else hi=mid;
   }
   fraction=lo+0.5*(hi-lo);
  }
  out.cut=(1.0-fraction)*interval.cut_lower+fraction*interval.cut_upper;
  // Floating-point interpolation must realize the partition whose mass
  // was selected: lower < c <= upper, including adjacent double values.
  out.cut=std::min(interval.cut_upper,std::max(out.cut,
   std::nextafter(interval.cut_lower,interval.cut_upper)));
 }
 out.child_n[0]=interval.n_left;
 out.child_n[1]=proposal.n_residual-interval.n_left;
 out.child_ll[0]=interval.left_loglik;
 out.child_ll[1]=interval.right_loglik;
 out.child_st[0].clear(); out.child_st[1].clear();
}

// Quantile-cut move: a thread-parallel DRAW pass (each particle touches only
// its own RNG stream and the level-cached trie state) plus a serial COMMIT
// pass in particle order, so results are identical for every ncores.  Draw
// order: cuts, then S, then J.
void SMC::draw_random_cut_move(int p, int id, std::mt19937_64& rng,
	RandomCutMove& out){

	out.active = false;
 out.integrated_wn = false;
	auto* nd = find_node(p,id);
	if(!nd || nd->act!=ACT_PENDING){ return; } // absent / empty / terminal: no-op
 if(baseline=="WhiteNoise") {
  draw_integrated_wn_cut_move(p,id,rng,out);
  return;
 }
 // Nested PP integrates its own rank-cell prior and owns a different cache.
 // M controls only ordinary PP proposals, never the nested transition.
 if(nested()){ draw_nested_pp_cut_move(p, id, rng, out); return; }
 if(cut_candidates>1) {
  draw_multi_cut_move(p,id,rng,out);
  return;
 }

	const int32_t t = nd->trie_at;
	const double loglik_base = trie[t].loglik_base;
	const double loglik_knot = trie[t].loglik_knot;
	const int n_res = trie[t].postknot.residual.size();
	if(!trie[t].postknot.ready){
		// OpenMP worker: never throw here.  Record the node id and return;
		// sample_level raises the R error after the parallel region.
		int expected = -1;
		internal_error_node.compare_exchange_strong(expected, id);
		return;
	}

	// ---- quantile-sampled cuts: a node with no residual rows consumes no
	// randomness and takes the middle cut.
	Eigen::VectorXd cuts(dim);
	if(n_res==0){
		for(int J=0; J<dim; J++){
			cuts(J) = (trie[t].box_min(J) + trie[t].box_max(J)) / 2.0;
		}
	}else{
		const Eigen::MatrixXd& Srt = trie[t].cut_workspace.sorted;
		const int nv = n_res;
		for(int J=0; J<dim; J++){
			double pos;
			if(cut_method=="balanced"){
				std::gamma_distribution<double> rgam(2.0, 1.0);
				double g1 = rgam(rng);
				double g2 = rgam(rng);
				double u = (g1+g2 > 0.0) ? g1/(g1+g2) : 0.5;
				pos = u * (double)(nv-1);
			}else{ // "uniform": uniform in rank space
				std::uniform_real_distribution<double> runif(0.0, 1.0);
				pos = runif(rng) * (double)(nv-1);
			}
			int lo = (int) std::floor(pos);
			int hi = (lo+1<nv) ? (lo+1) : lo;
			double w = pos - (double)lo;
			cuts(J) = (1.0-w)*Srt(lo,J) + w*Srt(hi,J);
		}
	}

	// ---- candidate children for this particle's cuts ----
	Eigen::VectorXd child_loglik(dim);
	Eigen::MatrixXd child_loglik_lr(dim, 2);
	std::vector<BlockStats> child_stats(2*dim);
	Eigen::ArrayXXi child_n(dim, 2);
	{
		const Eigen::MatrixXd& Xres = trie[t].cut_workspace.input;
		const Eigen::VectorXd& zres = trie[t].postknot.residual;
		const Eigen::MatrixXd& Qn = trie[t].postknot.coefficients;
		const Eigen::VectorXd bmin = trie[t].box_min, bmax = trie[t].box_max;
		for(int J=0; J<dim; J++){
			const double delta = cuts(J);
			int nl=0, nr=0;
			Eigen::VectorXi ind_left(n_res), ind_right(n_res);
			for(int i=0; i<n_res; i++){
				if(Xres(i,J) < delta){ ind_left(nl++) = i; }
				else{ ind_right(nr++) = i; }
			}
			child_n(J,0) = nl; child_n(J,1) = nr;
			double ll_l = 0.0, ll_r = 0.0;
			// Score-only leaves on the node's block through the row lists
			// (no gathered copies; terminal_loglik_rows).
			if(nl>0){
				Eigen::VectorXd bmax_l = bmax; bmax_l(J) = delta;
				ll_l = terminal_loglik_rows(Xres, zres, Qn, ind_left.data(), nl,
					bmin, bmax_l, child_stats[2*J+0]);
			}
			if(nr>0){
				Eigen::VectorXd bmin_r = bmin; bmin_r(J) = delta;
				ll_r = terminal_loglik_rows(Xres, zres, Qn, ind_right.data(), nr,
					bmin_r, bmax, child_stats[2*J+1]);
			}
			child_loglik_lr(J,0) = ll_l; child_loglik_lr(J,1) = ll_r;
			if(nl==0 && nr==0){ child_loglik(J) = 0.0; }
			else if(nl==0){ child_loglik(J) = ll_r; }
			else if(nr==0){ child_loglik(J) = ll_l; }
			else{ child_loglik(J) = ll_l + ll_r; }
		}
	}

	// ---- posteriors at this particle's cuts (clamped away from 0 and 1) ----
	const double rho_prior = prior_stop(trie[trie_at_value(p, id)].level);
	double Phi_ss = spike_slab_loglik(loglik_base, loglik_knot,
		child_loglik, rho_prior, prior_lambda);
	double rho = std::exp(std::log(rho_prior) + loglik_base - Phi_ss);
	if(rho<0.0){ rho = 0.0; }
	if(rho>1.0){ rho = 1.0; }
	Eigen::VectorXd lambda(dim), lambda_log(dim);
	{
		Eigen::VectorXd tmp2 = prior_lambda.array().log() + loglik_knot + child_loglik.array();
		double loglik_slab = log_sum_exp(tmp2);
		for(int J=0; J<dim; J++){
			lambda_log(J) = tmp2(J) - loglik_slab;
			lambda(J) = std::exp(lambda_log(J));
			if(lambda(J)<0.0){ lambda(J) = 0.0; }
			if(lambda(J)>1.0){ lambda(J) = 1.0; }
		}
	}
	nd->rhoval = rho; // model state: the POSTERIOR, not the proposal

	// ---- draws (per-particle stream; order: S, then J) ----
	// out.rho / out.lambda_log_Ji carry the exact draw probabilities, so the
	// commit's weight update is coherent by construction.
	std::bernoulli_distribution rbern(rho);
	out.active = true;
	out.rho = rho;
	out.S = rbern(rng) ? 1 : 0;
	if(out.S==1){ return; } // stop: cut draws above were still consumed

	std::discrete_distribution<int> rcat(lambda.data(), lambda.data()+dim);
	const int Ji = rcat(rng);
	out.Ji = Ji;
	out.cut = cuts(Ji);
	out.lambda_log_Ji = lambda_log(Ji);
	for(int side=0; side<2; ++side){
		out.child_n[side] = child_n(Ji, side);
		out.child_ll[side] = child_loglik_lr(Ji, side);
		out.child_st[side] = child_stats[2*Ji+side];
	}
}

// M-candidate PP move (cut_candidates > 1).  For every split dimension J,
// M cuts are drawn from the cut prior (uniform or Beta(2,2) in rank space) and
// the PP child pair is evaluated at each.  With
//   A_J = log( (1/M) sum_m exp(child_loglik(J, m)) )
// an unbiased estimate of the cut-integrated slab marginal of dimension J,
//   Phi_hat = rho e^{loglik_base} + (1 - rho) e^{loglik_knot} sum_J lambda_J e^{A_J}
// is an unbiased estimate of the one-step normaliser over (stop, J, cut).
// The move draws stop with probability rho e^{base}/Phi_hat, otherwise J with
// probability proportional to lambda_J e^{A_J} and the candidate m within J
// with probability proportional to exp(child_loglik(J, m)); the weight
// increment is log Phi_hat - loglik_base for every outcome.  This is the
// fully adapted move on the extended target (the M d candidates and the
// selected index), so the increment does not depend on the outcome and log Z
// stays unbiased. IID draws are not exact enumeration at any finite M.
void SMC::draw_multi_cut_move(int p, int id, std::mt19937_64& rng,
	RandomCutMove& out){

	auto* nd=worker_node(p,id); if(!nd) return;
	const int32_t t = nd->trie_at;
	const double loglik_base = trie[t].loglik_base;
	const double loglik_knot = trie[t].loglik_knot;
	const int n_res = trie[t].postknot.residual.size();
	if(!trie[t].postknot.ready){
		int expected = -1;
		internal_error_node.compare_exchange_strong(expected, id);
		return;
	}
	const int M = cut_candidates;

	// ---- M prior cuts per dimension ----
	Eigen::MatrixXd cuts(dim, M);
	if(n_res==0){
		for(int J=0; J<dim; J++){
			cuts.row(J).setConstant((trie[t].box_min(J) + trie[t].box_max(J)) / 2.0);
		}
	}else{
		const Eigen::MatrixXd& Srt = trie[t].cut_workspace.sorted;
		const int nv = n_res;
		std::gamma_distribution<double> rgam(2.0, 1.0);
		std::uniform_real_distribution<double> runif(0.0, 1.0);
		for(int J=0; J<dim; J++){
			for(int m=0; m<M; m++){
				double pos;
				if(cut_method=="balanced"){
					double g1 = rgam(rng), g2 = rgam(rng);
					double u = (g1+g2 > 0.0) ? g1/(g1+g2) : 0.5;
					pos = u * (double)(nv-1);
				}else{
					pos = runif(rng) * (double)(nv-1);
				}
				int lo = (int) std::floor(pos);
				int hi = (lo+1<nv) ? (lo+1) : lo;
				double w = pos - (double)lo;
				cuts(J,m) = (1.0-w)*Srt(lo,J) + w*Srt(hi,J);
			}
		}
	}

	// ---- PP child pair at every candidate ----
	Eigen::MatrixXd child_loglik(dim, M);            // ll_left + ll_right
	std::vector<double> child_ll_lr((size_t)dim*M*2);
	std::vector<BlockStats> child_stats((size_t)dim*M*2);
	std::vector<int> child_n((size_t)dim*M*2);
	{
		const Eigen::MatrixXd& Xres = trie[t].cut_workspace.input;
		const Eigen::VectorXd& zres = trie[t].postknot.residual;
		const Eigen::MatrixXd& Qn = trie[t].postknot.coefficients;
		const Eigen::VectorXd bmin = trie[t].box_min, bmax = trie[t].box_max;
		Eigen::VectorXi ind_left(n_res), ind_right(n_res);
		for(int J=0; J<dim; J++){
			for(int m=0; m<M; m++){
				const double delta = cuts(J,m);
				int nl=0, nr=0;
				for(int i=0; i<n_res; i++){
					if(Xres(i,J) < delta){ ind_left(nl++) = i; }
					else{ ind_right(nr++) = i; }
				}
				const size_t kL = ((size_t)J*M + m)*2, kR = kL+1;
				child_n[kL] = nl; child_n[kR] = nr;
				double ll_l = 0.0, ll_r = 0.0;
				if(nl>0){
					Eigen::VectorXd bmax_l = bmax; bmax_l(J) = delta;
					ll_l = terminal_loglik_rows(Xres, zres, Qn, ind_left.data(), nl,
						bmin, bmax_l, child_stats[kL]);
				}
				if(nr>0){
					Eigen::VectorXd bmin_r = bmin; bmin_r(J) = delta;
					ll_r = terminal_loglik_rows(Xres, zres, Qn, ind_right.data(), nr,
						bmin_r, bmax, child_stats[kR]);
				}
				child_ll_lr[kL] = ll_l; child_ll_lr[kR] = ll_r;
				child_loglik(J,m) = (nl==0 && nr==0) ? 0.0 : (nl==0 ? ll_r : (nr==0 ? ll_l : ll_l + ll_r));
			}
		}
	}

	// ---- per-dimension estimate of the integrated slab marginal ----
	Eigen::VectorXd A(dim);
	for(int J=0; J<dim; J++){
		A(J) = log_sum_exp(child_loglik.row(J).transpose()) - std::log((double)M);
	}
	const double rho_prior = prior_stop(trie[t].level);
	const double log_Phi_hat = spike_slab_loglik(loglik_base, loglik_knot, A, rho_prior, prior_lambda);
	double rho = std::exp(std::log(rho_prior) + loglik_base - log_Phi_hat);
	if(rho<0.0){ rho = 0.0; }
	if(rho>1.0){ rho = 1.0; }
	Eigen::VectorXd lambda(dim);
	{
		Eigen::VectorXd tmp2 = prior_lambda.array().log() + A.array();
		const double lse = log_sum_exp(tmp2);
		for(int J=0; J<dim; J++){
			lambda(J) = std::exp(tmp2(J) - lse);
			if(lambda(J)<0.0){ lambda(J) = 0.0; }
			if(lambda(J)>1.0){ lambda(J) = 1.0; }
		}
	}
	nd->rhoval = rho;

	// ---- draws: stop / dimension / candidate; the increment is outcome-free ----
	out.active = true;
	out.direct_increment = true;
	out.log_weight_increment = log_Phi_hat - loglik_base;
	out.rho = rho;
	std::bernoulli_distribution rbern(rho);
	out.S = rbern(rng) ? 1 : 0;
	if(out.S==1){ return; }
	std::discrete_distribution<int> rcat(lambda.data(), lambda.data()+dim);
	const int Ji = rcat(rng);
	Eigen::VectorXd pm(M);
	{
		const double mx = child_loglik.row(Ji).maxCoeff();
		for(int m=0; m<M; m++){ pm(m) = std::exp(child_loglik(Ji,m) - mx); }
	}
	std::discrete_distribution<int> rm(pm.data(), pm.data()+M);
	const int mi = rm(rng);
	out.Ji = Ji;
	out.cut = cuts(Ji, mi);
	out.lambda_log_Ji = std::log(std::max(lambda(Ji), 1e-300));  // informational only
	for(int side=0; side<2; ++side){
		const size_t k = ((size_t)Ji*M + mi)*2 + side;
		out.child_n[side] = child_n[k];
		out.child_ll[side] = child_ll_lr[k];
		out.child_st[side] = child_stats[k];
	}
}

// ---- PP random cuts, shared candidate blocks (three phases; see the
// declaration in ResTree.h). Phase 1: prior-stratified midpoint coverage
// for uniform, the original IID draws for balanced, and left-row counts.
void SMC::pp_draw_cuts(int p, int id, std::mt19937_64& rng, PPCutDraw& d){
	d.active = false;
	const auto* nd=find_node(p,id);
	if(!nd || nd->act!=ACT_PENDING){ return; }
	const int32_t t = nd->trie_at;
	d.t = t;
	if(!trie[t].postknot.ready){
		int expected = -1;
		internal_error_node.compare_exchange_strong(expected, id);
		return;
	}
	const int n_res = trie[t].postknot.residual.size();
	// Always cover the rank cell containing (or nearest to) the region
	// midpoint: one draw there and M draws from the complementary cells.
	// This stratifies the ORIGINAL uniform rank prior; it adds no point mass.
	const bool midpoint_strata = cut_method=="uniform";
	const int gaps = std::max(1, n_res-1);
	const int M = midpoint_strata ? (gaps>1 ? cut_candidates+1 : 1) : cut_candidates;
	d.cuts.resize(dim, M);
	d.n_left.resize(dim, M);
	d.log_mass.resize(midpoint_strata ? M : 0);
	if(midpoint_strata){
		d.log_mass(0) = -std::log((double)gaps);
		if(M>1) d.log_mass.tail(M-1).setConstant(
			std::log1p(-1.0/(double)gaps)-std::log((double)cut_candidates));
		std::uniform_real_distribution<double> runif(0.0, 1.0);
		for(int J=0; J<dim; ++J){
			const double midpoint = 0.5*(trie[t].box_min(J)+trie[t].box_max(J));
			if(n_res<2){
				d.cuts(J,0) = n_res==0 ? midpoint : trie[t].cut_workspace.sorted(0,J);
				continue;
			}
			const double* sorted = trie[t].cut_workspace.sorted.col(J).data();
			const int below = (int)(std::lower_bound(sorted, sorted+n_res, midpoint)-sorted);
			const int focus = std::max(0, std::min(gaps-1, below-1));
			for(int m=0; m<M; ++m){
				double pos;
				if(m==0){ pos = (double)focus + runif(rng); }
				else{
					pos = runif(rng)*(double)(gaps-1);
					if(pos >= (double)focus) pos += 1.0;
				}
				const int lo = std::min(gaps-1, (int)std::floor(pos));
				const double fraction = pos-lo;
				const double a = sorted[lo], b = sorted[lo+1];
				d.cuts(J,m) = (1.0-fraction)*a + fraction*b;
				// Keep a positive-width cell's lower endpoint on the left if
				// interpolation rounds to it. Flat cells retain their atom.
				if(a<b && d.cuts(J,m)<=a) d.cuts(J,m) = std::nextafter(a,b);
			}
		}
	}else if(n_res==0){
		for(int J=0; J<dim; J++){
			d.cuts.row(J).setConstant((trie[t].box_min(J) + trie[t].box_max(J)) / 2.0);
		}
	}else{
		const Eigen::MatrixXd& Srt = trie[t].cut_workspace.sorted;
		const int nv = n_res;
		if(M==1){
			for(int J=0; J<dim; J++){
				double pos;
				if(cut_method=="balanced"){
					std::gamma_distribution<double> rgam(2.0, 1.0);
					double g1 = rgam(rng);
					double g2 = rgam(rng);
					double u = (g1+g2 > 0.0) ? g1/(g1+g2) : 0.5;
					pos = u * (double)(nv-1);
				}else{
					std::uniform_real_distribution<double> runif(0.0, 1.0);
					pos = runif(rng) * (double)(nv-1);
				}
				int lo = (int) std::floor(pos);
				int hi = (lo+1<nv) ? (lo+1) : lo;
				double w = pos - (double)lo;
				d.cuts(J,0) = (1.0-w)*Srt(lo,J) + w*Srt(hi,J);
			}
		}else{
			std::gamma_distribution<double> rgam(2.0, 1.0);
			std::uniform_real_distribution<double> runif(0.0, 1.0);
			for(int J=0; J<dim; J++){
				for(int m=0; m<M; m++){
					double pos;
					if(cut_method=="balanced"){
						double g1 = rgam(rng), g2 = rgam(rng);
						double u = (g1+g2 > 0.0) ? g1/(g1+g2) : 0.5;
						pos = u * (double)(nv-1);
					}else{
						pos = runif(rng) * (double)(nv-1);
					}
					int lo = (int) std::floor(pos);
					int hi = (lo+1<nv) ? (lo+1) : lo;
					double w = pos - (double)lo;
					d.cuts(J,m) = (1.0-w)*Srt(lo,J) + w*Srt(hi,J);
				}
			}
		}
	}
	// left-row count of every cut: the rows with X_J below it.  cut_workspace.sorted
	// (expand(): every column of cut_workspace.input sorted ascending) holds the same
	// values, so the count of rows with X_J < delta is the position of the
	// first sorted value >= delta -- the integer the row scan over cut_workspace.input
	// returned, in O(log n_res) per cut instead of O(n_res).
	const Eigen::MatrixXd& sorted_cols = trie[t].cut_workspace.sorted;
	for(int J=0; J<dim; J++){
		const double* col = sorted_cols.col(J).data();
		for(int m=0; m<M; m++){
			d.n_left(J,m) = (int)(std::lower_bound(col, col+n_res, d.cuts(J,m)) - col);
		}
	}
	d.active = true;
}

// Phase 2: the PP child pair of one distinct (node, J, left count) -- the
// very computation the one-pass move did per particle, on the node's rows
// in node order.
void SMC::pp_eval_child_pair(PPChildPair& pr) const{
	const PathNode& v = trie[pr.t];
	const Eigen::MatrixXd& Xres = v.cut_workspace.input;
	const Eigen::VectorXd& zres = v.postknot.residual;
	const Eigen::MatrixXd& Qn = v.postknot.coefficients;
	const Eigen::VectorXd bmin = v.box_min, bmax = v.box_max;
	const int n_res = zres.size();
	const int J = pr.J;
	const double delta = pr.delta;
	Eigen::VectorXi ind_left(n_res), ind_right(n_res);
	int nl=0, nr=0;
	for(int i=0; i<n_res; i++){
		if(Xres(i,J) < delta){ ind_left(nl++) = i; }
		else{ ind_right(nr++) = i; }
	}
	pr.nl = nl; pr.nr = nr;
	pr.ll[0] = 0.0; pr.ll[1] = 0.0;
	pr.st[0].clear(); pr.st[1].clear();
	// Score-only leaves on the node's block through the row lists: no
	// gathered copy of the child's rows / values / coefficient columns and no
	// stored cross block; one 1024-row panel at a time (terminal_loglik_rows).
	if(nl>0){
		Eigen::VectorXd bmax_l = bmax; bmax_l(J) = delta;
		pr.ll[0] = terminal_loglik_rows(Xres, zres, Qn, ind_left.data(), nl,
			bmin, bmax_l, pr.st[0]);
	}
	if(nr>0){
		Eigen::VectorXd bmin_r = bmin; bmin_r(J) = delta;
		pr.ll[1] = terminal_loglik_rows(Xres, zres, Qn, ind_right.data(), nr,
			bmin_r, bmax, pr.st[1]);
	}
}

// Phase 3: the particle's posteriors from the shared pairs and its stop /
// dimension (/ candidate) draws -- the tail of draw_random_cut_move (M = 1)
// or of draw_multi_cut_move (M > 1), reading the blocks from the table.
void SMC::pp_finish_move(int p, int id, std::mt19937_64& rng, const PPCutDraw& d,
	const std::vector<PPChildPair>& pairs, RandomCutMove& out){

	out.active = false;
	out.integrated_wn = false;
	out.direct_increment = false;
	if(!d.active){ return; }
	auto* nd=worker_node(p,id); if(!nd) return;
	const int32_t t = d.t;
	const double loglik_base = trie[t].loglik_base;
	const double loglik_knot = trie[t].loglik_knot;
	const int M = (int)d.cuts.cols();
	const double rho_prior = prior_stop(trie[t].level);

	if(M==1){
		Eigen::VectorXd child_loglik(dim);
		for(int J=0; J<dim; J++){
			const PPChildPair& pr = pairs[d.pair(J,0)];
			const int nl = pr.nl, nr = pr.nr;
			const double ll_l = pr.ll[0], ll_r = pr.ll[1];
			if(nl==0 && nr==0){ child_loglik(J) = 0.0; }
			else if(nl==0){ child_loglik(J) = ll_r; }
			else if(nr==0){ child_loglik(J) = ll_l; }
			else{ child_loglik(J) = ll_l + ll_r; }
		}
		double Phi_ss = spike_slab_loglik(loglik_base, loglik_knot,
			child_loglik, rho_prior, prior_lambda);
		double rho = std::exp(std::log(rho_prior) + loglik_base - Phi_ss);
		if(rho<0.0){ rho = 0.0; }
		if(rho>1.0){ rho = 1.0; }
		Eigen::VectorXd lambda(dim), lambda_log(dim);
		{
			Eigen::VectorXd tmp2 = prior_lambda.array().log() + loglik_knot + child_loglik.array();
			double loglik_slab = log_sum_exp(tmp2);
			for(int J=0; J<dim; J++){
				lambda_log(J) = tmp2(J) - loglik_slab;
				lambda(J) = std::exp(lambda_log(J));
				if(lambda(J)<0.0){ lambda(J) = 0.0; }
				if(lambda(J)>1.0){ lambda(J) = 1.0; }
			}
		}
		nd->rhoval = rho;
		std::bernoulli_distribution rbern(rho);
		out.active = true;
		out.rho = rho;
		out.S = rbern(rng) ? 1 : 0;
		if(out.S==1){ return; }
		std::discrete_distribution<int> rcat(lambda.data(), lambda.data()+dim);
		const int Ji = rcat(rng);
		const PPChildPair& pr = pairs[d.pair(Ji,0)];
		out.Ji = Ji;
		out.cut = d.cuts(Ji,0);
		out.lambda_log_Ji = lambda_log(Ji);
		out.child_n[0] = pr.nl; out.child_n[1] = pr.nr;
		out.child_ll[0] = pr.ll[0]; out.child_ll[1] = pr.ll[1];
		out.child_st[0] = pr.st[0]; out.child_st[1] = pr.st[1];
		return;
	}

	// Uniform candidates carry stratum masses; balanced candidates retain
	// the original equally weighted IID estimator.
	Eigen::MatrixXd child_loglik(dim, M);
	for(int J=0; J<dim; J++){
		for(int m=0; m<M; m++){
			const PPChildPair& pr = pairs[d.pair(J,m)];
			const int nl = pr.nl, nr = pr.nr;
			const double ll_l = pr.ll[0], ll_r = pr.ll[1];
			child_loglik(J,m) = (nl==0 && nr==0) ? 0.0 : (nl==0 ? ll_r : (nr==0 ? ll_l : ll_l + ll_r));
			if(d.log_mass.size()) child_loglik(J,m) += d.log_mass(m);
		}
	}
	Eigen::VectorXd A(dim);
	for(int J=0; J<dim; J++){
		A(J) = log_sum_exp(child_loglik.row(J).transpose());
		if(!d.log_mass.size()) A(J) -= std::log((double)M);
	}
	const double log_Phi_hat = spike_slab_loglik(loglik_base, loglik_knot, A, rho_prior, prior_lambda);
	double rho = std::exp(std::log(rho_prior) + loglik_base - log_Phi_hat);
	if(rho<0.0){ rho = 0.0; }
	if(rho>1.0){ rho = 1.0; }
	Eigen::VectorXd lambda(dim);
	{
		Eigen::VectorXd tmp2 = prior_lambda.array().log() + A.array();
		const double lse = log_sum_exp(tmp2);
		for(int J=0; J<dim; J++){
			lambda(J) = std::exp(tmp2(J) - lse);
			if(lambda(J)<0.0){ lambda(J) = 0.0; }
			if(lambda(J)>1.0){ lambda(J) = 1.0; }
		}
	}
	nd->rhoval = rho;
	out.active = true;
	out.direct_increment = true;
	out.log_weight_increment = log_Phi_hat - loglik_base;
	out.rho = rho;
	std::bernoulli_distribution rbern(rho);
	out.S = rbern(rng) ? 1 : 0;
	if(out.S==1){ return; }
	std::discrete_distribution<int> rcat(lambda.data(), lambda.data()+dim);
	const int Ji = rcat(rng);
	Eigen::VectorXd pm(M);
	{
		const double mx = child_loglik.row(Ji).maxCoeff();
		for(int m=0; m<M; m++){ pm(m) = std::exp(child_loglik(Ji,m) - mx); }
	}
	std::discrete_distribution<int> rm(pm.data(), pm.data()+M);
	const int mi = rm(rng);
	const PPChildPair& pr = pairs[d.pair(Ji,mi)];
	out.Ji = Ji;
	out.cut = d.cuts(Ji, mi);
	out.lambda_log_Ji = std::log(std::max(lambda(Ji), 1e-300));
	out.child_n[0] = pr.nl; out.child_n[1] = pr.nr;
	out.child_ll[0] = pr.ll[0]; out.child_ll[1] = pr.ll[1];
	out.child_st[0] = pr.st[0]; out.child_st[1] = pr.st[1];
}

void SMC::commit_random_cut_move(int p, int id, const RandomCutMove& out,
	double& Phi, double& w_log){

	if(!out.active){ return; }
	auto& nd=existing_node(p,id);
	const int32_t t = nd.trie_at;
	const double rho_prior = prior_stop(trie[t].level);
	const bool direct = out.integrated_wn || out.direct_increment;

 if(direct) w_log += out.log_weight_increment;
	if(out.S==1){ // stop
		nd.act = ACT_STOP;
  if(!direct)
		 w_log = w_log + std::log(rho_prior) - std::log(out.rho);
		return;
	}

	const int Ji = out.Ji;
	record_split_depth(trie[t].level);
	nd.act = ACT_SPLIT;
	nd.jdim = (int8_t)Ji;
	nd.cutval = out.cut;

	for(int side=0; side<2; ++side){
		const int cid = 2*id + side;
		const int cn = out.child_n[side];
		SparseNode child;
		child.Loglik = out.child_ll[side];
		if(cn==0){
			child.act = ACT_EMPTY;
		}else{
			// Integrated WN under a box-free knot design: the subtree likelihood
			// is a function of the row sets alone, so particles drawing the
			// same rank cell share one child (the box enters pick_knots only
			// under "boundary", which keeps cut-keyed children).
			const bool share = out.integrated_wn && design!="boundary";
			const int ct = share ?
				trie.child_cell(t, Ji, side, out.cell, out.cut_rep, cn,
					out.child_ll[side], out.child_st[side]) :
				trie.child_random(t, Ji, side, out.cut, cn,
					out.child_ll[side], out.child_st[side]);
			child.trie_at = ct;
			child.act = compare_node(trie[ct].level,cn) ? ACT_PENDING : ACT_BOTTOM;
		}
		insert_node(p,cid,child);
	}

	// ---- bottom-up likelihood recursion and the weight update ----
	nd.Loglik = trie[t].loglik_knot + (out.child_ll[0] + out.child_ll[1]);
	const double Phi_new = update_ancestor_loglik(p,id,nd.Loglik);
 if(direct) { Phi=Phi_new; return; }
	const double logBF = Phi_new - Phi;
	const double log1m_rho_post = std::log1p(-out.rho);
	const double log_proposal_ratio = std::log1p(-rho_prior) - log1m_rho_post
		+ std::log(prior_lambda(Ji)) - out.lambda_log_Ji;
	w_log = w_log + log_proposal_ratio + logBF;
	Phi = Phi_new;
}

// ==================== Nested PP particle draws ====================

/*============================== the particle draw ==============================*/

// draw_integrated_wn_cut_move on the nested proposal: the same draws from the
// same table (stop | J, cell, then the cut inside the cell), with the cell's
// Gaussian block statistics handed to the child (child_cell stores them as
// the child's arrival stat_leaf; particle_loglik_tree and the sig2 refresh
// read them).
static bool draw_left(double left, double right, std::mt19937_64& rng){
	const double neginf = -std::numeric_limits<double>::infinity();
	if(left==neginf) return false;
	if(right==neginf) return true;
	const double log_total = log_sum_exp2(left, right);
	const bool left_smaller = left<right;
	std::bernoulli_distribution rare(std::exp(std::min(left,right)-log_total));
	return rare(rng) ? left_smaller : !left_smaller;
}

void SMC::draw_nested_pp_cut_move(int p, int id, std::mt19937_64& rng,
	RandomCutMove& out){
	auto* nd=worker_node(p,id); if(!nd) return;
	const PathNode& node = trie[nd->trie_at];
	out.integrated_wn = true;
	if(!node.proposals.nested){
		int expected = -1;
		internal_error_node.compare_exchange_strong(expected, id);
		return;
	}
	const NestedCutProposal& prop = *node.proposals.nested;
	const WNCutProposal& tab = prop.table;
	out.active = true;
	out.rho = node.rho;
	nd->rhoval = out.rho;
	out.log_weight_increment = tab.log_weight_increment;
	out.S = draw_left(tab.log_stop_mass, tab.log_split_mass, rng) ? 1 : 0;
	if(out.S==1){ return; }
	size_t index = 1;
	while(index < tab.leaf_offset){
		const bool left = draw_left(tab.log_mass_tree[2*index], tab.log_mass_tree[2*index+1], rng);
		index = 2*index + (left ? 0 : 1);
	}
	const size_t cell_idx = index - tab.leaf_offset;
	const WNCutInterval& interval = tab.intervals[cell_idx];
	out.Ji = interval.coordinate;
	out.cell = interval.rank_index;
	out.cut_rep = interval.cut_upper;
	out.cut = interval.cut_lower;
	if(interval.cut_lower!=interval.cut_upper){
		const int m = tab.n_residual;
		const double a = (double)interval.rank_index/(m-1), b = (double)(interval.rank_index+1)/(m-1);
		std::uniform_real_distribution<double> uniform(0.0, 1.0);
		double fraction = uniform(rng);
		if(cut_method=="balanced"){
			const double target = fraction*beta22_interval_mass(a, b);
			double lo = 0.0, hi = 1.0;
			for(int k=0; k<52; ++k){
				const double mid = lo + 0.5*(hi-lo);
				if(beta22_interval_mass(a, a+(b-a)*mid) < target){ lo = mid; } else { hi = mid; }
			}
			fraction = lo + 0.5*(hi-lo);
		}
		out.cut = (1.0-fraction)*interval.cut_lower + fraction*interval.cut_upper;
		out.cut = std::min(interval.cut_upper, std::max(out.cut,
			std::nextafter(interval.cut_lower, interval.cut_upper)));
	}
	out.child_n[0] = interval.n_left;
	out.child_n[1] = tab.n_residual - interval.n_left;
	out.child_ll[0] = interval.left_loglik;
	out.child_ll[1] = interval.right_loglik;
	out.child_st[0] = prop.left_stat[cell_idx];
	out.child_st[1] = prop.right_stat[cell_idx];
}

// ==================== Sampler cache admission ====================

// Keep a node's level-scoped state across sig2-only reuse runs?  Under
// deterministic cuts that is the post-knot cache (postknot.row_ids, postknot.residual,
// postknot.coefficients); under integrated-cut WN it is the proposal table the particles
// draw from (the root's unconditionally, deeper nodes within the budget).
// Decides once per node and accounts the retained bytes; a node kept in an
// earlier run stays kept (its bytes are already counted) until the trie is
// cleared.
bool SMC::retain_cache(PathNode& v){
	if(!keep_shallow_caches){ return false; }
	if(v.retention != CacheRetention::none){ return true; }
	size_t b = 0;
	if(random_cuts){
		if(!wn_integrated_reuse() || !v.proposals.wn){ return false; }
		b = proposal_bytes_of(*v.proposals.wn);
		if(v.level > 0 && trie.retained_bytes.total() + b > keep_cache_budget_bytes){ return false; }
	}else{
		if(!v.postknot.ready){ return false; }
		b = cache_bytes_of(v);
		if(v.level >= keep_cache_levels && trie.retained_bytes.total() + b > keep_cache_budget_bytes){
			return false;
		}
	}
	if(random_cuts){
		trie.retained_bytes.wn_proposal += b;
		v.retention = CacheRetention::wn_proposal;
	}else{
		trie.retained_bytes.postknot += b;
		v.retention = CacheRetention::postknot;
	}
	return true;
}

// ==================== Frontier processing and resampling ====================

void SMC::expand_level(int level){
	// Re-derive the frontier from the LIVE particles (resampling may have
	// killed states whose children were marked during sample_level), then stamp
	// the ancestor paths so descend() only walks what is needed.
	for(size_t t=0; t<trie.size(); ++t){ trie[t].need_expand = false; }
	const int lo = 1 << level, hi = 2*lo;
	for(int p=0; p<N; ++p){
		for_each_node(p, lo, hi, [&](int id){
			if(act_value(p, id)==ACT_PENDING){
				const int32_t t = trie_at_value(p, id);
				if(t>=0 && !trie[t].expanded){ trie[t].need_expand = true; }
			}
		});
	}
	trie.path_stamp.resize(trie.size(), 0);
	trie.stamp++;
	int n_frontier = 0;
	for(int t=0; t<(int)trie.size(); ++t){
		if(trie[t].need_expand && !trie[t].expanded){
			n_frontier++;
			int a = t;
			while(a>=0 && trie.path_stamp[a]!=trie.stamp){
				trie.path_stamp[a] = trie.stamp;
				a = trie[a].parent;
			}
		}
	}
	if(n_frontier==0){ return; }
	if(level<diag.distinct_per_level.size()){
		diag.distinct_per_level(level) = n_frontier;
	}
	diag.n_expanded += n_frontier;

	// A frontier node whose parent still holds its cached post-knot state
	// (expanded in the previous level and carried within the level budget) is
	// expanded straight from that state -- the same (idx, resid, Q) descend()
	// would hand it after re-deriving the whole root-to-parent path, obtained
	// by pure selection instead.  The others (their parent's state was not
	// carried: over budget, or never stored) go through descend() below, along
	// their own stamped paths only.  Per node, not all-or-nothing: one
	// uncached parent no longer sends the whole frontier through the root.
	std::vector<int> uncached;
	{
		std::vector<int> frontier;
		frontier.reserve(n_frontier);
		for(int t=0; t<(int)trie.size(); ++t){
			if(!(trie[t].need_expand && !trie[t].expanded)){ continue; }
			const int par = trie[t].parent;
			if(par<0 || !trie[par].postknot.ready){ uncached.push_back(t); }
			else{ frontier.push_back(t); }
		}
		if(!frontier.empty()){
			// Release a random-cut parent's post-knot state after its last
			// frontier child finishes reading. Counts/release share the cache-list
			// critical section; partition_child copies rows before expand().
			// Unvisited parents wait for end-of-level cleanup. Deterministic and
			// nested designs use their own release paths. Retained WN proposals
			// are independent of these numeric matrices.
			const bool early_release = random_cuts && !nested();
			std::vector<int> slot_of(frontier.size(), -1);   // frontier position -> parent slot
			std::vector<int> par_of_slot;                     // parent slot -> trie index
			std::vector<int> left_in_slot;                    // frontier children still to expand
			if(early_release){
				std::unordered_map<int,int> slot;
				for(size_t u=0; u<frontier.size(); ++u){
					const int par = trie[frontier[u]].parent;
					auto it = slot.find(par);
					if(it==slot.end()){
						it = slot.emplace(par, (int)par_of_slot.size()).first;
						par_of_slot.push_back(par);
						left_in_slot.push_back(0);
					}
					slot_of[u] = it->second;
					left_in_slot[(size_t)it->second]++;
				}
			}
			// The per-child work, serial or as one task per frontier node: it
			// writes only its own trie entry (and, under the critical, the
			// cache-list counters), so the numbers are identical for every ncores.
			auto expand_from_parent = [&](size_t u){
				PathNode& c = trie[frontier[u]];
				const PathNode& par = trie[c.parent];
				std::vector<int> cidx;
				Eigen::VectorXd cres;
				Eigen::MatrixXd cQ;
				partition_child(par.postknot.row_ids, par.postknot.residual, par.postknot.coefficients,
					c.j_from_parent, c.side_from_parent,
					random_cuts ? c.cut_from_parent : par.cuts(c.j_from_parent),
					cidx, cres, cQ);
				expand(trie, c, cidx, cres, cQ);
				if(!early_release){ return; }
				const size_t s = (size_t)slot_of[u];
				bool last = false;
				#ifdef _OPENMP
				#pragma omp critical(restree_cache_list)
				#endif
				{
					left_in_slot[s]--;
					last = (left_in_slot[s]==0);
					if(last){
						const size_t b = cache_bytes_of(par);
						trie.level_cache.previous.bytes -= std::min(b, trie.level_cache.previous.bytes);
					}
				}
				if(last){ release_postknot(trie[c.parent]); }
			};
			#ifdef _OPENMP
			if(ncores>1){
				// One team, one task per frontier node: the expansions write to
				// disjoint trie entries and the trie is not resized here, so the
				// tasks are race-free and the results are identical for every
				// ncores.  The closure is a local of this function (shared in the
				// parallel region, hence in the task; it holds a reference to
				// this, so the store it reaches is the real trie, never a copy);
				// the loop index is firstprivate.
				#pragma omp parallel num_threads(ncores)
				{
					#pragma omp single nowait
					{
						for(size_t u=0; u<frontier.size(); ++u){
							#pragma omp task default(shared) firstprivate(u)
							worker_error.run("expand",[&]{ expand_from_parent(u); });
						}
					}
				}
				worker_error.rethrow();
			}else
			#endif
			{
				for(size_t u=0; u<frontier.size(); ++u){ worker_error.run("expand",[&]{ expand_from_parent(u); }); }
				worker_error.rethrow();
			}
		}
	}
	if(uncached.empty()){ return; }
	// Re-stamp the ancestor paths of the uncached frontier nodes alone, so
	// descend() walks only those (the nodes just expanded above are
	// `expanded` and would be skipped anyway, but their paths would still be
	// partitioned).
	trie.stamp++;
	for(size_t u=0; u<uncached.size(); ++u){
		int a = uncached[u];
		while(a>=0 && trie.path_stamp[a]!=trie.stamp){
			trie.path_stamp[a] = trie.stamp;
			a = trie[a].parent;
		}
	}

	std::vector<int> root_idx((size_t)X.rows());
	std::iota(root_idx.begin(), root_idx.end(), 0);
	Eigen::VectorXd root_resid = y;
	Eigen::MatrixXd root_Q(0, X.rows());
	#ifdef _OPENMP
	if(ncores>1){
		// One team; the DFS spawns a task per sibling subtree.  The implicit
		// barrier at the end of the region completes all outstanding tasks.
		#pragma omp parallel num_threads(ncores)
		{
			#pragma omp single nowait
			{
				worker_error.run("descend-root",[&]{
				descend(trie, 0, std::move(root_idx), std::move(root_resid), std::move(root_Q));
				});
			}
		}
		worker_error.rethrow();
		return;
	}
	#endif
	descend(trie, 0, std::move(root_idx), std::move(root_resid), std::move(root_Q));
}

/************ sample_level: particle draws, weights, resampling ************/

// REALIZED partial-tree likelihood only, never the stop/split mixture.
void SMC::resample_particles(const Eigen::VectorXd& weight){

	Eigen::VectorXd w_star = weight.array().pow(alpha);
	Eigen::VectorXi I(N);
	if(resampling_method=="multinomial"){
		std::discrete_distribution<int> dist(w_star.data(), w_star.data()+N);
		for(int i=0; i<N; i++){ I(i) = dist(master_rng); }
	}else{ // stratified: one uniform per stratum, inverse CDF
		Eigen::VectorXd lb = Eigen::VectorXd::LinSpaced(N, 1.0, N) / ((double)N);
		Eigen::VectorXd u(N);
		std::uniform_real_distribution<double> unif(0.0, 1.0);
		u(0) = unif(master_rng) * lb(0);
		for(int i=0; i<N-1; i++){
			u(i+1) = lb(i) + unif(master_rng) * (lb(i+1)-lb(i));
		}
		Eigen::VectorXd wnorm = w_star / w_star.sum();
		Eigen::VectorXd cum(N);
		double s = 0.0;
		for(int i=0; i<N; i++){ s += wnorm(i); cum(i) = s; }
		// Stratified thresholds are ordered, so the CDF cursor is retained
		// across draws and advances at most N-1 times in total (linear work);
		// the strict comparison fixes the treatment of CDF-boundary ties.
		int count = 0;
		for(int i=0; i<N; i++){
			while(count<N-1 && u(i)>cum(count)){ count++; }
			I(i) = count;
		}
	}

	// Offspring i takes ancestor I(i)'s node map.  The FIRST offspring of an
	// ancestor takes it by move (no copy, no allocation); every further
	// offspring copies the moved map.  The maps are equal whichever
	// offspring holds the original, and nothing reads them through the
	// ancestor's index again, so the resampled system is the same as with
	// N copies -- at the cost of the duplicates alone.  cutval (and, under
	// quantile cuts, rhoval) are per-state quantities: cutval feeds the
	// distinct-tree grouping and the native prediction edges, so both
	// follow the resampled states with the rest of the record.
	std::vector<ParticleNodes> sparse_t(N);
	Eigen::VectorXd loglik_temp(N), w_log_temp(N);
	std::vector<int> first_use(N, -1);   // offspring that took ancestor a by move
	for(int i=0; i<N; i++){
		const int a = I(i);
		if(first_use[a] < 0){
			first_use[a] = i;
			sparse_t[i] = std::move(sparse_particles[a]);
		}else{
			sparse_t[i] = sparse_t[first_use[a]];
		}
		loglik_temp(i) = particle_loglik(a);
		w_log_temp(i) = (1.0-alpha) * std::log(weight(a));
	}
	sparse_particles.swap(sparse_t);
	// Proper-weighting mass correction: exp(logZ) stays unbiased under the
	// tempered scheme; identically zero at alpha = 1.
	logZ += std::log(w_star.sum()) + log_sum_exp(w_log_temp) - std::log((double)N);
	particle_loglik = loglik_temp;
	particle_log_weight = w_log_temp;
	n_resample++;
}

std::vector<int> SMC::pending_ids(int level) const {
	const int lo=1<<level, hi=2*lo;
	return merge_pending_particles(sparse_particles,lo,hi);
}

bool SMC::has_pending(int level) const {
	const int lo=1<<level, hi=2*lo;
	return has_pending_particles(sparse_particles,lo,hi);
}

void SMC::record_split_depth(int level) {
	if(open_depth() && level>=max_tree_depth)
		throw AdaptiveDepthError("WN selected a split at level " + std::to_string(level) +
			", but the supported tree depth is " + std::to_string(max_tree_depth) +
			". No forced terminal leaf was accepted; increase r or change the model settings.");
	tree_depth=std::max(tree_depth,level+1);
	if(open_depth()) n_internal=1<<tree_depth;   // PP: n_internal = 2^depth, fixed
}

size_t SMC::stored_particle_nodes() const {
	size_t out=0;
	for(const auto& p : sparse_particles) out+=p.size();
	return out;
}

void SMC::append_ess_run(int end, double value){
	if(!record_ess_history || end<=ess_history_length()) return;
	if(!ess_runs.empty() && ess_runs.back().value==value) ess_runs.back().end=end;
	else ess_runs.push_back(ESSRun{end,value});
}

Eigen::VectorXd SMC::collect_ess_history() const{
	Eigen::VectorXd out(ess_history_length());
	int begin=0;
	for(const auto& run:ess_runs){
		out.segment(begin,run.end-begin).setConstant(run.value);
		begin=run.end;
	}
	return out;
}

void SMC::sample_level(int level){
	const int lo = 1 << level, hi = 2*lo;
	// Process the union of pending node IDs. Inactive positions consume no
	// draws and leave weights unchanged; PP ESS diagnostics still include
	// these gaps and level-end bookkeeping.
	const std::vector<int> positions=pending_ids(level);
	Eigen::VectorXd Phi = particle_loglik;
	Eigen::VectorXd w_log_curr = particle_log_weight;
	const double logZ_level_start = logZ;
	// A PP gap before the first processed id carries the arrival weights.
	// Subsequent gaps are recorded once, with the preceding position's ESS.
	// Empty levels still do the same resampling/bookkeeping below; histories
	// never control a weight update. Value-only engines collect no history.
	if(record_ess_history && !open_depth()){
		const int end=positions.empty() ? hi : positions.front();
		const Eigen::VectorXd w0 = (particle_log_weight.array() - log_sum_exp(particle_log_weight)).exp();
		append_ess_run(end,1.0 / w0.squaredNorm());
	}

	std::vector<RandomCutMove> qout;
	if(random_cuts){ qout.resize(N); }

	const bool pp_shared = random_cuts && baseline=="PP" && !nested();
	std::vector<PPCutDraw> pdraw;
	if(pp_shared){ pdraw.resize(N); }
	for(size_t pos=0; pos<positions.size(); ++pos){
		const int t=positions[pos];
		if(pp_shared){
			// PP random cuts, candidate blocks shared across particles (see
			// PPCutDraw in ResTree.h).  Phase 1: cut draws, per particle.
			parallel_for(worker_error,"cut-draw",N,ncores,
				[&](int i){ pp_draw_cuts(i, t, par_rng[i], pdraw[i]); });
			check_particle_worker_error(internal_error_node);
			// Distinct (node, J, left count) -- plus the cut itself under the
			// boundary design, whose blocks depend on the box.
			std::vector<PPChildPair> pairs;
			{
				const bool key_delta = (design=="boundary");
				std::unordered_map<std::string, int> index;
				std::string key;
				for(int i=0; i<N; ++i){
					PPCutDraw& d = pdraw[i];
					if(!d.active){ continue; }
					const int M = (int)d.cuts.cols();
					d.pair.resize(dim, M);
					for(int J=0; J<dim; ++J){
						for(int m=0; m<M; ++m){
							key.assign((const char*)&d.t, sizeof(d.t));
							key.append((const char*)&J, sizeof(J));
							const int nl = d.n_left(J,m);
							key.append((const char*)&nl, sizeof(nl));
							if(key_delta){ const double dl = d.cuts(J,m); key.append((const char*)&dl, sizeof(dl)); }
							auto it = index.find(key);
							if(it==index.end()){
								PPChildPair pr;
								pr.t = d.t; pr.J = J; pr.nl = nl; pr.delta = d.cuts(J,m);
								it = index.emplace(key, (int)pairs.size()).first;
								pairs.push_back(pr);
							}
							d.pair(J,m) = it->second;
						}
					}
				}
			}
			// Evaluate distinct PP child pairs independently. Each worker gathers
			// rows/coefficients and factors its own kernels. Bound concurrency by
			// the widest estimated workspace, allowing at least one worker;
			// this budget is not a whole-process memory cap.
			// The per-pair transient of the streamed leaves (terminal_loglik_rows):
			// the row lists and the per-row terms, 8 (d + 5) bytes per row, one
			// panel of rows, values and coefficient columns for the child being
			// evaluated, and the r_k x r_k knot blocks.
			int pair_threads = ncores;
			if(ncores>1 && !pairs.empty()){
				size_t widest = 0;
				for(size_t k=0; k<pairs.size(); ++k){
					const PathNode& pv = trie[pairs[k].t];
					const size_t q2 = (size_t)pv.postknot.coefficients.rows();
					const size_t b = (size_t)pv.postknot.residual.size() * sizeof(double) * (size_t)(dim + 5)
						+ (size_t)restree_solve_panel * (q2 + (size_t)r_knot + (size_t)dim + 1) * 2 * sizeof(double)
						+ 3 * (size_t)r_knot * (size_t)r_knot * sizeof(double);
					if(b>widest){ widest = b; }
				}
				if(widest>0){
					// Bounded by the transient-workspace budget (task_budget_bytes),
					// not by the level cache budget: a cache limit bounds retained
					// state, never the concurrent temporaries of the workers.
					const size_t fits = task_budget_bytes / widest;
					const int cap = fits<1 ? 1 :
						(fits>(size_t)ncores ? ncores : (int)fits);
					if(cap<pair_threads){ pair_threads = cap; }
				}
			}
			parallel_for(worker_error,"candidate",(int)pairs.size(),pair_threads,
				[&](int k){ pp_eval_child_pair(pairs[k]); });
			// Phase 3: posteriors and the stop / dimension (/ candidate) draws.
			parallel_for(worker_error,"cut-finish",N,ncores,
				[&](int i){ pp_finish_move(i, t, par_rng[i], pdraw[i], pairs, qout[i]); });
			check_particle_worker_error(internal_error_node);
			for(int i=0; i<N; ++i){
				double phi_i = particle_loglik(i), w_i = particle_log_weight(i);
				commit_random_cut_move(i, t, qout[i], phi_i, w_i);
				Phi(i) = phi_i;
				w_log_curr(i) = w_i;
			}
		}else if(random_cuts){
			// Draw pass, thread-parallel: each particle touches only its own
			// RNG stream and outcome slot, so results match every ncores.
			parallel_for(worker_error,"cut-move",N,ncores,[&](int i){
				draw_random_cut_move(i, t, par_rng[i], qout[i]);
			});
			// Serial context again: raise any error a worker deferred.
			check_particle_worker_error(internal_error_node);
			// Commit pass, serial in particle order (trie children are created
			// in the same order as the single-threaded engine).
			for(int i=0; i<N; ++i){
				double phi_i = particle_loglik(i), w_i = particle_log_weight(i);
				commit_random_cut_move(i, t, qout[i], phi_i, w_i);
				Phi(i) = phi_i;
				w_log_curr(i) = w_i;
			}
		}else{
			for(int i=0; i<N; ++i){
				double phi_i = particle_loglik(i), w_i = particle_log_weight(i);
				transition(i, t, par_rng[i], phi_i, w_i);
				Phi(i) = phi_i;
				w_log_curr(i) = w_i;
			}
		}
		logZ += log_sum_exp(w_log_curr) - log_sum_exp(particle_log_weight);
		particle_loglik = Phi;
		particle_log_weight = w_log_curr;
		particle_weight = (w_log_curr.array() - log_sum_exp(w_log_curr)).exp();
		const double ess_now=1.0 / particle_weight.squaredNorm();
		if(record_ess_history){
			const int end=open_depth() ? ess_history_length()+1 :
				(pos+1<positions.size() ? positions[pos+1] : hi);
			append_ess_run(end,ess_now);
		}
		// Resample at LEVEL ENDS only, never between the sibling nodes of one
		// level: mid-level weights are transient and not comparable across
		// particles (a particle whose gain sits in an unprocessed sibling is
		// temporarily under-weighted).  The last level never resamples --
		// the final mixture weights feed prediction.
		++processed_steps;
		const bool level_done = pos+1==positions.size();
		if(level_done){
			// level-end diagnostics (bookkeeping only; see ResTree.h)
			level_ess(level) = ess_now;
			level_ess_after(level) = ess_now;
			level_max_weight(level) = particle_weight.maxCoeff();
			level_resampled(level) = 0;
			level_last_node(level) = open_depth() ? t : hi-1;
			level_last_step(level) = processed_steps;
		}
		const bool more_levels = level_done && (open_depth() ? has_pending(level+1) : hi-1<n_internal-1);
		if(level_done && ess_now<minESS && more_levels){
			resample_particles(particle_weight);
			w_log_curr = particle_log_weight;
			Phi = particle_loglik;
			level_resampled(level) = 1;
			{
				// ESS of the carried (tempered) weights after the resample:
				// N at alpha = 1, less as alpha decreases
				const double lse = log_sum_exp(particle_log_weight);
				const double s2 = (2.0*(particle_log_weight.array()-lse)).exp().sum();
				level_ess_after(level) = 1.0/s2;
			}
		}
		if(record_ess_history){
			step_diag.push_back(StepDiagnostic{processed_steps,t,level,
				level_done ? level_resampled(level) : 0,ess_now,
				level_done ? level_ess_after(level) : ess_now,logZ});
		}
		if(level_done){ level_logZ_inc(level) = logZ - logZ_level_start; }
	}
	if(positions.empty() && !open_depth()){
		// A capped level at which no particle is pending (every particle
		// stopped or emptied above it).  The dense loop still ran its no-op
		// passes here and did the level-end bookkeeping, including a possible
		// resample of the carried weights; reproduce exactly that.
		particle_weight = (particle_log_weight.array() - log_sum_exp(particle_log_weight)).exp();
		const double ess_now = 1.0 / particle_weight.squaredNorm();
		level_ess(level) = ess_now;
		level_ess_after(level) = ess_now;
		level_max_weight(level) = particle_weight.maxCoeff();
		level_resampled(level) = 0;
		level_last_node(level) = hi-1;
		level_last_step(level) = processed_steps;
		if(ess_now<minESS && hi-1<n_internal-1){
			resample_particles(particle_weight);
			level_resampled(level) = 1;
			const double lse = log_sum_exp(particle_log_weight);
			const double s2 = (2.0*(particle_log_weight.array()-lse)).exp().sum();
			level_ess_after(level) = 1.0/s2;
		}
		level_logZ_inc(level) = 0.0;
	}
}

// ==================== Run lifecycle and retained tree summaries ====================

void SMC::run(bool reuse_trie, bool collect_ess, bool collect_tree_scores){
	worker_error.reset();
	// Everything a run reads is (re)initialised here, so a persistent engine
	// (smc_engine_new_cpp) reruns at a new theta / seed exactly as a fresh
	// one would: the particle arrays, the trie and its level caches, the RNG
	// streams, the diagnostics, the evidence accumulators -- and the
	// deferred worker error slot, which nothing else clears.  Only the data,
	// the configuration and the path cache (theta-independent) persist.
	//
	// reuse_trie (sig2-only re-evaluation, see refresh_sig2_values): the
	// trie of the previous completed run is kept with its block statistics,
	// every sig2-derived value is refreshed from them, and the sampler runs
	// again; a path the particles did not visit before is expanded exactly
	// as in a fresh run, from its parent's retained post-knot cache when the
	// engine kept one (retain_cache) and otherwise through descend(), which
	// re-derives the ancestors' post-knot state.  Both routes hand expand()
	// the same numbers.  Valid only when the caller checked sig2_only_change().
	internal_error_node.store(-1);
	reuse_trie = reuse_trie && trie_reuse_ok && !trie.nodes.empty() && trie[0].expanded;
	trie_reuse_ok = false;   // set again only by a completed run
	dim = X.cols();
	n_internal = 1 << depth;
	tree_depth=depth;
	const int n = (int)y.size();
	prior_lambda = (1.0/(double)dim) * Eigen::VectorXd::Ones(dim);
	random_cuts = (cut_method=="uniform" || cut_method=="balanced");

	// The PP and WN node maps start empty and
	// insert only reached ids (root first, then the children of committed
	// splits).
	sparse_particles.assign(N, ParticleNodes());
	trie.level_cache.current.paths.clear();
	trie.level_cache.previous.paths.clear();
	trie.level_cache.current.bytes = 0;
	trie.level_cache.previous.bytes = 0;

	// RNG streams: one master stream (resampling) and one stream per particle.
	master_rng.seed(seed);
	par_rng.resize(N);
	for(int i=0; i<N; i++){ par_rng[i].seed(master_rng()); }

	// Trie root (tree node id 1).
	if(reuse_trie){
		// Keep every expanded node; nothing is pending, no level cache is live.
		// Integrated-cut WN: a node whose proposal table was not retained is
		// marked unexpanded, so a particle reaching it re-expands it through
		// descend() (the same rows, knots and sweeps give the same table).
		const bool wn_reuse = wn_integrated_reuse();
		for(size_t t=0; t<trie.size(); ++t){
			PathNode& v = trie[t];
			v.need_expand = false;
			if(!retain_cache(v)){
				release_cache(v);
				if(wn_reuse && v.expanded && compare_node(v.level,v.n_arrival) &&
				   prior_stop(v.level) < 1.0){
					v.expanded = false;
				}
			}
		}
		trie.path_stamp.assign(trie.size(), 0);
		trie.stamp = 0;
		refresh_sig2_values();
		diag_reused_trie = true;
	}else{
		trie.clear();
		trie.reserve(1024);
		PathNode root;
		root.level = 0; root.tree_id = 1; root.parent = -1;
		root.n_arrival = n;
		root.box_min = Eigen::VectorXd::Zero(dim);
		root.box_max = Eigen::VectorXd::Ones(dim);
		root.need_expand = true;
		trie.push_back(std::move(root));
		trie.path_stamp.assign(1, 0);
		trie.stamp = 0;
		diag_reused_trie = false;
	}

	diag = EngineDiagnostics();
	const int initial_levels=open_depth() ? std::max(1,depth) : depth;
	diag.distinct_per_level = Eigen::VectorXi::Zero(initial_levels);

	particle_weight = Eigen::VectorXd::Constant(N, 1.0/N);
	// Reset history on every run, including persistent sigma^2 reuse. PP
	// preserves its per-id export without allocating 2^depth entries here.
	// WN preserves one exported entry per processed position. Neither
	// history is collected by likelihood-only callers (EB/PMMH included).
	std::vector<ESSRun>().swap(ess_runs);
	record_ess_history=collect_ess;
	std::vector<StepDiagnostic>().swap(step_diag);
	processed_steps=0;
	append_ess_run(1,(double)N);
	n_resample = 0;
	level_ess = Eigen::VectorXd::Constant(initial_levels, (double)N);
	level_ess_after = Eigen::VectorXd::Constant(initial_levels, (double)N);
	level_max_weight = Eigen::VectorXd::Constant(initial_levels, 1.0/(double)N);
	level_logZ_inc = Eigen::VectorXd::Zero(initial_levels);
	level_resampled = Eigen::VectorXi::Zero(initial_levels);
	level_last_node = Eigen::VectorXi::Zero(initial_levels);
	level_last_step = Eigen::VectorXi::Zero(initial_levels);

	// Level 0: expand the root directly (Phi0 = the root-leaf arrival value).
	auto ta0 = std::chrono::steady_clock::now();
	if(!reuse_trie){
		std::vector<int> root_idx((size_t)n);
		std::iota(root_idx.begin(), root_idx.end(), 0);
		Eigen::VectorXd root_resid = y;
		Eigen::MatrixXd root_Q(0, n);
		expand(trie, trie[0], root_idx, root_resid, root_Q);
		if(compare_node(0,n)){ diag.distinct_per_level(0) = 1; }
		diag.n_expanded += 1;
	}
	diag.expand_sec += std::chrono::duration<double>(
		std::chrono::steady_clock::now()-ta0).count();
	if(covariance_error_occurred()){
		logZ = -std::numeric_limits<double>::infinity();
		return;
	}

	Phi0 = trie[0].loglik_base;
	logZ = Phi0;
	if(record_ess_history) step_diag.push_back(StepDiagnostic{0,0,-1,0,(double)N,(double)N,logZ});
	particle_loglik = Phi0 * Eigen::VectorXd::Ones(N);
	particle_log_weight = -std::log((double)N) * Eigen::VectorXd::Ones(N);
	for(int i=0; i<N; i++){
		SparseNode root;
		if(n==0){ root.act=ACT_EMPTY; }
		else { root.act=compare_node(0,n) ? ACT_PENDING : ACT_STOP; root.trie_at=0; }
		root.Loglik=Phi0;
		insert_node(i,1,root);
	}

	int processed_levels=0;
	for(int level=0; level<(open_depth() ? max_tree_depth+1 : depth); ++level){
		if(open_depth() && !has_pending(level)) break;
		if(level>=level_ess.size()) {
			level_ess.conservativeResize(level+1); level_ess(level)=N;
			level_ess_after.conservativeResize(level+1); level_ess_after(level)=N;
			level_max_weight.conservativeResize(level+1); level_max_weight(level)=1.0/N;
			level_logZ_inc.conservativeResize(level+1); level_logZ_inc(level)=0;
			level_resampled.conservativeResize(level+1); level_resampled(level)=0;
			level_last_node.conservativeResize(level+1); level_last_node(level)=0;
			level_last_step.conservativeResize(level+1); level_last_step(level)=0;
			diag.distinct_per_level.conservativeResize(level+1); diag.distinct_per_level(level)=0;
		}
		processed_levels=level+1;
		// The ONLY interrupt point: once per level, serial context, never
		// inside an OpenMP region.  The unwind runs every destructor and
		// surfaces as a regular R interrupt (caught by the PMMH loop).
		Rcpp::checkUserInterrupt();
		// Previous-level post-knot bytes before this level's expand_level, which
		// (random cuts) releases a parent as soon as its frontier children are
		// expanded and takes its bytes off level_cache.previous.bytes.
		const size_t prev_bytes_before = trie.level_cache.previous.bytes;
		if(level>0){
			auto s0 = std::chrono::steady_clock::now();
			expand_level(level);
			diag.expand_sec += std::chrono::duration<double>(
				std::chrono::steady_clock::now()-s0).count();
			if(covariance_error_occurred()){
				logZ = -std::numeric_limits<double>::infinity();
				return;
			}
		}
		auto s1 = std::chrono::steady_clock::now();
		sample_level(level);
		// Nested proposals serve this level's draws only. Their lifetime must
		// not depend on admission to the separate post-knot cache registry.
		for(int at:trie.nested_proposal_paths) trie[at].proposals.nested.reset();
		trie.nested_proposal_paths.clear();
		if(covariance_error_occurred()){ logZ=-std::numeric_limits<double>::infinity(); return; }
		diag.sample_sec += std::chrono::duration<double>(
			std::chrono::steady_clock::now()-s1).count();
		// Upper bound of the live post-knot bytes during this level: the whole
		// previous level (as it stood before expand_level's early releases)
		// plus the whole current level.  The true within-level peak lies
		// between max(prev_bytes_before, prev_remaining + this) and this bound;
		// under deterministic cuts nothing is released early and the bound is
		// exact (prev_bytes_before == level_cache.previous.bytes here).
		diag.cache_peak_bytes = std::max(diag.cache_peak_bytes,
			prev_bytes_before + trie.level_cache.current.bytes);
		if(random_cuts && !nested()){
			// This level's draws consumed the draw-time state (PP candidate
			// workspace, WN proposal -- kept when a persistent engine retains
			// it).  The previous level's post-knot state was consumed by this
			// level's expand_level, which already released every parent of a
			// frontier node; whatever is left of it is released here
			// (release_postknot is idempotent -- an empty node stays empty).
			for(size_t k=0; k<trie.level_cache.previous.paths.size(); ++k){
				release_postknot(trie[trie.level_cache.previous.paths[k]]);
			}
			trie.level_cache.previous.paths.clear();
			for(size_t k=0; k<trie.level_cache.current.paths.size(); ++k){
				PathNode& c = trie[trie.level_cache.current.paths[k]];
				release_draw_state(c);
				if(!retain_cache(c)){ c.proposals.wn.reset(); }
			}
			// What this level's post-knot state is still for: the next
			// expand_level, which reads it for the parents of the LIVE
			// particles' pending children -- known now, after this level's
			// resample.  Everything else (nodes whose particles all stopped or
			// died in the resample) is released here rather than at the end of
			// the next level, so it is not resident through the next level's
			// expansion and sampling.  The carried set is then admitted, in
			// trie order (deterministic across ncores), within
			// cache_budget_bytes: a node beyond the budget is released too and
			// its children re-derive their input through descend() -- the same
			// numbers, more work, never a different draw.  The state the
			// level's OWN draws needed was never subject to this budget: it is
			// required, not optional.
			trie.path_stamp.resize(trie.size(), 0);
			trie.stamp++;
			{
				const int clo = 2*(1 << level), chi = 2*clo;
				for(int p=0; p<N; ++p){
					for_each_node(p, clo, chi, [&](int id){
						const auto* nd = find_node(p, id);
						if(nd && nd->act==ACT_PENDING && nd->trie_at>=0){
							const int par = trie[nd->trie_at].parent;
							if(par>=0){ trie.path_stamp[par] = trie.stamp; }
						}
					});
				}
			}
			std::vector<int> carried;
			carried.reserve(trie.level_cache.current.paths.size());
			for(size_t k=0; k<trie.level_cache.current.paths.size(); ++k){
				const int t = trie.level_cache.current.paths[k];
				if(trie.path_stamp[t]==trie.stamp && trie[t].postknot.ready){ carried.push_back(t); }
				else{ release_postknot(trie[t]); }
			}
			std::sort(carried.begin(), carried.end());
			size_t carried_bytes = 0;
			trie.level_cache.previous.paths.clear();
			for(size_t k=0; k<carried.size(); ++k){
				PathNode& c = trie[carried[k]];
				const size_t b = cache_bytes_of(c);
				if(carried_bytes + b <= trie.cache_budget_bytes){
					carried_bytes += b;
					trie.level_cache.previous.paths.push_back(carried[k]);
				}else{
					release_postknot(c);
				}
			}
			trie.level_cache.current.paths.clear();
			trie.level_cache.previous.bytes = carried_bytes;
			trie.level_cache.current.bytes = 0;
		}else{
			// Deterministic cuts: the previous level's cached post-knot states
			// were consumed by this level's expand_level; this level's states
			// feed the next one.  At most two levels of caches are live.
			for(size_t k=0; k<trie.level_cache.previous.paths.size(); ++k){
				PathNode& c = trie[trie.level_cache.previous.paths[k]];
				if(!retain_cache(c)){ release_cache(c); }
			}
			trie.level_cache.previous.paths.clear();
			trie.level_cache.previous.paths.swap(trie.level_cache.current.paths);
			trie.level_cache.previous.bytes = trie.level_cache.current.bytes;
			trie.level_cache.current.bytes = 0;
		}
	}
	if(open_depth()) {
		// tree_depth grew monotonically with every split committed during the
		// run, including splits of particles that resampling later removed.
		// Recompute it from the SURVIVING particles so that structure$depth,
		// n_internal and the matrix exports describe the trees actually kept
		// (never below the initial depth).
		int deepest = 0;
		for(const auto& particle : sparse_particles)
			for(const auto& entry : particle)
				if(entry.second.act==ACT_SPLIT){
					int level = 0; for(int a=entry.first; a>1; a>>=1) ++level;
					deepest = std::max(deepest, level+1);
				}
		tree_depth = std::max(depth, deepest);
		n_internal = 1<<tree_depth;
		level_ess.conservativeResize(processed_levels);
		level_ess_after.conservativeResize(processed_levels);
		level_max_weight.conservativeResize(processed_levels);
		level_logZ_inc.conservativeResize(processed_levels);
		level_resampled.conservativeResize(processed_levels);
		level_last_node.conservativeResize(processed_levels);
		level_last_step.conservativeResize(processed_levels);
		diag.distinct_per_level.conservativeResize(processed_levels);
	}
	// Nothing consumes the caches after the last level; the retained engine
	// state (prediction) must not carry them.  Random cuts keep only a
	// persistent engine's retained WN proposals (retain_cache).
	const bool random_plain = random_cuts && !nested();
	for(size_t k=0; k<trie.level_cache.previous.paths.size(); ++k){
		PathNode& c = trie[trie.level_cache.previous.paths[k]];
		if(random_plain){ release_postknot(c); }
		else if(!retain_cache(c)){ release_cache(c); }
	}
	trie.level_cache.previous.paths.clear();
	for(size_t k=0; k<trie.level_cache.current.paths.size(); ++k){
		PathNode& c = trie[trie.level_cache.current.paths[k]];
		if(random_plain){
			release_draw_state(c);
			if(!retain_cache(c)){ c.proposals.wn.reset(); }
			release_postknot(c);
		}else if(!retain_cache(c)){ release_cache(c); }
	}
	trie.level_cache.current.paths.clear();
	trie.level_cache.current.bytes = 0;
	trie.level_cache.previous.bytes = 0;

	// Realized-tree log-likelihood per particle, re-derived from the per-node
	// Gaussian statistics the trie holds (terminals keep their arrival
	// stats, splits their knot blocks) in the fixed-structure evaluator's
	// aggregation form: ONE global (logdet, quad, n) triple summed in the
	// DFS order of loglik_subtree()/combine() (node, left subtree, right
	// subtree) and the Gaussian formula applied once at the top, plus the
	// WhiteNoise leaf marginals.  The per-node stats are bit-identical to
	// loglik_subtree()'s, so this value is identical() to
	// restree_loglik(tree = <this structure>); it agrees with the sampler's
	// incremental particle_loglik (a per-node sum) to round-off.
	particle_loglik_tree.resize(collect_tree_scores ? N : 0);
	for(int p=0; collect_tree_scores && p<N; ++p){
		struct Agg { double quad=0.0, logdet=0.0, wn=0.0; int n=0; };
		std::function<Agg(int)> walk = [&](int id)->Agg{
			Agg out;
			if(id>=2*n_internal){ return out; }
			const int8_t a = act_value(p, id);
			if(a==ACT_ABSENT || a==ACT_EMPTY){ return out; }
			if(a==ACT_SPLIT){
				const PathNode& v = trie[trie_at_value(p, id)];
				out.quad = v.stat_knot.qd; out.logdet = v.stat_knot.ld; out.n = v.stat_knot.n;
				const Agg l = walk(2*id);
				out.quad += l.quad; out.logdet += l.logdet; out.wn += l.wn; out.n += l.n;
				const Agg r = walk(2*id+1);
				out.quad += r.quad; out.logdet += r.logdet; out.wn += r.wn; out.n += r.n;
				return out;
			}
			// Terminal (stop / bottom): the Gaussian block stats if
			// the leaf model holds them (PP, dense), else the WN marginal.  A
			// terminal with a trie node (every stop; with random cuts every
			// non-empty child, child_random) carries its arrival stats in
			// stat_leaf; a trie-less terminal (deterministic cuts: bottom
			// children) reads its parent's candidate-child stats, the
			// values sample_level copied into Loglik.
			BlockStats st;
			const int32_t tnode = trie_at_value(p, id);
			if(tnode>=0){
				st = trie[tnode].stat_leaf;
			}else{
				const int par = id/2;
				const PathNode& pv = trie[trie_at_value(p, par)];
				const size_t slot = (size_t)(2*jdim_value(p, par) + (id%2));
				if(slot < pv.child_stats.size()){ st = pv.child_stats[slot]; }
			}
			if(st.n>0){ out.quad = st.qd; out.logdet = st.ld; out.n = st.n; }
			else{ out.wn = Loglik_value(p, id); }
			return out;
		};
		const Agg t = walk(1);
		particle_loglik_tree(p) = gaussian_loglik(t.logdet, t.quad, t.n) + t.wn;
	}

	diag.n_trie = (int)trie.size();
	// The trie's block statistics are sig2-free; under deterministic cuts
	// (the candidate children and posteriors are path functions) a later
	// run at the same (range, nugget, ...) and another sig2 may reuse it.
	trie_reuse_ok = !random_cuts ||
		(wn_integrated_reuse() && keep_shallow_caches && trie.size() <= reuse_node_cap);
	trie_seed = seed;
	trie_covpar = covpar;
}

// sig2-only change: every covariance quantity except sig2 equal to the one
// the retained trie was expanded under.  Exact comparisons -- a different
// range in the last bit is a different kernel.
bool SMC::sig2_only_change(const GPM& th, unsigned int seed) const{
	// Integrated-cut WN reuses under common random numbers only (see
	// wn_integrated_reuse): another seed draws other cells almost everywhere.
	if(wn_integrated_reuse() && seed!=trie_seed){ return false; }
	if(!trie_reuse_ok){ return false; }
	const GPM& c = trie_covpar;
	if(th.range.size()!=c.range.size()){ return false; }
	for(Eigen::Index i=0; i<th.range.size(); ++i){ if(th.range(i)!=c.range(i)){ return false; } }
	if(th.coef.size()!=c.coef.size()){ return false; }
	for(Eigen::Index i=0; i<th.coef.size(); ++i){ if(th.coef(i)!=c.coef(i)){ return false; } }
	return th.nugget==c.nugget && th.nu==c.nu && th.tail==c.tail &&
		th.form==c.form && th.family==c.family && th.dtype==c.dtype &&
		th.sig2>0.0 && std::isfinite(th.sig2);
}

void SMC::save_trie(){
	// Nothing to protect when the trie could not be reused anyway (random
	// cuts, or an incomplete run): keeping a second populated trie alive
	// through a joint move would only double the sampler's largest object.
	if(!trie_reuse_ok){
		trie_saved.clear();
		std::vector<PathNode>().swap(trie_saved.nodes);
		trie_saved_ok = false;
		return;
	}
	const size_t budget = trie.cache_budget_bytes;
	std::swap(trie_saved, trie);
	trie.cache_budget_bytes = budget;   // configuration, not state
	trie_saved_covpar = trie_covpar;
	trie_saved_seed = trie_seed;
	trie_saved_ok = trie_reuse_ok;
	trie.clear();
	trie.level_cache.current.paths.clear();
	trie.level_cache.previous.paths.clear();
	trie.level_cache.current.bytes = 0;
	trie.level_cache.previous.bytes = 0;
	trie_reuse_ok = false;
}

bool SMC::restore_trie(){
	if(trie_saved.nodes.empty()){ return false; }
	std::swap(trie_saved, trie);
	trie_covpar = trie_saved_covpar;
	trie_seed = trie_saved_seed;
	trie_reuse_ok = trie_saved_ok;
	trie_saved.clear();
	std::vector<PathNode>().swap(trie_saved.nodes);   // release the capacity too
	trie_saved_ok = false;
	return trie_reuse_ok;
}

// Refresh every sig2-derived value of the retained trie from the stored
// block statistics, in exactly the expressions expand() uses:
//   loglik_knot       = restree_loglik_conditional(stat_knot, sig2)
//   loglik_base       = restree_loglik_conditional(stat_leaf, sig2)   (PP leaf
//                       with a valid block; WhiteNoise / empty / failed leaves
//                       carry no block and keep their sig2-free value)
//   child_loglik_lr   = the same per candidate child from child_stats
//   child_loglik      = the nl/nr combination of expand()
//   rho, lambda       = node_posterior()
// so a node refreshed at sig2' holds the numbers a fresh expansion at sig2'
// would have produced from the same (sig2-free) linear algebra.
void SMC::refresh_sig2_values(){
	for(size_t t=0; t<trie.size(); ++t){
		PathNode& v = trie[t];
		// Arrival value: also for nodes created by child() but never expanded
		// (a stop there reads the parent's refreshed child_loglik_lr, and a
		// later expansion recomputes it; keep the copy coherent anyway).
		if(v.stat_leaf.n>0){
			v.loglik_base = restree_loglik_conditional(v.stat_leaf.ld, v.stat_leaf.qd,
				v.stat_leaf.n, covpar.sig2);
		}
		if(!v.expanded || v.n_arrival<1){ continue; }
		if(baseline=="WhiteNoise" && prior_stop(v.level)==1.0){ continue; } // shortcut node: sig2-free
		if(v.stat_knot.n>0){
			v.loglik_knot = restree_loglik_conditional(v.stat_knot.ld, v.stat_knot.qd,
				v.stat_knot.n, covpar.sig2);
		}
		if(v.proposals.wn){
			// Integrated-cut WN: the interval table is sigma^2-free; only the
			// split mass carries loglik_knot.  The same expressions as
			// prepare_wn_cut_proposal, so the refreshed node holds the numbers
			// a fresh expansion at this sigma^2 would have produced.
			WNCutProposal& pr = *v.proposals.wn;
			const double prior = prior_stop(v.level);
			pr.log_split_mass = std::log1p(-prior) + v.loglik_knot + pr.log_mass_tree[1];
			const double log_normalizer = log_sum_exp2(pr.log_stop_mass, pr.log_split_mass);
			pr.log_weight_increment = log_normalizer - v.loglik_base;
			v.rho_log = pr.log_stop_mass - log_normalizer;
			v.rho = std::min(1.0, std::exp(v.rho_log));
			continue;
		}
		if((int)v.child_stats.size()!=2*dim || v.child_loglik_lr.rows()!=dim ||
		   v.child_n.rows()!=dim || v.child_loglik.size()!=dim){ continue; }
		for(int J=0; J<dim; J++){
			for(int side=0; side<2; ++side){
				const BlockStats& st = v.child_stats[(size_t)(2*J+side)];
				if(st.n>0){
					v.child_loglik_lr(J,side) = restree_loglik_conditional(st.ld, st.qd,
						st.n, covpar.sig2);
				}
			}
			const int nl = v.child_n(J,0), nr = v.child_n(J,1);
			const double ll_l = v.child_loglik_lr(J,0), ll_r = v.child_loglik_lr(J,1);
			if(nl==0 && nr==0){ v.child_loglik(J) = 0.0; }
			else if(nl==0){ v.child_loglik(J) = ll_r; }
			else if(nr==0){ v.child_loglik(J) = ll_l; }
			else{ v.child_loglik(J) = ll_l + ll_r; }
		}
		node_posterior(v);
	}
}

/*************************** native prediction ****************************/
// Prediction for ALL distinct trees in one recursion over their union:
// per-node terms are computed once and shared.  Sibling subtrees descend as
// OpenMP tasks; means and variances go to disjoint (new point, tree) cells,
// the joint LPD of a tree is accumulated atomically.
void SMC::distinct_trees(std::vector<int>& reps, Eigen::VectorXd& w_distinct,
	Eigen::VectorXi& group_of_particle) const{

	// Hash candidates, then compare EVERY structural field (including cut bits).
	// A collision never merges different trees. First representatives and weight
	// additions retain their original particle order, including under collisions.
	Eigen::VectorXd w = (particle_log_weight.array() - log_sum_exp(particle_log_weight)).exp();
	ParticleGroups groups=group_particle_structures(sparse_particles);
	reps=std::move(groups.representatives);
	group_of_particle.resize(N);
	std::vector<double> wsum(reps.size(),0.0);
	for(int p=0; p<N; ++p) {
		const int g=groups.group[p];
		group_of_particle(p)=g; wsum[g]+=w(p);
	}
	w_distinct=Eigen::Map<Eigen::VectorXd>(wsum.data(),wsum.size());
}

FittedTrees SMC::fitted_trees(const std::vector<int>& keep) const {
    FittedTrees out;
    out.nodes.reserve(keep.size());
    out.source_indices=keep;
    for(int p:keep) {
        if(p<0 || p>=N) throw std::out_of_range("fitted_trees: particle index out of range");
        out.nodes.push_back(PredictionNodes::compact(sparse_particles[(size_t)p]));
    }
    return out;
}

void SMC::collect_outputs(Eigen::MatrixXd& S_out,
	Eigen::MatrixXd& J_out, Eigen::MatrixXd& cut_out, Eigen::MatrixXd& rho_out,
	const std::vector<int>* representatives) const{

	const int K = representatives ? (int)representatives->size() : N;

	S_out.setConstant(n_internal, K, NA_REAL);
	J_out.setConstant(n_internal, K, NA_REAL);
	cut_out.setConstant(n_internal, K, NA_REAL);
	rho_out.setConstant(n_internal, K, NA_REAL);
	// Internal-node positions 1 .. 2^depth - 1; at depth 0 the single
	// position is the root itself (a terminal: S = 1).
	const int id_max = (n_internal==1) ? 1 : n_internal-1;
	for(int k=0; k<K; k++){
		const int i = representatives ? (*representatives)[k] : k;
		for_each_node(i, 1, id_max+1, [&](int id){
			const int8_t a = act_value(i, id);
			if(a==ACT_ABSENT || a==ACT_EMPTY){ return; }
			if(a==ACT_SPLIT){
				const PathNode& v = trie[trie_at_value(i, id)];
				rho_out(id-1, k) = random_cuts ? rhoval_value(i, id) : v.rho;
				S_out(id-1, k) = 0.0;
				const int Ji = jdim_value(i, id);
				J_out(id-1, k) = Ji + 1.0;
				cut_out(id-1, k) = random_cuts ? cutval_value(i, id) : v.cuts(Ji);
			}else if(a==ACT_STOP){
				const int32_t t = trie_at_value(i, id);
				rho_out(id-1, k) = random_cuts ? rhoval_value(i, id) : (t>=0 ? trie[t].rho : 1.0);
				S_out(id-1, k) = 1.0;
			}else if(a==ACT_BOTTOM){
				S_out(id-1,k)=1.0; // size-terminal branch above another branch's deepest level
			}
			// ACT_PENDING cannot remain after the last level's sample_level.
		});
	}
}


void SMC::collect_nodes(const std::vector<int>& representatives, NodeTable& out) const{
	const double nan = std::numeric_limits<double>::quiet_NaN();
	const int K = (int)representatives.size();
	const int n_ids = 2*n_internal;   // ids 1 .. 2^(depth+1) - 1
	for(int k=0; k<K; ++k){
		const int i = representatives[(size_t)k];
		for_each_node(i, 1, n_ids, [&](int id){
			const int8_t a = act_value(i, id);
			if(a==ACT_ABSENT || a==ACT_PENDING || a==ACT_EMPTY){ return; }
			int split = 0, J = -1, n = 0;
			double cut = nan, rho = nan;
			const int32_t t = trie_at_value(i, id);
			if(a==ACT_SPLIT){
				const PathNode& v = trie[t];
				split = 1; J = jdim_value(i, id); n = v.n_arrival;
				cut = random_cuts ? cutval_value(i, id) : v.cuts(J);
				rho = random_cuts ? rhoval_value(i, id) : v.rho;
			}else if(a==ACT_STOP){
				const PathNode& v = trie[t];
				n = v.n_arrival;
				rho = random_cuts ? rhoval_value(i, id) : v.rho;
			}else if(a==ACT_BOTTOM){
				if(t>=0){ n = trie[t].n_arrival; }
				else{   // deterministic cuts: the parent's candidate-child count
					const int par = id/2;
					const PathNode& pv = trie[trie_at_value(i, par)];
					n = pv.child_n(jdim_value(i, par), id%2);
				}
			}
			out.tree.push_back(k); out.id.push_back(id); out.split.push_back(split);
			out.J.push_back(J); out.n.push_back(n); out.cut.push_back(cut); out.rho.push_back(rho);
		});
	}
}

// ==================== Main-thread R configuration and diagnostics ====================

// Callers validate scalar sampler options in R, or at restree_loglik's direct
// native boundary. Configuration only copies them into the engine.
void configure_smc(SMC& eng, const ModelSpec& sp, int n_particles,
	const std::string& resampling, double temper_alpha, int seed, int ncores){
	if(sp.leaf_model=="Full")
		Rcpp::stop("SMC over trees needs leaf_model PP or WN; Full uses one fixed tree.\n");
	eng.depth = sp.depth;
	eng.r_knot = sp.r;
	eng.baseline = sp.leaf_model;
	eng.cut_method = sp.cut_method;
	eng.cut_candidates = sp.cut_candidates;
	eng.design = sp.design;
	eng.nested_cand_factor = sp.nested_factor;
	eng.resampling_method = resampling;
	// CGM prior: stop prob 1 - (1-prior_rho)/(1+depth)^prior_beta.
	eng.prior_rho = sp.prior_rho;
	eng.prior_beta = sp.prior_beta;
	eng.N = n_particles;
	eng.minESS = 0.5 * static_cast<double>(n_particles);
	eng.alpha = temper_alpha;
	eng.seed = static_cast<unsigned int>(seed);
	configure_threads(eng, ncores);
}

Rcpp::DataFrame level_diag_frame(const SMC& eng){
	const int L = eng.level_ess.size();
	Rcpp::IntegerVector level(L), last_node(L), last_step(L), resampled(L), n_paths(L);
	Rcpp::NumericVector ess(L), ess_after(L), maxw(L), inc(L);
	for(int l=0; l<L; ++l){
		level[l] = l;
		last_node[l] = eng.level_last_node(l);
		last_step[l] = eng.level_last_step(l);
		ess[l] = eng.level_ess(l);
		ess_after[l] = eng.level_ess_after(l);
		resampled[l] = eng.level_resampled(l);
		maxw[l] = eng.level_max_weight(l);
		inc[l] = eng.level_logZ_inc(l);
		n_paths[l] = (l < eng.diag.distinct_per_level.size()) ? eng.diag.distinct_per_level(l) : NA_INTEGER;
	}
	return Rcpp::DataFrame::create(
		Rcpp::_["level"] = level,
		Rcpp::_["last_node"] = last_node,
		Rcpp::_["last_step"] = last_step,
		Rcpp::_["ess"] = ess,
		Rcpp::_["resampled"] = resampled,
		Rcpp::_["ess_after"] = ess_after,
		Rcpp::_["max_weight"] = maxw,
		Rcpp::_["logZ_increment"] = inc,
		Rcpp::_["n_paths"] = n_paths);
}
// One row per processed node position, plus initialization. Inactive PP
// positions are omitted; node_id is the binary-tree position, step is ordinal.
Rcpp::DataFrame step_diag_frame(const SMC& eng){
	const int n = eng.step_diag.size();
	Rcpp::IntegerVector step(n), node_id(n), level(n), resampled(n);
	Rcpp::NumericVector ess(n), ess_after(n), logZ(n);
	for(int i=0; i<n; ++i){
		const auto& x=eng.step_diag[i];
		step[i]=x.step;
		node_id[i]=x.step==0 ? NA_INTEGER : x.node_id;
		level[i]=x.step==0 ? NA_INTEGER : x.level;
		resampled[i]=x.resampled; ess[i]=x.ess; ess_after[i]=x.ess_after;
		logZ[i]=x.logZ;
	}
	return Rcpp::DataFrame::create(Rcpp::_["step"]=step,
		Rcpp::_["node_id"]=node_id,Rcpp::_["level"]=level,
		Rcpp::_["ess"]=ess,Rcpp::_["resampled"]=resampled,
		Rcpp::_["ess_after"]=ess_after,Rcpp::_["logZ"]=logZ);
}
// (structure$nodes): tree 1-based, id, split (logical), J 1-based (NA at
// terminals), cut and rho (NA where undefined), n.
Rcpp::DataFrame nodes_frame(const SMC& eng, const std::vector<int>& reps){
	SMC::NodeTable t;
	eng.collect_nodes(reps, t);
	const int m = (int)t.id.size();
	Rcpp::IntegerVector tree(m), id(m), J(m), n(m);
	Rcpp::LogicalVector split(m);
	Rcpp::NumericVector cut(m), rho(m);
	for(int i=0; i<m; ++i){
		tree[i] = t.tree[(size_t)i] + 1;
		id[i] = t.id[(size_t)i];
		split[i] = (t.split[(size_t)i]!=0);
		J[i] = t.J[(size_t)i]>=0 ? t.J[(size_t)i] + 1 : NA_INTEGER;
		n[i] = t.n[(size_t)i];
		cut[i] = std::isnan(t.cut[(size_t)i]) ? NA_REAL : t.cut[(size_t)i];
		rho[i] = std::isnan(t.rho[(size_t)i]) ? NA_REAL : t.rho[(size_t)i];
	}
	return Rcpp::DataFrame::create(
		Rcpp::_["tree"] = tree, Rcpp::_["id"] = id, Rcpp::_["split"] = split,
		Rcpp::_["J"] = J, Rcpp::_["cut"] = cut, Rcpp::_["rho"] = rho, Rcpp::_["n"] = n);
}

// Internal knots spent by one particle's realized tree (the knot budget that
// a Vecchia neighbour count is compared against).
int particle_internal_knots(const SMC& eng, int p){
	int total = 0;
	eng.for_each_node(p, 1, eng.n_internal, [&](int id){
		if(eng.act_value(p,id)!=ACT_SPLIT) return;
		const PathNode& v = eng.trie[eng.trie_at_value(p,id)];
		total += eng.knot_count(v.n_arrival);
	});
	return total;
}


// Shared reporting only: this function never transfers sampler ownership.
Rcpp::List smc_info(const SMC& eng){
	size_t entries = 0, bytes = 0;
	unsigned long hits = 0, misses = 0, flushes = 0;
	#ifdef _OPENMP
	#pragma omp critical(restree_path_cache)
	#endif
	{
		entries = eng.path_cache.size();
		bytes = eng.path_cache_bytes;
		hits = eng.path_cache_hits;
		misses = eng.path_cache_misses;
		flushes = eng.path_cache_flushes;
	}
	// Rcpp::List::create takes at most 20 named arguments; the retained-cache
	// fields are appended.
	Rcpp::List out = Rcpp::List::create(
		Rcpp::_["n"] = (int)eng.y.size(),
		Rcpp::_["d"] = (int)eng.X.cols(),
		Rcpp::_["depth"] = eng.depth,
		Rcpp::_["r"] = eng.r_knot,
		Rcpp::_["leaf_model"] = eng.baseline,
		Rcpp::_["cut_method"] = eng.cut_method,
		Rcpp::_["cut_candidates"] = eng.cut_candidates,
		Rcpp::_["design"] = eng.design,
		Rcpp::_["nparticles"] = eng.N,
		Rcpp::_["resampling"] = eng.resampling_method,
		Rcpp::_["temper_alpha"] = eng.alpha,
		Rcpp::_["ncores"] = eng.ncores,
		Rcpp::_["cache_enabled"] = eng.path_cache_enabled,
		Rcpp::_["cache_hits"] = (double)hits,
		Rcpp::_["cache_misses"] = (double)misses,
		Rcpp::_["cache_entries"] = (double)entries,
		Rcpp::_["cache_bytes"] = (double)bytes,
		Rcpp::_["cache_budget_bytes"] = (double)eng.path_cache_budget_bytes,
		Rcpp::_["cache_flushes"] = (double)flushes);
	size_t nested_proposals=0;
	for(const auto& node:eng.trie.nodes) if(node.proposals.nested) ++nested_proposals;
	out.push_back((double)nested_proposals, "nested_proposals");
	out.push_back((double)eng.particle_loglik_tree.size(), "realized_tree_scores");
	out.push_back(eng.ncores_request, "ncores_request");   // the caller's value
	out.push_back(eng.eigen_threads, "eigen_threads");     // Eigen's threads outside the regions
	out.push_back((double)eng.trie.retained_bytes.total(), "kept_cache_bytes");
	out.push_back((double)eng.trie.retained_bytes.postknot, "kept_postknot_bytes");
	out.push_back((double)eng.trie.retained_bytes.wn_proposal, "kept_wn_proposal_bytes");
	out.push_back((double)eng.trie.level_cache.current.bytes, "current_level_cache_bytes");
	out.push_back((double)eng.trie.level_cache.previous.bytes, "previous_level_cache_bytes");
	out.push_back((double)eng.keep_cache_budget_bytes, "kept_cache_budget_bytes");
	out.push_back((double)eng.task_budget_bytes, "task_budget_bytes");
	out.push_back(eng.task_depth(), "task_depth");
	// last run: phase split and peak live post-knot cache (diagnostics)
	out.push_back(eng.diag.expand_sec, "last_expand_seconds");
	out.push_back(eng.diag.sample_sec, "last_sample_seconds");
	out.push_back((double)eng.diag.cache_peak_bytes, "last_cache_peak_bytes");
	// ESS history is compact in the sampler,
	// absent from likelihood-only engines and released from prediction state.
	out.push_back((double)eng.ess_runs.size(), "ess_history_runs");
	out.push_back(eng.ess_history_length(), "ess_history_length");
	// Native prediction-state storage statistics.
	out.push_back(false, "compact_prediction");
	out.push_back(0.0, "prediction_nodes");
	out.push_back(0.0, "prediction_storage_bytes");
	out.push_back((double)sizeof(PredictionNode), "prediction_record_bytes");
	size_t active_nodes=0;
	for(const auto& particle:eng.sparse_particles) active_nodes+=particle.size();
	out.push_back((double)active_nodes, "active_particle_nodes");
	out.push_back("SMC", "state_kind");
	return out;
}

// ==================== Main-thread R sampling entry points ====================

// SMC over trees at fixed theta for a restree_model, retaining model context
// and compact trees in a prediction-only external pointer ("state"). Optional
// prediction (input_new) and joint
// log predictive density (output_new) run in the same traversal.
// [[Rcpp::export]]
Rcpp::List smc_sample_cpp(SEXP model, SEXP theta, int n_particles,
	int seed=42, int ncores=1, double temper_alpha=1.0,
	std::string resampling="stratified",
	Rcpp::Nullable<Eigen::MatrixXd> input_new=R_NilValue,
	Rcpp::Nullable<Eigen::VectorXd> output_new=R_NilValue){

	// Every option and covariance value was validated when the model / theta
	// objects were built; the readers repeat only the structural checks.
	const ModelSpec sp = read_model(model);
	const GPM th = read_theta(theta, (int)sp.X.cols());
	const double sig2 = th.sig2;
	auto t0 = std::chrono::steady_clock::now();

	// Scoped ownership releases all sampler state on success and on errors.
	// At return, move the base context into prediction ownership without copying
	// training matrices; repeated predictions need no rebuild through R.
	std::unique_ptr<SMC> eng_ptr(new SMC(sp.y, sp.X));
	SMC& eng = *eng_ptr;
	configure_smc(eng, sp, n_particles, resampling, temper_alpha, seed, ncores);
	eng.covpar = th;
	reset_covariance_error();

	eng.run();
	stop_on_covariance_error("smc_sample");
	// A numerically singular covariance (duplicated inputs, zero nugget)
	// yields a NaN evidence without tripping the GSL flag: surface it.
	if(std::isnan(eng.logZ)){
		Rcpp::stop("smc_sample: the evidence is NaN -- the residual covariance "
			"is numerically singular (duplicated inputs with a zero nugget?); "
			"increase theta$nugget.\n");
	}

	auto t1 = std::chrono::steady_clock::now();
	double smc_sec = std::chrono::duration<double>(t1-t0).count();

	const int n_internal = eng.n_internal;

	Eigen::VectorXd w = (eng.particle_log_weight.array() - log_sum_exp(eng.particle_log_weight)).exp();

	// ---- distinct trees and (optionally) native trie-shared prediction ----
	std::vector<int> reps;
	Eigen::VectorXd w_distinct;
	Eigen::VectorXi group_of_particle;
	eng.distinct_trees(reps, w_distinct, group_of_particle);
	// The realized trees as one node table over the K distinct trees
	// (structure_of_particle maps particles to its tree index).
	Rcpp::DataFrame nodes = nodes_frame(eng, reps);
	FittedTrees fitted=eng.fitted_trees(reps);
	std::vector<int> prediction_reps(reps.size());
	std::iota(prediction_reps.begin(),prediction_reps.end(),0);

	// sigma^2 is a covariance parameter: known here, or estimated by EB/PMMH.
	const double shared_sig2 = sig2;

	Eigen::MatrixXd par_mean(0,0), par_var(0,0), par_df(0,0);
	Eigen::VectorXd pred_mean(0), pred_var(0);
	Eigen::VectorXd lpd_tree(0);
	double lpd_mix = NA_REAL;
	double pred_sec = NA_REAL;
	if(input_new.isNotNull()){
		// input_new / output_new shapes are validated at the R interface
		// (.restree_xnew), the only route into this entry point.
		Eigen::MatrixXd Xn = Rcpp::as<Eigen::MatrixXd>(input_new.get());
		Eigen::VectorXd yn;
		bool has_y = output_new.isNotNull();
		if(has_y){
			yn = Rcpp::as<Eigen::VectorXd>(output_new.get());
		}
		if(Xn.rows()>0){
			auto tp0 = std::chrono::steady_clock::now();
			eng.covpar.sig2 = shared_sig2;
			// Native trie-shared prediction; when ynew is supplied the same
			// traversal also accumulates the per-tree JOINT log predictive
			// density.
			Predictor(eng,fitted).predict(Xn, prediction_reps, shared_sig2, par_mean, par_var,
				has_y ? &yn : nullptr, has_y ? &lpd_tree : nullptr,
        eng.baseline=="WhiteNoise" ? &par_df : nullptr);
			stop_on_covariance_error("smc_sample prediction");
			// weighted mixture over the distinct trees
			const PredictiveWeights weights = predictive_weights(w_distinct);
			const Eigen::VectorXd& wd = weights.probability;
			predictive_moments(par_mean, par_var, wd, pred_mean, pred_var);
			if(has_y && lpd_tree.size()>0){
				// mixture of per-tree joints: LSE(log wd + lpd_tree)
				lpd_mix = predictive_log_mixture(lpd_tree, weights);
			}
			pred_sec = std::chrono::duration<double>(
				std::chrono::steady_clock::now()-tp0).count();
		}
	}

	// Per-particle run outputs, taken before the state is compacted.
	// particle_loglik is the realized partial-tree value; particle_loglik_tree
	// re-derives it from the per-node block statistics in the
	// fixed-structure evaluator's form (identical() to restree_loglik(tree =
	// .) for the same structure; the two agree to round-off).
	const Eigen::VectorXd particle_loglik = eng.particle_loglik;
	const Eigen::VectorXd particle_loglik_tree = eng.particle_loglik_tree;
	const Eigen::VectorXd ESS = eng.collect_ess_history();
	const double stored_nodes=(double)eng.stored_particle_nodes();
	// FittedTrees already contains only the structural fields. Capture numeric
	// outputs before moving model ownership and destroying this sampler.

	Rcpp::List out = Rcpp::List::create(
		// no weight history: the ESS trajectory is the summary; final weights in "w"
		Rcpp::_["ESS"] = ESS,
		Rcpp::_["nodes"] = nodes,
		Rcpp::_["logZ"] = eng.logZ,
		Rcpp::_["Phi0"] = eng.Phi0,
		Rcpp::_["particle_loglik"] = particle_loglik,
		Rcpp::_["particle_loglik_tree"] = particle_loglik_tree,
		Rcpp::_["loglik"] = eng.logZ,
		Rcpp::_["n_resample"] = eng.n_resample,
		Rcpp::_["w"] = w,
		Rcpp::_["smc_time"] = smc_sec,
		Rcpp::_["n_trie"] = eng.diag.n_trie,
		Rcpp::_["n_expanded"] = eng.diag.n_expanded,
		Rcpp::_["distinct_per_level"] = eng.diag.distinct_per_level
		);
	out["expand_time"] = eng.diag.expand_sec;
	out["sample_time"] = eng.diag.sample_sec;
	out["level_diag"] = level_diag_frame(eng);
	out["step_diag"] = step_diag_frame(eng);
	out["n_internal"] = n_internal;
	out["tree_depth"] = eng.tree_depth;
	out["initial_depth"] = sp.depth;
	out["stored_particle_nodes"] = stored_nodes;
	out["n_distinct_trees"] = (int)reps.size();
	out["structure_of_particle"] = (group_of_particle.array()+1).matrix(); // 1-based
	out["w_distinct"] = w_distinct;
	out["distinct_rep"] = Rcpp::wrap([&]{ std::vector<int> r1(reps.size());
		for(size_t k=0;k<reps.size();++k){ r1[k]=reps[k]+1; } return r1; }());
	// Transfer the owned data and compact trees, not the sampler. The scoped
	// SMC owner releases particles, RNGs, trie and histories before returning.
	const Rcpp::List info=smc_info(eng);
	out["state"] = make_prediction_handle(std::move(static_cast<ResTree&>(eng)),
		std::move(fitted),info);
	if(pred_mean.size()>0){
		out["mean"] = pred_mean;
		out["var"] = pred_var;
		out["par_mean"] = par_mean;
		out["par_var"] = par_var;
        if(par_df.size()){ out["par_df"] = par_df; }
		out["pred_time"] = pred_sec;
		if(lpd_tree.size()>0){
			out["lpd"] = lpd_mix;
			out["lpd_particle"] = lpd_tree; // per DISTINCT tree, order of distinct_rep
		}
	}
	return out;
}


/*==================== persistent evidence engine (EB / PMMH) ====================*/
/* restree_fit(method = "ebayes" / "pmmh") evaluates log Z(theta) hundreds to
   thousands of times on ONE model.  restree_loglik() builds a fresh engine
   per call (one copy of X and y, particle arrays, RNG streams -- cheap)
   and, more to the point, re-derives at every expanded path the geometry
   that does not depend on theta at all: the maximin knot selection on the
   node rows, the cuts, and the PP leaf knot selection inside every
   candidate child.  The handle below owns one engine (one copy of X, y, the
   sampler configuration) whose ResTree keeps that geometry in a
   path-keyed cache (PathCacheEntry) across runs; each evaluation resets the
   run state, sets theta and the seed, and runs the very same sampler, so
   the evidence is bit-identical to restree_loglik() at the same arguments.
   Internal (not in NAMESPACE); the R drivers .restree_eb_fit / .restree_pmmh
   create one handle per fit and .restree_log_evidence() calls it.            */

// [[Rcpp::export]]
SEXP smc_engine_new_cpp(SEXP model, int n_particles, std::string resampling,
	double temper_alpha, int ncores,
	double path_cache_mb = 256.0, int keep_cache_levels = 2,
	double keep_cache_mb = 512.0, double task_budget_mb = 512.0, double level_cache_mb = 512.0){

	const ModelSpec sp = read_model(model);
	if(!std::isfinite(path_cache_mb) || path_cache_mb<0.0)
		Rcpp::stop("path_cache_mb must be a nonnegative number.\n");
	if(keep_cache_levels<0) Rcpp::stop("keep_cache_levels must be nonnegative.\n");
	if(!std::isfinite(keep_cache_mb) || keep_cache_mb<0.0)
		Rcpp::stop("keep_cache_mb must be a nonnegative number.\n");
	if(!std::isfinite(task_budget_mb) || task_budget_mb<0.0)
		Rcpp::stop("task_budget_mb must be a nonnegative number.\n");
    // Integer-representability validation, not a RAM budget/preflight.
    const auto cache_bytes = [](double mb, const char* name){
        const double bytes = mb * 1048576.0;
        const double upper = std::ldexp(1.0,std::numeric_limits<size_t>::digits);
        if(!std::isfinite(bytes) || bytes>=upper)
            Rcpp::stop("%s exceeds the cache-size integer range.\n",name);
        return static_cast<size_t>(bytes);
    };
    const size_t path_bytes = cache_bytes(path_cache_mb,"path_cache_mb");
    const size_t keep_bytes = cache_bytes(keep_cache_mb,"keep_cache_mb");
    const size_t task_bytes = cache_bytes(task_budget_mb,"task_budget_mb");
    const size_t level_bytes = cache_bytes(level_cache_mb,"level_cache_mb");
	Rcpp::XPtr<SMC> eng_ptr(new SMC(sp.y, sp.X), true, smc_handle_tag());
	SMC& eng = *eng_ptr;
	configure_smc(eng, sp, n_particles, resampling, temper_alpha, 0, ncores);
	// The path cache serves deterministic cuts only (under quantile cuts the
	// realized cuts, hence the row sets, are per particle draws).
	const bool deterministic = (sp.cut_method=="middle" || sp.cut_method=="median");
	eng.path_cache_enabled = deterministic && path_cache_mb>0.0;
	// A persistent engine keeps sig2-free level state across sig2-only reuse
	// runs (SMC::retain_cache).  Deterministic cuts: the post-knot state
	// of expanded nodes, the shallowest keep_cache_levels levels
	// unconditionally and deeper nodes within keep_cache_mb.  Integrated-cut
	// WN (maximin design): the rank-cell proposal tables the particles draw
	// from, the root's unconditionally and deeper nodes within keep_cache_mb;
	// a node whose table was not kept is re-expanded on demand.
	const bool wn_integrated = (sp.leaf_model=="WhiteNoise" && sp.design=="maximin" &&
		(sp.cut_method=="uniform" || sp.cut_method=="balanced"));
	eng.keep_shallow_caches = (deterministic || wn_integrated) &&
		(keep_cache_levels>0 || keep_cache_mb>0.0);
	eng.keep_cache_levels = keep_cache_levels;
	eng.keep_cache_budget_bytes = keep_bytes;
	eng.path_cache_budget_bytes = path_bytes;
	// How much of the fixed-structure recursion may be in flight at once
	// (ResTree::task_depth): the tasked top levels are all resident, so this
	// and not ncores is what sets their peak.
	eng.task_budget_bytes = task_bytes;
	eng.trie.cache_budget_bytes = level_bytes;
	return eng_ptr;
}

// log Z(theta) from a persistent engine: the restree_loglik() SMC route
// (same error handling, same attributes) on the retained engine.
// diagnostics = FALSE skips the attributes (distinct-tree grouping, knot
// counts, final ESS), which the optimisers never read; the value is the
// same either way. No per-node ESS history is collected on either route.
// [[Rcpp::export]]
Rcpp::NumericVector smc_engine_logZ_cpp(SEXP handle, SEXP theta, int seed,
	bool diagnostics = true){
	SMC& eng = smc_from_handle(handle);
	const GPM th = read_theta(theta, (int)eng.X.cols());
	reapply_threads(eng);
	reset_covariance_error();
	// sig2-only change since the last completed run: the retained trie's
	// block statistics are sig2-free, so only the sampler runs again
	// (SMC::refresh_sig2_values).  Bit-identical to a fresh run at the
	// same (theta, seed): the same statistics feed the same expressions.
	// Integrated-cut WN reuses only at the seed of the retained run.
	const bool reuse = eng.sig2_only_change(th, static_cast<unsigned int>(seed));
	eng.covpar = th;
	eng.seed = static_cast<unsigned int>(seed);
	eng.run(reuse,false,false); // re-initialises run state, without per-node ESS history
	stop_on_covariance_error("restree_loglik");
	if(!std::isfinite(eng.logZ))
		Rcpp::stop("restree_loglik: the evidence is not finite -- the residual covariance is numerically singular (duplicated inputs with a zero nugget?); increase theta$nugget.\n");

	Rcpp::NumericVector out = Rcpp::NumericVector::create(eng.logZ);
	if(!diagnostics){ return out; }
	std::vector<int> reps; Eigen::VectorXd w_distinct; Eigen::VectorXi group;
	eng.distinct_trees(reps, w_distinct, group);
	const double wsum = w_distinct.sum();
	double n_knot = 0.0;
	for(size_t k=0; k<reps.size(); ++k)
		n_knot += (w_distinct(k)/wsum) * particle_internal_knots(eng, reps[k]);
	out.attr("n_knot") = n_knot;
	Eigen::VectorXd w = (eng.particle_log_weight.array() - log_sum_exp(eng.particle_log_weight)).exp();
	out.attr("ess") = 1.0 / w.squaredNorm();
	out.attr("n_distinct_trees") = (int)reps.size();
	out.attr("reused_trie") = eng.diag_reused_trie;
	out.attr("n_expanded") = eng.diag.n_expanded;
	// The value carries deterministic attributes only (it is compared with
	// identical() downstream); the run's timings are in smc_engine_info_cpp.
	return out;
}

// Move the engine's retained trie aside / back (see SMC::save_trie):
// PMMH saves before a (range, nugget) proposal and restores after a
// rejection, so the sig2-only moves at the current state keep reusing it.
// [[Rcpp::export]]
void smc_engine_save_cpp(SEXP handle){
	smc_from_handle(handle).save_trie();
}

// [[Rcpp::export]]
bool smc_engine_restore_cpp(SEXP handle){
	return smc_from_handle(handle).restore_trie();
}

// Release an inference or prediction handle before waiting for R's GC, which
// cannot see the native data/cache allocations. Clear the pointer first so
// its finalizer is a no-op. NULL/non-pointer/already-cleared handles are ignored;
// unrelated nonempty pointers are rejected without being cleared or deleted.
// [[Rcpp::export]]
void smc_engine_release_cpp(SEXP handle){
	if(TYPEOF(handle)!=EXTPTRSXP) return;
	void* p=R_ExternalPtrAddr(handle);
	if(!p) return;
	if(R_ExternalPtrTag(handle)==prediction_handle_tag()){
		R_ClearExternalPtr(handle);
		delete static_cast<PredictionState*>(p);
	}else if(R_ExternalPtrTag(handle)==smc_handle_tag()){
		R_ClearExternalPtr(handle);
		delete static_cast<SMC*>(p);
	}else Rcpp::stop("unrecognized native engine handle; nothing was released.");
}
