// Prediction conditional on stored residual-tree structures.
#include "prediction.h"
#include "wrapper.h"

// ==================== WN predictive moments ====================

// Predictive moments under the same WN leaf prior.
inline void wn_leaf_moments(const Eigen::VectorXd& y, double& mean, double& variance){
	const double a0=2.0, b0=0.01, mu0=0.0, kappa0=1.0;
	const double nd=(double)y.size();
	const double ybar=y.size()>0 ? y.mean() : 0.0;
	const double sse=y.size()>0 ? (y.array()-ybar).square().sum() : 0.0;
	const double kappa_n=kappa0+nd, a_n=a0+0.5*nd;
	const double b_n=b0+0.5*sse+
		0.5*kappa0*nd/kappa_n*(ybar-mu0)*(ybar-mu0);
	mean=(kappa0*mu0+nd*ybar)/kappa_n;
	variance=b_n*(kappa_n+1.0)/((a_n-1.0)*kappa_n);
}

// ==================== Prediction traversal and compact fitted trees ====================

// The post-knot state of a split node of the prediction traversal: the
// residualised values and propagated coefficients of the data rows
// (zres_c, Qnext: (q+r_k) x n_res) and of the new points (Qnew_next:
// (q+r_k) x nb).  Heap-held and shared_ptr-owned so that the child tasks
// predict_subtree() spawns -- which gather their own column blocks from it
// INSIDE the task, after the parent frame has returned -- keep it alive until
// the last of them has done so.
namespace {
struct PredictNodeState {
	int qr = 0;                 // q + r_k, the row count of the blocks
	Eigen::VectorXd zres_c;
	Eigen::MatrixXd Qnext;
	Eigen::MatrixXd Qnew_next;
};
}

// One child's (resid, Q, Qnew) by column selection from the parent's state:
// lpos / npos are the node-order positions of the rows / new points routed to
// the child.  Exactly the copies the parent loop used to make (same columns,
// same order), now callable from inside the child's task.
static void gather_child_blocks(const PredictNodeState& ps,
	const std::vector<int>& lpos, const std::vector<int>& npos,
	Eigen::VectorXd& cres, Eigen::MatrixXd& cQ, Eigen::MatrixXd& cQn){
	const int nc = (int)lpos.size(), nn = (int)npos.size();
	cres.resize(nc);
	cQ.resize(ps.qr, nc);
	for(int i=0; i<nc; ++i){
		cres(i) = ps.zres_c(lpos[i]);
		cQ.col(i) = ps.Qnext.col(lpos[i]);
	}
	cQn.resize(ps.qr, nn);
	for(int i=0; i<nn; ++i){ cQn.col(i) = ps.Qnew_next.col(npos[i]); }
}

void Predictor::predict_subtree(int tau, std::vector<int>&& T,
	std::vector<int>&& idx, Eigen::VectorXd&& resid, Eigen::MatrixXd&& Q,
	std::vector<int>&& nidx, Eigen::MatrixXd&& Qnew,
	Eigen::VectorXd bmin, Eigen::VectorXd bmax, PredictWork& W,
	std::vector<int> parent_cand){

	const int n_arr = (int)idx.size();
	const int nb = (int)nidx.size();
	if(nb==0 || T.empty() || n_arr==0){ return; }

	// Classify the trees at this node of their realized structure.
	std::vector<int> splitters, stop_pp, stop_wn, term_small;
	struct Edge { int8_t J; double cut; std::vector<int> trees; };
	std::vector<Edge> edges;
	for(size_t u=0; u<T.size(); ++u){
		const int k = T[u];
		const int p = (*W.reps)[k];
		const PredictionNode node=trees.node(p,tau);
		const int8_t a = node.act;
		if(a==ACT_SPLIT){
			splitters.push_back(k);
			bool found=false;
			for(auto& edge:edges) {
				if(edge.J==node.jdim && edge.cut==node.cutval) {
					edge.trees.push_back(k); found=true; break;
				}
			}
			if(!found) edges.push_back(Edge{node.jdim,node.cutval,{k}});
		}else if(a==ACT_STOP || a==ACT_BOTTOM){
			if(eng.baseline=="WhiteNoise"){ stop_wn.push_back(k); }
			else if(eng.baseline=="Full"){ term_small.push_back(k); }
			else if(n_arr>eng.r_knot){ stop_pp.push_back(k); }
			else{ term_small.push_back(k); }
		}
		// ACT_EMPTY / ACT_ABSENT: nothing reaches prediction (the assigned
		// new points keep only their prefix terms).
	}
	// NOTE: a STOP/BOTTOM node with n_arr <= r ends in the dense block
	// (term_small); a PP STOP/BOTTOM with n_arr > r reuses the split-node
	// knot structure (identical maximin selection on the same points).

	Eigen::MatrixXd Xb(nb, eng.dim);
	for(int i=0; i<nb; ++i){ Xb.row(i) = W.Xnew->row(nidx[i]); }

	// ---- WhiteNoise terminals: NIG moments of the arrival residuals ----
	if(!stop_wn.empty()){
		double mu_hat = 0.0, var_hat = 0.0;
		wn_leaf_moments(resid, mu_hat, var_hat);
		const double wn_base = (W.ynew!=nullptr) ? wn_leaf_loglik(resid) : 0.0;
		for(size_t u=0; u<stop_wn.size(); ++u){
			const int k = stop_wn[u];
			if(W.ynew!=nullptr){
				// JOINT augmented NIG marginal of this leaf's test points,
				// exactly as predictLPD: wnll([z; z*]) - wnll(z), with the
				// test residuals taken BEFORE the leaf mean is added.
				Eigen::VectorXd vaug(n_arr + nb);
				vaug.head(n_arr) = resid;
				for(int i=0; i<nb; ++i){
					vaug(n_arr+i) = (*W.ynew)(nidx[i]) - (*W.pm)(nidx[i], k);
				}
				const double add = wn_leaf_loglik(vaug) - wn_base;
				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				(*W.lpd)(k) += add;
			}
			for(int i=0; i<nb; ++i){
				(*W.pm)(nidx[i], k) += mu_hat;
				// the leaf's own predictive variance replaces the path's
				// explained variance in this cell (the terminal is its last
				// writer; lok tells predict() to keep it as it is)
				(*W.pexpl)(nidx[i], k) = var_hat;
				if(W.df){ (*W.df)(nidx[i],k)=4.0+n_arr; }
				if(n_arr>=1){ (*W.lok)[(size_t)nidx[i]*W.reps->size()+k] = 1; }
			}
		}
	}

	// ---- dense (leafFull) terminals: all arrival points act as knots ----
	if(!term_small.empty()){
		Eigen::MatrixXd Xn(n_arr, eng.dim);
		for(int i=0; i<n_arr; ++i){ Xn.row(i) = eng.X.row(idx[i]); }
		// dense_block: the same factor the likelihood's dense leaf uses.
		BlockStats st_dense;
		Eigen::MatrixXd LA;
		if(!eng.dense_block(Xn, resid, Q, st_dense, &LA)){ record_cholesky_error(); return; }
		Eigen::MatrixXd Bf = ikernel(Xb, Xn, eng.covpar);
		if(Q.rows()>0){ Bf.noalias() -= Qnew.transpose()*Q; }
		Eigen::VectorXd t1 = LA.triangularView<Eigen::Lower>().solve(resid);
		Eigen::VectorXd al = LA.transpose().triangularView<Eigen::Upper>().solve(t1);
		Eigen::VectorXd mu_f = Bf * al;
		Eigen::MatrixXd LBf = LA.triangularView<Eigen::Lower>().solve(Bf.transpose());
		Eigen::VectorXd ex_f = LBf.array().square().colwise().sum().transpose();
		// Full baseline: the JOINT dense-leaf conditional needs the Schur block
		// among the new points once per node (shared by every tree here):
		// Snew = C(new,new | path) + tau^2 I - LBf' LBf.
		Eigen::MatrixXd LSf;
		bool full_ok = false;
		if(W.ynew!=nullptr && eng.baseline=="Full"){
			Eigen::MatrixXd Snew = ikernel(Xb, Xb, eng.covpar);
			if(Q.rows()>0){ sym_downdate(Snew, Qnew); }
			Snew.diagonal().array() += eng.covpar.nugget;
			sym_downdate(Snew, LBf);
			if(!checked_cholesky(Snew, LSf)){ return; }
			full_ok = LSf.allFinite() && (LSf.diagonal().array()>0.0).all();
		}
		for(size_t u=0; u<term_small.size(); ++u){
			const int k = term_small[u];
			for(int i=0; i<nb; ++i){
				(*W.pm)(nidx[i], k) += mu_f(i);
				(*W.pexpl)(nidx[i], k) += ex_f(i);
			}
			if(W.ynew!=nullptr){
				double add = 0.0;
				const double s2 = eng.covpar.sig2;
				if(eng.baseline=="Full"){
					// JOINT within-leaf conditional Gaussian at the plug-in
					// sigma^2 (a Full terminal keeps the complete residual
					// covariance among new points).
					if(!full_ok){
						add = -std::numeric_limits<double>::infinity();
					}else{
						Eigen::VectorXd zc(nb);
						for(int i=0; i<nb; ++i){
							zc(i) = (*W.ynew)(nidx[i]) - (*W.pm)(nidx[i], k);
						}
						Eigen::VectorXd t2 = LSf.triangularView<Eigen::Lower>().solve(zc);
						const double logdet_new = 2.0*LSf.diagonal().array().log().sum();
						add = -0.5*(nb*std::log(2.0*M_PI*s2) + logdet_new + t2.dot(t2)/s2);
					}
				}else{
					// Dense terminal under the PP baseline: per-point conditional
					// Gaussian at the plug-in sigma^2 (predictLPD diagonalizes the
					// new-point Schur block for PP), lambda = (1+tau^2) - e_total.
					for(int i=0; i<nb; ++i){
						const double lam = (1.0 + eng.covpar.nugget) - (*W.pexpl)(nidx[i], k);
						if(!(lam>0.0)){ add = -std::numeric_limits<double>::infinity(); break; }
						const double zc = (*W.ynew)(nidx[i]) - (*W.pm)(nidx[i], k);
						add += -0.5*(std::log(2.0*M_PI*s2*lam) + zc*zc/(s2*lam));
					}
				}
				#ifdef _OPENMP
				#pragma omp atomic
				#endif
				(*W.lpd)(k) += add;
			}
		}
	}

	if(splitters.empty() && stop_pp.empty()){ return; }

	// ---- shared knot structure: identical for split nodes and PP stops ----
	Eigen::MatrixXd Xn(n_arr, eng.dim);
	for(int i=0; i<n_arr; ++i){ Xn.row(i) = eng.X.row(idx[i]); }
	Eigen::VectorXi ind_knot, ind_res;
	if(eng.nested()){
		eng.nested_node_knots(idx, tau>1 ? &parent_cand : nullptr, bmin, bmax, ind_knot, ind_res);
	}else{
		eng.pick_knots(Xn, bmin, bmax, ind_knot, ind_res, eng.knot_count(n_arr));
	}
	const int rk = ind_knot.size();
	const int n_res = ind_res.size();
	Eigen::MatrixXd Kx(rk, eng.dim), Xres(n_res, eng.dim);
	for(int i=0; i<rk; ++i){ Kx.row(i) = Xn.row(ind_knot(i)); }
	for(int i=0; i<n_res; ++i){ Xres.row(i) = Xn.row(ind_res(i)); }
	Eigen::VectorXd knot_data(rk), zres(n_res);
	for(int i=0; i<rk; ++i){ knot_data(i) = resid(ind_knot(i)); }
	for(int i=0; i<n_res; ++i){ zres(i) = resid(ind_res(i)); }
	const int q = Q.rows();
	Eigen::MatrixXd Qk(q, rk), Qx(q, n_res);
	for(int i=0; i<rk; ++i){ Qk.col(i) = Q.col(ind_knot(i)); }
	for(int i=0; i<n_res; ++i){ Qx.col(i) = Q.col(ind_res(i)); }

	// knot_step: the same knot block the likelihood forms at this node.  A
	// PP terminal needs only the factor; a split also propagates and
	// residualises the data side for the children.
	KnotBlock kb;
	const int kstep_level = splitters.empty() ? (int)KNOT_FACTOR : (int)KNOT_RESIDUALIZE;
	const bool kstep_ok = eng.nested() ?
		eng.nested_knot_step(Kx, Xres, knot_data, zres, Qk, Qx, kstep_level, kb) :
		eng.knot_step(Kx, Xres, knot_data, zres, Qk, Qx, kstep_level, kb);
	if(!kstep_ok){ record_cholesky_error(); return; }
	const Eigen::MatrixXd& Vchol = kb.L;

	// Predictive-process term for the new points at this node (void when the
	// node holds no knots, which only the nested design allows).
	Eigen::VectorXd mu_b = Eigen::VectorXd::Zero(nb), ex_b = Eigen::VectorXd::Zero(nb);
	Eigen::MatrixXd LBn(rk, nb);
	if(rk>0){
		Eigen::MatrixXd Bnew = ikernel(Xb, Kx, eng.covpar);
		if(q>0){ Bnew.noalias() -= Qnew.transpose()*Qk; }
		Eigen::VectorXd al = Vchol.transpose().triangularView<Eigen::Upper>().solve(kb.Ry);
		mu_b = Bnew * al;
		LBn = Vchol.triangularView<Eigen::Lower>().solve(Bnew.transpose());
		ex_b = LBn.array().square().colwise().sum().transpose();
	}

	for(size_t u=0; u<splitters.size(); ++u){
		const int k = splitters[u];
		for(int i=0; i<nb; ++i){
			(*W.pm)(nidx[i], k) += mu_b(i);
			(*W.pexpl)(nidx[i], k) += ex_b(i);
		}
	}
	for(size_t u=0; u<stop_pp.size(); ++u){
		const int k = stop_pp[u];
		for(int i=0; i<nb; ++i){
			(*W.pm)(nidx[i], k) += mu_b(i);
			(*W.pexpl)(nidx[i], k) += ex_b(i);
		}
		if(W.ynew!=nullptr){
			// PP terminal: per-point conditional Gaussian at the plug-in
			// sigma^2 with the diagonal Schur remainder,
			// lambda = (1+tau^2) - e_total (path + terminal knot block).
			double add = 0.0;
			const double s2 = eng.covpar.sig2;
			for(int i=0; i<nb; ++i){
				const double lam = (1.0 + eng.covpar.nugget) - (*W.pexpl)(nidx[i], k);
				if(!(lam>0.0)){ add = -std::numeric_limits<double>::infinity(); break; }
				const double zc = (*W.ynew)(nidx[i]) - (*W.pm)(nidx[i], k);
				add += -0.5*(std::log(2.0*M_PI*s2*lam) + zc*zc/(s2*lam));
			}
			#ifdef _OPENMP
			#pragma omp atomic
			#endif
			(*W.lpd)(k) += add;
		}
	}

	if(splitters.empty()){ return; }

	// ---- continue: data-side coefficients and residualised values (knot_step) ----
	// Moved out of kb into a shared_ptr-owned block: the child tasks below
	// gather their column blocks from it inside the task, after this frame has
	// returned (see PredictNodeState).  A move changes no value.
	std::shared_ptr<PredictNodeState> ps = std::make_shared<PredictNodeState>();
	ps->qr = q + rk;
	ps->Qnext = std::move(kb.Qnext);
	ps->zres_c = std::move(kb.zres_c);
	Eigen::MatrixXd& Qnew_next = ps->Qnew_next;
	Qnew_next.resize(q+rk, nb);
	if(q>0){ Qnew_next.topRows(q) = Qnew; }
	Qnew_next.bottomRows(rk) = LBn;
	// Nothing below reads the node's own blocks any more (the children gather
	// from ps): free them now rather than at the frame's end, which -- with
	// the taskwait below -- is after every descendant has run.  Per frame at
	// level l these are (2 l r + r) x n_l and (l r + 2 r) x nb doubles, i.e.
	// about a gigabyte along one root-to-leaf chain at n = np = 1e5, r = 60.
	Q.resize(0,0); Qx.resize(0,0); Qk.resize(0,0); Qnew.resize(0,0);
	LBn.resize(0,0); Xn.resize(0,0); Kx.resize(0,0); resid.resize(0);
	kb = KnotBlock();

	// Group the splitting trees by realized edge (J and exact cut value).
	std::vector<int> gidx(n_res);
	for(int i=0; i<n_res; ++i){ gidx[i] = idx[ind_res(i)]; }
	// nested design: this node's candidates (global ids) define the children's knots
	std::vector<int> ccand;
	if(eng.nested()){
		std::vector<int> cp = eng.nested_candidates(Xres);
		ccand.resize(cp.size());
		for(size_t a=0; a<cp.size(); ++a){ ccand[a] = gidx[(size_t)cp[a]]; }
	}

	#ifdef _OPENMP
	const bool spawn_tasks = omp_in_parallel() && edges.size()>1;
	#else
	const bool spawn_tasks = false;
	#endif
	for(size_t e=0; e<edges.size(); ++e){
		const int J = edges[e].J;
		const double cut = edges[e].cut;
		for(int side=0; side<2; ++side){
			// data partition (post-knot rows)
			std::vector<int> cidx; cidx.reserve(n_res);
			std::vector<int> lpos; lpos.reserve(n_res);
			for(int i=0; i<n_res; ++i){
				const bool left = Xres(i,J) < cut;
				if((side==0)==left){ cidx.push_back(gidx[i]); lpos.push_back(i); }
			}
			// new-point partition (the same routing rule as the data)
			std::vector<int> cnidx; cnidx.reserve(nb);
			std::vector<int> npos; npos.reserve(nb);
			for(int i=0; i<nb; ++i){
				const bool left = Xb(i,J) < cut;
				if((side==0)==left){ cnidx.push_back(nidx[i]); npos.push_back(i); }
			}
			if(cnidx.empty() || cidx.empty()){
				if(W.ynew!=nullptr && cidx.empty() && !cnidx.empty()){
					// Test points routed into an empty region: plug-in
					// Gaussian with the unexplained prior residual variance
					// (floored at the noise floor).
					const double s2 = eng.covpar.sig2;
					for(size_t uu=0; uu<edges[e].trees.size(); ++uu){
						const int k = edges[e].trees[uu];
						double add = 0.0;
						for(size_t ii=0; ii<cnidx.size(); ++ii){
							double lam = s2*((1.0 + eng.covpar.nugget) -
								(*W.pexpl)(cnidx[ii], k));
							if(lam < s2*eng.covpar.nugget){ lam = s2*eng.covpar.nugget + 1e-12; }
							const double zz = (*W.ynew)(cnidx[ii]) -
								(*W.pm)(cnidx[ii], k);
							add += -0.5*std::log(2.0*M_PI*lam) - 0.5*zz*zz/lam;
						}
						#ifdef _OPENMP
						#pragma omp atomic
						#endif
						(*W.lpd)(k) += add;
					}
				}
				continue;
			}
			Eigen::VectorXd cbmin = bmin, cbmax = bmax;
			if(side==0){ cbmax(J) = cut; } else { cbmin(J) = cut; }
			std::vector<int> cT = edges[e].trees;
			const int ctau = 2*tau + side;
			if(spawn_tasks){
				// The task carries only the small index vectors and a handle on
				// the parent's post-knot state; the child's (resid, Q, Qnew)
				// blocks are gathered INSIDE the task (gather_child_blocks), so
				// at most one child's blocks exist per running task instead of
				// every sibling's from the moment this loop spawns them (E
				// copies of the parent's blocks at once).  The payload is
				// heap-allocated and reaches the task through one firstprivate
				// pointer (no deep copies).
				struct SubtreeInput {
					std::shared_ptr<const PredictNodeState> parent;
					std::vector<int> T, idx, lpos, nidx, npos, cand;
					Eigen::VectorXd bmin, bmax;
				};
				SubtreeInput* in = new SubtreeInput{ps, std::move(cT), std::move(cidx), std::move(lpos),
					std::move(cnidx), std::move(npos), ccand, cbmin, cbmax};
				#ifdef _OPENMP
				#pragma omp task shared(W) firstprivate(ctau, in)
				#endif
				{
					std::unique_ptr<SubtreeInput> own(in);
					eng.worker_error.run("predict-child",[&]{
					Eigen::VectorXd cres;
					Eigen::MatrixXd cQ, cQn;
					gather_child_blocks(*own->parent, own->lpos, own->npos, cres, cQ, cQn);
					std::vector<int> tT = std::move(own->T);
					std::vector<int> tidx = std::move(own->idx);
					std::vector<int> tnidx = std::move(own->nidx);
					std::vector<int> tcand = std::move(own->cand);
					Eigen::VectorXd tbmin = std::move(own->bmin), tbmax = std::move(own->bmax);
					// Drop this task's hold on the parent's state before
					// recursing, so the parent's blocks are freed as soon as the
					// last sibling has gathered from them rather than when the
					// deepest descendant returns.
					own.reset();
					predict_subtree(ctau, std::move(tT), std::move(tidx), std::move(cres),
						std::move(cQ), std::move(tnidx), std::move(cQn),
						std::move(tbmin), std::move(tbmax), W, std::move(tcand));
					});
				}
			}else{
				Eigen::VectorXd cres;
				Eigen::MatrixXd cQ, cQn;
				gather_child_blocks(*ps, lpos, npos, cres, cQ, cQn);
				predict_subtree(ctau, std::move(cT), std::move(cidx), std::move(cres),
					std::move(cQ), std::move(cnidx), std::move(cQn), cbmin, cbmax, W, ccand);
			}
		}
	}
	#ifdef _OPENMP
	// Wait for this node's subtrees before the frame returns, for the reason
	// given at the end of ResTree::descend: a spawn-and-return recursion is
	// bounded only by the runtime's task throttle, which grows with the thread
	// count, and each outstanding task pins its parent's (resid, Q, Qnew)
	// blocks through SubtreeInput::parent.  Prediction is the costlier of the
	// two because the pinned state also carries the n_new coefficient columns.
	// The frame's own handle goes first so the block is still released as soon
	// as the last sibling has gathered from it.
	if(spawn_tasks){
		ps.reset();
		#pragma omp taskwait
	}
	#endif
}

void Predictor::predict(const Eigen::Ref<const Eigen::MatrixXd>& Xnew, const std::vector<int>& reps,
	double sig2_use,
	Eigen::MatrixXd& par_mean, Eigen::MatrixXd& par_var,
	const Eigen::VectorXd* ynew, Eigen::VectorXd* lpd_tree,
	Eigen::MatrixXd* par_df){

	for(int p:reps)
		if(p<0 || (size_t)p>=trees.nodes.size())
			throw std::out_of_range("prediction tree index out of range");
	eng.worker_error.reset();
	const int np = (int)Xnew.rows();
	const int K = (int)reps.size();
	// par_var accumulates the explained variance during the traversal (a WN
	// leaf overwrites its cell with the leaf's predictive variance and sets
	// lok) and is finished in place below: two np x K buffers fewer than a
	// separate accumulator and leaf-variance matrix.
	// lok is written by WhiteNoise terminals only; PP and Full never set it,
	// so the np x K flag block exists for WN alone.
	std::vector<uint8_t> lok(eng.baseline=="WhiteNoise" ? (size_t)np*K : (size_t)0, 0);
	par_mean = Eigen::MatrixXd::Zero(np, K);
	par_var = Eigen::MatrixXd::Zero(np, K);

	if(par_df){ par_df->setConstant(np,K,std::numeric_limits<double>::infinity()); }
	PredictWork W;
	W.df=par_df;
	W.Xnew = &Xnew; W.reps = &reps;
	W.pm = &par_mean; W.pexpl = &par_var; W.lok = &lok;
	if(ynew!=nullptr && lpd_tree!=nullptr){
		lpd_tree->setZero(K);
		W.ynew = ynew;
		W.lpd = lpd_tree;
	}

	std::vector<int> T(K);
	std::iota(T.begin(), T.end(), 0);
	std::vector<int> idx((size_t)eng.X.rows());
	std::iota(idx.begin(), idx.end(), 0);
	Eigen::VectorXd resid = eng.y;
	Eigen::MatrixXd Q(0, eng.X.rows());
	std::vector<int> nidx((size_t)np);
	std::iota(nidx.begin(), nidx.end(), 0);
	Eigen::MatrixXd Qnew(0, np);
	Eigen::VectorXd bmin = Eigen::VectorXd::Zero(eng.dim);
	Eigen::VectorXd bmax = Eigen::VectorXd::Ones(eng.dim);

	#ifdef _OPENMP
	if(eng.ncores>1){
		#pragma omp parallel num_threads(eng.ncores)
		{
			#pragma omp single nowait
			{
				eng.worker_error.run("predict-root",[&]{
				predict_subtree(1, std::move(T), std::move(idx), std::move(resid),
					std::move(Q), std::move(nidx), std::move(Qnew), bmin, bmax, W);
				});
			}
		}
	}else
	#endif
	{
		predict_subtree(1, std::move(T), std::move(idx), std::move(resid),
			std::move(Q), std::move(nidx), std::move(Qnew), bmin, bmax, W);
	}
	eng.worker_error.rethrow();

	// Variance assembly, per tree: the leaf model's predictive variance where
	// a leaf model exists (for WN the NIG posterior predictive variance; the
	// knot layers' contribution is then not added), sig2 * unexplained
	// residual variance otherwise.
	for(int k=0; k<K; ++k){
		for(int i=0; i<np; ++i){
			if(!lok.empty() && lok[(size_t)i*K+k]){ continue; }   // the leaf's predictive variance stands
			double resid_var = (1.0 + eng.covpar.nugget) - par_var(i,k);
			if(resid_var<0.0){ resid_var = 0.0; }
			par_var(i,k) = sig2_use * resid_var;
		}
	}
}


void FittedTrees::load_structures(const ResTree& model, const Eigen::Ref<const Eigen::MatrixXd>& S,
	const Eigen::Ref<const Eigen::MatrixXd>& J,
	const Eigen::Ref<const Eigen::MatrixXd>& cuts){

	const int n_internal=(int)S.rows();
	const int dim=model.dim;
	const int K=(int)S.cols();
	const int n=(int)model.y.size();
	const auto& X=model.X;
	std::vector<PredictionNodes> trees;
	trees.reserve(K);

	for(int k=0; k<K; ++k){
		// Route the observations down the recorded splits to recover per-node
		// counts (S: 0 = split, >= 0.5 = terminal, NaN = unreachable).
		std::vector<int> cnt(2*n_internal, 0);
		for(int i=0; i<n; ++i){
			int id = 1;
			cnt[1]++;
			while(id < n_internal && S(id-1, k)==0.0){
				const int Ji = (int)J(id-1, k) - 1;
				if(Ji<0 || Ji>=dim){ throw std::invalid_argument("stored structure has an invalid split dimension."); }
				id = 2*id + (X(i, Ji) < cuts(id-1, k) ? 0 : 1);
				cnt[id]++;
			}
		}
		// Reachability and action codes, parents before children.
		std::vector<char> reach(2*n_internal, 0);
		reach[1] = 1;
		std::vector<PredictionNode> records;
		for(int id=1; id<2*n_internal; ++id){
			if(!reach[id]){ continue; }
			PredictionNode node{id,ACT_STOP,-1,std::numeric_limits<double>::quiet_NaN()};
			if(cnt[id]==0 && id>1){ node.act=ACT_EMPTY; records.push_back(node); continue; }
			if(id>=n_internal){ node.act=ACT_BOTTOM; records.push_back(node); continue; }
			const double s = S(id-1, k);
			if(s==0.0){
				node.act = ACT_SPLIT;
				node.jdim = (int8_t)((int)J(id-1, k) - 1);
				node.cutval = cuts(id-1, k);
				reach[2*id] = 1; reach[2*id+1] = 1;
			}else{
				node.act = ACT_STOP;   // prediction re-derives n at the node
			}
			records.push_back(node);
		}
		trees.emplace_back(std::move(records));
	}
	nodes=std::move(trees); source_indices.clear();
}


FittedTrees full_prediction_trees(const FullTreeGeometry& geometry) {
    std::vector<PredictionNode> records;
    records.reserve(geometry.nodes.size());
    for(const auto& nd:geometry.nodes)
        records.push_back({nd.id,nd.split?ACT_SPLIT:ACT_STOP,static_cast<int8_t>(nd.J),nd.cut});
    FittedTrees out;
    out.nodes.emplace_back(std::move(records));
    return out;
}

// ==================== Main-thread R prediction and restoration ====================

PredictiveWeights predictive_weights(const Eigen::VectorXd& w){
	if(w.size()==0 || !w.allFinite() || (w.array()<0.0).any() || w.maxCoeff()<=0.0)
		Rcpp::stop("weights must be finite and nonnegative with positive total mass.\n");
	PredictiveWeights out;
	const double total = w.sum();
	double log_total;
	if(std::isfinite(total)){
		out.probability = w/total;
		log_total = std::log(total);
	}else{
		const double scale = w.maxCoeff();
		out.probability = w/scale;
		const double scaled_total = out.probability.sum();
		out.probability /= scaled_total;
		log_total = std::log(scale) + std::log(scaled_total);
	}
	out.log_probability.resize(w.size());
	for(Eigen::Index k=0; k<w.size(); ++k)
		out.log_probability(k) = w(k)>0.0 ? std::log(w(k))-log_total :
			-std::numeric_limits<double>::infinity();
	return out;
}
double predictive_log_mixture(const Eigen::VectorXd& log_density,
	const PredictiveWeights& weights){
	if(log_density.size()!=weights.log_probability.size())
		Rcpp::stop("predictive densities and weights must have matching lengths.\n");
	Eigen::VectorXd terms = Eigen::VectorXd::Constant(log_density.size(),
		-std::numeric_limits<double>::infinity());
	for(Eigen::Index k=0; k<terms.size(); ++k){
		if(weights.log_probability(k)==-std::numeric_limits<double>::infinity()) continue;
		if(std::isnan(log_density(k)))
			Rcpp::stop("a positive-weight component has an undefined predictive density.\n");
		terms(k) = log_density(k) + weights.log_probability(k);
	}
	return log_sum_exp(terms); // all -Inf is a valid zero predictive density
}


// [[Rcpp::export]]
Rcpp::List predict_state_cpp(SEXP state, const Eigen::Map<Eigen::MatrixXd> input_new,
	Rcpp::IntegerVector distinct_rep, Eigen::VectorXd w_distinct,
	double sig2,
	Rcpp::Nullable<Eigen::VectorXd> output_new=R_NilValue, int ncores=1){

	if(TYPEOF(state)==EXTPTRSXP && R_ExternalPtrAddr(state)==nullptr) return R_NilValue;
	PredictionState& fitted=prediction_from_handle(state);
	ResTree& eng=fitted.model;
	// input_new / output_new shapes and sig2 are validated at the R interface
	// (.restree_xnew / .restree_theta) on every route here.  distinct_rep and
	// w_distinct are INTERNAL stored-fit fields with no R-level counterpart,
	// so their consistency guards stay.
	if((int)distinct_rep.size()!=(int)w_distinct.size() || distinct_rep.size()==0){
		Rcpp::stop("distinct_rep and w_distinct must have equal positive length.\n");
	}
	const PredictiveWeights weights = predictive_weights(w_distinct);
	std::vector<int> reps(distinct_rep.size());
	for(int k=0; k<(int)distinct_rep.size(); ++k){
		// run index of the representative -> its column in the retained state
		const int p = fitted.trees.position(distinct_rep[k]-1);
		if(p<0){ Rcpp::stop("distinct_rep names a particle the state does not hold.\n"); }
		reps[(size_t)k] = p;
	}
	configure_threads(eng, ncores);
	eng.covpar.sig2 = sig2;

	Eigen::VectorXd yn; bool has_y = output_new.isNotNull();
	if(has_y){
		yn = Rcpp::as<Eigen::VectorXd>(output_new.get());
	}

	auto t0 = std::chrono::steady_clock::now();
	Eigen::MatrixXd par_mean, par_var, par_df;
	Eigen::VectorXd lpd_tree;
	reset_covariance_error();
	Predictor(eng,fitted.trees).predict(input_new, reps, sig2, par_mean, par_var,
		has_y ? &yn : nullptr, has_y ? &lpd_tree : nullptr,
        eng.baseline=="WhiteNoise" ? &par_df : nullptr);
	stop_on_covariance_error("predict_state");

	const Eigen::VectorXd& wd = weights.probability;
	Eigen::VectorXd pred_mean, pred_var;
	predictive_moments(par_mean, par_var, wd, pred_mean, pred_var);
	double sec = std::chrono::duration<double>(
		std::chrono::steady_clock::now()-t0).count();

	Rcpp::List out = Rcpp::List::create(
		Rcpp::_["mean"] = pred_mean,
		Rcpp::_["var"] = pred_var,
		Rcpp::_["par_mean"] = par_mean,
		Rcpp::_["par_var"] = par_var,
		Rcpp::_["w"] = wd,
		Rcpp::_["pred_time"] = sec);
	if(par_df.size()){ out["par_df"] = par_df; }
	if(has_y && lpd_tree.size()>0){
		out["lpd"] = predictive_log_mixture(lpd_tree, weights);
		out["lpd_particle"] = lpd_tree;
	}else{
		out["lpd"] = NA_REAL;
	}
	return out;
}

// Reconstruct a live prediction state from a STORED fit by pure integer
// routing (O(n * depth * K), no SMC rerun); expensive quantities are
// re-derived inside predict() once per distinct path.  Pair with
// predict_state_cpp using distinct_rep = 1..K.
// [[Rcpp::export]]
SEXP rebuild_state_cpp(SEXP model, SEXP theta,
	const Eigen::Map<Eigen::MatrixXd> S, const Eigen::Map<Eigen::MatrixXd> J,
	const Eigen::Map<Eigen::MatrixXd> cuts){

	const ModelSpec sp = read_model(model);
	const GPM th = read_theta(theta, (int)sp.X.cols());
	// The prediction state serves the PP and WhiteNoise leaf models; Full
	// leaves predict through predict_structures_cpp.
	if(sp.leaf_model!="PP" && sp.leaf_model!="WhiteNoise"){
		Rcpp::stop("the prediction state supports the PP and WhiteNoise leaf models.\n");
	}
	// Saved WN fits predict through the sparse node table (rebuild_nodes_cpp);
	// this dense 2^depth x K matrix route is a fallback (and the manual test
	// fixtures), always at the model depth (capped at restree_max_model_depth),
	// not the possibly deeper realized WN depth.
	const bool valid_depth = sp.leaf_model=="WhiteNoise" && sp.depth>0 ?
		(S.rows()>=1 && S.rows()<=(1<<restree_max_model_depth) && !(S.rows() & (S.rows()-1))) : S.rows()==(1<<sp.depth);
	if(!valid_depth || J.rows()!=S.rows() || cuts.rows()!=S.rows() ||
	   J.cols()!=S.cols() || cuts.cols()!=S.cols())
		Rcpp::stop("S, J, cuts must be conformable 2^depth x K matrices.\n");
	ResTree eng(sp.y,sp.X);
	// the fixed-structure configuration of the fit (depth, knots, design and
	// its candidate factor), so the rebuilt state predicts with the knots the
	// fit used; ncores is set by predict_state_cpp at prediction time
	configure_fixed(eng, sp, th, 1);
	FittedTrees trees;
	trees.load_structures(eng,S,J,cuts);
	return make_prediction_handle(std::move(eng),std::move(trees));
}


// Restore a PP or WN prediction state directly from its node table (the
// persistent form of a fit): compact sorted prediction records, directly,
// without intermediate active maps. No 2^depth x K matrix or tree inference
// or covariance work is run.  PP ids are bounded by the model depth (its cap),
// WN ids by the realized-depth limit.
// [[Rcpp::export]]
SEXP rebuild_nodes_cpp(SEXP model, SEXP theta, Rcpp::DataFrame nodes, int K){
	const ModelSpec sp=read_model(model);
	const GPM th=read_theta(theta,(int)sp.X.cols());
	if((sp.leaf_model!="WhiteNoise" && sp.leaf_model!="PP" && sp.leaf_model!="Full") || K<1)
		Rcpp::stop("node-table restoration requires a PP, WN or Full model and at least one tree.");
	for(const char* name : {"tree","id","split","J","cut"})
		if(!nodes.containsElementNamed(name)) Rcpp::stop("the stored node table is incomplete.");
	Rcpp::IntegerVector tree=nodes["tree"], id=nodes["id"], J=nodes["J"];
	Rcpp::LogicalVector split=nodes["split"];
	Rcpp::NumericVector cut=nodes["cut"];
	const int nr=id.size();
	if(tree.size()!=nr || J.size()!=nr || split.size()!=nr || cut.size()!=nr)
		Rcpp::stop("stored node columns have different lengths.");
	const bool wn = sp.leaf_model=="WhiteNoise";
	// ids below 2^(D+1), splits below 2^D: D = the realized-depth limit for
	// WN (its trees may be deeper than the model depth), the model depth for PP
	const int D = wn ? restree_max_wn_depth : sp.depth;
	ResTree eng(sp.y,sp.X);
	configure_fixed(eng,sp,th,1);
	std::vector<size_t> sizes(K,0);
	for(int i=0;i<nr;++i){
		if(tree[i]<1 || tree[i]>K) Rcpp::stop("invalid tree/node id or split flag in the stored tree.");
		++sizes[tree[i]-1];
	}
	std::vector<std::vector<PredictionNode>> rows(K);
	for(int k=0;k<K;++k) rows[k].reserve(sizes[k]);
	for(int i=0;i<nr;++i){
		if(sp.depth==0 && (id[i]!=1 || split[i]!=FALSE))
			Rcpp::stop("a depth-zero model requires a single root leaf.");
		if(tree[i]<1 || tree[i]>K || id[i]<1 || id[i]>=(2<<D) || split[i]==NA_LOGICAL)
			Rcpp::stop("invalid tree/node id or split flag in the stored tree.");
		if(split[i] && (id[i]>=(1<<D) || J[i]<1 || J[i]>sp.X.cols() || !std::isfinite(cut[i])))
			Rcpp::stop("invalid split in the stored tree.");
		PredictionNode node{id[i],split[i] ? ACT_SPLIT : ACT_STOP,
			-1,std::numeric_limits<double>::quiet_NaN()};
		if(split[i]) { node.jdim=(int8_t)(J[i]-1); node.cutval=cut[i]; }
		rows[tree[i]-1].push_back(node);
	}
	std::vector<PredictionNodes> trees;
	trees.reserve(K);
	for(auto& records:rows) trees.emplace_back(std::move(records));
	FittedTrees fitted; fitted.nodes=std::move(trees);
	return make_prediction_handle(std::move(eng),std::move(fitted));
}


namespace {
// One prediction pass over K stored structures (columns of S/J/cuts) on an
// engine already holding the data and the fixed-structure configuration
// (configure_fixed): the same engine that scored the structures, so the
// entry point owns ONE copy of X and y for evaluation and prediction alike.
void predict_structures(ResTree& eng, const Eigen::Ref<const Eigen::MatrixXd>& S,
	const Eigen::Ref<const Eigen::MatrixXd>& J, const Eigen::Ref<const Eigen::MatrixXd>& cuts,
	const Eigen::Ref<const Eigen::MatrixXd>& input_new, const Eigen::VectorXd* ynew,
	Eigen::MatrixXd& pm, Eigen::MatrixXd& pv, Eigen::VectorXd* lpd_tree){

	FittedTrees trees;
	trees.load_structures(eng,S,J,cuts);
	std::vector<int> reps((size_t)S.cols());
	std::iota(reps.begin(), reps.end(), 0);
	Predictor(eng,trees).predict(input_new, reps, eng.covpar.sig2,
		pm, pv, ynew, lpd_tree);
}


}
// Prediction from K stored tree structures (columns of S / J / cuts) with
// mixture weights w: one shared prediction pass over all K structures.
// The structures are rebuilt by integer routing (load_structures); nothing
// is rescored -- the mixture weights are the caller's (a fit's structure$w),
// and the per-tree likelihoods play no part in the predictive mixture.  A
// stored split spends min(r, n) knots whatever the leaf model.
// [[Rcpp::export]]
Rcpp::List predict_structures_cpp(SEXP model, SEXP theta, const Eigen::Map<Eigen::MatrixXd> input_new,
	const Eigen::Map<Eigen::MatrixXd> S, const Eigen::Map<Eigen::MatrixXd> J,
	const Eigen::Map<Eigen::MatrixXd> cuts, Eigen::VectorXd w,
	Rcpp::Nullable<Eigen::VectorXd> output_new=R_NilValue, int ncores=1){

	const ModelSpec sp = read_model(model);
	const GPM th = read_theta(theta, (int)sp.X.cols());
	reset_covariance_error();
	const int tree_depth = sp.depth;
	if(S.rows()!=(1<<tree_depth) || J.rows()!=S.rows() || cuts.rows()!=S.rows() ||
	   J.cols()!=S.cols() || cuts.cols()!=S.cols() || w.size()!=S.cols())
		Rcpp::stop("S, J, cuts must be 2^depth x K matrices and w must have length K.\n");
	const PredictiveWeights weights = predictive_weights(w);

	const int K = S.cols(), np = input_new.rows();
	const bool has_y = output_new.isNotNull();
	Eigen::VectorXd ynew;
	if(has_y){ ynew = Rcpp::as<Eigen::VectorXd>(output_new.get()); }

	auto t0 = std::chrono::steady_clock::now();
	ResTree eng(sp.y, sp.X);
	configure_fixed(eng, sp, th, ncores);
	Eigen::MatrixXd pm(np, K), pv(np, K);
	Eigen::VectorXd lpd_i = Eigen::VectorXd::Constant(K, NA_REAL);
	Eigen::VectorXd lpd_tree(0);
	predict_structures(eng, S, J, cuts, input_new, (has_y ? &ynew : nullptr),
		pm, pv, (has_y ? &lpd_tree : nullptr));
	stop_on_covariance_error("restree prediction");
	if(has_y && lpd_tree.size()==K){ lpd_i = lpd_tree; }
	double sec = std::chrono::duration<double>(
		std::chrono::steady_clock::now()-t0).count();

	if(!pm.allFinite() || !pv.allFinite()){
		Rcpp::stop("prediction produced a non-finite fitted result.\n");
	}
	w = weights.probability;
	Eigen::VectorXd mu, vv;
	predictive_moments(pm, pv, w, mu, vv);
	if(!mu.allFinite() || !vv.allFinite()){
		Rcpp::stop("prediction produced a non-finite mixture moment.\n");
	}
	double lpd = NA_REAL;
	if(has_y){
		lpd = predictive_log_mixture(lpd_i, weights);
	}
	return Rcpp::List::create(
		Rcpp::_["mean"] = mu, Rcpp::_["var"] = vv,
		Rcpp::_["par_mean"] = pm, Rcpp::_["par_var"] = pv,
		Rcpp::_["w"] = w, Rcpp::_["lpd"] = lpd,
		Rcpp::_["lpd_particle"] = lpd_i,
		Rcpp::_["pred_time"] = sec);
}

// ==================== Mixture moments and prediction-state diagnostics ====================

// Internal R/PMMH pooling shares the native prediction assembly.
// [[Rcpp::export]]
Rcpp::List mixture_moments_cpp(const Eigen::Map<Eigen::MatrixXd> mean,
	const Eigen::Map<Eigen::MatrixXd> variance, const Eigen::VectorXd& w){
	const PredictiveWeights weights=predictive_weights(w);
	Eigen::VectorXd mu, vv;
	predictive_moments(mean, variance, weights.probability, mu, vv);
	return Rcpp::List::create(Rcpp::_["mean"]=mu, Rcpp::_["var"]=vv);
}

// Historical run metadata is optional and kept as R data, never as a sampler.
// Live allocation counters below describe the prediction-only owner, not the
// sampler snapshot. Restored fits need no sampling metadata at all.
Rcpp::List prediction_state_info(SEXP handle){
    const auto& state=prediction_from_handle(handle);
    const auto& eng=state.model;
    SEXP saved=R_ExternalPtrProtected(handle);
    Rcpp::List out=saved==R_NilValue ? Rcpp::List::create() : Rcpp::clone(Rcpp::List(saved));
    out["n"]=(int)eng.y.size(); out["d"]=(int)eng.X.cols();
    out["depth"]=eng.depth; out["r"]=eng.r_knot;
    out["leaf_model"]=eng.baseline; out["cut_method"]=eng.cut_method;
    out["cut_candidates"]=eng.cut_candidates; out["design"]=eng.design;
    out["nparticles"]=(int)state.trees.nodes.size();
    if(saved==R_NilValue){
        out["resampling"]="multinomial"; out["temper_alpha"]=0.01;
        out["cache_flushes"]=0.0; out["cache_hits"]=0.0; out["cache_misses"]=0.0;
        out["kept_cache_budget_bytes"]=static_cast<double>(static_cast<size_t>(512)<<20);
        out["last_expand_seconds"]=0.0; out["last_sample_seconds"]=0.0;
        out["last_cache_peak_bytes"]=0.0;
    }
    out["ncores"]=eng.ncores; out["ncores_request"]=eng.ncores_request;
    out["eigen_threads"]=eng.eigen_threads; out["cache_enabled"]=eng.path_cache_enabled;
    out["cache_entries"]=(double)eng.path_cache.size();
    out["cache_bytes"]=(double)eng.path_cache_bytes;
    out["cache_budget_bytes"]=(double)eng.path_cache_budget_bytes;
    out["task_budget_bytes"]=(double)eng.task_budget_bytes;
    out["task_depth"]=eng.task_depth();
    for(const char* name:{"nested_proposals","realized_tree_scores","kept_cache_bytes",
        "kept_postknot_bytes","kept_wn_proposal_bytes","current_level_cache_bytes",
        "previous_level_cache_bytes","ess_history_runs","active_particle_nodes"})
        out[name]=0.0;
    out["ess_history_length"]=0;
    out["compact_prediction"]=true;
    out["prediction_nodes"]=(double)state.trees.node_count();
    out["prediction_storage_bytes"]=(double)state.trees.storage_bytes();
    out["prediction_record_bytes"]=(double)sizeof(PredictionNode);
    out["state_kind"]="prediction";
    return out;
}
