// Residual-tree construction, shared node calculations and fixed-tree scoring.
#include "ResTree.h"
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

// ==================== Numerical helpers ====================

// Private shared numerical helpers. Keep expression order identical across callers.




inline const double log2pi_const = std::log(2.0*M_PI);


// Conjugate WN leaf marginal under the fixed normal/inverse-gamma prior.
double wn_leaf_loglik(const Eigen::VectorXd& y){
	const int n = y.size();
	if(n==0){ return 0.0; }
	const double a0=2.0, b0=0.01, mu0=0.0, kappa0=1.0;
	const double nd=(double)n, ybar=y.mean();
	const double sse=(y.array()-ybar).square().sum();
	const double kappa_n=kappa0+nd, a_n=a0+0.5*nd;
	const double b_n=b0+0.5*sse+
		0.5*kappa0*nd/kappa_n*(ybar-mu0)*(ybar-mu0);
	return -0.5*nd*log2pi_const + 0.5*(std::log(kappa0)-std::log(kappa_n)) +
		a0*std::log(b0)-a_n*std::log(b_n)+std::lgamma(a_n)-std::lgamma(a0);
}


inline bool llt_factor(const Eigen::MatrixXd& V, Eigen::MatrixXd& L){
	Eigen::LLT<Eigen::MatrixXd> llt(V);
	if(llt.info()!=Eigen::Success){ return false; }
	L = llt.matrixL();
	return L.allFinite() && (L.diagonal().array()>0.0).all();
}

inline Eigen::VectorXi knot_complement(const Eigen::VectorXi& kn, int n){
	std::vector<char> taken((size_t)n, 0);
	for(Eigen::Index i=0; i<kn.size(); ++i){ taken[(size_t)kn(i)] = 1; }
	Eigen::VectorXi res(n - (int)kn.size());
	int c = 0;
	for(int i=0; i<n; ++i){ if(!taken[(size_t)i]){ res(c++) = i; } }
	return res;
}



// ==================== Shared algebra and knots ====================

/************************ shared numerical utilities ************************/

double restree_loglik_conditional(double ld, double qd, int n, double sig2){
	if(n<=0){ return 0.0; }
	if(!(sig2>0.0) || !std::isfinite(ld) || !std::isfinite(qd)){
		return -std::numeric_limits<double>::infinity();
	}
	return -0.5*(n*std::log(2.0*M_PI) + n*std::log(sig2) + ld + qd/sig2);
}

Eigen::VectorXi ancestor_ids(int id){
	if(id<=1){ // the root node has no ancestors (id<=0 is invalid input)
		Eigen::VectorXi index(1);
		index(0) = -1;
		return index;
	}
	int m = (int)(std::log2((double)id));
	if(m==0){
		Eigen::VectorXi index(1);
		index(0) = -1;
		return index;
	}
	Eigen::VectorXi index(m);
	int c=id, i=1;
	while(c>1){ // stop at the root; c=0 would write index[-1]
		c = c >> 1;
		index[m-i] = c;
		i++;
	}
	return index;
}

double log_sum_exp2(double x, double y){
	if(std::isnan(x) || std::isnan(y)){
		return std::numeric_limits<double>::quiet_NaN();
	}
	if(x==std::numeric_limits<double>::infinity() ||
	   y==std::numeric_limits<double>::infinity()){
		return std::numeric_limits<double>::infinity();
	}
	if(x==-std::numeric_limits<double>::infinity()) return y;
	if(y==-std::numeric_limits<double>::infinity()) return x;
	const double hi = std::max(x,y);
	const double lo = std::min(x,y);
	return hi + std::log1p(std::exp(lo-hi));
}

double log_sum_exp(const Eigen::VectorXd & x){
	if(x.size()==0){
		return -std::numeric_limits<double>::infinity();
	}
	double c = -std::numeric_limits<double>::infinity();
	for(Eigen::Index i=0; i<x.size(); ++i){
		if(std::isnan(x(i))) return std::numeric_limits<double>::quiet_NaN();
		if(x(i)==std::numeric_limits<double>::infinity())
			return std::numeric_limits<double>::infinity();
		c = std::max(c,x(i));
	}
	if(c==-std::numeric_limits<double>::infinity()) return c;
	return c + std::log((x.array()-c).exp().sum());
}

double spike_slab_loglik(const double& loglik_baseline, const double& loglik_knot,
	const Eigen::VectorXd& loglik_child, const double& rho, const Eigen::VectorXd& lambda){
	double loglik=0.0;
	Eigen::VectorXd tmp;
	tmp = lambda.array().log() + loglik_child.array();
	if(rho==1.0){
		loglik = loglik_baseline;
	}else if(rho==0.0){
		loglik = loglik_knot + log_sum_exp(tmp);
	}else{
		double part1, part2;
		part1 = std::log(rho) + loglik_baseline;
		part2 = std::log(1.0-rho) + loglik_knot + log_sum_exp(tmp);
		loglik = log_sum_exp2(part1, part2);
	}
	return loglik;
}


/*************************** knot selection ****************************/
// Knot selection on the rows of a node: "maximin" -> seq_maximin over the
// node rows; "boundary" -> half the knots near the candidate cutting planes,
// half maximin.
void ResTree::pick_knots(const Eigen::MatrixXd& Xn, const Eigen::VectorXd& bmin,
	const Eigen::VectorXd& bmax, Eigen::VectorXi& ind_knot, Eigen::VectorXi& ind_res,
	int r_use) const{

	const int n_node = Xn.rows();
	const int rk = std::min(r_use, n_node);
	ind_knot.resize(rk);
	ind_res.resize(n_node-rk);

	if(design=="boundary"){
		Eigen::VectorXd dplane(n_node);
		for(int i=0; i<n_node; i++){
			double dmin = std::numeric_limits<double>::max();
			for(int j=0; j<dim; j++){
				double mid = 0.5*(bmin(j) + bmax(j));
				dmin = std::min(dmin, std::abs(Xn(i,j) - mid));
			}
			dplane(i) = dmin;
		}
		int r_b = rk/2;
		int n_cand = std::min(n_node, 3*rk);
		std::vector<int> idx(n_node);
		std::iota(idx.begin(), idx.end(), 0);
		std::partial_sort(idx.begin(), idx.begin()+n_cand, idx.end(),
			[&dplane](int a, int b){ return dplane(a) < dplane(b); });
		Eigen::MatrixXd cand(n_cand, dim);
		for(int i=0; i<n_cand; i++){ cand.row(i) = Xn.row(idx[i]); }
		Eigen::VectorXi bk(r_b), br(n_cand-r_b);
		seq_maximin(cand, r_b, bk, br);
		std::vector<bool> taken(n_node, false);
		for(int i=0; i<r_b; i++){ ind_knot(i) = idx[bk(i)]; taken[idx[bk(i)]] = true; }
		int n_rest = n_node - r_b, r_m = rk - r_b;
		Eigen::MatrixXd rest(n_rest, dim);
		std::vector<int> restidx(n_rest);
		int c = 0;
		for(int i=0; i<n_node; i++){
			if(!taken[i]){ restidx[c] = i; rest.row(c) = Xn.row(i); c++; }
		}
		Eigen::VectorXi mk(r_m), mr(n_rest-r_m);
		seq_maximin(rest, r_m, mk, mr);
		for(int i=0; i<r_m; i++){ ind_knot(r_b+i) = restidx[mk(i)]; taken[restidx[mk(i)]] = true; }
		c = 0;
		for(int i=0; i<n_node; i++){ if(!taken[i]){ ind_res(c++) = i; } }
	}else if(design=="maximin_full"){
		// full deterministic maximin ordering; O(n^2), fixed-method fits only
		Eigen::VectorXi ord = maximin_order(Xn);
		ind_knot = ord.head(rk);
		ind_res = ord.tail(n_node-rk);
	}else{ // "maximin"
		seq_maximin(Xn, rk, ind_knot, ind_res);
	}
}

/*************************** the two block factorisations ****************************/
// Operation order, fixed here for every route: kernel, nugget on the
// diagonal, THEN the symmetric downdate by the ancestors' coefficients;
// Eigen LLT as formed (V is symmetric by construction: syrk + mirrored lower
// triangle in ikernel / sym_downdate); a factor with a non-finite or
// non-positive diagonal counts as failure.
bool ResTree::knot_factor(const Eigen::MatrixXd& Kx, const Eigen::VectorXd& knot_data,
	const Eigen::MatrixXd& Qk, KnotBlock& kb) const{

	const int rk = (int)Kx.rows();
	const int q = (int)Qk.rows();
	Eigen::MatrixXd V = ikernel(Kx, Kx, covpar);
	V.diagonal().array() += covpar.nugget;
	if(q>0){ sym_downdate(V, Qk); }
	kb.stat.clear();
	if(!llt_factor(V, kb.L)){ return false; }
	kb.Ry = kb.L.triangularView<Eigen::Lower>().solve(knot_data);
	kb.stat.qd = kb.Ry.squaredNorm();
	kb.stat.ld = 2.0*kb.L.diagonal().array().log().sum();
	kb.stat.n  = rk;
	return true;
}

// One column panel of the residual-sized work: the cross block of the panel
// rows against the knots, its downdate by the ancestors' coefficients (Qxp:
// the panel's q coefficient columns), and the triangular solve that turns it
// into the panel's W columns.  ONE definition for every route (knot_step's
// split, the leaf evaluator, prediction): the same expressions on the same
// operands, so the numbers cannot differ between a stored block and a
// streamed one.  Bp is n_panel x r_k, Wp is r_k x n_panel.
void ResTree::knot_cross_panel(const Eigen::MatrixXd& Xp, const Eigen::MatrixXd& Kx,
	const Eigen::Ref<const Eigen::MatrixXd>& Qxp, const Eigen::MatrixXd& Qk,
	const Eigen::MatrixXd& L, Eigen::MatrixXd& Bp, Eigen::MatrixXd& Wp) const{
	Bp = ikernel(Xp, Kx, covpar);
	if(Qk.rows()>0){ Bp.noalias() -= Qxp.transpose()*Qk; }
	Wp = Bp.transpose();
	L.triangularView<Eigen::Lower>().solveInPlace(Wp);
}

bool ResTree::knot_step(const Eigen::MatrixXd& Kx, const Eigen::MatrixXd& Xres,
	const Eigen::VectorXd& knot_data, const Eigen::VectorXd& zres,
	const Eigen::MatrixXd& Qk, const Eigen::MatrixXd& Qx,
	int level, KnotBlock& kb) const{

	const int rk = (int)Kx.rows();
	const int n_res = (int)Xres.rows();
	const int q = (int)Qk.rows();
	if(!knot_factor(Kx, knot_data, Qk, kb)){ return false; }
	if(level<KNOT_PROPAGATE){ return true; }
	// (The knot block above is r_k x r_k and stays on the calling thread; the
	// cross block and its solve below are the residual-sized part and the only
	// place where a large node has work worth spreading.  The factorisation
	// comes first so a non-positive-definite block costs no kernel evaluations.)

	// One column panel at a time: B's rows, their downdate by the ancestors'
	// coefficients, and the triangular solve that turns them into W / Qnext all
	// act on one residual row independently of the others.  restree_solve_panel
	// (ResTree.h) fixes the width, so which sums Eigen groups together is a
	// property of the panel and not of the thread count, and a node that fits in
	// one panel is computed exactly as the unpanelled call computed it.
	//
	// A split (KNOT_RESIDUALIZE) residualises the panel's values in the same
	// pass -- zres_c(panel) -= Bp * alpha, with alpha = L^-T Ry known before the
	// loop -- so the full n_res x r_k cross block B is never stored: Eigen's
	// column-major matrix-vector kernel accumulates each row independently
	// over the knot columns and treats rows in fixed blocks from the panel
	// start (a multiple of the panel width), so the panel product is the
	// unpanelled product row for row.  Only the fixed-structure evaluator's
	// KNOT_PROPAGATE still materialises B and W.
	Eigen::VectorXd alpha;
	if(level>=KNOT_RESIDUALIZE){
		kb.zres_c = zres;
		if(n_res>0){ alpha = kb.L.transpose().triangularView<Eigen::Upper>().solve(kb.Ry); }
		kb.B.resize(0,0); kb.W.resize(0,0);
	}
	Eigen::MatrixXd* solved = nullptr;   // W, or the bottom r_k rows of Qnext
	int solved_row0 = 0;
	if(level==KNOT_PROPAGATE){
		kb.B.resize(n_res, rk);
		kb.W.resize(rk, n_res);
		solved = &kb.W;
	}else{
		kb.Qnext.resize(q+rk, n_res);
		if(q>0){ kb.Qnext.topRows(q) = Qx; }
		solved = &kb.Qnext;
		solved_row0 = q;
	}
	const int n_panel = n_res>0 ?
		(n_res + restree_solve_panel - 1)/restree_solve_panel : 0;
	auto solve_panel = [&](int p){
		const int j0 = p*restree_solve_panel;
		const int jn = std::min(restree_solve_panel, n_res - j0);
		const Eigen::MatrixXd Xp = Xres.middleRows(j0, jn);
		Eigen::MatrixXd Bp, Sp;
		knot_cross_panel(Xp, Kx, Qx.middleCols(j0, jn), Qk, kb.L, Bp, Sp);
		if(level==KNOT_PROPAGATE){ kb.B.middleRows(j0, jn) = Bp; }
		else{ kb.zres_c.segment(j0, jn).noalias() -= Bp*alpha; }
		solved->block(solved_row0, j0, rk, jn) = Sp;
	};
	#ifdef _OPENMP
	// Tasks of the team the caller already opened -- never a new team, which
	// would fork ncores^2 threads.  A node small enough to be one panel stays
	// on the calling thread, so leaf evaluations add nothing to the queue.
	if(n_panel>1 && omp_in_parallel()){
		for(int p=0; p<n_panel; ++p){
			#pragma omp task default(shared) firstprivate(p)
			worker_error.run("panel",[&]{ solve_panel(p); });
		}
		#pragma omp taskwait
	}else{
		for(int p=0; p<n_panel; ++p){ worker_error.run("panel",[&]{ solve_panel(p); }); }
	}
	#else
	for(int p=0; p<n_panel; ++p){ worker_error.run("panel",[&]{ solve_panel(p); }); }
	#endif
	worker_error.rethrow();
	return true;
}

bool ResTree::dense_block(const Eigen::MatrixXd& Xn, const Eigen::VectorXd& z,
	const Eigen::MatrixXd& Q, BlockStats& st, Eigen::MatrixXd* L_out) const{

	st.clear();
	const int n = (int)z.size();
	if(n<=0){ return true; }
	Eigen::MatrixXd V = ikernel(Xn, Xn, covpar);
	V.diagonal().array() += covpar.nugget;
	if(Q.rows()>0){ sym_downdate(V, Q); }
	Eigen::MatrixXd L;
	if(!llt_factor(V, L)){ return false; }
	Eigen::VectorXd t1 = L.triangularView<Eigen::Lower>().solve(z);
	st.ld = 2.0*L.diagonal().array().log().sum();
	st.qd = t1.squaredNorm();
	st.n  = n;
	if(L_out){ *L_out = std::move(L); }
	return true;
}

/*************************** terminal evaluations ****************************/
// The dense leaf: every row of the block a knot.
double ResTree::leaf_full_loglik(const Eigen::MatrixXd& Xn, const Eigen::VectorXd& z,
	const Eigen::MatrixXd& Q, BlockStats& st) const{

	if(z.size()<=0){ st.clear(); return 0.0; }
	if(!dense_block(Xn, z, Q, st, nullptr)){
		record_cholesky_error();
		st.clear();
		return -std::numeric_limits<double>::infinity();
	}
	return restree_loglik_conditional(st.ld, st.qd, st.n, covpar.sig2);
}

// The residual rows left by a knot selection on n rows: the ascending
// complement of kn, which is exactly the input_ind that seq_maximin (and the
// boundary design's final pass in pick_knots) return.
double ResTree::leaf_pp_loglik(const Eigen::MatrixXd& Xn, const Eigen::VectorXd& z,
	const Eigen::MatrixXd& Q, const Eigen::VectorXd& bmin,
	const Eigen::VectorXd& bmax, BlockStats& st,
	const Eigen::VectorXi* kn_in, Eigen::VectorXi* kn_out) const{
	return leaf_pp_loglik_rows(Xn, z, Q, nullptr, (int)z.size(), bmin, bmax, st, kn_in, kn_out);
}

// The PP leaf on rows `rows` (positions into Xsrc / zsrc / Qsrc, ascending;
// nullptr = every row) WITHOUT gathering the block: only the node's
// coordinates (only when selecting knots, not on a geometry-cache hit) and,
// per 1024-row panel, the panel's rows, values and coefficient columns are
// copied. The per-row remainder
// terms (the residualised value and the diagonal Schur variance) are kept in
// two n_res vectors and reduced once, exactly as the block form reduced them,
// so the result is the block form's bit for bit:
//   * pick_knots sees the same coordinate matrix;
//   * each panel's cross block and W columns come from knot_cross_panel on
//     the same operands the split's knot_step gives it (a panel starts at a
//     multiple of the panel width, so Eigen's blocking and the column
//     alignment classes are those of the block form);
//   * zr = zres - B alpha is the same row-independent matrix-vector product,
//     panel by panel;
//   * the column sums of squares of Q and W are per-column reductions;
//   * the final sums run over the full n_res vectors in the block form's
//     expressions.
// A cached indexed leaf also avoids n_total * dim coordinate doubles. The
// uncached selection and small dense-leaf paths retain their coordinate gather.
double ResTree::leaf_pp_loglik_rows(const Eigen::MatrixXd& Xsrc, const Eigen::VectorXd& zsrc,
	const Eigen::MatrixXd& Qsrc, const int* rows, int n_total,
	const Eigen::VectorXd& bmin, const Eigen::VectorXd& bmax, BlockStats& st,
	const Eigen::VectorXi* kn_in, Eigen::VectorXi* kn_out) const{

	st.clear();
	if(n_total<=0){ return 0.0; }
	const int q = (int)Qsrc.rows();
	auto at = [&](int i){ return rows ? rows[i] : i; };
	// Knot selection needs contiguous child coordinates; known knots need only
	// their own rows and the current panel. Keep the small dense gather intact.
	const bool cached_knots = kn_in!=nullptr && kn_in->size()>0;
	const bool indexed_coordinates = rows && cached_knots && n_total>r_knot;
	Eigen::MatrixXd Xn_gathered;
	const Eigen::MatrixXd* Xn = &Xsrc;
	if(rows && !indexed_coordinates){
		Xn_gathered.resize(n_total, dim);
		for(int i=0; i<n_total; ++i){ Xn_gathered.row(i) = Xsrc.row(rows[i]); }
		Xn = &Xn_gathered;
	}
	if(n_total<=r_knot){
		// The dense block on every row (leaf_full_loglik): a small gather.
		if(!rows){ return leaf_full_loglik(Xsrc, zsrc, Qsrc, st); }
		Eigen::VectorXd zn(n_total);
		Eigen::MatrixXd Qn(q, n_total);
		for(int i=0; i<n_total; ++i){ zn(i) = zsrc(rows[i]); if(q>0){ Qn.col(i) = Qsrc.col(rows[i]); } }
		return leaf_full_loglik(*Xn, zn, Qn, st);
	}

	Eigen::VectorXi kn, res;
	if(cached_knots){
		kn = *kn_in;
		res = knot_complement(kn, n_total);
	}else{
		pick_knots(*Xn, bmin, bmax, kn, res, r_knot);
	}
	if(kn_out!=nullptr){ *kn_out = kn; }
	const int rk_leaf = kn.size();
	if(rk_leaf!=r_knot){ return -std::numeric_limits<double>::infinity(); }
	const int n_res = res.size();

	// The knot block: r_k rows, the same factor step as the block form.
	Eigen::MatrixXd Kx(rk_leaf, dim), Qk(q, rk_leaf);
	Eigen::VectorXd zk_data(rk_leaf);
	for(int i=0; i<rk_leaf; ++i){
		Kx.row(i) = Xn->row(indexed_coordinates ? rows[kn(i)] : kn(i));
		zk_data(i) = zsrc(at(kn(i)));
		if(q>0){ Qk.col(i) = Qsrc.col(at(kn(i))); }
	}
	KnotBlock kb;
	if(!knot_factor(Kx, zk_data, Qk, kb)){
		record_cholesky_error();
		st.clear();
		return -std::numeric_limits<double>::infinity();
	}
	const Eigen::MatrixXd& L = kb.L;
	const Eigen::VectorXd& zk = kb.Ry;
	if(!(L.allFinite() && (L.diagonal().array()>0.0).all())){
		return -std::numeric_limits<double>::infinity();
	}
	const Eigen::VectorXd alpha = L.transpose().triangularView<Eigen::Upper>().solve(zk);

	// Per residual row: 1 + tau^2 - |Q_i|^2 (dbase), then - |W_i|^2 (diag),
	// and the residualised value zr_i = zres_i - (B alpha)_i; panel by panel.
	Eigen::VectorXd dbase = Eigen::VectorXd::Constant(n_res, 1.0+covpar.nugget);
	Eigen::VectorXd diag(n_res), zr(n_res);
	const int n_panel = n_res>0 ?
		(n_res + restree_solve_panel - 1)/restree_solve_panel : 0;
	auto leaf_panel = [&](int p){
		const int j0 = p*restree_solve_panel;
		const int jn = std::min(restree_solve_panel, n_res - j0);
		Eigen::MatrixXd Xp(jn, dim), Qxp(q, jn);
		Eigen::VectorXd zres_p(jn);
		for(int i=0; i<jn; ++i){
			const int r0 = res(j0+i);
			Xp.row(i) = Xn->row(indexed_coordinates ? rows[r0] : r0);
			zres_p(i) = zsrc(at(r0));
			if(q>0){ Qxp.col(i) = Qsrc.col(at(r0)); }
		}
		Eigen::MatrixXd Bp, Wp;
		knot_cross_panel(Xp, Kx, Qxp, Qk, L, Bp, Wp);
		if(q>0){
			dbase.segment(j0, jn).array() -= Qxp.array().square().colwise().sum().transpose();
		}
		Eigen::VectorXd diag_p = dbase.segment(j0, jn);
		diag_p.array() -= Wp.array().square().colwise().sum().transpose();
		diag.segment(j0, jn) = diag_p;
		Eigen::VectorXd zr_p = zres_p - Bp*alpha;
		zr.segment(j0, jn) = zr_p;
	};
	#ifdef _OPENMP
	if(n_panel>1 && omp_in_parallel()){
		for(int p=0; p<n_panel; ++p){
			#pragma omp task default(shared) firstprivate(p)
			worker_error.run("leaf-panel",[&]{ leaf_panel(p); });
		}
		#pragma omp taskwait
	}else{
		for(int p=0; p<n_panel; ++p){ worker_error.run("leaf-panel",[&]{ leaf_panel(p); }); }
	}
	#else
	for(int p=0; p<n_panel; ++p){ worker_error.run("leaf-panel",[&]{ leaf_panel(p); }); }
	#endif
	worker_error.rethrow();

	double loglik = -std::numeric_limits<double>::infinity();
	const bool valid = zk.allFinite() && zr.allFinite() && diag.allFinite() &&
	             (diag.array()>-1e-9).all();
	if(valid){
		diag = diag.array().max(1e-12);
		st.qd = zk.squaredNorm() + (zr.array().square()/diag.array()).sum();
		st.ld = 2.0*L.diagonal().array().log().sum()+diag.array().log().sum();
		st.n  = n_total;
		loglik = restree_loglik_conditional(st.ld, st.qd, st.n, covpar.sig2);
	}
	return loglik;
}

// The arrival leaf of a split node from the factors the split's knot_step
// already holds: L, W = the new rows of Qnext, zk = L^-1 zk_data, and the
// residualised values zr = zres - B alpha, which the split formed panel by
// panel as zres_c (the same product, the same rounding).
double ResTree::leaf_pp_loglik_from_factors(int n_total, bool chol_ok,
	const Eigen::MatrixXd& L, const Eigen::Ref<const Eigen::MatrixXd>& W,
	const Eigen::VectorXd& zk, const Eigen::VectorXd& zr,
	const Eigen::MatrixXd& Qx, BlockStats& st) const{

	st.clear();
	const int n_res = (int)zr.size();
	const int q = (int)Qx.rows();
	Eigen::VectorXd dbase = Eigen::VectorXd::Constant(n_res, 1.0+covpar.nugget);
	if(q>0){
		dbase.array() -= Qx.array().square().colwise().sum().transpose();
	}
	double loglik = -std::numeric_limits<double>::infinity();
	if(chol_ok){
		if(L.allFinite() && (L.diagonal().array()>0.0).all()){
			Eigen::VectorXd diag = dbase;
			diag.array() -= W.array().square().colwise().sum().transpose();
			bool valid = zk.allFinite() && zr.allFinite() && diag.allFinite() &&
			             (diag.array()>-1e-9).all();
			if(valid){
				diag = diag.array().max(1e-12);
				st.qd = zk.squaredNorm() + (zr.array().square()/diag.array()).sum();
				st.ld = 2.0*L.diagonal().array().log().sum()+diag.array().log().sum();
				st.n  = n_total;
				loglik = restree_loglik_conditional(st.ld, st.qd, st.n, covpar.sig2);
			}
		}
	}
	return loglik;
}

// Arrival / candidate-child terminal model, dispatching on the baseline
// (PP leaf, else the conjugate white-noise leaf; the SMC never runs on Full).
double ResTree::terminal_loglik(const Eigen::MatrixXd& Xn, const Eigen::VectorXd& z,
	const Eigen::MatrixXd& Q, const Eigen::VectorXd& bmin,
	const Eigen::VectorXd& bmax, BlockStats& st) const{
	if(baseline=="PP"){
		return leaf_pp_loglik(Xn, z, Q, bmin, bmax, st);
	}
	st.clear();               // WhiteNoise terminals carry no Gaussian block
	return wn_leaf_loglik(z);
}

// terminal_loglik on the rows `rows` of a parent's (Xres, zres, Q) without
// gathering the block: PP through leaf_pp_loglik_rows, WhiteNoise from the
// gathered values alone (it reads nothing else).
double ResTree::terminal_loglik_rows(const Eigen::MatrixXd& Xsrc, const Eigen::VectorXd& zsrc,
	const Eigen::MatrixXd& Qsrc, const int* rows, int n_rows,
	const Eigen::VectorXd& bmin, const Eigen::VectorXd& bmax, BlockStats& st,
	const Eigen::VectorXi* kn_in, Eigen::VectorXi* kn_out) const{
	if(baseline=="PP"){
		return leaf_pp_loglik_rows(Xsrc, zsrc, Qsrc, rows, n_rows, bmin, bmax, st, kn_in, kn_out);
	}
	st.clear();
	Eigen::VectorXd zn(n_rows);
	for(int i=0; i<n_rows; ++i){ zn(i) = zsrc(rows[i]); }
	return wn_leaf_loglik(zn);
}

// ==================== Path construction, cuts and model expansion ====================

namespace {
// Welford summaries avoid cancellation; use extended precision for row sums.
struct WNRunningStats {
 int n = 0;
 long double mean = 0.0L, m2 = 0.0L;
 void add(double z) {
  ++n;
  const long double delta = (long double)z - mean;
  mean += delta / n;
  m2 += delta * ((long double)z - mean);
 }
 double loglik() const {
  if(n==0) return 0.0;
  const double an = 2.0 + 0.5*n;
  const long double bn = 0.01L + 0.5L*std::max(0.0L,m2)
   + 0.5L*n/(1.0L+n)*mean*mean;
  return -0.5*n*log2pi_const - 0.5*std::log1p((double)n)
   + 2.0*std::log(0.01) - an*std::log((double)bn) + std::lgamma(an);
 }
};
} // namespace

// Integral of 6*u*(1-u) over [a,b], without subtracting two CDFs near one.
double beta22_interval_mass(double a, double b) {
 const double width = b-a, mid = a + 0.5*width;
 return 6.0*width*(mid*(1.0-mid)-width*width/12.0);
}


void ResTree::prepare_wn_cut_proposal(PathNode& v,
 const Eigen::MatrixXd& Xres, const Eigen::VectorXd& residual) const {
 auto proposal = std::make_shared<WNCutProposal>();
 const int m = residual.size(), gaps = std::max(1,m-1);
 proposal->n_residual = m;
 proposal->intervals.reserve((size_t)dim*gaps);
 std::vector<double> log_masses;
 log_masses.reserve((size_t)dim*gaps);
 for(int j=0; j<dim; ++j) {
  std::vector<int> order(m);
  std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](int a,int b) {
   return Xres(a,j)<Xres(b,j);
  });
  std::vector<double> prefix(m+1), suffix(m+1);
  WNRunningStats left,right;
  for(int k=0; k<m; ++k) {
   left.add(residual(order[k])); prefix[k+1]=left.loglik();
   right.add(residual(order[m-1-k])); suffix[m-1-k]=right.loglik();
  }
  int first_equal = 0;
  for(int k=0; k<gaps; ++k) {
   WNCutInterval interval;
   interval.coordinate=j; interval.rank_index=k;
   double mass=1.0;
   if(m==0) {
    interval.cut_lower=interval.cut_upper=0.5*(v.box_min(j)+v.box_max(j));
   } else if(m==1) {
    interval.cut_lower=interval.cut_upper=Xres(order[0],j);
   } else {
    interval.cut_lower=Xres(order[k],j);
    interval.cut_upper=Xres(order[k+1],j);
    if(k==0 || Xres(order[k],j)!=Xres(order[k-1],j)) first_equal=k;
    // A flat rank segment has positive prior mass at c=x. All ties go right.
    interval.n_left=(interval.cut_lower==interval.cut_upper) ? first_equal : k+1;
    const double a=(double)k/(m-1), b=(double)(k+1)/(m-1);
    mass=(cut_method=="balanced") ? beta22_interval_mass(a,b) : b-a;
   }
   interval.left_loglik=prefix[interval.n_left];
   interval.right_loglik=suffix[interval.n_left];
   log_masses.push_back(std::log(prior_lambda(j))+std::log(mass)
    + interval.left_loglik+interval.right_loglik);
   proposal->intervals.push_back(interval);
  }
 }
 while(proposal->leaf_offset<log_masses.size()) proposal->leaf_offset*=2;
 proposal->log_mass_tree.assign(2*proposal->leaf_offset,
  -std::numeric_limits<double>::infinity());
 for(size_t k=0; k<log_masses.size(); ++k)
  proposal->log_mass_tree[proposal->leaf_offset+k]=log_masses[k];
 for(size_t k=proposal->leaf_offset-1; k>0; --k)
  proposal->log_mass_tree[k]=log_sum_exp2(proposal->log_mass_tree[2*k],
   proposal->log_mass_tree[2*k+1]);
 const double prior=prior_stop(v.level);
 proposal->log_stop_mass=std::log(prior)+v.loglik_base;
 proposal->log_split_mass=std::log1p(-prior)+v.loglik_knot+proposal->log_mass_tree[1];
 const double log_normalizer=log_sum_exp2(proposal->log_stop_mass,proposal->log_split_mass);
 proposal->log_weight_increment=log_normalizer-v.loglik_base;
 v.rho_log=proposal->log_stop_mass-log_normalizer;
 v.rho=std::min(1.0,std::exp(v.rho_log)); // diagnostic only, not used for drawing
 v.proposals.wn=std::move(proposal);
}

// The stop/split posterior of an expanded node from its loglik_base,
// loglik_knot and child_loglik (probabilities clamped away from 0 and 1).  Shared by
// expand() and by the sig2-only refresh, so the two cannot drift apart.
void ResTree::node_posterior(PathNode& v) const{
	const double rho_prior = prior_stop(v.level);
	double Phi_ss = spike_slab_loglik(v.loglik_base, v.loglik_knot,
		v.child_loglik, rho_prior, prior_lambda);
	v.rho_log = std::log(rho_prior) + v.loglik_base - Phi_ss;
	v.rho = std::exp(v.rho_log);
	if(v.rho<0.0){ v.rho = 0.0; }
	if(v.rho>1.0){ v.rho = 1.0; }
	v.lambda.resize(dim); v.lambda_log.resize(dim);
	Eigen::VectorXd tmp2 = prior_lambda.array().log() + v.loglik_knot + v.child_loglik.array();
	double loglik_slab = log_sum_exp(tmp2);
	for(int J=0; J<dim; J++){
		v.lambda_log(J) = tmp2(J) - loglik_slab;
		v.lambda(J) = std::exp(v.lambda_log(J));
		if(v.lambda(J)<0.0){ v.lambda(J) = 0.0; }
		if(v.lambda(J)>1.0){ v.lambda(J) = 1.0; }
	}
}

/**************** expand_level: shared per-path expansion ****************/

int PathTrie::child(int parent_idx, int J, int side, int dim){
	// NOTE: nodes.push_back below may reallocate the vector, so no reference
	// into trie may be held across it -- everything is copied out first and
	// the parent slot is re-addressed afterwards.
	{
		PathNode& par = nodes[parent_idx];
		if((int)par.child_path.size()!=2*dim){ par.child_path.assign(2*dim, -1); }
		const int existing = par.child_path[2*J+side];
		if(existing>=0){ return existing; }
	}
	PathNode v;
	{
		const PathNode& par = nodes[parent_idx];
		v.level = par.level + 1;
		v.tree_id = 2*par.tree_id + side;
		v.parent = parent_idx;
		v.j_from_parent = (int8_t)J;
		v.side_from_parent = (int8_t)side;
		v.n_arrival = par.child_n(J, side);
		v.loglik_base = par.child_loglik_lr(J, side);
		v.stat_leaf = par.child_stats[2*J+side];
		v.box_min = par.box_min; v.box_max = par.box_max;
		const double delta = par.cuts(J);
		v.cut_from_parent = delta;
		if(side==0){ v.box_max(J) = delta; } else { v.box_min(J) = delta; }
	}
	nodes.push_back(std::move(v));
	const int slot = (int)nodes.size()-1;
	nodes[parent_idx].child_path[2*J+side] = slot;
	return slot;
}

// Quantile-cut mode: children are discriminated by the realized cut value
// (exact bit pattern), so particles that split the same path at the same
// cut -- e.g. resampling duplicates -- share one child, while distinct cuts
// get distinct children.  The child carries the per-particle candidate
// evaluation made at its creation (arrival log-likelihood and block stats),
// which is a pure function of (path, J, side, cut) and therefore shared.
int PathTrie::child_random(int parent_idx, int J, int side, double cut,
	int n_child, double ll, const BlockStats& st){

	uint64_t bits; std::memcpy(&bits, &cut, sizeof(double));
	{
		// the node's edges with this cut pattern (child_lookup), then the
		// (J, side) check the full scan made
		const PathNode& par = nodes[parent_idx];
		const auto range = par.child_lookup.equal_range(bits);
		for(auto it=range.first; it!=range.second; ++it){
			const ChildEdge& ce = par.child_edges[(size_t)it->second];
			if(ce.J==(int8_t)J && ce.side==(int8_t)side){
				uint64_t b2; std::memcpy(&b2, &ce.cut, sizeof(double));
				if(b2==bits){ return ce.idx; }
			}
		}
	}
	PathNode v;
	{
		const PathNode& par = nodes[parent_idx];
		v.level = par.level + 1;
		v.tree_id = 2*par.tree_id + side;
		v.parent = parent_idx;
		v.j_from_parent = (int8_t)J;
		v.side_from_parent = (int8_t)side;
		v.cut_from_parent = cut;
		v.n_arrival = n_child;
		v.loglik_base = ll;
		v.stat_leaf = st;
		v.box_min = par.box_min; v.box_max = par.box_max;
		if(side==0){ v.box_max(J) = cut; } else { v.box_min(J) = cut; }
	}
	nodes.push_back(std::move(v));
	const int slot = (int)nodes.size()-1;
	ChildEdge e; e.J = (int8_t)J; e.side = (int8_t)side; e.cut = cut; e.idx = slot;
	nodes[parent_idx].child_edges.push_back(e);
	nodes[parent_idx].child_lookup.emplace(bits, (int)nodes[parent_idx].child_edges.size()-1);
	return slot;
}

int PathTrie::child_cell(int parent_idx, int J, int side, int cell, double cut_rep,
	int n_child, double ll, const BlockStats& st){

	// (cell, J, side) packed into one key: cell in its own 32 bits (a rank
	// cell index below 2^31), J and side in the low 16
	const uint64_t key = ((uint64_t)(uint32_t)cell << 16) |
		((uint64_t)(uint8_t)J << 8) | (uint64_t)(uint8_t)side;
	{
		const PathNode& par = nodes[parent_idx];
		const auto range = par.child_lookup.equal_range(key);
		for(auto it=range.first; it!=range.second; ++it){
			const ChildEdge& ce = par.child_edges[(size_t)it->second];
			if(ce.cell==cell && ce.J==(int8_t)J && ce.side==(int8_t)side){ return ce.idx; }
		}
	}
	PathNode v;
	{
		const PathNode& par = nodes[parent_idx];
		v.level = par.level + 1;
		v.tree_id = 2*par.tree_id + side;
		v.parent = parent_idx;
		v.j_from_parent = (int8_t)J;
		v.side_from_parent = (int8_t)side;
		v.cut_from_parent = cut_rep;
		v.n_arrival = n_child;
		v.loglik_base = ll;
		v.stat_leaf = st;
		v.box_min = par.box_min; v.box_max = par.box_max;
		if(side==0){ v.box_max(J) = cut_rep; } else { v.box_min(J) = cut_rep; }
	}
	nodes.push_back(std::move(v));
	const int slot = (int)nodes.size()-1;
	ChildEdge e; e.J = (int8_t)J; e.side = (int8_t)side; e.cut = cut_rep; e.cell = cell; e.idx = slot;
	nodes[parent_idx].child_edges.push_back(e);
	nodes[parent_idx].child_lookup.emplace(key, (int)nodes[parent_idx].child_edges.size()-1);
	return slot;
}

// Bytes of a node's deterministic-cut post-knot cache.
void ResTree::expand(PathTrie& trie, PathNode& v, const std::vector<int>& idx,
	const Eigen::VectorXd& resid, const Eigen::MatrixXd& Q) const{

	const int n = (int)idx.size();

	v.n_arrival = n;
	if((int)v.child_path.size()!=2*dim){ v.child_path.assign(2*dim, -1); }
	if(n < 1){
		// Arrival evaluation of an empty node (0 for every leaf model) and a
		// terminal by construction; sample_level never draws here.  (A
		// nonempty node always flows through the ordinary expansion, a small
		// node spending all knot_count(n) = n of its rows as knots, so its
		// candidate children are empty.)
		Eigen::MatrixXd Xn(0, dim);
		v.loglik_base = terminal_loglik(Xn, resid, Q, v.box_min, v.box_max, v.stat_leaf);
		v.expanded = true;
		v.need_expand = false;
		return;
	}

	// The nested knot design (design = "nested"): its own expansion
	// (nested PP section below) -- inherited knots, exact cut integration, shared children.
	if(nested()){ expand_nested(trie, v, idx, resid, Q); return; }

  if(baseline=="WhiteNoise" && (!compare_node(v.level,n) || prior_stop(v.level)==1.0)){
    v.loglik_base=wn_leaf_loglik(resid); v.loglik_knot=0.0;
    v.rho=1.0; v.rho_log=0.0;
    v.lambda=Eigen::VectorXd::Constant(dim,1.0/dim);
    v.lambda_log=v.lambda.array().log();
    v.child_loglik=Eigen::VectorXd::Zero(dim);
    v.cut_workspace.input.resize(0,dim); v.postknot.residual.resize(0); v.postknot.coefficients.resize(0,0);
    v.postknot.ready=false; v.expanded=true; v.need_expand=false;   // no children, no post-knot state
    return;
  }

	// Theta-independent geometry of this path: from the cache when this
	// engine keeps one and has seen the path (a previous run at another
	// theta), otherwise derived here exactly as before and stored.
	const bool use_cache = path_cache_enabled && !random_cuts;
	std::string key;
	std::shared_ptr<const PathCacheEntry> hit;
	if(use_cache){
		key = path_key(trie, v);
		hit = path_cache_get(key);
	}
	std::shared_ptr<PathCacheEntry> fresh;
	if(use_cache && !hit){ fresh = std::make_shared<PathCacheEntry>(); }

	// Knot selection and permutation.  The node rows Xn are
	// only gathered when something reads them: the selection itself (a
	// miss) or the dense arrival leaf of a small PP node; every other
	// consumer gathers its rows straight from X through idx, which yields
	// the very same doubles Xn.row(.) would have copied.
	const bool arrival_dense = (baseline=="PP" && n<=r_knot);
	Eigen::MatrixXd Xn;
	if(!hit || arrival_dense){
		Xn.resize(n, dim);
		for(int i=0; i<n; ++i){ Xn.row(i) = X.row(idx[i]); }
	}
	Eigen::VectorXi ind_knot, ind_res;
	if(hit){
		ind_knot = hit->ind_knot;
		ind_res = hit->ind_res;
	}else{
		pick_knots(Xn, v.box_min, v.box_max, ind_knot, ind_res, knot_count(n));
	}
	const int rk = ind_knot.size();
	const int n_res = ind_res.size();

	Eigen::MatrixXd Kx(rk, dim), Xres(n_res, dim);
	for(int i=0; i<rk; ++i){ Kx.row(i) = X.row(idx[ind_knot(i)]); }
	for(int i=0; i<n_res; ++i){ Xres.row(i) = X.row(idx[ind_res(i)]); }
	Eigen::VectorXd knot_data(rk), zres(n_res);
	for(int i=0; i<rk; ++i){ knot_data(i) = resid(ind_knot(i)); }
	for(int i=0; i<n_res; ++i){ zres(i) = resid(ind_res(i)); }
	const int q = Q.rows();
	Eigen::MatrixXd Qk(q, rk), Qx(q, n_res);
	for(int i=0; i<rk; ++i){ Qk.col(i) = Q.col(ind_knot(i)); }
	for(int i=0; i<n_res; ++i){ Qx.col(i) = Q.col(ind_res(i)); }

	// The knot block of this node (knot_step): kernels, Cholesky factor,
	// Qnext = [Qx; L^-1 B'], Ry = L^-1 knot_data and the residualised values.
	// The factors are shared with the arrival (stop) evaluation of a PP node
	// with n > r below: its leaf selects the same knots by the same pick_knots
	// call on the same rows (knot_count(n) = r there), so the leaf block is
	// evaluated on these factors instead of repeating the selection and the
	// kernels; a small PP node is the dense block on Xn; WhiteNoise reads the
	// residuals only.
	KnotBlock kb;
	if(!knot_step(Kx, Xres, knot_data, zres, Qk, Qx, KNOT_RESIDUALIZE, kb)){
		record_cholesky_error();
		return;
	}
	Eigen::MatrixXd& Vchol = kb.L;
	Eigen::MatrixXd& Qnext = kb.Qnext;
	Eigen::VectorXd& Ry = kb.Ry;
	Eigen::VectorXd& zres_c = kb.zres_c;

	if(baseline=="PP"){
		if(arrival_dense){
			v.loglik_base = leaf_pp_loglik(Xn, resid, Q, v.box_min, v.box_max, v.stat_leaf);
		}else{
			v.loglik_base = leaf_pp_loglik_from_factors(n, true,
				Vchol, Qnext.bottomRows(rk), Ry, zres_c, Qx, v.stat_leaf);
		}
	}else if(random_cuts && v.parent>=0){
		// Integrated WN random cuts: the arrival value was set by child_cell /
		// child_random to the parent's candidate evaluation (the prefix/suffix
		// Welford sweep), and that is the number the particles' partial-tree
		// likelihood already carries.  Keep it, so the increment at this node
		// (log Phi* - loglik_base) subtracts exactly what was added; a
		// two-pass re-evaluation of the same rows would differ in the last
		// bits.  The root has no parent and is evaluated here as before.
		v.stat_leaf.clear();
	}else{
		v.loglik_base = terminal_loglik(Xn, resid, Q, v.box_min, v.box_max, v.stat_leaf);
	}

	// The internal knot block.
	v.stat_knot = kb.stat;
	v.loglik_knot = restree_loglik_conditional(v.stat_knot.ld, v.stat_knot.qd,
		v.stat_knot.n, covpar.sig2);

	// Ineligible terminal: PP at its cap, or WN with <= r rows at/beyond
	// its initial boundary. Oversized WN nodes still prepare the proposal.
	if(!compare_node(v.level,n)){
		v.rho = 1.0; v.rho_log = 0.0;
		v.cuts.resize(0);
		v.child_loglik.resize(0); v.child_loglik_lr.resize(0, 2);
		v.child_stats.clear(); v.child_n.resize(0, 2);
		v.expanded = true;
		v.need_expand = false;
		return;
	}
	if(random_cuts){
		// Random cuts.  Draw-time state: WN integrates the cut prior into its
		// proposal table here; PP keeps the workspace its particle-specific
		// candidates read (rows, sorted columns).  Post-knot state (gidx,
		// zres_c, Qnext) for both leaf models: the next level's expand_level
		// builds every realised child's (idx, resid, Q) from it by row
		// selection, so a node's knot step is computed once per run instead
		// of once per deeper level through descend().  Stored without a
		// budget -- the numbers are the ones descend() would recompute, so
		// nothing depends on whether they are cached -- except where nothing
		// can ever read them (WN at the last level, below).
		if(baseline=="WhiteNoise"){
			prepare_wn_cut_proposal(v, Xres, zres_c);
		}else{
			v.cut_workspace.sorted.resize(n_res, dim);
			for(int J=0; J<dim; J++){
				Eigen::VectorXd col = Xres.col(J);
				std::sort(col.data(), col.data()+col.size());
				v.cut_workspace.sorted.col(J) = col;
			}
			v.cut_workspace.input = std::move(Xres);   // the node keeps its rows; nothing below reads Xres
		}
		// When no possible WN child is eligible (the next level reaches the
		// boundary and at most r residual rows remain), the children are
		// terminal (ACT_BOTTOM) and never expanded, so no
		// expand_level / descend ever reads this node's post-knot state.  WN
		// draws from proposals.wn alone (draw_integrated_wn_cut_move), so
		// its post-knot state is dropped here -- the numbers it would hold are
		// never read.  PP's per-particle candidate draws (draw_random_cut_move,
		// draw_multi_cut_move, pp_eval_child_pair) read postknot.residual / postknot.coefficients /
		// cut_workspace.input of the node at draw time, so PP keeps it at every level.
		// (The nested design never reaches here: it returned above.)  The node
		// is listed in level_cache.current.paths either way -- the end-of-level
		// loop in run() releases its draw-time state / proposal through that
		// list -- but accounts 0 bytes.
		// A boundary child can still compare if it receives > r residual rows.
		const bool store_postknot = baseline!="WhiteNoise" || compare_node(v.level+1,n_res);
		size_t bytes = 0;
		if(store_postknot){
			v.postknot.row_ids.resize(n_res);
			for(int i=0; i<n_res; ++i){ v.postknot.row_ids[i] = idx[ind_res(i)]; }
			v.postknot.residual = std::move(zres_c);
			v.postknot.coefficients = std::move(Qnext);
			v.postknot.ready = true;
		}else{
			v.postknot.row_ids.clear(); v.postknot.row_ids.shrink_to_fit();
			v.postknot.residual.resize(0); v.postknot.coefficients.resize(0,0);
			v.postknot.ready = false;
		}
		// Accounted: the post-knot state and the PP draw-time workspace (the
		// level's draws read both; the workspace is released at the level end
		// and the post-knot state carried to the next level's expansion, see
		// SMC::run, which recounts what it carries).
		bytes = cache_bytes_of(v);
		#ifdef _OPENMP
		#pragma omp critical(restree_cache_list)
		#endif
		{
			worker_error.run("cache-list",[&]{
				trie.level_cache.current.paths.push_back((int)(&v - trie.data()));
				trie.level_cache.current.bytes += bytes;
			});
		}
		worker_error.rethrow();
		v.expanded = true;
		v.need_expand = false;
		return;
	}

	// Deterministic cut methods.  The cuts are a function of
	// the path (the box, or the residual rows), so a cache hit copies them.
	v.cuts.resize(dim);
	if(hit){
		v.cuts = hit->cuts;
	}else{
		for(int J=0; J<dim; J++){
			if(cut_method=="median" && n_res>0){
				Eigen::VectorXd col = Xres.col(J);
				const int nv = col.size(), mid = nv/2;
				if(baseline=="PP"){
					std::nth_element(col.data(), col.data()+mid, col.data()+nv);
					if(col(mid)==0.0){
						// Equivalent +0/-0 values can be reordered by selection.
						// Re-sort the ORIGINAL column to retain the old cut bits.
						col = Xres.col(J);
						std::sort(col.data(), col.data()+nv);
					}else if(nv%2==0){
						col(mid-1) = *std::max_element(col.data(), col.data()+mid);
					}
				}else{
					std::sort(col.data(), col.data()+nv);
				}
				v.cuts(J) = (nv%2==0) ? 0.5*(col(nv/2-1)+col(nv/2)) : col(nv/2);
			}else{ // "middle"; also the median of an EMPTY residual set (adaptive
				// small-node split: all observations became knots, both children
				// are empty whatever the cut, so any finite cut is equivalent)
				v.cuts(J) = (v.box_min(J) + v.box_max(J)) / 2.0;
			}
		}
	}

	// Candidate children for every dimension.  The PP
	// leaf of a candidate child selects its own knots (leaf_pp_loglik ->
	// pick_knots on the child rows): a path function too, read from the
	// cache on a hit and recorded on a miss ([2*J+side] slots are disjoint
	// across the parallel J loop; the entry is published after it).
	v.child_loglik.resize(dim);
	v.child_loglik_lr.resize(dim, 2);
	v.child_stats.assign(2*dim, BlockStats());
	v.child_n.resize(dim, 2);
	const bool pp_leaf = (baseline=="PP");
	if(fresh){ fresh->child_kn.assign((size_t)2*dim, Eigen::VectorXi()); }
	// The row partition of every dimension first (serial, O(d n_res)), then
	// the 2d candidate leaves as independent work items: each (J, side)
	// writes only its own child_loglik_lr / child_stats / child_kn slot, so
	// the parallel loop runs over 2d items instead of d and keeps every
	// thread busy when d is small.  The arithmetic per leaf is unchanged.
	std::vector<Eigen::VectorXi> ind_side((size_t)2*dim);
	for(int J=0; J<dim; J++){
		const double delta = v.cuts(J);
		int nl=0, nr=0;
		Eigen::VectorXi ind_left(n_res), ind_right(n_res);
		for(int i=0; i<n_res; i++){
			if(Xres(i,J) < delta){ ind_left(nl++) = i; }
			else{ ind_right(nr++) = i; }
		}
		v.child_n(J,0) = nl; v.child_n(J,1) = nr;
		ind_side[(size_t)2*J]   = ind_left.head(nl);
		ind_side[(size_t)2*J+1] = ind_right.head(nr);
	}
	// The 2d candidate leaves as independent work items, each writing only its
	// own child_loglik_lr / child_stats / child_kn slot, so serial or parallel
	// they produce identical numbers.  Outside any OpenMP region (the root
	// expansion) they form a parallel loop.  Inside expand_level's task team
	// (every level > 0 at ncores > 1) a nested team is never forked -- where
	// the runtime allows nested teams, ncores concurrent expansions would fork
	// ncores^2 threads for no speed-up and, on glibc, a resident-memory
	// blow-up; instead the PP items are spawned as tasks of the CURRENT team,
	// so its idle threads take them whenever the frontier of distinct paths is
	// narrower than ncores (typically after a resampling collapse).  WN items
	// (a Welford sum each) are too small to be worth a task.
	auto candidate_item = [&](int item){
		const int J = item/2, side = item%2;
		const double delta = v.cuts(J);
		const Eigen::VectorXi& ind_c = ind_side[(size_t)item];
		const int nc = (int)ind_c.size();
		double ll_c = 0.0;
		if(nc>0){
			if(pp_leaf){
				// Score-only leaf on the node's post-knot block through the
				// child's row list (no gathered copy; leaf_pp_loglik_rows).
				Eigen::VectorXd bmin_c = v.box_min, bmax_c = v.box_max;
				if(side==0){ bmax_c(J) = delta; } else { bmin_c(J) = delta; }
				const Eigen::VectorXi* kn_in =
					(hit && (int)hit->child_kn.size()==2*dim) ?
					&hit->child_kn[(size_t)(2*J+side)] : nullptr;
				Eigen::VectorXi* kn_out =
					fresh ? &fresh->child_kn[(size_t)(2*J+side)] : nullptr;
				ll_c = leaf_pp_loglik_rows(Xres, zres_c, Qnext, ind_c.data(), nc,
					bmin_c, bmax_c, v.child_stats[2*J+side], kn_in, kn_out);
			}else{
				// WhiteNoise terminals read the residuals only (no rows, no
				// coefficients are gathered for them).
				Eigen::VectorXd zc(nc);
				for(int i=0; i<nc; ++i){ zc(i) = zres_c(ind_c(i)); }
				v.child_stats[2*J+side].clear();
				ll_c = wn_leaf_loglik(zc);
			}
		}
		v.child_loglik_lr(J,side) = ll_c;
	};
	#ifdef _OPENMP
	const bool heavy_items = ncores>1 && n_res>4*r_knot;
	if(heavy_items && pp_leaf && omp_in_parallel()){
		for(int item=0; item<2*dim; item++){
			#pragma omp task default(shared) firstprivate(item)
			worker_error.run("deterministic-candidate",[&]{ candidate_item(item); });
		}
		#pragma omp taskwait
		worker_error.rethrow();
	}else{
		parallel_for(worker_error,"deterministic-candidate",2*dim,
			heavy_items ? ncores : 1,candidate_item);
	}
	#else
	for(int item=0; item<2*dim; item++){ candidate_item(item); }
	#endif
	for(int J=0; J<dim; J++){
		const int nl = v.child_n(J,0), nr = v.child_n(J,1);
		const double ll_l = v.child_loglik_lr(J,0), ll_r = v.child_loglik_lr(J,1);
		if(nl==0 && nr==0){ v.child_loglik(J) = 0.0; }
		else if(nl==0){ v.child_loglik(J) = ll_r; }
		else if(nr==0){ v.child_loglik(J) = ll_l; }
		else{ v.child_loglik(J) = ll_l + ll_r; }
	}

	if(fresh){
		fresh->ind_knot = ind_knot;
		fresh->ind_res = ind_res;
		fresh->cuts = v.cuts;
		fresh->bytes = path_cache_entry_bytes(*fresh, key.size());
		path_cache_put(key, std::move(fresh));
	}

	// Stop / split posterior of the node.
	node_posterior(v);

	// Deterministic cuts: keep the post-knot state so the next level's
	// expand_level can build every realized child's (idx, resid, Q) by pure
	// selection instead of re-deriving this node's knots, kernels and
	// Cholesky through descend().  The stored numbers are exactly the ones
	// descend() would recompute (same code on the same inputs), so the
	// children's expansions are bit-identical either way. Cache when a child
	// can be eligible; a level whose caches would exceed
	// the budget is left uncached (expand_level then falls back to descend).
	if(compare_node(v.level+1,n_res)){
		const size_t bytes = (size_t)n_res * (sizeof(int) + sizeof(double))
			+ (size_t)(q+rk) * (size_t)n_res * sizeof(double);
		bool do_cache = false;
		#ifdef _OPENMP
		#pragma omp critical(restree_cache_list)
		#endif
		{
			worker_error.run("cache-list",[&]{
			if(trie.level_cache.current.bytes + bytes <= trie.cache_budget_bytes){
				trie.level_cache.current.paths.push_back((int)(&v - trie.data()));
				trie.level_cache.current.bytes += bytes;
				do_cache = true;
			}
			});
		}
		worker_error.rethrow();
		if(do_cache){
			v.postknot.row_ids.resize(n_res);
			for(int i=0; i<n_res; ++i){ v.postknot.row_ids[i] = idx[ind_res(i)]; }
			v.postknot.residual = std::move(zres_c);
			v.postknot.coefficients = std::move(Qnext);
			v.postknot.ready = true;
		}
	}

	v.expanded = true;
	v.need_expand = false;
}

void ResTree::partition_child(const std::vector<int>& gidx, const Eigen::VectorXd& zres_c,
	const Eigen::MatrixXd& Qnext, int J, int side, double cut,
	std::vector<int>& cidx, Eigen::VectorXd& cres, Eigen::MatrixXd& cQ) const{

	const int n_res = (int)gidx.size();
	cidx.clear();
	cidx.reserve(n_res);
	std::vector<int> lpos;
	lpos.reserve(n_res);
	for(int i=0; i<n_res; ++i){
		// X(gidx[i], J) is the very double Xres(i, J) was copied from.
		const bool left = X(gidx[i], J) < cut;
		if((side==0) == left){ cidx.push_back(gidx[i]); lpos.push_back(i); }
	}
	const int nc = (int)cidx.size();
	cres.resize(nc);
	cQ.resize(Qnext.rows(), nc);
	for(int i=0; i<nc; ++i){
		cres(i) = zres_c(lpos[i]);
		cQ.col(i) = Qnext.col(lpos[i]);
	}
}

// A node's re-derived post-knot state (gidx, zres_c, Qnext) when descend()
// cannot read it from the trie's cache.  Heap-held and shared_ptr-owned so
// that the child tasks descend() spawns -- which build their own input from
// it INSIDE the task, after the parent frame has returned -- keep it alive
// until the last of them has partitioned it.
namespace {
struct DescendNodeState {
	std::vector<int> gidx;
	Eigen::VectorXd zres_c;
	Eigen::MatrixXd Qnext;
};
}

// Depth-first re-derivation of (idx, resid, Q) along paths that lead to
// nodes marked need_expand.  Interior nodes repeat only the "shared" part of
// their expansion (knot removal, coefficient append, pp subtraction), which
// is a small fraction of the candidate-split work done once in expand().
void ResTree::descend(PathTrie& trie, int trie_idx, std::vector<int>&& idx,
	Eigen::VectorXd&& resid, Eigen::MatrixXd&& Q) const{

	PathNode& v = trie[trie_idx];
	if(v.need_expand && !v.expanded){
		expand(trie, v, idx, resid, Q);
		return;
	}
	if(!v.expanded){ return; }

	// Which realized children lead to marked nodes?
	std::vector<ChildEdge> need;
	if(random_cuts){
		for(size_t e=0; e<v.child_edges.size(); ++e){
			const ChildEdge& ce = v.child_edges[e];
			if(ce.idx>=0 && trie.path_stamp[ce.idx]==trie.stamp){ need.push_back(ce); }
		}
	}else{
		for(int s=0; s<(int)v.child_path.size(); ++s){
			int c = v.child_path[s];
			if(c>=0 && trie.path_stamp[c]==trie.stamp){
				ChildEdge ce;
				ce.J = (int8_t)(s/2); ce.side = (int8_t)(s%2);
				ce.cut = v.cuts(s/2); ce.idx = c;
				need.push_back(ce);
			}
		}
	}
	if(need.empty()){ return; }

	// The node's post-knot state: (gidx, zres_c, Qnext) -- global ids of the
	// residual rows in node order, the residual after the pp_curr
	// subtraction, and the propagated coefficients.  Under deterministic cuts
	// expand() may still hold exactly these numbers from this node's own
	// expansion; otherwise they are re-derived by the same code.
	//
	// Ownership: in the cached case the three arrays live in the trie, which
	// is not resized during expand_level and whose caches are released only
	// after the level (serial context, after the parallel region's barrier),
	// so they outlive every task spawned below.  In the re-derived case they
	// live in a shared_ptr-owned heap block: each child task holds a copy of
	// the shared_ptr and the block is freed when the last task has built its
	// input from it (the parent's own copy goes out of scope with this frame).
	std::shared_ptr<DescendNodeState> own_state;
	const std::vector<int>* gidx_p = nullptr;
	const Eigen::VectorXd* zres_p = nullptr;
	const Eigen::MatrixXd* Qnext_p = nullptr;
	if(v.postknot.ready){
		gidx_p = &v.postknot.row_ids;
		zres_p = &v.postknot.residual;
		Qnext_p = &v.postknot.coefficients;
	}else{
		own_state = std::make_shared<DescendNodeState>();
		std::vector<int>& gidx = own_state->gidx;
		Eigen::VectorXd& zres_c = own_state->zres_c;
		Eigen::MatrixXd& Qnext = own_state->Qnext;
		gidx_p = &own_state->gidx;
		zres_p = &own_state->zres_c;
		Qnext_p = &own_state->Qnext;
		const int n = (int)idx.size();

		// The knot selection of this (already expanded) path: from the path
		// cache when the engine keeps one, else pick_knots on the node rows.
		Eigen::VectorXi ind_knot, ind_res;
		std::shared_ptr<const PathCacheEntry> hit;
		if(path_cache_enabled && !random_cuts){
			hit = path_cache_get(path_key(trie, v));
		}
		if(hit){
			ind_knot = hit->ind_knot;
			ind_res = hit->ind_res;
		}else if(nested()){
			// nested design: the parent's candidates that reach this node
			nested_node_knots(idx, v.parent>=0 ? &trie[v.parent].nested_cand : nullptr,
				v.box_min, v.box_max, ind_knot, ind_res);
		}else{
			Eigen::MatrixXd Xn(n, dim);
			for(int i=0; i<n; ++i){ Xn.row(i) = X.row(idx[i]); }
			pick_knots(Xn, v.box_min, v.box_max, ind_knot, ind_res, knot_count(n));
		}
		const int rk = ind_knot.size();
		const int n_res = ind_res.size();
		Eigen::MatrixXd Kx(rk, dim), Xres(n_res, dim);
		for(int i=0; i<rk; ++i){ Kx.row(i) = X.row(idx[ind_knot(i)]); }
		for(int i=0; i<n_res; ++i){ Xres.row(i) = X.row(idx[ind_res(i)]); }
		Eigen::VectorXd knot_data(rk), zres(n_res);
		for(int i=0; i<rk; ++i){ knot_data(i) = resid(ind_knot(i)); }
		for(int i=0; i<n_res; ++i){ zres(i) = resid(ind_res(i)); }
		const int q = Q.rows();
		Eigen::MatrixXd Qk(q, rk), Qx(q, n_res);
		for(int i=0; i<rk; ++i){ Qk.col(i) = Q.col(ind_knot(i)); }
		for(int i=0; i<n_res; ++i){ Qx.col(i) = Q.col(ind_res(i)); }

		// The same knot step expand() took on this node (knot_step): the
		// children's inputs are then the identical numbers.
		KnotBlock kb;
		const bool step_ok = nested() ?
			nested_knot_step(Kx, Xres, knot_data, zres, Qk, Qx, KNOT_RESIDUALIZE, kb) :
			knot_step(Kx, Xres, knot_data, zres, Qk, Qx, KNOT_RESIDUALIZE, kb);
		if(!step_ok){
			record_cholesky_error();
			return;
		}
		Qnext = std::move(kb.Qnext);
		zres_c = std::move(kb.zres_c);

		// Global row ids of the residual rows, in node order.
		gidx.resize(n_res);
		for(int i=0; i<n_res; ++i){ gidx[i] = idx[ind_res(i)]; }
	}
	// Free what the children do not need before recursing.
	idx.clear(); idx.shrink_to_fit(); resid.resize(0); Q.resize(0,0);

	#ifdef _OPENMP
	const bool spawn_tasks = omp_in_parallel() && need.size()>1;
	#else
	const bool spawn_tasks = false;
	#endif
	for(size_t u=0; u<need.size(); ++u){
		const int J = need[u].J, side = need[u].side;
		const int child = need[u].idx;
		const double delta = need[u].cut;
		// Sibling subtrees are independent: expansions write to disjoint trie
		// entries and the trie is not resized during expand_level, so descending
		// them as OpenMP tasks is race-free and leaves the results identical
		// for every ncores.
		if(spawn_tasks){
			// The task carries only the edge (J, side, cut) and a handle on the
			// parent's post-knot state; partition_child runs INSIDE the task,
			// so at most one child's (idx, resid, Q) block exists per running
			// task instead of one per sibling from the moment the parent loop
			// spawns them (need.size() copies of the parent's block at once).
			// The payload is heap-allocated and reaches the task through one
			// firstprivate pointer (no deep copies).  trie must be SHARED
			// explicitly: it is a reference parameter, and OpenMP's implicit
			// firstprivate for an unlisted reference in a task would copy the
			// referenced store, discarding the expansions.
			struct ChildInput {
				std::shared_ptr<const DescendNodeState> keep;   // null when the trie's cache is read
				const std::vector<int>* gidx;
				const Eigen::VectorXd* zres;
				const Eigen::MatrixXd* Qnext;
				int J, side;
				double delta;
			};
			ChildInput* in = new ChildInput{own_state, gidx_p, zres_p, Qnext_p, J, side, delta};
			#ifdef _OPENMP
			#pragma omp task shared(trie) firstprivate(child, in)
			#endif
			{
				std::unique_ptr<ChildInput> own(in);
				worker_error.run("descend",[&]{
				std::vector<int> cidx;
				Eigen::VectorXd cres;
				Eigen::MatrixXd cQ;
				partition_child(*own->gidx, *own->zres, *own->Qnext, own->J, own->side, own->delta,
					cidx, cres, cQ);
				// Drop this task's hold on the parent's state before recursing,
				// so the parent's block is freed as soon as the last sibling has
				// partitioned it rather than when the deepest descendant returns.
				own.reset();
				descend(trie, child, std::move(cidx), std::move(cres), std::move(cQ));
				});
			}
		}else{
			std::vector<int> cidx;
			Eigen::VectorXd cres;
			Eigen::MatrixXd cQ;
			partition_child(*gidx_p, *zres_p, *Qnext_p, J, side, delta, cidx, cres, cQ);
			descend(trie, child, std::move(cidx), std::move(cres), std::move(cQ));
		}
	}
	#ifdef _OPENMP
	// Wait for this node's children before the frame returns.  Without the
	// wait the recursion is not a depth-first descent but a task flood: a
	// frame spawns its children and returns, they spawn theirs, and the only
	// bound on how many are outstanding is the runtime's own throttle, which
	// is itself proportional to the thread count (libgomp defers a task until
	// the team holds 64 per thread).  Every outstanding task pins its parent's
	// post-knot block through ChildInput::keep, so the resident set grew with
	// ncores for no gain in concurrency.  With the wait, the live blocks are
	// the ones on the threads' active paths -- O(depth) per path, and siblings
	// share their ancestors -- which is the serial profile at full width.
	//
	// This frame's own handle is dropped first: the tasks hold their copies
	// and each releases its one before recursing, so the block is still freed
	// as soon as the last sibling has partitioned it, not when the deepest
	// descendant returns.
	if(spawn_tasks){
		own_state.reset();
		#pragma omp taskwait
	}
	#endif
}

// ==================== Nested PP model and integrated cut proposals ====================

// Nested PP: exact integration of the uniform / balanced rank-cut prior at every
// node, cell-keyed sharing of children between particles, and the nested
// knot rule that makes the integration affordable.
//
// MODEL.  The root spends r maximin knots on all its rows (as under
// "maximin").  Every other node v inherits its knots from its parent u:
//   K_v = C_u \cap rows(v)                (in the order of C_u)
// where C_u, the CANDIDATE set of u, is the first min(k*r, n_res(u)) residual
// rows of u in the maximin order of u's residual rows.  |K_v| therefore lies
// in 0..k*r (k = nested_cand_factor, default 2) and depends on the cut; nesting
// use across a partition.  A node with n <= r rows keeps the engine's
// small-node convention: all rows are knots (a split then leaves both
// children empty; the leaf is the dense block).  The leaf of a node with
// n > r rows is the modified predictive process on K_v (diagonal remainder),
// exactly as leaf_pp_loglik_core forms it for maximin knots.
//
// WHY IT IS CHEAP.  For a candidate split of u on coordinate J, the left child
// of rank cell k is the set of the k+1 smallest residual rows in x_J and its
// knots are C_u restricted to that set: sweeping k upward only ever APPENDS
// rows -- a residual row costs one triangular solve, O(|K|^2); a knot extends
// the Cholesky factor by one row, O(|K|^2), and gives every residual row one
// more coefficient, O(|K| n). With at most k*r knot appends, all n_res - 1 cell
// values of one (J, side) cost O(n_res (k*r)^2), against O(n_res^2 r^2) for
// re-selecting maximin knots inside every cell.  The right child is the same
// sweep from the top.  The sweep over all cells of every coordinate is done
// ONCE per distinct node path and shared by every particle there, as
// prepare_wn_cut_proposal does for WN; particles draw (stop | J, cell) from
// the exact node posterior with a constant weight increment log Phi* -
// loglik_base, and children are keyed by (J, side, cell) (PathTrie::child_cell).
//
// EVERY ROUTE AGREES.  The fixed-structure evaluator (loglik_subtree) and
// the predictor (predict_subtree) apply the same rule through
// nested_node_knots / nested_candidates; the sweep's incremental arithmetic
// and their direct block factorisations agree to round-off (the same
// quantities in another operation order). restree_loglik(verify = TRUE)
// checks their agreement at 1e-6.
//
// expand() dispatches here; in SMC.cpp, draw_random_cut_move() dispatches to
// draw_nested_pp_cut_move(), and commit_random_cut_move() treats the move as
// an integrated one (out.integrated_wn) with the cell's block statistics.

/*============================ the nested knot rule ============================*/

// Knots of a node under the nested rule: the root (parent_cand == nullptr) and
// small nodes (n <= r) select by pick_knots exactly as every other design
// does (maximin r, or all rows); any other node takes its parent's candidate
// rows that reach it, in candidate order.  ind_res is the ascending
// complement, the convention of pick_knots / knot_complement.
void ResTree::nested_node_knots(const std::vector<int>& idx,
	const std::vector<int>* parent_cand, const Eigen::VectorXd& bmin,
	const Eigen::VectorXd& bmax, Eigen::VectorXi& ind_knot,
	Eigen::VectorXi& ind_res) const{

	const int n = (int)idx.size();
	if(parent_cand==nullptr || n<=r_knot){
		Eigen::MatrixXd Xn(n, dim);
		for(int i=0; i<n; ++i){ Xn.row(i) = X.row(idx[(size_t)i]); }
		pick_knots(Xn, bmin, bmax, ind_knot, ind_res, knot_count(n));
		return;
	}
	// position of every row id in idx (a node's rows are distinct), so each
	// candidate is located in O(1) instead of a scan over the node's rows
	std::unordered_map<int,int> where;
	where.reserve((size_t)n);
	for(int i=0; i<n; ++i){ where.emplace(idx[(size_t)i], i); }
	std::vector<int> kpos;
	kpos.reserve(parent_cand->size());
	std::vector<char> taken((size_t)n, 0);
	for(size_t a=0; a<parent_cand->size(); ++a){
		const auto it = where.find((*parent_cand)[a]);
		if(it!=where.end()){ kpos.push_back(it->second); taken[(size_t)it->second] = 1; }
	}
	ind_knot.resize((int)kpos.size());
	for(size_t a=0; a<kpos.size(); ++a){ ind_knot((int)a) = kpos[a]; }
	ind_res.resize(n - (int)kpos.size());
	int c = 0;
	for(int i=0; i<n; ++i){ if(!taken[(size_t)i]){ ind_res(c++) = i; } }
}

// The candidate set: the first min(nested_cand_factor*r, n_res)
// rows in the maximin order of Xres (seq_maximin), as positions into Xres in
// selection order.
std::vector<int> ResTree::nested_candidates(const Eigen::MatrixXd& Xres) const{
	const int m = (int)Xres.rows();
	const int rc = static_cast<int>(std::min<long long>(
    static_cast<long long>(std::max(1, nested_cand_factor)) * r_knot, m));
	std::vector<int> cand;
	if(rc<=0){ return cand; }
	Eigen::VectorXi bk(rc), br(m-rc);
	seq_maximin(Xres, rc, bk, br);
	cand.resize((size_t)rc);
	for(int a=0; a<rc; ++a){ cand[(size_t)a] = bk(a); }
	return cand;
}

// knot_step that also accepts an EMPTY knot set (a node none of whose
// parent's candidates reached): the block is void, the residual rows pass
// through unchanged.
bool ResTree::nested_knot_step(const Eigen::MatrixXd& Kx, const Eigen::MatrixXd& Xres,
	const Eigen::VectorXd& knot_data, const Eigen::VectorXd& zres,
	const Eigen::MatrixXd& Qk, const Eigen::MatrixXd& Qx,
	int level, KnotBlock& kb) const{

	if(Kx.rows()>0){ return knot_step(Kx, Xres, knot_data, zres, Qk, Qx, level, kb); }
	const int n_res = (int)Xres.rows();
	kb.L.resize(0,0); kb.B.resize(n_res, 0); kb.Ry.resize(0);
	kb.stat.clear();
	if(level==KNOT_PROPAGATE){ kb.W.resize(0, n_res); }
	if(level>=KNOT_RESIDUALIZE){ kb.Qnext = Qx; kb.zres_c = zres; }
	return true;
}

/*========================= the all-cell prefix sweep =========================*/

namespace {

// Node-level precomputation shared by every sweep of one node.
struct SweepInput {
	const Eigen::MatrixXd* Xres;    // n_res x d residual rows (node order)
	const Eigen::MatrixXd* Qnext;   // (q + r_k) x n_res propagated coefficients
	const Eigen::VectorXd* z;       // residualised values
	const Eigen::MatrixXd* Ball;    // n_res x r_c: k(x_i, x_c) - Q_i'Q_c for every row i and candidate c
	const Eigen::MatrixXd* Vcc;     // r_c x r_c: Ball on the candidate rows + nugget I
	const Eigen::VectorXd* dbase;   // 1 + nugget - |Q_i|^2
	const Eigen::VectorXi* crank;   // candidate rank of row i, -1 if not a candidate
	const std::vector<int>* cand_rows; // row position of candidate rank a
	const GPM* cp;                  // covariance parameters (the model's)
	double nugget;
	int r;        // dense threshold: leaves with at most r rows are dense
	int rc_max;   // largest possible knot count (candidates per node)
	double sig2;
};

// Leaf value and Gaussian block statistics of every prefix of `seq`: out[j]
// describes the set {seq[0], ..., seq[j-1]} as a terminal (j = 0 empty).
//   j <= r : dense block, every row a knot (incremental Cholesky in sequence
//            order)
//   j > r  : nested PP leaf, knots = candidates in the set (in candidate
//            order at the restart, then in arrival order), residual rows
//            in the diagonal Schur remainder
// A factorisation failure (a non-positive pivot, or a residual variance
// below -1e-9) makes that prefix and every later one -inf: such cells carry
// no proposal mass and are never realised.
void nested_sweep(const SweepInput& P, const std::vector<int>& seq,
	std::vector<double>& ll, std::vector<BlockStats>& st){

	const int m = (int)seq.size();
	const int r = P.r;
	const int qq = (int)P.Qnext->rows();
	ll.assign((size_t)m+1, 0.0);
	st.assign((size_t)m+1, BlockStats());
	if(m==0){ return; }
	const double neg_inf = -std::numeric_limits<double>::infinity();
	bool valid = true;
	auto fail_from = [&](int j){
		for(int t=j; t<=m; ++t){ ll[(size_t)t] = neg_inf; st[(size_t)t].clear(); }
	};

	// ---- dense phase: rows seq[0..rd-1], all knots ----
	const int rd = std::min(m, r);
	Eigen::MatrixXd Kd;
	{
		Eigen::MatrixXd Xd(rd, P.Xres->cols());
		for(int j=0; j<rd; ++j){ Xd.row(j) = P.Xres->row(seq[(size_t)j]); }
		Kd = ikernel(Xd, Xd, *P.cp);
		if(qq>0){
			Eigen::MatrixXd Qd(qq, rd);
			for(int j=0; j<rd; ++j){ Qd.col(j) = P.Qnext->col(seq[(size_t)j]); }
			sym_downdate(Kd, Qd);
		}
		Kd.diagonal().array() += P.nugget;
	}
	const int rmax = std::max(r, P.rc_max);
	Eigen::MatrixXd L = Eigen::MatrixXd::Zero(rmax, rmax);   // knot factor (at most rmax knots)
	Eigen::VectorXd u = Eigen::VectorXd::Zero(rmax);         // L^-1 z_K
	int mk = 0;                                        // current knot count
	double ld_K = 0.0, qd_K = 0.0;
	for(int j=0; j<rd; ++j){
		// append row seq[j] as knot number mk
		double ljj2 = Kd(j,j);
		if(mk>0){
			Eigen::VectorXd v = Kd.row(j).head(mk).transpose();
			Eigen::VectorXd l = L.topLeftCorner(mk,mk).triangularView<Eigen::Lower>().solve(v);
			ljj2 -= l.squaredNorm();
			L.row(mk).head(mk) = l.transpose();
			u(mk) = (*P.z)(seq[(size_t)j]) - l.dot(u.head(mk));
		}else{
			u(0) = (*P.z)(seq[(size_t)j]);
		}
		if(!(ljj2>0.0) || !std::isfinite(ljj2)){ valid = false; fail_from(j+1); break; }
		const double ljj = std::sqrt(ljj2);
		L(mk,mk) = ljj;
		u(mk) /= ljj;
		ld_K += 2.0*std::log(ljj);
		qd_K += u(mk)*u(mk);
		++mk;
		st[(size_t)(j+1)].ld = ld_K; st[(size_t)(j+1)].qd = qd_K; st[(size_t)(j+1)].n = j+1;
		ll[(size_t)(j+1)] = restree_loglik_conditional(ld_K, qd_K, j+1, P.sig2);
	}
	if(!valid || m<=r){ return; }

	// ---- restart at j = r+1: knots = candidates among seq[0..r], residual rows the rest ----
	std::vector<int> K;          // candidate ranks of the current knots, in knot order
	std::vector<int> R;          // residual rows (positions into Xres), in arrival order
	K.reserve((size_t)rmax); R.reserve((size_t)m);
	{
		std::vector<int> present;
		for(int j=0; j<=r; ++j){
			const int i = seq[(size_t)j];
			if((*P.crank)(i)>=0){ present.push_back((*P.crank)(i)); } else { R.push_back(i); }
		}
		std::sort(present.begin(), present.end());
		K = present;
	}
	mk = (int)K.size();
	L.setZero(); u.setZero();
	ld_K = 0.0; qd_K = 0.0;
	if(mk>0){
		Eigen::MatrixXd Vk(mk, mk);
		for(int a=0; a<mk; ++a){ for(int b=0; b<mk; ++b){ Vk(a,b) = (*P.Vcc)(K[(size_t)a], K[(size_t)b]); } }
		Eigen::LLT<Eigen::MatrixXd> llt(Vk);
		if(llt.info()!=Eigen::Success){ fail_from(r+1); return; }
		Eigen::MatrixXd Lk = llt.matrixL();
		if(!Lk.allFinite() || (Lk.diagonal().array()<=0.0).any()){ fail_from(r+1); return; }
		L.topLeftCorner(mk,mk) = Lk;
		Eigen::VectorXd zk(mk);
		for(int a=0; a<mk; ++a){ zk(a) = (*P.z)((*P.cand_rows)[(size_t)K[(size_t)a]]); }
		u.head(mk) = Lk.triangularView<Eigen::Lower>().solve(zk);
		ld_K = 2.0*Lk.diagonal().array().log().sum();
		qd_K = u.head(mk).squaredNorm();
	}
	// residual rows: coefficients w_i = L^-1 b_i (stored column-wise, r x n_R),
	// remainder variance d_i and mean-corrected value e_i
	Eigen::MatrixXd Wm = Eigen::MatrixXd::Zero(rmax, m);
	// per residual row: remainder variance d, mean-corrected value e, and
	// their current contributions (log d, e^2/d) so a knot append updates a
	// row's terms without recomputing the old log
	std::vector<double> dv, ev, ldv, qdv;
	dv.reserve((size_t)m); ev.reserve((size_t)m); ldv.reserve((size_t)m); qdv.reserve((size_t)m);
	double ld_R = 0.0, qd_R = 0.0;
	auto contrib = [&](double d, double e, double& ldc, double& qdc){
		const double df = std::max(d, 1e-12);
		ldc = std::log(df); qdc = e*e/df;
	};
	// row work buffers, allocated once per sweep (the head(mk) segments start
	// at the buffers' first element, as the per-row vectors did)
	Eigen::VectorXd wbuf(std::max(1, rmax)), bbuf(std::max(1, rmax));
	auto add_residual = [&](int i)->bool{
		auto w = wbuf.head(mk);
		if(mk>0){
			auto b = bbuf.head(mk);
			for(int a=0; a<mk; ++a){ b(a) = (*P.Ball)(i, K[(size_t)a]); }
			w = L.topLeftCorner(mk,mk).triangularView<Eigen::Lower>().solve(b);
		}
		const double d = (*P.dbase)(i) - w.squaredNorm();
		const double e = (*P.z)(i) - w.dot(u.head(mk));
		if(!std::isfinite(d) || !std::isfinite(e) || d <= -1e-9){ return false; }
		const int slot = (int)dv.size();
		Wm.col(slot).head(mk) = w;
		dv.push_back(d); ev.push_back(e);
		double ldc, qdc; contrib(d, e, ldc, qdc);
		ldv.push_back(ldc); qdv.push_back(qdc);
		ld_R += ldc; qd_R += qdc;
		return true;
	};
	for(size_t t=0; t<R.size(); ++t){
		if(!add_residual(R[t])){ fail_from(r+1); return; }
	}
	st[(size_t)(r+1)].ld = ld_K + ld_R; st[(size_t)(r+1)].qd = qd_K + qd_R; st[(size_t)(r+1)].n = r+1;
	ll[(size_t)(r+1)] = restree_loglik_conditional(ld_K + ld_R, qd_K + qd_R, r+1, P.sig2);

	// ---- append rows seq[r+1..m-1] ----
	for(int j=r+1; j<m; ++j){
		const int i = seq[(size_t)j];
		const int rank = (*P.crank)(i);
		if(rank>=0){
			// knot append: column of Vcc on the current knots
			double ljj2 = (*P.Vcc)(rank, rank);
			Eigen::VectorXd l(mk);
			if(mk>0){
				Eigen::VectorXd v(mk);
				for(int a=0; a<mk; ++a){ v(a) = (*P.Vcc)(K[(size_t)a], rank); }
				l = L.topLeftCorner(mk,mk).triangularView<Eigen::Lower>().solve(v);
				ljj2 -= l.squaredNorm();
			}
			if(!(ljj2>0.0) || !std::isfinite(ljj2)){ fail_from(j+1); return; }
			const double ljj = std::sqrt(ljj2);
			const double unew = ((*P.z)(i) - (mk>0 ? l.dot(u.head(mk)) : 0.0)) / ljj;
			L.row(mk).head(mk) = l.transpose();
			L(mk,mk) = ljj;
			u(mk) = unew;
			ld_K += 2.0*std::log(ljj);
			qd_K += unew*unew;
			// every residual row gains one coefficient
			const int nR = (int)dv.size();
			for(int s=0; s<nR; ++s){
				const int irow = R[(size_t)s];
				const double wnew = ((*P.Ball)(irow, rank) - (mk>0 ? l.dot(Wm.col(s).head(mk)) : 0.0)) / ljj;
				const double ldo = ldv[(size_t)s], qdo = qdv[(size_t)s];   // the row's current terms
				dv[(size_t)s] -= wnew*wnew;
				ev[(size_t)s] -= wnew*unew;
				if(!std::isfinite(dv[(size_t)s]) || dv[(size_t)s] <= -1e-9){ fail_from(j+1); return; }
				double ldn, qdn; contrib(dv[(size_t)s], ev[(size_t)s], ldn, qdn);
				ld_R += ldn - ldo; qd_R += qdn - qdo;
				ldv[(size_t)s] = ldn; qdv[(size_t)s] = qdn;
				Wm(mk, s) = wnew;
			}
			K.push_back(rank);
			++mk;
		}else{
			R.push_back(i);
			if(!add_residual(i)){ fail_from(j+1); return; }
		}
		st[(size_t)(j+1)].ld = ld_K + ld_R; st[(size_t)(j+1)].qd = qd_K + qd_R; st[(size_t)(j+1)].n = j+1;
		ll[(size_t)(j+1)] = restree_loglik_conditional(ld_K + ld_R, qd_K + qd_R, j+1, P.sig2);
	}
}


} // namespace

/*==================== the integrated proposal of a nested PP node ====================*/

// The rank-cell table of prepare_wn_cut_proposal with PP leaf values from
// the sweeps: for every coordinate the prefix values (left children) and the
// suffix values (right children) of the sorted residual rows, one interval
// per rank cell with the WN tie rule (a flat segment is a point mass; ties go
// right), masses from the cut prior, the log-mass binary tree for O(log)
// draws, and the node's stop/split masses.
void ResTree::prepare_nested_cut_proposal(PathNode& v, const Eigen::MatrixXd& Xres,
	const Eigen::VectorXd& zres_c, const Eigen::MatrixXd& Qnext,
	const std::vector<int>& cand_pos) const{

	auto prop = std::make_shared<NestedCutProposal>();
	WNCutProposal& tab = prop->table;
	const int m = (int)Xres.rows(), gaps = std::max(1, m-1);
	const int rc = (int)cand_pos.size();
	const int qq = (int)Qnext.rows();
	tab.n_residual = m;
	tab.intervals.reserve((size_t)dim*gaps);
	prop->left_stat.reserve((size_t)dim*gaps);
	prop->right_stat.reserve((size_t)dim*gaps);
	std::vector<double> log_masses;
	log_masses.reserve((size_t)dim*gaps);

	// ---- node-level precomputation ----
	Eigen::MatrixXd Ball(m, rc), Vcc(rc, rc);
	if(rc>0){
		Eigen::MatrixXd Xc(rc, dim);
		for(int a=0; a<rc; ++a){ Xc.row(a) = Xres.row(cand_pos[(size_t)a]); }
		Ball = ikernel(Xres, Xc, covpar);
		if(qq>0){
			Eigen::MatrixXd Qc(qq, rc);
			for(int a=0; a<rc; ++a){ Qc.col(a) = Qnext.col(cand_pos[(size_t)a]); }
			Ball.noalias() -= Qnext.transpose()*Qc;
		}
		for(int a=0; a<rc; ++a){
			for(int b=0; b<rc; ++b){ Vcc(a,b) = Ball(cand_pos[(size_t)a], b); }
			Vcc(a,a) += covpar.nugget;
		}
	}
	Eigen::VectorXd dbase = Eigen::VectorXd::Constant(m, 1.0 + covpar.nugget);
	if(qq>0){ dbase.array() -= Qnext.array().square().colwise().sum().transpose(); }
	Eigen::VectorXi crank = Eigen::VectorXi::Constant(m, -1);
	for(int a=0; a<rc; ++a){ crank(cand_pos[(size_t)a]) = a; }
	SweepInput P;
	P.Xres = &Xres; P.Qnext = &Qnext; P.z = &zres_c; P.Ball = &Ball; P.Vcc = &Vcc;
	P.dbase = &dbase; P.crank = &crank; P.cand_rows = &cand_pos; P.cp = &covpar; P.nugget = covpar.nugget;
	P.r = r_knot; P.rc_max = rc; P.sig2 = covpar.sig2;

	std::vector<double> pre_ll, suf_ll;
	std::vector<BlockStats> pre_st, suf_st;
	for(int j=0; j<dim; ++j){
		std::vector<int> order((size_t)m);
		std::iota(order.begin(), order.end(), 0);
		std::stable_sort(order.begin(), order.end(), [&](int a, int b){ return Xres(a,j) < Xres(b,j); });
		nested_sweep(P, order, pre_ll, pre_st);
		std::vector<int> rev(order.rbegin(), order.rend());
		nested_sweep(P, rev, suf_ll, suf_st);   // suf[c] = the c largest rows
		int first_equal = 0;
		for(int k=0; k<gaps; ++k){
			WNCutInterval interval;
			interval.coordinate = j; interval.rank_index = k;
			double mass = 1.0;
			if(m==0){
				interval.cut_lower = interval.cut_upper = 0.5*(v.box_min(j)+v.box_max(j));
			}else if(m==1){
				interval.cut_lower = interval.cut_upper = Xres(order[0], j);
			}else{
				interval.cut_lower = Xres(order[(size_t)k], j);
				interval.cut_upper = Xres(order[(size_t)k+1], j);
				if(k==0 || Xres(order[(size_t)k], j)!=Xres(order[(size_t)k-1], j)){ first_equal = k; }
				interval.n_left = (interval.cut_lower==interval.cut_upper) ? first_equal : k+1;
				const double a = (double)k/(m-1), b = (double)(k+1)/(m-1);
				mass = (cut_method=="balanced") ? beta22_interval_mass(a, b) : b-a;
			}
			const int nl = interval.n_left, nr = m - nl;
			interval.left_loglik = pre_ll[(size_t)nl];
			interval.right_loglik = suf_ll[(size_t)nr];
			log_masses.push_back(std::log(prior_lambda(j)) + std::log(mass)
				+ interval.left_loglik + interval.right_loglik);
			tab.intervals.push_back(interval);
			prop->left_stat.push_back(pre_st[(size_t)nl]);
			prop->right_stat.push_back(suf_st[(size_t)nr]);
		}
	}
	while(tab.leaf_offset < log_masses.size()){ tab.leaf_offset *= 2; }
	tab.log_mass_tree.assign(2*tab.leaf_offset, -std::numeric_limits<double>::infinity());
	for(size_t k=0; k<log_masses.size(); ++k){ tab.log_mass_tree[tab.leaf_offset+k] = log_masses[k]; }
	for(size_t k=tab.leaf_offset-1; k>0; --k){
		tab.log_mass_tree[k] = log_sum_exp2(tab.log_mass_tree[2*k], tab.log_mass_tree[2*k+1]);
	}
	const double prior = prior_stop(v.level);
	tab.log_stop_mass = std::log(prior) + v.loglik_base;
	tab.log_split_mass = std::log1p(-prior) + v.loglik_knot + tab.log_mass_tree[1];
	const double log_normalizer = log_sum_exp2(tab.log_stop_mass, tab.log_split_mass);
	tab.log_weight_increment = log_normalizer - v.loglik_base;
	v.rho_log = tab.log_stop_mass - log_normalizer;
	v.rho = std::min(1.0, std::exp(v.rho_log));
	v.proposals.nested = std::move(prop);
}

/*========================= expansion of a nested PP node =========================*/

void ResTree::expand_nested(PathTrie& trie, PathNode& v, const std::vector<int>& idx,
	const Eigen::VectorXd& resid, const Eigen::MatrixXd& Q) const{

	const int n = (int)idx.size();
	v.n_arrival = n;
	if((int)v.child_path.size()!=2*dim){ v.child_path.assign(2*dim, -1); }

	// ---- knots: root / small node by pick_knots, otherwise inherited ----
	const std::vector<int>* parent_cand = (v.parent>=0) ? &trie[v.parent].nested_cand : nullptr;
	Eigen::VectorXi ind_knot, ind_res;
	nested_node_knots(idx, parent_cand, v.box_min, v.box_max, ind_knot, ind_res);
	const int rk = ind_knot.size();
	const int n_res = ind_res.size();
	Eigen::MatrixXd Kx(rk, dim), Xres(n_res, dim);
	for(int i=0; i<rk; ++i){ Kx.row(i) = X.row(idx[(size_t)ind_knot(i)]); }
	for(int i=0; i<n_res; ++i){ Xres.row(i) = X.row(idx[(size_t)ind_res(i)]); }
	Eigen::VectorXd knot_data(rk), zres(n_res);
	for(int i=0; i<rk; ++i){ knot_data(i) = resid(ind_knot(i)); }
	for(int i=0; i<n_res; ++i){ zres(i) = resid(ind_res(i)); }
	const int q = (int)Q.rows();
	Eigen::MatrixXd Qk(q, rk), Qx(q, n_res);
	for(int i=0; i<rk; ++i){ Qk.col(i) = Q.col(ind_knot(i)); }
	for(int i=0; i<n_res; ++i){ Qx.col(i) = Q.col(ind_res(i)); }

	// ---- the knot block ----
	KnotBlock kb;
	if(!nested_knot_step(Kx, Xres, knot_data, zres, Qk, Qx, KNOT_RESIDUALIZE, kb)){
		record_cholesky_error();
		return;
	}

	// ---- arrival (stop) value ----
	if(v.parent<0){
		// the root: dense when n <= r, else the PP leaf on the maximin knots
		// (the same factors as the split block, as expand() does)
		if(n<=r_knot){
			Eigen::MatrixXd Xn(n, dim);
			for(int i=0; i<n; ++i){ Xn.row(i) = X.row(idx[(size_t)i]); }
			v.loglik_base = leaf_pp_loglik(Xn, resid, Q, v.box_min, v.box_max, v.stat_leaf);
		}else{
			v.loglik_base = leaf_pp_loglik_from_factors(n, true, kb.L,
				kb.Qnext.bottomRows(rk), kb.Ry, kb.zres_c, Qx, v.stat_leaf);
		}
	}
	// else: set by child_cell from the parent's sweep -- the number the
	// particles' partial-tree likelihood already carries (see expand()'s
	// integrated-WN branch).

	v.stat_knot = kb.stat;
	v.loglik_knot = restree_loglik_conditional(v.stat_knot.ld, v.stat_knot.qd,
		v.stat_knot.n, covpar.sig2);

	if(v.level >= depth){
		v.rho = 1.0; v.rho_log = 0.0;
		v.cuts.resize(0);
		v.child_loglik.resize(0); v.child_loglik_lr.resize(0, 2);
		v.child_stats.clear(); v.child_n.resize(0, 2);
		v.expanded = true;
		v.need_expand = false;
		return;
	}

	// ---- candidates for the children, then the integrated proposal ----
	std::vector<int> cand_pos = nested_candidates(Xres);
	v.nested_cand.resize(cand_pos.size());
	for(size_t a=0; a<cand_pos.size(); ++a){ v.nested_cand[a] = idx[(size_t)ind_res(cand_pos[a])]; }
	#ifdef _OPENMP
	#pragma omp critical(restree_nested_proposal_list)
	#endif
	{
		worker_error.run("proposal-list",[&]{
			trie.nested_proposal_paths.push_back(static_cast<int>(&v-trie.data()));
		});
	}
	worker_error.rethrow();
	prepare_nested_cut_proposal(v, Xres, kb.zres_c, kb.Qnext, cand_pos);
	v.cuts.resize(0);
	v.child_loglik.resize(0); v.child_loglik_lr.resize(0, 2);
	v.child_stats.clear(); v.child_n.resize(0, 2);

	// ---- post-knot state for the next level's expansions (as deterministic
	// cuts do: under the budget, else descend() re-derives it) ----
	if(v.level + 1 < depth){
		const size_t bytes = (size_t)n_res * (sizeof(int) + sizeof(double))
			+ (size_t)(q+rk) * (size_t)n_res * sizeof(double);
		bool do_cache = false;
		#ifdef _OPENMP
		#pragma omp critical(restree_cache_list)
		#endif
		{
			worker_error.run("cache-list",[&]{
			if(trie.level_cache.current.bytes + bytes <= trie.cache_budget_bytes){
				trie.level_cache.current.paths.push_back((int)(&v - trie.data()));
				trie.level_cache.current.bytes += bytes;
				do_cache = true;
			}
			});
		}
		worker_error.rethrow();
		if(do_cache){
			v.postknot.row_ids.resize((size_t)n_res);
			for(int i=0; i<n_res; ++i){ v.postknot.row_ids[(size_t)i] = idx[(size_t)ind_res(i)]; }
			v.postknot.residual = std::move(kb.zres_c);
			v.postknot.coefficients = std::move(kb.Qnext);
			v.postknot.ready = true;
		}
	}
	v.expanded = true;
	v.need_expand = false;
}

// ==================== Tree cache ownership and release ====================

// Bytes a node's level-scoped state holds NOW: the post-knot state (row
// ids, residual, coefficients) and, while it lives, the PP draw-time
// workspace (the residual rows and their sorted columns, 2 d doubles per
// row).  Callers subtract this at release time, so it must count what is
// allocated at the call, not what was stored.
size_t cache_bytes_of(const PathNode& v){
	return v.postknot.row_ids.size() * sizeof(int)
		+ (size_t)v.postknot.residual.size() * sizeof(double)
		+ (size_t)v.postknot.coefficients.size() * sizeof(double)
		+ (size_t)v.cut_workspace.input.size() * sizeof(double)
		+ (size_t)v.cut_workspace.sorted.size() * sizeof(double);
}

// Bytes of an integrated-cut WN proposal table.
size_t proposal_bytes_of(const WNCutProposal& p){
	return p.intervals.size()*sizeof(WNCutInterval)
		+ p.log_mass_tree.size()*sizeof(double) + sizeof(WNCutProposal);
}

// Random cuts: the post-knot state (gidx, z, Q) feeds the next level's
// expand_level; the draw-time state (PP candidate workspace X / sorted
// columns, WN proposal) is consumed by the node's own level.  They are
// released separately so the post-knot state lives exactly one level longer.
void release_postknot(PathNode& v){
	v.postknot.row_ids.clear(); v.postknot.row_ids.shrink_to_fit();
	v.postknot.residual.resize(0); v.postknot.coefficients.resize(0,0);
	v.postknot.ready = false;
}
void release_draw_state(PathNode& v){
	v.cut_workspace.input.resize(0,0); v.cut_workspace.sorted.resize(0,0);
}

// Drop a node's level-scoped expansion cache (both cut modes).
void release_cache(PathNode& v){
	v.postknot.row_ids.clear(); v.postknot.row_ids.shrink_to_fit();
	v.cut_workspace.input.resize(0,0); v.postknot.coefficients.resize(0,0);
	v.cut_workspace.sorted.resize(0,0); v.postknot.residual.resize(0);
	v.postknot.ready = false;
	v.retention = CacheRetention::none;
	v.proposals.wn.reset();
	v.proposals.nested.reset();
}

/**************** path cache: theta-independent expansion geometry ****************/


std::string ResTree::path_key(const PathTrie& trie, const PathNode& v){
	std::string key;
	key.reserve((size_t)2*(size_t)std::max(0, v.level));
	const PathNode* node = &v;
	while(node->parent>=0){
		key.push_back((char)node->j_from_parent);
		key.push_back((char)node->side_from_parent);
		node = &trie[node->parent];
	}
	// built leaf-to-root; reverse the (J, side) pairs to root-to-leaf order
	std::string out(key.size(), '\0');
	for(size_t k=0; k<key.size(); k+=2){
		out[key.size()-2-k] = key[k];
		out[key.size()-1-k] = key[k+1];
	}
	return out;
}

std::shared_ptr<const PathCacheEntry> ResTree::path_cache_get(const std::string& key) const{
	std::shared_ptr<const PathCacheEntry> hit;
	#ifdef _OPENMP
	#pragma omp critical(restree_path_cache)
	#endif
	{
		auto it = path_cache.find(key);
		if(it!=path_cache.end()){ hit = it->second; path_cache_hits++; }
		else{ path_cache_misses++; }
	}
	return hit;
}

void ResTree::path_cache_put(const std::string& key,
	std::shared_ptr<const PathCacheEntry> entry) const{
	#ifdef _OPENMP
	#pragma omp critical(restree_path_cache)
	#endif
	{
		worker_error.run("cache",[&]{
		if(path_cache.find(key)==path_cache.end()){
			if(path_cache_bytes + entry->bytes > path_cache_budget_bytes &&
			   !path_cache.empty()){
				path_cache.clear();
				path_cache_bytes = 0;
				path_cache_flushes++;
			}
			if(path_cache_bytes + entry->bytes <= path_cache_budget_bytes){
				const size_t bytes = entry->bytes;
				path_cache.emplace(key, std::move(entry));
				path_cache_bytes += bytes;
			}
		}
		});
	}
	worker_error.rethrow();
}

void ResTree::path_cache_clear() const{
	#ifdef _OPENMP
	#pragma omp critical(restree_path_cache)
	#endif
	{
		path_cache.clear();
		path_cache_bytes = 0;
	}
}

size_t path_cache_entry_bytes(const PathCacheEntry& e, size_t key_len){
	size_t b = sizeof(PathCacheEntry) + key_len + 64;
	b += (size_t)(e.ind_knot.size() + e.ind_res.size()) * sizeof(int);
	b += (size_t)e.cuts.size() * sizeof(double);
	for(size_t k=0; k<e.child_kn.size(); ++k){
		b += sizeof(Eigen::VectorXi) + (size_t)e.child_kn[k].size() * sizeof(int);
	}
	return b;
}

// ==================== Fixed-tree evaluation and Full geometry ====================

/**************** known-theta fixed-structure evaluation ****************/
// Top-down rescoring of one tree structure, with the tree geometry -- knot
// selection, cuts, row partitioning, boxes -- carried by this class.  The
// likelihood aggregates one global Gaussian (logdet, quad, n) triple plus the
// WN leaf marginals, assembled once at the top.  This is a computation path
// INDEPENDENT of the SMC sampler's incremental trie bookkeeping, so rescoring
// a sampled tree (restree_loglik(verify = TRUE)) is a real check.

double ResTree::gaussian_loglik(double logdet, double quad, int n) const{
	if(n==0){ return 0.0; }
	if(!(covpar.sig2>0.0) || !std::isfinite(logdet) || !std::isfinite(quad)){
		return -std::numeric_limits<double>::infinity();
	}
	return -0.5*(static_cast<double>(n)*std::log(2.0*M_PI) +
		static_cast<double>(n)*std::log(covpar.sig2) + logdet +
		quad/covpar.sig2);
}

double ResTree::white_noise_loglik(const Eigen::VectorXd& z) const{
	const int n = z.size();
	if(n==0){ return 0.0; }
	// Fully proper Normal-Inverse-Gamma leaf prior:
	//   sigma2 ~ IG(a0, b0),  mu | sigma2 ~ N(mu0, sigma2 / kappa0).
	// kappa0 = 1 gives the prior mean one pseudo-observation at zero; this
	// defines Bayes factors across different numbers of leaves and is finite
	// for singleton and constant leaves.
	const double a0=2.0, b0=0.01, mu0=0.0, kappa0=1.0;
	const double nd=(double)n, ybar=z.mean();
	const double sse=(z.array()-ybar).square().sum();
	const double kappa_n=kappa0+nd, a_n=a0+0.5*nd;
	const double b_n=b0+0.5*sse+0.5*kappa0*nd/kappa_n*(ybar-mu0)*(ybar-mu0);
	if(!(b_n>0.0) || !std::isfinite(b_n)){
		return -std::numeric_limits<double>::infinity();
	}
	return -0.5*nd*std::log(2.0*M_PI) +
		0.5*(std::log(kappa0)-std::log(kappa_n)) +
		a0*std::log(b0)-a_n*std::log(b_n)+
		std::lgamma(a_n)-std::lgamma(a0);
}

void ResTree::combine(StructureStats& into, const StructureStats& from){
	into.quad += from.quad;
	into.logdet += from.logdet;
	into.white_noise_loglik += from.white_noise_loglik;
	into.n_gauss += from.n_gauss;
	into.n_knot += from.n_knot;
	into.n_knot_live += from.n_knot_live;
	into.n_leaf_knot += from.n_leaf_knot;
	into.n_white_noise_leaf += from.n_white_noise_leaf;
	into.n_white_noise_obs += from.n_white_noise_obs;
	if(!from.ok && into.ok){
		into.ok = false;
		into.error = from.error;
	}
}

// Gather the knot rows / residual rows of a node (node-local positions kn /
// res over rows idx of X, values z, coefficients Q) into the inputs of
// knot_step.  Shared by loglik_subtree and loglik_geometry_subtree.
static void gather_knot_inputs(const Eigen::MatrixXd& X, const std::vector<int>& idx,
	const Eigen::VectorXd& z, const Eigen::MatrixXd& Q,
	const Eigen::VectorXi& kn, const Eigen::VectorXi& res,
	Eigen::MatrixXd& Kx, Eigen::MatrixXd& Xres, Eigen::VectorXd& knot_data,
	Eigen::VectorXd& zres, Eigen::MatrixXd& Qk, Eigen::MatrixXd& Qx){
	const int rk = (int)kn.size(), nres = (int)res.size(), q = (int)Q.rows(), dim = (int)X.cols();
	Kx.resize(rk, dim); Xres.resize(nres, dim);
	knot_data.resize(rk); zres.resize(nres);
	Qk.resize(q, rk); Qx.resize(q, nres);
	for(int i=0; i<rk; ++i){
		Kx.row(i) = X.row(idx[(size_t)kn(i)]);
		knot_data(i) = z(kn(i));
		if(q>0){ Qk.col(i) = Q.col(kn(i)); }
	}
	for(int i=0; i<nres; ++i){
		Xres.row(i) = X.row(idx[(size_t)res(i)]);
		zres(i) = z(res(i));
		if(q>0){ Qx.col(i) = Q.col(res(i)); }
	}
}

// The level up to which the fixed-structure recursion spawns sibling tasks.
// Every node at levels 0..d is live at once (each waits on both children), and
// a level's coefficient blocks total (l+1) * r * n * 8 bytes however many nodes
// it holds, so levels 0..d cost r * n * 8 * (d+1)(d+2)/2.  Take the largest d
// inside task_budget_bytes, never deeper than ceil(log2(ncores)) -- past that
// there is no team member left to take a task, only memory to pay.  d = 0 is a
// serial recursion inside the team; knot_step's panels still use the threads.
int ResTree::task_depth() const{
	if(ncores<2){ return 0; }
	int dmax = 0;
	while((1 << dmax) < ncores){ ++dmax; }
	// double throughout: (d+1)(d+2)/2 * r * n * 8 overflows size_t for no
	// realistic problem, but the comparison must not wrap if it ever could.
	const double unit = (double)r_knot * (double)y.size() * (double)sizeof(double);
	const double budget = (double)task_budget_bytes;
	int d = 0;
	while(d<dmax && unit*(double)(d+2)*(double)(d+3)*0.5 <= budget){ ++d; }
	return d;
}

StructureStats ResTree::loglik_subtree(StructureContext& ctx, int id, int level,
	std::vector<int>&& idx, Eigen::VectorXd&& z, Eigen::MatrixXd&& Q,
	Eigen::VectorXd bmin, Eigen::VectorXd bmax,
	const std::vector<int>* parent_cand) const{

	StructureStats out;
	const int n = (int)idx.size();
	if(n==0){ return out; }
	if((int)Q.cols()!=n){
		out.ok = false;
		out.error = "coefficient workspace is not aligned with node observations";
		return out;
	}

	const bool bottom = !ctx.sparse && (level==ctx.tree_depth);
	bool stop = bottom;
	if(!bottom && ctx.fixed){
		double s;
		if(ctx.sparse){
			const auto at=ctx.sparse->find(id);
			if(at==ctx.sparse->end()){
				out.ok=false;
				out.error="nonempty node "+std::to_string(id)+" is missing from the sparse tree";
				return out;
			}
			s=at->second.split ? 0.0 : 1.0;
		}else s=(*ctx.fixed_S)(id-1);
		if(std::isnan(s)){
			// NA marks a position no observation reaches; rows arriving here
			// mean the supplied S disagrees with the data routing, and
			// skipping them would silently drop their likelihood.
			out.ok = false;
			out.error = "the supplied tree marks node " + std::to_string(id) +
				" as unreachable (S is NA) although " + std::to_string(n) +
				" observation(s) reach it; every non-empty node needs S = 0 (split) or 1 (terminal)";
			return out;
		}
		stop = s>=0.5;
	}else if(!bottom){
		// The generated (Full) tree keeps its global-r stopping rule: a node
		// with n <= r stops, so every split spends exactly r knots.  Sampled /
		// supplied structures (ctx.fixed) have their S/J decisions rebuilt
		// exactly and spend min(r, n) at a split.
		stop = ctx.leaf_model=="Full" ? n <= r_knot : n < r_knot;
	}

	Eigen::MatrixXd Xn(n, dim);
	for(int i=0; i<n; ++i){ Xn.row(i) = X.row(idx[i]); }

	// ---- terminals ----
	if(stop && ctx.leaf_model=="Full" && n>r_knot){
		out.ok = false;
		out.error = "Full terminal contains more than r observations; rebuild the size-constrained tree";
		return out;
	}
	if(stop && ctx.leaf_model=="WhiteNoise"){
		out.white_noise_loglik = white_noise_loglik(z);
		if(!std::isfinite(out.white_noise_loglik)){
			out.ok = false;
			out.error = "non-finite WhiteNoise Normal-Inverse-Gamma marginal";
			return out;
		}
		out.n_white_noise_leaf = 1;
		out.n_white_noise_obs = n;
		if(!bottom && ctx.S){ (*ctx.S)(id-1) = 1.0; }
		return out;
	}
	if(stop && (ctx.leaf_model=="Full" || n<=r_knot)){
		// Dense block: a Full leaf, or a PP leaf with no more rows than knots.
		BlockStats st;
		if(!dense_block(Xn, z, Q, st, nullptr) || !std::isfinite(st.qd) || !std::isfinite(st.ld)){
			out.ok = false;
			out.error = ctx.leaf_model=="Full" ? "non-positive-definite full leaf covariance"
				: "non-positive-definite PP leaf covariance";
			return out;
		}
		out.quad = st.qd; out.logdet = st.ld; out.n_gauss = n;
		if(ctx.leaf_model=="PP"){ out.n_leaf_knot = n; }
		if(!bottom && ctx.S){ (*ctx.S)(id-1) = 1.0; }
		return out;
	}
	if(stop){
		// PP terminal with r knots on the node rows; the remaining
		// observations stay in the diagonal Schur correction.  (Nested
		// design: the knots are the parent's candidates that reach the node.)
		Eigen::VectorXi kn, res;
		if(nested()){
			nested_node_knots(idx, parent_cand,
				bmin, bmax, kn, res);
		}else{
			pick_knots(Xn, bmin, bmax, kn, res, r_knot);
		}
		Eigen::MatrixXd Kx, Xres, Qk, Qr;
		Eigen::VectorXd knot_data, zres;
		gather_knot_inputs(X, idx, z, Q, kn, res, Kx, Xres, knot_data, zres, Qk, Qr);
		KnotBlock kb;
		const bool leaf_ok = nested() ?
			nested_knot_step(Kx, Xres, knot_data, zres, Qk, Qr, KNOT_PROPAGATE, kb) :
			knot_step(Kx, Xres, knot_data, zres, Qk, Qr, KNOT_PROPAGATE, kb);
		if(!leaf_ok || !std::isfinite(kb.stat.qd) || !std::isfinite(kb.stat.ld)){
			out.ok = false;
			out.error = "non-positive-definite PP knot covariance";
			return out;
		}
		const int rk = kb.stat.n, nr = (int)res.size(), q = (int)Q.rows();
		Eigen::VectorXd alpha = kb.L.transpose().triangularView<Eigen::Upper>().solve(kb.Ry);
		Eigen::VectorXd zr = zres - kb.B*alpha;
		Eigen::VectorXd diag = Eigen::VectorXd::Constant(nr, 1.0 + covpar.nugget);
		if(q>0){
			diag.array() -= Qr.array().square().colwise().sum().transpose();
		}
		diag.array() -= kb.W.array().square().colwise().sum().transpose();
		if(!zr.allFinite() || !diag.allFinite() ||
			(diag.array() <= -1e-9).any()){
			out.ok = false;
			out.error = "invalid PP diagonal residual covariance";
			return out;
		}
		diag = diag.array().max(1e-12);
		out.quad = kb.stat.qd + (zr.array().square()/diag.array()).sum();
		out.logdet = kb.stat.ld + diag.array().log().sum();
		out.n_gauss = n;
		out.n_leaf_knot = rk;
		if(!bottom && ctx.S){ (*ctx.S)(id-1) = 1.0; }
		return out;
	}

	// ---- split: internal knot block, then the two children ----
	Eigen::VectorXi kn, res;
	if(nested()){
		nested_node_knots(idx, parent_cand,
			bmin, bmax, kn, res);
	}else{
		pick_knots(Xn, bmin, bmax, kn, res, knot_count(n));
	}
	Eigen::MatrixXd Kx, Xres, Qk, Qx;
	Eigen::VectorXd knot_data, zres;
	gather_knot_inputs(X, idx, z, Q, kn, res, Kx, Xres, knot_data, zres, Qk, Qx);
	Q.resize(0,0);
	KnotBlock kb;
	const bool split_ok = nested() ?
		nested_knot_step(Kx, Xres, knot_data, zres, Qk, Qx, KNOT_RESIDUALIZE, kb) :
		knot_step(Kx, Xres, knot_data, zres, Qk, Qx, KNOT_RESIDUALIZE, kb);
	if(!split_ok || !std::isfinite(kb.stat.qd) || !std::isfinite(kb.stat.ld)){
		out.ok = false;
		out.error = "non-positive-definite internal knot covariance";
		return out;
	}
	// This frame owns the candidates until both children finish. Siblings
	// read the same immutable vector; no depth-sized table or concurrently
	// mutated map is needed, including inside OpenMP tasks.
	std::vector<int> candidates;
	if(nested()){
		// the candidates this node hands to its children (global ids)
		std::vector<int> cp = nested_candidates(Xres);
		candidates.resize(cp.size());
		for(size_t a=0; a<cp.size(); ++a){ candidates[a] = idx[(size_t)res(cp[a])]; }
	}
	const std::vector<int>* child_cand = nested() ? &candidates : nullptr;
	const int rk = kb.stat.n;
	const int nres = (int)res.size();
	out.quad = kb.stat.qd;
	out.logdet = kb.stat.ld;
	out.n_gauss = rk;

	int split_dim = 0;
	Eigen::VectorXd cut_point;
	if(ctx.fixed){
		split_dim = ctx.sparse ? ctx.sparse->at(id).J :
			static_cast<int>((*ctx.fixed_J)(id-1)) - 1;
		const double fixed_cut=ctx.sparse ? ctx.sparse->at(id).cut : (*ctx.fixed_cuts)(id-1);
		if(split_dim<0 || split_dim>=dim || !std::isfinite(fixed_cut)){
			out.ok = false;
			out.error = "invalid fixed split dimension or cut";
			return out;
		}
		cut_point = Eigen::VectorXd::Constant(dim, fixed_cut);
	}else{
		split_dim = dim>1 ? level % dim : 0;
		cut_point = generated_cut_points(Xres, ctx.cut_method, ctx.seed, id, bmin, bmax);
	}
	const double cut = cut_point(split_dim);

	int nl = 0;
	for(int i=0; i<nres; ++i){ if(Xres(i,split_dim) < cut){ ++nl; } }
	const int nrr = nres - nl;
	std::vector<int> idxl; idxl.reserve(nl);
	std::vector<int> idxr; idxr.reserve(nrr);
	Eigen::VectorXd zl(nl), zr2(nrr);
	Eigen::MatrixXd Qleft(kb.Qnext.rows(), nl), Qright(kb.Qnext.rows(), nrr);
	int il = 0, ir = 0;
	for(int i=0; i<nres; ++i){
		if(Xres(i,split_dim) < cut){
			idxl.push_back(idx[res(i)]);
			zl(il) = kb.zres_c(i);
			Qleft.col(il++) = kb.Qnext.col(i);
		}else{
			idxr.push_back(idx[res(i)]);
			zr2(ir) = kb.zres_c(i);
			Qright.col(ir++) = kb.Qnext.col(i);
		}
	}
	kb = KnotBlock();
	if(ctx.S){
		(*ctx.S)(id-1) = 0.0;
		(*ctx.J)(id-1) = split_dim + 1.0;
		(*ctx.cuts)(id-1) = cut;
	}
	// Count the knot block even when it consumes the entire node: both child
	// likelihoods are then the neutral factor one, but p(y(R)) remains a real
	// part of the model likelihood and must stay visible in diagnostics.
	out.n_knot += rk;
	if(nres>0){ out.n_knot_live += rk; }

	Eigen::VectorXd bminL = bmin, bmaxL = bmax;
	bmaxL(split_dim) = cut;
	Eigen::VectorXd bminR = bmin, bmaxR = bmax;
	bminR(split_dim) = cut;

	StructureStats left, right;
#ifdef _OPENMP
	if(level < ctx.task_depth){
		#pragma omp task shared(ctx, left, idxl, zl, Qleft, bminL, bmaxL) firstprivate(id, level, child_cand)
		worker_error.run("fixed-child",[&]{
			left = loglik_subtree(ctx, 2*id, level+1, std::move(idxl), std::move(zl),
				std::move(Qleft), bminL, bmaxL, child_cand);
		});
		#pragma omp task shared(ctx, right, idxr, zr2, Qright, bminR, bmaxR) firstprivate(id, level, child_cand)
		worker_error.run("fixed-child",[&]{
			right = loglik_subtree(ctx, 2*id+1, level+1, std::move(idxr), std::move(zr2),
				std::move(Qright), bminR, bmaxR, child_cand);
		});
		#pragma omp taskwait
		worker_error.rethrow();
	}else
#endif
	{
		left = loglik_subtree(ctx, 2*id, level+1, std::move(idxl), std::move(zl),
			std::move(Qleft), bminL, bmaxL, child_cand);
		right = loglik_subtree(ctx, 2*id+1, level+1, std::move(idxr), std::move(zr2),
			std::move(Qright), bminR, bmaxR, child_cand);
	}
	combine(out, left);
	combine(out, right);
	return out;
}

StructureEval ResTree::evaluate(const Eigen::VectorXd* fixed_S,
	const Eigen::VectorXd* fixed_J, const Eigen::VectorXd* fixed_cuts,
	const std::string& leaf_model, const std::string& cut_method_use,
	unsigned int seed_use) const{

	worker_error.reset();
	StructureEval ans;
	int eval_depth=depth;
	if(fixed_S && leaf_model=="WhiteNoise") {
		const int rows=(int)fixed_S->size();
		if(depth==0 && rows!=1) {
			ans.stats.ok=false; ans.stats.error="depth-zero WN requires a single root leaf";
			return ans;
		}
		if(rows<1 || rows>(1<<restree_max_wn_depth) || (rows & (rows-1))) {
			ans.stats.ok=false;
			ans.stats.error="WN tree arrays must have power-of-two length, at most 2^"
				+ std::to_string(restree_max_wn_depth);
			return ans;
		}
		eval_depth=0; while((1<<eval_depth)<rows) ++eval_depth;
	}
	const int numI = 1 << eval_depth;
	ans.S = Eigen::VectorXd::Constant(numI, NA_REAL);
	ans.J = Eigen::VectorXd::Constant(numI, NA_REAL);
	ans.cuts = Eigen::VectorXd::Constant(numI, NA_REAL);
	if(fixed_S!=nullptr &&
		(fixed_S->size()!=numI || fixed_J->size()!=numI ||
		 fixed_cuts->size()!=numI)){
		ans.stats.ok = false;
		ans.stats.error = "S, J, and cuts must have length 2^depth";
		return ans;
	}

	const int td = task_depth();
	StructureContext ctx = {fixed_S!=nullptr, fixed_S, fixed_J,
		fixed_cuts, &ans.S, &ans.J, &ans.cuts, leaf_model, cut_method_use,
		seed_use, td, eval_depth};
	const int n = (int)y.size();
	std::vector<int> idx((size_t)n);
	std::iota(idx.begin(), idx.end(), 0);
	Eigen::VectorXd z = y;
	Eigen::MatrixXd Q0(0, n);
	Eigen::VectorXd bmin = Eigen::VectorXd::Zero(dim);
	Eigen::VectorXd bmax = Eigen::VectorXd::Ones(dim);
#ifdef _OPENMP
	if(ncores>1){
		#pragma omp parallel num_threads(ncores)
		{
			#pragma omp single
			worker_error.run("fixed-root",[&]{
				ans.stats = loglik_subtree(ctx, 1, 0, std::move(idx), std::move(z),
					std::move(Q0), bmin, bmax);
			});
		}
	}else
#endif
	{
		ans.stats = loglik_subtree(ctx, 1, 0, std::move(idx), std::move(z),
			std::move(Q0), bmin, bmax);
	}
	worker_error.rethrow();
	if(ans.stats.ok){
		ans.loglik = gaussian_loglik(ans.stats.logdet, ans.stats.quad,
			ans.stats.n_gauss) + ans.stats.white_noise_loglik;
	}
	return ans;
}

StructureEval ResTree::evaluate_nodes(const FixedNodes& nodes,
	const std::string& leaf_model) const{
	worker_error.reset();
	StructureEval ans;
	const bool wn = leaf_model=="WhiteNoise";
	if((!wn && leaf_model!="PP" && leaf_model!="Full") || nodes.empty() || !nodes.count(1)){
		ans.stats.ok=false;
		ans.stats.error="sparse fixed-tree evaluation requires a PP, WN or Full tree containing its root";
		return ans;
	}
	if(depth==0 && (nodes.size()!=1 || nodes.at(1).split)){
		ans.stats.ok=false; ans.stats.error="a depth-zero model requires a single root leaf";
		return ans;
	}
	// PP's supplied cap is structural: bottom nodes must be explicit leaves.
	// Positive-depth WN remains open up to its separate representation limit.
	const int limit = wn ? restree_max_wn_depth : depth;
	for(const auto& item:nodes){
		const int id=item.first;
		if(id<1 || id>=(2<<limit) ||
			(item.second.split && id>=(1<<limit)) ||
			(id>1 && (!nodes.count(id/2) || !nodes.at(id/2).split))){
			ans.stats.ok=false; ans.stats.error="invalid sparse node address or parent";
			return ans;
		}
	}
	const int td = task_depth();
	StructureContext ctx={true,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
		leaf_model,"middle",0u,td,limit};
	ctx.sparse=&nodes;
	const int n=(int)y.size();
	std::vector<int> idx((size_t)n); std::iota(idx.begin(),idx.end(),0);
	Eigen::VectorXd z=y;
	Eigen::MatrixXd Q(0,n);
	Eigen::VectorXd lo=Eigen::VectorXd::Zero(dim), hi=Eigen::VectorXd::Ones(dim);
#ifdef _OPENMP
	if(ncores>1){
		#pragma omp parallel num_threads(ncores)
		{
			#pragma omp single
			worker_error.run("fixed-root",[&]{
				ans.stats=loglik_subtree(ctx,1,0,std::move(idx),std::move(z),std::move(Q),lo,hi);
			});
		}
	}else
#endif
	ans.stats=loglik_subtree(ctx,1,0,std::move(idx),std::move(z),std::move(Q),lo,hi);
	worker_error.rethrow();
	if(ans.stats.ok)
		ans.loglik=gaussian_loglik(ans.stats.logdet,ans.stats.quad,ans.stats.n_gauss)+ans.stats.white_noise_loglik;
	return ans;
}

StructureEval ResTree::evaluate_structure(const Eigen::VectorXd& S,
	const Eigen::VectorXd& J, const Eigen::VectorXd& cuts,
	const std::string& leaf_model) const{
	return evaluate(&S, &J, &cuts, leaf_model, "middle", 0u);
}

StructureEval ResTree::evaluate_generated(const std::string& leaf_model,
	const std::string& cut_method_use, unsigned int seed_use) const{
	if(leaf_model=="Full"){
		auto geom = build_generated_geometry(cut_method_use, seed_use);
		return evaluate_geometry(*geom);
	}
	return evaluate(nullptr, nullptr, nullptr, leaf_model, cut_method_use,
		seed_use);
}

// Random cut draws of the generated Full tree on the post-knot residual rows
// -- ALL dimensions in order from one per-node RNG stream; an empty residual
// block leaves the cut unset (middle fallback at the end).
Eigen::VectorXd ResTree::generated_cut_points(const Eigen::MatrixXd& Xres,
	const std::string& cut_method_use, unsigned int seed_use, int id,
	const Eigen::VectorXd& bmin, const Eigen::VectorXd& bmax) const{

	const int nres = (int)Xres.rows();
	Eigen::VectorXd cut_point;
	if(nres>0){
		cut_point.resize(dim);
		if(cut_method_use=="median"){
			for(int Jd=0; Jd<dim; ++Jd){
				Eigen::VectorXd v = Xres.col(Jd);
				std::sort(v.data(), v.data()+v.size());
				const int nv = v.size();
				cut_point(Jd) = (nv%2==0) ? 0.5*(v(nv/2-1)+v(nv/2)) : v(nv/2);
			}
		}else if(cut_method_use=="balanced" || cut_method_use=="uniform"){
			std::mt19937_64 rng(seed_use +
				2654435761u * static_cast<unsigned int>(id));
			for(int Jd=0; Jd<dim; ++Jd){
				Eigen::VectorXd v = Xres.col(Jd);
				std::sort(v.data(), v.data()+v.size());
				const int nv = v.size();
				double pos;
				if(cut_method_use=="balanced"){
					std::gamma_distribution<double> rgam(2.0, 1.0);
					const double g1 = rgam(rng), g2 = rgam(rng);
					const double u = (g1+g2 > 0.0) ? g1/(g1+g2) : 0.5;
					pos = u * (double)(nv-1);
				}else{
					std::uniform_real_distribution<double> runif(0.0, 1.0);
					pos = runif(rng) * (double)(nv-1);
				}
				const int lo = (int) std::floor(pos);
				const int hi = (lo+1<nv) ? (lo+1) : lo;
				const double w = pos - (double)lo;
				cut_point(Jd) = (1.0-w)*v(lo) + w*v(hi);
			}
		}else{ // "middle" and any unprepared method
			for(int Jd=0; Jd<dim; ++Jd){
				cut_point(Jd) = (bmin(Jd) + bmax(Jd)) / 2.0;
			}
		}
	}
	if(cut_point.size()!=dim){ cut_point = 0.5*(bmin + bmax); }
	return cut_point;
}

/*================ generated Full tree: geometry once, thetas many ================*/
// Full stops exactly when n <= r. Extend the storage depth when a larger
// node reaches the requested depth; that depth is not a leaf-size cutoff.
// Spend r knots chosen by pick_knots on the node rows, cut the
// residual rows with generated_cut_points, and route them by X(., J) < cut in
// node order.  Nothing here reads theta.
int ResTree::geometry_subtree(FullTreeGeometry& geom, int id, int level,
	std::vector<int>&& idx, Eigen::VectorXd bmin, Eigen::VectorXd bmax) const{

	const int n = (int)idx.size();
	if(n>r_knot && level>=geom.depth){
		if(level>=20){
			Rcpp::stop("Full needs a tree deeper than the supported depth 20 to keep leaves at most r; increase r. No oversized leaf was accepted.");
		}
		geom.depth = level + 1;
	}
	const int position=static_cast<int>(geom.nodes.size());
	geom.nodes.emplace_back();
	FullTreeNodeGeometry& node=geom.nodes.back();
	node.level=level; node.id=id;
	if(n==0) return position;
	const bool stop = n <= r_knot;
	if(stop){
		node.split = false;
		node.idx = std::move(idx);
		geom.bytes += node.idx.size()*sizeof(int);
		return position;
	}

	Eigen::MatrixXd Xn(n, dim);
	for(int i=0; i<n; ++i){ Xn.row(i) = X.row(idx[i]); }
	const int split_r = knot_count(n);   // == r here (n > r)
	Eigen::VectorXi kn, res;
	pick_knots(Xn, bmin, bmax, kn, res, split_r);
	const int nres = res.size();
	Eigen::MatrixXd Xres(nres, dim);
	for(int i=0; i<nres; ++i){ Xres.row(i) = Xn.row(res(i)); }

	const int split_dim = dim>1 ? level % dim : 0;
	const Eigen::VectorXd cut_point = generated_cut_points(Xres, geom.cut_method,
		geom.seed, id, bmin, bmax);
	const double cut = cut_point(split_dim);

	node.split = true;
	node.kn = kn;
	node.res = res;
	node.side.assign((size_t)nres, 0);
	std::vector<int> idxl, idxr;
	idxl.reserve((size_t)nres); idxr.reserve((size_t)nres);
	for(int i=0; i<nres; ++i){
		if(Xres(i,split_dim) < cut){ idxl.push_back(idx[res(i)]); }
		else{ node.side[(size_t)i] = 1; idxr.push_back(idx[res(i)]); }
	}
	node.idx = std::move(idx);
	geom.bytes += node.idx.size()*sizeof(int) + (kn.size()+res.size())*sizeof(int)
		+ node.side.size();
	node.J=split_dim; node.cut=cut;

	Eigen::VectorXd bminL = bmin, bmaxL = bmax; bmaxL(split_dim) = cut;
	Eigen::VectorXd bminR = bmin, bmaxR = bmax; bminR(split_dim) = cut;
	// Recursive appends may move the vector: never retain node references.
	const int left=geometry_subtree(geom,2*id,level+1,std::move(idxl),bminL,bmaxL);
	const int right=geometry_subtree(geom,2*id+1,level+1,std::move(idxr),bminR,bmaxR);
	geom.nodes[position].left=left; geom.nodes[position].right=right;
	return position;
}

std::shared_ptr<FullTreeGeometry> ResTree::build_generated_geometry(
	const std::string& cut_method_use, unsigned int seed_use) const{

	auto geom = std::make_shared<FullTreeGeometry>();
	geom->n = (int)y.size(); geom->dim = dim; geom->depth = depth; geom->r = r_knot;
	geom->cut_method = cut_method_use; geom->design = design; geom->seed = seed_use;
	std::vector<int> idx((size_t)geom->n);
	std::iota(idx.begin(),idx.end(),0);
	geometry_subtree(*geom,1,0,std::move(idx),
		Eigen::VectorXd::Zero(dim),Eigen::VectorXd::Ones(dim));
	return geom;
}

// The arithmetic of loglik_subtree for the Full leaf model on a prepared
// node: same rows in the same order (node.idx), same knot rows (node.kn),
// same residual rows (node.res), same routing (node.side); the kernels,
// Schur corrections, factorizations and propagations are the same
// statements in the same order.
StructureStats ResTree::loglik_geometry_subtree(const FullTreeGeometry& geom,
	int id, Eigen::VectorXd&& z, Eigen::MatrixXd&& Q, int task_depth) const{

	StructureStats out;
	const FullTreeNodeGeometry& node = geom.nodes[(size_t)id];
	const int n = (int)node.idx.size();
	if(n==0){ return out; }
	if((int)Q.cols()!=n){
		out.ok = false;
		out.error = "coefficient workspace is not aligned with node observations";
		return out;
	}
	if(!node.split){
		Eigen::MatrixXd Xn(n, dim);
		for(int i=0; i<n; ++i){ Xn.row(i) = X.row(node.idx[(size_t)i]); }
		BlockStats st;
		if(!dense_block(Xn, z, Q, st, nullptr) || !std::isfinite(st.qd) || !std::isfinite(st.ld)){
			out.ok = false;
			out.error = "non-positive-definite full leaf covariance";
			return out;
		}
		out.quad = st.qd; out.logdet = st.ld; out.n_gauss = n;
		return out;
	}

	// ---- split: internal knot block, then the two children ----
	const int rk = (int)node.kn.size();
	const int nres = (int)node.res.size();
	Eigen::MatrixXd Kx, Xres, Qk, Qx;
	Eigen::VectorXd knot_data, zres;
	gather_knot_inputs(X, node.idx, z, Q, node.kn, node.res, Kx, Xres, knot_data, zres, Qk, Qx);
	Q.resize(0,0);
	KnotBlock kb;
	if(!knot_step(Kx, Xres, knot_data, zres, Qk, Qx, KNOT_RESIDUALIZE, kb) ||
	   !std::isfinite(kb.stat.qd) || !std::isfinite(kb.stat.ld)){
		out.ok = false;
		out.error = "non-positive-definite internal knot covariance";
		return out;
	}
	out.quad = kb.stat.qd;
	out.logdet = kb.stat.ld;
	out.n_gauss = rk;
	Eigen::MatrixXd& Qnext = kb.Qnext;
	Eigen::VectorXd& zres_c = kb.zres_c;

	int nl = 0;
	for(int i=0; i<nres; ++i){ if(node.side[(size_t)i]==0){ ++nl; } }
	const int nrr = nres - nl;
	Eigen::VectorXd zl(nl), zr2(nrr);
	Eigen::MatrixXd Qleft(Qnext.rows(), nl), Qright(Qnext.rows(), nrr);
	int il = 0, ir = 0;
	for(int i=0; i<nres; ++i){
		if(node.side[(size_t)i]==0){
			zl(il) = zres_c(i);
			Qleft.col(il++) = Qnext.col(i);
		}else{
			zr2(ir) = zres_c(i);
			Qright.col(ir++) = Qnext.col(i);
		}
	}
	kb = KnotBlock();
	out.n_knot += rk;
	if(nres>0){ out.n_knot_live += rk; }

	StructureStats left, right;
	const int left_index=node.left,right_index=node.right;
#ifdef _OPENMP
	if(node.level < task_depth){
		#pragma omp task shared(geom, left, zl, Qleft) firstprivate(left_index, right_index, task_depth)
		worker_error.run("full-child",[&]{
			left = loglik_geometry_subtree(geom, left_index, std::move(zl), std::move(Qleft), task_depth);
		});
		#pragma omp task shared(geom, right, zr2, Qright) firstprivate(left_index, right_index, task_depth)
		worker_error.run("full-child",[&]{
			right = loglik_geometry_subtree(geom, right_index, std::move(zr2), std::move(Qright), task_depth);
		});
		#pragma omp taskwait
		worker_error.rethrow();
	}else
#endif
	{
		left = loglik_geometry_subtree(geom, left_index, std::move(zl), std::move(Qleft), task_depth);
		right = loglik_geometry_subtree(geom, right_index, std::move(zr2), std::move(Qright), task_depth);
	}
	combine(out, left);
	combine(out, right);
	return out;
}

StructureEval ResTree::evaluate_geometry(const FullTreeGeometry& geom) const{
	worker_error.reset();
	StructureEval ans;
	// Matrix compatibility exports are constructed only when requested.
	if(geom.n != (int)y.size() || geom.dim != dim ||
		geom.r != r_knot || geom.design != design){
		ans.stats.ok = false;
		ans.stats.error = "the prepared tree geometry does not match this model";
		return ans;
	}
	const int td = task_depth();
	const int n = (int)y.size();
	Eigen::VectorXd z = y;
	Eigen::MatrixXd Q0(0, n);
#ifdef _OPENMP
	if(ncores>1){
		#pragma omp parallel num_threads(ncores)
		{
			#pragma omp single
			worker_error.run("full-root",[&]{
				ans.stats = loglik_geometry_subtree(geom, 0, std::move(z), std::move(Q0), td);
			});
		}
	}else
#endif
	{
		ans.stats = loglik_geometry_subtree(geom, 0, std::move(z), std::move(Q0), td);
	}
	worker_error.rethrow();
	if(ans.stats.ok){
		ans.loglik = gaussian_loglik(ans.stats.logdet, ans.stats.quad,
			ans.stats.n_gauss) + ans.stats.white_noise_loglik;
	}
	return ans;
}
