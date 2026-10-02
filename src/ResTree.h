// -*- mode: c++; -*-
//
// ResTree.h: the shared-computation residual-tree engine.
//
// Class layout (one responsibility each):
//   * PathTrie      -- the core dynamic-array storage: a trie of distinct
//                      node-paths (PathNode entries) with child creation/lookup,
//                      path stamps for frontier walks, and the level-scoped
//                      quantile-cut cache list.
//   * ResTree       -- the model and tree machinery: data and covariance, tree
//                      configuration (depth, knots, leaf model, cut methods,
//                      designs, CGM prior), knot selection, terminal evaluations,
//                      shared per-path expansion into a PathTrie, and the
//                      known-theta likelihood of a FIXED structure
//                      (evaluate_structure / evaluate_generated).
//   * SMC (SMC.h) -- the derived sampler: reached-node records,
//                      per-particle RNG streams, transitions, weights,
//                      resampling, and evidence.
//   * Predictor (prediction.h) -- prediction using ResTree and FittedTrees,
//                      independent of the algorithm that supplied the trees.
//
// All path-shared work (knot selection, Cholesky factors, whitening
// coefficients, arrival evaluations) is a deterministic function of a node's
// PATH, computed once per distinct path in the trie (expand_level); particles
// store reached-node records whose draws replay fixed per-particle
// mt19937_64 streams (sample_level).  Under deterministic cuts
// ("middle"/"median") the candidate evaluations and the stop/split posteriors
// (rho, lambda) are also path functions. With the random cut priors
// ("uniform" = the CGM Bayesian CART prior, uniform over the rank intervals
// between consecutive order statistics of the node's residual coordinates;
// "balanced" = the same intervals reweighted by the Beta(2,2) density on the
// rank scale), WN and the nested PP design integrate all rank intervals once
// per path; other PP designs evaluate per-particle candidate cuts.
// SMC sampling: design "maximin"/"boundary"/"nested", baseline "PP"/"WhiteNoise"
// (checked by the wrapper).  Fixed-structure evaluation and prediction
// additionally serve baseline "Full" (dense terminals).

#ifndef RESTREE_MODEL_H
#define RESTREE_MODEL_H

#ifndef _USE_RcppEigen
#define _USE_RcppEigen
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#endif

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <map>
#include <random>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "GPutils.h"   // GPM, ikernel, seq_maximin, maximin_order, covariance-error flags

// ==================== Depth limits and native shared storage ====================

// Single source of truth for the two tree-depth limits, shared by the engine
// (ResTree.h) and the capability query (wrapper.cpp -> R startup.R,
// which hands them to R/validation.R and R/S4classes.R).  See ResTree.h
// ResTree::compare_node for the WN model that makes the two differ.
// Largest INITIAL depth a model may declare (restree_model(depth=)); every leaf
// model. PP and Full also use it as the realized-tree representation cap;
// active trees use reached-node storage, not dense depth-sized arrays.
constexpr int restree_max_model_depth = 20;

// Largest REALIZED WhiteNoise tree depth.  WN has no structural depth boundary
// and grows past the initial depth as the data need; this ceiling is a chosen
// representation limit (25), below the 32-bit node-id bound: ids are int and
// 2*n_internal = 2^(tree_depth+1) must stay below 2^31, so any value up to 29
// is arithmetically safe.  A WN split beyond the ceiling raises
// AdaptiveDepthError (never a forced leaf).  Unbounded WN depth would require
// 64-bit node ids.
constexpr int restree_max_wn_depth = 25;

// BEGIN RESTREE_NATIVE_STORAGE
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

// Broken sampler invariants are not invalid covariance proposals. Keep a
// distinct type through native EB/PMMH error handling and a stable R prefix.
class ParticleStateError : public std::runtime_error {
public:
	explicit ParticleStateError(const std::string& detail)
		: std::runtime_error("SMC internal error: " + detail) {}
};

enum Action : int8_t {
	ACT_ABSENT = -1, // unreached: never materialized in a particle
	ACT_SPLIT = 0,   // internal knot-bearing node
	ACT_STOP = 1,    // model-selected terminal
	ACT_EMPTY = 2,   // no observations reached this child
	ACT_BOTTOM = 3,  // capped PP / fixed-structure terminal
	ACT_PENDING = 4  // reached child awaiting its own comparison
};

struct ParticleNode {
	int8_t act = ACT_ABSENT, jdim = -1;
	int32_t trie_at = -1;
	double Loglik = 0.0, cutval = std::numeric_limits<double>::quiet_NaN(), rhoval = 1.0;
};

// No operator[]: missing-node reads and mutations must never insert a record.
// Creation is explicit, serial, and supplies an initialized reached-node action.
class ParticleNodes {
	std::map<int, ParticleNode> records;
public:
	using iterator = std::map<int, ParticleNode>::iterator;
	using const_iterator = std::map<int, ParticleNode>::const_iterator;
	iterator begin() { return records.begin(); }
	iterator end() { return records.end(); }
	const_iterator begin() const { return records.begin(); }
	const_iterator end() const { return records.end(); }
	const_iterator lower_bound(int id) const { return records.lower_bound(id); }
	const_iterator find(int id) const { return records.find(id); }
	size_t size() const { return records.size(); }
	bool empty() const { return records.empty(); }
	size_t count(int id) const { return records.count(id); }
	ParticleNode* find_node(int id) {
		auto at=records.find(id);
		return at==records.end() ? nullptr : &at->second;
	}
	const ParticleNode* find_node(int id) const {
		auto at=records.find(id);
		return at==records.end() ? nullptr : &at->second;
	}
	ParticleNode& existing(int id) {
		auto* node=find_node(id);
		if(!node) throw ParticleStateError("missing particle node " + std::to_string(id));
		return *node;
	}
	const ParticleNode& existing(int id) const {
		const auto* node=find_node(id);
		if(!node) throw ParticleStateError("missing particle node " + std::to_string(id));
		return *node;
	}
	ParticleNode& insert(int id, const ParticleNode& node) {
		if(id<1 || node.act<ACT_SPLIT || node.act>ACT_PENDING)
			throw ParticleStateError("inserting an invalid/unreached particle node");
		auto added=records.emplace(id,node);
		if(!added.second) throw ParticleStateError("duplicate particle node " + std::to_string(id));
		return added.first->second;
	}
};

inline ParticleNode* worker_particle_node(ParticleNodes& nodes, int id,
	std::atomic<int>& error_node) {
	auto* node=nodes.find_node(id);
	if(!node) { int expected=-1; error_node.compare_exchange_strong(expected,id); }
	return node;
}

// Call only after the parallel region has joined. Worker lookup itself never
// allocates an error message, throws, or calls R.
inline void check_particle_worker_error(const std::atomic<int>& error_node) {
	const int id=error_node.load();
	if(id>=0) throw ParticleStateError("particle/expansion state missing at node " + std::to_string(id));
}

// Carry the changed child's value upward. Two lookups per ancestor, preserving
// exactly knot + (left + right), nearest ancestor first; never use a delta update.
template<class KnotLoglik>
double propagate_particle_loglik(ParticleNodes& nodes, int id, double value,
	const KnotLoglik& knot_loglik) {
	for(int parent_id=id>>1; parent_id>=1; id=parent_id, parent_id>>=1) {
		auto& parent=nodes.existing(parent_id);
		if(parent.act!=ACT_SPLIT || parent.trie_at<0)
			throw ParticleStateError("likelihood ancestor is not a realized split");
		const double sibling=nodes.existing(id^1).Loglik;
		const double left=(id&1) ? sibling : value;
		const double right=(id&1) ? value : sibling;
		parent.Loglik=knot_loglik(parent.trie_at)+(left+right);
		value=parent.Loglik;
	}
	return value;
}

inline uint64_t particle_cut_bits(const double& cut) {
	static_assert(sizeof(double)==sizeof(uint64_t), "64-bit double required");
	uint64_t bits;
	std::memcpy(&bits,&cut,sizeof(bits));
	return bits;
}

// Fingerprints are lookup hints only, never persistent IDs or proof of equality.
// Feed fields, not struct padding or runtime likelihood/trie/stopping state.
struct ParticleFingerprint {
	uint64_t operator()(const ParticleNodes& nodes) const {
		uint64_t hash=UINT64_C(14695981039346656037);
		auto append=[&hash](uint64_t value,int bytes) {
			for(int i=0;i<bytes;++i) {
				hash^=value&UINT64_C(255);
				hash*=UINT64_C(1099511628211);
				value>>=8;
			}
		};
		for(const auto& item:nodes) {
			append((uint32_t)item.first,4);
			append((uint8_t)item.second.act,1);
			append((uint8_t)item.second.jdim,1);
			append(particle_cut_bits(item.second.cutval),8);
		}
		append(nodes.size(),8);
		return hash;
	}
};

inline bool same_particle_structure(const ParticleNodes& a,const ParticleNodes& b) {
	if(a.size()!=b.size()) return false;
	auto j=b.begin();
	for(const auto& i:a) {
		if(i.first!=j->first || i.second.act!=j->second.act ||
			i.second.jdim!=j->second.jdim ||
			particle_cut_bits(i.second.cutval)!=particle_cut_bits(j->second.cutval)) return false;
		++j;
	}
	return true;
}

struct ParticleGroups {
	std::vector<int> representatives, group;
};

// Prediction never reads runtime likelihood, trie index or posterior rho.
// Ordinary alignment: 16 bytes on platforms with 8-byte aligned doubles.
struct PredictionNode {
	int32_t id;
	int8_t act, jdim;
	double cutval;
};

class PredictionNodes {
	std::vector<PredictionNode> records;
public:
	PredictionNodes() = default;
	// Saved rows may be unordered. Keep duplicate/root/parent checks independent
	// of PP/WN's model-specific depth and split-value checks in the wrapper.
	explicit PredictionNodes(std::vector<PredictionNode> rows): records(std::move(rows)) {
		std::sort(records.begin(),records.end(),[](const PredictionNode& a,const PredictionNode& b){return a.id<b.id;});
		if(records.empty() || records.front().id!=1)
			throw std::invalid_argument("every stored tree needs a root");
		int previous=0;
		for(const auto& node:records) {
			if(node.id==previous) throw std::invalid_argument("duplicate stored tree/node id");
			if(node.id<1 || node.act<ACT_SPLIT || node.act>ACT_BOTTOM)
				throw std::invalid_argument("invalid stored node action or id");
			if(node.id>1) {
				const auto* parent=find_node(node.id/2);
				if(!parent || parent->act!=ACT_SPLIT)
					throw std::invalid_argument("a stored node has no splitting parent");
			}
			previous=node.id;
		}
	}
	// The sampler's map is already sorted and reached-node validated. Copy only
	// final fields, once, without an intermediate deep copy of any map.
	static PredictionNodes compact(const ParticleNodes& source) {
		PredictionNodes out;
		out.records.reserve(source.size());
		for(const auto& item:source) {
			const auto& node=item.second;
			if(node.act<ACT_SPLIT || node.act>ACT_BOTTOM)
				throw ParticleStateError("prediction requires completed particle nodes");
			out.records.push_back({item.first,node.act,node.jdim,node.cutval});
		}
		return out;
	}
	const PredictionNode* find_node(int id) const {
		auto at=std::lower_bound(records.begin(),records.end(),id,
			[](const PredictionNode& node,int key){return node.id<key;});
		return at==records.end() || at->id!=id ? nullptr : &*at;
	}
	size_t size() const { return records.size(); }
	size_t storage_bytes() const { return records.capacity()*sizeof(PredictionNode); }
	std::vector<PredictionNode>::const_iterator begin() const { return records.begin(); }
	std::vector<PredictionNode>::const_iterator end() const { return records.end(); }
};

inline bool has_pending_particles(const std::vector<ParticleNodes>& particles,int lo,int hi) {
	for(const auto& particle:particles)
		for(auto at=particle.lower_bound(lo);at!=particle.end() && at->first<hi;++at)
			if(at->second.act==ACT_PENDING) return true;
	return false;
}

// Merge already ordered particle frontiers. No per-record allocation and no
// depth-sized bitmap or vector of all duplicated IDs: O(P + U) auxiliary space,
// where P is particles and U is the number of distinct pending IDs returned.
inline std::vector<int> merge_pending_particles(const std::vector<ParticleNodes>& particles,
	int lo,int hi) {
	struct Cursor { ParticleNodes::const_iterator at,end; };
	std::vector<Cursor> heap;
	heap.reserve(particles.size());
	auto advance=[hi](Cursor& cursor) {
		while(cursor.at!=cursor.end && cursor.at->first<hi && cursor.at->second.act!=ACT_PENDING)
			++cursor.at;
		return cursor.at!=cursor.end && cursor.at->first<hi;
	};
	auto later=[](const Cursor& a,const Cursor& b){return a.at->first>b.at->first;};
	int max_id=-1;
	for(const auto& particle:particles) {
		Cursor cursor{particle.lower_bound(lo),particle.end()};
		if(advance(cursor)) {
			max_id=std::max(max_id,cursor.at->first);
			heap.push_back(cursor);
		}
	}
	std::make_heap(heap.begin(),heap.end(),later);
	std::vector<int> ids;
	while(!heap.empty()) {
		// Resampling often makes every live frontier identical. Advance that
		// shared ID in one linear pass, without P separate heap pop/push pairs.
		// max_id remains exact: outside this branch the removed minimum is
		// strictly below the maximum, and cursors only move forward.
		if(heap.front().at->first==max_id) {
			if(ids.empty() || ids.back()!=max_id) ids.push_back(max_id);
			size_t live=0;
			max_id=-1;
			int min_id=std::numeric_limits<int>::max();
			for(size_t k=0;k<heap.size();++k) {
				Cursor cursor=heap[k]; ++cursor.at;
				if(advance(cursor)) {
					max_id=std::max(max_id,cursor.at->first);
					min_id=std::min(min_id,cursor.at->first);
					heap[live++]=cursor;
				}
			}
			heap.resize(live);
			if(min_id!=max_id) std::make_heap(heap.begin(),heap.end(),later);
			continue;
		}
		std::pop_heap(heap.begin(),heap.end(),later);
		Cursor cursor=heap.back(); heap.pop_back();
		const int id=cursor.at->first;
		if(ids.empty() || ids.back()!=id) ids.push_back(id);
		++cursor.at;
		if(advance(cursor)) {
			max_id=std::max(max_id,cursor.at->first);
			heap.push_back(cursor);
			std::push_heap(heap.begin(),heap.end(),later);
		}
	}
	return ids;
}

// Injectable hasher permits forced-collision native tests without any R export
// or production test switch. Collision chains occupy O(number of groups) space.
template<class Fingerprint=ParticleFingerprint>
ParticleGroups group_particle_structures(const std::vector<ParticleNodes>& particles,
	const Fingerprint& fingerprint=Fingerprint()) {
	ParticleGroups out;
	out.group.resize(particles.size());
	std::unordered_map<uint64_t,int> heads;
	std::vector<int> next;
	for(size_t p=0;p<particles.size();++p) {
		const uint64_t hash=fingerprint(particles[p]);
		auto head=heads.find(hash);
		int g=head==heads.end() ? -1 : head->second;
		while(g>=0 && !same_particle_structure(particles[p],particles[out.representatives[g]]))
			g=next[g];
		if(g<0) {
			g=(int)out.representatives.size();
			out.representatives.push_back((int)p);
			next.push_back(head==heads.end() ? -1 : head->second);
			if(head==heads.end()) heads.emplace(hash,g);
			else head->second=g;
		}
		out.group[p]=g;
	}
	return out;
}
// END RESTREE_NATIVE_STORAGE

// BEGIN RESTREE_NATIVE_PARALLEL
#include <atomic>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef RESTREE_TEST_FAULTS
#include <cstdlib>
#include <cstring>
#include <new>
#endif

// Constructed on the calling thread, after every worker has joined. Keep a
// distinct fatal type: optimizers must not turn a broken worker into -Inf.
class WorkerFailure : public std::runtime_error {
public:
    explicit WorkerFailure(const char* reason)
        : std::runtime_error(std::string("ResTree worker error: ") + reason) {}
};

// One error slot per evaluation, shared by its tasks, never by separate
// models. The winning worker publishes an exception_ptr without formatting
// strings or calling R. Reset/rethrow are only used after joining all workers.
class ParallelError {
    // 0 = clear, 1 = publishing, 2 = ready. A different task can notice the
    // failure before the winner finishes publishing its exception pointer.
    std::atomic<int> state_{0};
    std::exception_ptr exception_;
#ifdef RESTREE_TEST_FAULTS
    std::string fault_site_;
    unsigned run_ = 0;
    bool inject_ = false;
    std::atomic<bool> injected_{false};
#endif
public:
    ParallelError() = default;
    // ResTree is a value type. Copies start with an independent empty slot.
    ParallelError(const ParallelError&) noexcept {}
    ParallelError& operator=(const ParallelError&) noexcept {
        exception_ = nullptr; state_.store(0); return *this;
    }
    void reset() {
        exception_ = nullptr;
        state_.store(0, std::memory_order_relaxed);
#ifdef RESTREE_TEST_FAULTS
        ++run_;
        const char* site = std::getenv("RESTREE_TEST_FAULT");
        fault_site_ = site ? site : "";
        const char* run = std::getenv("RESTREE_TEST_FAIL_RUN");
        inject_ = !run || std::strtoul(run,nullptr,10)==run_;
        injected_.store(false);
#endif
    }
    bool failed() const noexcept { return state_.load(std::memory_order_relaxed)!=0; }
    void capture() noexcept {
        int expected = 0;
        if(state_.compare_exchange_strong(expected,1,std::memory_order_relaxed)) {
            exception_ = std::current_exception();
            state_.store(2,std::memory_order_release);
        }
    }
    template<class F> void run(const char* site, F&& work) noexcept {
        if(failed()) return;
        try {
#ifdef RESTREE_TEST_FAULTS
            if(inject_ && fault_site_==site && !injected_.exchange(true))
                throw std::bad_alloc();
#else
            (void)site;
#endif
            work();
        } catch(...) { capture(); }
    }
    // After a child taskwait, stop this computation before reading partially
    // filled outputs. Inside a worker its own run() catches this exception;
    // outside workers the original failure reaches the R boundary directly.
    void rethrow() const {
        if(!failed()) return;
        while(state_.load(std::memory_order_acquire)==1) {}
        try { std::rethrow_exception(exception_); }
        catch(const std::bad_alloc&) { throw; }
        catch(const WorkerFailure&) { throw; }
        catch(const std::exception& e) { throw WorkerFailure(e.what()); }
        catch(...) { throw WorkerFailure("unknown native exception"); }
    }
};

// An ordinary C++ loop is essential on the serial branch. An OpenMP
// parallel-for with if(false) creates an inactive inner region, hiding an
// active outer team from Eigen's omp_get_num_threads() nesting guard.
template<class F> void parallel_for(ParallelError& error, const char* site,
    int count, int threads, F&& work) {
#ifdef _OPENMP
    if(threads>1 && !omp_in_parallel()) {
        #pragma omp parallel for schedule(dynamic) num_threads(threads)
        for(int i=0;i<count;++i) error.run(site,[&]{ work(i); });
    } else
#endif
    {
        for(int i=0;i<count;++i) error.run(site,[&]{ work(i); });
    }
    // This helper is called from serial sampling or from an enclosing guarded
    // task, never from an unguarded OpenMP structured block.
    error.rethrow();
}
// END RESTREE_NATIVE_PARALLEL

// Sufficient statistics of one block of the residual-tree factorisation, on the
// CORRELATION scale.  Blocks compose by addition; node routines return these,
// not likelihoods.
struct BlockStats {
	double ld = 0.0;   // log|Sigma_b|
	double qd = 0.0;   // z_b' Sigma_b^{-1} z_b
	int    n  = 0;     // rows in the block
	void clear(){ ld = 0.0; qd = 0.0; n = 0; }
	BlockStats& operator+=(const BlockStats& o){ ld+=o.ld; qd+=o.qd; n+=o.n; return *this; }
	BlockStats& operator-=(const BlockStats& o){ ld-=o.ld; qd-=o.qd; n-=o.n; return *this; }
};
inline BlockStats operator+(BlockStats a, const BlockStats& b){ a+=b; return a; }
inline BlockStats operator-(BlockStats a, const BlockStats& b){ a-=b; return a; }

// The one global sigma^2 enters the likelihood ONLY through this conditional
// closed form in (ld, qd, n): sigma^2 is never profiled or integrated out.
double restree_loglik_conditional(double ld, double qd, int n, double sig2);

double log_sum_exp(const Eigen::VectorXd& x);
double log_sum_exp2(double x, double y);
double spike_slab_loglik(const double& loglik_baseline,
	const double& loglik_knot, const Eigen::VectorXd& loglik_child,
	const double& rho, const Eigen::VectorXd& lambda);
Eigen::VectorXi ancestor_ids(int id);

// The knot-block step of the residual-tree factorisation at one node, on
// the rows the caller has already gathered: knot rows Kx with values
// knot_data, residual rows Xres with values zres, and the propagated
// coefficients Qk / Qx of the ancestors (q rows; q = 0 at the root).
//   V      = K(Kx, Kx) + nugget I - Qk' Qk          (knot covariance)
//   B      = K(Xres, Kx) - Qx' Qk                    (residual-knot cross block)
//   L      = chol(V),  Ry = L^-1 knot_data,  stat = (log|V|, Ry'Ry, r_k)
//   Qnext  = [Qx ; L^-1 B']                          (coefficients incl. this level)
//   zres_c = zres - B L^-T Ry                        (residualised values)
// Every route of the engine that factorises a knot block -- the SMC
// expansion (expand / descend), the fixed-structure evaluators
// (loglik_subtree, loglik_geometry_subtree), the PP leaf
// (leaf_pp_loglik_core) and prediction (predict_subtree) -- calls
// ResTree::knot_step, so the numbers agree by construction rather than by
// matching statements across functions.
enum KnotStepLevel : int {
	KNOT_FACTOR = 0,      // V, L, Ry, stat only        (a PP terminal in prediction)
	KNOT_PROPAGATE = 1,   // + B, W = L^-1 B'           (the PP leaf)
	KNOT_RESIDUALIZE = 2  // + Qnext = [Qx; W], zres_c  (a split node; W is then
	                      //   the bottom rows of Qnext and kb.W stays empty)
};

// knot_step's residual-sized work -- the cross kernel B, its downdate by the
// ancestors' coefficients, and the triangular solve L^-1 B' that forms W /
// Qnext -- is independent across residual rows, so it is done in fixed-width
// column panels.  Two reasons the width is a constant rather than a function
// of n_res or of the thread count:
//   * the panels are the unit of intra-node parallelism.  Without them the
//     only parallelism in the engine is across sibling subtrees, and the top
//     levels of a binary tree have almost none: the root's O(r^2 n) solve runs
//     on one thread while the rest of the team waits at the task barrier.
//     With depth D levels of roughly equal work and min(2^l, T) parallelism at
//     level l, that caps the speedup at sum_l (l+1) / sum_l (l+1)/2^l, about
//     16x at D = 11 whatever T is.
//   * Eigen chooses its own blocking from the operand shapes, so the grouping
//     of the sums -- hence the last bits -- depends on how many columns a call
//     is given.  A width fixed here makes that grouping a property of the
//     panel and not of ncores, which is what keeps a run reproducible across
//     thread counts.  A node with n_res <= this width is one panel and is
//     computed exactly as an unpanelled call would compute it.
static const int restree_solve_panel = 1024;
struct KnotBlock {
	Eigen::MatrixXd L;       // Cholesky factor of the knot covariance
	Eigen::MatrixXd B;       // residual-knot cross block after the downdate (KNOT_PROPAGATE only)
	Eigen::MatrixXd W;       // L^-1 B'  (KNOT_PROPAGATE only)
	Eigen::MatrixXd Qnext;   // (q + r_k) x n_res  (KNOT_RESIDUALIZE only; B is then never stored)
	Eigen::VectorXd Ry;      // L^-1 knot_data
	Eigen::VectorXd zres_c;  // residualised values of the residual rows
	BlockStats stat;         // (ld, qd, n = r_k) of the knot block
};

// Trie child edge.  Deterministic cuts have at most one child per
// (J, side); quantile-sampled cuts discriminate by the realized cut value.
struct ChildEdge {
	int8_t J = -1;
	int8_t side = 0;
	double cut = 0.0;   // realized cut (cut-keyed) or the cell's representative cut (cell-keyed)
	int cell = -1;      // rank cell of the integrated WN proposal, -1 when cut-keyed
	int idx = -1;
};

// One interval in the interpolated rank-cut prior. Equal endpoints denote
// a point mass from tied coordinates; n_left uses the exact x < cut rule.
struct WNCutInterval {
 int coordinate = 0, rank_index = 0, n_left = 0;
 double cut_lower = 0.0, cut_upper = 0.0;
 double left_loglik = 0.0, right_loglik = 0.0;
};

// Immutable, level-scoped WN proposal shared by all requests for a path.
// Leaves of log_mass_tree store log(lambda_j * prior_interval_mass * L * R).
// Internal entries are log sums, supporting O(log(number of intervals)) draws
// without subtracting nearly equal cumulative probabilities.
struct WNCutProposal {
 int n_residual = 0;
 size_t leaf_offset = 1;
 std::vector<WNCutInterval> intervals;
 std::vector<double> log_mass_tree;
 double log_stop_mass = 0.0, log_split_mass = 0.0;
 double log_weight_increment = 0.0;
};

// The integrated cut proposal of a PP node under the nested knot design
// (design = "nested", tree_engine.cpp): WN's rank-cell table (intervals,
// masses, log-mass tree, stop/split masses) with PP leaf values from the
// all-cell sweeps, plus the Gaussian block statistics of both candidate
// children per interval (the child's arrival stat_leaf once realised).
struct NestedCutProposal {
	WNCutProposal table;
	std::vector<BlockStats> left_stat, right_stat;
};

// Numeric residual state: consumed by child expansion, optionally retained
// across sig2-only runs. The knot/cut geometry lives separately on ResTree.
struct PostKnotCache {
	std::vector<int> row_ids;
	Eigen::VectorXd residual;
	Eigen::MatrixXd coefficients;
	bool ready = false;
};
// Ordinary PP candidate draws consume these matrices in their own level.
struct CutDrawWorkspace {
	Eigen::MatrixXd input, sorted;
};
// Nested proposals are always level-local. WN proposals may survive scale
// refresh; retaining one does not imply retaining the post-knot matrices.
struct CutProposalState {
	std::shared_ptr<WNCutProposal> wn;
	std::shared_ptr<const NestedCutProposal> nested;
};
enum class CacheRetention { none, postknot, wn_proposal };
struct CacheLevel {
	std::vector<int> paths;
	size_t bytes = 0;
};
struct LevelCacheRegistry { CacheLevel current, previous; };
struct RetainedCacheBytes {
	size_t postknot = 0, wn_proposal = 0;
	size_t total() const { return postknot + wn_proposal; }
};

// Everything later stages need about one distinct node-path.
struct PathNode {
	int level = 0;
	int tree_id = 0;        // node id in the complete binary tree
	int parent = -1;        // trie index of the parent path (-1 for the root)
	int8_t j_from_parent = -1;
	int8_t side_from_parent = 0;
	double cut_from_parent = 0.0; // realized cut of the edge from the parent
	int n_arrival = 0;

	double loglik_base = 0.0;  // arrival (stop) evaluation
	BlockStats stat_leaf;      // Gaussian stats of the arrival terminal block (PP)

	double loglik_knot = 0.0;  // internal knot block
	BlockStats stat_knot;

	// ---- deterministic-cut mode only: per-path candidate results ----
	Eigen::VectorXd cuts;          // realized cut per dimension
	Eigen::VectorXd child_loglik;  // per dim: left + right child arrival loglik
	Eigen::MatrixXd child_loglik_lr;     // dim x 2 individual child arrival logliks
	std::vector<BlockStats> child_stats; // 2*dim entries: [2*J+side]
	Eigen::ArrayXXi child_n;       // dim x 2 child sizes
	double rho = 1.0, rho_log = 0.0;
	Eigen::VectorXd lambda, lambda_log;

	PostKnotCache postknot;
	CutDrawWorkspace cut_workspace;
	CutProposalState proposals;
	CacheRetention retention = CacheRetention::none;

	Eigen::VectorXd box_min, box_max;

	// ---- nested knot design (PP, random cuts; tree_engine.cpp) ----
	std::vector<int> nested_cand;   // global row ids of the node's candidate knots C_v (maximin order); persists

	std::vector<int> child_path;      // deterministic mode: [2*J+side] -> trie idx
	std::vector<ChildEdge> child_edges; // quantile mode: realized edges
	// Quantile mode: index of child_edges by key -- the cut's 64-bit pattern
	// (cut-keyed edges, PathTrie::child_random) or (cell, J, side) packed
	// (cell-keyed edges, child_cell); a node holds one kind only.  The commit
	// pass finds candidates by hash, then checks the exact edge identity.
	std::unordered_multimap<uint64_t, int> child_lookup;
	bool expanded = false;
	bool need_expand = false;
};

// The core dynamic-array path store: distinct node-paths in creation order,
// child creation/lookup, frontier stamps, and the level-scoped cache list.
// The SMC sampler (SMC) drives one of these.
class PathTrie {
public:
	std::vector<PathNode> nodes;
	std::vector<int> path_stamp;
	int stamp = 0;
	// Current/previous expansion levels are independent of proposal ownership.
	// Admission uses current bytes; uncached children re-derive via descend().
	LevelCacheRegistry level_cache;
	std::vector<int> nested_proposal_paths;
	size_t cache_budget_bytes = static_cast<size_t>(512) << 20;
	RetainedCacheBytes retained_bytes;

	PathNode& operator[](size_t i){ return nodes[i]; }
	const PathNode& operator[](size_t i) const { return nodes[i]; }
	size_t size() const { return nodes.size(); }
	PathNode* data(){ return nodes.data(); }
	void push_back(PathNode&& v){ nodes.push_back(std::move(v)); }
	void reserve(size_t n){ nodes.reserve(n); }
	void clear(){
		nodes.clear();
		nested_proposal_paths.clear();
		level_cache.current.paths.clear();
		level_cache.previous.paths.clear();
		level_cache.current.bytes = level_cache.previous.bytes = 0;
		retained_bytes = RetainedCacheBytes{};
	}
	void shrink_to_fit(){ nodes.shrink_to_fit(); }

	// Deterministic-cut child: at most one per (J, side).
	int child(int parent_idx, int J, int side, int dim);
	// Quantile-cut child: discriminated by the realized cut's bit pattern.
	int child_random(int parent_idx, int J, int side, double cut,
		int n_child, double ll, const BlockStats& st);
	// Integrated-WN child under a box-free knot design: discriminated by the
	// rank cell, so every particle that draws the same cell shares one node.
	// cut_rep (the cell's upper endpoint) partitions the rows exactly as any
	// realized cut inside the cell does; the realized cut itself stays per
	// particle in SMC::cutval.
	int child_cell(int parent_idx, int J, int side, int cell, double cut_rep,
		int n_child, double ll, const BlockStats& st);
};

struct EngineDiagnostics {
	double expand_sec = 0.0;
	double sample_sec = 0.0;
	int n_trie = 0;
	int n_expanded = 0;
	size_t cache_peak_bytes = 0;   // largest live post-knot cache (two levels) during the run
	Eigen::VectorXi distinct_per_level;
};

// ---- known-theta fixed-structure evaluation results ----
struct StructureStats {
	double quad = 0.0;
	double logdet = 0.0;
	double white_noise_loglik = 0.0;
	int n_gauss = 0;
	int n_knot = 0;
	// Internal knot blocks whose split actually conditions a populated child
	// (the conditioning budget to line up against a Vecchia neighbour count;
	// a block whose split leaves both children empty conditions nothing).
	int n_knot_live = 0;
	int n_leaf_knot = 0;
	int n_white_noise_leaf = 0;
	int n_white_noise_obs = 0;
	bool ok = true;
	std::string error;
};

struct StructureEval {
	StructureStats stats;
	Eigen::VectorXd S;
	Eigen::VectorXd J;
	Eigen::VectorXd cuts;
	double loglik = -std::numeric_limits<double>::infinity();
};

// The theta-INDEPENDENT geometry of one distinct node-path under
// deterministic cuts ("middle"/"median"): everything expand() derives from
// the node's rows alone.  The row set of a path and its order are fixed by
// (X, path) -- the root holds rows 0..n-1 and partition_child() keeps node
// order -- so the maximin knot selection (pick_knots on those rows), the
// residual rows it leaves, the cuts (box midpoint, or the median of the
// residual rows) and, for the PP leaf model, the knot selection inside every
// candidate child are all pure functions of (X, path); theta enters only the
// kernels and factorizations computed afterwards.  An engine that evaluates
// the same data at many thetas (EB, PMMH) keeps these across runs, keyed by
// the path (ResTree::path_cache); the numbers are exactly the ones
// pick_knots / the cut rule would recompute, so the results are bit-identical.
struct PathCacheEntry {
	Eigen::VectorXi ind_knot;  // pick_knots output on the node rows (node-local ids)
	Eigen::VectorXi ind_res;   // pick_knots output: residual rows, node-local, node order
	Eigen::VectorXd cuts;      // realized cut per dimension
	// PP only: per candidate child [2*J+side] with n_child > r, the leaf
	// knot selection of leaf_pp_loglik on the child rows (child-local ids;
	// the residual rows are the ascending complement).  Empty otherwise.
	std::vector<Eigen::VectorXi> child_kn;
	size_t bytes = 0;
};

// The theta-INDEPENDENT geometry of the generated Full tree
// (fixed_tree_fit_cpp): which rows reach each node and in what order, which
// of them the knot design selects, the cut, and which residual rows go left
// or right.  Every one of these is a pure function of (X, depth, r, design,
// cut_method, seed) -- the stopping rule is "n <= r"; storage depth extends
// as needed. Knots come from pick_knots on node rows, cuts from residual
// rows and the per-node RNG stream. The R model stores these choices once;
// Full MLE prepares this native object once per fit; public Full likelihoods
// use a model-local cache with exact invalidation. Numerical factors are rebuilt.
struct FullTreeNodeGeometry {
	std::vector<int> idx;      // global rows of the node, in node order
	Eigen::VectorXi kn, res;   // node-local knot / residual positions (pick_knots output)
	std::vector<char> side;    // per residual position: 0 = left child, 1 = right child
	bool split = false;        // false: terminal Full leaf on idx
	int level = 0, id = 1, J = -1, left = -1, right = -1;
	double cut = std::numeric_limits<double>::quiet_NaN();
};
struct FullTreeGeometry {
	int n = 0, dim = 0, depth = 0, r = 0;
	std::string cut_method, design;
	unsigned int seed = 0;
	std::vector<FullTreeNodeGeometry> nodes; // reached records; explicit child indices
	size_t bytes = 0;
};

// A selected adaptive split cannot be represented; never turn this into
// an invalid-theta rejection inside EB or PMMH.
class AdaptiveDepthError : public std::runtime_error {
public:
	using std::runtime_error::runtime_error;
};

// The residual-tree model and its tree machinery.  Owns the data (not
// references): a fitted engine can be retained as an external pointer beyond
// the constructing call, so it must not alias R memory or stack-lifetime
// conversions.  The constructors take the data BY VALUE and move it into the
// members, so the one owned copy is made exactly at the construction site:
// from an Eigen::Map over R memory (the wrapper's ModelSpec) that is one
// copy; from an rvalue MatrixXd it is a move and no copy at all.
// The C++ tree engine is a global class. R's S4 class remains `restree`.
class ResTree {
public:
	Eigen::VectorXd y;
	Eigen::MatrixXd X;
	GPM covpar;
	int depth = 1, r_knot = 1, dim = 1;
	std::string baseline = "PP", cut_method = "middle", design = "maximin";
	// Ordinary PP random cuts: balanced uses M IID prior draws per dimension.
	// Uniform uses one draw from the midpoint-containing (or nearest) rank
	// cell plus M draws from its complement. Stratum weights are 1/g and
	// (1-1/g)/M for g rank cells, preserving the original cut prior. A single
	// cell needs one draw only. The weighted child marginals estimate the
	// integrated slab marginal without bias; stop/dimension/candidate draws
	// and the direct increment use this SAME estimate (unbiased Zhat, not
	// log Zhat). Repeated random draws are not general exact enumeration.
	// Ignored by the
	// WN leaf and by the nested design (both integrate the cut exactly) and
	// by deterministic cuts.
	// The model reader supplies M = 1 outside ordinary uniform PP when omitted.
	int cut_candidates = 30;
	double prior_rho = 0.5, prior_beta = 0.0;
	Eigen::VectorXd prior_lambda;
	// Knots a split spends: min(r, n).  The SMC always splits adaptively -- a
	// nonempty node with n < r may split, spending every row as a knot (the
	// children are then empty and the split is a terminal dense Gaussian
	// block) -- and the fixed-structure evaluator, the stored-structure
	// rebuild and prediction rescore a split with the same rule, so every
	// route agrees.  An empty node spends nothing.
	int knot_count(int n) const { return n <= 0 ? 0 : std::min(r_knot, n); }
	// WN has no structural depth boundary: every nonempty node compares stop
	// versus split, at every level, until it runs out of data.  A node with
	// n <= r spends all n rows as knots (knot_count = min(r,n)); its residual
	// leaf is then empty, both children are empty, and the split still competes
	// with the stop leaf.  Because each split passes only n - min(r,n) residual
	// rows down, every branch reaches n = 0 in at most ceil(n / r) levels, so
	// the tree is always finite; the depth ceiling (max_tree_depth) is an
	// representation guard, not a termination guard.  Positive WN depth is
	// only the initial comparison depth (the default particle budget is 100 * depth,
	// wrapper.cpp restree_default_particles); depth zero specifies a single
	// WN leaf irrespective of size.  PP keeps the level < depth cap.  Full
	// does not sample trees: its one fixed tree is built by the size rule
	// (fixed_tree_geometry, which extends past the requested depth until
	// every terminal holds at most r rows).  The local prior/posterior is
	// unchanged.
	bool compare_node(int level, int n) const {
		return n > 0 && depth > 0 && (level < depth || baseline=="WhiteNoise");
	}

	// Threads (wrapper.cpp, restree_thread_policy): ncores = the thread
	// count of the engine's own OpenMP regions (1 = none is opened),
	// eigen_threads = Eigen::setNbThreads() outside them, ncores_request =
	// the caller's value.  Set together by the wrapper's
	// configure_threads; re-applied by restree_apply_threads at every
	// evaluation of a retained engine.
	int ncores = 1;
	int eigen_threads = 1;
	int ncores_request = 1;
	mutable ParallelError worker_error;
	bool random_cuts = false;  // set from cut_method in SMC::run()

	// The tasked top of the fixed-structure recursion (loglik_subtree,
	// loglik_geometry_subtree) is fully resident: a node spawns both children
	// as tasks and waits, so every node at levels 0..d holds its own
	// (q + r_k) x n_res coefficient block at the same time.  The 2^l nodes of
	// level l partition the n rows and carry q = l*r coefficients each, so the
	// level totals (l+1)*r*n*8 bytes whatever 2^l is, and levels 0..d total
	//     r * n * 8 * (d+1)(d+2)/2.
	// Taking d = ceil(log2(ncores)) -- one task per team member at the deepest
	// tasked level -- therefore makes peak memory grow like log(ncores)^2 for
	// a speedup that is already flat by then (see restree_solve_panel): at
	// r = 60, n = 1e5 it is 0.19 GB serial, 1.01 GB at 20 threads and 1.34 GB
	// at 40; at r = 200, n = 1e6, 6.4 / 33.6 / 44.8 GB.  task_depth() instead
	// takes the largest d whose closed-form footprint fits task_budget_bytes,
	// capped by ceil(log2(ncores)) because a deeper cut buys no extra
	// concurrency.  d = 0 leaves the recursion serial inside the team, which
	// is not a loss of parallelism: knot_step's panels keep the team busy
	// within each node.
	size_t task_budget_bytes = static_cast<size_t>(512) << 20;
	int task_depth() const;

	// ---- path-keyed cache of the theta-independent expansion geometry ----
	// Off by default (a one-shot engine expands every path once, so there is
	// nothing to reuse); the persistent evidence engine (smc_engine_new_cpp)
	// turns it on.  Never consulted under quantile cuts.  Key: the path as
	// two bytes (J, side) per level from the root -- exact for any depth and
	// d <= 255 (the engine accepts d <= 127).  Budget: when an insertion
	// would exceed path_cache_budget_bytes the whole cache is dropped and
	// refilled (recent paths are the ones that recur).  The map is shared by
	// the OpenMP expansion tasks: every lookup and insertion runs inside
	// `omp critical(restree_path_cache)`; entries are immutable once
	// inserted and handed out as shared_ptr, so a flush cannot pull an entry
	// from under a task still reading it.
	bool path_cache_enabled = false;
	size_t path_cache_budget_bytes = static_cast<size_t>(256) << 20;
	mutable std::unordered_map<std::string,
		std::shared_ptr<const PathCacheEntry>> path_cache;
	mutable size_t path_cache_bytes = 0;
	mutable unsigned long path_cache_hits = 0, path_cache_misses = 0,
		path_cache_flushes = 0;
	std::shared_ptr<const PathCacheEntry> path_cache_get(const std::string& key) const;
	void path_cache_put(const std::string& key,
		std::shared_ptr<const PathCacheEntry> entry) const;
	void path_cache_clear() const;
	// The key of a trie node: its (J, side) edge labels from the root.
	static std::string path_key(const PathTrie& trie, const PathNode& v);

	ResTree(Eigen::VectorXd y_, Eigen::MatrixXd X_)
		: y(std::move(y_)), X(std::move(X_)) { dim = (int)X.cols(); }

	// CGM tree prior: p_split(level) = (1 - prior_rho) / (1 + level)^beta.
	// beta = 0 must return prior_rho EXACTLY (no 1-(1-rho) round trip) for
	// seed-for-seed reproducibility.
	double prior_stop(int level) const {
		if(prior_beta==0.0){ return prior_rho; }
		const double split = (1.0-prior_rho) /
			std::pow(1.0 + static_cast<double>(level), prior_beta);
		return 1.0 - std::max(0.0, std::min(1.0, split));
	}

	// r_use: number of knots to spend (callers pass knot_count(n) at split
	// nodes and r_knot for PP leaf evaluations).
	void pick_knots(const Eigen::MatrixXd& Xn, const Eigen::VectorXd& bmin,
		const Eigen::VectorXd& bmax, Eigen::VectorXi& ind_knot,
		Eigen::VectorXi& ind_res, int r_use) const;

	// ---- the two block factorisations every route is built from ----
	// knot_step: see KnotBlock.  Returns false when the knot covariance is
	// not positive definite (kb.L is then unusable); it does NOT raise the
	// global covariance-error flag -- the caller decides (the SMC records it,
	// the fixed-structure evaluator reports it in StructureStats::error).
	// The (ld, qd) of kb.stat are computed whenever the factorisation
	// succeeds; the caller checks finiteness where it matters.
	bool knot_step(const Eigen::MatrixXd& Kx, const Eigen::MatrixXd& Xres,
		const Eigen::VectorXd& knot_data, const Eigen::VectorXd& zres,
		const Eigen::MatrixXd& Qk, const Eigen::MatrixXd& Qx,
		int level, KnotBlock& kb) const;
	// The two pieces knot_step is built from, shared with the leaf evaluator:
	// the r_k x r_k knot block (V, L, Ry, stat) and one residual column panel
	// (cross block, downdate, W columns).  See knot_cross_panel in
	// tree_engine.cpp for why the panel is the unit.
	bool knot_factor(const Eigen::MatrixXd& Kx, const Eigen::VectorXd& knot_data,
		const Eigen::MatrixXd& Qk, KnotBlock& kb) const;
	void knot_cross_panel(const Eigen::MatrixXd& Xp, const Eigen::MatrixXd& Kx,
		const Eigen::Ref<const Eigen::MatrixXd>& Qxp, const Eigen::MatrixXd& Qk,
		const Eigen::MatrixXd& L, Eigen::MatrixXd& Bp, Eigen::MatrixXd& Wp) const;
	// dense_block: the dense Gaussian block of a terminal with every row a
	// knot, V = K(Xn, Xn) + nugget I - Q'Q, L = chol(V), stat = (log|V|,
	// z'V^-1 z, n).  Same contract as knot_step for failure; L is returned
	// for prediction (the leaf's conditional needs it), pass nullptr when
	// only the statistics are wanted.
	bool dense_block(const Eigen::MatrixXd& Xn, const Eigen::VectorXd& z,
		const Eigen::MatrixXd& Q, BlockStats& st, Eigen::MatrixXd* L_out) const;

	// Terminal-model evaluations (pure; box needed only by the boundary design).
	double terminal_loglik(const Eigen::MatrixXd& Xn, const Eigen::VectorXd& z,
		const Eigen::MatrixXd& Q, const Eigen::VectorXd& bmin,
		const Eigen::VectorXd& bmax, BlockStats& st) const;
	// terminal_loglik on the rows `rows` of a parent's block, ungathered.
	double terminal_loglik_rows(const Eigen::MatrixXd& Xsrc, const Eigen::VectorXd& zsrc,
		const Eigen::MatrixXd& Qsrc, const int* rows, int n_rows,
		const Eigen::VectorXd& bmin, const Eigen::VectorXd& bmax, BlockStats& st,
		const Eigen::VectorXi* kn_in = nullptr,
		Eigen::VectorXi* kn_out = nullptr) const;
	// kn_in: a stored leaf knot selection to use instead of pick_knots (the
	// residual rows are its ascending complement, exactly what pick_knots
	// returns); kn_out: receives the selection made.  Both optional.
	double leaf_pp_loglik(const Eigen::MatrixXd& Xn, const Eigen::VectorXd& z,
		const Eigen::MatrixXd& Q, const Eigen::VectorXd& bmin,
		const Eigen::VectorXd& bmax, BlockStats& st,
		const Eigen::VectorXi* kn_in = nullptr,
		Eigen::VectorXi* kn_out = nullptr) const;
	// The same leaf on the rows `rows` (positions into Xsrc / zsrc / Qsrc,
	// ascending; nullptr = all n_total rows) without gathering the block:
	// score-only, panel by panel, bit-identical to leaf_pp_loglik on the
	// gathered block (tree_engine.cpp).  Every candidate-child evaluation
	// of the sampler goes through this form.
	double leaf_pp_loglik_rows(const Eigen::MatrixXd& Xsrc, const Eigen::VectorXd& zsrc,
		const Eigen::MatrixXd& Qsrc, const int* rows, int n_total,
		const Eigen::VectorXd& bmin, const Eigen::VectorXd& bmax, BlockStats& st,
		const Eigen::VectorXi* kn_in = nullptr,
		Eigen::VectorXi* kn_out = nullptr) const;
	// The leaf from the factors expand's split branch holds: the Cholesky
	// factor L of V (chol_ok: its LLT reported success), W = L^-1 B' (a block
	// of the split's Qnext), zk = L^-1 zk_data and the residualised values
	// zr = zres - B alpha (the split's zres_c).  V is used as formed (no
	// symmetrisation), the split's convention.
	double leaf_pp_loglik_from_factors(int n_total, bool chol_ok,
		const Eigen::MatrixXd& L, const Eigen::Ref<const Eigen::MatrixXd>& W,
		const Eigen::VectorXd& zk, const Eigen::VectorXd& zr,
		const Eigen::MatrixXd& Qx, BlockStats& st) const;
	double leaf_full_loglik(const Eigen::MatrixXd& Xn, const Eigen::VectorXd& z,
		const Eigen::MatrixXd& Q, BlockStats& st) const;

	// Stop/split posterior (rho, lambda) of an expanded node from its
	// loglik_base / loglik_knot / child_loglik; shared by expand() and the
	// sig2-only refresh of SMC.
	void node_posterior(PathNode& v) const;

	// Exact WN integration over the uniform/Beta(2,2) rank-cut prior.
 void prepare_wn_cut_proposal(PathNode& v, const Eigen::MatrixXd& Xres,
  const Eigen::VectorXd& residual) const;

	// ---- the nested knot design (design = "nested"; tree_engine.cpp) ----
	// PP leaf model with uniform/balanced cuts only (the wrapper checks).
	bool nested() const { return design=="nested"; }
	// Candidates per node: min(nested_cand_factor * r, n_res).  With factor 1
	// the children of a node share its r candidates, so the knot count
	// halves at every level; factor 2 keeps a balanced child at about r.
	int nested_cand_factor = 2;
	// Knots of a node: the root (parent_cand == nullptr) and small nodes
	// (n <= r) by pick_knots; otherwise the parent's candidate rows that
	// reach the node, in candidate order.  ind_res: ascending complement.
	void nested_node_knots(const std::vector<int>& idx,
		const std::vector<int>* parent_cand, const Eigen::VectorXd& bmin,
		const Eigen::VectorXd& bmax, Eigen::VectorXi& ind_knot,
		Eigen::VectorXi& ind_res) const;
	// The candidate set of a node: the first min(nested_cand_factor*r, n_res) residual rows in
	// the maximin order of Xres, as positions into Xres (selection order).
	std::vector<int> nested_candidates(const Eigen::MatrixXd& Xres) const;
	// knot_step that also accepts an empty knot set (void block).
	bool nested_knot_step(const Eigen::MatrixXd& Kx, const Eigen::MatrixXd& Xres,
		const Eigen::VectorXd& knot_data, const Eigen::VectorXd& zres,
		const Eigen::MatrixXd& Qk, const Eigen::MatrixXd& Qx,
		int level, KnotBlock& kb) const;
	// Exact PP integration over the rank-cut prior by the all-cell sweeps.
	void prepare_nested_cut_proposal(PathNode& v, const Eigen::MatrixXd& Xres,
		const Eigen::VectorXd& zres_c, const Eigen::MatrixXd& Qnext,
		const std::vector<int>& cand_pos) const;
	// expand() of a nested PP node (expand() dispatches here).
	void expand_nested(PathTrie& trie, PathNode& v, const std::vector<int>& idx,
		const Eigen::VectorXd& resid, const Eigen::MatrixXd& Q) const;

 // ---- shared per-path expansion into a store ----
	// expand: one distinct node-path, done once for every particle that
	// shares it.  descend: depth-first re-derivation of (idx, resid, Q) along
	// stamped paths to nodes marked need_expand.
	void expand(PathTrie& trie, PathNode& v, const std::vector<int>& idx,
		const Eigen::VectorXd& resid, const Eigen::MatrixXd& Q) const;
	void descend(PathTrie& trie, int trie_idx, std::vector<int>&& idx,
		Eigen::VectorXd&& resid, Eigen::MatrixXd&& Q) const;
	// The expansion input of child (J, side, cut) of a node whose post-knot
	// state is (gidx, zres_c, Qnext): row i goes left iff X(gidx[i], J) < cut;
	// selection preserves node order.  Shared by descend() and the cached
	// direct expansion so both do the identical arithmetic.
	void partition_child(const std::vector<int>& gidx, const Eigen::VectorXd& zres_c,
		const Eigen::MatrixXd& Qnext, int J, int side, double cut,
		std::vector<int>& cidx, Eigen::VectorXd& cres, Eigen::MatrixXd& cQ) const;

	// ---- known-theta likelihood of a FIXED tree structure ----
	// Direct top-down rescoring of one structure (S: 1 stop / 0 split / NaN
	// unreached; J, cuts at split nodes), aggregating one Gaussian
	// (logdet, quad, n) triple plus the WN leaf marginals.  This is a
	// computation path INDEPENDENT of the SMC sampler's incremental trie
	// bookkeeping, so rescoring a sampled tree (verify = TRUE) is a real check.
	// A split of the supplied structure spends min(r, n) knots (the SMC
	// structure convention; deep splits may consume the whole node).
	struct FixedNode { bool split; int J; double cut; };
	using FixedNodes = std::map<int, FixedNode>;
	// Reached-node PP/WN scoring, independent of the SMC incremental weights.
	StructureEval evaluate_nodes(const FixedNodes& nodes, const std::string& leaf_model) const;
	StructureEval evaluate_structure(const Eigen::VectorXd& S,
		const Eigen::VectorXd& J, const Eigen::VectorXd& cuts,
		const std::string& leaf_model) const;
// Deterministic generated tree (the Full leaf model): split while n > r,
// extending storage depth as needed (each split spends exactly r knots), split
	// dimension cycling level % dim, cuts from the cut method (uniform/balanced
	// consume a per-node mt19937_64 seeded seed + 2654435761u * id).
	StructureEval evaluate_generated(const std::string& leaf_model,
		const std::string& cut_method, unsigned int seed) const;
	// The generated Full tree's theta-independent geometry (see
	// FullTreeGeometry), computed by the same decisions evaluate_generated
	// takes, without any kernel or factorization.
	std::shared_ptr<FullTreeGeometry> build_generated_geometry(
		const std::string& cut_method, unsigned int seed) const;
	// evaluate_generated("Full", ...) on a prepared geometry: kernels,
	// factorizations and the residual recursion only.
	StructureEval evaluate_geometry(const FullTreeGeometry& geom) const;

	double gaussian_loglik(double logdet, double quad, int n) const;
	double white_noise_loglik(const Eigen::VectorXd& z) const;

private:
	struct StructureContext {
		bool fixed;
		const Eigen::VectorXd* fixed_S;
		const Eigen::VectorXd* fixed_J;
		const Eigen::VectorXd* fixed_cuts;
		Eigen::VectorXd* S;
		Eigen::VectorXd* J;
		Eigen::VectorXd* cuts;
		std::string leaf_model;
		std::string cut_method;
		unsigned int seed;
		int task_depth;
		int tree_depth;  // representation depth, independent of WN's initial boundary
		const FixedNodes* sparse = nullptr;
	};
	// Recursive worker of evaluate(): the log-likelihood sufficient statistics
	// (logdet, quad, n, WN marginal) of the subtree rooted at node id -- the
	// knot block of a split node, the leaf block of a terminal node, and the
	// two children combined.  ("loglik", not "score": no derivative is taken.)
	StructureStats loglik_subtree(StructureContext& ctx, int id, int level,
		std::vector<int>&& idx, Eigen::VectorXd&& z, Eigen::MatrixXd&& Q,
		Eigen::VectorXd bmin, Eigen::VectorXd bmax,
		const std::vector<int>* parent_cand = nullptr) const;
	StructureEval evaluate(const Eigen::VectorXd* S, const Eigen::VectorXd* J,
		const Eigen::VectorXd* cuts, const std::string& leaf_model,
		const std::string& cut_method, unsigned int seed) const;
	static void combine(StructureStats& into, const StructureStats& from);
	// The generated tree's cut points at node id from the post-knot residual
	// rows Xres (all dimensions, in order, so the per-node RNG stream is the
	// same whichever dimension is used): "median", "balanced"/"uniform"
	// (mt19937_64 seeded seed + 2654435761u * id), else the box midpoint.  An
	// empty residual block falls back to the midpoint.  Shared by
	// loglik_subtree and geometry_subtree so the two cannot drift apart.
	Eigen::VectorXd generated_cut_points(const Eigen::MatrixXd& Xres,
		const std::string& cut_method, unsigned int seed, int id,
		const Eigen::VectorXd& bmin, const Eigen::VectorXd& bmax) const;
	int geometry_subtree(FullTreeGeometry& geom, int id, int level,
		std::vector<int>&& idx, Eigen::VectorXd bmin, Eigen::VectorXd bmax) const;
	StructureStats loglik_geometry_subtree(const FullTreeGeometry& geom, int id,
		Eigen::VectorXd&& z, Eigen::MatrixXd&& Q, int task_depth) const;
};

// Shared tree mathematics and cache lifetimes; implemented in tree_engine.cpp.
double wn_leaf_loglik(const Eigen::VectorXd& y);
size_t cache_bytes_of(const PathNode& v);
double beta22_interval_mass(double a, double b);
size_t proposal_bytes_of(const WNCutProposal& p);
size_t path_cache_entry_bytes(const PathCacheEntry& e, size_t key_len);
void release_postknot(PathNode& v);
void release_draw_state(PathNode& v);
void release_cache(PathNode& v);
#endif
