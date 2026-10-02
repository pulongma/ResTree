#ifndef RESTREE_SMC_H
#define RESTREE_SMC_H
#include "ResTree.h"
#include "prediction.h"

// The SMC sampler over tree space, driving the model's shared expansion.
class SMC : public ResTree {
public:
	int n_internal = 2;
	std::string resampling_method = "multinomial";

	int N = 24;
	double minESS = 6.0, alpha = 0.01;
	unsigned int seed = 42;

	PathTrie trie;

	// Per-particle node storage for PP and WN. Every
	// particle keeps one record per REACHED node id -- the root and every
	// child created by one of its committed splits -- in an ordered map keyed
	// by the heap id; an id that is not in the map is ACT_ABSENT. The
	// accessors below are the only
	// way the sampler and its exports read or write active nodes. Prediction
	// reads a separate FittedTrees snapshot, never these active maps.
	//   open_depth()    whether the sampler runs past the initial depth
	//                   (WN: every nonempty node compares at every level;
	//                   PP: level < depth is the cap) -- the MODEL rule.
	using SparseNode = ParticleNode;
	std::vector<ParticleNodes> sparse_particles;
	const SparseNode* find_node(int p, int id) const {
		return p<0 || (size_t)p>=sparse_particles.size() ? nullptr : sparse_particles[p].find_node(id);
	}
	SparseNode* find_node(int p, int id) {
		return p<0 || (size_t)p>=sparse_particles.size() ? nullptr : sparse_particles[p].find_node(id);
	}
	SparseNode& existing_node(int p, int id) { return sparse_particles.at(p).existing(id); }
	SparseNode& insert_node(int p, int id, const SparseNode& node) {
		return sparse_particles.at(p).insert(id,node);
	}
	// Worker mutations never throw on a miss. The caller must stop its outcome;
	// sample_level reports the deferred error from the main thread before commit.
	SparseNode* worker_node(int p, int id) {
		return worker_particle_node(sparse_particles[p],id,internal_error_node);
	}
	// Compact selected completed particles for any prediction consumer.
	FittedTrees fitted_trees(const std::vector<int>& keep) const;
	bool open_depth() const { return baseline=="WhiteNoise"; }
	int tree_depth = 0; // maximum(initial depth, deepest split's child level)
	static constexpr int max_tree_depth = restree_max_wn_depth;
#define RESTREE_NODE_ACCESS(TYPE, NAME, DEFAULT_VALUE) \
	TYPE NAME##_value(int p, int id) const { \
		const auto* node = find_node(p,id); \
		return node ? node->NAME : (TYPE)(DEFAULT_VALUE); \
	}
	RESTREE_NODE_ACCESS(int8_t, act, ACT_ABSENT)
	RESTREE_NODE_ACCESS(int8_t, jdim, -1)
	RESTREE_NODE_ACCESS(int32_t, trie_at, -1)
	RESTREE_NODE_ACCESS(double, Loglik, 0.0)
	RESTREE_NODE_ACCESS(double, cutval, std::numeric_limits<double>::quiet_NaN())
	RESTREE_NODE_ACCESS(double, rhoval, 1.0)
#undef RESTREE_NODE_ACCESS
	// Visit the node ids of particle p in [lo, hi) in ascending order by
	// walking its map from lower_bound(lo); no per-particle vector of
	// 2^level ids is ever materialised.
	template<class F> void for_each_node(int p, int lo, int hi, F&& f) const {
		const auto& nodes = sparse_particles[p];
		for(auto it=nodes.lower_bound(lo); it!=nodes.end() && it->first<hi; ++it) f(it->first);
	}
	// Pending ids of a level over all particles: the union of the particles'
	// PENDING entries in [2^level, 2^(level+1)).
	std::vector<int> pending_ids(int level) const;
	bool has_pending(int level) const;
	void record_split_depth(int level);
	size_t stored_particle_nodes() const;

	// particle_loglik: REALIZED partial-tree log-likelihood (terminals keep arrival
	// values; realized splits hold loglik_knot + realized children).  Never
	// the stop/split mixture -- that exists only inside transition() as the
	// action normalizer.  particle_loglik_tree re-derives the same quantity at
	// the end of run() from the per-node block statistics in the
	// fixed-structure evaluator's aggregation form (one global Gaussian
	// triple, DFS order), so it is identical() to restree_loglik(tree = .);
	// the two must agree to round-off.
	Eigen::VectorXd particle_log_weight, particle_loglik, particle_loglik_tree;
	// Current normalized weights only; no weight history is kept (the ESS
	// vector carries the per-step trajectory).
	Eigen::VectorXd particle_weight;
	// ESS history is diagnostic only, never read by resampling. PP stores
	// equal-value runs over heap positions; WN stores runs over processed
	// positions. Only fitted-output export materializes the legacy vector.
	struct ESSRun { int end; double value; }; // half-open end, ascending
	std::vector<ESSRun> ess_runs;
	// Exact processed-position history; the legacy ESS vector remains unchanged.
	struct StepDiagnostic {
		int step, node_id, level, resampled;
		double ess, ess_after, logZ;
	};
	std::vector<StepDiagnostic> step_diag;
	int processed_steps = 0;
	bool record_ess_history = true;
	int ess_history_length() const { return ess_runs.empty() ? 0 : ess_runs.back().end; }
	void append_ess_run(int end, double value);
	Eigen::VectorXd collect_ess_history() const;
	double logZ = 0.0, Phi0 = 0.0;
	int n_resample = 0;

	// Per-LEVEL sampler diagnostics, recorded at the level end (the only
	// point where a resampling decision is made; mid-level weights rank
	// particles by node-processing order and are not reported).  Pure
	// bookkeeping: nothing in the sampling path reads these.
	//   level_ess        ESS of the level-end weights, before any resample
	//   level_ess_after  ESS after the level's resample (== level_ess if none)
	//   level_max_weight largest normalised weight at the level end
	//   level_logZ_inc   increment of logZ contributed by the level (all its
	//                    nodes plus the tempered-resampling mass correction)
	//   level_resampled  1 if the level-end resample fired
	//   level_last_node  index of the last node position of the level
	Eigen::VectorXd level_ess, level_ess_after, level_max_weight, level_logZ_inc;
	Eigen::VectorXi level_resampled, level_last_node, level_last_step;

	std::mt19937_64 master_rng;
	std::vector<std::mt19937_64> par_rng;

	// Never throw inside an OpenMP worker: workers record the failing node id
	// here (first writer wins) and return; the serial caller raises the error.
	std::atomic<int> internal_error_node{-1};

	EngineDiagnostics diag;

	SMC(Eigen::VectorXd y_, Eigen::MatrixXd X_)
		: ResTree(std::move(y_), std::move(X_)) {}

void run(bool reuse_trie = false, bool collect_ess = true, bool collect_tree_scores = true);

	// ---- sig2-only re-evaluation (persistent engine) ----
	// None of the expansion linear algebra involves sig2 (kernels, knot
	// downdates, Cholesky factors, propagated coefficients, the residual
	// subtraction); sig2 enters only through restree_loglik_conditional(ld,
	// qd, n, sig2) at the stored block statistics.  After a completed run
	// under deterministic cuts the trie is therefore valid for every sig2 at
	// the same (range, nugget, nu, ...): run(true) refreshes its values and
	// re-runs only the sampler.  Random-cut PP keeps no per-path candidate
	// state across levels, so it never reuses.  Integrated-cut WN under the
	// maximin design does (wn_integrated_reuse): its children are keyed by
	// rank cell, sigma^2 enters a node only through loglik_knot inside the
	// proposal's split mass, and the interval table is sigma^2-free -- so a
	// persistent engine keeps the proposals (retain_cache) and refreshes the
	// masses.  Reuse there is restricted to the same seed (common random
	// numbers): at another seed the particles draw other cells almost
	// everywhere and the retained trie would only grow.  A node whose
	// proposal was dropped for the budget is re-expanded on demand.
	bool trie_reuse_ok = false;
	GPM trie_covpar;
	unsigned int trie_seed = 0;
	bool diag_reused_trie = false;
	bool wn_integrated_reuse() const {
		return baseline=="WhiteNoise" && design=="maximin" &&
			(cut_method=="uniform" || cut_method=="balanced");
	}
	// Nodes above which a retained integrated-cut trie is rebuilt instead
	// of reused (bounds the memory of the trie itself; the proposals are
	// bounded separately by keep_cache_budget_bytes).
	size_t reuse_node_cap = static_cast<size_t>(1) << 21;
	// Persistent deterministic engines retain sig2-free post-knot state.
	// Integrated WN retains proposal tables instead. Shallow levels (WN: root)
	// are unconditional; deeper admission uses keep_cache_budget_bytes.
	// PathTrie::retained_bytes accounts these two categories separately.
	bool keep_shallow_caches = false;
	int keep_cache_levels = 2;
	size_t keep_cache_budget_bytes = static_cast<size_t>(512) << 20;
	bool retain_cache(PathNode& v);
	bool sig2_only_change(const GPM& th, unsigned int seed) const;
	void refresh_sig2_values();
	// Protect the retained trie across a run at another (range, nugget):
	// save_trie() moves it aside (O(1) swap) before the run, restore_trie()
	// moves it back after a rejected proposal so sig2-only moves at the
	// current state keep reusing it.  Nothing is copied.
	PathTrie trie_saved;
	GPM trie_saved_covpar;
	unsigned int trie_saved_seed = 0;
	bool trie_saved_ok = false;
	void save_trie();
	bool restore_trie();

	// Export selected particle columns directly; nullptr retains the full
	// per-particle export for internal callers that explicitly need it.
	// (S: 0 split / 1 terminal / NA unreached; J 1-based; cuts; rho.)
	void collect_outputs(Eigen::MatrixXd& S_out,
		Eigen::MatrixXd& J_out, Eigen::MatrixXd& cut_out, Eigen::MatrixXd& rho_out,
		const std::vector<int>* representatives = nullptr) const;

	// The realized trees of the given particles as one long-format node
	// table (the R-side `structure$nodes`): one row per NONEMPTY node position
	// a tree reaches -- splits, stops and bottom-level terminals (the empty
	// children of a split that consumed every row carry nothing and are
	// omitted) -- with tree (0-based index into
	// reps), id, split flag, J (0-based; -1 at terminals), cut (NaN at
	// terminals), rho (the posterior stop probability of an expanded node;
	// NaN where no draw was made) and n (rows reaching the node).  Row count
	// = the realized positions, independent of depth.
	struct NodeTable {
		std::vector<int> tree, id, split, J, n;
		std::vector<double> cut, rho;
	};
	void collect_nodes(const std::vector<int>& representatives, NodeTable& out) const;

	// Group the particle system into distinct realized trees; returns one
	// representative particle index per distinct tree, the aggregated weight
	// of each, and the per-particle group index (0-based).
	void distinct_trees(std::vector<int>& reps, Eigen::VectorXd& w_distinct,
		Eigen::VectorXi& group_of_particle) const;

private:
	// ---- expand_level: mark the live frontier, then shared expansion ----
	void expand_level(int level);

	// ---- sample_level: particle draws, weights, resampling ----
	// Per-particle outcome of one quantile-cut move, computed in the
	// thread-parallel draw pass and applied in the serial commit pass.
	struct RandomCutMove {
		bool active = false;
  bool integrated_wn = false;
  // direct_increment: the draw already computed the exact increment of the
  // weight (log of the estimated one-step normaliser minus loglik_base) --
  // used by the M-candidate PP move; the commit then skips the
  // proposal-ratio bookkeeping of the basic move.  integrated_wn implies it.
  bool direct_increment = false;
  double log_weight_increment = 0.0;
  int cell = -1;           // integrated WN: rank cell of the drawn interval
  double cut_rep = 0.0;    // integrated WN: the cell's representative cut
		int S = 1;
		int Ji = -1;
		double cut = 0.0;
		double rho = 1.0;
		double lambda_log_Ji = 0.0;
		int child_n[2] = {0, 0};
		double child_ll[2] = {0.0, 0.0};
		BlockStats child_st[2];
	};
	void sample_level(int level);
	// The PP arrays (act, jdim, trie_at, Loglik, cutval, rhoval) hold
	// ids 1 .. n_ids - 1.  A run allocates them for the root level only and
	// grows them at every level to the ids that level's moves can touch
	// (children of level l have ids < 2^(l+2)), so a resample at level l copies
	// 2^(l+2) entries per particle instead of 2^(depth+1); after the last level
	// they hold every id. WN uses sparse maps and this helper is a no-op.
	void transition(int p, int id, std::mt19937_64& rng,
		double& Phi, double& w_log);
	void draw_random_cut_move(int p, int id, std::mt19937_64& rng, RandomCutMove& out);
	// ---- PP random cuts with candidate blocks shared across particles ----
	// The basic and the M-candidate PP moves are the same computation in
	// three phases: (1) each particle draws its cuts per dimension from
	// its own stream and records the left-row count of every cut; (2) the
	// PP child pair of every DISTINCT (node, J, left count) is evaluated
	// once -- under the maximin design the block depends on the rows only,
	// and the left rows of a cut are exactly the left-count smallest X_J of
	// the node, so particles whose cuts fall in the same rank cell share the
	// pair; (3) each particle assembles its posteriors from the shared
	// pairs and draws stop / dimension (/ candidate) from its stream, in the
	// order cuts, then S, then J, then candidate. Balanced retains the
	// one-pass move's draws; uniform uses the midpoint stratification above.
	struct PPCutDraw {
		bool active = false;
		int32_t t = -1;
		Eigen::MatrixXd cuts;       // dim x M
		Eigen::VectorXd log_mass;   // uniform strata weights; empty for balanced IID draws
		Eigen::MatrixXi n_left;     // dim x M
		Eigen::MatrixXi pair;       // dim x M: index into the shared pair table
	};
	struct PPChildPair {
		int32_t t = -1; int J = 0; int nl = 0, nr = 0; double delta = 0.0;
		double ll[2] = {0.0, 0.0};
		BlockStats st[2];
	};
	void pp_draw_cuts(int p, int id, std::mt19937_64& rng, PPCutDraw& d);
	void pp_eval_child_pair(PPChildPair& pr) const;
	void pp_finish_move(int p, int id, std::mt19937_64& rng, const PPCutDraw& d,
		const std::vector<PPChildPair>& pairs, RandomCutMove& out);
	void draw_integrated_wn_cut_move(int p, int id, std::mt19937_64& rng,
  RandomCutMove& out);
	void draw_multi_cut_move(int p, int id, std::mt19937_64& rng, RandomCutMove& out);
	// nested PP design: the same draws from the nested proposal (SMC.cpp)
	void draw_nested_pp_cut_move(int p, int id, std::mt19937_64& rng,
		RandomCutMove& out);
 void commit_random_cut_move(int p, int id, const RandomCutMove& out,
		double& Phi, double& w_log);
	double update_ancestor_loglik(int p, int id, double value);
	void resample_particles(const Eigen::VectorXd& weight);
};

#endif
