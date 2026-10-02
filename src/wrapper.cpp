// R data conversion, validation and thread configuration.
#include "SMC.h"
#include "wrapper.h"
#ifdef _OPENMP
#include <omp.h>
#endif

// ==================== Physical-core detection ====================

// BEGIN RESTREE_NATIVE_TOPOLOGY
#include <cerrno>
#include <cstddef>
#include <set>
#include <utility>

// Bound only the OS probe, not model memory or the requested worker count.
// A million CPU IDs needs 128 KiB, well beyond current Linux affinity masks.
// Persistent EINVAL must not cause unlimited allocation/retries.
constexpr std::size_t max_affinity_cpus = std::size_t(1) << 20;

// Affinity supplies resize/query/capacity/contains; topology supplies the
// (package, core) IDs for an allowed CPU.
// query returns an errno value, or zero on success. A failed/empty probe,
// allocation failure or incomplete topology yields a conservative ONE core,
// never a machine-wide or partial count. This also bounds default Eigen teams.
template<class Affinity, class Topology>
int affinity_physical_cores(Affinity& affinity, const Topology& topology) noexcept {
    try {
        for(std::size_t cpus=1024; cpus<=max_affinity_cpus; cpus*=2) {
            if(!affinity.resize(cpus)) return 1;
            const int error=affinity.query();
            if(error==EINVAL) continue; // kernel mask requires a larger buffer
            if(error!=0) return 1;
            std::set<std::pair<int,int>> cores;
            // Inspect the entire allocated mask, including rounded-up bits.
            for(std::size_t cpu=0; cpu<affinity.capacity(); ++cpu) {
                if(!affinity.contains(cpu)) continue;
                int package=-1, core=-1;
                if(!topology(cpu,package,core) || package<0 || core<0) return 1;
                cores.emplace(package,core);
            }
            return cores.empty() ? 1 : static_cast<int>(cores.size());
        }
    } catch(...) {
        // Pure C++/OS probing only: no R calls or model evaluation to mask.
    }
    return 1;
}

// No R calls here. Count physical cores, not SMT threads or the OpenMP default
// team size. Cache OS topology once, outside the worker regions. Linux uses
// one core conservatively if affinity/topology cannot be established. On the
// other platforms zero means unknown (the request is not silently reduced).
//
// On Linux the count is restricted to the calling R thread's allowed CPUs.
// Probe before starting workers. Under a batch
// scheduler (Slurm cgroup / cpuset, taskset, a container) the job owns a subset
// of the node, and /sys/devices/system/cpu enumerates the whole node: it is a
// kernel topology view and is unaffected by cpusets or CPU affinity.  Sizing an
// OpenMP team from the node count then oversubscribes the allocation by the
// ratio between them and, because every per-thread working block is live at
// once while the memory limit stays per job, is the ordinary way a large fit
// dies of OOM on a shared node.  sched_getaffinity is the allocation-aware
// count on Linux; macOS uses hw.physicalcpu. Windows reports physical topology
// across processor groups; Windows affinity/job-object quotas are not queried.
// Affinity changes after the first probe require a fresh R process. CPU-time
// quotas (as distinct from cpusets) are not inferred from physical topology.
#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <climits>
#include <fstream>
#include <sched.h>
#include <string>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vector>
#endif

#if defined(__linux__) && defined(CPU_ALLOC) && defined(CPU_ALLOC_SIZE) && \
    defined(CPU_FREE) && defined(CPU_ZERO_S) && defined(CPU_ISSET_S)
class LinuxAffinity {
    cpu_set_t* mask_=nullptr;
    std::size_t bytes_=0;
public:
    LinuxAffinity() = default;
    LinuxAffinity(const LinuxAffinity&) = delete;
    LinuxAffinity& operator=(const LinuxAffinity&) = delete;
    ~LinuxAffinity() { if(mask_) CPU_FREE(mask_); }
    bool resize(std::size_t cpus) {
        if(mask_) CPU_FREE(mask_);
        mask_=CPU_ALLOC(cpus);
        bytes_=CPU_ALLOC_SIZE(cpus);
        if(!mask_) return false;
        CPU_ZERO_S(bytes_,mask_);
        return true;
    }
    int query() { return sched_getaffinity(0,bytes_,mask_)==0 ? 0 : errno; }
    std::size_t capacity() const { return bytes_*CHAR_BIT; }
    bool contains(std::size_t cpu) const {
        return cpu<capacity() && CPU_ISSET_S(cpu,bytes_,mask_);
    }
    cpu_set_t* data() { return mask_; }
    std::size_t bytes() const { return bytes_; }
};
inline bool linux_cpu_topology(std::size_t cpu, int& package, int& core) {
    const std::string path="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
    std::ifstream package_file(path+"physical_package_id");
    std::ifstream core_file(path+"core_id");
    return bool(package_file>>package) && bool(core_file>>core);
}
#endif

inline int restree_detect_physical_cores(){
#if defined(__APPLE__)
    int count = 0;
    size_t size = sizeof(count);
    if(sysctlbyname("hw.physicalcpu", &count, &size, nullptr, 0)==0 && count>0)
        return count;
#elif defined(__linux__)
#if defined(CPU_ALLOC) && defined(CPU_ALLOC_SIZE) && defined(CPU_FREE) && \
    defined(CPU_ZERO_S) && defined(CPU_ISSET_S)
    LinuxAffinity allowed;
    return affinity_physical_cores(allowed,linux_cpu_topology);
#else
    // A libc without the dynamic-affinity API must still compile safely.
    return 1;
#endif
#elif defined(_WIN32)
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore,nullptr,&bytes);
    if(GetLastError()!=ERROR_INSUFFICIENT_BUFFER || bytes==0) return 0;
    std::vector<unsigned char> buffer(bytes);
    auto info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    if(!GetLogicalProcessorInformationEx(RelationProcessorCore,info,&bytes)) return 0;
    int count = 0;
    for(size_t offset=0; offset<bytes; ){
        const auto record = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
            buffer.data()+offset);
        if(record->Size==0 || record->Size>bytes-offset) return 0;
        if(record->Relationship==RelationProcessorCore) ++count;
        offset += record->Size;
    }
    return count;
#endif
    return 0;
}

inline int restree_physical_cores(){
    static const int count = restree_detect_physical_cores();
    return count;
}
// END RESTREE_NATIVE_TOPOLOGY

// ==================== Input and thread policy helpers ====================

#include <RcppEigen.h>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <cerrno>
#include <algorithm>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

void restree_validate_covariance(
	int dimension, double sig2, const Eigen::VectorXd& range,
	double tail, double nu, double nugget, const std::string& form,
	const std::string& family, const std::string& dtype){
	if(form!="isotropic" && form!="ARD" && form!="tensor")
		Rcpp::stop("form must be isotropic, ARD, or tensor.\n");
	if(family!="CH" && family!="matern" && family!="gauss" &&
	   family!="powexp" && family!="cauchy")
		Rcpp::stop("unsupported covariance family.\n");
	if(dtype!="Euclidean" && dtype!="GCD")
		Rcpp::stop("dtype must be Euclidean or GCD.\n");
	if(!std::isfinite(sig2) || sig2<=0.0)
		Rcpp::stop("sig2 must be finite and positive.\n");
	if(!std::isfinite(nugget) || nugget<0.0)
		Rcpp::stop("nugget must be finite and nonnegative.\n");
	if(!std::isfinite(nu) || nu<=0.0)
		Rcpp::stop("nu must be finite and positive.\n");
	if((family=="powexp" || family=="cauchy") && nu>2.0)
		Rcpp::stop("nu must not exceed 2 for powexp and cauchy kernels.\n");
	if((family=="CH" || family=="cauchy") &&
	   (!std::isfinite(tail) || tail<=0.0))
		Rcpp::stop("tail must be finite and positive for CH and cauchy kernels.\n");
	if((form=="isotropic" && range.size()!=1) ||
	   ((form=="ARD" || form=="tensor") && range.size()!=dimension) || range.size()==0 ||
	   !range.allFinite() || (range.array()<=0.0).any())
		Rcpp::stop("range must be positive and scalar for isotropic or one per input column for ARD/tensor.\n");
	if(dtype=="GCD" && (form!="isotropic" || dimension!=2))
		Rcpp::stop("GCD distance requires two-column input and isotropic covariance.\n");
}

inline std::string restree_validate_baseline(std::string baseline,
	bool allow_full=true){
	if(baseline=="WN") baseline="WhiteNoise";
	if((allow_full && baseline=="Full") || baseline=="PP" ||
	   baseline=="WhiteNoise") return baseline;
	Rcpp::stop(allow_full ?
		"baseline must be Full, PP, or WhiteNoise (WN is an alias).\n" :
		"baseline must be PP or WhiteNoise (WN is an alias).\n");
	return baseline;
}

// Resolve one policy on the main thread. Serial tree recursion may still use
// allocation-bounded Eigen GEMM threads; a no-OpenMP build is fully serial.
// These counts govern this package, not unrelated BLAS or package threads.
struct restree_threads {
 int request = 1, omp = 1, eigen = 1;
 const char* source = "serial build";
};

// Strict scheduler integers; OpenMP additionally permits a comma-separated
// list of positive counts. Invalid or out-of-range values provide no cap.
inline int restree_env_thread_cap(const char* name){
 const char* value=std::getenv(name);
 if(!value || !*value) return 0;
 const bool list=std::string(name)=="OMP_NUM_THREADS";
 int first=0;
 const char* p=value;
 for(;;){
  while(std::isspace(static_cast<unsigned char>(*p))) ++p;
  if(!std::isdigit(static_cast<unsigned char>(*p))) return 0;
  errno=0; char* end=nullptr; const long k=std::strtol(p,&end,10);
  if(errno==ERANGE || k<1 || k>1000000L) return 0;
  if(!first) first=static_cast<int>(k);
  p=end; while(std::isspace(static_cast<unsigned char>(*p))) ++p;
  if(!*p) return first;
  if(!list || *p!=',') return 0;
  ++p;
 }
}

struct restree_core_limit {
 int cores = 0;
 const char* source = "physical cores in this process's CPU affinity";
};
inline restree_core_limit restree_core_cap(){
 restree_core_limit cap;
 cap.cores=restree_physical_cores();
 for(const char* name : {"OMP_NUM_THREADS","SLURM_CPUS_PER_TASK","PBS_NP"}){
  const int k=restree_env_thread_cap(name);
  if(k>0 && (cap.cores<1 || k<cap.cores)){cap.cores=k;cap.source=name;}
 }
 return cap;
}
inline restree_threads restree_thread_policy(int request, bool warn=true){
 if(request<1) Rcpp::stop("ncores must be a positive integer.");
 restree_threads t; t.request=request;
#ifdef _OPENMP
 const auto cap=restree_core_cap();
 t.source=cap.source;
 t.omp=cap.cores>0 ? std::min(request,cap.cores) : request;
 t.eigen=request==1 ? (cap.cores>0 ? cap.cores : std::max(1,omp_get_num_procs())) : t.omp;
 if(warn && t.omp<request)
  Rcpp::warning("ncores = %d exceeds the %d cores this process may use (%s); using %d",
    request,t.omp,t.source,t.omp);
#endif
 return t;
}
void restree_validate_ncores(int& ncores){
 ncores=restree_thread_policy(ncores).omp;
}

inline void restree_apply_threads(const restree_threads& t){
#ifdef _OPENMP
	Eigen::setNbThreads(t.eigen);
#else
	(void)t;
#endif
}

inline void restree_validate_cut_method(const std::string& cut_method){
	if(cut_method!="middle" && cut_method!="median" &&
	   cut_method!="uniform" && cut_method!="balanced")
		Rcpp::stop("cut_method must be one of: middle, median, uniform, balanced.\n");
}

inline void restree_validate_design(const std::string& design,
	bool allow_maximin_full=false){
	if(design=="maximin" || design=="boundary" || design=="nested" ||
	   (allow_maximin_full && design=="maximin_full")) return;
	Rcpp::stop(allow_maximin_full ?
		"design must be one of: maximin, maximin_full, boundary, nested.\n" :
		"design must be one of: maximin, boundary, nested.\n");
}

// ==================== R objects, native handles and shared configuration ====================

using namespace Rcpp;

// Tags distinguish prediction-only ownership from a live evidence sampler.
// Untyped casts between these handles would be unsafe after serialization or
// accidental use of a prediction handle in an inference-only entry point.
SEXP smc_handle_tag(){ return Rf_install("ResTree::SMC::v2"); }
SEXP prediction_handle_tag(){ return Rf_install("ResTree::PredictionState::v1"); }
SMC& smc_from_handle(SEXP handle){
    if(TYPEOF(handle)!=EXTPTRSXP || R_ExternalPtrTag(handle)!=smc_handle_tag())
        Rcpp::stop("expected a live SMC engine handle.");
    auto* p=static_cast<SMC*>(R_ExternalPtrAddr(handle));
    if(!p) Rcpp::stop("the SMC engine handle is empty.");
    return *p;
}
PredictionState& prediction_from_handle(SEXP handle){
    if(TYPEOF(handle)!=EXTPTRSXP || R_ExternalPtrTag(handle)!=prediction_handle_tag())
        Rcpp::stop("expected a prediction-state handle.");
    auto* p=static_cast<PredictionState*>(R_ExternalPtrAddr(handle));
    if(!p) Rcpp::stop("the prediction-state handle is empty.");
    return *p;
}
SEXP make_prediction_handle(ResTree&& model,FittedTrees&& trees,SEXP info){
    std::unique_ptr<PredictionState> state(new PredictionState(std::move(model),std::move(trees)));
    Rcpp::XPtr<PredictionState> ptr(state.get(),true,prediction_handle_tag(),info);
    state.release();
    return ptr;
}

// Theta conversion accepts:
//   * A `restree_theta` S4 object (or a subclass): the class is checked with
//     R's inheritance test, and every slot the class declares is read
//     directly -- there is no "missing slot takes a default" path, because a
//     restree_theta cannot lack a slot; an S4 object of any other class is
//     rejected.  Cost: one `is()` dispatch and eight slot reads.
//   * A named list: the documented convenience (`restree_loglik(m, list(sig2
//     = 1, range = 0.3))`); unknown names are rejected, and a component that
//     is absent takes the value of the class PROTOTYPE, obtained from R once
//     per call and only when something is absent, so the defaults have a
//     single source (setClass("restree_theta", prototype = ...)) and cannot
//     drift from restree_theta().
// Either way the values are then checked against the engine's covariance
// preconditions (restree_validate_covariance), so an object altered after
// construction cannot reach a kernel with, e.g., sig2 <= 0.
GPM read_theta(SEXP theta_sexp, int p){
	static const char* fields[] = {"sig2","range","nugget","nu","tail","form","family","dtype"};
	GPM th;
	th.coef = Eigen::VectorXd::Zero(1);
	Rcpp::S4 src;               // the object the eight fields are read from
	Rcpp::List lst;
	bool from_list = false;
	if(Rf_isS4(theta_sexp)){
		Rcpp::S4 obj(theta_sexp);
		if(!obj.is("restree_theta"))
			Rcpp::stop("theta must be a restree_theta object (see ?restree_theta) or a named list; "
				"got an S4 object of class %s.\n",
				Rcpp::as<std::string>(obj.attr("class")).c_str());
		src = obj;
	}else if(Rf_isNewList(theta_sexp)){
		from_list = true;
		lst = Rcpp::List(theta_sexp);
		bool complete = (lst.size()==8);
		if(lst.size()){
			SEXP nm_sexp = lst.names();
			if(Rf_isNull(nm_sexp) || Rf_length(nm_sexp)!=lst.size())
				Rcpp::stop("a list theta must be fully named (sig2, range, nugget, nu, tail, form, family, dtype).\n");
			Rcpp::CharacterVector nm(nm_sexp);
			for(int i=0; i<nm.size(); ++i){
				const std::string k = Rcpp::as<std::string>(nm[i]);
				if(k.empty())
					Rcpp::stop("a list theta must be fully named (sig2, range, nugget, nu, tail, form, family, dtype).\n");
				bool ok=false; for(const char* kk : fields) ok = ok || (k==kk);
				if(!ok) Rcpp::stop("unknown theta component(s): %s", k.c_str());
			}
			for(const char* kk : fields) complete = complete && lst.containsElementNamed(kk);
		}else{
			complete = false;
		}
		// absent components: the class prototype, fetched only when needed
		if(!complete){ src = Rcpp::S4("restree_theta"); }
	}else{
		Rcpp::stop("theta must be a restree_theta object or a named list.\n");
	}
	auto get = [&](const char* name) -> SEXP {
		if(from_list && lst.containsElementNamed(name)) return (SEXP)lst[name];
		return (SEXP)src.slot(name);   // S4: always present; list: the prototype
	};
	th.sig2   = Rcpp::as<double>(get("sig2"));
	th.nugget = Rcpp::as<double>(get("nugget"));
	th.nu     = Rcpp::as<double>(get("nu"));
	th.tail   = Rcpp::as<double>(get("tail"));
	th.form   = Rcpp::as<std::string>(get("form"));
	th.family = Rcpp::as<std::string>(get("family"));
	th.dtype  = Rcpp::as<std::string>(get("dtype"));
	th.range  = Rcpp::as<Eigen::VectorXd>(get("range"));
	if((th.form=="ARD" || th.form=="tensor") && th.range.size()==1 && p>1)
		th.range = Eigen::VectorXd::Constant(p, th.range(0));
	restree_validate_covariance(p, th.sig2, th.range, th.tail, th.nu,
		th.nugget, th.form, th.family, th.dtype);
	return th;
}

// Read validated S4 slots without rescanning training values. Fit/prediction
// validate in R; the public native likelihood trusts constructor-valid data.
// Keep cheap shape/index guards for that direct entry and its raw tree inputs.
ModelSpec read_model(SEXP model_sexp){
	Rcpp::S4 m(model_sexp);
	if(!m.is("restree_model"))
		Rcpp::stop("model must be a restree_model (or fitted restree) object.\n");
	ModelSpec sp(Rcpp::as<Rcpp::NumericMatrix>(m.slot("x")),
		Rcpp::as<Rcpp::NumericVector>(m.slot("y")));
	sp.depth = Rcpp::as<int>(m.slot("depth"));
	sp.r = Rcpp::as<int>(m.slot("r"));
	sp.leaf_model = restree_validate_baseline(Rcpp::as<std::string>(m.slot("leaf_model")), true);
	sp.design = Rcpp::as<std::string>(m.slot("design"));
	restree_validate_design(sp.design, false);
	sp.cut_method = Rcpp::as<std::string>(m.slot("cut_method"));
	restree_validate_cut_method(sp.cut_method);
	sp.prior_rho = Rcpp::as<double>(m.slot("prior_rho"));
	sp.prior_beta = Rcpp::as<double>(m.slot("prior_beta"));
	sp.cut_candidates = Rcpp::as<int>(m.slot("cut_candidates"));
	if(sp.cut_candidates<1) Rcpp::stop("cut_candidates must be at least 1.\n");
	if(sp.leaf_model=="PP" && sp.design!="nested" && sp.cut_method=="uniform" &&
		sp.cut_candidates==std::numeric_limits<int>::max())
		Rcpp::stop("cut_candidates must leave room for the additional midpoint-stratum candidate.\n");
	sp.seed = Rcpp::as<int>(m.slot("seed"));
	sp.nested_factor = Rcpp::as<int>(m.slot("nested_factor"));
	if(sp.nested_factor<1) Rcpp::stop("nested_factor must be at least 1.\n");
	sp.tree = Rcpp::List(m.slot("tree"));
	if(sp.X.rows()!=sp.y.size() || sp.X.rows()==0)
		Rcpp::stop("model x and y must have the same positive number of rows.\n");
	if(sp.X.cols()<1) Rcpp::stop("model x must have at least one column.\n");
	if(sp.X.cols()>127)
		Rcpp::stop("the tree engine supports at most 127 input dimensions.\n");
	if(sp.depth<0 || sp.depth>restree_max_model_depth)
		Rcpp::stop("depth must lie in 0..%d.\n", restree_max_model_depth);
	if(sp.r<1) Rcpp::stop("r must be at least 1.\n");
	return sp;
}

// The public native Full likelihood accepts valid data edits and invalidates
// its cache. Only that cache-miss path must validate the edited values; normal
// fit/prediction calls have already passed R validation and do not scan here.
void validate_model_data(const ModelSpec& sp){
	for(Eigen::Index j=0; j<sp.X.cols(); ++j){
		for(Eigen::Index i=0; i<sp.X.rows(); ++i){
			const double v = sp.X(i, j);
			if(!(v>=0.0 && v<=1.0))   // false for NaN as well
				Rcpp::stop("model x must contain only finite values in the unit hypercube [0, 1]^d "
					"(row %d, column %d is %g).\n", (int)i+1, (int)j+1, v);
		}
	}
	for(Eigen::Index i=0; i<sp.y.size(); ++i){
		if(!std::isfinite(sp.y(i)))
			Rcpp::stop("model y must contain only finite values (element %d).\n", (int)i+1);
	}
	if(!(sp.prior_rho>=0.0 && sp.prior_rho<=1.0))
		Rcpp::stop("prior_rho must lie in [0, 1].\n");
	if(!std::isfinite(sp.prior_beta) || sp.prior_beta<0.0)
		Rcpp::stop("prior_beta must be finite and nonnegative.\n");
	if(sp.seed<0) Rcpp::stop("model seed must be nonnegative.\n");
	if(sp.leaf_model=="Full"){
		if(sp.tree.size()==0)
			Rcpp::stop("a Full model must carry its fixed tree (slot tree); rebuild it with restree_model().\n");
	}else if(sp.tree.size()!=0){
		Rcpp::stop("only a Full model carries a fixed tree; slot tree must be empty for PP / WN.\n");
	}
}

// Default SMC particle count (the rule of .restree_model_nparticles in
// R/validation.R): one particle for depth zero,
// otherwise 100 * the configured model depth for both PP and WN.
// A deeper realized WN tree does not change the particle budget.
int restree_default_particles(const ModelSpec& sp){
	if(sp.depth==0) return 1;
	return 100 * sp.depth;
}

// The caller's ncores (a positive integer; 1 = the default) resolved into the
// engine's thread counts (wrapper.cpp: restree_thread_policy), applied to
// Eigen now and stored on the engine, so a retained engine (the persistent
// evidence engine, a fitted prediction state, the Full MLE workspace)
// re-applies its own counts at every later evaluation (reapply_threads).
void configure_threads(ResTree& m, int ncores){
	const restree_threads t = restree_thread_policy(ncores);
	m.ncores_request = t.request;
	m.ncores = t.omp;
	m.eigen_threads = t.eigen;
	restree_apply_threads(t);
}
void reapply_threads(const ResTree& m){
	restree_threads t;
	t.request = m.ncores_request; t.omp = m.ncores; t.eigen = m.eigen_threads;
	restree_apply_threads(t);
}

// The fixed-structure configuration (depth, knots, design, threads, theta)
// shared by the known-theta evaluator and the stored-structure predictor.
void configure_fixed(ResTree& m, const ModelSpec& sp, const GPM& th,
	int ncores){
	m.depth = sp.depth;
	m.r_knot = sp.r;
	m.baseline = sp.leaf_model;
	m.design = sp.design;
	m.nested_cand_factor = sp.nested_factor;
	configure_threads(m, ncores);
	m.covpar = th;
}

// A known-theta evaluator over the model data: the ONE owned copy of X and y
// is made here, from the mapped R memory, by the by-value constructor.
ResTree make_model(const ModelSpec& sp, const GPM& th, int ncores){
	ResTree m(sp.y, sp.X);
	configure_fixed(m, sp, th, ncores);
	return m;
}

// ==================== Fitted-state fingerprint and handle diagnostics ====================

// Fitted-state fingerprint (R/validation.R: .restree_fitted_state): FNV-1a
// 64-bit over the raw bytes of the numeric payload (so NA / NaN patterns and
// every last bit count) and over the strings, in order, with a separator
// between fields.  Returned as 16 hex digits.  Bit-exact and O(bytes); no
// serialization header, so a fit saved under one R version verifies under
// another.
// [[Rcpp::export]]
std::string fingerprint_cpp(SEXP num, Rcpp::CharacterVector chr){
	uint64_t h = 1469598103934665603ULL;
	auto mix = [&](const unsigned char* p, size_t n){
		for(size_t k=0; k<n; ++k){ h ^= (uint64_t)p[k]; h *= 1099511628211ULL; }
	};
	// Stream existing R arrays without concatenating the data. Convert integer
	// fields one value at a time in double representation.
	auto numeric_block = [&](SEXP block){
		if(Rf_isNull(block)) return;
		if(TYPEOF(block)==REALSXP){
			mix((const unsigned char*)REAL(block), (size_t)XLENGTH(block)*sizeof(double));
		}else if(TYPEOF(block)==INTSXP || TYPEOF(block)==LGLSXP){
			const int* p=TYPEOF(block)==INTSXP ? INTEGER(block) : LOGICAL(block);
			for(R_xlen_t i=0; i<XLENGTH(block); ++i){
				const double value=p[i]==NA_INTEGER ? NA_REAL : (double)p[i];
				mix((const unsigned char*)&value, sizeof(value));
			}
		}else Rcpp::stop("fingerprint payload must contain numeric arrays.");
	};
	if(TYPEOF(num)==VECSXP){
		for(R_xlen_t i=0; i<XLENGTH(num); ++i) numeric_block(VECTOR_ELT(num,i));
	}else numeric_block(num);
	const unsigned char sep = 0x1f;
	for(R_xlen_t k=0; k<chr.size(); ++k){
		mix(&sep, 1);
		if(Rcpp::CharacterVector::is_na(chr[k])){ continue; }
		const char* c = CHAR(STRING_ELT(chr, k));
		mix((const unsigned char*)c, std::strlen(c));
	}
	char buf[17];
	std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
	return std::string(buf);
}

// Compatibility diagnostic for both persistent inference and fitted handles.
// [[Rcpp::export]]
Rcpp::List smc_engine_info_cpp(SEXP handle){
    if(TYPEOF(handle)==EXTPTRSXP && R_ExternalPtrTag(handle)==prediction_handle_tag())
        return prediction_state_info(handle);
    return smc_info(smc_from_handle(handle));
}

// ==================== Backend capabilities and thread policy ====================

// Capability query of the compiled backend (R/startup.R restree_capabilities).
//   openmp        whether the package was built with OpenMP
//   physical_cores physical-core cap, independent of SMT (NA if unknown on
//                 macOS/Windows). Linux filters by the calling R thread's
//                 affinity; a failed/empty probe or incomplete topology gives
//                 a conservative cap of 1, not a node-wide/partial count.
//                 macOS/Windows report system topology, not Windows job quotas.
//                 A further cap comes
//                 from OMP_NUM_THREADS / SLURM_CPUS_PER_TASK / PBS_NP when set
//                 (wrapper.cpp, restree_core_cap).
//   max_threads   legacy informational logical-processor count, not the cap;
//                 omp_get_num_procs() is node-wide on older runtimes.
//   omp_max_threads  omp_get_max_threads(), informational: the default team
//                 size of a region without a num_threads clause. The package's
//                 effective Eigen count comes from thread_policy_cpp instead.
// [[Rcpp::export]]
Rcpp::List capabilities_cpp(){
    const int physical = restree_physical_cores();
#ifdef _OPENMP
	return Rcpp::List::create(
		Rcpp::_["openmp"] = true,
        Rcpp::_["physical_cores"] = physical>0 ? physical : NA_INTEGER,
		Rcpp::_["max_threads"] = omp_get_num_procs(),
		Rcpp::_["omp_max_threads"] = omp_get_max_threads(),
		Rcpp::_["parallel_backend"] = "OpenMP",
		Rcpp::_["max_model_depth"] = restree_max_model_depth,
		Rcpp::_["max_wn_depth"] = restree_max_wn_depth);
#else
	return Rcpp::List::create(
		Rcpp::_["openmp"] = false,
        Rcpp::_["physical_cores"] = physical>0 ? physical : NA_INTEGER,
		Rcpp::_["max_threads"] = 1,
		Rcpp::_["omp_max_threads"] = 1,
		Rcpp::_["parallel_backend"] = "serial",
		Rcpp::_["max_model_depth"] = restree_max_model_depth,
		Rcpp::_["max_wn_depth"] = restree_max_wn_depth);
#endif
}

// Main-thread policy shared by R validation and every native engine.
// [[Rcpp::export]]
Rcpp::List thread_policy_cpp(int request, bool warn=true){
    const auto t=restree_thread_policy(request,warn);
    return Rcpp::List::create(Rcpp::_["request"]=t.request,
        Rcpp::_["ncores"]=t.omp,Rcpp::_["eigen_threads"]=t.eigen,
        Rcpp::_["cap_source"]=t.source);
}
