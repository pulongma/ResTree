#ifndef RESTREE_PREDICTION_H
#define RESTREE_PREDICTION_H
#include "ResTree.h"

// Inference-independent reached trees. Source indices map the original SMC
// representatives to compact positions; empty means identity (restored/Full).
// Mixture weights remain explicit inputs at the prediction/R boundary.
struct FittedTrees {
    std::vector<PredictionNodes> nodes;
    std::vector<int> source_indices;
    int position(int source) const {
        if(source_indices.empty()) return source>=0 && (size_t)source<nodes.size() ? source : -1;
        for(size_t k=0;k<source_indices.size();++k)
            if(source_indices[k]==source) return (int)k;
        return -1;
    }
    PredictionNode node(int tree, int id) const {
        const auto* found=nodes[(size_t)tree].find_node(id);
        return found ? *found : PredictionNode{id,ACT_ABSENT,-1,std::numeric_limits<double>::quiet_NaN()};
    }
    size_t node_count() const {
        size_t n=0; for(const auto& tree:nodes) n+=tree.size(); return n;
    }
    size_t storage_bytes() const {
        size_t n=0; for(const auto& tree:nodes) n+=tree.storage_bytes(); return n;
    }
    void load_structures(const ResTree& model, const Eigen::Ref<const Eigen::MatrixXd>& S,
        const Eigen::Ref<const Eigen::MatrixXd>& J, const Eigen::Ref<const Eigen::MatrixXd>& cuts);
};

FittedTrees full_prediction_trees(const FullTreeGeometry& geometry);

// Owns only shared model context and fitted trees. No sampler, RNG streams,
// active particles, ESS histories or proposal caches are needed for prediction.
struct PredictionState {
    ResTree model;
    FittedTrees trees;
    PredictionState(ResTree&& model_, FittedTrees&& trees_)
        : model(std::move(model_)), trees(std::move(trees_)) {}
};

// Traversal depends on a concrete base model and fitted trees, never on SMC.
class Predictor {
public:
    Predictor(const ResTree& model, const FittedTrees& fitted) : eng(model), trees(fitted) {}
    void predict(const Eigen::Ref<const Eigen::MatrixXd>& Xnew, const std::vector<int>& reps,
        double sig2_use, Eigen::MatrixXd& par_mean, Eigen::MatrixXd& par_var,
        const Eigen::VectorXd* ynew=nullptr, Eigen::VectorXd* lpd_tree=nullptr,
        Eigen::MatrixXd* par_df=nullptr);
private:
    const ResTree& eng;
    const FittedTrees& trees;
    struct PredictWork {
        const Eigen::Ref<const Eigen::MatrixXd>* Xnew=nullptr;
        const std::vector<int>* reps=nullptr;
        Eigen::MatrixXd* df=nullptr;
        Eigen::MatrixXd* pm=nullptr;
        // Explained variance until finishing; a WN terminal replaces its cell
        // with leaf variance and sets lok, exactly as in the original traversal.
        Eigen::MatrixXd* pexpl=nullptr;
        std::vector<uint8_t>* lok=nullptr;
        const Eigen::VectorXd* ynew=nullptr;
        // Atomic additions from sibling tasks can differ in their last bits.
        Eigen::VectorXd* lpd=nullptr;
    };
    void predict_subtree(int tau, std::vector<int>&& T, std::vector<int>&& idx,
        Eigen::VectorXd&& resid, Eigen::MatrixXd&& Q, std::vector<int>&& nidx,
        Eigen::MatrixXd&& Qnew, Eigen::VectorXd bmin, Eigen::VectorXd bmax,
        PredictWork& W, std::vector<int> parent_cand=std::vector<int>());
};
#endif
