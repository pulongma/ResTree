##########################################################################
## The restree object family (formal S4 classes):
##
##   restree_theta  covariance parameters of the base Gaussian process
##                  (sig2, range, nugget, nu, tail, form, family, dtype);
##                  validated once, cheap to build, passed to every
##                  likelihood / fitting / prediction routine.
##   restree_model  the model SPECIFICATION: data (x, y) and every
##                  theta-independent tree setting (depth, r, leaf model,
##                  knot design, cut rule, CGM tree prior).
##                  Validated once at construction; the compiled routines
##                  read its slots directly.
##   restree        an object, fit or prediction: a restree_model plus theta,
##                  a kind marker and plain result/prediction/history lists.
##                  A fit keeps its estimation method, the
##                  realized tree structures with weights, and the
##                  method-specific results.
##
## restree_loglik(model, theta) and restree_fit(model, ...) take the
## specification; restree_predict(fit, xnew) takes the fit.  `$` reads slots
## first and then the results list, so a fit reads like a plain list
## (fit$logZ, fit$fit$state, fit$theta$sig2, ...).
##########################################################################

## ---------------------------------------------------------------------------
## restree_theta
## ---------------------------------------------------------------------------

#' Covariance parameters of the base Gaussian process
#'
#' Builds a validated \code{restree_theta} object.  The nugget is
#' \emph{relative}: the noise variance is \code{sig2 * nugget}.
#' \code{range} is a scalar for \code{form = "isotropic"} and one value per
#' input dimension for \code{form = "ARD"} or \code{"tensor"}.
#' For ARD/tensor, when a scalar range is supplied, likelihood evaluation or fitting expands
#' it to \code{rep(range, ncol(x))}. For example, \code{range = 0.2} with
#' five input columns starts all five ranges at 0.2; ARD/tensor fitting can
#' subsequently estimate a different range for each column. The supplied
#' theta object is not modified by this expansion.
#'
#' @param sig2 marginal variance (> 0).
#' @param range correlation range(s) (> 0).
#' @param nugget relative nugget (>= 0).
#' @param nu smoothness (> 0; at most 2 for \code{powexp} and \code{cauchy}).
#' @param tail tail parameter of the \code{CH} and \code{cauchy} families.
#' @param form covariance form:
#'   \itemize{
#'     \item \code{"isotropic"}: one shared \code{range} for every
#'       input dimension.
#'     \item \code{"ARD"} (the default): one \code{range} per input dimension
#'       (automatic relevance determination). Applies one correlation function
#'       to the Euclidean norm of coordinate differences divided by their ranges.
#'     \item \code{"tensor"}: a product of one-dimensional correlations,
#'       \eqn{\prod_{j=1}^d c(|x_j-x'_j|/\mathrm{range}_j)}. Each dimension
#'       has its own range; \code{family}, \code{nu} and \code{tail} are shared.
#'       This generally differs from ARD, but agrees for \code{family = "gauss"}.
#'   }
#' @param family covariance family (\eqn{h} the scaled distance):
#'   \itemize{
#'     \item \code{"matern"} (the default): Matern with smoothness \code{nu}.
#'     \item \code{"CH"}: confluent hypergeometric, smoothness \code{nu} and
#'       polynomial tail decay governed by \code{tail}.
#'     \item \code{"gauss"}: squared exponential \eqn{\exp(-h^2)}.
#'     \item \code{"powexp"}: powered exponential \eqn{\exp(-h^{\nu})},
#'       \code{nu} in \eqn{(0, 2]}.
#'     \item \code{"cauchy"}: generalised Cauchy \eqn{(1 + h^{\nu})^{-\mathrm{tail}/\nu}},
#'       \code{nu} in \eqn{(0, 2]}.
#'   }
#' @param dtype distance between inputs:
#'   \itemize{
#'     \item \code{"Euclidean"} (the default).
#'     \item \code{"GCD"}: great-circle distance; requires two input columns
#'       (longitude, then latitude) and \code{form = "isotropic"}.
#'   }
#' @return An object of class \code{restree_theta}.  Fields are read with
#'   \code{$} (e.g. \code{th$range}).
#' @examples
#' th <- restree_theta(sig2 = 1, range = 0.2, nugget = 0.1)
#' th$range
#' @export
restree_theta <- function(sig2 = 1, range = 0.1, nugget = 1e-3, nu = 2.5,
                          tail = 0.5, form = "ARD", family = "matern",
                          dtype = "Euclidean") {
  methods::new("restree_theta",
    sig2 = as.numeric(sig2), range = as.numeric(range),
    nugget = as.numeric(nugget), nu = as.numeric(nu), tail = as.numeric(tail),
    form = as.character(form), family = as.character(family),
    dtype = as.character(dtype))
}

#' @rdname restree_theta
#' @export
setClass("restree_theta", representation(
    sig2 = "numeric", range = "numeric", nugget = "numeric", nu = "numeric",
    tail = "numeric", form = "character", family = "character",
    dtype = "character"),
  prototype(sig2 = 1, range = 0.1, nugget = 1e-3, nu = 2.5, tail = 0.5,
            form = "ARD", family = "matern", dtype = "Euclidean"),
  validity = function(object) {
    msg <- character()
    one_pos <- function(v, name, zero_ok = FALSE) {
      if (length(v) != 1L || !is.finite(v) || (if (zero_ok) v < 0 else v <= 0))
        paste0(name, " must be one finite ", if (zero_ok) "nonnegative" else "positive",
               " number") else character()
    }
    msg <- c(msg, one_pos(object@sig2, "sig2"), one_pos(object@nugget, "nugget", TRUE),
             one_pos(object@nu, "nu"), one_pos(object@tail, "tail"))
    if (!length(object@range) || any(!is.finite(object@range)) || any(object@range <= 0))
      msg <- c(msg, "range must contain only finite positive values")
    if (length(object@form) != 1L || !object@form %in% c("isotropic", "ARD", "tensor"))
      msg <- c(msg, "form must be isotropic, ARD, or tensor")
    if (identical(object@form, "isotropic") && length(object@range) != 1L)
      msg <- c(msg, "form = 'isotropic' needs a scalar range; use form = 'ARD' for one range per dimension")
    if (length(object@family) != 1L ||
        !object@family %in% c("matern", "CH", "gauss", "powexp", "cauchy"))
      msg <- c(msg, "family must be one of: matern, CH, gauss, powexp, cauchy")
    ## validity, not preference: powexp exp(-(h/range)^nu) and generalized
    ## cauchy (1 + (h/range)^nu)^(-tail/nu) are positive definite only for
    ## nu <= 2 (Gneiting and Schlather 2004)
    if (length(object@family) == 1L && object@family %in% c("powexp", "cauchy") &&
        length(object@nu) == 1L && is.finite(object@nu) && object@nu > 2)
      msg <- c(msg, paste0("nu must lie in (0, 2] for family = ", object@family,
                           " (positive definiteness)"))
    if (length(object@dtype) != 1L || !object@dtype %in% c("Euclidean", "GCD"))
      msg <- c(msg, "dtype must be Euclidean or GCD")
    if (identical(object@dtype, "GCD") && !identical(object@form, "isotropic"))
      msg <- c(msg, "dtype = 'GCD' requires isotropic covariance")
    if (length(msg)) msg else TRUE
  })

## Coerce a theta given as a restree_theta or a named list, validate it, and
## resolve the data-dependent parts (ARD/tensor range length, GCD dimension).
## Returns a restree_theta.
.restree_theta_for <- function(theta, p, name = "theta") {
  if (is.null(theta)) theta <- list()
  if (methods::is(theta, "restree_theta")) {
    th <- theta
  } else if (is.list(theta)) {
    bad <- setdiff(names(theta), c("sig2", "range", "nugget", "nu", "tail",
                                   "form", "family", "dtype"))
    if (length(bad)) stop("unknown ", name, " component(s): ",
                          paste(bad, collapse = ", "), call. = FALSE)
    th <- do.call(restree_theta, theta)
  } else {
    stop(name, " must be a restree_theta object or a named list", call. = FALSE)
  }
  if (th@form %in% c("ARD", "tensor") && length(th@range) == 1L) th@range <- rep(th@range, p)
  if (th@form %in% c("ARD", "tensor") && length(th@range) != p)
    stop("form = 'ARD' or 'tensor' needs length(range) == ncol(x)", call. = FALSE)
  if (th@dtype == "GCD" && p != 2L)
    stop("dtype = 'GCD' requires two input columns", call. = FALSE)
  th
}

#' @rdname restree_theta
#' @param x,object a \code{restree_theta} (for \code{$} and \code{show}).
#' @param name slot name for \code{$}.
#' @export
setMethod("$", "restree_theta", function(x, name) methods::slot(x, name))

#' @rdname restree_theta
#' @export
setMethod("names", "restree_theta", function(x) methods::slotNames("restree_theta"))

#' @rdname restree_theta
#' @export
setMethod("show", "restree_theta", function(object) {
  cat("restree_theta: ", object@family, " (", object@form, ", ", object@dtype, ")\n",
      "  sig2 ", format(object@sig2, digits = 5), " ; range ",
      paste(format(object@range, digits = 5), collapse = " "),
      " ; nugget ", format(object@nugget, digits = 5),
      " ; nu ", format(object@nu, digits = 4),
      if (object@family %in% c("CH", "cauchy")) paste0(" ; tail ", format(object@tail, digits = 4)) else "",
      "\n", sep = "")
  invisible(object)
})

## ---------------------------------------------------------------------------
## restree_model
## ---------------------------------------------------------------------------

.restree_model_slots <- c("x", "y", "depth", "r", "leaf_model", "design",
                          "cut_method", "prior_rho", "prior_beta", "seed", "tree",
                          "nested_factor", "cut_candidates")

#' Residual-tree model specification
#'
#' Builds a validated \code{restree_model}: the data and every
#' theta-independent setting of a residual-tree Gaussian process.  The object
#' is what \code{\link{restree_loglik}} evaluates and \code{\link{restree_fit}}
#' fits; validation happens once here, so repeated likelihood evaluations at
#' different covariance parameters carry no R-level checking cost.
#'
#' The covariance is built from a binary tree over the unit cube: internal
#' nodes carry up to \code{r} space-filling knots (nested predictive-process
#' layers), and each terminal node closes its branch with the leaf model.
#' \code{Full} keeps the dense residual covariance within every leaf and
#' therefore uses one fixed tree, which is built here, once, and stored in the
#' object (\code{m$tree}: the realised \code{S}, \code{J}, \code{cuts} and,
#' per node, the rows it holds and the knots the design selects); every later
#' likelihood evaluation, the MLE of \code{\link{restree_fit}} and prediction
#' read that stored tree instead of rebuilding it.  \code{PP} (predictive-process leaf, at most
#' \code{r} knots plus a diagonal remainder) and \code{WN} (conjugate
#' white-noise leaf with a Normal-Inverse-Gamma marginal) admit tree
#' inference by sequential Monte Carlo.
#'
#' @param x numeric matrix of inputs, one row per observation, every
#'   coordinate in \code{[0, 1]}.
#' @param y numeric response, one value per row of \code{x}.
#' @param depth tree depth (\code{0} to \code{20}). For \code{PP} it is a
#'   hard ceiling. For \code{WN}, zero specifies one WN leaf regardless of
#'   \code{n} and \code{r}; a positive depth has no structural boundary: every
#'   nonempty node compares stop versus split, at every level. A node with at
#'   most \code{r} observations spends all of them as knots (empty residual
#'   leaf, empty children) and still competes its split with the WN stop leaf.
#'   Each split passes only \code{n - min(r, n)} rows down, so the tree is
#'   finite (at most \code{ceil(n / r)} levels) and grows past the initial
#'   depth as the data need; a positive depth only sets the initial
#'   comparison depth and the default particle budget (see
#'   \code{nparticles} in \code{\link{restree_fit}}). The initial depth is
#'   capped at 20, but the realized WN tree may be deeper (up to 25, the
#'   representation limit). A selected split beyond that limit raises an
#'   explicit error rather than silently accepting a leaf or rejecting an
#'   EB/PMMH parameter proposal.
#'   \code{NULL} (default) uses \eqn{\max(0, \lceil \log_2(n / r) \rceil)},
#'   a convenience rule, not a feasibility bound for a complete tree or a
#'   statistically optimal depth. Every later default reads the stored
#'   model depth: the
#'   SMC particle count of \code{\link{restree_fit}} and
#'   \code{\link{restree_loglik}} defaults to \eqn{100\,\mathrm{depth}} for
#'   both PP and WN, using the configured model depth, not the realized WN
#'   tree depth.
#'   Depth-zero PP/WN models use one exact terminal structure.
#'   For \code{Full}, this is an initial storage depth, not a stopping
#'   limit: branches split while they have more than \code{r} observations,
#'   and the stored depth increases as needed. Every Full terminal has at
#'   most \code{r} observations. The existing depth-20 representation limit
#'   raises an error if the size constraint needs a deeper tree. Uniform
#'   cuts remain the default; explicit middle cuts use the same size rule.
#'   A Full root is a single dense GP only when \code{n <= r}, even if
#'   \code{depth = 0} was requested. PP/WN depth zero remains a single leaf.
#'   If \code{n < r}, Full and WN keep the requested knot budget (Full stops
#'   at the root; a WN node spends all \code{n} rows as knots); PP resets
#'   the budget and preserves the selected depth.
#' @param r knots per internal node and maximum Full leaf size (default
#'   \code{60}). For WN it is the knot budget, not a maximum leaf size: a WN
#'   node with at most \code{r} rows spends all of them as knots and still
#'   compares stop versus split. For PP only, if \code{n < r}, a warning is issued and
#'   \code{r} is reset to \code{min(n, 30)}. Knots are taken
#'   from the node's own observations. A hypothetical complete depth-D tree
#'   spending \code{r} knots at every split needs at least
#'   \code{r * (2^D - 1)} training rows just for its internal knots.
#'   The actual Full tree need not be complete. This is not a depth bound
#'   for adaptive PP/WN trees, whose nodes may split with
#'   fewer than \code{r} observations and then spend all available rows.
#' @param leaf_model terminal model: \code{"PP"} (default), \code{"WN"}, or
#'   \code{"Full"}. See \strong{Leaf models} for their covariance and
#'   tree-inference rules.
#' @section Leaf models:
#' Internal nodes contribute predictive-process components based on their
#' knots. The terminal model describes the residual response after those
#' ancestor contributions have been removed:
#' \describe{
#'   \item{\code{PP}}{A predictive-process approximation to the residual GP,
#'     with a diagonal conditional remainder. A leaf with at most \code{r}
#'     observations uses its dense residual covariance. Tree uncertainty is
#'     integrated by SMC up to the configured depth.}
#'   \item{\code{WN}}{A white-noise model with a conjugate prior on the leaf
#'     mean and variance. Tree uncertainty is integrated by SMC. At positive
#'     depth, nonempty nodes remain eligible for stop/split comparison beyond
#'     the initial depth; depth zero specifies a single WN leaf.}
#'   \item{\code{Full}}{The dense residual GP covariance at each leaf of one
#'     fixed tree, without SMC. The tree splits until each terminal holds at
#'     most \code{r} observations; the root is an exact dense GP when
#'     \code{r >= nrow(x)}.}
#' }
#' Full and PP retain the base GP covariance parameters in their leaf
#' likelihoods; WN integrates its local leaf parameters. These models have
#' different likelihood targets, so tree averaging does not guarantee a
#' closer approximation to the dense GP likelihood.
#' @param design knot placement at each internal node:
#'   \itemize{
#'     \item \code{"maximin"} (the default): \code{r} space-filling knots
#'       chosen by sequential maximin among the node's observations.
#'     \item \code{"boundary"}: half of the knots near the node's mid-planes
#'       (the candidate cutting planes), half space-filling.
#'     \item \code{"nested"} (\code{PP} with \code{uniform} / \code{balanced}
#'       cuts only): the root spends \code{r} maximin knots and every other
#'       node inherits its knots -- the first \code{nested_factor * r}
#'       residual rows of its parent in maximin order (the parent's candidate
#'       set) that fall on its side of the cut -- so a node holds between 0
#'       and \code{nested_factor * r} knots depending on the cut (about
#'       \code{r} for a balanced cut at the default \code{nested_factor =
#'       2}); a node with at most \code{r} rows keeps all of them as knots.
#'       The nesting lets the sampler integrate the random-cut prior exactly
#'       at every node (as it does for \code{WN}): the candidate leaves of
#'       all rank cells of a coordinate come from one sweep costing
#'       \eqn{O(n_v r^2)}, particles draw the split and the cell from the
#'       exact node posterior, and particles that draw the same cell share
#'       the child, which removes the per-particle candidate evaluation and
#'       its memory.
#'   }
#' @param nested_factor (\code{design = "nested"} only) candidates per node
#'   as a multiple of \code{r}: a node hands its first
#'   \code{nested_factor * r} residual rows in maximin order to its children,
#'   so a child holds between 0 and \code{nested_factor * r} knots, about
#'   \code{nested_factor * r / 2} for a balanced cut.  The default \code{2}
#'   keeps a balanced child near \code{r} knots. Larger values increase the
#'   candidate budget and change the model, with sweep cost growing with
#'   the square of that budget; better GP approximation is not guaranteed.
#' @param cut_method the cut rule of a split: the prior on the cut point
#'   \eqn{c} of the split coordinate \eqn{J}, given the node.  Write
#'   \eqn{x_{(1)} \le \dots \le x_{(m)}} for the sorted \eqn{J}-coordinates
#'   of the node's residual observations (its rows that are not knots).
#'   \itemize{
#'     \item \code{"uniform"} (the default): the Bayesian CART cut prior of
#'       Chipman, George and McCulloch (1998).  Each of the \eqn{m - 1} rank
#'       intervals \eqn{[x_{(k)}, x_{(k+1)}]} -- each inducing one distinct
#'       partition of the node, with \eqn{k} observations on the left -- has
#'       prior probability \eqn{1/(m-1)}, i.e. the prior is uniform over the
#'       quantiles of the node's observations, and the cut is placed
#'       uniformly within the interval.  A run of tied coordinates keeps its
#'       point mass and its ties go right.
#'     \item \code{"balanced"}: the same rank intervals, reweighted by the
#'       Beta(2, 2) density on the rank scale: interval
#'       \eqn{[x_{(k)}, x_{(k+1)}]}, for \eqn{k = 1, \ldots, m-1}, receives
#'       \eqn{\int_{(k-1)/(m-1)}^{k/(m-1)} 6u(1-u)\,du}, so
#'       cuts near the median rank are favoured and near-empty children
#'       discouraged while every partition keeps positive prior mass.
#'     \item \code{"middle"}: the midpoint of the node's box (deterministic).
#'     \item \code{"median"}: the median of the residual coordinates
#'       (deterministic).
#'   }
#'   Under the two deterministic rules the cut is a function of the node
#'   path, so particles that share a path share its candidate splits.
#'   Sampling, resampling, storage and export still depend on particle count,
#'   and more particles can visit more paths. Under the two
#'   random rules, WN (and PP with \code{design = "nested"}) integrates the
#'   prior over all rank intervals exactly, once per distinct node path, and
#'   draws the split and a cut within its interval from the node posterior;
#'   PP with the other designs samples candidates per coordinate and particle
#'   (below), so its cost grows with the particle
#'   count.
#' @param cut_candidates Ordinary PP with a random cut rule: integer \eqn{M}
#'   (default 30 for ordinary PP \code{"uniform"}; 1 otherwise).
#'   Under \code{"uniform"}, every coordinate always includes
#'   one candidate from the rank interval containing the node-region midpoint
#'   (the nearest end interval if the midpoint is outside the cut support),
#'   and \eqn{M} independent candidates from the complementary rank intervals.
#'   Thus the default uses 31 candidates per coordinate when there is more
#'   than one rank interval; \code{cut_candidates = 1} restores the smaller
#'   budget of two candidates. Explicit budgets are retained unchanged.
#'   With \eqn{g > 1} rank intervals, their integration weights are respectively
#'   \eqn{1/g} and \eqn{(1-1/g)/M}. Cuts are sampled within their intervals:
#'   the geometric midpoint is a guide, not an added point mass in the prior.
#'   With at most one interval, one candidate suffices. Existing point masses
#'   from tied coordinates are retained. Under \code{"balanced"}, the
#'   \eqn{M} candidates remain independent draws from the Beta rank prior with
#'   weights \eqn{1/M}. Weighted child marginals estimate the cut-integrated
#'   marginal without bias, and the decisions
#'   (stop, coordinate, and the candidate within the coordinate) are drawn
#'   from the posterior built on those estimates, and the particle weight
#'   increment is the log of the estimated one-step normaliser minus the
#'   node's stopped log-likelihood, for every outcome. The evidence estimate
#'   targets the same tree-prior evidence for every \eqn{M}. Candidates are
#'   sampled with replacement within their strata: setting \eqn{M} equal to the number of rank
#'   intervals is not exact enumeration, and neither log-evidence accuracy
#'   nor total runtime is guaranteed to improve monotonically. Distinct
#'   candidate child blocks can be shared across particles.
#'   Ignored by WN, by deterministic cuts and by \code{design = "nested"}:
#'   nested PP integrates its rank-cell cut prior exactly at every split, so
#'   it has no candidate draw and any \code{cut_candidates} value leaves it
#'   unchanged (the default is 1 there).
#' @param prior_rho,prior_beta CGM (1998) tree prior: a node
#'   at depth \eqn{d} splits with probability \eqn{(1-\rho)(1+d)^{-\beta}};
#'   \code{prior_rho} is the root stop probability and \code{prior_beta} the
#'   depth penalty (\code{0} gives a constant stop probability).
#' @param seed seed of the random cut rules (\code{"uniform"},
#'   \code{"balanced"}) of the fixed \code{Full} tree, which is built when the
#'   model is created; \code{"middle"} and \code{"median"} cuts do not use it.
#'   The trees of \code{PP} and \code{WN} are inferred, so for them this
#'   argument is stored but unused -- their sampler seed is the \code{seed} of
#'   \code{\link{restree_loglik}} / \code{\link{restree_fit}}.
#'
#' @section Knots at a split:
#' How many knots a split spends is fixed by the leaf model, not by an
#' option.  \code{Full} uses one fixed tree, built when the model is created
#' and stored in it: a node splits exactly when it has more than \code{r}
#' observations, every split spends exactly \code{r} knots (chosen by
#' \code{design} among the node's rows), the split dimension cycles over the
#' coordinates with the depth, and the cut comes from \code{cut_method}.  \code{PP} and \code{WN} always split
#' adaptively: a node with fewer than \code{r} observations may still
#' split, spending all its rows as knots (the children are then empty and
#' the split is a terminal dense Gaussian block; for \code{WN} such a split
#' adds a predictive-process layer).  A supplied or stored structure is
#' rescored with \eqn{r_v = \min(r, n_v)} knots at every split, whatever
#' the leaf model.
#'
#' @section Full-tree storage and compatibility:
#' The Full size rule counts the rows arriving at a node, after ancestor
#' knots have been removed. Each split removes exactly \code{r} more rows
#' as knots and routes the remaining rows by \code{x[, J] < cut}; ties go
#' right. Requested depth controls initial storage, not tree truncation.
#' A larger requested depth can allocate larger compatibility arrays without
#' changing the reached tree. These arrays grow as \eqn{2^{\mathrm{depth}}};
#' small leaves alone do not guarantee small total memory or linear runtime.
#' Rebuild old Full models with \code{restree_model()} and refit if their
#' stored tree violates the new split rule. A rebuilt model can have different
#' likelihoods, estimates and predictions; this is a model change, not just
#' a speed optimization. For a dense GP reference use \code{r = nrow(x)}.
#' In \code{model$tree}, \code{knots} contains concatenated node-local row
#' positions, not a vector of global root-knot indices. Select a node's block
#' with \code{knot_offset} and \code{knot_size}, then map through its
#' \code{rows} block when global training-row indices are needed.
#' @return An object of class \code{restree_model}.  Slots are read with
#'   \code{$} (e.g. \code{m$depth}; \code{m$tree} is the stored fixed tree of a
#'   \code{Full} model, an empty list otherwise).
#' @references Chipman, H. A., George, E. I. and McCulloch, R. E. (1998)
#'   Bayesian CART model search. \emph{Journal of the American Statistical
#'   Association} 93, 935--948.
#' @examples
#' set.seed(1)
#' X <- matrix(runif(400), 200, 2); y <- sin(4 * pi * X[, 1]) + rnorm(200, sd = 0.3)
#' m <- restree_model(X, y, depth = 3, r = 15, leaf_model = "PP")
#' m
#' @export
restree_model <- function(x, y, depth = NULL, r = 60L, leaf_model = c("PP", "WN", "Full"),
                          design = c("maximin", "boundary", "nested"),
                          cut_method = c("uniform", "middle", "median", "balanced"),
                          prior_rho = 0.05, prior_beta = 2, seed = 42L,
                          nested_factor = 2L, cut_candidates = 30L) {
  leaf_model <- .restree_leaf_model(if (missing(leaf_model)) "PP" else leaf_model,
                                    "leaf_model")
  nested_factor <- .restree_integer(nested_factor, "nested_factor", 1L)
  design <- match.arg(design)
  cut_method <- match.arg(cut_method)
  # Increase only ordinary PP uniform proposals. In particular, nested PP
  # integrates cuts directly and must retain its supported default M = 1.
  if (missing(cut_candidates) &&
      (leaf_model != "PP" || design == "nested" || cut_method != "uniform"))
    cut_candidates <- 1L
  if (design == "nested" && (leaf_model != "PP" || !cut_method %in% c("uniform", "balanced")))
    stop("design = \"nested\" needs leaf_model = \"PP\" and cut_method \"uniform\" or \"balanced\"",
         call. = FALSE)
  x <- as.matrix(x)
  if (!is.numeric(x) || !length(x)) stop("x must be a numeric matrix", call. = FALSE)
  storage.mode(x) <- "double"
  r <- .restree_integer(r, "r", 1L)
  ## The initial depth is capped for every leaf model. For WN it is only the
  ## initial comparison depth: the realized WN tree grows past it as the data
  ## need (up to .restree_max_wn_depth()). The default particle budget is
  ## 100 times this configured depth, not the realized WN tree depth.
  md <- .restree_max_model_depth()
  if (is.null(depth))   # initial convenience boundary, not complete-tree feasibility
    depth <- min(md, max(0L, as.integer(ceiling(log2(max(1, nrow(x)) / r)))))
  depth <- .restree_integer(depth, "depth", 0L, md)
  seed <- .restree_integer(seed, "seed", 0L)
  cut_candidates <- .restree_integer(cut_candidates, "cut_candidates", 1L)
  if (length(x) && (any(!is.finite(x)) || any(x < 0) || any(x > 1)))
    stop("x must lie in the unit hypercube: every coordinate in [0, 1] (the tree ",
         "partition is defined on it). Rescale each column, e.g. (x - min) / (max - min)",
         call. = FALSE)
  ## PP leaves evaluate with r knots and need r < n; a WN node with n <= r
  ## simply spends all n rows as knots (knot_count = min(r, n)) and Full stops
  ## at the root, so only PP resets the budget.
  if (nrow(x) < r && leaf_model == "PP") {
    old_r <- r
    r <- min(nrow(x), 30L)
    warning(sprintf("n = %d is smaller than r = %d; resetting r to min(n, 30) = %d. The selected depth remains %d.",
                    nrow(x), old_r, r, depth), call. = FALSE)
  }
  ## The Full leaf model's tree does not depend on the covariance parameters,
  ## so it is built exactly once, here, and travels with the model: every
  ## likelihood evaluation, optimizer step and prediction reads it.
  tree <- if (leaf_model == "Full")
    fixed_tree_geometry_cpp(x, depth, r, design, cut_method, seed, 1L) else list()
  ## Full extends its requested depth until every terminal holds at most r
  ## rows. All consumers, including stored-structure prediction, use that depth.
  if (leaf_model == "Full") depth <- as.integer(tree$depth)
  methods::new("restree_model",
    x = x, y = as.numeric(y), depth = depth, r = r,
    leaf_model = leaf_model, design = design, cut_method = cut_method,
    prior_rho = .restree_number(prior_rho, "prior_rho", 0, 1),
    prior_beta = .restree_number(prior_beta, "prior_beta", 0, Inf),
    seed = seed, tree = tree, nested_factor = nested_factor,
    cut_candidates = cut_candidates)
}

## ---- validity of restree_model -----------------------------------------
## One shared core (the data and the tree settings every leaf model reads),
## then a check specific to the leaf model, because the three are different
## models of the same slots:
##   Full : one FIXED tree built at construction and stored in `tree`; the
##          CGM prior (prior_rho, prior_beta) is not read.  The stored tree
##          uses reached records (format 2); native validation checks the size
##          rule and exact consistency of rows, knots and routing. It does not
##          re-run knot selection or claim to regenerate every original choice.
##   PP   : the tree is inferred (SMC / EB / PMMH); `tree` is empty; the CGM
##          prior is read at every split decision.
##   WN   : as PP for every model slot (the leaf differs in the covariance
##          parameters and the terminal marginal, i.e. in restree_theta and
##          the engine, not in the model); kept as its own entry so a
##          WN-specific model constraint has one place to go.
## Each returns a character vector of messages (empty = valid).

.restree_validity_core <- function(object) {
  msg <- character()
  if (!is.matrix(object@x) || !is.double(object@x))
    msg <- c(msg, "x must be a double matrix")
  if (nrow(object@x) != length(object@y))
    msg <- c(msg, "x and y have different numbers of observations")
  if (!length(object@y)) msg <- c(msg, "x and y must not be empty")
  if (is.matrix(object@x) && ncol(object@x) < 1L)
    msg <- c(msg, "x must have at least one column")
  if (length(object@y) && any(!is.finite(object@y)))
    msg <- c(msg, "y must contain only finite values")
  if (length(object@x) && any(!is.finite(object@x)))
    msg <- c(msg, "x must contain only finite values")
  if (length(object@x) && (any(object@x < 0) || any(object@x > 1)))
    msg <- c(msg, paste0("x must lie in the unit hypercube: every coordinate in [0, 1] ",
                         "(the tree partition is defined on it). Rescale each column, e.g. ",
                         "(x - min) / (max - min)"))
  md <- .restree_max_model_depth()
  if (length(object@depth) != 1L || is.na(object@depth) ||
      object@depth < 0L || object@depth > md)
    msg <- c(msg, sprintf("depth must be one integer in [0, %d]", md))
  if (length(object@r) != 1L || is.na(object@r) || object@r < 1L)
    msg <- c(msg, "r must be one integer >= 1")
  if (length(object@leaf_model) != 1L ||
      !object@leaf_model %in% c("Full", "PP", "WhiteNoise"))
    msg <- c(msg, "leaf_model must be Full, PP, or WN")
  if (length(object@design) != 1L || !object@design %in% c("maximin", "boundary", "nested"))
    msg <- c(msg, "design must be maximin, boundary or nested")
  if (identical(object@design, "nested") &&
      (!identical(object@leaf_model, "PP") || !object@cut_method %in% c("uniform", "balanced")))
    msg <- c(msg, "design nested needs leaf_model PP and cut_method uniform or balanced")
  if (length(object@nested_factor) != 1L || is.na(object@nested_factor) || object@nested_factor < 1L)
    msg <- c(msg, "nested_factor must be one integer >= 1")
  if (length(object@cut_method) != 1L ||
      !object@cut_method %in% c("middle", "median", "uniform", "balanced"))
    msg <- c(msg, "cut_method must be one of: middle, median, uniform, balanced")
  if (length(object@cut_candidates) != 1L || is.na(object@cut_candidates) ||
      object@cut_candidates < 1L)
    msg <- c(msg, "cut_candidates must be one integer >= 1")
  if (length(object@seed) != 1L || is.na(object@seed) || object@seed < 0L)
    msg <- c(msg, "seed must be one nonnegative integer")
  if (!is.list(object@tree)) msg <- c(msg, "slot tree must be a list")
  msg
}

## The CGM tree prior, read by the inferred-tree leaf models only.
.restree_validity_prior <- function(object) {
  msg <- character()
  if (length(object@prior_rho) != 1L || !is.finite(object@prior_rho) ||
      object@prior_rho < 0 || object@prior_rho > 1)
    msg <- c(msg, "prior_rho must lie in [0, 1]")
  if (length(object@prior_beta) != 1L || !is.finite(object@prior_beta) ||
      object@prior_beta < 0)
    msg <- c(msg, "prior_beta must be finite and nonnegative")
  msg
}

## Full geometry uses sorted reached-node ids with contiguous row/knot blocks.
## Native conversion validates both this compact format and legacy dense slots:
## exact size-based splits, child routing, knot positions and root row order.
.restree_validity_full <- function(object) {
  tryCatch({
    full_tree_validity_cpp(object)
    character()
  }, error = function(e) conditionMessage(e))
}

## PP: inferred tree, so no stored tree; the CGM prior is read.
.restree_validity_pp <- function(object) {
  msg <- .restree_validity_prior(object)
  if (length(object@tree))
    msg <- c(msg, "a PP model infers its tree; slot tree must be empty")
  msg
}

## WN: the same model slots as PP (the leaf differs in restree_theta and the
## engine, not here).  Own entry so a WN-only model constraint has one home.
.restree_validity_wn <- function(object) {
  msg <- .restree_validity_prior(object)
  if (length(object@tree))
    msg <- c(msg, "a WN model infers its tree; slot tree must be empty")
  msg
}

#' @rdname restree_model
#' @export
setClass("restree_model", representation(
    x = "matrix", y = "numeric", depth = "integer", r = "integer",
    leaf_model = "character", design = "character", cut_method = "character",
    prior_rho = "numeric", prior_beta = "numeric",
    seed = "integer", tree = "list", nested_factor = "integer",
    cut_candidates = "integer"),
  prototype(x = matrix(numeric(), 0L, 0L), y = numeric(), depth = 1L, r = 1L,
            leaf_model = "PP", design = "maximin", cut_method = "uniform",
            prior_rho = 0.05, prior_beta = 2, seed = 42L, tree = list(),
            nested_factor = 2L, cut_candidates = 30L),
  validity = function(object) {
    msg <- .restree_validity_core(object)
    if (length(msg)) return(msg)          # the leaf checks assume valid shapes
    msg <- switch(object@leaf_model,
      Full = .restree_validity_full(object),
      PP = .restree_validity_pp(object),
      WhiteNoise = .restree_validity_wn(object))
    if (length(msg)) msg else TRUE
  })

#' @rdname restree_model
#' @param object a \code{restree_model} (for \code{show}); for \code{$} and
#'   \code{names}, \code{x} is the object.
#' @param name slot name for \code{$}.
#' @export
setMethod("$", "restree_model", function(x, name) methods::slot(x, name))

#' @rdname restree_model
#' @export
setMethod("names", "restree_model", function(x) .restree_model_slots)

#' @rdname restree_model
#' @export
setMethod("show", "restree_model", function(object) {
  cat("restree_model\n")
  cat("  data      :", nrow(object@x), "observations,", ncol(object@x), "inputs\n")
  cat("  tree      : depth", object@depth, "; r =", object@r,
      "; leaf model", object@leaf_model, "; design", object@design,
      if (identical(object@design, "nested")) paste0("(", object@nested_factor, " r candidates)") else "",
      "; cuts", object@cut_method,
      if (object@leaf_model == "PP" && object@design != "nested" &&
          object@cut_method %in% c("uniform", "balanced"))
        if (object@cut_method == "uniform")
          paste0(" (M = ", object@cut_candidates, "; up to ",
                 as.double(object@cut_candidates) + 1, " candidates per coordinate)")
        else paste0(" (", object@cut_candidates, " candidates per coordinate)") else "",
      "\n")
  if (identical(object@leaf_model, "Full") && length(object@tree)) {
    cat("  fixed tree: ", object@tree$n_split, " splits, ", object@tree$n_leaf,
        " non-empty leaves (built at construction; cut seed ", object@seed, ")\n", sep = "")
  } else {
    cat("  tree prior: p(split at depth d) = ", format(1 - object@prior_rho, digits = 4),
        "/(1+d)^", format(object@prior_beta, digits = 4), "\n", sep = "")
  }
  invisible(object)
})

## ---------------------------------------------------------------------------
## restree (a fit)
## ---------------------------------------------------------------------------

.restree_fit_slots <- c("theta", "method", "structure", "kind", "prediction", "results", "smc", "mcmc")

#' Residual-tree objects, fits and predictions
#'
#' \code{\link{restree_fit}} returns a formal (S4) object of class
#' \code{restree}.  It inherits every slot of the \code{\link{restree_model}}
#' it was fitted from (\code{x}, \code{y}, \code{depth}, \code{r},
#' \code{leaf_model}, \code{design}, \code{cut_method}, \code{prior_rho},
#' \code{prior_beta}) and adds \code{theta} (the
#' \code{\link{restree_theta}} used or estimated), \code{method}, the
#' realized tree structures in \code{structure} (the node table
#' \code{nodes} and the weights \code{w}; \code{\link{restree_trees}}),
#' method-specific \code{results}, and ordinary \code{smc}/\code{mcmc} lists.
#' The package defines only three S4 classes: \code{restree_model},
#' \code{restree_theta} and \code{restree}. The \code{kind} slot distinguishes
#' an assembled \code{"object"}, a \code{"fit"}, and a \code{"prediction"}.
#' The \code{prediction} slot is an ordinary list, including on a fit made
#' with \code{xnew}. Prediction objects retain the fit's SMC state, MCMC
#' history and stored trees for \code{restree_diagnostics} and
#' \code{restree_trees}. Continue to use the original fit for new predictions.
#' The \code{smc} slot holds the SMC record (under PMMH, the final SMC at
#' point-estimate theta); it is empty for Full MLE. The \code{mcmc} slot
#' holds PMMH chain/proposals on the natural parameter scale, acceptance,
#' evidence and adaptation histories, settings and retained prediction indices.
#' After prediction, \code{mcmc$prediction} caches moments by selected chain
#' iteration; final SMC ensembles and their weights are saved in \code{mcmc$trees}.
#' It does not retain one native SMC engine per iteration.
#' Histories are shared via R copy-on-write, not recomputed when wrapping predictions.
#'
#' \code{$} is read-only: it reads slots first, then \code{results} entries,
#' then read-only fit/MCMC aliases and prediction-list entries. Thus \code{pred$prediction$mean} and
#' \code{pred$mean} both work. Training slots take precedence; held-out
#' responses are \code{pred$prediction$y}, not \code{pred$y}.
#' No ResTree \code{$<-} method is provided. Direct slot edits are still
#' possible in R and do not establish fitted-state integrity: refit after
#' changing inference-defining quantities. Older saved objects without the
#' new slots must be rebuilt and refitted.
#'
#' Methods: \code{show}, \code{summary} (sampler and optimizer
#' diagnostics), \code{plot} (the maximum a posteriori partition, using
#' the suggested ggplot2 package), and \code{predict} (dispatching to
#' \code{\link{restree_predict}}).
#' These are S4 methods only. After \code{library(ResTree)}, use the
#' unqualified generics. Without attaching the package, use
#' \code{ResTree::plot}, \code{ResTree::predict} and \code{ResTree::summary};
#' the original package-qualified S3 generics
#' (such as \code{stats::predict}) do not provide ResTree dispatch.
#' Display uses \code{methods::show(object)}; ordinary \code{print(object)}
#' with no formatting arguments also invokes \code{show} when the methods
#' package is attached. There is no separate \code{ResTree::print} export.
#'
#' @name restree-class
NULL

#' Assemble a residual-tree object
#'
#' Constructs and validates an S4 \code{restree} without fitting, sampling,
#' likelihood evaluation or prediction. Use \code{restree_fit} to estimate a
#' model and \code{restree_predict} to predict from an intact fit.
#' @rdname restree-class
#' @param model a \code{restree_model}; a \code{restree} contributes only its
#'   inherited model specification.
#' @param theta a \code{restree_theta} or named parameter list.
#' @param method \code{"none"} for an assembled object, or the originating
#'   inference method: \code{"mle"}, \code{"smc"}, \code{"ebayes"}, \code{"pmmh"}.
#' @param structure stored tree nodes and weights for a fit.
#' @param results ordinary named list of fit results.
#' @param prediction ordinary named list of predictive moments, components,
#'   weights and optional held-out responses.
#' @param smc ordinary named list of SMC tree records, weights, diagnostics
#'   and optional native cache; empty when no SMC was run.
#' @param mcmc ordinary named list of PMMH parameters, histories and optional
#'   per-iteration prediction cache; empty for non-PMMH fits.
#' @param kind \code{"object"} (default), \code{"fit"}, or \code{"prediction"}.
#' @return \code{restree()} returns an S4 \code{restree}. Its constructor
#'   checks representation, not the provenance of user-supplied fit results.
#' @export
restree <- function(model, theta = NULL, method = "none", structure = list(),
                    results = list(), prediction = list(),
                    kind = c("object", "fit", "prediction"), smc = list(), mcmc = list()) {
  if (!methods::is(model, "restree_model"))
    stop("model must be a restree_model", call. = FALSE)
  kind <- match.arg(kind)
  methods::new("restree", .restree_model_of(model),
    theta = .restree_theta_for(theta, ncol(model@x)),
    method = as.character(method), kind = kind,
    structure = structure, results = results, prediction = prediction, smc = smc, mcmc = mcmc)
}

#' @rdname restree-class
#' @export
setClass("restree", contains = "restree_model",
  representation(theta = "restree_theta", method = "character",
                 structure = "list", results = "list", kind = "character",
                 prediction = "list", smc = "list", mcmc = "list"),
  prototype(theta = methods::new("restree_theta"), method = "none",
            structure = list(), results = list(), kind = "object", prediction = list(),
            smc = list(), mcmc = list()),
  validity = function(object) {
    msg <- character()
    if (length(object@kind) != 1L ||
        !object@kind %in% c("object", "fit", "prediction"))
      return("kind must be object, fit, or prediction")
    theta_valid <- methods::validObject(object@theta, test = TRUE)
    if (!isTRUE(theta_valid)) msg <- c(msg, paste("theta:", theta_valid))
    if (length(object@method) != 1L ||
        !object@method %in% c("none", "mle", "smc", "ebayes", "pmmh"))
      msg <- c(msg, "method must be none, mle, smc, ebayes, or pmmh")
    if (object@kind == "fit" && identical(object@method, "none"))
      msg <- c(msg, "a fit must specify an inference method")
    if (object@kind == "object" &&
        (length(object@results) || length(object@prediction) || length(object@structure) ||
         length(object@smc) || length(object@mcmc) ||
         !identical(object@method, "none")))
      msg <- c(msg, "an assembled object has method none and no fitted or predictive results")
    if (object@kind == "prediction" && !length(object@prediction))
      msg <- c(msg, "a prediction needs a prediction list")
    for (nm in c("results", "prediction", "smc", "mcmc")) {
      z <- methods::slot(object, nm)
      if (isS4(z) || (length(z) &&
          (is.null(names(z)) || anyNA(names(z)) || any(!nzchar(names(z))) || anyDuplicated(names(z)))))
        msg <- c(msg, paste(nm, "must be an ordinary list with unique nonempty names"))
    }
    if (length(object@prediction))
      msg <- c(msg, .restree_validate_prediction(object@prediction, ncol(object@x)))
    if (length(object@structure)) {
      st <- object@structure
      if (!all(c("nodes", "w") %in% names(st)))
        msg <- c(msg, "structure must carry nodes and w")
      else
        msg <- c(msg, .restree_validate_nodes(st,
          if (object@leaf_model == "WhiteNoise") .restree_stored_depth(st, object@depth) else object@depth,
          ncol(object@x)))
    }
    if (length(msg)) msg else TRUE
  })

## list-like access: slots first, then the results list
#' @rdname restree-class
#' @param x a \code{restree} object.
#' @param name field name for \code{$} access: a slot, results entry,
#'   read-only fit/MCMC alias, or prediction entry, in that order.
#' @export
setMethod("$", "restree", function(x, name) {
  if (name %in% c(.restree_model_slots, .restree_fit_slots)) methods::slot(x, name)
  else if (name %in% names(x@results)) x@results[[name]]
  else if (name == "fit" && length(x@mcmc)) c(x@mcmc, list(fit = x@smc))
  else if (name == "fit" && length(x@smc)) x@smc
  else if (name %in% names(x@mcmc)) x@mcmc[[name]]
  else x@prediction[[name]]
})

## No custom replacement methods: `$` and `[[` are read-only conveniences.

#' @rdname restree-class
#' @param i a single field name for \code{[[}.
#' @param j unused; only one field name is supported.
#' @param exact must be TRUE: partial field matching is not supported.
#' @export
setMethod("[[", "restree", function(x, i, j, ..., exact = TRUE) {
  if (!missing(j) || length(list(...)) || !identical(exact, TRUE) ||
      !is.character(i) || length(i) != 1L || is.na(i))
    stop("use one exact field name with [[", call. = FALSE)
  do.call("$", list(x, i))
})

#' @rdname restree-class
#' @export
setMethod("names", "restree", function(x)
  unique(c(.restree_model_slots, .restree_fit_slots, names(x@results),
    if (length(x@smc) || length(x@mcmc)) "fit", names(x@mcmc), names(x@prediction))))

## Formal constructor: a validated fit from its model, theta, method,
## structures, and the list of method-specific results.  new() runs the
## validity method, so this IS the formal validation point.
.restree_new <- function(model, theta, method, structure, results) {
  results <- results[!vapply(results, is.null, logical(1))]
  smc <- mcmc <- list()
  if (method == "pmmh") {
    mcmc <- results$fit; smc <- mcmc$fit; mcmc$fit <- NULL
    mcmc$prediction_index <- results$prediction_index
    results$fit <- NULL
    # Canonical histories live only in mcmc; dollar aliases preserve read access.
    for (nm in intersect(names(results), names(mcmc)))
      if (identical(results[[nm]], mcmc[[nm]])) results[[nm]] <- NULL
  } else if (identical(results$fit$engine, "smc")) {
    smc <- results$fit; results$fit <- NULL
  }
  obj <- restree(model, theta, method, if (is.null(structure)) list() else structure,
                 results, kind = "fit", smc = smc, mcmc = mcmc)
  ## the fitted-state record (see .restree_fitted_state): computed on the
  ## finished object so it covers exactly what determines its predictive
  obj@results$.fitted_state <- .restree_fitted_state(obj)
  obj
}

## The model specification carried by a fit (or a model), as a restree_model.
.restree_model_of <- function(object) {
  if (methods::is(object, "restree")) methods::as(object, "restree_model") else object
}

#' @rdname restree-class
#' @param object a \code{restree} object.
#' @export
setMethod("show", "restree", function(object) {
  kind <- .restree_kind(object)
  if (kind == "fit") .restree_show_fit(object)
  else if (kind == "prediction") .restree_show_prediction(object@prediction)
  else {
    cat("restree: assembled model and theta (not fitted)\n")
    show(.restree_model_of(object)); show(object@theta)
  }
  invisible(object)
})

#' @rdname restree-class
#' @param ... passed through: summary/trace options to
#'   \code{restree_diagnostics}, or \code{ynew}/\code{ncores}/\code{keep_history}
#'   to prediction.
#' @return \code{summary} returns an ordinary diagnostics list for a fit,
#'   or a compact object/prediction summary. \code{plot} draws and returns a ggplot invisibly, which can be
#'   customized with additional layers; \code{predict} dispatches to
#'   \code{\link{restree_predict}}. Numerical workflows do not need ggplot2.
#' @export
setMethod("summary", "restree", function(object, ...) {
  kind <- .restree_kind(object)
  if (kind == "fit") return(restree_diagnostics(object, ...))
  if (length(list(...))) stop("additional summary arguments apply only to fits", call. = FALSE)
  if (kind == "prediction") {
    p <- object@prediction
    return(list(kind = kind, n = length(p$mean),
      components = if (is.null(p$par_mean)) 1L else ncol(p$par_mean),
      mean_sd = mean(sqrt(p$var)), joint_lpd = p$joint_lpd))
  }
  list(kind = kind, n = nrow(object@x), d = ncol(object@x),
       leaf_model = object@leaf_model, depth = object@depth, theta = object@theta)
})

#' @rdname restree-class
#' @param xnew prediction inputs (for \code{predict}).
#' @importFrom stats predict
#' @export
setMethod("predict", "restree", function(object, xnew, ...)
  restree_predict(object, xnew, ...))

## The plot method and its helpers live in plotfuns.R.
