##########################################################################
## restree_predict(fit, xnew): prediction from a fitted restree object.
##
## Routes, in order of preference:
##   * PMMH: plug-in at posterior medians by default, using the final
##     conditional SMC trees below; or compiled MCMC averaging from saved
##     accepted final SMC ensembles, then equally weighted iterations.
##   * smc / ebayes fits with a live native prediction state (owned model and
##     compact trees in fit$fit$state): shared-path prediction, no sampler.
##   * smc / ebayes fits restored from disk: prediction state is REBUILT
##     from the stored distinct node records (no SMC rerun)
##     and prediction runs through the same native recursion; the rebuilt
##     state is cached in the fit's smc$rebuilt environment (keyed by a
##     fingerprint of theta, data and structure), so later predictions from
##     the same reloaded fit skip the rebuild.
##   * mle (Full) fits: stored-structure prediction through
##     predict_structures_cpp (no rescoring: the one stored tree is routed
##     and predicted from).
## Every route returns a `restree` containing a plain prediction list (mean / var / sd,
## per-tree moments, weights, and the joint log predictive density when
## joint_lpd = TRUE and ynew is supplied).
##########################################################################

#' Predict from a fitted residual tree
#'
#' The one prediction interface of the package.  Fits made with
#' \code{xnew} carry their prediction already (\code{fit$prediction}); this
#' function predicts afterwards, at any inputs, from the fitted object alone
#' -- also after it has been saved to disk and reloaded, in which case the
#' prediction state is rebuilt from the stored tree structures without
#' refitting.
#'
#' @param object a fitted \code{\link[=restree-class]{restree}} object from
#'   \code{\link{restree_fit}}.
#' @param xnew matrix of prediction inputs (unit hypercube, same number of
#'   columns as the training inputs).
#' @param ynew optional held-out responses at \code{xnew}, retained for scoring.
#' Joint density is disabled by default; \code{joint_lpd = TRUE} in
#' \code{...} enables \eqn{\log p(y^* \mid y)} and requires \code{ynew}.
#' Full and EB fits use
#'   plug-in covariance estimates. PMMH also defaults to plug-in prediction,
#'   at coordinatewise posterior medians (or means if selected during fitting).
#'   Parameter uncertainty is omitted, but tree uncertainty conditional on
#'   that estimate is retained. With \code{prediction = "mcmc"}, every
#'   post-burn-in accepted parameter/SMC state contributes equally, including
#' repeated states after rejection. Within each iteration, predictions use all
#' distinct final SMC trees weighted by their aggregated particle weights. Both modes
#'   use saved trees without rerunning SMC. Depth-zero WN MLE fits use
#'   independent Gaussian predictions at the fitted mean and variance.
#'   Mixture densities exclude zero-weight components exactly and retain
#'   positive subnormal weights on the log scale. A zero represented density
#'   returns \code{-Inf}, including when every supported component does so.
#'   In the current Full predictor, several test points routed to the same
#'   empty training branch are scored as independent Gaussian residuals.
#'   Their means and marginal variances are retained, but \code{joint_lpd}
#'   omits their within-branch residual correlation; it is not the exact
#'   joint density of the Full covariance in this case.
#' @param ncores positive thread request, with the allocation-bounded policy
#'   described for \code{\link{restree_fit}}. Without OpenMP prediction is serial.
#' @param keep_history keep the fit's \code{smc} and \code{mcmc} histories,
#'   stored trees and native state in the returned object (the default,
#'   \code{TRUE}); \code{FALSE} returns a lightweight prediction (see Value).
#' @param ... \code{joint_lpd = FALSE} (the default) skips joint-density evaluation
#' for all fit types, even when \code{ynew} is supplied. Set it to \code{TRUE}
#' to compute the joint density. For PMMH fits, optional \code{prediction = "plugin"} or
#'   \code{"mcmc"} overrides the fitting-time \code{control$prediction}.
#'   Without an override that saved choice is used. The point estimate is
#'   chosen during fitting via \code{control$point_estimate}; changing it
#'   requires new conditional trees, so it cannot be changed here.
#'   Other entries are rejected. The S4 \code{predict} method forwards these.
#'
#' @return An S4 \code{\link{restree}} with \code{kind = "prediction"} and an
#'   ordinary \code{prediction} list. It contains \code{xnew} and the mixture
#'   \code{mean}, \code{var} and \code{sd}, the per-tree moments
#'   \code{par_mean} and \code{par_var}, optional \code{par_df}
#'   (Student-t degrees of freedom; \code{Inf} denotes Gaussian components), weights \code{w}, and
#'   \code{joint_lpd} (\code{NA} unless joint scoring is explicitly enabled). Values can be read as
#'   \code{pred$prediction$mean} or \code{pred$mean}; held-out responses are
#'   \code{pred$prediction$y}. The object retains model/theta context,
#'   \code{smc}/\code{mcmc} histories and native state for diagnostics and
#'   tree inspection; with \code{keep_history = FALSE} the object omits the
#'   \code{smc}/\code{mcmc} histories, posterior trees and native prediction state
#'   (\code{restree_trees} / \code{restree_diagnostics} then do not apply
#'   to it), reducing saved history -- under PMMH only the
#'   \code{mcmc$prediction} cache described below is kept.
#'   Use the original fit for further prediction; it is unchanged.
#'   Under PMMH in MCMC mode, \code{pred$mcmc$prediction} contains \code{index} (retained
#'   chain iterations), \code{theta} (their parameter rows), \code{mean}/\code{var}
#'   matrices (prediction rows by selected iteration), \code{joint_lpd} per
#'   selected iteration, \code{component_iteration} mapping the pooled tree
#'   components to chain iterations, and the prediction inputs/responses.
#'   In MCMC mode every post-burn-in iteration is predicted, including
#'   rejection repeats; \code{control$predict_draws} no longer thins this
#'   average. With \code{control$prediction = "mcmc"} and \code{xnew}
#'   supplied to fitting, these values are computed in the compiled loop
#'   immediately after each retained iteration. Otherwise
#'   this function computes them from \code{mcmc$trees}, a compact serialized
#'   ensemble/weight record with shared entries for repeated accepted states.
#' Older PMMH fits with only one sampled tree per iteration, no iteration
#' ensembles, or only a subset of iterations must be refitted for MCMC averaging.
#'   Plug-in prediction can use their stored point estimate
#'   and its conditional trees. The prediction list records
#'   \code{parameter_method}, and \code{point_estimate} in plug-in mode.
#'   Components retain their Gaussian or Student-t distributions, not a
#' moment-matched approximation. With \eqn{S} retained iterations and
#' \eqn{K_s} distinct trees in state \eqn{s}, component storage grows as
#' \eqn{n_{\rm new}\sum_s K_s}; iteration summaries require
#' \eqn{n_{\rm new}S}. Weights are normally normalized; if dividing by
#' \eqn{S} would erase positive subnormal support, proportional weights are
#' returned instead. Scoring normalizes them on the log scale.
#' No per-iteration native engines are retained. Serialization saves the
#'   training data and model/theta context even with \code{keep_history = FALSE}.
#'   That option removes fitted history, not training inputs/responses or the
#'   Full model's fixed geometry; the result is not a lightweight vector.
#' @section Fitted-object integrity:
#' Reuse an intact fit. Changing its theta, data, model settings, method, stored
#' tree distribution or sampler settings does not refit its tree distribution, so
#' such an object is refused: every fit carries a byte-exact record of those
#' quantities (\code{results$.fitted_state}) and prediction stops with a
#' \code{refit} error when the record no longer matches (or is absent, as in a fit
#' saved by an older version). Build a new model and refit instead.
#' For SMC, EB and PMMH plug-in prediction, a fit restored from disk
#' rebuilds its native prediction state on the first
#' prediction and caches it inside the fit (a reference shared by every copy
#' of the object); the cache is keyed by a fingerprint of the fit's theta,
#' data and stored structures, which is a guard against such edits, not a
#' guarantee. MCMC prediction reconstructs the compact iteration trees
#' per call and uses one temporary native model context.
#' @seealso \code{\link{restree_score}}
#' @examples
#' set.seed(2)
#' n <- 300; X <- matrix(runif(2 * n), n, 2)
#' y <- sin(4 * pi * X[, 1]) + rnorm(n, sd = 0.3)
#' th <- restree_theta(sig2 = 1, range = 0.2, nu = 2.5, nugget = 0.1)
#' fit <- restree_fit(restree_model(X, y, depth = 3, r = 15, leaf_model = "PP"),
#'                    theta = th, method = "smc", nparticles = 40, seed = 2)
#'
#' ## predictive mean and standard deviation at new inputs
#' Xt <- matrix(runif(100), 50, 2)
#' pred <- restree_predict(fit, xnew = Xt)
#'
#' ## joint density is opt-in; held-out responses alone only store scoring data
#' yt <- sin(4 * pi * Xt[, 1]) + rnorm(50, sd = 0.3)
#' scored <- restree_predict(fit, xnew = Xt, ynew = yt, joint_lpd = TRUE)
#' scored$joint_lpd
#'
#' ## the predict generic dispatches to the same function
#' all.equal(predict(fit, Xt)$mean, pred$mean)
#' @export
restree_predict <- function(object, xnew, ynew = NULL, ncores = 1L, keep_history = TRUE, ...) {
  checked <- .restree_check_fit(object)
  nd <- .restree_xnew(xnew, ynew, checked$p, allow_null = FALSE)
  Xn <- nd$x; yn <- nd$y; th <- checked$theta
  ncores <- .restree_ncores(ncores)
  keep_history <- .restree_flag(keep_history, "keep_history")
  dots <- list(...)
  if (length(dots) && (is.null(names(dots)) ||
      any(!names(dots) %in% c("prediction", "joint_lpd")) || anyDuplicated(names(dots))))
    stop("only prediction and joint_lpd are supported in ...", call. = FALSE)
  joint <- if (is.null(dots$joint_lpd)) FALSE else
    .restree_flag(dots$joint_lpd, "joint_lpd")
  if (joint && is.null(yn))
    stop("joint_lpd = TRUE requires ynew", call. = FALSE)
  if (!joint) yn <- NULL
  mode <- if (is.null(dots$prediction)) {
    if (is.null(object@mcmc$prediction_method)) "mcmc" else object@mcmc$prediction_method
  } else .restree_choice(dots$prediction, c("plugin", "mcmc"), "prediction")
  if (!is.null(dots$prediction) && object@method != "pmmh")
    stop("prediction mode is only applicable to PMMH fits", call. = FALSE)
  model <- .restree_model_of(object)
  assemble <- function(ans) {
    if (!is.null(ans$mcmc_prediction)) ans$mcmc_prediction$y <- nd$y
    .restree_prediction_object(object,
    .restree_prediction(ans, nd$y, Xn,
      if (object@method == "pmmh") mode else NULL, object@mcmc$point_estimate),
    ans$mcmc_prediction, keep_history)
  }
  if (object@depth == 0L && object@leaf_model == "WhiteNoise" &&
      object@method == "ebayes" && identical(object$leaf_mle$family,"gaussian_wn"))
    return(assemble(.restree_predict_wn_mle(object$leaf_mle,Xn,yn)))
  if (object@method == "pmmh" && mode == "mcmc") {
    ans <- .restree_predict_pmmh(object, model, Xn, yn, th, ncores)
    return(assemble(ans))
  }
  ## Fast path: the fit retains independent native prediction state as an external
  ## pointer, so prediction (and the joint LPD) go straight into the native
  ## trie-shared recursion -- no rebuild, no data copies through R.  A fit
  ## restored from disk has a stale pointer: predict_state_cpp returns an
  ## empty list for that (and only that), and the exact rebuild below covers
  ## it.  Any other condition is a real error and propagates.
  st <- object@smc$state
  if (!is.null(st) && object@method %in% c("smc", "ebayes", "pmmh") &&
      !is.null(object@smc$distinct_rep) && !is.null(object@smc$w_distinct)) {
    keep <- object@smc$w_distinct > 0
    ans <- predict_state_cpp(st, Xn,
      as.integer(object@smc$distinct_rep[keep]),
      as.numeric(object@smc$w_distinct[keep]),
      th@sig2, yn, ncores)
    if (length(ans)) return(assemble(ans))
  }
  ## A fit restored from disk has a stale pointer, but the stored distinct
  ## structures are everything the native prediction needs: rebuild the
  ## compact state directly from its reached-node records (no SMC rerun or
  ## kernel work), then predict through the same shared-path
  ## recursion.  This path is AUTHORITATIVE for smc/ebayes fits (always
  ## PP/WhiteNoise): errors here are real errors.
  st_ <- object@structure
  if (is.null(st_$nodes) || is.null(st_$w))
    stop("the fit carries no stored tree structures to predict from", call. = FALSE)
  if (object@method %in% c("smc", "ebayes", "pmmh")) {
    ## The rebuilt state is cached in the fit's smc$rebuilt environment
    ## (created by .restree_smc_fit; NULL or not an environment in an older
    ## object, which then simply rebuilds every time), so repeated
    ## predictions from a fit restored from disk rebuild once. The
    ## environment is a reference: the cache is visible through every copy of
    ## the fit object, and it survives serialization (with, at most, a stale
    ## pointer that predict_state_cpp reports as an empty result).  The
    ## fingerprint is a cache key against edits of a shared object -- theta,
    ## data, structure -- not a security check.  Sums are taken in double so
    ## that a large table cannot overflow an integer sum.
    env <- object@smc$rebuilt
    cache <- is.environment(env)
    fp <- if (cache) list(theta = th, n = length(object@y), sx = sum(object@x), sy = sum(object@y),
                          r = object@r, depth = object@depth, leaf_model = object@leaf_model,
                          design = object@design, cut_method = object@cut_method,
                          nested_factor = object@nested_factor,
                          K = length(st_$w), sw = sum(st_$w), nn = nrow(st_$nodes),
                          sid = sum(as.numeric(st_$nodes$id)),
                          sJ = sum(as.numeric(st_$nodes$J), na.rm = TRUE),
                          scut = sum(st_$nodes$cut, na.rm = TRUE),
                          ssplit = sum(as.numeric(st_$nodes$split))) else NULL
    if (cache && identical(env$fingerprint, fp) && !is.null(env$state) && !is.null(env$K)) {
      ans <- predict_state_cpp(env$state, Xn, seq_len(env$K), as.numeric(st_$w),
                               th@sig2, yn, ncores)
      if (length(ans)) return(assemble(ans))
    }
    ## PP and WN restore the prediction state directly from the node table
    ## (rho is not read by prediction); no 2^depth x K matrix is formed
    K <- length(st_$w)
    st2 <- rebuild_nodes_cpp(model, th, st_$nodes, K)
    ans <- predict_state_cpp(st2, Xn, seq_len(K), as.numeric(st_$w),
                             th@sig2, yn, ncores)
    if (cache) {
      assign("state", st2, envir = env)
      assign("fingerprint", fp, envir = env)
      assign("K", K, envir = env)
    }
    return(assemble(ans))
  }
  ## Full uses the same sparse restoration path, without depth-sized matrices.
  h <- rebuild_nodes_cpp(model, th, st_$nodes, length(st_$w))
  on.exit(smc_engine_release_cpp(h), add = TRUE)
  ans <- predict_state_cpp(h, Xn, seq_along(st_$w), as.numeric(st_$w),
                           th@sig2, yn, ncores)
  # Preserve Full's existing per-tree missing-density placeholder when ynew
  # is omitted; the generic sparse predictor omits that field in this case.
  if (is.null(yn)) ans$lpd_particle <- rep(NA_real_, length(st_$w))
  assemble(ans)
}

## PMMH prediction uses saved weighted final SMC ensembles in C++, then
## equal MCMC iteration weights. Joint LPD uses nested log mixtures.

## Internal log-mixture, with exact zero support and logs taken before weight
## normalization (which can underflow for positive subnormals). Rows are test
## outcomes/draw groups; columns are components. All -Inf remains -Inf.
.restree_log_mixture <- function(log_density, w) {
  log_density <- as.matrix(log_density)
  if (!is.numeric(w) || length(w) != ncol(log_density) || !length(w) ||
      any(!is.finite(w) | w < 0) || !any(w > 0))
    stop("weights must match the components and have finite nonnegative positive total mass", call. = FALSE)
  keep <- w > 0
  L <- log_density[, keep, drop = FALSE]
  if (anyNA(L)) stop("a positive-weight component has an undefined predictive density", call. = FALSE)
  total <- sum(w)
  log_total <- if (is.finite(total)) log(total) else log(max(w)) + log(sum(w/max(w)))
  L <- sweep(L, 2L, log(w[keep]) - log_total, "+")
  vapply(seq_len(nrow(L)), function(i) {
    hi <- max(L[i, ])
    if (is.infinite(hi)) hi else hi + log(sum(exp(L[i, ] - hi)))
  }, 0)
}

.restree_predict_pmmh <- function(object, model, Xn, yn, th, ncores) {
  history <- object@mcmc$trees
  if (is.null(history))
    stop("this PMMH fit has no retained iteration trees; refit with the current package",
         call. = FALSE)
  if (!identical(as.integer(history$index),
                 seq.int(object$burnin + 1L, nrow(object$chain))))
    stop("MCMC prediction requires every post-burn-in joint state; this fit saved a subset -- refit",
         call. = FALSE)
  ans <- pmmh_predict_cpp(model, th, object$chain, history, Xn, yn, ncores)
  ans$mcmc_prediction$theta <- object$chain[history$index, , drop = FALSE]
  ans
}

## Assemble an ordinary prediction list from compiled values (no new S4 class).
.restree_prediction <- function(fit, ynew = NULL, xnew = NULL,
                                prediction_method = NULL, point_estimate = NULL) {
  pm <- fit$par_mean; pv <- fit$par_var; w <- fit$w
  ## SMC fits carry per-DISTINCT-tree moments with aggregated weights
  if (!is.null(pm) && !is.null(fit$w_distinct) &&
      length(fit$w_distinct) == ncol(pm)) w <- fit$w_distinct
  if (is.null(pm)) { pm <- matrix(fit$mean, ncol = 1L); pv <- matrix(fit$var, ncol = 1L); w <- 1 }
  out <- list(mean = as.numeric(fit$mean), var = as.numeric(fit$var),
                 sd = sqrt(pmax(as.numeric(fit$var), 0)),
                 par_mean = pm, par_var = pv, par_df = fit$par_df, w = as.numeric(w),
                 joint_lpd = if (is.null(fit$lpd)) NA_real_ else fit$lpd,
                 lpd_particle = fit$lpd_particle,
                 y = ynew, xnew = xnew)
  if (!is.null(prediction_method)) {
    out$parameter_method <- prediction_method
    if (prediction_method == "plugin")
      out$point_estimate <- if (is.null(point_estimate)) "mean" else point_estimate
  }
  out
}

.restree_show_prediction <- function(x, ...) {
  cat("restree predictive distribution at", length(x$mean), "points\n")
  cat("  components:", if (is.null(x$par_mean)) 1L else ncol(x$par_mean), "tree(s)\n")
  cat("  mean sd   :", format(mean(sqrt(pmax(x$var, 0))), digits = 4), "\n")
  if (length(x$joint_lpd) == 1L && !is.na(x$joint_lpd))
    cat("  joint lpd :", format(x$joint_lpd, digits = 8), "\n")
  invisible(x)
}
