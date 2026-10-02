##########################################################################
## Sampler diagnostics: one entry point, dispatching on how the fit was made.
##
## The three samplers fail in different ways and need different numbers.
##   SMC      -- weight degeneracy. The effective sample size across the sweep, how
##               often resampling fired, the largest normalised weight, and how many
##               DISTINCT tree structures survive. A high final ESS with few distinct
##               trees means the particles are copies, which ESS alone will not show.
##   PMMH     -- two things at once: the theta chain must mix, and the SMC evidence
##               estimate it consumes must not be too noisy. sd(log Zhat) is the
##               quantity to tune on; Doucet, Pitt, Deligiannidis and Kohn (2015,
##               Biometrika 102:295-313) show the optimal setting is near 1.0-1.7,
##               and that far outside that range the chain sticks.
##
## Chain ESS uses Geyer's initial positive sequence estimator (1992, Statist Sci
## 7:473-483), so nothing outside base R is needed.
##########################################################################

## Geyer (1992) initial positive sequence estimate of the integrated
## autocorrelation time: Gamma_m = rho_{2m} + rho_{2m+1} (m = 0, 1, ...; rho_0 = 1)
## summed up to the last m before the first nonpositive Gamma, tau = -1 +
## 2 sum_m Gamma_m, ESS = n / tau (capped at n).
.restree_ess_geyer <- function(x) {
  x <- as.numeric(x); n <- length(x)
  if (n < 8L || !is.finite(stats::sd(x)) || stats::sd(x) == 0) return(NA_real_)
  rho <- stats::acf(x, lag.max = min(n - 1L, 2000L), plot = FALSE, demean = TRUE)$acf[, 1L, 1L]
  m_max <- floor((length(rho) - 2L) / 2)                  # pairs (2m, 2m+1) within the lags
  if (m_max < 0L) return(NA_real_)
  m <- 0L:m_max
  G <- rho[2L * m + 1L] + rho[2L * m + 2L]                # rho[k + 1] is rho_k
  k <- which(G <= 0)[1L]
  k <- if (is.na(k)) length(G) else k - 1L                # initial positive sequence
  if (k < 1L) return(as.numeric(n))
  tau <- -1 + 2 * sum(G[seq_len(k)])
  min(n, n / max(tau, .Machine$double.eps))
}

#' Fitting diagnostics and sampler traces
#'
#' \code{type = "summary"} reports fitting quality: particle degeneracy and
#' diversity for SMC; optimizer convergence and evidence gain for EB/Full;
#' parameter-chain ESS, acceptance and evidence-estimator noise for PMMH.
#' Recommendations are tuning heuristics, not convergence guarantees.
#'
#' \code{type = "trace"} returns detailed histories. For an SMC/EB fit it
#' extracts stored level-end diagnostics; for PMMH it extracts the accepted
#' chain, proposals, acceptance indicators, evidence traces and replicate
#' evidences. For a PP/WN model plus known theta it runs \code{nrep} new SMC
#' evaluations; their variability measures log-evidence-estimator noise at
#' that fixed theta. This model route is computation, not just extraction.
#'
#' @param object a fitted \code{\link{restree}} or its prediction object for summaries or stored
#'   traces; a PP/WN \code{\link{restree_model}} or assembled \code{restree}
#'   for new known-theta trace runs.
#' @param burnin PMMH iterations to discard; defaults to the fitted setting.
#' @param type \code{"summary"} (default) or \code{"trace"}.
#' @param theta covariance parameters for trace runs from a model. An
#'   assembled \code{restree} uses its theta when this argument is omitted.
#' @param nrep number of independent known-theta SMC runs.
#' @param nparticles,resampling,temper_alpha,seed,ncores sampler settings for
#'   known-theta trace runs, as in \code{\link{restree_loglik}} (\code{ncores
#'   = 1}, the default; see \code{\link{restree_fit}}). They do
#'   not change a trace extracted from an existing fit.
#' @return An ordinary list. Summary fields depend on the fitting method and
#'   include diagnostic values and explanatory \code{notes}.
#'   SMC trace lists contain \code{kind = "smc"}, \code{levels} (per run and
#'   level), \code{runs} (log evidence, final ESS, diversity, resampling,
#'   seed and timing), \code{sd_logZ} and sampler settings.
#'   Trace extraction from a newly fitted SMC/EB object additionally returns
#'   \code{steps}: step 0 is initialization and each subsequent step processes
#'   one active node position across particles. Columns are \code{step},
#'   \code{node_id} (binary-tree position), \code{level}, \code{ess}
#'   (after the node update, before resampling), \code{resampled} and
#'   \code{ess_after}, and cumulative \code{logZ} after the node update
#'   and any resampling correction. At step 0, \code{logZ} equals the
#'   root-leaf value \code{Phi0}. Inactive PP positions do not increment the step.
#'   The \code{levels$last_step} column places level-end resampling on this
#'   same axis, including any resampling at an empty PP level. Older saved fits
#'   have an empty \code{steps} table. The separate \code{ESS} vector is
#'   indexed by node position, not by processing step.
#'   Model runs additionally record root-split coverage in \code{runs}
#'   and include \code{evidence}: log mean evidence
#'   (\code{log_mean_Z}), evidence-scale relative Monte Carlo standard error
#'   (\code{relative_mcse}), and \code{replicate_ess}. These are empirical
#'   diagnostics, not proof of coverage; compare independent seeds and budgets.
#'   PMMH trace lists contain \code{kind = "pmmh"}, \code{chain},
#'   \code{chain_prop}, \code{logZ_trace}, \code{logZ_prop}, \code{accepted},
#'   \code{burnin}, \code{accept_rate}, \code{logZ_reps} and \code{sd_logZ}.
#'   Chains are on the natural parameter scale.
#' @section Plotting:
#' Use \code{plot(fit, type = "trace")} or
#' \code{plot(model, theta = th, type = "trace")} to request ggplots.
#' The latter performs new SMC runs. Numerical diagnostics do not require
#' ggplot2. Trace lists have no custom print or plot dispatch.
#' @section Interpretation:
#' SMC ESS is interpreted at level ends, where resampling is decided.
#' A high ESS alone does not establish diversity of distinct trees or coverage
#' of unvisited posterior modes. WN integrates cuts only one step ahead:
#' descendants may gain likelihood by splitting later. Small particle runs can
#' miss this mass. Compare evidence-scale replicate means, root-split coverage,
#' and error per second across particle budgets; averaging log evidence is
#' generally downward biased even when evidence itself is unbiased.
#' PMMH evidence-trace variation includes changing theta; it is not the
#' fixed-theta estimator noise measured by independent evidence replicates.
#' A root-only SMC fit has no level events; its trace has an empty
#' \code{levels} table and one run record, including WN at depth zero when
#' \code{n > r}. For positive WN depth, level records continue while eligible
#' nodes remain, so their number need not equal the initial model depth.
#' WN summaries report \code{initial_depth} and \code{tree_depth} separately.
#' Root-only models have no new trace runs; Full has no SMC trace.
#' @references Geyer, C. J. (1992) \emph{Statistical Science} 7, 473-483.
#'   Doucet, A., Pitt, M. K., Deligiannidis, G. and Kohn, R. (2015)
#'   \emph{Biometrika} 102, 295-313.
#' @examples
#' set.seed(4)
#' X <- matrix(runif(120), 60, 2); y <- sin(4 * X[,1])
#' th <- restree_theta(range = .2, nugget = .1)
#' m <- restree_model(X, y, depth = 2, r = 6)
#' fit <- restree_fit(m, th, method = "smc", nparticles = 8)
#' restree_diagnostics(fit)
#' restree_diagnostics(fit, type = "trace")$levels
#' restree_diagnostics(m, type = "trace", theta = th, nrep = 2, nparticles = 8)$runs
#' if (requireNamespace("ggplot2", quietly = TRUE)) plot(fit, type = "trace")
#' @seealso \code{\link{restree_trees}} for actual trees and posterior
#'   summaries of their decisions.
#' @export
restree_diagnostics <- function(object, burnin = NULL, type = c("summary", "trace"),
                                theta = NULL, nrep = 10L, nparticles = NULL,
                                resampling = "stratified", temper_alpha = 1,
                                seed = 42L, ncores = 1L) {
  type <- match.arg(type)
  if (type == "trace")
    return(.restree_collect_trace(object, theta, nrep, nparticles, resampling,
                                  temper_alpha, seed, ncores, burnin))
  .restree_require_history(object)
  m <- object$method
  out <- list(method = m, leaf_model = object$leaf_model, notes = character(0))
  if (object@leaf_model == "WhiteNoise") {
    out$initial_depth <- object@depth
    out$tree_depth <- .restree_stored_depth(object@structure, object@depth)
  }

  if (m == "smc") {
    ess <- as.numeric(object$fit$ESS); N <- object$nparticles
    w <- as.numeric(object$fit$w)
    ## realized per-tree log p(y | T_k, theta): particle_loglik and
    ## particle_loglik_tree agree to round-off, so either serves
    pl <- object$fit$particle_loglik_tree
    if (is.null(pl)) pl <- object$fit$particle_loglik
    pl <- as.numeric(pl)
    ## Level-end values only: a resampling decision is made at level ends,
    ## and mid-level weights rank particles by node-processing order (a
    ## particle whose subtree was visited later in the level is valued with
    ## its children still closed as leaves), so the per-node ESS vector is
    ## kept for completeness but the summaries come from level_diag.
    ld <- object$fit$level_diag
    if (is.null(ld)) ld <- .restree_empty_level_diag()
    lev_ess <- if (nrow(ld)) ld$ess else utils::tail(ess, 1L)
    out$nparticles <- N
    out$ESS <- ess
    out$level_diag <- ld
    out$step_diag <- object$fit$step_diag
    out$ESS_final <- utils::tail(ess, 1L)
    out$ESS_min <- min(lev_ess, na.rm = TRUE)
    out$ESS_min_level <- if (nrow(ld)) ld$level[which.min(ld$ess)] else NA_integer_
    out$ESS_median <- stats::median(lev_ess, na.rm = TRUE)
    out$ESS_frac <- out$ESS_final / N
    out$n_resample <- object$fit$n_resample
    out$resampled_levels <- if (nrow(ld)) ld$level[ld$resampled == 1L] else integer(0)
    out$max_weight <- max(w)
    out$n_distinct_trees <- object$fit$n_distinct_trees
    out$distinct_frac <- out$n_distinct_trees / N
    out$logZ <- object$fit$logZ; out$Phi0 <- object$fit$Phi0
    out$particle_loglik <- c(mean = mean(pl), sd = stats::sd(pl),
                             min = min(pl), max = max(pl))
    if (out$ESS_frac < 0.5) {
      out$notes <- c(out$notes, "final ESS below half the particle count: weights are degenerate")
      ta <- object$temper_alpha
      if (is.numeric(ta) && length(ta) == 1L && is.finite(ta) && ta < 1)
        out$notes <- c(out$notes, sprintf(paste0(
          "tempered resampling at temper_alpha = %g draws from w^%g and ",
          "keeps the fraction 1 - %g of the log-weight spread in the ",
          "residual weights, so a triggered resample rebuilds little ESS; ",
          "temper_alpha = 1 gives the standard equal-weight resample"),
          ta, ta, ta))
    }
    if (out$distinct_frac < 0.5)
      out$notes <- c(out$notes,
        "fewer than half the particles carry distinct trees: ESS overstates diversity")
    if (out$n_distinct_trees == 1L)
      out$notes <- c(out$notes,
        "every particle holds the same tree; inspect whether the tree posterior is genuinely concentrated")

  } else if (m == "pmmh") {
    ch <- object$chain; ni <- nrow(ch)
    bi <- if (!is.null(burnin)) .restree_integer(burnin, "burnin", 0L, ni - 1L) else
      if (!is.null(object$burnin)) object$burnin else floor(ni / 2)
    keep <- seq.int(bi + 1L, ni)
    lz <- as.numeric(object$fit$logZ_trace)
    par_names <- colnames(ch)
    out$n_iter <- ni; out$burnin <- bi
    out$accept_rate <- object$accept_rate
    out$accept_rate_post <- object$fit$accept_rate_post
    out$adapt <- object$fit$adapt
    out$prop_sd_final <- object$fit$prop_sd_final
    ## the sigma^2 block moves (deterministic cuts on the persistent engine)
    out$sig2_moves <- object$fit$sig2_moves
    out$accept_rate_sig2 <- object$fit$accept_rate_sig2
    out$n_evidence_evals <- object$fit$n_evidence_evals
    out$summary <- data.frame(
      parameter = par_names,
      mean   = apply(ch[keep, , drop = FALSE], 2L, mean),
      median = apply(ch[keep, , drop = FALSE], 2L, stats::median),
      sd     = apply(ch[keep, , drop = FALSE], 2L, stats::sd),
      ESS    = apply(ch[keep, , drop = FALSE], 2L, function(z) .restree_ess_geyer(log(z))),
      stringsAsFactors = FALSE)
    out$summary$ESS_per_iter <- out$summary$ESS / length(keep)
    ## Likelihood-estimator noise: sd of independent log Zhat replicates at
    ## the fixed point-estimate theta (fit$logZ_reps, control$sd_reps runs
    ## after the chain).  sd of the chain's logZ_trace is a different
    ## quantity -- theta moves along the chain, so it adds the posterior
    ## variation of log Z(theta), and it records accepted values only -- and
    ## is reported separately without a recommendation attached.
    reps <- object$logZ_reps
    if (is.null(reps)) reps <- object$fit$logZ_reps
    reps <- as.numeric(reps); reps <- reps[is.finite(reps)]
    out$sd_logZ_reps <- length(reps)
    out$sd_logZ <- if (length(reps) >= 2L) stats::sd(reps) else NA_real_
    out$sd_logZ_trace <- stats::sd(lz[keep])
    if (out$accept_rate < 0.1)
      out$notes <- c(out$notes, "acceptance below 10%: reduce prop_sd, or the evidence estimate is too noisy")
    if (out$accept_rate > 0.7)
      out$notes <- c(out$notes, "acceptance above 70%: proposal steps are too small")
    if (is.finite(out$sd_logZ) && out$sd_logZ > 1.7)
      out$notes <- c(out$notes, sprintf(
        "sd(log Zhat) at the point-estimate theta = %.2f over %d replicate SMC runs, above the 1.0-1.7 range: the chain will stick; raise nparticles",
        out$sd_logZ, length(reps)))
    if (is.finite(out$sd_logZ) && out$sd_logZ < 1.0)
      out$notes <- c(out$notes, sprintf(
        "sd(log Zhat) at the point-estimate theta = %.2f over %d replicate SMC runs, below the 1.0-1.7 range: nparticles could be cut for the same mixing",
        out$sd_logZ, length(reps)))
    if (!is.finite(out$sd_logZ))
      out$notes <- c(out$notes,
        "no replicate evidence evaluations (control$sd_reps = 0): the estimator noise sd(log Zhat) is unknown; sd of the logZ trace mixes it with the posterior variation of log Z(theta) and is not a particle-count guide")
    if (any(is.finite(out$summary$ESS) & out$summary$ESS < 20))
      out$notes <- c(out$notes,
        "effective sample size under 20 for at least one parameter: run longer")

  } else if (m == "ebayes") {
    opt <- object$optimization
    wn_mle <- identical(object$leaf_mle$family, "gaussian_wn")
    out$estimator <- if (!is.null(opt$estimator)) opt$estimator else
      if (wn_mle) "mle" else "map"
    if (wn_mle)
      out$notes <- c(out$notes,"Root-only WN: Gaussian mean and variance estimated by MLE; range, nugget, and control$prior are unused.")
    else if (.restree_root_only(object))
      out$notes <- c(out$notes, if (out$estimator == "mle")
        "Single-leaf maximum likelihood; no tree sampling or resampling." else
        "Single-leaf MAP on the natural parameter scale; no tree sampling or resampling.")
    th <- as.numeric(object$theta_hat)
    names(th) <- names(object$theta_hat)
    ess <- as.numeric(object$fit$ESS)
    out$nparticles <- object$nparticles
    out$seconds <- opt$seconds
    out$n_eval <- opt$n_eval
    out$n_eval_full <- opt$n_eval_full
    out$n_eval_cheap <- opt$n_eval_cheap
    out$profiled <- isTRUE(opt$profiled)
    out$start <- opt$start
    out$convergence <- opt$convergence
    out$theta <- data.frame(parameter = names(th), estimate = th,
                            row.names = NULL, stringsAsFactors = FALSE)
    ## logZ_initial and logZ_hat are the common-random-number objective at
    ## the start and at the optimum (same SMC seed), so their difference is
    ## the gain the optimizer saw; object$logZ is the final SMC at another
    ## seed and is reported separately.
    out$logZ_initial <- opt$logZ_initial
    out$logZ_hat <- opt$logZ_hat
    out$logZ <- object$logZ
    out$sd_logZ <- if (is.null(opt$sd_logZ)) NA_real_ else opt$sd_logZ
    out$sd_logZ_reps <- length(opt$logZ_reps)
    out$logZ_gain <- if (is.null(opt$logZ_initial) || is.null(opt$logZ_hat)) NA_real_ else
      opt$logZ_hat - opt$logZ_initial
    ## MAP includes the prior; MLE uses the likelihood/evidence alone.
    out$logpost_initial <- opt$logpost_initial
    out$logpost <- opt$logpost_hat
    out$logpost_gain <- if (is.null(opt$logpost_initial) || is.null(opt$logpost_hat)) NA_real_ else
      opt$logpost_hat - opt$logpost_initial
    out$prior <- opt$prior
    if (identical(out$estimator, "mle") && !.restree_root_only(object))
      out$notes <- c(out$notes, "no prior on theta: the estimate maximises the SMC evidence alone")
    out$ESS_final <- utils::tail(ess, 1L)
    out$n_distinct_trees <- object$fit$n_distinct_trees
    if (!identical(as.integer(opt$convergence), 0L))
      out$notes <- c(out$notes,
        sprintf("the %s optimizer did not converge (code %d)",
                toupper(out$estimator), opt$convergence))
    if (is.finite(out$logpost_gain) && out$logpost_gain < -1e-8)
      out$notes <- c(out$notes,
        if (out$estimator == "mle")
          "the reported estimate has lower fixed-seed log evidence than the supplied starting theta" else
          "the reported mode has lower fixed-seed log posterior (log Z + log prior) than the supplied starting theta")

  } else if (m == "mle") {
    opt <- object$optimization
    out$convergence <- opt$convergence
    out$n_eval <- unname(opt$counts[["function"]])
    out$loglik <- object$loglik
    out$theta <- data.frame(parameter = names(object$theta_hat),
                            estimate = as.numeric(object$theta_hat), row.names = NULL)
    if (!identical(as.integer(opt$convergence), 0L))
      out$notes <- c(out$notes,
        "the fixed-tree Gaussian MLE optimizer reached maxit before convergence")

  } else {
    out$notes <- c(out$notes, sprintf(
      "method = '%s' involves no sampler; nothing to diagnose", m))
  }
  out
}

## Zero-row diagnostics also support saved fits without a level record.
.restree_empty_level_diag <- function()
  data.frame(level = integer(0), last_node = integer(0), last_step = integer(0), ess = numeric(0),
             resampled = integer(0), ess_after = numeric(0),
             max_weight = numeric(0), logZ_increment = numeric(0),
             n_paths = integer(0))
.restree_empty_step_diag <- function()
  data.frame(step = integer(0), node_id = integer(0), level = integer(0),
             ess = numeric(0), resampled = integer(0), ess_after = numeric(0),
             logZ = numeric(0))
## Evidence-scale replicate summaries: scaling avoids exp(logZ) overflow.
## An observed MCSE cannot detect posterior regions missed by every replicate.
.restree_evidence_replicates <- function(logZ) {
  hi <- max(logZ)
  scaled <- exp(logZ - hi)
  n <- length(scaled); avg <- mean(scaled)
  list(log_mean_Z = hi + log(avg),
       relative_mcse = if (n > 1L) stats::sd(scaled) / (sqrt(n) * avg) else NA_real_,
       replicate_ess = sum(scaled)^2 / sum(scaled^2), nrep = n)
}
