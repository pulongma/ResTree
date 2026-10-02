##########################################################################
## restree_fit(model, ...): estimation for a restree_model.
##
## The compiled entry points differ in what they estimate, not in what they
## are for, so they are gathered behind one call with a `method` argument
## (PP / WN leaf models; a Full model has one fixed tree and always fits by
## Gaussian maximum likelihood -- its fit carries method = "mle"):
##
##   method = "smc"     SMC over PP/WN trees at fixed theta (theta required).
##   method = "ebayes"  joint empirical Bayes for (sigma^2, range, nugget):
##                      CRN Nelder-Mead over the conditional SMC evidence.
##   method = "pmmh"    particle-marginal MH for (sigma^2, theta) via .restree_pmmh().
##
## SMC defaults (one source, matched by src/SMC.cpp configure_smc and
## R/RcppExports.R): resampling "stratified", temper_alpha 1 -- the plain
## evidence estimator, so restree_fit()$logZ and restree_loglik() at the
## same arguments are the same estimator.
##
## Two prediction routes, both supported on purpose:
##   restree_fit(..., xnew = Xnew)   one pass; the fitted engine is live,
##                                      so nothing is refitted. Cheapest.
##   restree_predict(fit, Xnew)         deferred; see restree_predict.R.
##########################################################################

#' Fit a residual-tree Gaussian process
#'
#' Estimates the covariance parameters (and, for the PP and WN leaf models,
#' the tree) of the model specified by a \code{\link{restree_model}}.
#' \code{leaf_model = Full} uses one fixed tree with the complete residual
#' covariance in every leaf: its ranges and nugget are optimized by Gaussian
#' maximum likelihood (bounded multi-start L-BFGS-B by default),
#' \eqn{\sigma^2} is profiled analytically, and prediction plugs in those
#' estimates; \code{method} is ignored for it and the fit carries
#' \code{method = "mle"}.  The tree itself is part of the model (\code{model$tree}, built
#' once by \code{\link{restree_model}}). An internal workspace owns the data
#' and prepares the native geometry once per fit, reusing it across all
#' optimizer evaluations and starting points. Covariance-dependent kernels,
#' factorizations and residual propagation are recomputed for each theta.
#' The workspace is not stored in the fitted object. Full extends the requested depth
#' until every terminal holds at most \code{r} remaining observations.
#' \code{PP} and \code{WN} infer tree uncertainty with sequential
#' Monte Carlo and estimate the shared covariance parameters either by
#' empirical Bayes over the tree-integrated evidence (\code{ebayes}) or by
#' particle-marginal Metropolis--Hastings (\code{pmmh}); \code{smc} runs the
#' tree sampler at the supplied \code{theta} without estimating anything.
#' With positive initial depth, the WN sampler compares stop versus split at
#' every nonempty node, even beyond the model's initial \code{depth} and when
#' the node contains at most \code{r} observations. This rule
#' is shared by SMC, EB and PMMH; see \code{\link{restree_model}}.
#' The original model depth is retained, while \code{fit$structure$depth}
#' records the WN representation depth needed by the saved trees.
#'
#' Joint predictive density is disabled by default, even with \code{ynew}.
#' Enable it with \code{control = list(joint_lpd = TRUE)}; this requires
#' \code{ynew}. Held-out responses are still stored for marginal scoring.
#' The predictive distribution follows the estimator. Full MLE and EB use
#' fitted covariance parameters, including a plug-in \eqn{\sigma^2}. PMMH
#' defaults to \code{control$prediction = "plugin"}: fix each covariance
#' parameter at its posterior median on the natural scale after burn-in
#' (\code{control$point_estimate = "median"}), or its posterior mean with
#' \code{"mean"}. This omits covariance-parameter uncertainty, while still
#' averaging over the final SMC trees conditional on that fixed estimate.
#' With \code{control$prediction = "mcmc"}, average over every retained joint
#' parameter/SMC state, including repeated states after rejection. At each
#' iteration, predict from every distinct tree in the accepted final SMC
#' ensemble, using its aggregated final particle weights. Then average these
#' mixtures equally across iterations; rejected proposals never supply trees.
#' With \code{xnew}, MCMC predictions run inside the compiled chain; otherwise
#' compact saved trees support later prediction without rerunning SMC.
#' The deprecated \code{control$predict_draws} is ignored with a warning.
#' The WN
#' leaf's local mean and variance are integrated under its Normal-Inverse-Gamma
#' model for tree-based fits. Full uses ordinary zero-mean dense GP MLE
#' when \code{n <= r}; requesting depth zero alone does not ensure this.
#' For PP or WN at depth zero there is no tree sampling, regardless of
#' \code{n} or \code{r}. PP with \code{method="ebayes"}
#' uses its terminal PP likelihood for MAP by default, or MLE with
#' \code{control = list(prior = "none")}; WN with \code{method="ebayes"}
#' estimates a Gaussian mean and variance by MLE (variance divisor n) and
#' predicts with that fitted Gaussian. Constant/singleton WN data have no
#' finite positive-variance MLE and raise an error. The Gaussian WN MLE
#' does not use \code{control$prior}. Explicit \code{smc} and
#' \code{pmmh} retain their known-parameter and Bayesian interpretations,
#' including the conjugate WN terminal rather than the WN MLE.
#'
#' A running \code{pmmh} chain can be interrupted from R (Ctrl-C, or Esc in
#' RStudio): the completed iterations are kept and returned as a valid
#' shorter chain with a warning, and the SMC engine checks for interrupts
#' between tree levels. A large individual matrix operation may delay response.
#' If fewer than two iterations completed, interruption raises an error.
#'
#' Errors during PMMH initialization, sampling, evidence replicates or the
#' final SMC raise a \code{restree_pmmh_error} condition. Its \code{chain}
#' and \code{chain_prop} matrices contain completed iterations on the named,
#' natural parameter scale, as in a successful fit (zero rows if initialization
#' failed). It also records \code{phase}, \code{n_done}, \code{n_requested},
#' \code{chain_scale = "natural"}, \code{parameter_form}, \code{burnin},
#' \code{seed}, evidence traces, acceptance and proposal-scale history.
#' These are diagnostic partial results, not an exact-resume checkpoint:
#' \code{exact_resume = FALSE}. For example, inspect them with
#' \code{tryCatch(restree_fit(m, th, method = "pmmh"),
#' restree_pmmh_error = function(e) e)}. Invalid arguments can still fail
#' before a chain is initialized and do not promise partial results.
#'
#' @inheritSection restree_model Leaf models
#' @param model a \code{\link{restree_model}}, or an assembled
#'   \code{restree(model, theta)} object. For the latter, its stored theta
#'   is used when the \code{theta} argument is omitted.
#' @param theta covariance parameters (a \code{\link{restree_theta}} or a
#'   named list): fixed for \code{method = "smc"}; starting values for the
#'   estimators (the non-estimated fields \code{nu}, \code{tail},
#'   \code{form}, \code{family}, \code{dtype} are held fixed).  \code{NULL}
#'   uses the package defaults (\code{method = "ebayes"} then starts at
#'   them with \eqn{\sigma^2 = \mathrm{var}(y)}; \code{control$start},
#'   below, can start the search at the Full-tree MLE instead).
#' @param method estimator for the PP / WN leaf models:
#'   \itemize{
#'     \item \code{"ebayes"} (the default): empirical Bayes for the shared
#'       covariance parameters, maximising the tree-integrated SMC evidence
#'       (times the prior, unless \code{control$prior = "none"}).
#'     \item \code{"pmmh"}: particle-marginal Metropolis--Hastings over the
#'       covariance parameters and the trees.
#'     \item \code{"smc"}: the PP / WN tree posterior at the supplied fixed
#'       \code{theta}; nothing is estimated.
#'   }
#'   A Full model ignores it: its one fixed tree is fitted by Gaussian
#'   maximum likelihood, and the fit carries \code{method = "mle"}.
#' @param xnew optional matrix of prediction inputs. Supplying it predicts inside
#'   the fit, which refits nothing.
#' @param ynew optional held-out responses at \code{xnew}, retained for scoring.
#' The joint log predictive density requires \code{control$joint_lpd = TRUE}.
#' @param nparticles number of SMC particles. The default \code{NULL} uses
#'   \eqn{100\,\mathrm{depth}} for both PP and WN, using the configured
#'   model depth. WN may grow deeper, but its realized tree depth does not
#'   change this budget. Root-only models use one particle, including WN
#'   at depth zero with \code{n > r}. With the automatic depth of
#'   \code{\link{restree_model}}, \eqn{n = 10^4}, \eqn{r = 60} gives depth
#'   8 and 800 particles; \eqn{n = 10^5} gives depth 11 and 1100.
#'   Explicit \code{nparticles} values override this default for positive
#'   depth. Tune it against
#'   \code{summary(fit)$sd_logZ} of a \code{pmmh} fit (the estimator noise
#'   at the point-estimate theta; about 1.0--1.7 is a tuning heuristic,
#'   not a convergence guarantee). This particle budget does not apply to Full.
#' @param resampling SMC resampling scheme:
#'   \itemize{
#'     \item \code{"stratified"} (the default): one uniform draw per stratum
#'       of the cumulative weights.
#'     \item \code{"multinomial"}: independent draws from the weights.
#'   }
#' @param temper_alpha SMC tempering exponent in \code{(0, 1]}; the default
#'   \code{1} is the plain evidence estimator.  The two defaults are those of
#'   \code{\link{restree_loglik}}, so a fit's \code{logZ} and
#'   \code{restree_loglik()} at the same arguments are the same estimator.
#' @param n_iter iterations for \code{pmmh}. The default 250 is a
#'   smoke-test length: with the fixed random-walk proposal
#'   (\code{control$prop_sd}, 0.15 by default) a few hundred iterations do
#'   not converge.  Chains for reported inference typically need several
#'   thousand iterations; judge convergence by \code{summary()} (Geyer ESS
#'   per parameter) and, when feasible, split-Rhat over independent seeds.
#' @param seed random seed of the SMC (\code{smc}, \code{ebayes}, \code{pmmh}).
#'   The Full MLE is deterministic given the model (whose fixed tree carries
#'   its own \code{seed}) and does not use it.
#' @param ncores positive thread request. With \code{1}, tree traversal is
#'   serial; Eigen may parallelize large matrix products up to the available
#'   physical-core allocation. Requests \code{k >= 2} bound this package's
#'   engine and Eigen teams by \code{k}. Both policies respect physical-core
#'   detection and \code{OMP_NUM_THREADS}, \code{SLURM_CPUS_PER_TASK}, and
#'   \code{PBS_NP}; capped requests warn. Without OpenMP all computations are
#'   serial. Linux physical cores are affinity-filtered; unavailable affinity
#'   or incomplete topology conservatively gives one core. Restart R after
#'   changing CPU affinity because topology is cached. Cache/workspace budgets
#'   may limit concurrent tasks, but no memory preflight rejects a fit and
#'   there is no total-memory guarantee. These settings do not control other
#'   packages or external BLAS threads. See \code{fit@results$thread_policy}
#'   for the original request and resolved engine/Eigen counts.
#' @param verbose show a progress bar over PMMH iterations.
#' @param control list of method-specific settings.
#'   \itemize{
#'     \item All methods: \code{joint_lpd} (default \code{FALSE}).
#' Enable joint predictive-density evaluation only with held-out \code{ynew}.
#' \item Full MLE (\code{method = "mle"}): \code{maxit} (default 100),
#'       \code{reltol} (default \code{1e-6}), \code{optimizer}
#'       (\code{"L-BFGS-B"}, the default, or \code{"Nelder-Mead"}),
#'       \code{n_starts} (default 5, the caller's start plus data-scale
#'       starts), and the search box on the natural scale --
#'       \code{range_bounds} (default \code{c(1e-4, 10)}, applied to every
#'       range) and \code{nugget_bounds} (default \code{c(1e-8, 2)}) --
#'       which bound L-BFGS-B and clamp every start.
#'     \item EB (\code{method = "ebayes"}): \code{maxit} (default 100) and
#'       \code{tol} (default \code{1e-5}) of the Nelder--Mead search;
#'       \code{start}, the starting point: \code{"theta"} (the default)
#'       begins at \code{theta} (the package defaults when \code{NULL}),
#'       \code{"full_mle"} at the Full-tree MLE, and \code{"auto"} at the
#'       Full-tree MLE when \code{theta} supplies no range and at
#'       \code{theta} otherwise -- the MLE is the fit of
#'       \code{leaf_model = "Full"} on the model's \code{depth}, \code{r},
#'       \code{design} and \code{cut_method} with three L-BFGS-B starts and
#'       is recorded in \code{optimization$start}; depth zero always starts
#'       at \code{theta}; \code{sd_reps} (default 5, 0 disables), the number
#'       of fresh-seed evidence evaluations at the estimate whose standard
#'       deviation (\code{sd_logZ}) is the estimator noise -- the search
#'       itself runs on one seed and cannot show it; and \code{prior} (below).
#'     \item PMMH (\code{method = "pmmh"}): \code{burnin} (default half the
#'       chain); \code{adapt} (\code{"burnin"}, the default, or
#'       \code{"none"}) and \code{target_accept} (default 0.15): with
#'       \code{"burnin"} one common factor on \code{prop_sd} is adapted by
#'       Robbins--Monro steps towards the target acceptance during the
#'       burn-in and frozen afterwards, so the retained draws come from a
#'       fixed kernel (\code{prop_sd_final} and \code{prop_scale_trace}
#'       record it; \code{"none"} keeps \code{prop_sd} as given);
#'       \code{prediction} (\code{"plugin"}, the default, or \code{"mcmc"});
#'       \code{point_estimate} (\code{"median"}, the default, or \code{"mean"})
#'       for the coordinatewise natural-scale parameter summary. Plug-in
#'       prediction fixes this summary; MCMC prediction averages every
#'       post-burn-in joint state, with rejection repeats included.
#'       \code{predict_draws} is deprecated and ignored with a warning;
#'       \code{prop_sd} (default 0.15), the random-walk standard
#'       deviations on the log scale, ordered \code{(sig2, range..., nugget)}
#'       (a scalar is recycled); \code{sig2_moves} (default 3), the number
#'       of random-walk moves on \eqn{\log\sigma^2} alone that precede the
#'       joint move of every iteration -- under the deterministic cut rules
#'       they re-use the engine's retained block statistics and cost a small
#'       fraction of a full evidence evaluation, since \eqn{\sigma^2} scales
#'       every Gaussian block and enters none of the linear algebra (the same
#'       fact lets \code{ebayes} nest a one-dimensional Brent search over
#'       \eqn{\log\sigma^2}, made of such \eqn{\sigma^2}-only
#'       re-evaluations at the common seed, inside its search over range and
#'       nugget under the deterministic cut rules); \code{sd_reps}
#'       (default 10, 0 disables), the number of independent SMC evaluations
#'       of \eqn{\log \hat Z} at the point-estimate theta run after the
#'       chain, each costing one PMMH iteration -- their standard deviation is
#'       the likelihood-estimator noise that governs PMMH mixing
#'       (\code{summary(fit)$sd_logZ}, target about 1.0--1.7), which the
#'       chain's own \code{logZ_trace} cannot give because theta moves along
#'       it; and \code{prior} (below).
#'     \item \code{prior} (EB and PMMH), a list with
#'       \itemize{
#'         \item \code{sig2 = c(shape, rate)}: an inverse-gamma prior on
#'           \eqn{\sigma^2} (default shape 2, rate \code{var(y)}, so the
#'           prior mean of \eqn{\sigma^2} is the sample variance); or
#'           \code{sig2 = list(mode, factor, mass)}, the inverse-gamma with
#'           that mode (default \code{var(y)}) whose shape puts \code{mass}
#'           (default 0.9) of its probability in \code{[mode/factor,
#'           mode*factor]} (default factor 2, shape 7.55).
#'         \item \code{range} (scalar or one per input) and \code{nugget}:
#'           half-Cauchy scales, default 1 each -- the density
#'           \eqn{2 / (\pi s (1 + (\theta/s)^2))} on \eqn{\theta > 0} with
#'           scale \eqn{s}.
#'       }
#'       EB returns the posterior mode of \eqn{\theta} on the natural scale
#'       (objective \eqn{\log Z(\theta) + \log p(\theta)}); PMMH samples the
#'       same posterior in \eqn{\log\theta}, Jacobian included.
#'       \code{prior = "none"} makes EB the maximum marginal-likelihood
#'       (maximum-evidence) estimate with no prior at all; PMMH does not
#'       accept it.
#'   }
#'
#' @return An object of class \code{\link[=restree-class]{restree}}: the
#'   model slots, the fitted or supplied \code{theta}, \code{method}, the
#'   realized tree \code{structure} (the node table \code{nodes} and the
#'   weights \code{w}; see \code{\link{restree_trees}}), the ordinary
#'   \code{smc} and \code{mcmc} record lists, and the results (\code{fit} --
#'   a read-only alias for the native Full result or sampler record,
#'   \code{theta_hat}, \code{loglik} / \code{logZ}, \code{diagnostics},
#'   and \code{prediction} when \code{xnew} was supplied), all readable
#'   with \code{$}. For EB, \code{optimization$estimator} and
#'   \code{diagnostics$estimator} report \code{map} or \code{mle}.  PMMH
#'   fits carry coordinatewise posterior medians by default (or means with
#'   \code{control$point_estimate = "mean"}) as \code{theta}
#'   and \code{theta_hat}; the final SMC (\code{smc}, \code{structure}) and
#'   the \code{sd_logZ} replicates are run at that same theta, and the
#'   evidence record of the chain lives in \code{mcmc}, with top-level
#'   read-only aliases: \code{logZ_trace}
#'   (the SMC log evidence of the current state after every iteration),
#'   \code{logZ_prop} (of every proposal), \code{accepted},
#'   \code{logZ_reps}, \code{n_evidence_evals}, next to \code{chain},
#'   \code{accept_rate} and \code{accept_rate_sig2}. Depth-zero
#'   PP uses \code{leaf_map} or \code{leaf_mle} as its diagnostic/optimization
#'   engine label and result entry, respectively; the likelihood engine
#'   remains the exact terminal calculation in both cases.
#'   If \code{xnew} and \code{control$prediction = "mcmc"} were supplied,
#'   PMMH stores per-iteration predictive moments in \code{mcmc$prediction};
#'   plug-in prediction is stored only in \code{prediction}.
#'   \code{mcmc$prediction_method} and \code{mcmc$point_estimate} record the
#'   chosen modes; see \code{restree_predict}.
#'   \code{mcmc$trees} stores compact accepted final SMC ensembles, their
#' weights and iteration mapping, whether or not prediction inputs were supplied.
#'   Repeated accepted states share geometry, but retain separate iteration
#'   weights. The summary trees in \code{structure} remain the final SMC at
#'   point-estimate theta, not the retained iteration trees.
#' @seealso \code{\link{restree_model}}, \code{\link{restree_theta}},
#'   \code{\link{restree_predict}}, \code{\link{restree_score}},
#'   \code{\link{restree_loglik}}
#' @examples
#' set.seed(1)
#' n <- 400; X <- matrix(runif(2 * n), n, 2)
#' y <- sin(4 * pi * X[, 1]) * cos(2 * pi * X[, 2]) + rnorm(n, sd = 0.3)
#' th <- restree_theta(sig2 = 1, range = 0.2, nu = 2.5, nugget = 0.1)
#' m <- restree_model(X, y, depth = 4, r = 15, leaf_model = "PP")
#'
#' ## tree posterior by SMC at fixed covariance parameters (PP leaves)
#' fit <- restree_fit(m, theta = th, method = "smc", nparticles = 50, seed = 1)
#' fit
#'
#' ## predict at new inputs; joint density needs ynew and control$joint_lpd = TRUE
#' Xt <- matrix(runif(200), 100, 2)
#' pred <- restree_predict(fit, xnew = Xt)
#' head(cbind(mean = pred$mean, sd = pred$sd))
#' \donttest{
#' ## joint empirical Bayes for (sigma2, range, nugget), WN leaves
#' fe <- restree_fit(restree_model(X, y, depth = 4, r = 15, leaf_model = "WN"),
#'                   theta = th, method = "ebayes", nparticles = 50,
#'                   seed = 1, control = list(maxit = 25))
#' summary(fe)
#'
#' ## fixed-tree Full leaf model: Gaussian MLE with plug-in prediction
#' ff <- restree_fit(restree_model(X, y, depth = 2, r = 15, leaf_model = "Full"),
#'                   theta = th)
#' ff
#' }
#' @export
restree_fit <- function(model, theta = NULL,
                        method = c("ebayes", "pmmh", "smc"),
                        xnew = NULL, ynew = NULL, nparticles = NULL,
                        resampling = "stratified", temper_alpha = 1,
                        n_iter = NULL, seed = 42L, ncores = 1L, verbose = FALSE,
                        control = list()) {

  if (methods::is(model, "restree")) {
    if (.restree_kind(model) == "object" && is.null(theta)) theta <- model@theta
    model <- .restree_model_of(model)
  }
  model <- .restree_check_model(model)
  leaf <- model@leaf_model
  ## A Full model has one fixed tree: Gaussian MLE, whatever `method` says.
  method <- if (leaf == "Full") "mle" else match.arg(method)
  x <- model@x; y <- model@y; n <- nrow(x); p <- ncol(x)
  depth <- model@depth; r <- model@r
  if (is.null(nparticles))
    nparticles <- .restree_model_nparticles(model)
  nparticles <- .restree_integer(nparticles, "nparticles", 1L)
  opts <- .restree_smc_options(temper_alpha, resampling)
  temper_alpha <- opts$temper_alpha; resampling <- opts$resampling
  seed <- .restree_integer(seed, "seed", 0L)
  thread_policy <- .restree_thread_policy(ncores)
  ncores <- thread_policy$ncores
  verbose <- .restree_flag(verbose, "verbose")
  th <- .restree_theta_for(theta, p)
  nd <- .restree_xnew(xnew, ynew, p)
  Xn <- nd$x; yn <- nd$y
  if (.restree_root_only(model) && leaf == "WhiteNoise" && method == "pmmh")
    warning("The Bayesian depth-zero WN likelihood integrates its leaf parameters; the GP covariance parameter posterior is its prior. Use method='ebayes' for WN mean/variance MLE.",
            call. = FALSE)
  if (.restree_root_only(model)) nparticles <- 1L
  if (method == "pmmh" && is.null(n_iter)) n_iter <- 250L
  if (!is.null(n_iter)) n_iter <- .restree_integer(n_iter, "n_iter", 2L)
  control <- .restree_controls(control, method, n_iter, th, model)
  if (control$joint_lpd && is.null(yn))
    stop("control$joint_lpd = TRUE requires ynew", call. = FALSE)
  # Keep held-out responses for marginal scoring, but only pass them to native
  # prediction when the potentially expensive joint density was requested.
  if (!control$joint_lpd) yn <- NULL

  res <- list(call = match.call(), seed = seed, ncores = ncores,
              thread_policy = thread_policy,
              resampling = resampling, temper_alpha = temper_alpha,
              control = control)
  structure <- NULL

  if (method == "mle") {
    fx <- .restree_fixed_mle(model, th, Xn, yn, control, ncores)
    th <- fx$theta; structure <- fx$structure
    res <- c(res, fx$results)

  } else if (method == "smc") {
    t_fit0 <- proc.time()[3]
    fit <- .restree_smc_fit(model, th, nparticles, Xn, yn, temper_alpha,
                            resampling, seed, ncores)
    structure <- fit$structure_distinct
    res <- c(res, list(fit = fit, loglik = fit$loglik, logZ = fit$logZ,
                       ESS = fit$ESS, nparticles = nparticles,
                       diagnostics = list(engine = fit$engine,
                         n_distinct_trees = fit$n_distinct_trees,
                         n_resample = fit$n_resample,
                         final_ESS = utils::tail(fit$ESS, 1L),
                         smc_time = fit$smc_time,
                         expand_time = fit$expand_time,
                         sample_time = fit$sample_time,
                         pred_time = fit$pred_time,
                         time_s = proc.time()[3] - t_fit0)))

  } else if (method == "ebayes" && .restree_root_only(model) && leaf == "WhiteNoise") {
    wn <- .restree_wn_mle(model, th, Xn, yn)
    th <- wn$theta; structure <- wn$structure
    res <- c(res, wn$results)

  } else if (method == "ebayes") {
    t_fit0 <- proc.time()[3]
    ## The conditional evidence log Z(sigma2, range, nugget) is a genuine
    ## likelihood in all three parameters, so sigma^2 is estimated JOINTLY
    ## with range and nugget (common-random-number Nelder-Mead over the SMC
    ## evidence), then a final SMC runs at the estimate.  The search starts
    ## at theta with sigma^2 = var(y) unless theta supplied one, or at the
    ## Full-tree MLE when control$start asks for it.
    st <- .restree_eb_start(model, th, theta, control$start, ncores)
    th <- st$theta
    sig2_start <- if (st$method == "full_mle") th@sig2 else
      if (!is.null(theta) && (methods::is(theta, "restree_theta") ||
                              !is.null(theta$sig2))) th@sig2 else stats::var(y)
    if (depth == 0L && (!is.finite(sig2_start) || sig2_start <= 0)) sig2_start <- th@sig2
    eb <- .restree_eb_fit(model, th, sig2_start, nparticles, temper_alpha,
                          resampling, control$maxit, control$tol, seed, ncores,
                          control$prior, control$sd_reps)
    eb$start <- st$record
    th2 <- th
    th2@sig2 <- as.numeric(eb$sig2_hat)
    th2@range <- as.numeric(eb$range_hat)
    th2@nugget <- as.numeric(eb$nugget_hat)
    smc <- .restree_smc_fit(model, th2, nparticles, Xn, yn, temper_alpha,
                            resampling, .restree_seed_add(seed, 1L), ncores)
    th <- th2; structure <- smc$structure_distinct
    res <- c(res, list(fit = smc, optimization = eb,
                       theta_hat = .restree_theta_hat(th2),
                       loglik = smc$loglik, logZ = smc$logZ, ESS = smc$ESS,
                       logZ_reps = eb$logZ_reps, sd_logZ = eb$sd_logZ,
                       nparticles = nparticles,
                       diagnostics = list(engine = smc$engine, estimator = eb$estimator,
                         eb_start = st$method, sd_logZ = eb$sd_logZ, sd_reps = length(eb$logZ_reps),
                         eb_convergence_code = eb$convergence,
                         n_distinct_trees = smc$n_distinct_trees,
                         n_resample = smc$n_resample,
                         final_ESS = utils::tail(smc$ESS, 1L),
                         eb_evals = eb$n_eval, eb_evals_full = eb$n_eval_full,
                         eb_evals_cheap = eb$n_eval_cheap, eb_profiled = eb$profiled,
                         eb_seconds = eb$seconds,
                         eb_convergence = eb$convergence,
                         smc_time = smc$smc_time, pred_time = smc$pred_time,
                         time_s = proc.time()[3] - t_fit0)))

  } else { # pmmh
    t_fit0 <- proc.time()[3]
    ## PMMH samples (log sigma2, log range, log nugget) jointly through the
    ## SMC engine: sigma^2 is a sampled parameter of the conditional
    ## evidence, never integrated out.
    sig2_start <- if (!is.null(theta) && (methods::is(theta, "restree_theta") ||
                                          !is.null(theta$sig2))) th@sig2 else stats::var(y)
    if (depth == 0L && (!is.finite(sig2_start) || sig2_start <= 0)) sig2_start <- th@sig2
    th0 <- th; th0@sig2 <- sig2_start
    fit <- .restree_pmmh(model, th0, nparticles = nparticles, n_iter = n_iter,
      burnin = control$burnin, prop_sd = control$prop_sd,
      prior = control$prior,
      sd_reps = control$sd_reps, sig2_moves = control$sig2_moves,
      adapt = control$adapt, target_accept = control$target_accept,
      xnew = if (nd$supplied) Xn else NULL, ynew = yn,
      prediction = control$prediction, point_estimate = control$point_estimate,
      resampling = resampling, temper_alpha = temper_alpha,
      seed = seed, ncores = ncores, verbose = verbose)
    ## an interrupted run (Ctrl-C) returns only the completed iterations,
    ## already a valid shorter chain with its burn-in reduced to fit
    ni <- nrow(fit$chain)
    fit$chain <- .restree_public_chain(fit$chain, th)
    fit$chain_prop <- .restree_public_chain(fit$chain_prop, th)
    burn <- fit$burnin
    prediction_index <- fit$trees$index
    online_prediction <- fit$prediction
    fit$prediction <- NULL
    if (!is.null(online_prediction$mcmc_prediction))
      online_prediction$mcmc_prediction$theta <- fit$chain[prediction_index, , drop = FALSE]
    ## ONE point summary for the whole object: the chosen posterior summary of the
    ## retained draws, at which .restree_pmmh ran the final SMC (fit$fit,
    ## the stored structures) and the sd_logZ replicates. Plug-in prediction
    ## uses this point; MCMC prediction averages all retained joint states.
    th <- fit$theta
    structure <- fit$fit$structure_distinct
    ## fit = the pmmh record (chain, logZ_trace, chain_prop, accepted, ...;
    ## its $fit is the final SMC at theta)
    ## Evidence diagnostics at the top level:
    ## logZ_trace / logZ_prop, the SMC log evidence of the chain's current
    ## state and of every proposal (one value per iteration, fresh seed
    ## each); logZ, the final SMC at the point estimate; logZ_reps, the
    ## sd_reps replicates at that same theta.
    res <- c(res, list(fit = fit, chain = fit$chain, burnin = burn,
                       accept_rate = fit$accept_rate,
                       accept_rate_post = fit$accept_rate_post,
                       accept_rate_sig2 = fit$accept_rate_sig2,
                       accepted = fit$accepted,
                       prop_sd_final = fit$prop_sd_final,
                       prop_scale_trace = fit$prop_scale_trace,
                       logZ_trace = as.numeric(fit$logZ_trace),
                       logZ_prop = as.numeric(fit$logZ_prop),
                       n_evidence_evals = fit$n_evidence_evals,
                       theta_hat = .restree_theta_hat(th),
                       nparticles = nparticles,
                       prediction_index = prediction_index,
                       logZ_reps = fit$logZ_reps, sd_logZ = fit$sd_logZ,
                       logZ = fit$fit$logZ, ESS = fit$fit$ESS,
                       diagnostics = list(engine = "smc",
                         accept_rate = fit$accept_rate, n_iter = ni,
                         burnin = burn, interrupted = isTRUE(fit$interrupted),
                         sd_logZ = fit$sd_logZ, sd_reps = length(fit$logZ_reps),
                         n_distinct_trees = fit$fit$n_distinct_trees,
                         final_ESS = utils::tail(fit$fit$ESS, 1L),
                         time_s = proc.time()[3] - t_fit0)))
  }

  ## Root-only model: no tree is sampled, so estimator labels name the leaf
  ## (the SMC engine label stays "smc" for smc / pmmh: the run is the
  ## engine's single root-stop structure).
  if (.restree_root_only(model) && method %in% c("mle", "ebayes")) {
    leaf_engine <- if (method == "ebayes" && leaf == "PP")
      paste0("leaf_", res$optimization$estimator) else "leaf_mle"
    res$diagnostics$engine <- leaf_engine
    if (method == "ebayes" && leaf == "PP") {
      res[[leaf_engine]] <- list(family = "PP")
      res$optimization$engine <- leaf_engine
    }
  }
  out <- .restree_new(model, th, method, structure, res)
  ## prediction, if it came free with the fit
  if (nd$supplied) {
    if (method == "pmmh") {
      out@prediction <- .restree_prediction(online_prediction, nd$y, Xn,
        control$prediction, control$point_estimate)
      out@mcmc$prediction <- online_prediction$mcmc_prediction
      if (!is.null(out@mcmc$prediction)) out@mcmc$prediction$y <- nd$y
    } else out@prediction <- .restree_prediction(out$fit, nd$y, Xn)
  }
  out
}

## Starting point of the EB search.  control$start = "theta" (the default)
## keeps theta.  With "full_mle" the Full leaf model's Gaussian MLE on the
## same tree settings (depth, r, design, cut_method; the nested design has no
## Full counterpart and falls back to maximin) provides (sigma^2, range,
## nugget); "auto" does so only when theta supplies no range. Requested depth 0
## keeps theta rather than constructing a separate starting fit. Full's own
## constructor may extend any initial depth until terminal sizes are <= r;
## a single dense root occurs only for n <= r. The start fit uses three
## L-BFGS-B starts with shared geometry. Its cost relative to SMC depends on
## the workload. Return theta, its selection method and optimization$start.
.restree_eb_start <- function(model, th, theta, start, ncores) {
  no_range <- is.null(theta) || (is.list(theta) && !methods::is(theta, "restree_theta") &&
                                 is.null(theta$range))
  use_full <- model@depth >= 1L &&
    (start == "full_mle" || (start == "auto" && no_range))
  if (!use_full)
    return(list(theta = th, method = "theta",
                record = list(method = "theta", theta = .restree_theta_hat(th))))
  t0 <- proc.time()[3]
  full <- restree_model(model@x, model@y, depth = model@depth, r = model@r,
                        leaf_model = "Full",
                        design = if (model@design == "nested") "maximin" else model@design,
                        cut_method = model@cut_method, seed = model@seed)
  ctl <- .restree_controls(list(n_starts = 3L), "mle", NULL, th, full)
  fx <- .restree_fixed_mle(full, th, matrix(numeric(), 0L, ncol(model@x)), NULL, ctl, ncores)
  th0 <- fx$theta
  list(theta = th0, method = "full_mle",
       record = list(method = "full_mle", theta = .restree_theta_hat(th0),
                     loglik = fx$results$loglik,
                     n_eval = unname(fx$results$optimization$counts[["function"]]),
                     convergence = fx$results$optimization$convergence,
                     n_starts = fx$results$optimization$n_starts,
                     seconds = unname(proc.time()[3] - t0)))
}

## Depth-zero WN MLE: iid N(mu, variance), with both parameters estimated.
## The divisor is n, not n-1. A constant sample has no finite positive-variance
## Gaussian MLE, so report it rather than silently flooring the variance.
.restree_wn_mle <- function(model, th, Xn, yn) {
  t0 <- proc.time()[3]
  y <- model@y
  mu <- mean(y); variance <- mean((y-mu)^2)
  if (!is.finite(variance) || variance <= 0)
    stop("WN MLE requires nonconstant data: no finite positive-variance Gaussian MLE exists for a constant or singleton sample.",
         call. = FALSE)
  pars <- list(family="gaussian_wn", mean=mu, variance=variance)
  ll <- sum(stats::dnorm(y, mu, sqrt(variance), log=TRUE))
  initial <- sum(stats::dnorm(y, 0, sqrt(th@sig2), log=TRUE))
  th@sig2 <- variance
  nodes <- data.frame(tree = 1L, id = 1L, split = FALSE, J = NA_integer_, cut = NA_real_,
                      rho = NA_real_, n = length(y))
  fit <- list(nodes=nodes,logZ=ll,loglik=ll,Phi0=ll,
    ESS=1,w=1,w_distinct=1,n_distinct_trees=1L,n_resample=0L,nparticles=1L,
    engine="leaf_mle",state=NULL,smc_time=0,expand_time=0,sample_time=0)
  pred <- .restree_predict_wn_mle(pars,Xn,yn)
  fit[names(pred)] <- pred
  seconds <- unname(proc.time()[3]-t0)
  list(theta=th,structure=list(nodes=nodes,w=1),
    results=list(fit=fit,leaf_mle=pars,theta_hat=c(mean=mu,sig2=variance),
      loglik=ll,logZ=ll,ESS=1,nparticles=1L,
      optimization=list(mean_hat=mu,sig2_hat=variance,logZ_hat=ll,
        logZ_initial=initial,n_eval=1L,convergence=0L,seconds=seconds,
        engine="leaf_mle",estimator="mle",optimizer="analytic Gaussian MLE"),
      diagnostics=list(engine="leaf_mle",estimator="mle",n_distinct_trees=1L,n_resample=0L,
        final_ESS=1,eb_evals=1L,eb_seconds=seconds,eb_convergence=0L,time_s=seconds)))
}

.restree_predict_wn_mle <- function(pars, Xn, yn=NULL) {
  mu <- .restree_number(pars$mean,"fitted WN mean")
  variance <- .restree_number(pars$variance,"fitted WN variance",0,Inf,lower_closed=FALSE)
  n <- nrow(Xn)
  ll <- if (is.null(yn)) NA_real_ else sum(stats::dnorm(yn,mu,sqrt(variance),log=TRUE))
  list(mean=rep(mu,n),var=rep(variance,n),par_mean=matrix(mu,n,1),
    par_var=matrix(variance,n,1),w=1,lpd=ll,lpd_particle=ll,pred_time=0)
}

## Named vector of the estimated covariance parameters.
.restree_theta_hat <- function(th) {
  c(sig2 = th@sig2,
    stats::setNames(th@range, if (th@form %in% c("ARD", "tensor"))
      paste0("range", seq_along(th@range)) else "range"),
    nugget = th@nugget)
}

## Fixed Full tree: the partition and knot design are fixed. Profile sigma2
## analytically and optimize the ARD/tensor/isotropic ranges and nugget by ordinary
## Gaussian MLE on the log scale from a deterministic multi-start set (the
## caller's start plus broad data-scale starts); L-BFGS-B box constraints
## prevent numerically singular correlation matrices.
##
## The tree itself -- which rows reach each node, the knots the design picks
## there, the cuts and the left/right routing -- is part of the model: it was
## built once by restree_model() and is stored in model$tree.  Every objective
## evaluation and the final fit reuse one owned native geometry prepared from
## that stored tree. Only theta-dependent kernels, factorizations and residual
## propagation are repeated. The handle is local to this fit, shared by all
## starting points, and never retained in the returned S4 object.
## `seed` of restree_fit() plays no role here.
# Native worker/allocation failures are not invalid covariance proposals.
# Keep the original condition so callers retain its type and error metadata.
.restree_rethrow_native_failure <- function(e) {
  msg <- conditionMessage(e)
  if (startsWith(msg, "SMC internal error:") ||
      startsWith(msg, "TreeSMC internal error:") ||
      startsWith(msg, "ResTree worker error:") ||
      inherits(e, "std::bad_alloc") || grepl("bad_alloc|cannot allocate", msg)) stop(e)
  invisible(NULL)
}

.restree_fixed_mle <- function(model, th, Xn, yn, control, ncores) {
  y <- model@y; p <- ncol(model@x)
  empty_new <- matrix(numeric(), 0L, p)
  handle <- full_fit_engine_new_cpp(model, th, ncores)
  on.exit(full_fit_engine_release_cpp(handle), add = TRUE)
  evaluate <- function(z, predict = FALSE) {
    if (any(!is.finite(z)) || any(z < -30) || any(z > 30)) return(NULL)
    tz <- th
    tz@range <- exp(z[seq_along(th@range)])
    tz@nugget <- exp(z[length(z)])
    tz@sig2 <- 1
    f <- tryCatch(full_fit_engine_eval_cpp(handle, tz, if (predict) Xn else empty_new,
      if (predict) yn else NULL), error = function(e) {
        .restree_rethrow_native_failure(e); NULL
      })
    if (is.null(f) || !is.finite(f$quad) || !is.finite(f$logdet) ||
        f$n_gauss != length(y) || f$quad <= 0) return(NULL)
    f$sig2_mle <- f$quad / length(y)
    f$gaussian_loglik <- -0.5 * (length(y) *
      (log(2 * pi) + 1 + log(f$sig2_mle)) + f$logdet)
    list(fit = f, theta = tz)
  }
  objective <- function(z) {
    ev <- evaluate(z, FALSE)
    if (is.null(ev)) 1e100 else -ev$fit$gaussian_loglik
  }
  nr <- length(th@range)
  z0 <- c(log(th@range), log(th@nugget))
  ## the search box (control$range_bounds / control$nugget_bounds, natural
  ## scale): L-BFGS-B bounds, and every start is clamped into it
  lower <- c(rep(log(control$range_bounds[1L]), nr), log(control$nugget_bounds[1L]))
  upper <- c(rep(log(control$range_bounds[2L]), nr), log(control$nugget_bounds[2L]))
  clamp <- function(z) pmin(upper, pmax(lower, z))
  starts <- list(clamp(z0))
  if (control$n_starts > 1L) {
    base_range <- exp(seq(log(0.03), log(1.0), length.out = control$n_starts - 1L))
    base_nugget <- exp(seq(log(0.003), log(0.20), length.out = control$n_starts - 1L))
    ard_offset <- if (nr == 1L) 0 else seq(-0.20, 0.20, length.out = nr)
    for (j in seq_len(control$n_starts - 1L)) {
      starts[[length(starts) + 1L]] <- clamp(c(log(base_range[j]) + ard_offset,
                                               log(base_nugget[j])))
    }
  }
  start_key <- vapply(starts, function(z) paste(formatC(z, digits = 16L,
                                                          format = "fg"), collapse = ","),
                      character(1L))
  starts <- starts[!duplicated(start_key)]
  run_one <- function(start) {
    if (control$optimizer == "L-BFGS-B") {
      stats::optim(start, objective, method = "L-BFGS-B", lower = lower, upper = upper,
        control = list(maxit = control$maxit,
                       factr = max(1, control$reltol / .Machine$double.eps),
                       pgtol = sqrt(control$reltol)))
    } else {
      stats::optim(start, objective, method = "Nelder-Mead",
        control = list(maxit = control$maxit, reltol = control$reltol))
    }
  }
  opt_all <- lapply(starts, function(start) tryCatch(run_one(start), error = function(e) {
    .restree_rethrow_native_failure(e); NULL
  }))
  valid_opt <- vapply(opt_all, function(z) !is.null(z) && is.finite(z$value), logical(1L))
  if (!any(valid_opt)) stop("fixed-tree covariance MLE failed", call. = FALSE)
  opt_all <- opt_all[valid_opt]
  opt <- opt_all[[which.min(vapply(opt_all, `[[`, numeric(1L), "value"))]]
  opt$optimizer <- control$optimizer
  opt$n_starts <- length(starts)
  opt$all_starts <- opt_all
  ev <- evaluate(opt$par, FALSE)
  if (is.null(ev)) stop("fixed-tree covariance MLE failed", call. = FALSE)
  th2 <- ev$theta; th2@sig2 <- ev$fit$sig2_mle
  fit <- full_fit_engine_eval_cpp(handle, th2, Xn, yn)
  ## Release native data and geometry without waiting for pointer finalization.
  full_fit_engine_release_cpp(handle)
  fit$gaussian_loglik <- ev$fit$gaussian_loglik
  fit$loglik <- fit$gaussian_loglik
  list(theta = th2,
       ## the one fixed tree, with its node sizes, from the model's tree slot
       structure = list(nodes = .restree_nodes_from_fixed_tree(model@tree, model@depth), w = 1),
       results = list(fit = fit, optimization = opt,
                      theta_hat = .restree_theta_hat(th2),
                      loglik = fit$gaussian_loglik))
}

## Run the SMC (PP/WhiteNoise only) and return a fit list.  par_mean /
## par_var / lpd_particle are per DISTINCT tree (with matching w_distinct);
## structure_distinct holds the distinct realized structures (node table)
## with aggregated weights -- the mixture over distinct trees is identical
## to the per-particle mixture, and later restree_predict() rebuilds only the
## distinct trees.
.restree_smc_fit <- function(model, th, nparticles, Xn, yn, temper_alpha,
                             resampling, seed, ncores) {
  fit <- .restree_smc(model, th, nparticles = nparticles,
    xnew = if (nrow(Xn) > 0) Xn else NULL, ynew = yn,
    resampling = resampling, temper_alpha = temper_alpha,
    seed = seed, ncores = ncores)
  ## fit$nodes arrives per DISTINCT tree (tree = distinct_rep order), so
  ## keeping the positive-weight trees is a row subset with renumbering.
  keep <- which(fit$w_distinct > 0)
  nodes <- fit$nodes[fit$nodes$tree %in% keep, , drop = FALSE]
  nodes$tree <- match(nodes$tree, keep)
  rownames(nodes) <- NULL
  # Preserve the exact positive SMC masses used by live prediction. Native
  # prediction normalizes them once; normalizing here too causes restoration
  # roundoff and can erase positive subnormal support.
  fit$structure_distinct <- list(nodes = nodes, w = fit$w_distinct[keep])
  if (model@leaf_model == "WhiteNoise") fit$structure_distinct$depth <- as.integer(fit$tree_depth)
  ## structure_distinct owns the node table; do not retain a duplicate in smc.
  fit$nodes <- NULL
  ## The shared environment caches rebuilt native prediction state after
  ## serialization. A reloaded fit rebuilds lazily on its first prediction.
  fit$rebuilt <- new.env(parent = emptyenv())
  fit
}

## Empirical Bayes: common-random-number Nelder-Mead over the conditional SMC
## evidence in ALL THREE parameters (log sigma^2, log range, log nugget).
## sigma^2 is a free parameter of the conditional likelihood and is estimated
## jointly.  A fixed SMC seed makes the stochastic objective deterministic
## across evaluations.
.restree_eb_fit <- function(model, th, sig2_start, nparticles, temper_alpha,
                            resampling, maxit, tol, seed, ncores, prior,
                            sd_reps = 5L) {
  smc <- list(temper_alpha = temper_alpha, resampling = resampling,
              seed = seed, ncores = ncores,
              engine = .restree_engine_handle(model, nparticles, temper_alpha,
                                              resampling, ncores))
  ## Release the persistent engine on every exit, including an error thrown by an
  ## evaluation (an OOM at large n, or a WN split beyond the depth ceiling). The
  ## external pointer has a deleting finalizer, so R's collector does reclaim the
  ## engine eventually -- but R only sees the pointer's few bytes, not the data
  ## copy and up to path_cache_mb + keep_cache_mb of caches behind it, so that
  ## collection can be arbitrarily late. The success path releases early (below);
  ## this is the deterministic release for the error paths.
  ## smc_engine_release_cpp is idempotent.
  on.exit(if (!is.null(smc$engine)) smc_engine_release_cpp(smc$engine), add = TRUE)
  nr <- length(th@range)
  n_full <- 0L; n_cheap <- 0L
  logZ_last <- NA_real_
  ## Posterior MODE of theta on the natural scale: the objective is
  ## -(log Z(theta) + log p(theta)) with the prior of .restree_log_prior
  ## (inverse-gamma on sig2, half-Cauchy on every range and on the nugget);
  ## the search variable is log theta but no Jacobian is added, so the
  ## optimum is the mode of the theta-scale posterior.  (The pmmh sampler
  ## targets the same posterior in u = log theta, Jacobian included.)
  ## With control$prior = "none" the prior term is zero and the optimum is
  ## the maximum marginal-likelihood (maximum-evidence) estimate.
  obj <- function(p) {
    z <- .restree_log_evidence(model, th, nparticles,
      exp(p[1L]), exp(p[1L + seq_len(nr)]), exp(p[2L + nr]), smc)
    logZ_last <<- z
    lpr <- .restree_log_prior(exp(p[1L]), exp(p[1L + seq_len(nr)]), exp(p[2L + nr]), prior)
    if (is.finite(z) && is.finite(lpr)) -(z + lpr) else 1e100
  }
  t0 <- proc.time()[3]
  p0 <- c(log(sig2_start), log(th@range), log(th@nugget))
  logpost_initial <- -obj(p0); logZ_initial <- logZ_last
  ## sigma^2 is a pure scale of every Gaussian block: none of the engine's
  ## linear algebra depends on it, and the persistent engine re-evaluates a
  ## sig2-only change from its retained block statistics by rerunning the
  ## sampler alone (SMC::refresh_sig2_values).  The search is therefore
  ## nested (numerically -- nothing is profiled in closed form):
  ## common-random-number Nelder-Mead over (log range, log nugget), each
  ## evaluation minimising over log sigma^2 by Brent's method on the same
  ## seed -- one full evaluation per outer point (the first sig2 value at a
  ## new (range, nugget)), the rest sig2-only reruns.  The nested objective
  ## has the same maximiser as the joint one.  Used under deterministic cuts
  ## only.  The engine also reuses its trie for integrated-cut WN under the
  ## maximin design (same seed; see SMC::wn_integrated_reuse), but the
  ## nested search is not routed there: Brent's sigma^2 steps flip stop/split
  ## draws and, with one sequential RNG stream per particle, every later
  ## node of a flipped particle can land in a new rank cell that requires
  ## expansion, reducing the benefit of sigma^2-only evaluations.
  profile_ok <- !is.null(smc$engine) && model@cut_method %in% c("middle", "median")
  if (profile_ok) {
    ls2_cur <- p0[1L]
    inner <- function(p2) {
      f1 <- function(ls2) { n_cheap <<- n_cheap + 1L; obj(c(ls2, p2)) }
      n_full <<- n_full + 1L; n_cheap <<- n_cheap - 1L
      half <- 1.0   # bracket in log sigma^2 around the last profile optimum
      for (widen in 1:4) {
        o <- stats::optimize(f1, c(ls2_cur - half, ls2_cur + half), tol = 1e-3)
        at_edge <- min(o$minimum - (ls2_cur - half), (ls2_cur + half) - o$minimum) < 0.02 * half
        if (!at_edge) break
        ls2_cur <- o$minimum; half <- 2 * half
      }
      ls2_cur <<- o$minimum
      o$objective
    }
    op2 <- stats::optim(p0[-1L], inner, method = "Nelder-Mead",
                        control = list(maxit = maxit, reltol = tol))
    ## final polish of sigma^2 at the outer optimum, then report the joint point
    f1 <- function(ls2) obj(c(ls2, op2$par))
    o <- stats::optimize(f1, c(ls2_cur - 1.5, ls2_cur + 1.5), tol = 1e-6)
    n_cheap <- n_cheap + 1L
    par <- c(o$minimum, op2$par); value <- o$objective
    convergence <- op2$convergence
    counts <- c(full = n_full, cheap = n_cheap)
  } else {
    ## Single-stage CRN Nelder-Mead over all three parameters (random-cut
    ## samplers keep no per-path candidate state to reuse).
    op <- stats::optim(p0, obj, method = "Nelder-Mead",
                       control = list(maxit = maxit, reltol = tol))
    par <- op$par; value <- op$value; convergence <- op$convergence
    counts <- c(full = unname(op$counts[1L]) + 1L, cheap = 0L)   # optim() names it "function"
  }
  ## evidence and prior at the estimate, reported separately
  logpost_hat <- -obj(par); logZ_hat <- logZ_last
  if (isTRUE(convergence == 1L))
    warning(sprintf(paste0("the EB search stopped at control$maxit = %d Nelder-Mead iterations ",
                           "without meeting control$tol; raise maxit or restart from theta_hat"), maxit),
            call. = FALSE)
  ## Estimator noise at the estimate: sd_reps fresh-seed evidence evaluations
  ## at theta_hat (seeds continue past the final SMC's seed + 1).  The CRN
  ## objective cannot show this -- every value it saw shares one seed -- and
  ## logZ_hat inherits the maximum of that seed's realisation over theta.
  logZ_reps <- numeric(0)
  if (sd_reps > 0L) {
    logZ_reps <- vapply(seq_len(sd_reps), function(k) {
      smc_k <- smc; smc_k$seed <- .restree_seed_add(seed, 1L + k)
      as.numeric(.restree_log_evidence(model, th, nparticles, exp(par[1L]),
        exp(par[1L + seq_len(nr)]), exp(par[2L + nr]), smc_k))
    }, numeric(1))
  }
  ## Release the optimizer's native data and caches before the final SMC.
  if (!is.null(smc$engine)) smc_engine_release_cpp(smc$engine)
  list(sig2_hat = exp(par[1L]),
       range_hat = exp(par[1L + seq_len(nr)]),
       nugget_hat = exp(par[2L + nr]),
       logZ_hat = logZ_hat, logpost_hat = logpost_hat,
       logZ_initial = logZ_initial, logpost_initial = logpost_initial,
       logZ_reps = logZ_reps,
       sd_logZ = if (sum(is.finite(logZ_reps)) >= 2L) stats::sd(logZ_reps[is.finite(logZ_reps)]) else NA_real_,
       prior = prior, estimator = if (is.null(prior)) "mle" else "map",
       n_eval = unname(counts["full"] + counts["cheap"]) + 1L,
       n_eval_full = unname(counts["full"]), n_eval_cheap = unname(counts["cheap"]),
       profiled = profile_ok,
       convergence = convergence, seconds = proc.time()[3] - t0,
       engine = "smc")
}

.restree_show_fit <- function(x, ...) {
  cat("restree fit\n")
  cat("  method    :", x$method, "\n")
  cat("  data      :", length(x$y), "observations,", ncol(x$x), "inputs\n")
  cat("  tree      : depth", x$depth, "; r =", x$r, "; leaf model", x$leaf_model,
      "; design", x$design, "; cuts", x$cut_method, "\n")
  th <- x$theta
  cat("  theta     : sig2", format(th$sig2, digits = 4),
      "; range", paste(format(th$range, digits = 4), collapse = " "),
      "; nugget", format(th$nugget, digits = 4),
      "; nu", th$nu, ";", th$family, th$form, "\n")
  if (identical(x$leaf_mle$family, "gaussian_wn"))
    cat("  WN MLE    : mean", format(x$leaf_mle$mean, digits=4),
        "; variance", format(x$leaf_mle$variance,digits=4), "\n")
  if (!identical(x$leaf_mle$family, "gaussian_wn") &&
      !is.null(x$theta_hat) && "sig2" %in% names(x$theta_hat) &&
      x$method %in% c("ebayes", "pmmh"))
    cat("  sigma2_hat:", format(unname(x$theta_hat["sig2"]), digits = 4),
        if (x$method == "ebayes") {
          if (identical(x$optimization$estimator, "mle")) "(joint maximum-evidence MLE;"
          else "(joint EB posterior mode;"
        } else
          paste0("(posterior ", if (is.null(x$point_estimate)) "mean" else x$point_estimate, ";"),
        "identified noise variance sig2*nugget =",
        format(unname(x$theta_hat["sig2"] * x$theta_hat["nugget"]),
               digits = 4), ")\n")
  if (!is.null(x$nparticles)) cat("  particles :", x$nparticles,
      if (!is.null(x$ESS)) paste0(" (final ESS ", format(utils::tail(x$ESS, 1L), digits = 3), ")") else "", "\n")
  if (!is.null(x$logZ))   cat("  log Z     :", format(x$logZ, digits = 8), "\n")
  if (!is.null(x$loglik)) cat("  loglik    :", format(x$loglik, digits = 8), "\n")
  if (!is.null(x$accept_rate)) cat("  accept    :", format(x$accept_rate, digits = 3), "\n")
  if (!is.null(x$prediction)) cat("  prediction:", length(x$prediction$mean), "new points\n")
  if (!is.null(x$diagnostics)) {
    dg <- x$diagnostics
    cat("  diagnostics: ",
        if (!is.null(dg$n_distinct_trees)) paste0(dg$n_distinct_trees, " distinct trees; ") else "",
        if (!is.null(dg$n_resample)) paste0(dg$n_resample, " resampling events; ") else "",
        if (!is.null(dg$accept_rate)) paste0("accept ", format(dg$accept_rate, digits = 3), "; ") else "",
        if (!is.null(dg$time_s)) paste0(format(dg$time_s, digits = 3), "s total") else "",
        "\n", sep = "")
  }
  invisible(x)
}

## S4 display, summary and predict methods live in S4classes.R;
## the S4 plot method lives in plotfuns.R.

##########################################################################
## Internal drivers of the compiled sampler (src/SMC.cpp), called
## only from restree_fit() / restree_predict():
##   .restree_smc(model, theta)    one SMC run over trees at fixed theta, with
##                                 optional native prediction / joint LPD; the
##                                 fitted engine is kept as an external pointer.
##   .restree_pmmh(model, theta0)  particle-marginal MH over
##                                 (log sig2, log range, log nugget).
## Empirical Bayes (.restree_eb_fit above) reuses .restree_log_evidence() below.
## The pure evidence number is restree_loglik() (compiled, src/restree_loglik.cpp).
##
## With deterministic cuts ("middle"/"median"), candidate split computation
## is deduplicated across particles that share a node path. With the random
## cut priors ("uniform" = the CGM Bayesian CART prior, uniform over the rank
## intervals of the node's residual observations; "balanced" = the same rank
## intervals reweighted by the Beta(2,2) density on the rank scale) the PP
## candidate-split evaluations are per particle (the stop/split posteriors
## depend on each particle's own drawn cuts) unless design = "nested", so
## time scales with nparticles; WN integrates the prior exactly per path and
## shares children by rank cell, which is what lets its persistent engine
## reuse the trie across sigma^2 at a common seed (design = "maximin").
## Runtime and memory must be benchmarked for the intended n, d, depth, r and
## particle count.
##########################################################################

#' SMC over residual trees at fixed covariance parameters
#'
#' Runs the sequential Monte Carlo tree sampler for a
#' \code{\link{restree_model}} at the covariance parameters \code{theta},
#' with shared computation where the cut rule permits it. Deterministic cuts
#' share candidate calculations among particles following the same node path;
#' random cuts require particle-specific candidate evaluations. This is the
#' sampler behind \code{restree_fit(method = "smc")}; the evidence alone is
#' \code{\link{restree_loglik}}.
#'
#' @param model a \code{\link{restree_model}} with \code{leaf_model} PP or WN.
#' @param theta a \code{\link{restree_theta}} or named list.
#' @param nparticles number of SMC particles; \code{NULL} (default) uses
#'   \eqn{100\,\mathrm{depth}} for both PP and WN, using the configured
#'   model depth; depth zero uses one particle.
#' @param xnew,ynew optional prediction inputs and held-out outputs (joint LPD).
#' @param resampling \code{"stratified"} (default) or \code{"multinomial"}.
#' @param temper_alpha tempering exponent of the SMC weights in \code{(0, 1]}
#'   (default 1, the plain evidence estimator).
#' @param seed,ncores random seed and threads (\code{NULL} = auto; the
#'   contract of \code{ncores} in \code{\link{restree_fit}}).
#' @return Ordinary list with the sampler fields (\code{logZ}; \code{nodes}, the
#'   node table of the DISTINCT trees in \code{distinct_rep} order (see
#'   \code{\link{restree_trees}}), particles mapped to trees by
#'   \code{structure_of_particle}; \code{ESS}, \code{w}, ...),
#'   engine diagnostics (\code{n_trie}, \code{n_expanded},
#'   \code{distinct_per_level}, \code{expand_time}, \code{sample_time},
#'   \code{n_distinct_trees}), native prediction context in \code{state}, and, when
#'   \code{xnew} is given, \code{mean}, \code{var}, \code{lpd},
#'   \code{lpd_particle} (per distinct tree).
#' @keywords internal
#' @noRd
.restree_smc <- function(model, theta, nparticles = NULL,
                        xnew = NULL, ynew = NULL,
                        resampling = "stratified", temper_alpha = 1,
                        seed = 42L, ncores = 1L) {
  model <- .restree_check_model(model, smc = TRUE)
  th <- .restree_theta_for(theta, ncol(model@x))
  if (is.null(nparticles))
    nparticles <- .restree_model_nparticles(model)
  nparticles <- .restree_integer(nparticles, "nparticles", 1L)
  seed <- .restree_integer(seed, "seed", 0L)
  ncores <- .restree_ncores(ncores)
  opts <- .restree_smc_options(temper_alpha, resampling)
  nd <- .restree_xnew(xnew, ynew, ncol(model@x))

  ## Sampling, prediction AND (when ynew is given) the joint log predictive
  ## density all run natively in the engine: the predictive-process terms
  ## are computed once per distinct node-path and shared by every tree
  ## through it, leaves are handled separately, sibling subtrees descend as
  ## OpenMP tasks, and the LPD accumulates per distinct tree in the SAME
  ## traversal (no per-tree fixed-structure rebuild). Root-only models need
  ## one particle, including a depth-zero WN model with n > r.
  if (.restree_root_only(model)) nparticles <- 1L
  fit <- smc_sample_cpp(model, th, nparticles, seed, ncores,
    opts$temper_alpha, opts$resampling,
    if (nd$supplied) nd$x else NULL,
    if (is.null(nd$y)) NULL else as.numeric(nd$y))
  fit$nparticles <- nparticles
  fit$engine <- "smc"
  fit
}


## ---------------------------------------------------------------------------
## Estimation of the global covariance parameters.
##
## For the PP and WhiteNoise leaf models there is exactly ONE coherent
## estimation target: the CONDITIONAL evidence log Z(sigma2, range, nugget).
## The tree marginal Q(A) = S Q0(A) + (1-S) QR(A) Q(Al) Q(Ar) is a product of
## Gaussian factors sharing the one scale sigma2, so it is multiplicative --
## and hence computable by the recursion -- only conditionally on sigma2;
## node-wise integration would give every block its own scale and is wrong,
## and global integration does not commute with the recursion.  sigma2 is
## therefore a free parameter estimated JOINTLY with (range, nugget).
## ---------------------------------------------------------------------------

## log Z(theta) for the optimizers / samplers: the same sampler as
## restree_fit(method = "smc") (same resampling and tempering) and the
## same likelihood convention (restree_loglik and .restree_smc both score a
## small-node WN split as the dense knot block), so the objective that is
## optimised and the log Z reported by the final fit agree. Invalid covariance
## proposals return -Inf; internal-state errors and WN depth/allocation limits
## propagate as explicit errors.
.restree_log_evidence <- function(model, th, nparticles, sig2, range, nugget,
                                  smc) {
  th2 <- th; th2@sig2 <- sig2; th2@range <- range; th2@nugget <- nugget
  reject_theta <- function(e) {
    .restree_rethrow_native_failure(e)
    # A selected WN split beyond the representation limit must abort, not
    # become a zero likelihood that biases the estimator toward shallow trees.
    if (model@leaf_model == "WhiteNoise" &&
        grepl("WN selected a split at level", conditionMessage(e))) stop(e)
    -Inf
  }
  if (!is.null(smc$engine))
    ## the persistent engine of this fit (see .restree_engine_handle): the
    ## same sampler on the same configuration, with the theta-independent
    ## path geometry kept across evaluations.  diagnostics = FALSE: the
    ## optimizers and samplers read the value alone, so the distinct-tree
    ## grouping, knot counts and ESS attributes are not computed.
    return(tryCatch(smc_engine_logZ_cpp(smc$engine, th2, smc$seed, FALSE),
                    error = reject_theta))
  tryCatch(restree_loglik(model, th2, nparticles = nparticles,
    resampling = smc$resampling, temper_alpha = smc$temper_alpha,
    seed = smc$seed, ncores = smc$ncores), error = reject_theta)
}

## One SMC engine per estimator run (EB / PMMH): the compiled handle owns a
## single copy of the data and the sampler configuration (nparticles,
## resampling, temper_alpha, ncores -- validated against known physical cores
## at the R/native entry points, with no memory preflight) and keeps the
## theta-independent geometry of every
## expanded node-path (maximin knots, cuts, candidate-child leaf knots) in a
## path-keyed cache across evaluations.  Deterministic cuts benefit; under
## quantile cuts the handle still saves the per-call setup.  Every evidence
## value is bit-identical to restree_loglik() at the same arguments.  NULL
## (the restree_loglik route: a fresh engine and data copy per evaluation,
## no reuse of anything) if the compiled handle cannot be created -- with a
## warning because the fallback loses the persistent engine's reuse.
.restree_engine_handle <- function(model, nparticles, temper_alpha, resampling,
                                   ncores) {
  if (.restree_root_only(model)) return(NULL)
  ## invalid sampler options are the caller's error and propagate; only a
  ## failure to CREATE the engine (memory, native state) takes the fallback
  opts <- .restree_smc_options(temper_alpha, resampling)
  tryCatch(smc_engine_new_cpp(model, nparticles, opts$resampling, opts$temper_alpha,
                              ncores),
           error = function(e) {
             .restree_rethrow_native_failure(e)
             warning("the persistent SMC engine could not be created (",
                     conditionMessage(e), "); falling back to one fresh engine ",
                     "per evidence evaluation, which is much slower", call. = FALSE)
             NULL
           })
}

#' Particle-marginal Metropolis--Hastings for PP / WN trees
#'
#' Random-walk MH on \eqn{(\log\sigma^2, \log\mathrm{range},
#' \log\mathrm{nugget})} under an inverse-gamma prior on \eqn{\sigma^2}
#' and half-Cauchy priors on every range and on the nugget (densities on
#' the natural scale; the chain's log parameterisation adds its Jacobian), using the
#' conditional SMC evidence in the acceptance ratio (Andrieu, Doucet &
#' Holenstein, 2010).  \eqn{\sigma^2} is a sampled parameter -- never
#' integrated or profiled.  Each iteration refreshes the SMC seed, as PMMH
#' requires.
#'
#' @param model a \code{\link{restree_model}} with \code{leaf_model} PP or WN.
#' @param theta0 starting values and fixed quantities (a
#'   \code{\link{restree_theta}} or named list): \code{range} (scalar, or
#'   length-d for \code{form = "ARD"} or \code{"tensor"}), \code{nugget}, optionally \code{sig2}
#'   (default \code{var(y)}), and the fixed \code{nu}, \code{tail},
#'   \code{form}, \code{family}, \code{dtype}.
#' @param nparticles SMC particles per evidence evaluation.
#' @param n_iter,burnin chain length and discarded burn-in.
#' @param prop_sd random-walk standard deviations on the log scale, ordered
#'   \code{(sig2, range..., nugget)}; a scalar is recycled.
#' @param prior the prior list of \code{.restree_prior_control}: inverse-gamma
#'   on \eqn{\sigma^2}, half-Cauchy on every range and on the nugget.
#' @param sd_reps independent SMC evaluations of \eqn{\log \hat Z} at the
#'   point-estimate theta after the chain (seeds continue the chain's
#'   sequence); their sd is the likelihood-estimator noise (\code{sd_logZ}).
#'   0 disables.
#' @param adapt,target_accept proposal-scale adaptation: \code{"burnin"}
#'   rescales every \code{prop_sd} by one common factor, updated after each
#'   burn-in iteration by a Robbins--Monro step towards \code{target_accept}
#'   and frozen afterwards; \code{"none"} keeps \code{prop_sd}.
#' @param xnew,ynew optional prediction inputs/responses.
#' @param prediction plug-in at the point estimate, or MCMC averaging of all
#'   retained joint states. The latter runs inside the compiled chain.
#' @param point_estimate coordinatewise natural-scale median (default) or mean.
#' @param resampling,temper_alpha,seed,ncores as in \code{.restree_smc()}
#'   (defaults \code{stratified}, \code{1}, \code{42}, \code{1}).
#' @param verbose show a progress bar over the MCMC iterations.
#' @return list with \code{chain} (n_iter x (2 + nr), log scale),
#'   \code{chain_prop}, \code{logZ_prop} and \code{accepted} (the proposed
#'   theta, its evidence estimate and the accept indicator per iteration),
#'   \code{theta_hat} (posterior medians or means on the natural scale after burn-in),
#'   \code{accept_rate}, \code{time_s}, \code{interrupted}, and \code{fit}
#'   at \code{theta_hat}. The chain can be interrupted from R (Ctrl-C or
#'   Esc): the completed iterations are returned as a valid shorter run,
#'   with a warning and \code{interrupted = TRUE}; the burn-in is reduced
#'   to fit if necessary. The engine also honors interrupts between tree
#'   levels inside each evidence evaluation.
#' @keywords internal
#' @noRd
.restree_pmmh <- function(model, theta0, nparticles = NULL,
                         n_iter = 250L, burnin = 50L,
                         prop_sd = 0.15, prior = NULL,
                         sd_reps = 10L, sig2_moves = 3L,
                         adapt = "burnin", target_accept = 0.15,
                         xnew = NULL, ynew = NULL,
                         resampling = "stratified", temper_alpha = 1,
                         seed = 42L, ncores = 1L, verbose = FALSE,
                         prediction = "plugin", point_estimate = "median") {
  ## The Metropolis loop runs in the compiled layer (smc_engine_pmmh_cpp,
  ## src/PMMH.cpp): proposals, acceptance tests, seed sequence and scale
  ## adaptation are the same arithmetic on the same R random numbers as the
  ## R loop it replaced, so a chain is bit-identical to one from that loop
  ## at the same seed.  R keeps the argument handling, the prior set-up, the
  ## post-processing (parameter summaries, the sd_reps replicates, the final
  ## fit) and the interrupt bookkeeping.  verbose: one progress-bar update
  ## per iteration, printed by the compiled loop.
  model <- .restree_check_model(model, smc = TRUE)
  p <- ncol(model@x)
  if (is.list(theta0) && is.null(theta0$sig2)) theta0$sig2 <- stats::var(model@y)
  fx <- .restree_theta_for(theta0, p, "theta0")
  if (is.null(nparticles))
    nparticles <- .restree_model_nparticles(model)
  nparticles <- .restree_integer(nparticles, "nparticles", 1L)
  verbose <- .restree_flag(verbose, "verbose")
  n_iter <- .restree_integer(n_iter, "n_iter", 2L)
  burnin <- .restree_integer(burnin, "burnin", 0L, n_iter - 1L)
  sd_reps <- .restree_integer(sd_reps, "sd_reps", 0L)
  sig2_moves <- .restree_integer(sig2_moves, "sig2_moves", 0L)
  seed <- .restree_integer(seed, "seed", 0L)
  ncores <- .restree_ncores(ncores)
  opts <- .restree_smc_options(temper_alpha, resampling)
  nr <- length(fx@range)
  nd <- .restree_xnew(xnew, ynew, p)
  prediction <- .restree_choice(prediction, c("plugin", "mcmc"), "prediction")
  point_estimate <- .restree_choice(point_estimate, c("median", "mean"), "point_estimate")
  nq <- 2L + nr
  p0 <- c(log(fx@sig2), log(fx@range), log(fx@nugget))
  if (!is.numeric(prop_sd) || !length(prop_sd) %in% c(1L, nq) ||
      any(!is.finite(prop_sd)) || any(prop_sd <= 0))
    stop("prop_sd must be finite and positive, length 1 or ", nq, call. = FALSE)
  prop_sd <- rep_len(as.numeric(prop_sd), nq)
  prior <- .restree_prior_control(prior, nr, model)
  ## The persistent engine of this chain (one copy of the data, the
  ## theta-independent path geometry kept across evaluations; every value
  ## bit-identical to restree_loglik() at the same arguments). Only root-only
  ## models bypass the engine, including every depth-zero WN model.
  partial_error <- function(e, phase, mc = NULL) {
    count <- if (is.null(mc)) 0L else as.integer(mc$it_done)
    take <- seq_len(count)
    empty <- matrix(numeric(), 0L, nq)
    ch <- if (is.null(mc)) empty else mc$chain[take, , drop = FALSE]
    cp <- if (is.null(mc)) empty else mc$chain_prop[take, , drop = FALSE]
    message <- if (inherits(e, "condition")) conditionMessage(e) else as.character(e)
    stop(structure(list(
      message = sprintf("PMMH stopped during %s after %d of %d iterations: %s",
                        phase, count, n_iter, message),
      call = NULL, phase = phase, n_done = count, n_requested = n_iter,
      chain = .restree_public_chain(ch, fx),
      chain_prop = .restree_public_chain(cp, fx),
      chain_scale = "natural", parameter_form = fx@form,
      burnin = min(burnin, max(0L, count - 1L)), seed = seed,
      logZ_trace = if (is.null(mc)) numeric() else mc$logZ_trace[take],
      logZ_prop = if (is.null(mc)) numeric() else mc$logZ_prop[take],
      accepted = if (is.null(mc)) logical() else mc$accepted[take],
      proposal_scale_trace = if (is.null(mc)) numeric() else mc$scale_trace[take],
      trees = if (is.null(mc)) NULL else mc$trees,
      prediction = if (is.null(mc)) NULL else mc$prediction,
      exact_resume = FALSE),
      class = c("restree_pmmh_error", "error", "condition")))
  }
  engine <- tryCatch(if (.restree_root_only(model)) NULL else
    smc_engine_new_cpp(model, nparticles, opts$resampling, opts$temper_alpha, ncores),
    error = function(e) partial_error(e, "initialization"))
  ## Release the chain's engine on every exit, including an evaluation error (an
  ## OOM at large n, or a WN split beyond the depth ceiling). The external
  ## pointer's finalizer would free the engine at some later collection, but R
  ## does not account for the native allocation behind it, so that collection
  ## can be arbitrarily late; this makes the release deterministic. The success
  ## path releases early (below). smc_engine_release_cpp is idempotent, and the
  ## .Random.seed on.exit below also uses add = TRUE.
  on.exit(if (!is.null(engine)) smc_engine_release_cpp(engine), add = TRUE)
  smc <- list(temper_alpha = opts$temper_alpha, resampling = opts$resampling,
              seed = seed, ncores = ncores, engine = engine)
  Zof <- function(p, it_seed) {
    smc$seed <- it_seed
    .restree_log_evidence(model, fx, nparticles, exp(p[1L]),
                          exp(p[2L:(1L + nr)]), exp(p[2L + nr]), smc)
  }
  ## Blocked moves.  sigma^2 is a pure scale of every Gaussian block, so the
  ## persistent engine re-evaluates a sig2-only proposal from its retained
  ## block statistics (SMC::refresh_sig2_values) at a small fraction of
  ## a full evaluation; a (range, nugget) proposal is a full evaluation.
  ## Each iteration therefore runs sig2_moves random-walk moves on log
  ## sigma^2 alone, then one joint random-walk move of all of theta.  Every
  ## evaluation draws a fresh SMC seed, as PMMH requires; the trie of the
  ## current state is saved before the joint proposal and restored after a
  ## rejection (SMC::save_trie / restore_trie), so the next sig2 block
  ## reuses it.  Metropolis-within-Gibbs over blocks of theta with fresh
  ## unbiased evidence estimates leaves the extended pseudo-marginal target
  ## invariant.  Random-cut samplers keep no reusable state: sig2_moves is 0
  ## there and the move is the joint one.
  blocked <- !is.null(engine) && model@cut_method %in% c("middle", "median")
  if (!blocked) sig2_moves <- 0L
  ## Proposal-scale adaptation (adapt = "burnin"): one common factor
  ## exp(log_scale) on every prop_sd, updated after each burn-in iteration
  ## by a Robbins-Monro step towards target_accept on the joint move's
  ## acceptance indicator (step it^-0.6; Andrieu & Thoms 2008, Algorithm 4)
  ## and frozen after the burn-in, so the retained chain is drawn by a fixed
  ## kernel and the pseudo-marginal target stays invariant.  The default
  ## target 0.15 is in the range the pseudo-marginal literature recommends
  ## for noisy evidence estimates (Sherlock, Thiery, Roberts & Rosenthal 2015).
  ##
  ## The chain's own random-walk draws use R's RNG, seeded here and restored
  ## on exit so the caller's stream is untouched; the compiled loop draws
  ## from that stream (norm_rand / unif_rand) in the order the R loop did.
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    old_seed <- get(".Random.seed", envir = globalenv())
    on.exit(assign(".Random.seed", old_seed, envir = globalenv()), add = TRUE)
  }
  set.seed(seed)
  t0 <- proc.time()[3]
  ## The chain is interruptible from R (Ctrl-C / Esc): the engine checks for
  ## interrupts between tree levels, the compiled loop ends at the first one
  ## and returns the completed iterations, kept below as a valid shorter run.
  mc <- tryCatch(smc_engine_pmmh_cpp(engine, model, fx, p0, n_iter, burnin, prop_sd,
                            sig2_moves, blocked, .restree_prior_for_cpp(prior, nr),
                            adapt, target_accept, seed, ncores, verbose,
                            if (nd$supplied && prediction == "mcmc") nd$x else NULL,
                            if (prediction == "mcmc") nd$y else NULL, point_estimate),
                 error = function(e) partial_error(e, "initialization"))
  chain <- mc$chain; zs <- mc$logZ_trace
  ## proposal record: the proposed theta, its evidence estimate and whether
  ## it was accepted, per iteration -- the accepted-only chain cannot show
  ## whether rejections come from the proposal scale or from evidence noise
  chain_prop <- mc$chain_prop; zs_prop <- mc$logZ_prop; accepted <- mc$accepted
  scale_trace <- mc$scale_trace
  acc <- mc$acc; acc_sig2 <- mc$acc_sig2; n_sig2 <- mc$n_sig2
  interrupted <- isTRUE(mc$interrupted); it_done <- mc$it_done
  ## An evidence evaluation that errors (a WN split beyond the depth ceiling,
  ## an allocation failure or a broken sampler invariant) is still an error --
  ## never a -Inf proposal -- but
  ## the iterations completed before it are not thrown away: the condition
  ## carries them (class restree_pmmh_error; fields chain, logZ_trace,
  ## chain_prop, logZ_prop, accepted, n_done), e.g.
  ## tryCatch(restree_fit(m, th, method = "pmmh"), error = function(e) e$chain).
  if (is.character(mc$error) && nzchar(mc$error)) {
    partial_error(mc$error, "sampling", mc)
  }
  log_scale <- mc$log_scale
  step_sd <- function() prop_sd * exp(log_scale)
  ## the derived-seed sequence continues past the chain for the replicates
  seed_counter <- as.integer(mc$seed_counter)
  next_seed <- function() { seed_counter <<- seed_counter + 1L; .restree_seed_add(seed, seed_counter) }
  if (interrupted) {
    if (it_done < 2L)
      stop("PMMH interrupted before two iterations completed; nothing usable to return",
           call. = FALSE)
    warning(sprintf(paste0("PMMH interrupted after %d of %d iterations; ",
        "returning the completed part of the chain"), it_done, n_iter),
        call. = FALSE, immediate. = TRUE)
    n_iter <- it_done
    chain <- chain[seq_len(it_done), , drop = FALSE]
    zs <- zs[seq_len(it_done)]
    chain_prop <- chain_prop[seq_len(it_done), , drop = FALSE]
    zs_prop <- zs_prop[seq_len(it_done)]
    accepted <- accepted[seq_len(it_done)]
    scale_trace <- scale_trace[seq_len(it_done)]
    acc <- min(acc, it_done)
    if (burnin >= n_iter) burnin <- n_iter - 1L
  }
  el <- proc.time()[3] - t0
  th <- fx
  th@sig2 <- mc$theta_hat[1L]
  th@range <- as.numeric(mc$theta_hat[2L:(1L + nr)])
  th@nugget <- mc$theta_hat[2L + nr]
  ## The likelihood-estimator noise that governs PMMH mixing is
  ## sd(log Zhat(theta)) at a FIXED theta over independent SMC runs
  ## (Doucet, Pitt, Deligiannidis & Kohn 2015; Sherlock, Thiery, Roberts &
  ## Rosenthal 2015): sd_reps replicate evaluations at the point-estimate
  ## theta, each with its own seed (continuing the chain's seed sequence),
  ## on the warm engine handle -- each costs one PMMH iteration, so the
  ## default 10 is 10 / n_iter of the run.  The chain's own logZ_trace
  ## cannot give this number: it moves theta along the chain (posterior
  ## variation of log Z(theta)) and records accepted values only.
  logZ_reps <- numeric(0)
  if (sd_reps > 0L) {
    p_hat <- c(log(th@sig2), log(th@range), log(th@nugget))
    ## the first replicate is a full evaluation at p_hat, the rest reuse
    ## its trie (same theta, new seeds)
    logZ_reps <- tryCatch(
      vapply(seq_len(sd_reps), function(k) Zof(p_hat, next_seed()), numeric(1)),
      error = function(e) partial_error(e, "evidence_replicates", mc))
  }
  el_reps <- proc.time()[3] - t0 - el
  ## Release the chain's native data and caches before the final SMC.
  if (!is.null(engine)) smc_engine_release_cpp(engine)
  ## the final SMC at the point-estimate theta, with its distinct structures
  ## (structure_distinct) -- the fit object's structure slot
  fit <- tryCatch(.restree_smc_fit(model, th, nparticles,
    if (prediction == "plugin") nd$x else matrix(numeric(), 0L, p),
    if (prediction == "plugin") nd$y else NULL,
    opts$temper_alpha, opts$resampling, seed, ncores),
    error = function(e) partial_error(e, "final_fit", mc))
  list(chain = chain, logZ_trace = zs,
                 chain_prop = chain_prop, logZ_prop = zs_prop,
                 accepted = accepted,
                 logZ_reps = logZ_reps,
                 sd_logZ = if (sum(is.finite(logZ_reps)) >= 2L)
                   stats::sd(logZ_reps[is.finite(logZ_reps)]) else NA_real_,
                 sd_reps_time_s = as.numeric(el_reps),
                 theta_hat = list(sig2 = th@sig2, range = th@range, nugget = th@nugget),
                 theta = th, prior = prior,
                 accept_rate = acc / n_iter,
                 accept_rate_post = if (n_iter > burnin) mean(accepted[seq.int(burnin + 1L, n_iter)]) else NA_real_,
                 accept_rate_sig2 = if (n_sig2 > 0L) acc_sig2 / n_sig2 else NA_real_,
                 adapt = adapt, target_accept = target_accept,
                 prop_sd = prop_sd, prop_sd_final = step_sd(), prop_scale_trace = scale_trace,
                 sig2_moves = sig2_moves, n_evidence_evals = seed_counter + 1L,
                 time_s = as.numeric(el),
                 burnin = burnin, interrupted = interrupted,
                 trees = mc$trees,
                 prediction = if (prediction == "mcmc") mc$prediction else
                   if (nd$supplied) fit else NULL,
                 prediction_method = prediction, point_estimate = point_estimate,
                 fit = fit, method = "pmmh")
}
## One public scale/naming contract for successful and partial PMMH chains.
.restree_public_chain <- function(chain, theta) {
  chain <- exp(chain)
  colnames(chain) <- c("sig2", if (theta@form %in% c("ARD", "tensor"))
    paste0("range", seq_along(theta@range)) else "range", "nugget")
  chain
}
