##########################################################################
## Central validation for the public R interface.
##
## Primitive checkers used by the S4 constructors (S4classes.R) and the
## entry points.  Data / tree settings are validated once, when a
## restree_model is built; theta once, when a restree_theta is built.  The
## compiled routines re-check only structure with no R-level counterpart.
##########################################################################

.restree_choice <- function(x, choices, name) {
  if (!is.character(x) || length(x) != 1L || is.na(x) || !x %in% choices)
    stop(name, " must be one of: ", paste(choices, collapse = ", "), call. = FALSE)
  x
}

.restree_number <- function(x, name, lower = -Inf, upper = Inf,
                            lower_closed = TRUE, upper_closed = TRUE) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x))
    stop(name, " must be one finite numeric value", call. = FALSE)
  low_bad <- if (lower_closed) x < lower else x <= lower
  high_bad <- if (upper_closed) x > upper else x >= upper
  if (low_bad || high_bad) {
    left <- if (lower_closed) "[" else "("
    right <- if (upper_closed) "]" else ")"
    stop(name, " must lie in ", left, lower, ", ", upper, right, call. = FALSE)
  }
  as.numeric(x)
}

.restree_integer <- function(x, name, lower = -.Machine$integer.max,
                             upper = .Machine$integer.max) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x != floor(x) ||
      x < lower || x > upper)
    stop(name, " must be one integer in [", lower, ", ", upper, "]", call. = FALSE)
  as.integer(x)
}

## Overflow-safe seed arithmetic: (seed + k) mod 2^31, in
## [0, .Machine$integer.max].  Every derived-seed site uses this helper --
## plain integer addition overflows to NA in R.  Exact for |seed + k| < 2^53.
.restree_seed_add <- function(seed, k) {
  as.integer((as.numeric(seed) + as.numeric(k)) %% 2^31)
}

## One native resolver supplies R and C++ entry points with the same physical/
## scheduler/backend caps. ncores=1 selects serial tree traversal with bounded
## Eigen GEMM parallelism; no-OpenMP builds use one thread throughout. Workspace
## budgets may limit concurrent work; they are not fit-rejecting memory preflights.
## Fitted-state integrity. A fit's
## predictive distribution is determined by its data, model settings, theta,
## inference method, stored tree distribution and sampler settings (and, for
## PMMH, the retained chain).  .restree_new() records a byte-exact FNV-1a
## fingerprint of exactly those (fingerprint_cpp; no serialization header,
## so it survives R versions) in results$.fitted_state, and
## .restree_check_fit() recomputes it before any prediction: an object whose
## inference state was edited after fitting -- or a legacy fit without the
## record -- is refused with a "refit" error instead of being rescored with
## tree weights that belong to other parameters.  Annotations elsewhere in
## results, the native engine state and the prediction slot are not covered.
.restree_fitted_state <- function(object) {
  th <- object@theta; st <- object@structure; nodes <- st$nodes
  res <- object@results; mc <- object@mcmc; tr <- object@tree
  num <- list(object@x, object@y,
           object@depth, object@r, object@prior_rho, object@prior_beta, object@seed,
           object@nested_factor, object@cut_candidates,
           th@sig2, th@range, th@nugget, th@nu, th@tail,
           tr$S, tr$J, tr$cuts, tr$id, tr$node_offset, tr$node_size,
           tr$node_split, tr$rows, tr$knot_offset, tr$knot_size, tr$knots,
           nodes$tree, nodes$id, nodes$split, nodes$J, nodes$cut,
           st$w, st$depth, res$nparticles, res$temper_alpha, res$seed,
           mc$chain, mc$prediction_index,
           res$leaf_mle$mean, res$leaf_mle$variance)
  # Saved PMMH particle geometry and its iteration mapping determine later
  # prediction. Cover every numeric field, including offsets and cut bits.
  if (!is.null(mc$trees)) num <- c(num, unname(mc$trees))
  chr <- c(object@leaf_model, object@design, object@cut_method, object@method,
           th@form, th@family, th@dtype, as.character(res$resampling),
           as.character(res$leaf_mle$family),
           if (object@leaf_model == "WhiteNoise")
             if (object@depth == 0L) "wn-root-only-depth-zero-v1" else "wn-adaptive-nodepth-v2")
  if (!is.null(mc$prediction_method))
    chr <- c(chr, mc$prediction_method, mc$point_estimate)
  fingerprint_cpp(num, chr)
}

.restree_verify_fitted_state <- function(object) {
  fs <- object@results$.fitted_state
  if (is.null(fs) || !identical(fs, .restree_fitted_state(object)))
    stop("the fitted object's data, model settings, theta, method, tree distribution or ",
         "sampler settings differ from those it was fitted with (or the fit predates ",
         "fitted-state records): its stored tree weights are not valid for them -- ",
         "build a new model and refit", call. = FALSE)
  leaf <- object@results$leaf_mle
  if (!is.null(leaf) && (!identical(leaf$family, "gaussian_wn") ||
      object@leaf_model != "WhiteNoise" || object@depth != 0L ||
      !is.numeric(leaf$mean) || length(leaf$mean) != 1L || !is.finite(leaf$mean) ||
      !is.numeric(leaf$variance) || length(leaf$variance) != 1L ||
      !identical(as.numeric(leaf$variance), as.numeric(object@theta@sig2))))
    stop("the fitted root-WN parameters are inconsistent: build a new model and refit",
         call. = FALSE)
  invisible(TRUE)
}

## NULL, zero, negative, NA, infinite and non-integer values are errors.
.restree_thread_policy <- function(ncores, warn = TRUE) {
  ncores <- .restree_integer(ncores, "ncores", 1L)
  thread_policy_cpp(ncores, warn)
}
.restree_ncores <- function(ncores) .restree_thread_policy(ncores)$ncores

.restree_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x))
    stop(name, " must be TRUE or FALSE", call. = FALSE)
  x
}

.restree_matrix <- function(x, name, ncol_expected = NULL, allow_empty = FALSE) {
  x <- as.matrix(x)
  if (length(dim(x)) != 2L || !typeof(x) %in% c("double", "integer"))
    stop(name, " must be a numeric matrix", call. = FALSE)
  if (ncol(x) < 1L || (!allow_empty && nrow(x) < 1L))
    stop(name, " must have at least one column",
         if (!allow_empty) " and one row" else "", call. = FALSE)
  if (!is.null(ncol_expected) && ncol(x) != ncol_expected)
    stop(name, " must have ", ncol_expected, " columns", call. = FALSE)
  if (any(!is.finite(x))) stop(name, " must contain only finite values", call. = FALSE)
  storage.mode(x) <- "double"
  x
}

.restree_vector <- function(x, name, length_expected = NULL, allow_empty = FALSE) {
  if (!is.numeric(x) || (!is.null(dim(x)) &&
      !(is.matrix(x) && ncol(x) == 1L)))
    stop(name, " must be a numeric vector", call. = FALSE)
  x <- as.numeric(x)
  if (!allow_empty && !length(x)) stop(name, " must not be empty", call. = FALSE)
  if (!is.null(length_expected) && length(x) != length_expected)
    stop(name, " must have length ", length_expected, call. = FALSE)
  if (any(!is.finite(x))) stop(name, " must contain only finite values", call. = FALSE)
  x
}

## The engine's root box is hardcoded to [0,1]^p, so inputs outside it
## produce degenerate partitions: membership is validated here (xnew) and in
## the restree_model validity method (x).
.restree_unit_cube <- function(x, name) {
  if (any(x < 0) || any(x > 1))
    stop(name, " must lie in the unit hypercube: every coordinate in [0, 1] ",
         "(the tree partition is defined on it). Rescale each column, e.g. ",
         "(x - min) / (max - min)", call. = FALSE)
  invisible(TRUE)
}

.restree_xnew <- function(xnew, ynew, p, allow_null = TRUE) {
  if (is.null(xnew)) {
    if (!is.null(ynew)) stop("ynew requires xnew", call. = FALSE)
    if (!allow_null) stop("xnew must be supplied", call. = FALSE)
    return(list(x = matrix(numeric(), 0L, p), y = NULL, supplied = FALSE))
  }
  xn <- .restree_matrix(xnew, "xnew", p)
  .restree_unit_cube(xn, "xnew")
  yn <- if (is.null(ynew)) NULL else .restree_vector(ynew, "ynew", nrow(xn))
  list(x = xn, y = yn, supplied = TRUE)
}

.restree_check_control_names <- function(control, method) {
  if (!is.list(control)) stop("control must be a list", call. = FALSE)
  allowed <- switch(method,
    mle = c("maxit", "reltol", "optimizer", "n_starts", "range_bounds", "nugget_bounds"),
    smc = character(),
    ebayes = c("maxit", "tol", "prior", "start", "sd_reps"),
    pmmh = c("prop_sd", "prior", "burnin", "predict_draws", "sd_reps", "sig2_moves",
             "adapt", "target_accept", "prediction", "point_estimate"))
  if (length(control) && (is.null(names(control)) || any(!nzchar(names(control)))))
    stop("every control entry must be named", call. = FALSE)
  duplicate <- unique(names(control)[duplicated(names(control))])
  if (length(duplicate))
    stop("duplicated control entry: ", paste(duplicate, collapse = ", "), call. = FALSE)
  bad <- setdiff(names(control), c(allowed, "joint_lpd"))
  if (length(bad))
    stop("unknown control entry for method = '", method, "': ",
         paste(bad, collapse = ", "), call. = FALSE)
  control
}

.restree_bounds <- function(x, name) {
  if (!is.numeric(x) || length(x) != 2L || any(!is.finite(x)) || any(x <= 0) ||
      x[1L] >= x[2L])
    stop(name, " must be c(lower, upper) with 0 < lower < upper", call. = FALSE)
  as.numeric(x)
}

## PMMH control vectors are ordered (sig2, range..., nugget): a scalar is
## recycled; for ARD a length-3 vector (sig2, shared range, nugget) is
## expanded; NULL keeps the fallback.
.restree_pmmh_control <- function(x, nr, name, fallback, positive = FALSE) {
  nq <- nr + 2L
  if (is.null(x)) return(fallback)
  if (!is.numeric(x) || !length(x) || any(!is.finite(x)))
    stop("control$", name, " must contain finite numeric values", call. = FALSE)
  x <- as.numeric(x)
  if (length(x) == 1L) x <- rep(x, nq)
  else if (length(x) == nq) x <- x
  else if (nr > 1L && length(x) == 3L) x <- c(x[1L], rep(x[2L], nr), x[3L])
  else stop("control$", name, " must have length 1, ",
            if (nr > 1L) "3, or " else "or ", nq,
            " (ordered sig2, range..., nugget)", call. = FALSE)
  if (positive && any(x <= 0))
    stop("control$", name, " must contain positive values", call. = FALSE)
  x
}

.restree_controls <- function(control, method, n_iter, theta, model = NULL) {
  control <- .restree_check_control_names(control, method)
  control$joint_lpd <- if (is.null(control$joint_lpd)) FALSE else
    .restree_flag(control$joint_lpd, "control$joint_lpd")
  if (method %in% c("ebayes", "pmmh"))
    control$prior <- .restree_prior_control(control$prior, length(theta@range), model,
                                            allow_none = (method == "ebayes"))
  if (method == "mle") {
    control$maxit <- .restree_integer(if (is.null(control$maxit)) 100L else control$maxit,
                                      "control$maxit", 1L)
    control$reltol <- .restree_number(if (is.null(control$reltol)) 1e-6 else control$reltol,
                                     "control$reltol", 0, lower_closed = FALSE)
    control$optimizer <- .restree_choice(
      if (is.null(control$optimizer)) "L-BFGS-B" else control$optimizer,
      c("L-BFGS-B", "Nelder-Mead"), "control$optimizer")
    control$n_starts <- .restree_integer(
      if (is.null(control$n_starts)) 5L else control$n_starts,
      "control$n_starts", 1L)
    ## the MLE search box on the natural scale: c(lower, upper), both
    ## positive, lower < upper (L-BFGS-B bounds; every start is clamped)
    control$range_bounds <- .restree_bounds(
      if (is.null(control$range_bounds)) c(1e-4, 10) else control$range_bounds,
      "control$range_bounds")
    control$nugget_bounds <- .restree_bounds(
      if (is.null(control$nugget_bounds)) c(1e-8, 2) else control$nugget_bounds,
      "control$nugget_bounds")
  } else if (method == "ebayes") {
    control$maxit <- .restree_integer(if (is.null(control$maxit)) 100L else control$maxit,
                                      "control$maxit", 1L)
    control$tol <- .restree_number(if (is.null(control$tol)) 1e-5 else control$tol,
                                   "control$tol", 0, lower_closed = FALSE)
    ## starting point of the search: "theta" (default) = the supplied theta or
    ## the package defaults; "auto" = the Full-tree MLE when no theta (or no
    ## range) was supplied, else theta; "full_mle" = always the Full-tree MLE
    control$start <- .restree_choice(if (is.null(control$start)) "theta" else control$start,
                                     c("theta", "auto", "full_mle"), "control$start")
    ## replicate evidence evaluations at the estimate (fresh seeds): the
    ## estimator noise the CRN search cannot show; 0 disables
    control$sd_reps <- .restree_integer(
      if (is.null(control$sd_reps)) 5L else control$sd_reps, "control$sd_reps", 0L)
  } else if (method == "pmmh") {
    burn <- if (is.null(control$burnin)) floor(n_iter / 2) else control$burnin
    control$burnin <- .restree_integer(burn, "control$burnin", 0L, n_iter - 1L)
    control$prediction <- .restree_choice(
      if (is.null(control$prediction)) "plugin" else control$prediction,
      c("plugin", "mcmc"), "control$prediction")
    control$point_estimate <- .restree_choice(
      if (is.null(control$point_estimate)) "median" else control$point_estimate,
      c("median", "mean"), "control$point_estimate")
    if (!is.null(control$predict_draws)) {
      control$predict_draws <- .restree_integer(control$predict_draws,
                                                 "control$predict_draws", 1L)
      warning("control$predict_draws is deprecated and ignored; MCMC prediction uses every post-burn-in state",
              call. = FALSE)
    }
    control$sd_reps <- .restree_integer(
      if (is.null(control$sd_reps)) 10L else control$sd_reps, "control$sd_reps", 0L)
    control$sig2_moves <- .restree_integer(
      if (is.null(control$sig2_moves)) 3L else control$sig2_moves, "control$sig2_moves", 0L)
    ## proposal-scale adaptation: "burnin" (default) rescales the random-walk
    ## step during the burn-in only (the kernel is fixed afterwards, so the
    ## retained chain is a valid MCMC sample); "none" keeps prop_sd as given
    control$adapt <- .restree_choice(if (is.null(control$adapt)) "burnin" else control$adapt,
                                     c("burnin", "none"), "control$adapt")
    control$target_accept <- .restree_number(
      if (is.null(control$target_accept)) 0.15 else control$target_accept,
      "control$target_accept", 0, 1, lower_closed = FALSE, upper_closed = FALSE)
    nr <- length(theta@range)
    control$prop_sd <- .restree_pmmh_control(
      control$prop_sd, nr, "prop_sd", rep(0.15, nr + 2L), positive = TRUE)
  }
  control
}

## ---------------------------------------------------------------------------
## Prior on the covariance parameters, shared by ebayes (posterior mode) and
## pmmh (posterior sampling):
##   sig2      ~ InverseGamma(shape, rate):  p(s) = rate^shape / Gamma(shape) *
##               s^-(shape+1) * exp(-rate / s)
##   range_i   ~ half-Cauchy(scale):  p(r) = 2 / (pi scale) / (1 + (r/scale)^2)
##   nugget    ~ half-Cauchy(scale)
## Defaults: shape 2, rate var(y) (prior mean of sig2 equal to the sample
## variance, mode var(y)/3, right tail proportional to sig2^-3), scale 1 for
## every range and for the nugget (the density 1/(1 + theta^2) on theta > 0).
## Densities are on the NATURAL scale; .restree_log_prior_u adds the
## Jacobian for a sampler that moves in u = log theta.
## `prior = "none"` (or FALSE) selects NO prior: ebayes then maximises the
## evidence alone (the maximum marginal-likelihood estimate); pmmh refuses
## it, since a flat prior on theta is improper along the evidence ridge.
## ---------------------------------------------------------------------------
.restree_prior_control <- function(prior, nr, model, allow_none = TRUE) {
  if (identical(prior, "none") || identical(prior, FALSE)) {
    if (!allow_none)
      stop("control$prior = \"none\" is not allowed here: pmmh needs a proper prior on theta",
           call. = FALSE)
    return(NULL)
  }
  if (is.list(prior) && isTRUE(prior$resolved)) return(prior)   # already validated
  y <- if (is.null(model)) NULL else model@y
  if (is.null(prior)) prior <- list()
  if (!is.list(prior)) stop("control$prior must be a list or \"none\"", call. = FALSE)
  bad <- setdiff(names(prior), c("sig2", "range", "nugget"))
  if (length(bad))
    stop("unknown control$prior entries: ", paste(bad, collapse = ", "),
         " (allowed: sig2, range, nugget)", call. = FALSE)
  s2 <- prior$sig2
  if (is.null(s2)) s2 <- c(shape = 2, rate = NA_real_)
  if (is.list(s2)) {
    ## mode-and-concentration form: mode m (default var(y)), and the shape
    ## solving P(m / factor <= sig2 <= factor * m) = mass; rate = m (shape + 1)
    bad <- setdiff(names(s2), c("mode", "factor", "mass"))
    if (length(bad))
      stop("unknown control$prior$sig2 entries: ", paste(bad, collapse = ", "),
           " (allowed: mode, factor, mass)", call. = FALSE)
    m <- if (is.null(s2$mode) || is.na(s2$mode)) {
      if (is.null(y)) stop("control$prior$sig2 mode is required here", call. = FALSE)
      stats::var(as.numeric(y)) } else s2$mode
    f <- if (is.null(s2$factor)) 2 else s2$factor
    mass <- if (is.null(s2$mass)) 0.9 else s2$mass
    if (!is.numeric(m) || length(m) != 1L || !is.finite(m) || m <= 0)
      stop("control$prior$sig2 mode must be positive", call. = FALSE)
    if (!is.numeric(f) || length(f) != 1L || !is.finite(f) || f <= 1)
      stop("control$prior$sig2 factor must be greater than 1", call. = FALSE)
    if (!is.numeric(mass) || length(mass) != 1L || !is.finite(mass) || mass <= 0 || mass >= 1)
      stop("control$prior$sig2 mass must be in (0, 1)", call. = FALSE)
    a <- .restree_ig_shape_for_mode(f, mass)
    s2 <- c(shape = a, rate = m * (a + 1))
  }
  if (!is.numeric(s2) || length(s2) != 2L)
    stop("control$prior$sig2 must be c(shape, rate) of the inverse-gamma prior, or list(mode, factor, mass)",
         call. = FALSE)
  s2 <- as.numeric(s2); names(s2) <- c("shape", "rate")
  if (!is.finite(s2[["shape"]]) || s2[["shape"]] <= 0)
    stop("control$prior$sig2 shape must be positive", call. = FALSE)
  if (is.na(s2[["rate"]])) {
    if (is.null(y)) stop("control$prior$sig2 rate is required here", call. = FALSE)
    s2[["rate"]] <- stats::var(as.numeric(y))
  }
  if (!is.finite(s2[["rate"]]) || s2[["rate"]] <= 0)
    stop("control$prior$sig2 rate must be positive", call. = FALSE)
  rs <- prior$range
  if (is.null(rs)) rs <- 1
  if (!is.numeric(rs) || !length(rs) %in% c(1L, nr) || any(!is.finite(rs)) || any(rs <= 0))
    stop("control$prior$range must be a positive half-Cauchy scale, length 1 or ", nr, call. = FALSE)
  rs <- rep_len(as.numeric(rs), nr)
  ns <- prior$nugget
  if (is.null(ns)) ns <- 1
  if (!is.numeric(ns) || length(ns) != 1L || !is.finite(ns) || ns <= 0)
    stop("control$prior$nugget must be a positive half-Cauchy scale", call. = FALSE)
  list(sig2 = s2, range = rs, nugget = as.numeric(ns), resolved = TRUE)
}

## Shape a of an inverse-gamma with FIXED mode m (rate = m (a + 1)) such that
## P(m / f <= X <= f m) = mass.  X / m ~ IG(a, a + 1) whatever m is, and
## P(X <= x m) = P(1/X >= 1/(x m)) = pgamma(1/x, a, rate = a + 1, lower = FALSE),
## so the mass in [m/f, f m] is pgamma(f, a, rate = a+1) - pgamma(1/f, a, rate = a+1),
## increasing in a: one root in log a.
.restree_ig_shape_for_mode <- function(f, mass) {
  g <- function(la) { a <- exp(la); stats::pgamma(f, a, rate = a + 1) - stats::pgamma(1 / f, a, rate = a + 1) - mass }
  exp(stats::uniroot(g, c(-8, 14), tol = 1e-10)$root)
}

## log prior density of (sig2, range, nugget) on the natural scale
.restree_log_prior <- function(sig2, range, nugget, prior) {
  if (any(!is.finite(c(sig2, range, nugget))) || sig2 <= 0 || any(range <= 0) || nugget <= 0)
    return(-Inf)
  if (is.null(prior)) return(0)   # no prior: the objective is the evidence alone
  a <- prior$sig2[["shape"]]; b <- prior$sig2[["rate"]]
  lp_sig2 <- a * log(b) - lgamma(a) - (a + 1) * log(sig2) - b / sig2
  lp_range <- sum(log(2 / pi) - log(prior$range) - log1p((range / prior$range)^2))
  lp_nugget <- log(2 / pi) - log(prior$nugget) - log1p((nugget / prior$nugget)^2)
  lp_sig2 + lp_range + lp_nugget
}

## the same prior for a sampler on u = (log sig2, log range, log nugget):
## p_u(u) = p_theta(exp(u)) * exp(sum(u))
.restree_log_prior_u <- function(u, nr, prior) {
  lp <- .restree_log_prior(exp(u[1L]), exp(u[1L + seq_len(nr)]), exp(u[2L + nr]), prior)
  if (is.null(prior)) lp else lp + sum(u)
}

## The same prior as the compiled PMMH loop evaluates it (LogPriorU in
## src/PMMH.cpp, term for term .restree_log_prior_u): the inverse-gamma
## constant a log(b) - lgamma(a) is evaluated here, by R's own lgamma, so the
## compiled value equals the R value to the last bit.
.restree_prior_for_cpp <- function(prior, nr) {
  if (is.null(prior)) return(list(proper = FALSE))
  a <- prior$sig2[["shape"]]; b <- prior$sig2[["rate"]]
  list(proper = TRUE, shape = a, rate = b, lp_sig2_const = a * log(b) - lgamma(a),
       range_scale = rep_len(as.numeric(prior$range), nr),
       nugget_scale = as.numeric(prior$nugget))
}

# Depth zero specifies a single PP/WN leaf, irrespective of the knot budget.
.restree_root_only <- function(model) {
  model@depth == 0L
}

.restree_model_nparticles <- function(model) {
  .restree_default_nparticles(model@depth)
}

# The model's depth slot is the initial depth (the PP cap; for WN only the
# initial comparison depth, which imposes no boundary). The saved tree depth
# is kept separately as the representation depth of the stored trees, so
# prediction/PMMH never feed a run's deepest realised level back into the
# model. A realized WN tree may be deeper than the initial depth (up to
# .restree_max_wn_depth()); PP never exceeds its model depth, and Full's
# stored depth is its own size-rule depth, so the bound is exact for them.
.restree_stored_depth <- function(st, initial_depth) {
  if (is.null(st$depth)) initial_depth else
    .restree_integer(st$depth, "structure$depth", initial_depth, .restree_max_wn_depth())
}

## A restree_model (or the model part of a fit) ready for the engine; with
## smc = TRUE permits PP/WN, including the exact depth-zero terminal.
.restree_check_model <- function(model, smc = FALSE) {
  if (!methods::is(model, "restree_model"))
    stop("model must be a restree_model (see ?restree_model)", call. = FALSE)
  methods::validObject(model)
  model <- .restree_model_of(model)
  if (smc) {
    if (model@leaf_model == "Full")
      stop("leaf_model = 'Full' uses one fixed tree; the SMC sampler needs PP or WN",
           call. = FALSE)
  }
  model
}

## A fitted restree object ready for prediction: the class validity plus the
## method-specific fields prediction reads.
.restree_check_fit <- function(object) {
  .restree_require_fit(object)
  ## the fitted-state record first: an object whose inference state was edited
  ## is refused as edited ("refit") whatever else the edit broke -- validity
  ## would otherwise report, e.g., an invalid theta family or unnormalised
  ## weights and hide that the stored tree weights no longer belong to it
  .restree_verify_fitted_state(object)
  methods::validObject(object)
  p <- ncol(object@x)
  th <- .restree_theta_for(object@theta, p)
  method <- object@method
  if (method == "pmmh") {
    ## nr+2 columns: (sig2, range..., nugget) -- sigma^2 sampled jointly
    ch <- object$chain
    if (!is.matrix(ch) || !is.numeric(ch) || ncol(ch) != length(th@range) + 2L ||
        nrow(ch) < 1L || any(!is.finite(ch)))
      stop("fitted PMMH chain has an invalid shape or values", call. = FALSE)
    ix <- object$prediction_index
    if (!is.numeric(ix) || !length(ix) || any(!is.finite(ix)) ||
        any(ix != floor(ix)) || any(ix < 1L | ix > nrow(ch)))
      stop("fitted PMMH prediction_index is invalid", call. = FALSE)
    .restree_integer(object$nparticles, "object$nparticles", 1L)
    .restree_smc_options(object$temper_alpha, object$resampling)
  } else {
    ## Class validity already checks node records. Prediction additionally
    ## requires a nonempty structure.
    if (!all(c("nodes", "w") %in% names(object@structure)))
      stop("fitted object has incomplete prediction structure", call. = FALSE)
  }
  list(p = p, theta = th, method = method)
}

.restree_smc_options <- function(temper_alpha, resampling) {
  list(temper_alpha = .restree_number(temper_alpha, "temper_alpha", 0, 1,
                                      lower_closed = FALSE),
       resampling = .restree_choice(resampling,
         c("stratified", "multinomial"), "resampling"))
}

.restree_leaf_model <- function(x, name = "baseline") {
  ## User-facing canonical value is WN; WhiteNoise is the accepted alias.
  ## The INTERNAL canonical value stays "WhiteNoise" (the compiled layer and
  ## every stored-object comparison use it), so both spellings normalize to
  ## it here and only here.
  if (length(x) != 1L || is.na(x))
    stop(name, " must be one of: Full, PP, WN (WhiteNoise is an alias for WN)",
         call. = FALSE)
  x <- as.character(x)
  if (identical(x, "WN")) x <- "WhiteNoise"
  if (!x %in% c("Full", "PP", "WhiteNoise"))
    stop(name, " must be one of: Full, PP, WN (WhiteNoise is an alias for WN)",
         call. = FALSE)
  x
}

## Default SMC particle count when nparticles is NULL: 100 * the model's
## configured depth for both PP and WN. Depth zero uses one particle.
## WN may grow deeper, but its realized tree depth never changes this budget.
## Full uses its fixed tree without SMC. Tune the
## count against summary(fit)$sd_logZ of a pmmh fit (the estimator noise at
## the point-estimate theta).  Every R entry point (restree_fit for smc /
## ebayes / pmmh, .restree_smc, .restree_pmmh, restree_diagnostics) passes
## through .restree_model_nparticles; restree_loglik() applies the same rule in C++
## (src/wrapper.cpp, restree_default_particles).
.restree_default_nparticles <- function(depth) {
  if (depth == 0L) 1L else 100L * as.integer(depth)
}
