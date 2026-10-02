##########################################################################
## Proper scoring rules for ResTGP predictive distributions.
##
## Naming, deliberately: the pointwise quantity summed over test points is the
## log pointwise predictive density, lppd (Gelman, Hwang & Vehtari 2014, Stat
## Comput 24:997-1016; Vehtari, Gelman & Gabry 2017, Stat Comput 27:1413-1432).
## It is positively oriented and scales with the number of test points. The
## negatively oriented per-point average, -lppd/m, is the logarithmic score
## (Good 1952; Gneiting & Raftery 2007, JASA 102:359-378) and is the quantity
## comparable with CRPS. `restree_lppd()` returns both from one pass. The
## JOINT log predictive density
## log p(y* | y) is a different object -- it is not pointwise and not an average
## -- and is carried separately as `joint_lpd`, never mixed into either. Its
## parameter treatment follows the fitted estimator: Full/EB are plug-in, while
## PMMH averages the plug-in predictive over selected retained theta draws and trees.
##########################################################################


crps_gaussian <- function(y, mean, sd) {
  ## CRPS of N(mean, sd^2) at y, closed form (Gneiting & Raftery 2007, eq. 21):
  ##   sd * [ z(2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],  z = (y - mean)/sd
  stopifnot(all(sd > 0))
  z <- (y - mean) / sd
  sd * (z * (2 * stats::pnorm(z) - 1) + 2 * stats::dnorm(z) - 1 / sqrt(pi))
}

## The compiled core (src/scores.cpp, score_mixture_cpp): every mixture score
## below is one call with the component table (m x K means and variances,
## optional m x K Student-t df with Inf for Gaussian components, weights) and
## flags for what is wanted.  Every requested quantity is computed
## independently of the others from the same per-point component set (the
## CRPS, log score and cdf from y, the quantiles from probs alone), so one
## call with several flags returns exactly what separate calls would.
## Rmath and QUADPACK calls stay on the main R thread. Requests are validated
## consistently with other entry points, but do not create scoring workers.
.restree_score_core <- function(y, mean, var, df, w, probs = numeric(0), crps = FALSE, logscore = FALSE,
                                cdf = FALSE, control = list(), ncores = 1L) {
  rel <- if (is.null(control$rel_tol)) 1e-6 else
    .restree_number(control$rel_tol, "control$rel_tol", 0, 1, lower_closed = FALSE)
  subdivisions <- if (is.null(control$subdivisions)) 300L else
    .restree_integer(control$subdivisions, "control$subdivisions", 20L)
  mean <- as.matrix(mean); var <- as.matrix(var)
  if (!is.null(df)) df <- as.matrix(df)
  ncores <- .restree_ncores(ncores)
  score_mixture_cpp(as.numeric(y), mean, var, df, as.numeric(w), as.numeric(probs),
                    crps, logscore, cdf, rel, subdivisions, ncores)
}

crps_mixture <- function(y, mean, var, w = NULL, ncores = 1L) {
  ## CRPS of a Gaussian mixture sum_k w_k N(mean[,k], var[,k]) in closed form
  ## (Grimit, Gneiting, Berrocal & Johnson 2006, QJRMS 132:2925-2942):
  ##   CRPS = sum_k w_k A(y - m_k, s_k^2)
  ##          - 0.5 sum_k sum_l w_k w_l A(m_k - m_l, s_k^2 + s_l^2)
  ## Exact for the SMC particle mixture; no sampling.  Compiled (src/scores.cpp).
  mean <- as.matrix(mean); var <- as.matrix(var)
  K <- ncol(mean)
  if (is.null(w)) w <- rep(1 / K, K)
  stopifnot(length(y) == nrow(mean), dim(var) == dim(mean), length(w) == K)
  .restree_score_core(y, mean, var, NULL, w, crps = TRUE, ncores = ncores)$crps
}

logs_gaussian <- function(y, mean, sd) -stats::dnorm(y, mean, sd, log = TRUE)

logs_mixture <- function(y, mean, var, w = NULL, ncores = 1L) {
  ## pointwise -log sum_k w_k N(y | m_k, s_k^2), by log-sum-exp (compiled)
  mean <- as.matrix(mean); var <- as.matrix(var)
  K <- ncol(mean)
  if (is.null(w)) w <- rep(1 / K, K)
  stopifnot(length(y) == nrow(mean), dim(var) == dim(mean), length(w) == K)
  .restree_score_core(y, mean, var, NULL, w, logscore = TRUE, ncores = ncores)$logscore
}


## Student-t components retain variance for moments, but densities use SCALE.
## Inf degrees of freedom denotes a Gaussian (including empty-region fallback).
.restree_t_scale <- function(var, df)
  sqrt(var) * sqrt(ifelse(is.finite(df), (df-2)/df, 1))

## Student-t mixture (wn_leaf = "student"): pointwise -log density (compiled)
.restree_logs_tmix <- function(y, p, ncores = 1L)
  .restree_score_core(y, p$mean, p$var, p$df, p$w, logscore = TRUE, ncores = ncores)$logscore

.restree_component_cdf <- function(x, mu, sc, df, w, lower=TRUE) {
  z <- sweep(outer(x,mu,"-"),2L,sc,"/")
  probs <- matrix(stats::pt(z,df=rep(df,each=length(x)),lower.tail=lower),length(x))
  drop(probs %*% w)
}

## Student-t mixture CRPS (compiled): a single component in closed form, a
## genuine mixture by adaptive quadrature of the squared mixture cdf on the
## standardised scale -- the line split at the component locations (at most 32
## quantiles of them) and at y, each piece integrated by QUADPACK dqags / dqagi
## through R's own C entry points (the routines stats::integrate() uses) at
## rel.tol = control$rel_tol and abs.tol = rel.tol / pieces on that side, all
## cross-component terms included.  Returns the values and the summed
## integration errors; a failed integration raises stats::integrate()'s error.
.restree_crps_tmix <- function(y, p, control = list(), ncores = 1L) {
  z <- .restree_score_core(y, p$mean, p$var, p$df, p$w, crps = TRUE, control = control, ncores = ncores)
  list(value = z$crps, error = z$crps_abs_error)
}

## F_mix(y_i) = sum_k w_k F_k(y_i) at one y per row (compiled): the coverage
## probabilities of restree_score() without root finding.
.restree_mixture_cdf <- function(y, p, ncores = 1L)
  .restree_score_core(y, p$mean, p$var, p$df, p$w, cdf = TRUE, ncores = ncores)$cdf_y

## Quantiles of the mixture predictive, one row per prediction point (compiled).
## The mixture cdf has no closed-form inverse; F_mix(x) - p is nondecreasing and
## the quantile lies in [min_k q_ik, max_k q_ik] (q_ik the component quantiles),
## so a bracketed bisection on that interval converges without any sign
## condition: the bracket halves until its width is below 1e-8 * max_k s_ik
## or it can no longer shrink in double precision. Component parameters are
## validated by .restree_pred_parts().
.restree_predictive_quantiles <- function(p, probs, ncores = 1L) {
  if (p$kind == "gaussian") return(outer(p$mu, rep(1, length(probs))) + outer(p$sd, stats::qnorm(probs)))
  .restree_score_core(rep(0, nrow(as.matrix(p$mean))), p$mean, p$var, p$df, p$w, probs = probs,
                      ncores = ncores)$quantiles
}

## ---- object-level entry points ---------------------------------------------
## Each accepts a restree prediction object, its plain prediction list, or
## a fit carrying a prediction (exact mixture treatment), or plain
## mean/sd vectors (single Gaussian).

## WN leaf treatment of the CRPS (wn_leaf).  Every score of a WN prediction is
## an average over the tree particles of the component quantity: the log
## score is -log sum_k w_k f_k(y), coverage uses sum_k w_k F_k(y), the width the
## quantiles of sum_k w_k F_k, with f_k / F_k the integrated (Normal-Inverse-
## Gamma) leaf predictive -- Student-t with df = 2 a_n = 4 + n_leaf -- for
## every component; these are cheap (vectorised dt / pt) and are never
## approximated.  Only the CRPS of a Student-t MIXTURE has no closed form, and
## wn_leaf chooses how it is computed:
##   "plugin"  (default) the leaf variance is fixed at its NIG posterior mean
##             E[sigma^2 | leaf] = b_n / (a_n - 1), so every component becomes
##             Gaussian with the SAME mean and variance the prediction stores
##             (wn_leaf_moments in src/prediction.cpp stores exactly
##             b_n (kappa_n + 1) / ((a_n - 1) kappa_n)), the tree mixture is
##             exactly a Gaussian mixture, and its CRPS is the closed form of
##             Grimit et al. (2006). This changes the represented predictive
##             distribution used for CRPS; its error depends on the component
##             degrees of freedom, locations, scales, weights and test values.
##   "student" the Student-t mixture CRPS by adaptive quadrature of the squared
##             mixture cdf (.restree_crps_tmix).
## The scorer uses the component moments/df supplied by prediction. This
## does not establish that every represented component or joint density is
## an exact posterior predictive distribution of an underlying dense GP.
.restree_wn_leaf <- function(wn_leaf) {
  if (!is.character(wn_leaf) || length(wn_leaf) != 1L || !wn_leaf %in% c("plugin", "student"))
    stop("wn_leaf must be \"plugin\" or \"student\"", call. = FALSE)
  wn_leaf
}

.restree_pred_parts <- function(object, sd = NULL, y = NULL, wn_leaf = "plugin") {
  ## Reject length-mismatched y (silent dnorm recycling otherwise) and
  ## nonpositive or nonfinite sd (NaN scores otherwise).
  wn_leaf <- .restree_wn_leaf(wn_leaf)
  if (!is.null(y)) {
    if (!is.numeric(y) || !length(y) || any(!is.finite(y)))
      stop("y must be a nonempty finite numeric vector", call. = FALSE)
  }
  check_len <- function(mu) {
    if (!is.null(y) && length(y) != length(mu))
      stop("y and the predictive means have different lengths (",
           length(y), " vs ", length(mu), ")", call. = FALSE)
  }
  if (methods::is(object, "restree")) {
    .restree_kind(object)
    if (!length(object@prediction))
      stop("the restree object has no prediction; call restree_predict() first", call. = FALSE)
    object <- object@prediction
  }
  if (is.list(object) && !isS4(object)) {
    if (!all(c("mean", "var") %in% names(object)))
      stop("a prediction list must contain mean and var", call. = FALSE)
    if (is.null(object$joint_lpd)) object$joint_lpd <- NA_real_
    if (!is.null(object$par_df)) {
      check_len(object$mean)
      pm <- as.matrix(object$par_mean); pv <- as.matrix(object$par_var)
      df <- as.matrix(object$par_df); w <- as.numeric(object$w)
      if (!identical(dim(pm),dim(pv)) || !identical(dim(pm),dim(df)) ||
          any(!is.finite(pm)) || any(!is.finite(pv) | pv<=0) ||
          any(is.na(df) | df<=2) || length(w)!=ncol(pm) ||
          any(!is.finite(w) | w<0) || sum(w)<=0)
        stop("invalid predictive component distributions",call.=FALSE)
      return(list(kind="student mixture",mean=pm,var=pv,df=df,w=w,
        mu=object$mean,sd=sqrt(object$var),joint=object$joint_lpd,
        crps_plugin = (wn_leaf == "plugin")))   # CRPS on the Gaussian plug-in components
    }
    if (!is.null(object$par_mean) && ncol(as.matrix(object$par_mean)) > 1L) {
      check_len(object$mean)
      pm <- as.matrix(object$par_mean); pv <- as.matrix(object$par_var); w <- as.numeric(object$w)
      if (!identical(dim(pm), dim(pv)) || any(!is.finite(pm)) || any(!is.finite(pv) | pv <= 0) ||
          length(w) != ncol(pm) || any(!is.finite(w) | w < 0) || sum(w) <= 0)
        stop("invalid predictive component distributions", call. = FALSE)
      return(list(kind = "mixture", mean = pm, var = pv, w = w,
                  mu = object$mean, sd = sqrt(object$var), joint = object$joint_lpd))
    }
    check_len(object$mean)
    return(list(kind = "gaussian", mu = object$mean, sd = sqrt(object$var),
                joint = object$joint_lpd))
  }
  if (is.null(sd)) stop("supply sd, a restree object containing a prediction, or a prediction list")
  mu <- as.numeric(object); s <- as.numeric(sd)
  if (any(!is.finite(mu)))
    stop("predictive means must be finite", call. = FALSE)
  if (length(s) != length(mu))
    stop("sd and the predictive means have different lengths (",
         length(s), " vs ", length(mu), ")", call. = FALSE)
  if (any(!is.finite(s)) || any(s <= 0))
    stop("sd must contain only finite positive values", call. = FALSE)
  check_len(mu)
  list(kind = "gaussian", mu = mu, sd = s, joint = NA_real_)
}

#' @noRd
## the label of the represented predictive a score was computed for
.restree_score_type <- function(p, exact) {
  if (!exact) return("moment-matched Gaussian")
  if (p$kind == "student mixture") return("Student-t / Gaussian mixture")
  if (p$kind == "mixture") return("Gaussian mixture")
  "Gaussian"
}

## `parts` and `core` let restree_score() parse the prediction once and
## scan the component matrices once: `parts` is the .restree_pred_parts()
## result for (object, sd, y, wn_leaf), and `core` a score_mixture_cpp result
## on the REPRESENTED components (parts$mean, parts$var, parts$df, parts$w,
## default integration control) whose requested fields are used when they
## are present.  Both are NULL (recomputed here) in a standalone call.
.restree_core_field <- function(core, field, y) {
  z <- core[[field]]
  if (is.numeric(z) && length(z) == length(y)) z else NULL
}

#' @noRd
restree_lppd <- function(object, y, sd = NULL, exact = TRUE, wn_leaf = "plugin", ncores = 1L,
                         parts = NULL, core = NULL) {
  ## Returns the log pointwise predictive density (sum, higher is better), the
  ## logarithmic score (mean of -log p, lower is better), the pointwise vector,
  ## and -- when the object carries it -- the joint log predictive density.
  p <- if (is.null(parts)) .restree_pred_parts(object, sd, y, wn_leaf) else parts
  mixture <- exact && p$kind %in% c("student mixture", "mixture")
  ls <- if (mixture) .restree_core_field(core, "logscore", y) else NULL
  pw <- if (!is.null(ls)) ls else
        if (p$kind == "student mixture" && exact) .restree_logs_tmix(y, p, ncores) else
        if (p$kind == "mixture" && exact) logs_mixture(y, p$mean, p$var, p$w, ncores)
        else logs_gaussian(y, p$mu, p$sd)
  list(pointwise = -pw, lppd = -sum(pw), logscore = mean(pw),
       joint_lpd = p$joint, n = length(y), type = .restree_score_type(p, exact))
}

#' @noRd
restree_crps <- function(object, y, sd = NULL, exact = TRUE, control = list(), wn_leaf = "plugin",
                         ncores = 1L, parts = NULL, core = NULL) {
  p <- if (is.null(parts)) .restree_pred_parts(object, sd, y, wn_leaf) else parts
  if (p$kind == "student mixture" && exact) {
    ## neither WN route can take `core`: the plug-in CRPS scores the Gaussian
    ## components (df = NULL), the Student-t CRPS the caller's `control`
    if (isTRUE(p$crps_plugin)) {
      ## WN leaf variance at its posterior mean: the Gaussian mixture on the
      ## stored component moments, closed form
      pw <- crps_mixture(y, p$mean, p$var, p$w, ncores)
      return(list(pointwise = pw, CRPS = mean(pw), n = length(y),
                  type = "Gaussian mixture (WN leaf variance plug-in)"))
    }
    z <- .restree_crps_tmix(y, p, control, ncores)
    return(list(pointwise=z$value,CRPS=mean(z$value),n=length(y),
      abs_error=z$error,type="Student-t / Gaussian mixture (analytic or numerical CRPS)"))
  }
  cr <- if (p$kind == "mixture" && exact) .restree_core_field(core, "crps", y) else NULL
  pw <- if (!is.null(cr)) cr else
        if (p$kind == "mixture" && exact) crps_mixture(y, p$mean, p$var, p$w, ncores)
        else crps_gaussian(y, p$mu, p$sd)
  list(pointwise = pw, CRPS = mean(pw), n = length(y), type = .restree_score_type(p, exact))
}

#' Proper scoring rules for a residual-tree predictive distribution
#'
#' One-call scoring of a predictive distribution against held-out data:
#' accuracy (RMSE, MAE), CRPS for the represented Gaussian or Student-t
#' predictive mixture, the log pointwise
#' predictive density lppd (sum, higher is better; Gelman, Hwang and
#' Vehtari 2014) with its per-point negative the logarithmic score, the
#' joint log predictive density when the prediction carries one (plug-in
#' for Full MLE and EB; a Monte Carlo approximation over selected PMMH draws
#' and SMC trees for PP/WN PMMH), and calibration
#' (empirical 90/95 coverage, interval width, z-variance).
#'
#' @param object a prediction from \code{\link{restree_predict}}, its ordinary
#'   \code{prediction} list, a fitted object with an attached prediction, or a
#'   numeric vector of predictive means (then \code{sd} must be given).
#' @param y observed responses at the prediction inputs.
#' @param sd predictive standard deviations, when \code{object} is a
#'   numeric vector.
#' @param label method name for the returned row.
#' @param exact score the represented predictive mixture over trees (one
#'   component per distinct tree; closed forms throughout for Gaussian
#'   components). \code{FALSE} scores and calibrates a single moment-matched
#'   Gaussian instead.
#' @param wn_leaf how the CRPS of a WN prediction is computed.  Every score
#'   of a WN prediction averages the component quantity over the tree
#'   weights with the integrated Normal-Inverse-Gamma leaf predictive
#'   (Student-t components with \eqn{4 + n_{leaf}} degrees of freedom): the
#'   log score is \eqn{-\log \sum_k w_k f_k(y)}, coverage and width use the
#'   mixture cdf \eqn{\sum_k w_k F_k}.  Only the CRPS of a Student-t mixture
#'   has no closed form.  \code{"plugin"} (the default) fixes the leaf
#'   variance at its posterior mean for the CRPS, so every component is
#'   Gaussian with its stored mean and variance and the CRPS is the closed
#'   form of Grimit et al. (2006); \code{"student"} computes the Student-t
#'   mixture CRPS by numerical cdf integration. The approximation error
#'   depends on the component distributions and held-out responses; there
#'   is no general percentage bound. The other columns are identical.
#'   Ignored for PP / Full predictions.
#' @param control numerical CRPS settings for \code{wn_leaf = "student"}:
#'   \code{rel_tol} (default \code{1e-6}) and \code{subdivisions} (default
#'   300). Integration errors are reported, and a failed integration raises an
#'   error.
#' @param ncores positive thread request, accepted for interface consistency.
#'   Distribution and quadrature calculations currently run on the main R
#'   thread because they use R APIs. Likelihood, fitting and prediction retain
#'   their separate parallel execution policy.
#' @return A one-row data frame with columns \code{RMSE}, \code{MAE},
#'   \code{CRPS}, \code{logscore}, \code{lppd}, \code{joint_lpd},
#'   \code{cover90}, \code{cover95}, \code{width95}, \code{z_var},
#'   \code{score_type}, \code{crps_type}, and \code{crps_abs_error}.
#'   With \code{exact=TRUE}, coverage is the represented predictive cdf
#'   evaluated at \code{y} (\eqn{q_a < y < q_b} is the event
#'   \eqn{a < F(y) < b}) and the width uses the mixture quantiles, found by
#'   a bracketed bisection on the monotone cdf.
#' @section Numerical scales and limitations:
#' Gaussian mixtures use the supplied positive component variances without an
#' absolute epsilon floor. Gaussian CRPS combines standard deviations with
#' overflow-safe arithmetic, and Student-t scale conversion takes square roots
#' before multiplication to retain positive subnormal variances. Supported
#' components with invalid means, variances or degrees of freedom are rejected
#' before evaluation; zero-weight components are ignored by the core. The reported
#' \code{joint_lpd} is copied from prediction and is \code{NA} unless
#' joint scoring was explicitly enabled; see the empty-Full-branch
#' limitation in \code{\link{restree_predict}}.
#' @references Gneiting, T. and Raftery, A. E. (2007) \emph{JASA} 102,
#'   359-378.  Grimit, E. P. et al. (2006) \emph{QJRMS} 132, 2925-2942.
#'   Gelman, A., Hwang, J. and Vehtari, A. (2014) \emph{Stat Comput} 24,
#'   997-1016.
#' @examples
#' \donttest{
#' set.seed(5)
#' n <- 300; X <- matrix(runif(2 * n), n, 2)
#' y <- sin(4 * pi * X[, 1]) + rnorm(n, sd = 0.3)
#' th <- restree_theta(sig2 = 1, range = 0.2, nu = 2.5, nugget = 0.1)
#' fit <- restree_fit(restree_model(X, y, depth = 3, r = 15, leaf_model = "PP"),
#'                    theta = th, nparticles = 40, seed = 5)
#' Xt <- matrix(runif(100), 50, 2)
#' yt <- sin(4 * pi * Xt[, 1]) + rnorm(50, sd = 0.3)
#' pred <- restree_predict(fit, xnew = Xt, ynew = yt)
#' restree_score(pred, yt, label = "PP-eBayes")
#' }
#' @export
restree_score <- function(object, y, sd = NULL, label = NA_character_, exact = TRUE,
                          control = list(), wn_leaf = "plugin", ncores = 1L) {
  ## One-row summary: accuracy, sharpness, calibration, and both log-density
  ## conventions side by side, every mixture quantity averaged over the tree
  ## weights (compiled core, src/scores.cpp).  wn_leaf selects the WN CRPS
  ## route only; exact = FALSE explicitly requests one moment-matched Gaussian.
  ncores <- .restree_ncores(ncores)
  p  <- .restree_pred_parts(object, sd, y, wn_leaf)
  mixture <- exact && p$kind != "gaussian"
  ## One compiled pass over the component matrices for every quantity whose
  ## inputs coincide: the log score and the cdf
  ## at y always score the represented components (df = p$df), and so does
  ## the CRPS of a Gaussian mixture; a WN prediction scores its CRPS on the
  ## Gaussian plug-in components (wn_leaf = "plugin", df = NULL) or with the
  ## caller's integration control ("student"), so restree_crps() keeps its
  ## own call there.  score_mixture_cpp computes every requested quantity
  ## independently, so the merged values are those of the separate calls.
  ## The Gaussian-mixture dimension check is the one logs_mixture() /
  ## crps_mixture() made first before; the Student-t route relies on the
  ## compiled checks, as before.
  core <- NULL
  if (mixture) {
    if (p$kind == "mixture")
      stopifnot(length(y) == nrow(p$mean), dim(p$var) == dim(p$mean), length(p$w) == ncol(p$mean))
    core <- .restree_score_core(y, p$mean, p$var, p$df, p$w, crps = (p$kind == "mixture"),
                                logscore = TRUE, cdf = TRUE, probs = c(.025, .975), ncores = ncores)
  }
  ll <- restree_lppd(object, y, sd, exact, wn_leaf, ncores, parts = p, core = core)
  cr <- restree_crps(object, y, sd, exact, control, wn_leaf, ncores, parts = p, core = core)
  ## Calibration.  Coverage needs no quantiles: for the continuous, strictly
  ## increasing predictive cdf F, q_a < y < q_b is the event a < F(y) < b, so
  ## the exact mixture coverage is one vectorised cdf evaluation at y (no root
  ## finding; taken from the merged pass above).  The 95% width needs the two
  ## quantiles themselves (.restree_predictive_quantiles, bracketed
  ## bisection).
  if (mixture) {
    Fy <- .restree_core_field(core, "cdf_y", y)
    if (is.null(Fy)) Fy <- .restree_mixture_cdf(y, p, ncores)
    q95 <- core$quantiles
  } else {
    Fy <- stats::pnorm(y, p$mu, p$sd)
    q95 <- outer(p$mu, rep(1, 2)) + outer(p$sd, stats::qnorm(c(.025, .975)))
  }
  z  <- (y - p$mu) / p$sd
  data.frame(
    method   = label,
    n        = length(y),
    RMSE     = sqrt(mean((y - p$mu)^2)),
    MAE      = mean(abs(y - p$mu)),
    CRPS     = cr$CRPS,
    logscore = ll$logscore,          # mean -log p(y*_j), lower is better
    lppd     = ll$lppd,              # sum log p(y*_j), higher is better
    joint_lpd = ll$joint_lpd,        # log p(y*_{1:m} | y), NOT pointwise
    cover90  = mean(Fy > 0.05 & Fy < 0.95),
    cover95  = mean(Fy > 0.025 & Fy < 0.975),
    width95  = mean(q95[, 2] - q95[, 1]),
    z_var    = stats::var(z),        # 1 = correctly dispersed
    score_type = ll$type,
    crps_type = cr$type,
    crps_abs_error = if (is.null(cr$abs_error)) 0 else mean(cr$abs_error),
    stringsAsFactors = FALSE)
}
