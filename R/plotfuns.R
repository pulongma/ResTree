##########################################################################
## Every plotting function of the package lives here.
##
##   plot(fit)              -- MAP partition of a fit (optional ggplot2)
##   restree_diagnostics(type = "trace", ...)     -- sampler diagnostics of the SMC (known theta:
##                             a model + theta over replicate runs, or an smc /
##                             ebayes fit) and of the PMMH chain (a pmmh fit)
##   plot(fit, type = "trace")    -- ggplot2 figures of the above (ggplot2 is in
##                             Suggests; the numbers are available without it)
##########################################################################

## ggplot2 non-standard-evaluation column names used in aes()
utils::globalVariables(c("level", "ess", "rep", "value", "quantity",
                         "resampled", "iteration", "parameter", "kind",
                         "lag", "acf", "logZ", "yint", "x_plot", "y_plot",
                         "response", "xmin", "xmax", "ymin", "ymax", "cut"))

.restree_need_ggplot2 <- function() {
  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("plotting needs the suggested ggplot2 package (install.packages(\"ggplot2\"))",
         call. = FALSE)
  invisible(TRUE)
}

## ---------------------------------------------------------------------------
## MAP-partition plot (ggplot2 is optional)
## ---------------------------------------------------------------------------

## leaf boxes of one stored tree, by descending its recorded splits. Reads the
## node table (id, split, J, cut) by id lookup, so it never materialises a dense
## 2^depth vector -- a WN tree may be deep (up to .restree_max_wn_depth()).
.restree_leaf_boxes <- function(nodes, p) {
  rows <- list()
  lookup <- new.env(hash = TRUE, parent = emptyenv(), size = max(29L, nrow(nodes)))
  for (i in seq_len(nrow(nodes))) assign(as.character(nodes$id[i]), i, envir = lookup)
  rec <- function(id, lo, hi) {
    i <- lookup[[as.character(id)]]
    if (!is.null(i) && isTRUE(nodes$split[i])) {
      j <- as.integer(nodes$J[i]); c0 <- nodes$cut[i]
      hi2 <- hi; hi2[j] <- c0
      rec(2L * id, lo, hi2)
      lo2 <- lo; lo2[j] <- c0
      rec(2L * id + 1L, lo2, hi)
    } else {
      rows[[length(rows) + 1L]] <<- c(lo, hi)
    }
  }
  rec(1L, rep(0, p), rep(1, p))
  m <- do.call(rbind, rows)
  colnames(m) <- c(paste0("lo", seq_len(p)), paste0("hi", seq_len(p)))
  as.data.frame(m)
}

## ggplot2 MAP partition. Constructing a figure never changes the fitted tree.
.restree_plot_map <- function(x, dims = c(1L, 2L), ...) {
  .restree_need_ggplot2()
  st <- x$structure
  if (is.null(st$nodes))
    stop("the object carries no stored tree structures to plot", call. = FALSE)
  w <- as.numeric(st$w)
  # Plot only the MAP tree; neither a deep WN fit nor the plot needs K dense trees.
  k <- which.max(w)
  map_nodes <- st$nodes[st$nodes$tree == k, , drop = FALSE]
  map_nodes$tree <- 1L
  map_weight <- w[k] / sum(w); n_trees <- length(w)
  X <- as.matrix(x$x); yv <- as.numeric(x$y); p <- ncol(X)
  ttl <- sprintf("restree MAP partition (leaf %s, weight %.2f of %d trees)",
                 x$leaf_model, map_weight, n_trees)
  if (p == 1L) {
    dat <- data.frame(x_plot = X[, 1], y_plot = yv)
    sp <- map_nodes$split & !is.na(map_nodes$J)
    cuts <- data.frame(cut = map_nodes$cut[sp])
    return(ggplot2::ggplot(dat, ggplot2::aes(x_plot, y_plot)) +
      ggplot2::geom_point(size = 1, colour = "grey25", alpha = .7) +
      ggplot2::geom_vline(data = cuts, ggplot2::aes(xintercept = cut),
                          linetype = 2, colour = "grey40") +
      ggplot2::labs(x = "x", y = "y", title = ttl) + .restree_theme())
  }
  if (!is.numeric(dims) || length(dims) != 2L || any(!is.finite(dims)) ||
      any(dims != floor(dims)) || any(dims < 1 | dims > p) || anyDuplicated(dims))
    stop("dims must select two distinct input dimensions using integer indices", call. = FALSE)
  dims <- as.integer(dims)
  boxes <- .restree_leaf_boxes(map_nodes, p)
  boxes <- data.frame(xmin = boxes[[paste0("lo", dims[1])]],
                      xmax = boxes[[paste0("hi", dims[1])]],
                      ymin = boxes[[paste0("lo", dims[2])]],
                      ymax = boxes[[paste0("hi", dims[2])]])
  dat <- data.frame(x_plot = X[, dims[1]], y_plot = X[, dims[2]], response = yv)
  ggplot2::ggplot(dat, ggplot2::aes(x_plot, y_plot)) +
    ggplot2::geom_point(ggplot2::aes(colour = response), size = 1, alpha = .7) +
    ggplot2::geom_rect(data = boxes, inherit.aes = FALSE,
      ggplot2::aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
      fill = NA, colour = "grey25", linewidth = .4) +
    ggplot2::scale_colour_gradient(low = "grey80", high = "grey10") +
    ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    ggplot2::labs(x = sprintf("x[%d]", dims[1]), y = sprintf("x[%d]", dims[2]),
      colour = "y", title = ttl,
      subtitle = if (p > 2L) sprintf(
        "projection onto dimensions (%d, %d); splits on other dimensions overlap",
        dims[1], dims[2]) else NULL) + .restree_theme()
}

#' @rdname restree-class
#' @param y unused (base \code{plot} generic signature).
#' @param dims for inputs with more than two columns, the two input
#'   dimensions to display (default \code{c(1, 2)}).
#' @param type plot \code{"tree"} (default for a fit) or \code{"trace"}.
#'   An unfitted model only supports trace plots at supplied theta.
#' @param which trace figure: SMC \code{"levels"}, \code{"logZ"}, or
#'   \code{"all"}; PMMH \code{"trace"}, \code{"running"}, \code{"acf"},
#'   \code{"logZ"}, or \code{"all"}. NULL selects the first figure.
#'   \code{"all"} returns a named list of ggplots without arranging a page.
#' @param max_lag maximum lag for the PMMH autocorrelation plot.
#' @export
setMethod("plot", signature(x = "restree", y = "missing"),
  function(x, y, dims = c(1L, 2L), type = c("tree", "trace"), which = NULL,
           max_lag = 50L, ...) {
    type <- match.arg(type)
    if (type == "trace") {
      .restree_need_ggplot2()
      return(.restree_plot_trace(restree_diagnostics(x, type = "trace", ...), which, max_lag))
    }
    .restree_require_history(x)
    p <- .restree_plot_map(x, dims, ...)
    print(p)
    invisible(p)
  })

#' @rdname restree-class
#' @export
setMethod("plot", signature(x = "restree_model", y = "missing"),
  function(x, y, type = "trace", which = NULL, max_lag = 50L, ...) {
    if (!identical(type, "trace"))
      stop("a model supports type = 'trace'; fit it first for a tree plot", call. = FALSE)
    .restree_need_ggplot2()
    .restree_plot_trace(restree_diagnostics(x, type = "trace", ...), which, max_lag)
  })

## ---------------------------------------------------------------------------
## Sampler traces
## ---------------------------------------------------------------------------

## Trace collection implementation for restree_diagnostics(type = "trace").
.restree_collect_trace <- function(object, theta = NULL, nrep = 10L, nparticles = NULL,
                          resampling = "stratified", temper_alpha = 1,
                          seed = 42L, ncores = 1L, burnin = NULL) {
  if (methods::is(object, "restree") && .restree_kind(object) == "object") {
    if (is.null(theta)) theta <- object@theta
    object <- .restree_model_of(object)
  }
  if (methods::is(object, "restree")) {
    .restree_require_history(object)
    m <- object$method
    if (m %in% c("smc", "ebayes")) return(.restree_trace_from_smc_fit(object))
    if (m == "pmmh") return(.restree_trace_from_pmmh_fit(object, burnin))
    stop(sprintf("method = '%s' involves no sampler; nothing to trace", m),
         call. = FALSE)
  }
  if (!methods::is(object, "restree_model"))
    stop("object must be a restree_model or a restree fit", call. = FALSE)
  if (is.null(theta))
    stop("theta is required for the SMC trace of a model", call. = FALSE)
  if (object@leaf_model == "Full")
    stop("the SMC runs for leaf_model PP or WN; a Full model uses one fixed tree",
         call. = FALSE)
  if (.restree_root_only(object))
    stop("the SMC sampler needs depth >= 1", call. = FALSE)
  th <- .restree_theta_for(theta, ncol(object@x))
  nrep <- .restree_integer(nrep, "nrep", 1L)
  seed <- .restree_integer(seed, "seed", 0L)
  if (is.null(nparticles)) nparticles <- .restree_model_nparticles(object)
  nparticles <- .restree_integer(nparticles, "nparticles", 1L)
  opts <- .restree_smc_options(temper_alpha, resampling)

  runs <- vector("list", nrep); levs <- vector("list", nrep)
  for (k in seq_len(nrep)) {
    sk <- .restree_seed_add(seed, k - 1L)
    t0 <- proc.time()[3]
    z <- restree_loglik(object, th, nparticles = nparticles,
                        resampling = opts$resampling,
                        temper_alpha = opts$temper_alpha,
                        seed = sk, ncores = ncores, diagnostics = TRUE)
    el <- proc.time()[3] - t0
    ld <- attr(z, "level_diag")
    if (is.null(ld)) ld <- .restree_empty_level_diag()
    levs[[k]] <- cbind(rep = k, ld)
    runs[[k]] <- data.frame(rep = k, seed = sk, logZ = as.numeric(z),
                            ess = as.numeric(attr(z, "ess")),
                            n_distinct_trees = as.integer(attr(z, "n_distinct_trees")),
                            n_resample = as.integer(attr(z, "n_resample")),
                            n_knot = as.numeric(attr(z, "n_knot")),
                            root_split_weight = as.numeric(attr(z, "root_split_weight")),
                            root_split_fraction = as.numeric(attr(z, "root_split_fraction")),
                            seconds = as.numeric(el))
  }
  runs <- do.call(rbind, runs); levs <- do.call(rbind, levs)
  list(kind = "smc", source = "model", leaf_model = object@leaf_model,
                 depth = object@depth, nparticles = nparticles,
                 resampling = opts$resampling, temper_alpha = opts$temper_alpha,
                 levels = levs, runs = runs,
                 evidence = .restree_evidence_replicates(runs$logZ),
                 notes = c("High particle ESS does not establish posterior coverage.",
                   if (object@leaf_model == "WhiteNoise" && all(runs$root_split_fraction == 0))
                     "No run split the WN root: compare larger particle budgets and independent seeds."),
                 sd_logZ = if (nrep >= 2L) stats::sd(runs$logZ) else NA_real_)
}

.restree_trace_from_smc_fit <- function(fit) {
  ld <- fit$fit$level_diag
  if (is.null(ld)) ld <- .restree_empty_level_diag()
  steps <- fit$fit$step_diag
  if (is.null(steps)) steps <- .restree_empty_step_diag()
  ess <- as.numeric(fit$fit$ESS)
  ## the sampler seed is the results entry (`fit$seed` is the model slot)
  runs <- data.frame(rep = 1L, seed = fit@results$seed, logZ = as.numeric(fit$logZ),
                     ess = utils::tail(ess, 1L),
                     n_distinct_trees = as.integer(fit$fit$n_distinct_trees),
                     n_resample = as.integer(fit$fit$n_resample),
                     n_knot = NA_real_,
                     seconds = as.numeric(fit$fit$smc_time))
  list(kind = "smc", source = fit$method, leaf_model = fit$leaf_model,
                 depth = fit@depth, nparticles = fit$nparticles,
                 resampling = fit$resampling, temper_alpha = fit$temper_alpha,
                 levels = cbind(rep = rep.int(1L, nrow(ld)), ld),
                 steps = cbind(rep = rep.int(1L, nrow(steps)), steps),
                 runs = runs, sd_logZ = NA_real_)
}

.restree_trace_from_pmmh_fit <- function(fit, burnin = NULL) {
  ch <- fit$chain; ni <- nrow(ch)
  bi <- if (!is.null(burnin)) .restree_integer(burnin, "burnin", 0L, ni - 1L) else
    if (!is.null(fit$burnin)) fit$burnin else floor(ni / 2)
  f <- fit$fit
  cp <- f$chain_prop
  if (is.null(cp)) {   # no proposal record: plot the accepted chain alone
    cp <- matrix(NA_real_, ni, ncol(ch), dimnames = dimnames(ch))
  }
  reps <- as.numeric(fit$logZ_reps); reps <- reps[is.finite(reps)]
  list(kind = "pmmh", source = "pmmh", leaf_model = fit$leaf_model,
                 depth = fit@depth, nparticles = fit$nparticles,
                 chain = ch, chain_prop = cp,
                 logZ_trace = as.numeric(f$logZ_trace),
                 logZ_prop = if (is.null(f$logZ_prop)) rep(NA_real_, ni) else as.numeric(f$logZ_prop),
                 accepted = if (is.null(f$accepted)) rep(NA, ni) else as.logical(f$accepted),
                 burnin = bi, n_iter = ni, accept_rate = fit$accept_rate,
                 logZ_reps = reps,
                 sd_logZ = if (length(reps) >= 2L) stats::sd(reps) else NA_real_)
}

## ---- ggplot2 figures ------------------------------------------------------

.restree_theme <- function() {
  ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(panel.grid.minor = ggplot2::element_blank(),
                   strip.background = ggplot2::element_rect(fill = "grey93"),
                   legend.position = "bottom")
}

## SMC: ESS / logZ increment / paths by level, one line per run
.restree_plot_smc_levels <- function(x) {
  ld <- x$levels
  if (!nrow(ld)) stop("no level diagnostics to plot (depth 0)", call. = FALSE)
  nr <- nrow(x$runs)
  long <- rbind(
    data.frame(rep = ld$rep, level = ld$level, quantity = "ESS at level end",
               value = ld$ess, resampled = ld$resampled == 1L),
    data.frame(rep = ld$rep, level = ld$level, quantity = "log-evidence increment",
               value = ld$logZ_increment, resampled = ld$resampled == 1L),
    data.frame(rep = ld$rep, level = ld$level, quantity = "distinct node-paths",
               value = ld$n_paths, resampled = ld$resampled == 1L))
  long$quantity <- factor(long$quantity, levels = c("ESS at level end",
                                                    "log-evidence increment",
                                                    "distinct node-paths"))
  long$rep <- factor(long$rep)
  ref <- data.frame(quantity = factor("ESS at level end", levels = levels(long$quantity)),
                    yint = x$nparticles / 2)
  p <- ggplot2::ggplot(long, ggplot2::aes(x = level, y = value, group = rep)) +
    ggplot2::geom_hline(data = ref, ggplot2::aes(yintercept = yint),
                        linetype = 2, colour = "grey45") +
    ggplot2::geom_line(colour = "grey35", alpha = if (nr > 1L) 0.45 else 1,
                       linewidth = 0.4) +
    ggplot2::geom_point(ggplot2::aes(shape = resampled, fill = resampled),
                        size = 2, colour = "grey15") +
    ggplot2::scale_shape_manual(values = c(`FALSE` = 21, `TRUE` = 24),
                                labels = c(`FALSE` = "no", `TRUE` = "yes"),
                                name = "resampled at level end", drop = FALSE) +
    ggplot2::scale_fill_manual(values = c(`FALSE` = "white", `TRUE` = "#D55E00"),
                               labels = c(`FALSE` = "no", `TRUE` = "yes"),
                               name = "resampled at level end", drop = FALSE) +
    ggplot2::scale_x_continuous(breaks = sort(unique(long$level))) +
    ggplot2::facet_wrap(~ quantity, ncol = 1L, scales = "free_y") +
    ggplot2::labs(x = "tree level", y = NULL,
                  title = sprintf("SMC over trees: %s leaves, depth %d, N = %d, %s resampling, alpha = %g",
                                  x$leaf_model, x$depth, x$nparticles, x$resampling, x$temper_alpha),
                  subtitle = if (nr > 1L) sprintf("%d independent runs; dashed line: resampling threshold N/2", nr)
                             else "dashed line: resampling threshold N/2") +
    .restree_theme()
  p
}

## SMC: replicate log evidences
.restree_plot_smc_logZ <- function(x) {
  r <- x$runs
  if (nrow(r) < 2L)
    stop("the replicate-evidence figure needs at least two runs (restree_diagnostics(model, type = \"trace\", theta = theta, nrep = ...))",
         call. = FALSE)
  m <- mean(r$logZ); s <- stats::sd(r$logZ)
  ggplot2::ggplot(r, ggplot2::aes(x = rep, y = logZ)) +
    ggplot2::geom_hline(yintercept = m, colour = "grey45") +
    ggplot2::geom_hline(yintercept = m + c(-1, 1) * s, colour = "grey45", linetype = 3) +
    ggplot2::geom_point(size = 2.2, colour = "#0072B2") +
    ggplot2::scale_x_continuous(breaks = r$rep) +
    ggplot2::labs(x = "run", y = expression(log~hat(Z)),
                  title = "Replicate SMC evidence estimates at fixed theta",
                  subtitle = sprintf("mean %.3f ; sd %.3f (lines: mean, mean +/- sd); PMMH target sd 1.0 - 1.7",
                                     m, s)) +
    .restree_theme()
}

## PMMH: long data of accepted chain and proposals
.restree_pmmh_long <- function(x) {
  ch <- log(x$chain); cp <- log(x$chain_prop)
  pn <- colnames(ch); ni <- nrow(ch)
  lab <- c(paste0("log(", pn, ")"), "log Z")
  acc <- data.frame(iteration = rep(seq_len(ni), ncol(ch) + 1L),
                    parameter = rep(lab, each = ni),
                    value = c(as.numeric(ch), x$logZ_trace), kind = "accepted chain")
  prop <- data.frame(iteration = rep(seq_len(ni), ncol(cp) + 1L),
                     parameter = rep(lab, each = ni),
                     value = c(as.numeric(cp), x$logZ_prop), kind = "rejected proposal")
  rej <- rep(!x$accepted, ncol(cp) + 1L)
  prop <- prop[!is.na(rej) & rej & is.finite(prop$value), , drop = FALSE]
  out <- rbind(acc, prop)
  out$parameter <- factor(out$parameter, levels = lab)
  out
}

.restree_plot_pmmh_trace <- function(x) {
  long <- .restree_pmmh_long(x)
  acc <- long[long$kind == "accepted chain", ]
  prop <- long[long$kind == "rejected proposal", ]
  p <- ggplot2::ggplot(acc, ggplot2::aes(x = iteration, y = value)) +
    ggplot2::annotate("rect", xmin = -Inf, xmax = x$burnin + 0.5, ymin = -Inf, ymax = Inf,
                      fill = "grey85", alpha = 0.5)
  if (nrow(prop))
    p <- p + ggplot2::geom_point(data = prop, colour = "#D55E00", size = 0.6, alpha = 0.45)
  p + ggplot2::geom_line(colour = "grey20", linewidth = 0.35) +
    ggplot2::facet_wrap(~ parameter, ncol = 1L, scales = "free_y") +
    ggplot2::labs(x = "iteration", y = NULL,
                  title = sprintf("PMMH: %s leaves, depth %d, N = %d, %d iterations, acceptance %.2f",
                                  x$leaf_model, x$depth, x$nparticles, x$n_iter, x$accept_rate),
                  subtitle = paste0("line: accepted chain; points: rejected proposals; shaded: burn-in",
                                    if (is.finite(x$sd_logZ)) sprintf("\nsd(log Zhat) at the point estimate %.2f over %d replicate runs (target 1.0 - 1.7)", x$sd_logZ, length(x$logZ_reps)) else "")) +
    .restree_theme()
}

.restree_plot_pmmh_running <- function(x) {
  ch <- log(x$chain); ni <- nrow(ch); keep <- seq.int(x$burnin + 1L, ni)
  pn <- colnames(ch)
  d <- do.call(rbind, lapply(seq_along(pn), function(j) {
    v <- ch[keep, j]
    data.frame(iteration = keep, parameter = paste0("log(", pn[j], ")"),
               value = cumsum(v) / seq_along(v))
  }))
  d$parameter <- factor(d$parameter, levels = paste0("log(", pn, ")"))
  ggplot2::ggplot(d, ggplot2::aes(x = iteration, y = value)) +
    ggplot2::geom_line(colour = "#0072B2", linewidth = 0.5) +
    ggplot2::facet_wrap(~ parameter, ncol = 1L, scales = "free_y") +
    ggplot2::labs(x = "iteration", y = "running mean",
                  title = "PMMH running posterior means after burn-in") +
    .restree_theme()
}

.restree_plot_pmmh_acf <- function(x, max_lag = 50L) {
  ch <- log(x$chain); ni <- nrow(ch); keep <- seq.int(x$burnin + 1L, ni)
  pn <- colnames(ch); L <- min(max_lag, length(keep) - 1L)
  d <- do.call(rbind, lapply(seq_along(pn), function(j) {
    a <- stats::acf(ch[keep, j], lag.max = L, plot = FALSE)$acf[, 1, 1]
    data.frame(lag = seq_along(a) - 1L, parameter = paste0("log(", pn[j], ")"), acf = a)
  }))
  d$parameter <- factor(d$parameter, levels = paste0("log(", pn, ")"))
  ci <- stats::qnorm(0.975) / sqrt(length(keep))
  ggplot2::ggplot(d, ggplot2::aes(x = lag, y = acf)) +
    ggplot2::geom_hline(yintercept = 0, colour = "grey40") +
    ggplot2::geom_hline(yintercept = c(-ci, ci), colour = "grey55", linetype = 3) +
    ggplot2::geom_segment(ggplot2::aes(xend = lag, yend = 0), colour = "#0072B2") +
    ggplot2::facet_wrap(~ parameter, ncol = 1L) +
    ggplot2::labs(x = "lag", y = "autocorrelation",
                  title = "PMMH autocorrelation of the retained draws") +
    .restree_theme()
}

.restree_plot_pmmh_logZ <- function(x) {
  keep <- seq.int(x$burnin + 1L, x$n_iter)
  tr <- data.frame(value = x$logZ_trace[keep], kind = "accepted chain after burn-in")
  reps <- x$logZ_reps
  d <- tr
  if (length(reps))
    d <- rbind(d, data.frame(value = reps, kind = "replicates at the point estimate"))
  d$kind <- factor(d$kind, levels = c("accepted chain after burn-in",
                                      "replicates at the point estimate"))
  ggplot2::ggplot(d, ggplot2::aes(x = kind, y = value)) +
    ggplot2::geom_boxplot(width = 0.35, outlier.shape = NA, colour = "grey30") +
    ggplot2::geom_jitter(width = 0.12, height = 0, size = 0.9, alpha = 0.5, colour = "#0072B2") +
    ggplot2::scale_x_discrete(labels = function(l) sub(" (after|at) ", "\n\\1 ", l)) +
    ggplot2::labs(x = NULL, y = expression(log~hat(Z)),
                  title = "PMMH evidence estimates",
                  subtitle = paste0(
                    sprintf("sd of the chain's log Z %.2f (posterior variation of log Z(theta) plus noise)",
                            stats::sd(tr$value)),
                    if (length(reps) >= 2L) sprintf("\nreplicate sd at fixed theta %.2f (%d runs; target 1.0 - 1.7)",
                                                    x$sd_logZ, length(reps))
                    else "\nno replicate runs (control$sd_reps = 0)")) +
    .restree_theme()
}

.restree_plot_trace <- function(x, which = NULL, max_lag = 50L) {
  .restree_need_ggplot2()
  if (x$kind == "smc") {
    choices <- c("levels", "logZ", "all")
    which <- if (is.null(which)) "levels" else match.arg(which, choices)
    if (which == "levels") { p <- .restree_plot_smc_levels(x); print(p); return(invisible(p)) }
    if (which == "logZ")   { p <- .restree_plot_smc_logZ(x); print(p); return(invisible(p)) }
    plots <- list(levels = .restree_plot_smc_levels(x))
    if (nrow(x$runs) >= 2L) plots$logZ <- .restree_plot_smc_logZ(x)
    return(invisible(plots))
  }
  choices <- c("trace", "running", "acf", "logZ", "all")
  which <- if (is.null(which)) "trace" else match.arg(which, choices)
  one <- switch(which,
    trace = .restree_plot_pmmh_trace(x),
    running = .restree_plot_pmmh_running(x),
    acf = .restree_plot_pmmh_acf(x, max_lag),
    logZ = .restree_plot_pmmh_logZ(x),
    NULL)
  if (!is.null(one)) { print(one); return(invisible(one)) }
  plots <- list(trace = .restree_plot_pmmh_trace(x),
                running = .restree_plot_pmmh_running(x),
                acf = .restree_plot_pmmh_acf(x, max_lag),
                logZ = .restree_plot_pmmh_logZ(x))
  invisible(plots)
}
