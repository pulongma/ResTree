#' Plain result lists and prediction objects
#'
#' ResTree defines only three S4 classes: \code{restree_model},
#' \code{restree_theta} and \code{restree}. Fits and predictions have class
#' \code{restree}; their \code{kind} slot distinguishes their roles.
#' Prediction values are an ordinary list in \code{object@prediction}, also
#' accessible as \code{object$prediction}. The same list is attached when
#' \code{restree_fit} is called with prediction inputs.
#' Predictions also retain ordinary \code{smc} and \code{mcmc} lists from
#' the fit. PMMH prediction adds per-selected-iteration moments in
#' \code{object@mcmc$prediction}, with chain indices and component mapping.
#'
#' \code{restree_trees} returns ordinary lists or a posterior-summary
#' data frame. \code{restree_diagnostics} returns ordinary summary or trace
#' lists. Internal SMC/PMMH records are ordinary lists. No auxiliary ResTree
#' S4 classes or ResTree S3 methods are registered.
#'
#' Use \code{plot(fit, type = "trace")} for sampler plots. Trace lists do not
#' have custom plot methods. The suggested ggplot2 package is needed only
#' when requesting plots, not when collecting numerical traces or summaries.
#'
#' @name restree-results
#' @seealso \code{\link{restree}}, \code{\link{restree_predict}},
#'   \code{\link{restree_trees}}, \code{\link{restree_diagnostics}}
NULL

.restree_kind <- function(object) {
  kind <- attr(object, "kind", exact = TRUE)
  if (!is.character(kind) || length(kind) != 1L || is.na(kind) ||
      !kind %in% c("object", "fit", "prediction"))
    stop("this restree object has no valid kind; rebuild the model and refit saved legacy objects",
         call. = FALSE)
  kind
}

.restree_require_fit <- function(object) {
  if (!methods::is(object, "restree") || .restree_kind(object) != "fit")
    stop("object must be a fitted 'restree' (kind = 'fit'); use restree_fit()",
         call. = FALSE)
  invisible(object)
}

# Read-only inspection can use the fitted state retained by a prediction.
# A manually assembled prediction without a fit history has nothing to inspect.
.restree_require_history <- function(object) {
  if (!methods::is(object, "restree") ||
      !.restree_kind(object) %in% c("fit", "prediction") ||
      identical(object@method, "none") || !length(object@structure))
    stop("object must carry fitted history; use a fit or its prediction object", call. = FALSE)
  invisible(object)
}

.restree_validate_prediction <- function(z, d) {
  if (!all(c("mean", "var") %in% names(z)))
    return("prediction must contain mean and var")
  m <- length(z$mean)
  if (!is.numeric(z$mean) || !is.numeric(z$var) || length(z$var) != m ||
      any(!is.finite(z$mean)) || any(!is.finite(z$var) | z$var < 0))
    return("prediction mean and var must be finite numeric vectors of equal length, with var >= 0")
  if (!is.null(z$xnew) && (!is.matrix(z$xnew) || !is.numeric(z$xnew) ||
      nrow(z$xnew) != m || ncol(z$xnew) != d ||
      any(!is.finite(z$xnew) | z$xnew < 0 | z$xnew > 1)))
    return("prediction xnew must have one unit-cube row per predicted mean")
  if (!is.null(z$y) && (!is.numeric(z$y) || length(z$y) != m || any(!is.finite(z$y))))
    return("prediction y must have one finite value per predicted mean")
  if (any(c("par_mean", "par_var", "w") %in% names(z))) {
    if (!is.matrix(z$par_mean) || !is.numeric(z$par_mean) ||
        !is.matrix(z$par_var) || !is.numeric(z$par_var) ||
        !identical(dim(z$par_mean), dim(z$par_var)) || nrow(z$par_mean) != m ||
        !is.numeric(z$w) || length(z$w) != ncol(z$par_mean) ||
        any(!is.finite(z$w) | z$w < 0) || !any(z$w > 0))
      return("prediction component matrices and nonnegative weights must have matching dimensions")
    if (!is.null(z$par_df) && (!is.matrix(z$par_df) || !is.numeric(z$par_df) ||
        !identical(dim(z$par_df), dim(z$par_mean))))
      return("prediction par_df must match the component matrices")
  }
  character()
}

## Inputs and source fit were validated by restree_predict. Reuse model/theta
## via copy-on-write without new()/a second full geometry validation. Retain
## histories, posterior structures and native handles for read-only inspection
## -- unless keep_history = FALSE: then the smc and mcmc histories, the
## posterior structures and sampler state are dropped, so a saved prediction does
## not carry the whole fit (the node table, the chain, the engine record);
## only the per-draw prediction cache of a PMMH fit, which belongs to this
## prediction and not to the fit, is kept in mcmc. Training data and model
## context (including Full geometry) remain. restree_trees() and
## restree_diagnostics() then refuse the object (no fitted history).
.restree_prediction_object <- function(object, prediction, mcmc_prediction = NULL,
                                       keep_history = TRUE) {
  out <- methods::as(object, "restree")
  out@kind <- "prediction"
  out@prediction <- prediction
  # Do not carry a stale MCMC prediction cache into a new plug-in prediction.
  out@mcmc$prediction <- mcmc_prediction
  if (!keep_history) {
    out@smc <- list()
    out@mcmc <- if (!is.null(mcmc_prediction)) list(prediction = mcmc_prediction) else list()
    out@structure <- list()
  }
  out
}
