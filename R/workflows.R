#' ResTree models and a first likelihood/prediction workflow
#'
#' These examples preserve the three S4 classes: \code{restree_model},
#' \code{restree_theta}, and \code{restree}. Fits and predictions use
#' \code{restree}; tree tables and diagnostics are lists. No preparation
#' object is required to call \code{\link{restree_loglik}} repeatedly.
#'
#' @section Choose the model:
#' \tabular{lll}{
#' Model \tab Tree rule \tab Parameter fitting\cr
#' Full \tab One fixed tree; split iff remaining n exceeds r \tab MLE\cr
#' PP \tab Inferred tree, capped at supplied depth \tab EB or PMMH\cr
#' WN \tab Inferred adaptive tree at positive initial depth \tab EB or PMMH
#' }
#' Full removes knots at each split and extends its initial depth until each
#' leaf has at most \code{r} rows (representational limit 20). PP keeps its
#' depth cap. At positive initial depth, every nonempty WN node compares stop
#' and split, including small nodes; splitting a node with at most \code{r}
#' rows spends every row as knots. WN depth zero is a single WN leaf. See
#' \code{\link{restree_model}} for priors, limits and root-only fitting rules.
#'
#' @section Knots are not cuts:
#' \code{design} selects knot rows (\code{maximin}, \code{boundary}, or PP's
#' \code{nested} design). \code{cut_method} selects the cut law (\code{middle},
#' \code{median}, \code{uniform}, \code{balanced}). Ordinary PP uniform cuts
#' default to \code{cut_candidates=30}: complementary draws plus a
#' midpoint-focused stratum, weighted to preserve the cut prior. Nested PP
#' integrates rank cells instead; these are not 31 equally weighted quantiles.
#'
#' @section Interpret comparisons:
#' Full returns its fixed covariance likelihood; PP and WN estimate evidence
#' for their own tree-prior mixtures. Their ordering or closeness to a dense
#' GP is not guaranteed. A Full model with \code{r=nrow(x)} and depth zero is
#' a small-data dense-GP reference. Separate model construction, first-call
#' preparation and warm evaluation when timing repeated likelihoods.
#' The example uses small particle counts for instruction, not certification
#' of evidence accuracy or PMMH convergence. \code{method="smc"} holds theta
#' fixed; use EB/PMMH for parameter inference and inspect diagnostics separately.
#'
#' @section Optional plotting and storage:
#' Numerical workflows do not require \pkg{ggplot2}. When installed, plotting
#' returns a ggplot object that callers can customize. Prediction objects keep
#' training/model context even with \code{keep_history=FALSE}. Saved native
#' pointers rebuild lazily; use intact fitted objects for later prediction.
#'
#' @name restree_workflows
#' @seealso \code{\link{restree_model}}, \code{\link{restree_theta}},
#'   \code{\link{restree_loglik}}, \code{\link{restree_fit}},
#'   \code{\link{restree_predict}}, \code{\link{restree_score}}
#' @examples
#' source(system.file("examples", "quickstart.R", package = "ResTree"), local = TRUE)
NULL
