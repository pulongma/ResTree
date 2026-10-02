##########################################################################
## The stored trees of a fit: one long-format node table
## (fit$structure$nodes) plus the mixture weights (fit$structure$w).
##
##   nodes: data.frame, one row per node position a tree reaches --
##          tree   1..K, the distinct tree
##          id     node id in the complete binary tree (root 1; children of
##                 id are 2 id and 2 id + 1; level = floor(log2(id)))
##          split  TRUE at a split, FALSE at a terminal
##          J      split dimension 1..d (NA at a terminal)
##          cut    realized cut (NA at a terminal)
##          rho    posterior stop probability of the node when the sampler
##                 drew there (NA at bottom-level and empty terminals)
##          n      observations reaching the node (empty children of a split
##                 that spent every row as a knot are not listed)
##   w:     nonnegative weights, one per tree.
##
## The table holds exactly the realized positions, so its size does not
## grow with the model's depth (a dense 2^depth x K layout does).  The
## likelihood and prediction paths consume reached-node records directly.
## The converter below supports the explicit matrix export and legacy inputs:
## S / J / cuts (2^depth rows: 0 split, 1 terminal, NA unreached).
## form = "posterior" aggregates the table into the posterior decision tree.
##########################################################################

.restree_node_columns <- c("tree", "id", "split", "J", "cut", "rho", "n")

## node table -> matrix convention over the internal positions 1 .. 2^depth
## (the compiled routines take 2^depth rows; row 2^depth is never read).
## The matrices are double (matrix(NA_real_, ...)): the compiled routines map
## them without a copy and require that storage mode.  rho = FALSE skips the
## rho matrix when a legacy caller needs decisions alone. Routine prediction
## uses node records and does not allocate these depth-sized matrices.
.restree_nodes_to_matrix <- function(nodes, depth, K = NULL, rho = TRUE) {
  numI <- 2L^depth
  if (is.null(K)) K <- max(nodes$tree)
  S <- J <- cuts <- matrix(NA_real_, numI, K)
  keep <- nodes$id < numI
  i <- cbind(nodes$id[keep], nodes$tree[keep])
  S[i] <- ifelse(nodes$split[keep], 0, 1)
  J[i] <- nodes$J[keep]
  cuts[i] <- nodes$cut[keep]
  out <- list(S = S, J = J, cuts = cuts)
  if (rho) {
    rho_m <- matrix(NA_real_, numI, K)
    rho_m[i] <- nodes$rho[keep]
    out$rho <- rho_m
  }
  out
}

## The node table of a Full model's stored fixed tree (model@tree).
.restree_nodes_from_fixed_tree <- function(tree, depth) {
  tr <- tree
  n_size <- as.integer(tr$node_size)
  at <- which(n_size > 0L)
  ids <- if (is.null(tr$id)) at else tr$id[at]
  split <- !is.na(tr$node_split[at]) & tr$node_split[at] == 1L
  from <- if (is.null(tr$id)) pmin(ids,2^depth) else at
  data.frame(tree = 1L, id = as.integer(ids), split = split,
    J = as.integer(ifelse(split,tr$J[from],NA_real_)),
    cut = as.numeric(ifelse(split,tr$cuts[from],NA_real_)),
    rho = NA_real_, n = n_size[at])
}

## Validation of a structure list (nodes + w); returns character(0) or messages.
.restree_validate_nodes <- function(st, depth, p) {
  msg <- character()
  nodes <- st$nodes
  if (!is.data.frame(nodes) || !all(.restree_node_columns %in% names(nodes)))
    return("structure$nodes must be a data.frame with columns tree, id, split, J, cut, rho, n")
  if (!nrow(nodes)) return("structure$nodes is empty")
  w <- st$w
  K <- max(nodes$tree)
  if (!is.numeric(w) || length(w) != K || any(!is.finite(w)) || any(w < 0) || sum(w) <= 0)
    msg <- c(msg, "structure$w must be nonnegative finite weights, one per tree")
  if (any(!is.finite(nodes$tree)) || any(nodes$tree < 1) || any(nodes$tree != floor(nodes$tree)) ||
      !setequal(unique(nodes$tree), seq_len(K)))
    msg <- c(msg, "structure$nodes$tree must be 1..K with every tree present")
  if (any(!is.finite(nodes$id)) || any(nodes$id < 1) || any(nodes$id >= 2^(depth + 1)))
    msg <- c(msg, "structure$nodes$id must lie in 1 .. 2^(depth+1) - 1")
  ## Encode (tree, id) as one integer key, tree * 2^(depth+1) + id.
  ## With id < 2^(depth+1)
  ## the key is unique per (tree, id); it is exact in double below 2^53, i.e.
  ## for tree * 2^(depth+1) < 2^53 -- at the deepest WN depth 25 that is
  ## tree < 2^27 (134M), always true in practice.  The parent key is the same
  ## expression at id %/% 2.  Non-numeric columns cannot be keyed: they are
  ## reported (the checks above already flag them) instead of erroring.
  if (!is.numeric(nodes$tree) || !is.numeric(nodes$id))
    return(c(msg, "structure$nodes$tree and structure$nodes$id must be numeric"))
  B <- 2^(depth + 1)
  key <- nodes$tree * B + nodes$id
  if (anyDuplicated(key)) msg <- c(msg, "structure$nodes has a duplicated (tree, id)")
  if (!all((seq_len(K) * B + 1) %in% key)) msg <- c(msg, "every tree needs its root (id 1)")
  if (!is.logical(nodes$split) || any(is.na(nodes$split)))
    msg <- c(msg, "structure$nodes$split must be TRUE / FALSE")
  sp <- nodes$split
  if (any(sp & (!is.finite(nodes$J) | nodes$J < 1 | nodes$J > p | nodes$J != floor(nodes$J))) ||
      any(sp & !is.finite(nodes$cut)))
    msg <- c(msg, "every split needs a dimension in 1..ncol(x) and a finite cut")
  if (any(sp & nodes$id >= 2^depth)) msg <- c(msg, "a bottom-level node cannot split")
  parent <- nodes$tree * B + nodes$id %/% 2L
  is_root <- nodes$id == 1L
  split_key <- key[sp]
  if (!all(parent[!is_root] %in% split_key))
    msg <- c(msg, "every non-root node needs its parent as a split of the same tree")
  msg
}

#' Stored trees and posterior tree summaries
#'
#' Extracts stored trees or summarizes their mixture weights, without running
#' inference. These are summaries of trees, not the covariance-parameter
#' posterior. For Full there is one fixed tree with weight one.
#' For WN the stored trees may extend beyond the model's initial depth: every
#' nonempty WN node compares stop versus split at every level, so the initial
#' depth is not a boundary. \code{object$structure$depth} records the
#' representation depth of the stored trees (never below the initial depth).
#'
#' @param object a fitted \code{\link{restree}} or its prediction object
#'   retaining fitted tree history.
#' @param form \code{"nodes"} (default), \code{"matrix"}, or \code{"posterior"}.
#' @param min_reach for \code{form = "posterior"}, keep node positions with
#'   posterior reach probability at least this value in [0,1].
#' @return \code{"nodes"} returns \code{list(nodes, w)}. The node table has
#'   one row per reached position per tree: \code{tree}, \code{id},
#'   \code{split}, \code{J}, \code{cut}, \code{rho}, \code{n}. Children of
#'   position \code{id} are \code{2*id} and \code{2*id+1}.
#'   \code{"matrix"} returns \code{list(S, J, cuts, rho, w)}; each decision
#'   matrix is \eqn{2^{D} \times K}, where \eqn{D} is the stored tree depth
#'   (which may exceed \code{object$depth} for WN), with \code{S} equal to 0 for a split,
#'   1 for a terminal, and NA for an unreached position.
#'   \code{"posterior"} returns an ordinary data frame, ordered by node id,
#'   with \code{id}, \code{level}, \code{parent}, \code{p_reach},
#'   \code{p_split} (conditional on reaching the position), \code{J_map},
#'   \code{p_J_map}, \code{cut_mean}, \code{cut_sd}, \code{rho_mean},
#'   \code{n_mean}, and \code{pJ1 ... pJd} for multiple inputs.
#'   Cut moments are conditional on splitting on \code{J_map}.
#'   Attributes \code{w} and \code{K} record the weights and tree count.
#' @section Interpretation:
#' Node ids denote binary-tree positions, not necessarily identical spatial
#' regions across different trees. Their posterior summary need not itself
#' define one realizable tree. Under PMMH, the stored trees come from the
#' final SMC at point-estimate theta; this is not a tree posterior integrated
#' over all retained theta draws. Prediction separately averages over draws.
#' @examples
#' set.seed(1)
#' X <- matrix(runif(120), 60, 2); y <- sin(4 * X[,1])
#' fit <- restree_fit(restree_model(X, y, depth = 2, r = 6),
#'   restree_theta(range = .2, nugget = .1), method = "smc", nparticles = 8)
#' head(restree_trees(fit)$nodes)
#' restree_trees(fit, form = "posterior")
#' @seealso \code{\link{restree_diagnostics}} for sampler quality, rather
#'   than tree decisions.
#' @export
restree_trees <- function(object, form = c("nodes", "matrix", "posterior"), min_reach = 0) {
  form <- match.arg(form)
  .restree_require_history(object)
  min_reach <- .restree_number(min_reach, "min_reach", 0, 1)
  if (form == "posterior") return(.restree_trees_posterior(object, min_reach))
  st <- object@structure
  if (is.null(st$nodes)) stop("the fit carries no stored tree structures", call. = FALSE)
  if (form == "nodes") return(list(nodes = st$nodes, w = as.numeric(st$w)))
  D <- .restree_stored_depth(st, object@depth)
  K <- length(st$w)
  ## The dense matrix form is 2^D rows per component (S, J, cuts, rho). A WN tree
  ## may be deep (up to .restree_max_wn_depth()), where 2^D x K is enormous and
  ## the node table (form = "nodes") holds the same information in O(reached
  ## positions). Refuse before allocating terabytes.
  cells <- 4 * (2^D) * K
  if (cells > 2^28)
    stop(sprintf(paste0("form = \"matrix\" would allocate %.3g doubles (2^%d x %d, four matrices). ",
                        "Use form = \"nodes\" (the same information, one row per reached position) ",
                        "or form = \"posterior\"."),
                 cells, D, K), call. = FALSE)
  m <- .restree_nodes_to_matrix(st$nodes, D, K)
  c(m, list(w = as.numeric(st$w)))
}

## Weighted posterior summaries of the stored node table (no new inference).
.restree_trees_posterior <- function(object, min_reach = 0) {
  st <- restree_trees(object, "nodes")
  nodes <- st$nodes; w <- st$w / sum(st$w)
  p <- ncol(object@x)
  wn <- w[nodes$tree]
  ids <- sort(unique(nodes$id))
  f <- factor(nodes$id, levels = ids)
  sum_by <- function(v) as.numeric(tapply(v, f, sum))
  p_reach <- sum_by(wn)
  w_split <- sum_by(wn * nodes$split)
  p_split <- w_split / p_reach
  ## Sum each (id, J) cell in node-table order. Keep sum()'s long-double
  ## accumulation rather than rowsum()'s double accumulation; empty cells are 0.
  on_j <- nodes$split & !is.na(nodes$J)
  pJ <- tapply(wn[on_j], list(factor(nodes$id[on_j], levels = ids),
                              factor(nodes$J[on_j], levels = seq_len(p))), sum)
  pJ <- matrix(as.numeric(pJ), nrow = length(ids), ncol = p)
  pJ[is.na(pJ)] <- 0
  pJ <- pJ / ifelse(w_split > 0, w_split, NA_real_)
  J_map <- ifelse(w_split > 0, max.col(replace(pJ, is.na(pJ), -Inf), ties.method = "first"), NA_integer_)
  p_J_map <- ifelse(w_split > 0, pJ[cbind(seq_along(ids), pmax(1L, J_map))], NA_real_)
  on_map <- nodes$split & !is.na(nodes$J) & nodes$J == J_map[as.integer(f)]
  w_map <- sum_by(wn * on_map)
  cut_mean <- sum_by(wn * on_map * ifelse(on_map, nodes$cut, 0)) / ifelse(w_map > 0, w_map, NA_real_)
  cut_sq <- sum_by(wn * on_map * ifelse(on_map, nodes$cut, 0)^2) / ifelse(w_map > 0, w_map, NA_real_)
  cut_sd <- sqrt(pmax(cut_sq - cut_mean^2, 0))
  has_rho <- !is.na(nodes$rho)
  w_rho <- sum_by(wn * has_rho)
  rho_mean <- sum_by(wn * ifelse(has_rho, nodes$rho, 0)) / ifelse(w_rho > 0, w_rho, NA_real_)
  has_n <- !is.na(nodes$n)
  w_n <- sum_by(wn * has_n)
  n_mean <- sum_by(wn * ifelse(has_n, nodes$n, 0)) / ifelse(w_n > 0, w_n, NA_real_)
  out <- data.frame(id = ids, level = floor(log2(ids)), parent = ifelse(ids > 1L, ids %/% 2L, NA_integer_),
                    p_reach = p_reach, p_split = p_split, J_map = J_map, p_J_map = p_J_map,
                    cut_mean = cut_mean, cut_sd = cut_sd, rho_mean = rho_mean, n_mean = n_mean)
  if (p > 1L) {
    colnames(pJ) <- paste0("pJ", seq_len(p))
    out <- cbind(out, pJ)
  }
  out <- out[out$p_reach >= min_reach, , drop = FALSE]
  rownames(out) <- NULL
  attr(out, "w") <- w; attr(out, "K") <- length(w)
  out
}
