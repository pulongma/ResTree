# Small instructional workflow, not a benchmark or a convergence study.
# No plotting package is needed for any numerical calculation.
library(ResTree)
set.seed(1701)
x <- matrix(stats::runif(160), 80, 2)
y <- sin(5*x[,1]) + (0.15 + 0.5*(x[,2] > 0.5))*stats::rnorm(80)
xnew <- matrix(stats::runif(40), 20, 2)
ynew <- sin(5*xnew[,1]) + (0.15 + 0.5*(xnew[,2] > 0.5))*stats::rnorm(20)
theta <- restree_theta(range = c(0.3, 0.4), nugget = 0.1) # ARD default

# Known theta: Full is one fixed tree; PP/WN estimate tree-prior evidence.
# Knot design and cut law are different options. Ordinary uniform PP defaults
# to 30 complementary candidates plus its midpoint-focused stratum.
models <- stats::setNames(lapply(c("Full", "PP", "WN"), function(leaf)
  restree_model(x, y, r = 12, depth = 3, leaf_model = leaf,
                design = "maximin", cut_method = "uniform")), c("Full", "PP", "WN"))
loglik <- vapply(models, function(m)
  as.numeric(restree_loglik(m, theta, nparticles = 16, seed = 71)), 0.0)
exact_gp <- restree_model(x, y, r = nrow(x), depth = 0, leaf_model = "Full")
exact_gp_loglik <- as.numeric(restree_loglik(exact_gp, theta))
# These numbers target different covariance/tree models, so no ordering is
# guaranteed. Repeated seeded estimates assess Monte Carlo variability.

# A fitted tree distribution at known theta, then prediction and proper scores.
# Use method="ebayes" to estimate theta, or method="pmmh" to sample it.
# Those modes need their own optimization/MCMC and particle-accuracy checks.
fit <- restree_fit(models$WN, theta, method = "smc", nparticles = 32, seed = 71)
prediction <- restree_predict(fit, xnew)
scores <- restree_score(prediction, ynew)
trees <- restree_trees(fit)
diagnostics <- restree_diagnostics(fit)
figure <- if (requireNamespace("ggplot2", quietly = TRUE)) plot(fit) else NULL

# No plotting object is required to serialize or reuse the fitted S4 object.
restored_prediction <- restree_predict(unserialize(serialize(fit, NULL)), xnew)
# Rebuilt floating-point calculations need numerical, not bitwise, equality.
stopifnot(isTRUE(all.equal(prediction$mean, restored_prediction$mean, tolerance=1e-12)),
          isTRUE(all.equal(prediction$var, restored_prediction$var, tolerance=1e-12)))
invisible(list(loglik=loglik, exact_gp_loglik=exact_gp_loglik, fit=fit,
               prediction=prediction, scores=scores, trees=trees,
               diagnostics=diagnostics, figure=figure))
