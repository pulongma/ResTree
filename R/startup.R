##########################################################################
## start-up and clean-up functions
##
## This software is distributed under the terms of the GNU GENERAL
## PUBLIC LICENSE Version 2 or later.
##
## Copyright (C) 2024-present by Pulong Ma
##    
##########################################################################

.onAttach <- function(...) {
	
	date <- date()
	x <- regexpr("[0-9]{4}", date)
	this.year <- substr(date, x[1], x[1] + attr(x, "match.length") - 1)
	
	# echo output to screen
	packageStartupMessage("\n#########################################################")
	packageStartupMessage("## ResTree: Residual Tree Gaussian Process Models for High-Dimensional Spatial Data")
	packageStartupMessage("## Copyright (C) 2024-", this.year,
			" by Pulong Ma <plma@iastate.edu>", sep="")
	packageStartupMessage("## Please cite ResTree including its version number: citation(\"ResTree\").")
	packageStartupMessage("##########################################################")
}

.onUnload <- function(libpath) {
	library.dynam.unload("ResTree", libpath)
}
## Internal capability probe (capabilities.cpp): whether the package was
## built with OpenMP, the physical-core cap (physical_cores), the logical
## processor count (max_threads), the default team size (omp_max_threads, the
## OMP_NUM_THREADS value), and the two tree-depth limits (max_model_depth,
## max_wn_depth).  Not exported -- ResTree:::restree_capabilities().
restree_capabilities <- function() capabilities_cpp()

## The compiled depth limits, as the single source of truth (src/ResTree.h):
##   .restree_max_model_depth  largest initial depth restree_model(depth=) allows
##                             (every leaf model; PP/Full realized cap too)
##   .restree_max_wn_depth     largest realized WhiteNoise tree depth (WN grows
##                             past the initial depth; the id-width overflow guard)
.restree_max_model_depth <- function() capabilities_cpp()$max_model_depth
.restree_max_wn_depth    <- function() capabilities_cpp()$max_wn_depth
