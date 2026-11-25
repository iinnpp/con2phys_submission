# Usage from any directory: Rscript path/to/q6_glm_stats.R [results directory]
args <- commandArgs(trailingOnly = TRUE)
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
repo <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)
output <- if (length(args)) args[1] else Sys.getenv("Q6_OUTPUT_DIR", file.path(repo, "results", "q6", "glm"))
result_prefix <- "q6_glm"
estimator_label <- "model"
estimator_description <- c(
  "For each source, the score is the held-out Poisson log-likelihood gain per target spike in bits",
  "after conditioning on target history, third-area history, and a slow past target count."
)
source(file.path(dirname(script_path), "q6_directed_stats.R"), local = TRUE)
