# Usage from any directory: Rscript path/to/q6_cte_stats.R [results directory]
args <- commandArgs(trailingOnly = TRUE)
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
repo <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)
output <- if (length(args)) args[1] else Sys.getenv(
  "Q6_CTE_OUTPUT_DIR", file.path(repo, "results", "q6", "conditional_transfer_entropy")
)
result_prefix <- "q6_cte"
estimator_label <- "conditional transfer entropy setting"
estimator_description <- c(
  "The score is the held-out gain in categorical log probability from adding source history,",
  "conditional on the target and third-area short histories and the slow target state.",
  "The primary comparison uses bits per original target spike. The score in bits per bin is also saved.",
  "The target response uses counts 0, 1, 2, 3, 4 and 5+; histories use training-fold count tertiles.",
  "Negative estimates can arise when the estimated full distribution predicts worse on held-out bins."
)
source(file.path(dirname(script_path), "q6_directed_stats.R"), local = TRUE)
