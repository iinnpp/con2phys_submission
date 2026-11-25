# Usage from any directory: Rscript path/to/q6_ccg_jittered_stats.R [results directory]
args <- commandArgs(trailingOnly = TRUE)
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
repo <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)
output <- if (length(args)) args[1] else Sys.getenv(
  "Q6_CCG_OUTPUT_DIR", file.path(repo, "results", "q6", "ccg_jittered")
)
result_prefix <- "q6_ccg"
estimator_description <- c(
  "The normalized correlogram subtracts the exact mean cross-count under independent interval jitter.",
  "That baseline preserves each area's spike count per jitter window. Positive lags mean the target follows the source.",
  "The asymmetry score is the mean corrected CCG at the positive lags minus its mean at mirrored negative lags.",
  "The zero-lag bin is excluded. Both sides use the same population count variance normalization."
)
stats_spec <- list(
  setting_columns = c("bin_ms", "jitter_ms", "max_lag_ms"), metric_column = "asymmetry",
  units = "normalized CCG units", show_shift = FALSE,
  timing_text = function(setting) sprintf("Primary CCG: %d ms lag bins, %d ms jitter, mirrored lags through %d ms.",
                                          setting$bin_ms, setting$jitter_ms, setting$max_lag_ms),
  timing_plot = function(setting, n) sprintf("%d ms bins, %d ms jitter, %d ms lag limit, %d mice",
                                              setting$bin_ms, setting$jitter_ms, setting$max_lag_ms, n),
  sensitivity_label = "bin width/jitter width/maximum lag (ms)",
  closing = c("The score measures pairwise lead-lag asymmetry in excess co-spiking.",
              "A common input with different delays can also produce an asymmetry.")
)
source(file.path(dirname(script_path), "q6_directed_stats.R"), local = TRUE)
