suppressPackageStartupMessages({
  library(lmerTest)
  library(emmeans)
})

repository_dir <- Sys.getenv(
  "CON2PHYS_REPOSITORY_DIR",
  unset = "D:/git_repos/con2phys_submission"
)
output_dir <- file.path(repository_dir, "results", "q14")
data_path <- file.path(output_dir, "q14_modularity_mouse_area.csv")
results_path <- file.path(output_dir, "q14_modularity_stats_results.txt")
means_path <- file.path(output_dir, "q14_modularity_stats_means.csv")
overall_path <- file.path(output_dir, "q14_modularity_stats_overall_effect.csv")
pairwise_path <- file.path(output_dir, "q14_modularity_stats_pairwise_tukey.csv")
permutation_path <- file.path(output_dir, "q14_modularity_permutation_results.csv")
answer_path <- file.path(output_dir, "answer.txt")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
modularity <- read.csv(data_path)

required_columns <- c(
  "mouse_id", "brain_area", "density", "gamma", "n_available_units",
  "n_sampled_units", "n_subsamples", "n_nulls_per_subsample",
  "mean_q_observed", "mean_q_null", "mean_delta_q", "mean_z_q"
)
missing_columns <- setdiff(required_columns, names(modularity))
if (length(missing_columns) > 0) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "))
}
if (nrow(modularity) != 42) {
  stop("Expected 42 primary rows (14 eligible mice x 3 brain areas)")
}
numeric_columns <- setdiff(required_columns, c("mouse_id", "brain_area"))
if (any(!is.finite(as.matrix(modularity[, numeric_columns])))) {
  stop("Primary modularity input contains non-finite values")
}
if (any(modularity$n_sampled_units < 10)) {
  stop("All primary networks must contain at least 10 matched neurons")
}
if (any(modularity$n_subsamples != 100)) {
  stop("Expected 100 neuron subsamples per primary mouse-area score")
}
if (any(modularity$n_nulls_per_subsample != 20)) {
  stop("Expected 20 degree-preserving nulls per primary subsample")
}
if (any(abs(modularity$density - 0.20) > 1e-12)) {
  stop("Primary density must be 0.20")
}
if (any(abs(modularity$gamma - 1.0) > 1e-12)) {
  stop("Primary resolution gamma must be 1.0")
}

area_levels <- c("1", "2", "3")
area_labels <- c(
  `1` = "Brain area 1",
  `2` = "Brain area 2",
  `3` = "Brain area 3"
)
modularity$mouse_id <- factor(modularity$mouse_id)
modularity$brain_area <- factor(modularity$brain_area, levels = area_levels)
if (nlevels(modularity$mouse_id) != 14) {
  stop("Expected 14 eligible mice")
}
if (anyNA(modularity$brain_area)) {
  stop("Unexpected or missing brain area")
}
if (any(table(modularity$mouse_id) != 3)) {
  stop("Every eligible mouse must have all three brain areas")
}
matched_within_mouse <- tapply(
  modularity$n_sampled_units,
  modularity$mouse_id,
  function(values) length(unique(values))
)
if (any(matched_within_mouse != 1)) {
  stop("Neuron counts are not matched across areas within mouse")
}

significance_label <- function(p_value) {
  ifelse(
    p_value < 0.001,
    "***",
    ifelse(p_value < 0.01, "**", ifelse(p_value < 0.05, "*", "ns"))
  )
}

############################################################
### Primary mouse-level repeated-measures model
############################################################

model <- lmer(
  mean_delta_q ~ brain_area + (1 | mouse_id),
  data = modularity,
  REML = TRUE
)
overall_effect <- as.data.frame(anova(model, ddf = "Satterthwaite"))
overall_effect$term <- rownames(overall_effect)
rownames(overall_effect) <- NULL
overall_effect <- overall_effect[, c("term", setdiff(names(overall_effect), "term"))]

area_means <- emmeans(model, ~ brain_area, lmer.df = "satterthwaite")
area_means_ci <- as.data.frame(confint(area_means, level = 0.95))
area_means_ci$brain_area_label <- area_labels[as.character(area_means_ci$brain_area)]

pairwise_tukey <- as.data.frame(summary(
  pairs(area_means, adjust = "tukey"),
  infer = c(TRUE, TRUE)
))
pairwise_tukey$significance <- significance_label(pairwise_tukey$p.value)
contrast_parts <- strsplit(as.character(pairwise_tukey$contrast), " - ")
pairwise_tukey$contrast_area_1 <- vapply(
  contrast_parts,
  function(parts) sub("^brain_area", "", parts[1]),
  character(1)
)
pairwise_tukey$contrast_area_2 <- vapply(
  contrast_parts,
  function(parts) sub("^brain_area", "", parts[2]),
  character(1)
)
pairwise_tukey$contrast_label <- vapply(
  seq_len(nrow(pairwise_tukey)),
  function(index) paste(
    area_labels[pairwise_tukey$contrast_area_1[index]],
    area_labels[pairwise_tukey$contrast_area_2[index]],
    sep = " - "
  ),
  character(1)
)

write.csv(area_means_ci, means_path, row.names = FALSE)
write.csv(overall_effect, overall_path, row.names = FALSE)
write.csv(pairwise_tukey, pairwise_path, row.names = FALSE)

############################################################
### Within-mouse randomization sensitivity analyses
############################################################

wide <- reshape(
  modularity[, c("mouse_id", "brain_area", "mean_delta_q")],
  idvar = "mouse_id",
  timevar = "brain_area",
  direction = "wide"
)
wide <- wide[order(as.integer(as.character(wide$mouse_id))), ]
value_matrix <- as.matrix(wide[, paste0("mean_delta_q.", area_levels)])
colnames(value_matrix) <- area_levels

omnibus_statistic <- function(values) {
  area_means_local <- colMeans(values)
  sum((area_means_local - mean(area_means_local))^2)
}

set.seed(14042)
n_omnibus_permutations <- 100000L
observed_omnibus <- omnibus_statistic(value_matrix)
null_omnibus <- numeric(n_omnibus_permutations)
for (iteration in seq_len(n_omnibus_permutations)) {
  permuted <- t(apply(value_matrix, 1, sample, replace = FALSE))
  null_omnibus[iteration] <- omnibus_statistic(permuted)
}
omnibus_permutation_p <- (
  1 + sum(null_omnibus >= observed_omnibus - 1e-15)
) / (n_omnibus_permutations + 1)

pair_names <- c("1 - 2", "1 - 3", "2 - 3")
pair_indices <- list(c(1, 2), c(1, 3), c(2, 3))
pairwise_permutation <- data.frame(
  contrast = pair_names,
  mean_difference = NA_real_,
  exact_two_sided_p = NA_real_
)
n_mice <- nrow(value_matrix)
sign_patterns <- as.matrix(expand.grid(rep(list(c(-1, 1)), n_mice)))
for (pair_index in seq_along(pair_indices)) {
  columns <- pair_indices[[pair_index]]
  differences <- value_matrix[, columns[1]] - value_matrix[, columns[2]]
  observed_difference <- mean(differences)
  null_differences <- as.vector(sign_patterns %*% differences / n_mice)
  exact_p <- mean(abs(null_differences) >= abs(observed_difference) - 1e-15)
  pairwise_permutation$mean_difference[pair_index] <- observed_difference
  pairwise_permutation$exact_two_sided_p[pair_index] <- exact_p
}
pairwise_permutation$holm_p <- p.adjust(
  pairwise_permutation$exact_two_sided_p,
  method = "holm"
)
pairwise_permutation$significance <- significance_label(
  pairwise_permutation$holm_p
)
pairwise_permutation$omnibus_statistic <- observed_omnibus
pairwise_permutation$omnibus_permutation_p <- omnibus_permutation_p
pairwise_permutation$n_omnibus_permutations <- n_omnibus_permutations
write.csv(pairwise_permutation, permutation_path, row.names = FALSE)

############################################################
### Multiple-choice decision rule
############################################################

lowest_row <- area_means_ci[which.min(area_means_ci$emmean), ]
lowest_area <- as.character(lowest_row$brain_area)
other_areas <- setdiff(area_levels, lowest_area)

is_significantly_lower <- function(candidate, comparison) {
  matching_row <- which(
    (pairwise_tukey$contrast_area_1 == candidate &
       pairwise_tukey$contrast_area_2 == comparison) |
      (pairwise_tukey$contrast_area_1 == comparison &
         pairwise_tukey$contrast_area_2 == candidate)
  )
  if (length(matching_row) != 1) {
    stop("Could not identify exactly one pairwise comparison")
  }
  estimate <- pairwise_tukey$estimate[matching_row]
  candidate_difference <- if (
    pairwise_tukey$contrast_area_1[matching_row] == candidate
  ) estimate else -estimate
  candidate_difference < 0 && pairwise_tukey$p.value[matching_row] < 0.05
}

lower_than_both <- all(vapply(
  other_areas,
  function(comparison) is_significantly_lower(lowest_area, comparison),
  logical(1)
))
answer_choice <- if (lower_than_both) {
  area_labels[lowest_area]
} else {
  "Not enough data / no differences"
}

sink(results_path)
cat("Q14 whole-recording spike-network modularity statistics\n")
cat("Input:", data_path, "\n")
cat("Independent unit: mouse\n")
cat("Eligible mice:", nlevels(modularity$mouse_id), "\n")
cat("Brain areas: 1, 2, 3\n")
cat("Primary response: mean chance-corrected modularity (delta Q)\n")
cat("Primary graph density: 0.20\n")
cat("Primary Leiden resolution gamma: 1.0\n")
cat("Model: mean_delta_q ~ brain_area + (1 | mouse_id)\n\n")
cat("Mouse-area input summary\n")
print(summary(modularity$mean_delta_q))
cat("\nMixed model summary\n")
print(summary(model))
cat("\nOverall effect of brain area\n")
print(overall_effect)
cat("\nModel-estimated mean delta Q and 95% confidence intervals\n")
print(area_means_ci)
cat("\nTukey-adjusted pairwise comparisons\n")
print(pairwise_tukey)
cat("\nWithin-mouse omnibus permutation test\n")
cat("Statistic:", observed_omnibus, "\n")
cat("Permutations:", n_omnibus_permutations, "\n")
cat("p-value:", omnibus_permutation_p, "\n")
cat("\nExact paired sign-flip tests with Holm adjustment\n")
print(pairwise_permutation)
cat("\nDecision rule result\n")
cat("Descriptively lowest area:", area_labels[lowest_area], "\n")
cat("Significantly lower than both other areas:", lower_than_both, "\n")
cat("Q14 answer:", answer_choice, "\n")
sink()

mean_lookup <- setNames(area_means_ci$emmean, area_means_ci$brain_area)
ci_low_lookup <- setNames(area_means_ci$lower.CL, area_means_ci$brain_area)
ci_high_lookup <- setNames(area_means_ci$upper.CL, area_means_ci$brain_area)
overall_p <- overall_effect[overall_effect$term == "brain_area", "Pr(>F)"]

answer_lines <- c(
  paste0(
    "Whole-recording spike-network modularity was estimated separately for each mouse and brain area from the ",
    "same pairwise STTC values used in Q4 (dt = 100 ms). Neuron counts were matched across areas within each ",
    "mouse by repeated subsampling. Mice with fewer than 10 neurons in any area were excluded, leaving 14 ",
    "complete mice. For each subsample, the strongest 20% of positive STTC edges formed a weighted graph. ",
    "Weighted Leiden community detection was repeated across random initializations at resolution gamma = 1."
  ),
  "",
  paste0(
    "Modularity was chance-corrected by subtracting the mean optimized modularity of degree-preserving, ",
    "edge-rewired null graphs with the observed edge weights shuffled. The resulting delta Q values were ",
    "averaged across 100 neuron subsamples to produce one value per mouse and brain area; subsamples and edges ",
    "were not treated as independent observations."
  ),
  "",
  paste0(
    "A linear mixed-effects model with brain area as a fixed effect and mouse as a random intercept gave an ",
    "overall brain-area p-value of ", format.pval(overall_p, digits = 4), ". Model-estimated mean delta Q values ",
    "(95% CI) were ",
    sprintf("%.4f (%.4f to %.4f)", mean_lookup["1"], ci_low_lookup["1"], ci_high_lookup["1"]),
    " for area 1, ",
    sprintf("%.4f (%.4f to %.4f)", mean_lookup["2"], ci_low_lookup["2"], ci_high_lookup["2"]),
    " for area 2, and ",
    sprintf("%.4f (%.4f to %.4f)", mean_lookup["3"], ci_low_lookup["3"], ci_high_lookup["3"]),
    " for area 3."
  ),
  "",
  paste0(
    "Decision rule: select an area only when it has the lowest model-estimated delta Q and is lower than both ",
    "other areas in Tukey-adjusted comparisons (p < 0.05). Q14 answer: ", answer_choice, "."
  )
)
writeLines(answer_lines, answer_path)

cat("Saved statistical results to", results_path, "\n")
cat("Saved Q14 answer to", answer_path, "\n")
