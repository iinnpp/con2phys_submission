suppressPackageStartupMessages({
  library(lmerTest)
  library(emmeans)
  library(ggplot2)
})

repository_dir <- Sys.getenv(
  "CON2PHYS_REPOSITORY_DIR",
  unset = "D:/git_repos/con2phys_submission"
)
output_dir <- file.path(repository_dir, "results", "q14")
data_path <- file.path(output_dir, "q14_modularity_mouse_area.csv")
sensitivity_data_path <- file.path(
  output_dir,
  "q14_modularity_sensitivity_mouse_area.csv"
)
results_path <- file.path(output_dir, "q14_modularity_stats_results.txt")
means_path <- file.path(output_dir, "q14_modularity_stats_means.csv")
overall_path <- file.path(output_dir, "q14_modularity_stats_overall_effect.csv")
pairwise_path <- file.path(output_dir, "q14_modularity_stats_pairwise_tukey.csv")
permutation_path <- file.path(output_dir, "q14_modularity_permutation_results.csv")
sensitivity_means_path <- file.path(
  output_dir,
  "q14_modularity_sensitivity_area_means.csv"
)
answer_path <- file.path(output_dir, "answer.txt")
mean_plot_path <- file.path(
  output_dir,
  "q14_modularity_stats_means_95ci.png"
)
pairwise_plot_path <- file.path(
  output_dir,
  "q14_modularity_stats_pairwise_tukey.png"
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
modularity <- read.csv(data_path)
sensitivity <- read.csv(sensitivity_data_path)

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

sensitivity_area_means <- aggregate(
  mean_delta_q ~ setting_type + density + gamma + brain_area,
  data = sensitivity,
  FUN = mean
)
sensitivity_area_means <- sensitivity_area_means[order(
  sensitivity_area_means$setting_type,
  sensitivity_area_means$density,
  sensitivity_area_means$gamma,
  sensitivity_area_means$brain_area
), ]
setting_key <- paste(
  sensitivity_area_means$setting_type,
  sensitivity_area_means$density,
  sensitivity_area_means$gamma,
  sep = ":"
)
lowest_by_setting <- tapply(
  seq_len(nrow(sensitivity_area_means)),
  setting_key,
  function(indices) sensitivity_area_means$brain_area[
    indices[which.min(sensitivity_area_means$mean_delta_q[indices])]
  ]
)
all_sensitivity_lowest_area_1 <- all(lowest_by_setting == 1)
write.csv(sensitivity_area_means, sensitivity_means_path, row.names = FALSE)

############################################################
### Usual statistical plots
############################################################

mean_ci_plot <- ggplot(
  area_means_ci,
  aes(x = brain_area, y = emmean)
) +
  geom_errorbar(
    aes(ymin = lower.CL, ymax = upper.CL),
    width = 0.08,
    linewidth = 0.7
  ) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  scale_x_discrete(labels = area_labels) +
  labs(
    title = "Model-estimated modularity and 95% CI",
    subtitle = "Whole-recording spike networks; chance-corrected delta Q",
    x = "Brain area",
    y = "Mean chance-corrected modularity (delta Q)"
  ) +
  theme_classic(base_size = 14) +
  theme(
    panel.grid.major = element_line(color = "grey75", linewidth = 0.4),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

ggsave(
  mean_plot_path,
  plot = mean_ci_plot,
  width = 7.0,
  height = 4.7,
  dpi = 150
)

contrast_span <- diff(range(c(
  pairwise_tukey$lower.CL,
  pairwise_tukey$upper.CL
)))
pairwise_tukey$label_x <- pairwise_tukey$upper.CL + max(
  contrast_span * 0.04,
  0.002
)

pairwise_plot <- ggplot(
  pairwise_tukey,
  aes(x = estimate, y = contrast_label)
) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbar(
    aes(xmin = lower.CL, xmax = upper.CL),
    orientation = "y",
    width = 0.12,
    linewidth = 0.7
  ) +
  geom_point(size = 3) +
  geom_text(
    aes(x = label_x, label = significance),
    hjust = 0,
    size = 5
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.08, 0.18))) +
  labs(
    title = "Pairwise brain-area differences and 95% CI",
    subtitle = "Tukey-adjusted confidence intervals and p-values",
    x = "Difference in chance-corrected modularity (delta Q)",
    y = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    panel.grid.major.x = element_line(color = "grey75", linewidth = 0.4),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

ggsave(
  pairwise_plot_path,
  plot = pairwise_plot,
  width = 9.0,
  height = 4.7,
  dpi = 150
)

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
cat("\nDensity and resolution sensitivity\n")
cat("Settings evaluated:", length(lowest_by_setting), "\n")
cat("Area 1 descriptively lowest in every setting:", all_sensitivity_lowest_area_1, "\n")
cat("\nDecision rule result\n")
cat("Descriptively lowest area:", area_labels[lowest_area], "\n")
cat("Significantly lower than both other areas:", lower_than_both, "\n")
cat("Q14 answer:", answer_choice, "\n")
cat("\nSaved statistical plots\n")
cat("Area means:", mean_plot_path, "\n")
cat("Pairwise comparisons:", pairwise_plot_path, "\n")
sink()

area_effect <- overall_effect[overall_effect$term == "brain_area", ]
mean_text <- sprintf(
  "%.4f for area %s (95%% CI %.4f to %.4f)",
  area_means_ci$emmean, area_means_ci$brain_area,
  area_means_ci$lower.CL, area_means_ci$upper.CL
)
contrast_text <- sprintf(
  "The area %s minus area %s difference was %.4f (95%% CI %.4f to %.4f, p = %.4f).",
  pairwise_tukey$contrast_area_1, pairwise_tukey$contrast_area_2,
  pairwise_tukey$estimate, pairwise_tukey$lower.CL, pairwise_tukey$upper.CL,
  pairwise_tukey$p.value
)
sensitivity_ranking <- if (length(unique(lowest_by_setting)) == 1L) {
  sprintf(
    "Brain area %s had the lowest mean delta Q across all %d settings, including the primary setting.",
    lowest_by_setting[[1]], length(lowest_by_setting)
  )
} else {
  "The area with the lowest mean delta Q varied across settings."
}
decision_text <- if (lower_than_both) {
  "It was lower than both other areas at Tukey-adjusted p < 0.05."
} else {
  "The Tukey-adjusted comparisons provided insufficient evidence that it was lower than both other areas."
}

answer_paragraphs <- c(
  paste0(
    "Modularity was estimated from single-unit spikes across the entire recording, including the intervals ",
    "between trials. We reused the pairwise Spike Time Tiling Coefficient (STTC) values from Q4. For each ",
    "mouse, Q4 used a window starting at the earliest recorded spike and extending for the full LFP-derived ",
    "recording duration. Two spikes counted as close in time if they occurred within 100 ms of each other. ",
    "STTC measures how often one neuron's spikes fall close to the other neuron's spikes, accounting for how ",
    "much of the recording is already close to spikes from each neuron. A positive value indicates more ",
    "coincidence than expected from that coverage. Only pairs recorded within the same mouse and area were used."
  ),
  paste0(
    "To compare networks of the same size within each mouse, we matched all three areas to the neuron count ",
    "in that mouse's smallest recorded area. Four of 18 mice had fewer than 10 neurons in at least one area. ",
    "This left ", nlevels(modularity$mouse_id), " mice with matched sets of ",
    min(modularity$n_sampled_units), " to ", max(modularity$n_sampled_units), " neurons per area. For each ",
    "mouse and area, we drew 100 random neuron sets, sampling without replacement within each set. The ",
    "smallest area contributed all its neurons each time. In each network, the neurons were the nodes and ",
    "the links were pairs with the highest positive STTC values. The number of links was 20% of all possible ",
    "pairs among the selected neurons, rounded up to a whole number. We used the original STTC values as ",
    "link weights, so stronger spike-time associations contributed more to the score."
  ),
  paste0(
    "Modularity (Q) measures how clearly neurons form groups with strong links within each group. For a ",
    "given grouping, Q compares the amount of link weight inside the groups with the amount expected from ",
    "each neuron's total link weight. A higher Q means that more of the network's weight is concentrated ",
    "within groups than expected. We used weighted Leiden community detection at resolution gamma = 1 to ",
    "find groups and kept the highest Q from 25 runs with different random seeds."
  ),
  paste0(
    "Even networks with randomly arranged links can yield a positive Q when the grouping is chosen to ",
    "maximize it. To estimate that baseline, we made 20 randomized versions of every observed network. We ",
    "swapped which neurons were linked while keeping the number of links per neuron fixed, then shuffled ",
    "the original STTC weights across those links. This kept the same number of neurons and links and the ",
    "same set of weights. We optimized Q separately for each randomized network at the same resolution, ",
    "using 10 runs per network. Chance-corrected modularity (delta Q) was the observed Q minus the mean Q ",
    "of the 20 randomized networks. A positive delta Q indicates stronger grouping than the randomized ",
    "baseline, and a value near zero indicates a similar amount of grouping. We averaged delta Q across ",
    "the 100 neuron sets to obtain one score for each mouse and area. The mouse was the independent unit."
  ),
  paste0(
    "We compared the three areas using a linear mixed-effects model with brain area as a fixed effect ",
    "and mouse as a random intercept to account for repeated measurements from each mouse. The overall ",
    sprintf(
      "area comparison gave F(%.0f, %.0f) = %.2f, p = %.4f. ",
      area_effect$NumDF, area_effect$DenDF, area_effect[["F value"]], area_effect[["Pr(>F)"]]
    ),
    "Model-estimated mean delta Q was ", paste(mean_text[1:2], collapse = ", "),
    ", and ", mean_text[3], "."
  ),
  paste0(
    "We adjusted the three pairwise comparisons using Tukey's method. ",
    paste(contrast_text, collapse = " ")
  ),
  paste0(
    if (lme4::isSingular(model)) {
      "The random-intercept variance was estimated at zero. "
    } else "",
    "As a robustness check, we ran a permutation test that preserved the pairing within each mouse. ",
    sprintf(
      "Shuffling the area labels within mice %s times gave p = %.4f.",
      format(n_omnibus_permutations, big.mark = ","), omnibus_permutation_p
    )
  ),
  paste0(
    "To check how much the ranking depended on how the networks were constructed and scored, we varied ",
    "the link density (10%, 15%, 20%, 25%, and 30% at gamma = 1) and the resolution parameter ",
    "(gamma = 0.5, 0.75, 1, 1.25, 1.5, and 2 at 20% density). The resolution parameter controls the ",
    "scale of the groups the score favors. Each additional setting used 20 neuron sets and 10 randomized ",
    "networks per set. ", sensitivity_ranking
  ),
  paste0(
    area_labels[lowest_area], " had the lowest estimated modularity in the primary analysis. ",
    decision_text, " The Q14 answer is \"", answer_choice, "\"."
  )
)
answer_lines <- c(
  unlist(lapply(answer_paragraphs, function(paragraph) c(strwrap(paragraph, width = 105), ""))),
  "Code files",
  "code/questions/q14_modularity.py",
  "code/questions/q14.ipynb",
  "code/questions/q14_modularity_stats.R",
  "",
  "Model output",
  "results/q14/q14_modularity_stats_results.txt"
)
writeLines(answer_lines, answer_path)

cat("Saved statistical results to", results_path, "\n")
cat("Saved Q14 answer to", answer_path, "\n")
