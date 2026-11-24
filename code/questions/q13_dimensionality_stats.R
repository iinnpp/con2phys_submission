suppressPackageStartupMessages({
  library(lmerTest)
  library(emmeans)
  library(ggplot2)
})

data_path <- paste0(
  "D:/git_repos/con2phys_submission/results/q13/",
  "q13_dimensionality_mouse_area.csv"
)
output_dir <- dirname(data_path)
results_path <- file.path(output_dir, "q13_dimensionality_stats_results.txt")
means_path <- file.path(output_dir, "q13_dimensionality_stats_means.csv")
overall_effect_path <- file.path(output_dir, "q13_dimensionality_stats_overall_effect.csv")
pairwise_path <- file.path(output_dir, "q13_dimensionality_stats_pairwise_tukey.csv")
normalized_path <- file.path(output_dir, "q13_normalized_pr_sensitivity.csv")
answer_path <- file.path(output_dir, "q13_answer.txt")
mean_plot_path <- file.path(output_dir, "q13_dimensionality_stats_means_95ci.png")
pairwise_plot_path <- file.path(output_dir, "q13_dimensionality_stats_pairwise_tukey.png")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dimensionality <- read.csv(data_path)

required_columns <- c(
  "mouse_id", "brain_area", "mean_participation_ratio",
  "mean_normalized_participation_ratio", "n_trials",
  "n_available_units", "n_sampled_units", "n_subsamples"
)
missing_columns <- setdiff(required_columns, names(dimensionality))
if (length(missing_columns) > 0) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "))
}
if (nrow(dimensionality) != 54) {
  stop("Expected 54 mouse-area rows (18 mice x 3 areas)")
}
numeric_columns <- c(
  "mean_participation_ratio", "mean_normalized_participation_ratio",
  "n_trials", "n_available_units", "n_sampled_units", "n_subsamples"
)
if (any(!is.finite(as.matrix(dimensionality[, numeric_columns])))) {
  stop("Input contains non-finite numeric values")
}
if (any(dimensionality$mean_participation_ratio <= 0)) {
  stop("Participation ratios must be positive")
}
if (any(
  dimensionality$mean_normalized_participation_ratio <= 0 |
    dimensionality$mean_normalized_participation_ratio > 1
)) {
  stop("Normalized participation ratios must be within (0, 1]")
}
if (any(dimensionality$n_subsamples != 100)) {
  stop("Expected 100 neuron subsamples per mouse-area")
}

area_levels <- c("1", "2", "3")
area_labels <- c(
  `1` = "Brain area 1",
  `2` = "Brain area 2",
  `3` = "Brain area 3"
)
dimensionality$mouse_id <- factor(dimensionality$mouse_id)
dimensionality$brain_area <- factor(
  dimensionality$brain_area,
  levels = area_levels
)
if (nlevels(dimensionality$mouse_id) != 18) {
  stop("Expected 18 mice")
}
if (anyNA(dimensionality$brain_area)) {
  stop("Unexpected or missing brain area")
}

mouse_area_key <- paste(
  dimensionality$mouse_id,
  dimensionality$brain_area,
  sep = ":"
)
if (anyDuplicated(mouse_area_key)) {
  stop("Duplicate mouse-area rows found")
}
rows_per_mouse <- table(dimensionality$mouse_id)
if (any(rows_per_mouse != 3)) {
  stop("Every mouse must have one row for each brain area")
}
sampled_counts_per_mouse <- tapply(
  dimensionality$n_sampled_units,
  dimensionality$mouse_id,
  function(values) length(unique(values))
)
if (any(sampled_counts_per_mouse != 1)) {
  stop("Neuron count is not matched across areas within mouse")
}

significance_label <- function(p_value) {
  ifelse(
    p_value < 0.001,
    "***",
    ifelse(p_value < 0.01, "**", ifelse(p_value < 0.05, "*", "ns"))
  )
}

############################################################
### Primary repeated-measures model: participation ratio
############################################################

# The response is averaged over neuron subsamples before inference. Mouse is
# the independent unit; the random intercept accounts for the three paired
# brain-area observations from each mouse.
model <- lmer(
  mean_participation_ratio ~ brain_area + (1 | mouse_id),
  data = dimensionality,
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
  character(1L)
)

write.csv(area_means_ci, means_path, row.names = FALSE)
write.csv(overall_effect, overall_effect_path, row.names = FALSE)
write.csv(pairwise_tukey, pairwise_path, row.names = FALSE)

############################################################
### Sensitivity model: participation ratio per sampled neuron
############################################################

normalized_model <- lmer(
  mean_normalized_participation_ratio ~ brain_area + (1 | mouse_id),
  data = dimensionality,
  REML = TRUE
)
normalized_means <- emmeans(
  normalized_model,
  ~ brain_area,
  lmer.df = "satterthwaite"
)
normalized_pairwise <- as.data.frame(summary(
  pairs(normalized_means, adjust = "tukey"),
  infer = c(TRUE, TRUE)
))
write.csv(normalized_pairwise, normalized_path, row.names = FALSE)

############################################################
### Multiple-choice decision rule
############################################################

highest_row <- area_means_ci[which.max(area_means_ci$emmean), ]
highest_area <- as.character(highest_row$brain_area)
other_areas <- setdiff(area_levels, highest_area)

is_significantly_higher <- vapply(other_areas, function(other_area) {
  row <- pairwise_tukey[
    (
      pairwise_tukey$contrast_area_1 == highest_area &
        pairwise_tukey$contrast_area_2 == other_area
    ) | (
      pairwise_tukey$contrast_area_1 == other_area &
        pairwise_tukey$contrast_area_2 == highest_area
    ),
  ]
  if (nrow(row) != 1) {
    stop("Could not identify a required pairwise contrast")
  }
  signed_estimate <- if (
    row$contrast_area_1 == highest_area
  ) row$estimate else -row$estimate
  isTRUE(signed_estimate > 0 && row$p.value < 0.05)
}, logical(1))

if (all(is_significantly_higher)) {
  q13_answer <- unname(area_labels[highest_area])
  decision_reason <- paste0(
    q13_answer,
    " has the largest model-estimated participation ratio and is greater ",
    "than both other areas in Tukey-adjusted comparisons (p < 0.05)."
  )
} else {
  q13_answer <- "Not enough data / no differences"
  decision_reason <- paste0(
    unname(area_labels[highest_area]),
    " has the largest model-estimated participation ratio, but is not ",
    "greater than both other areas in Tukey-adjusted comparisons (p < 0.05)."
  )
}
writeLines(
  c(
    paste("Q13 answer:", q13_answer),
    paste("Decision rule result:", decision_reason)
  ),
  answer_path
)

############################################################
### Statistical plots
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
  scale_x_discrete(labels = area_labels) +
  labs(
    title = "Model-estimated dimensionality and 95% CI",
    subtitle = "Trial-level mean rates from stimulus presentation to outcome",
    x = "Brain area",
    y = "PCA participation ratio"
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

contrast_span <- diff(range(c(pairwise_tukey$lower.CL, pairwise_tukey$upper.CL)))
pairwise_tukey$label_x <- pairwise_tukey$upper.CL + max(contrast_span * 0.04, 0.01)

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
    x = "Difference in participation ratio",
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
### Human-readable analysis record
############################################################

sink(results_path)
cat("Q13 stimulus-to-outcome population dimensionality statistics\n")
cat("============================================================\n\n")
cat("Input:", data_path, "\n")
cat("Primary response: mean PCA participation ratio across neuron subsamples\n")
cat("Independent unit: mouse\n")
cat("Mice:", nlevels(dimensionality$mouse_id), "\n")
cat("Brain areas:", paste(area_levels, collapse = ", "), "\n")
cat("Neuron matching: minimum area population separately within each mouse\n")
cat("Neuron subsamples per mouse-area: 100\n\n")

cat("Primary repeated-measures mixed model\n")
cat("-------------------------------------\n")
cat("Model: mean_participation_ratio ~ brain_area + (1 | mouse_id)\n")
cat("Degrees of freedom: Satterthwaite\n\n")
print(summary(model))

cat("\nOverall effect of brain area\n")
print(overall_effect)

cat("\nModel-estimated area means and 95% confidence intervals\n")
print(area_means_ci)

cat("\nTukey-adjusted pairwise brain-area comparisons\n")
print(pairwise_tukey)

cat("\nSensitivity model using participation ratio per sampled neuron\n")
print(summary(normalized_model))
cat("\nSensitivity-model Tukey comparisons\n")
print(normalized_pairwise)

cat("\nMultiple-choice decision\n")
cat("------------------------\n")
cat("Answer:", q13_answer, "\n")
cat(decision_reason, "\n")

cat("\nSaved statistical plots\n")
cat("Area means:", mean_plot_path, "\n")
cat("Pairwise comparisons:", pairwise_plot_path, "\n")

cat("\nSession information\n")
print(sessionInfo())
sink()

cat("Saved Q13 dimensionality statistics to", output_dir, "\n")
cat("Q13 answer:", q13_answer, "\n")
