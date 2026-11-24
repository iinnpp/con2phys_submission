suppressPackageStartupMessages({
  library(lmerTest)
  library(emmeans)
  library(ggplot2)
})

data_path <- paste0(
  "D:/git_repos/con2phys_submission/results/q11/outcome_trial_end/",
  "outcome_trial_end_auc_mouse_area.csv"
)
output_dir <- dirname(data_path)
results_path <- file.path(
  output_dir,
  "outcome_trial_end_stats_results.txt"
)
area_means_path <- file.path(
  output_dir,
  "outcome_trial_end_stats_area_means.csv"
)
chance_tests_path <- file.path(
  output_dir,
  "outcome_trial_end_stats_above_chance.csv"
)
overall_effect_path <- file.path(
  output_dir,
  "outcome_trial_end_stats_overall_area_effect.csv"
)
pairwise_path <- file.path(
  output_dir,
  "outcome_trial_end_stats_pairwise_tukey.csv"
)
mean_plot_path <- file.path(
  output_dir,
  "outcome_trial_end_stats_area_means_95ci.png"
)
pairwise_plot_path <- file.path(
  output_dir,
  "outcome_trial_end_stats_pairwise_tukey.png"
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
auc <- read.csv(data_path)

required_columns <- c(
  "mouse_id", "brain_area", "mean_roc_auc", "n_subsamples"
)
missing_columns <- setdiff(required_columns, names(auc))
if (length(missing_columns) > 0) {
  stop(
    "Missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}
if (nrow(auc) != 54) {
  stop("Expected 54 mouse-area rows (18 mice x 3 areas)")
}
if (any(!is.finite(auc$mean_roc_auc))) {
  stop("mean_roc_auc contains non-finite values")
}
if (any(auc$mean_roc_auc < 0 | auc$mean_roc_auc > 1)) {
  stop("mean_roc_auc values must be within [0, 1]")
}
if (any(auc$n_subsamples != 20)) {
  stop("Expected 20 neuron-subsample estimates per mouse and area")
}

auc$mouse_id <- factor(auc$mouse_id)
auc$brain_area <- factor(auc$brain_area, levels = c(1, 2, 3))
if (nlevels(auc$mouse_id) != 18) {
  stop("Expected 18 mice")
}
if (anyNA(auc$brain_area)) {
  stop("Expected brain-area levels 1, 2, and 3")
}

mouse_area_key <- paste(auc$mouse_id, auc$brain_area, sep = ":")
if (anyDuplicated(mouse_area_key)) {
  stop("Duplicate mouse-area rows found")
}
rows_per_mouse <- table(auc$mouse_id)
if (any(rows_per_mouse != 3)) {
  stop("Every mouse must have one row for each of the three brain areas")
}

significance_label <- function(p_value) {
  ifelse(
    p_value < 0.001,
    "***",
    ifelse(p_value < 0.01, "**", ifelse(p_value < 0.05, "*", "ns"))
  )
}

############################################################
### Above-chance tests: exact mouse-level sign-flip tests
############################################################

# The mouse is the independent unit. For each brain area, the statistic is
# the mean across mice of (mouse mean ROC-AUC - 0.5). All 2^18 sign patterns
# are enumerated, giving an exact one-sided randomization p-value under the
# usual symmetry assumption for the mouse-level differences.
n_mice <- nlevels(auc$mouse_id)
n_sign_patterns <- 2^n_mice
pattern_ids <- 0:(n_sign_patterns - 1)
sign_matrix <- vapply(
  0:(n_mice - 1),
  function(bit) {
    ifelse(
      bitwAnd(pattern_ids, bitwShiftL(1L, bit)) != 0,
      1,
      -1
    )
  },
  numeric(n_sign_patterns)
)

chance_rows <- lapply(levels(auc$brain_area), function(area) {
  area_auc <- auc$mean_roc_auc[auc$brain_area == area]
  differences <- area_auc - 0.5
  observed <- mean(differences)
  null_statistics <- as.vector(sign_matrix %*% differences) / n_mice
  p_value <- mean(null_statistics >= observed - 1e-15)

  data.frame(
    brain_area = area,
    n_mice = length(area_auc),
    mean_roc_auc = mean(area_auc),
    sd_mouse_auc = sd(area_auc),
    mean_above_chance = observed,
    median_roc_auc = median(area_auc),
    n_mice_above_chance = sum(area_auc > 0.5),
    chance_auc = 0.5,
    alternative = "mean ROC-AUC > 0.5",
    n_sign_patterns = n_sign_patterns,
    p_value = p_value,
    stringsAsFactors = FALSE
  )
})
chance_tests <- do.call(rbind, chance_rows)
chance_tests$significance <- significance_label(chance_tests$p_value)

############################################################
### Repeated-measures area model and Tukey comparisons
############################################################

# There is one response per mouse and area. The random mouse intercept
# accounts for the paired/repeated-measures design.
model <- lmer(
  mean_roc_auc ~ brain_area + (1 | mouse_id),
  data = auc,
  REML = TRUE
)

overall_effect <- as.data.frame(
  anova(model, ddf = "Satterthwaite")
)
overall_effect$term <- rownames(overall_effect)
rownames(overall_effect) <- NULL
overall_effect <- overall_effect[, c("term", setdiff(names(overall_effect), "term"))]

area_means <- emmeans(
  model,
  ~ brain_area,
  lmer.df = "satterthwaite"
)
area_means_ci <- as.data.frame(confint(area_means, level = 0.95))

pairwise_tukey <- as.data.frame(
  summary(
    pairs(area_means, adjust = "tukey"),
    infer = c(TRUE, TRUE)
  )
)
pairwise_tukey$significance <- significance_label(pairwise_tukey$p.value)

write.csv(area_means_ci, area_means_path, row.names = FALSE)
write.csv(chance_tests, chance_tests_path, row.names = FALSE)
write.csv(overall_effect, overall_effect_path, row.names = FALSE)
write.csv(pairwise_tukey, pairwise_path, row.names = FALSE)

############################################################
### Statistical plots
############################################################

mean_ci_plot <- ggplot(
  area_means_ci,
  aes(x = brain_area, y = emmean)
) +
  geom_hline(
    yintercept = 0.5,
    linetype = "dashed",
    color = "#D62728",
    linewidth = 0.7
  ) +
  geom_errorbar(
    aes(ymin = lower.CL, ymax = upper.CL),
    width = 0.08,
    linewidth = 0.7
  ) +
  geom_point(size = 3) +
  scale_y_continuous(
    limits = c(0.45, 1.0),
    breaks = seq(0.5, 1.0, by = 0.1)
  ) +
  labs(
    title = "Model-estimated mean and 95% CI",
    subtitle = "Outcome to trial end",
    x = "Brain area",
    y = "Mean cross-validated ROC-AUC"
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
  width = 5.7,
  height = 4.7,
  dpi = 150
)

contrast_span <- diff(range(
  c(pairwise_tukey$lower.CL, pairwise_tukey$upper.CL)
))
pairwise_tukey$label_x <- pairwise_tukey$upper.CL +
  max(contrast_span * 0.04, 0.005)

pairwise_plot <- ggplot(
  pairwise_tukey,
  aes(x = estimate, y = contrast)
) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey50"
  ) +
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
    title = "Pairwise differences and 95% CI",
    subtitle = "Tukey-adjusted confidence intervals and p-values",
    x = "Difference in mean ROC-AUC",
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
  width = 7.2,
  height = 4.7,
  dpi = 150
)

############################################################
### Human-readable analysis record
############################################################

sink(results_path)
cat("Q11 outcome-to-trial-end decoding statistics\n")
cat("=============================================\n\n")
cat("Input:", data_path, "\n")
cat("Response: mean cross-validated ROC-AUC across 20 neuron subsamples\n")
cat("Independent unit: mouse\n")
cat("Mice:", n_mice, "\n")
cat("Brain areas: 1, 2, 3\n\n")

cat("Above-chance analysis\n")
cat("---------------------\n")
cat(
  "Exact one-sided sign-flip tests of mouse mean ROC-AUC against 0.5.\n",
  "Tukey adjustment does not apply to tests against a fixed benchmark;\n",
  "these planned area-specific tests are reported separately.\n\n",
  sep = ""
)
print(chance_tests)

cat("\nRepeated-measures mixed model\n")
cat("-----------------------------\n")
cat("Model: mean_roc_auc ~ brain_area + (1 | mouse_id)\n")
cat("Degrees of freedom: Satterthwaite\n\n")
print(summary(model))

cat("\nOverall effect of brain area\n")
print(overall_effect)

cat("\nModel-estimated area means and 95% confidence intervals\n")
print(area_means_ci)

cat("\nTukey-adjusted pairwise area comparisons\n")
print(pairwise_tukey)

cat("\nSaved statistical plots\n")
cat("Area means:", mean_plot_path, "\n")
cat("Pairwise comparisons:", pairwise_plot_path, "\n")

cat("\nSession information\n")
print(sessionInfo())
sink()

cat("Saved statistical results to", output_dir, "\n")
