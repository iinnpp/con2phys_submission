suppressPackageStartupMessages({
  library(lmerTest)
  library(emmeans)
  library(ggplot2)
})

data_path <- paste0(
  "D:/git_repos/con2phys_submission/results/q12/",
  "q12_auc_mouse_segment.csv"
)
output_dir <- dirname(data_path)
results_path <- file.path(
  output_dir,
  "q12_segment_stats_results.txt"
)
segment_means_path <- file.path(
  output_dir,
  "q12_segment_stats_means.csv"
)
overall_effect_path <- file.path(
  output_dir,
  "q12_segment_stats_overall_effect.csv"
)
pairwise_path <- file.path(
  output_dir,
  "q12_segment_stats_pairwise_tukey.csv"
)
mean_plot_path <- file.path(
  output_dir,
  "q12_segment_stats_means_95ci.png"
)
pairwise_plot_path <- file.path(
  output_dir,
  "q12_segment_stats_pairwise_tukey.png"
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
auc <- read.csv(data_path)

required_columns <- c(
  "mouse_id", "segment", "mean_roc_auc", "n_areas"
)
missing_columns <- setdiff(required_columns, names(auc))
if (length(missing_columns) > 0) {
  stop(
    "Missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}
if (nrow(auc) != 54) {
  stop("Expected 54 mouse-segment rows (18 mice x 3 segments)")
}
if (any(!is.finite(auc$mean_roc_auc))) {
  stop("mean_roc_auc contains non-finite values")
}
if (any(auc$mean_roc_auc < 0 | auc$mean_roc_auc > 1)) {
  stop("mean_roc_auc values must be within [0, 1]")
}
if (any(auc$n_areas != 3)) {
  stop("Each mouse-segment mean must average three brain areas")
}

segment_levels <- c(
  "trial_start_to_stim",
  "stim_to_outcome",
  "outcome_to_trial_end"
)
segment_labels <- c(
  trial_start_to_stim = "Trial start to stimulus",
  stim_to_outcome = "Stimulus to outcome",
  outcome_to_trial_end = "Outcome to trial end"
)

auc$mouse_id <- factor(auc$mouse_id)
auc$segment <- factor(auc$segment, levels = segment_levels)
if (nlevels(auc$mouse_id) != 18) {
  stop("Expected 18 mice")
}
if (anyNA(auc$segment)) {
  stop("Unexpected or missing trial segment")
}

mouse_segment_key <- paste(auc$mouse_id, auc$segment, sep = ":")
if (anyDuplicated(mouse_segment_key)) {
  stop("Duplicate mouse-segment rows found")
}
rows_per_mouse <- table(auc$mouse_id)
if (any(rows_per_mouse != 3)) {
  stop("Every mouse must have one row for each trial segment")
}

significance_label <- function(p_value) {
  ifelse(
    p_value < 0.001,
    "***",
    ifelse(p_value < 0.01, "**", ifelse(p_value < 0.05, "*", "ns"))
  )
}

############################################################
### Repeated-measures segment model and Tukey comparisons
############################################################

# There is one response per mouse and segment. The random mouse intercept
# accounts for the paired/repeated-measures design.
model <- lmer(
  mean_roc_auc ~ segment + (1 | mouse_id),
  data = auc,
  REML = TRUE
)

overall_effect <- as.data.frame(
  anova(model, ddf = "Satterthwaite")
)
overall_effect$term <- rownames(overall_effect)
rownames(overall_effect) <- NULL
overall_effect <- overall_effect[, c(
  "term",
  setdiff(names(overall_effect), "term")
)]

segment_means <- emmeans(
  model,
  ~ segment,
  lmer.df = "satterthwaite"
)
segment_means_ci <- as.data.frame(
  confint(segment_means, level = 0.95)
)
segment_means_ci$segment_label <- segment_labels[
  as.character(segment_means_ci$segment)
]

pairwise_tukey <- as.data.frame(
  summary(
    pairs(segment_means, adjust = "tukey"),
    infer = c(TRUE, TRUE)
  )
)
pairwise_tukey$significance <- significance_label(pairwise_tukey$p.value)

contrast_parts <- strsplit(as.character(pairwise_tukey$contrast), " - ")
pairwise_tukey$contrast_label <- vapply(
  contrast_parts,
  function(parts) {
    paste(segment_labels[parts[1]], segment_labels[parts[2]], sep = " - ")
  },
  character(1)
)

write.csv(segment_means_ci, segment_means_path, row.names = FALSE)
write.csv(overall_effect, overall_effect_path, row.names = FALSE)
write.csv(pairwise_tukey, pairwise_path, row.names = FALSE)

############################################################
### Statistical plots
############################################################

mean_ci_plot <- ggplot(
  segment_means_ci,
  aes(x = segment, y = emmean)
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
  scale_x_discrete(labels = segment_labels) +
  scale_y_continuous(
    limits = c(0.45, 0.75),
    breaks = seq(0.45, 0.75, by = 0.05)
  ) +
  labs(
    title = "Model-estimated mean and 95% CI",
    subtitle = "Variable C decoding (0 versus 2)",
    x = "Trial segment",
    y = "Mean cross-validated ROC-AUC"
  ) +
  theme_classic(base_size = 14) +
  theme(
    panel.grid.major = element_line(color = "grey75", linewidth = 0.4),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 18, hjust = 1),
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

contrast_span <- diff(range(
  c(pairwise_tukey$lower.CL, pairwise_tukey$upper.CL)
))
pairwise_tukey$label_x <- pairwise_tukey$upper.CL +
  max(contrast_span * 0.04, 0.005)

pairwise_plot <- ggplot(
  pairwise_tukey,
  aes(x = estimate, y = contrast_label)
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
    title = "Pairwise segment differences and 95% CI",
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
  width = 9.0,
  height = 4.7,
  dpi = 150
)

############################################################
### Human-readable analysis record
############################################################

sink(results_path)
cat("Q12 trial-segment decoding statistics\n")
cat("=====================================\n\n")
cat("Input:", data_path, "\n")
cat("Response: mean cross-validated ROC-AUC across brain areas\n")
cat("Independent unit: mouse\n")
cat("Mice:", nlevels(auc$mouse_id), "\n")
cat("Segments:", paste(segment_levels, collapse = ", "), "\n\n")

cat("Repeated-measures mixed model\n")
cat("-----------------------------\n")
cat("Model: mean_roc_auc ~ segment + (1 | mouse_id)\n")
cat("Degrees of freedom: Satterthwaite\n\n")
print(summary(model))

cat("\nOverall effect of trial segment\n")
print(overall_effect)

cat("\nModel-estimated segment means and 95% confidence intervals\n")
print(segment_means_ci)

cat("\nTukey-adjusted pairwise segment comparisons\n")
print(pairwise_tukey)

cat("\nSaved statistical plots\n")
cat("Segment means:", mean_plot_path, "\n")
cat("Pairwise comparisons:", pairwise_plot_path, "\n")

cat("\nSession information\n")
print(sessionInfo())
sink()

cat("Saved Q12 segment statistics to", output_dir, "\n")
