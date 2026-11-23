suppressPackageStartupMessages({
  library(lmerTest)
  library(emmeans)
  library(ggplot2)
})

# The filename follows the existing question-script convention. The response
# is intentionally not log-transformed: STTC can be negative, and the raw
# scale gives directly interpretable model means and pairwise differences.
data_path <- "D:/git_repos/con2phys_submission/results/q4/sttc_pairs_100ms.csv"
output_dir <- "D:/git_repos/con2phys_submission/results/q4"
results_path <- file.path(output_dir, "sttc_mixed_model_results.txt")
area_means_path <- file.path(output_dir, "sttc_mixed_model_area_means.csv")
contrasts_path <- file.path(
  output_dir,
  "sttc_mixed_model_pairwise_contrasts.csv"
)
mean_plot_path <- file.path(
  output_dir,
  "sttc_mixed_model_mean_and_95_ci.png"
)
contrast_plot_path <- file.path(
  output_dir,
  "sttc_mixed_model_pairwise_contrasts.png"
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
sttc_pairs <- read.csv(data_path)

required_columns <- c(
  "mouse_id", "brain_area", "cluster_id_a", "cluster_id_b", "sttc"
)
missing_columns <- setdiff(required_columns, names(sttc_pairs))
if (length(missing_columns) > 0) {
  stop(
    "Missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}
if (nrow(sttc_pairs) == 0) {
  stop("The pair-level STTC table is empty")
}
if (any(!is.finite(sttc_pairs$sttc))) {
  stop("STTC contains non-finite values")
}
if (any(sttc_pairs$sttc < -1 | sttc_pairs$sttc > 1)) {
  stop("STTC values must be within [-1, 1]")
}
if (any(sttc_pairs$cluster_id_a >= sttc_pairs$cluster_id_b)) {
  stop("Expected every unordered pair to satisfy cluster_id_a < cluster_id_b")
}

pair_key <- paste(
  sttc_pairs$mouse_id,
  sttc_pairs$brain_area,
  sttc_pairs$cluster_id_a,
  sttc_pairs$cluster_id_b,
  sep = ":"
)
if (anyDuplicated(pair_key)) {
  stop("Duplicate mouse-area-unit pairs found")
}

sttc_pairs$mouse_id <- factor(sttc_pairs$mouse_id)
sttc_pairs$brain_area <- factor(sttc_pairs$brain_area)
if (nlevels(sttc_pairs$mouse_id) != 18) {
  stop("Expected 18 mice")
}
if (!identical(levels(sttc_pairs$brain_area), c("1", "2", "3"))) {
  stop("Expected brain-area levels 1, 2, and 3")
}

# emmeans disables lmerTest degrees-of-freedom calculations by default above
# 3,000 observations. Raise that limit so this analysis uses the same
# Satterthwaite approach as Q1 and Q10.
emm_options(lmerTest.limit = nrow(sttc_pairs) + 1)

sink(results_path)
cat("Input:", data_path, "\n")
cat("Method: plain pairwise STTC, dt = 100 ms\n")
cat("Pairing: all unordered pairs within mouse and brain area\n")
cat("Transformation: none; STTC modeled on its original [-1, 1] scale\n")
cat("Model: STTC ~ brain area + (1 | mouse_id)\n")
cat("Selected pairs:", nrow(sttc_pairs), "\n")
cat("Selected mice:", nlevels(sttc_pairs$mouse_id), "\n\n")
cat("Pairs per brain area\n")
print(table(sttc_pairs$brain_area))

################################################
### check if brain area has an effect at all
################################################
model <- lmer(
  sttc ~ brain_area + (1 | mouse_id),
  data = sttc_pairs,
  REML = TRUE
)
cat("\nMixed model summary\n")
print(summary(model))
cat("\nOverall effect of brain area on STTC\n")
print(anova(model, ddf = "Satterthwaite"))

################################################
### check pairwise contrasts
################################################
area_means <- emmeans(
  model,
  ~ brain_area,
  lmer.df = "satterthwaite"
)
area_means_ci <- as.data.frame(confint(area_means, level = 0.95))
cat("\nModel-estimated STTC means and 95% confidence intervals\n")
print(area_means_ci)

pairwise_contrasts <- as.data.frame(
  summary(
    pairs(area_means, adjust = "tukey"),
    infer = c(TRUE, TRUE)
  )
)
cat("\nTukey-adjusted pairwise STTC differences\n")
print(pairwise_contrasts)

write.csv(area_means_ci, area_means_path, row.names = FALSE)
write.csv(pairwise_contrasts, contrasts_path, row.names = FALSE)

################################################
### Plots
################################################

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
  labs(
    title = "Model-estimated mean and 95% CI",
    x = "Brain area",
    y = "Mean STTC (dt = 100 ms)"
  ) +
  theme_classic(base_size = 14) +
  theme(
    panel.grid.major = element_line(color = "grey75", linewidth = 0.4),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

ggsave(
  mean_plot_path,
  plot = mean_ci_plot,
  width = 5.7,
  height = 4.7,
  dpi = 150
)
cat("\nSaved plot to", mean_plot_path, "\n")

pairwise_contrasts$significance <- ifelse(
  pairwise_contrasts$p.value < 0.001,
  "***",
  ifelse(
    pairwise_contrasts$p.value < 0.01,
    "**",
    ifelse(pairwise_contrasts$p.value < 0.05, "*", "ns")
  )
)
contrast_span <- diff(range(
  c(pairwise_contrasts$lower.CL, pairwise_contrasts$upper.CL)
))
pairwise_contrasts$star_x <- pairwise_contrasts$upper.CL +
  max(contrast_span * 0.04, 0.002)

contrast_plot <- ggplot(
  pairwise_contrasts,
  aes(x = estimate, y = contrast)
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
    aes(x = star_x, label = significance),
    hjust = 0,
    size = 5
  ) +
  scale_x_continuous(expand = expansion(mult = c(0.08, 0.18))) +
  labs(
    title = "Pairwise contrasts and 95% CI",
    subtitle = "Tukey-adjusted confidence intervals and p-values",
    x = "Difference in mean STTC",
    y = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    panel.grid.major.x = element_line(color = "grey75", linewidth = 0.4),
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

ggsave(
  contrast_plot_path,
  plot = contrast_plot,
  width = 7.2,
  height = 4.7,
  dpi = 150
)
cat("Saved contrast plot to", contrast_plot_path, "\n")
cat("Saved area estimates to", area_means_path, "\n")
cat("Saved pairwise contrasts to", contrasts_path, "\n")

sink()
cat("Saved mixed-model output to", results_path, "\n")
