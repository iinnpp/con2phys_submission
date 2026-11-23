suppressPackageStartupMessages({
  library(lmerTest)
  library(emmeans)
  library(ggplot2)
})

data_path <- "D:/git_repos/con2phys_submission/results/q10/sua_it_methods.csv"
output_dir <- "D:/git_repos/con2phys_submission/results/q10"
#results_path <- file.path(
#  output_dir,
#  "sttc_concat_tau_log_mixed_model_results.txt"
#)
results_path <- file.path(
  output_dir,
  "pearsonr_avg_tau_log_mixed_model_results.txt"
)


dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
sink(results_path)

timescales <- read.csv(data_path)

required_columns <- c(
  "mouse_id", "brain_area", "method", "tau_ms", "qc_pass"
)
missing_columns <- setdiff(required_columns, names(timescales))
if (length(missing_columns) > 0) {
  stop(
    "Missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

# pandas writes Boolean CSV values as "True"/"False"; normalize them
# explicitly because read.csv may keep that column as character.
timescales$qc_pass <- tolower(trimws(as.character(timescales$qc_pass))) %in%
  c("true", "1")

tau <- timescales[
  #timescales$method == "sttc_trial_concat" &
  timescales$method == "pearsonr_trial_avg" &
    timescales$qc_pass %in% TRUE &
    is.finite(timescales$tau_ms) &
    timescales$tau_ms >= 50 &
    timescales$tau_ms <= 1000,
  ,
  drop = FALSE
]

if (nrow(tau) == 0) {
  stop("No units passed the STTC-concat QC and 50--1000 ms filters")
}
if (any(!is.finite(tau$tau_ms)) || any(tau$tau_ms <= 0)) {
  stop("Log transformation requires finite timescales greater than zero")
}

tau$mouse_id <- factor(tau$mouse_id)
tau$brain_area <- factor(tau$brain_area)

cat("Input:", data_path, "\n")
cat("Method: sttc_trial_concat\n")
cat("Selection: qc_pass and inclusive 50--1000 ms range\n")
cat("Selected units:", nrow(tau), "\n")
cat("Selected mice:", nlevels(tau$mouse_id), "\n\n")
cat("Units per brain area\n")
print(table(tau$brain_area))

################################################
### check if brain area has an effect at all
################################################
model <- lmer(
  log(tau_ms) ~ brain_area + (1 | mouse_id),
  data = tau,
  REML = TRUE
)
print(summary(model))
cat("\nOverall effect of brain area on log timescale\n")
print(anova(model, ddf = "Satterthwaite"))

################################################
### check pairwise contrasts
################################################
area_means <- emmeans(model, ~ brain_area)
area_means_ci <- as.data.frame(
  confint(area_means, level = 0.95, type = "response")
)
cat("\nBack-transformed geometric means and 95% confidence intervals\n")
print(area_means_ci)

pairwise_contrasts <- as.data.frame(
  summary(
    pairs(area_means, adjust = "tukey"),
    infer = c(TRUE, TRUE),
    type = "response"
  )
)
cat("\nTukey-adjusted pairwise ratios\n")
print(pairwise_contrasts)

################################################
### Plots
################################################

# plot back-transformed geometric means and 95% CI
plot_path <- file.path(
  output_dir,
  #"sttc_concat_tau_log_mixed_model_geometric_mean_and_95_ci.png"
  "pearsonr_avg_tau_log_mixed_model_geometric_mean_and_95_ci.png"
)

mean_ci_plot <- ggplot(
  area_means_ci,
  aes(x = brain_area, y = response)
) +
  geom_errorbar(
    aes(ymin = lower.CL, ymax = upper.CL),
    width = 0.08,
    linewidth = 0.7
  ) +
  geom_point(size = 3) +
  labs(
    title = "Geometric mean and 95% CI",
    x = "Brain area",
    y = "Geometric mean timescale (ms)"
  ) +
  theme_classic(base_size = 14) +
  theme(
    panel.grid.major = element_line(color = "grey75", linewidth = 0.4),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5)
  )

print(mean_ci_plot)
ggsave(
  plot_path,
  plot = mean_ci_plot,
  width = 5.7,
  height = 4.7,
  dpi = 150
)

cat("\nSaved plot to", plot_path, "\n")

# plot pairwise ratios and their CI
pairwise_contrasts$significance <- ifelse(
  pairwise_contrasts$p.value < 0.001,
  "***",
  ifelse(
    pairwise_contrasts$p.value < 0.01,
    "**",
    ifelse(pairwise_contrasts$p.value < 0.05, "*", "ns")
  )
)
pairwise_contrasts$star_x <- pairwise_contrasts$upper.CL * 1.05

contrast_plot <- ggplot(
  pairwise_contrasts,
  aes(x = ratio, y = contrast)
) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
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
    title = "Pairwise ratios and 95% CI",
    subtitle = "Tukey-adjusted confidence intervals and p-values",
    x = "Ratio of geometric mean timescales",
    y = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    panel.grid.major.x = element_line(color = "grey75", linewidth = 0.4),
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

contrast_plot_path <- file.path(
  output_dir,
  #"sttc_concat_tau_log_mixed_model_pairwise_ratios.png"
  "pearsonr_avg_tau_log_mixed_model_pairwise_ratios.png"
)

print(contrast_plot)
ggsave(
  contrast_plot_path,
  plot = contrast_plot,
  width = 7.2,
  height = 4.7,
  dpi = 150
)

cat("Saved contrast plot to", contrast_plot_path, "\n")

sink()
cat("Saved log-model output to", results_path, "\n")
