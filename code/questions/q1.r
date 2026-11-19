suppressPackageStartupMessages({
  library(lmerTest)
  library(emmeans)
  library(ggplot2)
})

data_path <- "E:/con2phys_submission/questions/q1/fr.csv"
results_path <- "D:/git_repos/con2phys_submission/results/q1/fr_mixed_model_results.txt"
sink(results_path)

fr <- read.csv(data_path)

fr$mouse_id <- factor(fr$mouse_id)
fr$brain_area <- factor(fr$brain_area)

################################################
### check if brain area has an effect at all
################################################
model <- lmer(
  fr_hz ~ brain_area + (1 | mouse_id),
  data = fr,
  REML = TRUE
)
print(summary(model))
cat("\nOverall effect of brain area\n")
print(anova(model, ddf = "Satterthwaite"))

################################################
### check pairwise contrast
################################################
area_means <- emmeans(model, ~ brain_area)
area_means_ci <- as.data.frame(confint(area_means, level = 0.95))
cat("\nEstimated area means and 95% confidence intervals\n")
print(area_means_ci)

pairwise_contrasts <- as.data.frame(
  summary(
    pairs(area_means, adjust = "tukey"),
    infer = c(TRUE, TRUE)
  )
)
cat("\nTukey adjusted pairwise contrasts\n")
print(pairwise_contrasts)

################################################
### Plots
################################################

# plot model-estimated means and 95% CI
plot_path <- file.path(
  dirname(data_path),
  "fr_mixed_model_mean_and_95_ci.png"
)

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
    title = "Mean and 95% CI",
    x = "Brain area",
    y = "Mean firing rate (Hz)"
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

# plot pairwise contrasts and their CI
pairwise_contrasts$significance <- ifelse(
  pairwise_contrasts$p.value < 0.001,
  "***",
  ifelse(
    pairwise_contrasts$p.value < 0.01,
    "**",
    ifelse(pairwise_contrasts$p.value < 0.05, "*", "ns")
  )
)
pairwise_contrasts$star_x <- pairwise_contrasts$upper.CL + 0.2

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
    x = "Difference in mean firing rate (Hz)",
    y = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    panel.grid.major.x = element_line(color = "grey75", linewidth = 0.4),
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

contrast_plot_path <- file.path(
  dirname(data_path),
  "fr_mixed_model_pairwise_contrasts.png"
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
cat("Saved model output to", results_path, "\n")
