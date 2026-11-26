# Usage from any directory: Rscript path/to/q7_waveform_mixed_model.R [results directory]
suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
repo <- normalizePath(file.path(dirname(script_path), "..", ".."), mustWork = TRUE)
output_dir <- if (length(args)) args[1] else file.path(repo, "results", "q7", "features_clustering_2features")
units <- read.csv(file.path(output_dir, "q7_sun_end_slope_gmm_bic_selected_units.csv"))
components <- read.csv(file.path(output_dir, "q7_sun_end_slope_gmm_bic_selected_components.csv"))

area_ids <- 1:3
area_labels <- paste("Area", area_ids)
if (!setequal(units$brain_area, area_ids) || !setequal(components$brain_area, area_ids) ||
    anyNA(units[c("mouse_id", "cluster_id", "brain_area", "component")]) ||
    anyDuplicated(units[c("mouse_id", "cluster_id")]) ||
    any(!is.finite(components$median_trough_to_peak_ms))) {
  stop("Expected GMM assignments and component summaries for three brain areas")
}

shortest <- do.call(rbind, lapply(split(components, components$brain_area), function(rows) {
  rows[which.min(rows$median_trough_to_peak_ms), , drop = FALSE]
}))
units$short_waveform <- as.integer(units$component == shortest$component[match(units$brain_area, shortest$brain_area)])
observed_short <- tapply(units$short_waveform, units$brain_area, sum)
if (!all(observed_short[as.character(shortest$brain_area)] == shortest$n_units)) {
  stop("The shortest-component counts disagree with the per-unit GMM assignments")
}

units$mouse_id <- factor(units$mouse_id)
units$brain_area <- factor(units$brain_area, levels = area_ids, labels = area_labels)
units$one <- 1L
mouse_area <- aggregate(
  cbind(n_short_waveform = short_waveform, n_units = one) ~ mouse_id + brain_area,
  data = units,
  FUN = sum
)
if (nrow(mouse_area) != nlevels(units$mouse_id) * length(area_ids)) {
  stop("Expected a measured proportion in each brain area for every mouse")
}
mouse_area$short_waveform_percent <- 100 * mouse_area$n_short_waveform / mouse_area$n_units

# Grouped binomial counts preserve the unit denominators and give the same fixed
# effect estimates as binary unit rows with these predictors.
model <- glmer(
  cbind(n_short_waveform, n_units - n_short_waveform) ~ brain_area + (1 | mouse_id),
  data = mouse_area,
  family = binomial(link = "logit"),
  nAGQ = 9,
  control = glmerControl(optimizer = "bobyqa")
)
null_model <- update(model, . ~ . - brain_area)
overall <- anova(null_model, model, test = "Chisq")

# The response-scale estimates are conditional probabilities at mouse effect zero.
area_means <- emmeans(model, ~ brain_area)
area_ci <- as.data.frame(confint(area_means, level = 0.95, type = "response"))
estimates <- data.frame(
  brain_area = area_ci$brain_area,
  estimate_percent = 100 * area_ci$prob,
  lower_95_percent = 100 * area_ci$asymp.LCL,
  upper_95_percent = 100 * area_ci$asymp.UCL
)

pairwise_odds <- function(estimated_means) {
  pairwise_ci <- as.data.frame(summary(
    pairs(estimated_means, reverse = TRUE, adjust = "tukey"), infer = c(TRUE, TRUE), type = "response"
  ))
  data.frame(
    contrast = sub(" / ", " vs ", pairwise_ci$contrast, fixed = TRUE),
    odds_ratio = pairwise_ci$odds.ratio,
    lower_95 = pairwise_ci$asymp.LCL,
    upper_95 = pairwise_ci$asymp.UCL,
    p_tukey = pairwise_ci$p.value
  )
}
contrasts <- pairwise_odds(area_means)
pearson_dispersion <- sum(residuals(model, type = "pearson")^2) / df.residual(model)

# Check the contrast with extra variability at the level of each binomial count.
sensitivity_data <- transform(mouse_area, mouse_area_id = factor(seq_len(nrow(mouse_area))))
sensitivity <- glmer(
  cbind(n_short_waveform, n_units - n_short_waveform) ~ brain_area + (1 | mouse_id) + (1 | mouse_area_id),
  data = sensitivity_data,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa")
)
sensitivity_contrasts <- pairwise_odds(emmeans(sensitivity, ~ brain_area))

prefix <- "q7_waveform_mixed_model"
write.csv(mouse_area, file.path(output_dir, "q7_waveform_short_component_by_mouse_area.csv"), row.names = FALSE)
write.csv(estimates, file.path(output_dir, paste0(prefix, "_area_estimates.csv")), row.names = FALSE)
write.csv(contrasts, file.path(output_dir, paste0(prefix, "_pairwise_contrasts.csv")), row.names = FALSE)
writeLines(capture.output({
  cat("Outcome: assignment to the BIC-selected GMM component with the lowest median trough-to-peak duration.\n")
  cat("GMM assignments held fixed. Grouped binomial counts for", nrow(units), "units from",
      nlevels(units$mouse_id), "mice.\n\n")
  print(summary(model))
  cat("\nSingular fit:", isSingular(model), "\n")
  cat("Pearson residual sum of squares / residual degrees of freedom:", pearson_dispersion, "\n")
  cat("\nArea likelihood-ratio test\n")
  print(overall)
  cat("\nModel-estimated percentages and pointwise 95% CIs at mouse effect zero\n")
  print(estimates, row.names = FALSE)
  cat("\nTukey-adjusted pairwise odds ratios\n")
  print(contrasts, row.names = FALSE)
  cat("\nSensitivity analysis with an additional random intercept for each mouse-area count\n")
  print(VarCorr(sensitivity))
  print(sensitivity_contrasts, row.names = FALSE)
}), file.path(output_dir, paste0(prefix, "_results.txt")))

area_colors <- c("Area 1" = "#0072B2", "Area 2" = "#D55E00", "Area 3" = "#00856A")
plot_theme <- theme_classic(base_size = 13) +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5))

estimate_plot <- ggplot(estimates, aes(x = brain_area, y = estimate_percent)) +
  geom_line(
    data = mouse_area, aes(x = brain_area, y = short_waveform_percent, group = mouse_id),
    inherit.aes = FALSE, color = "grey75", linewidth = 0.35
  ) +
  geom_point(
    data = mouse_area, aes(x = brain_area, y = short_waveform_percent),
    inherit.aes = FALSE, color = "grey58", size = 1.5, alpha = 0.75
  ) +
  geom_errorbar(aes(ymin = lower_95_percent, ymax = upper_95_percent, color = brain_area),
                width = 0.09, linewidth = 0.95) +
  geom_point(aes(color = brain_area), size = 3.5) +
  scale_color_manual(values = area_colors, guide = "none") +
  scale_y_continuous(labels = scales::label_number(suffix = "%"), limits = c(0, NA)) +
  labs(title = "Short-waveform proportion by area",
       subtitle = "Model estimates and 95% CIs in color. Individual mice in gray.",
       x = NULL, y = "Units assigned to the shortest-waveform component") +
  plot_theme + theme(panel.grid.major.y = element_line(color = "grey88", linewidth = 0.4))

contrasts$contrast <- factor(contrasts$contrast, levels = rev(contrasts$contrast))
contrasts$p_label <- ifelse(contrasts$p_tukey < .001, "p < 0.001", sprintf("p = %.3f", contrasts$p_tukey))
contrast_plot <- ggplot(contrasts, aes(x = odds_ratio, y = contrast)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbar(aes(xmin = lower_95, xmax = upper_95), orientation = "y", width = 0.13, linewidth = 0.75) +
  geom_point(size = 3, color = "#24415E") +
  geom_text(aes(x = upper_95 * 1.1, label = p_label), hjust = 0, size = 3.5) +
  scale_x_log10(expand = expansion(mult = c(0.06, 0.23))) +
  labs(title = "Pairwise contrasts and 95% CIs",
       subtitle = "Tukey-adjusted confidence intervals and p-values",
       x = "Odds ratio for short-waveform assignment (log scale)", y = NULL) +
  plot_theme + theme(panel.grid.major.x = element_line(color = "grey88", linewidth = 0.4))

ggsave(file.path(output_dir, paste0(prefix, "_mean_and_95_ci.png")), estimate_plot,
       width = 7.0, height = 4.7, dpi = 160, bg = "white")
ggsave(file.path(output_dir, paste0(prefix, "_pairwise_contrasts.png")), contrast_plot,
       width = 7.2, height = 4.7, dpi = 160, bg = "white")
print(estimates, row.names = FALSE)
print(contrasts, row.names = FALSE)
