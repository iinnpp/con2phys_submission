suppressPackageStartupMessages({
  library(lmerTest)
  library(emmeans)
  library(ggplot2)
})

output_dir <- Sys.getenv(
  "Q15_RESULTS_DIR",
  unset = "D:/git_repos/con2phys_submission/results/q15"
)
expected_subsamples <- as.integer(Sys.getenv(
  "Q15_EXPECTED_SUBSAMPLES",
  unset = "20"
))
if (!is.finite(expected_subsamples) || expected_subsamples <= 0) {
  stop("Q15_EXPECTED_SUBSAMPLES must be a positive integer")
}
data_path <- file.path(output_dir, "q15_lzc_mouse_area_bin.csv")
results_path <- file.path(output_dir, "q15_complexity_stats_results.txt")
means_path <- file.path(output_dir, "q15_complexity_stats_means.csv")
overall_effect_path <- file.path(output_dir, "q15_complexity_stats_overall_effect.csv")
pairwise_path <- file.path(output_dir, "q15_complexity_stats_pairwise_tukey.csv")
sensitivity_effects_path <- file.path(output_dir, "q15_complexity_bin_width_sensitivity_effects.csv")
sensitivity_means_path <- file.path(output_dir, "q15_complexity_bin_width_sensitivity_means.csv")
sensitivity_pairwise_path <- file.path(output_dir, "q15_complexity_bin_width_sensitivity_pairwise.csv")
answer_path <- file.path(output_dir, "answer.txt")
mean_plot_path <- file.path(output_dir, "q15_complexity_stats_means_95ci.png")
pairwise_plot_path <- file.path(output_dir, "q15_complexity_stats_pairwise_tukey.png")
sensitivity_plot_path <- file.path(output_dir, "q15_complexity_bin_width_sensitivity.png")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
complexity <- read.csv(data_path)

required_columns <- c(
  "mouse_id", "brain_area", "bin_width_ms",
  "mean_normalized_lz_complexity", "sd_normalized_lz_complexity",
  "mean_observed_compression_ratio", "mean_spike_bin_density",
  "n_trials", "n_time_bins", "total_analyzed_time_sec",
  "n_available_units", "n_sampled_units", "n_subsamples",
  "n_rate_surrogates"
)
missing_columns <- setdiff(required_columns, names(complexity))
if (length(missing_columns) > 0) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "))
}
if (nrow(complexity) != 162) {
  stop("Expected 162 mouse-area-bin rows (18 mice x 3 areas x 3 bin widths)")
}
numeric_columns <- setdiff(required_columns, c("mouse_id", "brain_area"))
if (any(!is.finite(as.matrix(complexity[, numeric_columns])))) {
  stop("Input contains non-finite numeric values")
}
if (any(complexity$mean_normalized_lz_complexity <= 0)) {
  stop("Normalized Lempel-Ziv complexity values must be positive")
}
if (any(complexity$mean_spike_bin_density < 0 | complexity$mean_spike_bin_density > 1)) {
  stop("Spike-bin density must be within [0, 1]")
}
if (any(complexity$n_subsamples != expected_subsamples)) {
  stop(
    "Expected ", expected_subsamples,
    " neuron subsamples per mouse-area-bin"
  )
}
if (any(complexity$n_rate_surrogates != 1)) {
  stop("Expected one rate-matched surrogate per subsample")
}

area_levels <- c("1", "2", "3")
area_labels <- c(
  `1` = "Brain area 1",
  `2` = "Brain area 2",
  `3` = "Brain area 3"
)
expected_bin_widths <- c(10, 20, 50)
primary_bin_width <- 20

complexity$mouse_id <- factor(complexity$mouse_id)
complexity$brain_area <- factor(complexity$brain_area, levels = area_levels)
complexity$bin_width_factor <- factor(
  complexity$bin_width_ms,
  levels = expected_bin_widths,
  labels = paste0(expected_bin_widths, " ms")
)
if (nlevels(complexity$mouse_id) != 18) {
  stop("Expected 18 mice")
}
if (anyNA(complexity$brain_area) || anyNA(complexity$bin_width_factor)) {
  stop("Unexpected brain area or bin width")
}

row_key <- paste(
  complexity$mouse_id,
  complexity$brain_area,
  complexity$bin_width_ms,
  sep = ":"
)
if (anyDuplicated(row_key)) {
  stop("Duplicate mouse-area-bin rows found")
}
rows_per_mouse <- table(complexity$mouse_id)
if (any(rows_per_mouse != 9)) {
  stop("Every mouse must have three areas at each of three bin widths")
}
sampled_counts_per_mouse_bin <- tapply(
  complexity$n_sampled_units,
  list(complexity$mouse_id, complexity$bin_width_ms),
  function(values) length(unique(values))
)
if (any(sampled_counts_per_mouse_bin != 1)) {
  stop("Neuron count is not matched across areas within mouse and bin width")
}

significance_label <- function(p_value) {
  ifelse(
    p_value < 0.001,
    "***",
    ifelse(p_value < 0.01, "**", ifelse(p_value < 0.05, "*", "ns"))
  )
}

############################################################
### Primary repeated-measures model: 20 ms bins
############################################################

primary <- complexity[complexity$bin_width_ms == primary_bin_width, ]
if (nrow(primary) != 54) {
  stop("Expected 54 mouse-area rows for the primary 20 ms analysis")
}

# Complexity is averaged over neuron subsamples before inference. Mouse is
# the independent unit; its random intercept accounts for the three paired
# brain-area observations.
primary_model <- lmer(
  mean_normalized_lz_complexity ~ brain_area + (1 | mouse_id),
  data = primary,
  REML = TRUE
)

overall_effect <- as.data.frame(anova(primary_model, ddf = "Satterthwaite"))
overall_effect$term <- rownames(overall_effect)
rownames(overall_effect) <- NULL
overall_effect <- overall_effect[, c("term", setdiff(names(overall_effect), "term"))]

area_means <- emmeans(primary_model, ~ brain_area, lmer.df = "satterthwaite")
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
### Bin-width sensitivity analysis
############################################################

sensitivity_model <- lmer(
  mean_normalized_lz_complexity ~ brain_area * bin_width_factor +
    (1 | mouse_id),
  data = complexity,
  REML = TRUE
)
sensitivity_effects <- as.data.frame(anova(
  sensitivity_model,
  ddf = "Satterthwaite"
))
sensitivity_effects$term <- rownames(sensitivity_effects)
rownames(sensitivity_effects) <- NULL
sensitivity_effects <- sensitivity_effects[, c(
  "term", setdiff(names(sensitivity_effects), "term")
)]

sensitivity_means <- emmeans(
  sensitivity_model,
  ~ brain_area | bin_width_factor,
  lmer.df = "satterthwaite"
)
sensitivity_means_ci <- as.data.frame(confint(sensitivity_means, level = 0.95))
sensitivity_means_ci$brain_area_label <- area_labels[
  as.character(sensitivity_means_ci$brain_area)
]

sensitivity_pairwise <- as.data.frame(summary(
  pairs(sensitivity_means, adjust = "tukey"),
  infer = c(TRUE, TRUE)
))
sensitivity_pairwise$significance <- significance_label(sensitivity_pairwise$p.value)

write.csv(sensitivity_effects, sensitivity_effects_path, row.names = FALSE)
write.csv(sensitivity_means_ci, sensitivity_means_path, row.names = FALSE)
write.csv(sensitivity_pairwise, sensitivity_pairwise_path, row.names = FALSE)

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
  q15_answer <- unname(area_labels[highest_area])
  decision_reason <- paste0(
    q15_answer,
    " has the largest model-estimated normalized Lempel-Ziv complexity and ",
    "is greater than both other areas in Tukey-adjusted comparisons (p < 0.05)."
  )
} else {
  q15_answer <- "Not enough data / no differences"
  decision_reason <- paste0(
    unname(area_labels[highest_area]),
    " has the largest model-estimated normalized Lempel-Ziv complexity, but ",
    "is not greater than both other areas in Tukey-adjusted comparisons (p < 0.05)."
  )
}
format_p <- function(value) {
  if (value < 0.001) {
    format(value, scientific = TRUE, digits = 3)
  } else {
    sprintf("%.4f", value)
  }
}

mean_1 <- area_means_ci[area_means_ci$brain_area == "1", ]
mean_2 <- area_means_ci[area_means_ci$brain_area == "2", ]
mean_3 <- area_means_ci[area_means_ci$brain_area == "3", ]
contrast_12 <- pairwise_tukey[
  pairwise_tukey$contrast_area_1 == "1" &
    pairwise_tukey$contrast_area_2 == "2",
]
contrast_13 <- pairwise_tukey[
  pairwise_tukey$contrast_area_1 == "1" &
    pairwise_tukey$contrast_area_2 == "3",
]
contrast_23 <- pairwise_tukey[
  pairwise_tukey$contrast_area_1 == "2" &
    pairwise_tukey$contrast_area_2 == "3",
]
if (
  nrow(mean_1) != 1 || nrow(mean_2) != 1 || nrow(mean_3) != 1 ||
    nrow(contrast_12) != 1 || nrow(contrast_13) != 1 ||
    nrow(contrast_23) != 1
) {
  stop("Could not identify all area means and primary pairwise contrasts")
}

primary_effect <- overall_effect[overall_effect$term == "brain_area", ]
sensitivity_area_effect <- sensitivity_effects[
  sensitivity_effects$term == "brain_area",
]
sensitivity_width_effect <- sensitivity_effects[
  sensitivity_effects$term == "bin_width_factor",
]
sensitivity_interaction <- sensitivity_effects[
  sensitivity_effects$term == "brain_area:bin_width_factor",
]

answer_paragraphs <- c(
  paste0(
    "Population spiking complexity was examined from stimulus presentation to trial end. Trial selection used exactly ",
    "the same filtered trial table as Q11-Q13. A trial was retained only if trial_start, stim_start, outcome, and ",
    "trial_end were all present and strictly ordered as trial_start < stim_start < outcome < trial_end. The final ",
    "recorded trial for each mouse, defined as the trial with the maximum trial_id, was excluded. The source table ",
    "contained 10,373 trials from 18 mice. The completeness and event-order criteria retained 10,361 trials; seven ",
    "final trials had already failed those criteria, and excluding the remaining 11 final trials produced a dataset ",
    "of 10,350 trials."
  ),
  paste0(
    "For each mouse and brain area, single-unit spikes in the half-open interval from stim_start to trial_end, ",
    "[stim_start, trial_end), were converted to binary population rasters. Only complete time bins were retained from ",
    "each trial, and trials were concatenated chronologically. The primary analysis used 20 ms bins; 10 and 50 ms ",
    "bins were analyzed as temporal-resolution sensitivity checks. Each raster was serialized in time-major order, ",
    "packed into bits, and compressed with level-1 DEFLATE, an LZ77-family compressor. Complexity was quantified as ",
    "the observed compressed size divided by the compressed size of a rate-matched surrogate raster. Thus, larger ",
    "values indicate less compressible population activity relative to independent spike timing with the same ",
    "per-neuron binned spike counts."
  ),
  paste0(
    "To prevent areas with more recorded neurons from receiving an advantage, the three areas were matched separately ",
    "within each mouse to the size of that mouse's smallest recorded area population. Matched population sizes ranged ",
    "from 5 to 32 neurons. Units were randomly subsampled and ordered ", expected_subsamples, " times. One independently ",
    "seeded rate-matched surrogate was generated for every subsample by redistributing each selected neuron's active ",
    "bins while preserving its total binned spike count. Complexity was averaged across the ", expected_subsamples,
    " subsamples, and the resulting mouse-level mean for each brain area was used as the independent observation for ",
    "statistical inference."
  ),
  paste0(
    "Differences among brain areas in the primary 20 ms analysis were tested with a linear mixed-effects model using ",
    "brain area as a fixed effect and mouse_id as a random intercept. The model showed an overall effect of brain area ",
    "on rate-normalized Lempel-Ziv complexity (F(", sprintf("%.0f", primary_effect$NumDF), ", ",
    sprintf("%.0f", primary_effect$DenDF), ") = ", sprintf("%.2f", primary_effect[["F value"]]), ", p = ",
    format_p(primary_effect[["Pr(>F)"]]), "). Model-estimated mean complexity was ",
    sprintf("%.4f", mean_1$emmean), " for area 1 (95% CI: ", sprintf("%.4f", mean_1$lower.CL), "-",
    sprintf("%.4f", mean_1$upper.CL), "), ", sprintf("%.4f", mean_2$emmean), " for area 2 (",
    sprintf("%.4f", mean_2$lower.CL), "-", sprintf("%.4f", mean_2$upper.CL), "), and ",
    sprintf("%.4f", mean_3$emmean), " for area 3 (", sprintf("%.4f", mean_3$lower.CL), "-",
    sprintf("%.4f", mean_3$upper.CL), ")."
  ),
  paste0(
    "Tukey-adjusted pairwise comparisons showed that complexity was higher in area 1 than in area 2. The area 1 minus ",
    "area 2 difference was ", sprintf("%.4f", contrast_12$estimate), " (95% CI: ",
    sprintf("%.4f", contrast_12$lower.CL), " to ", sprintf("%.4f", contrast_12$upper.CL), ", p = ",
    format_p(contrast_12$p.value), "). Area 1 had a descriptively higher value than area 3, but this contrast was not ",
    "significant after Tukey correction: the area 1 minus area 3 difference was ",
    sprintf("%.4f", contrast_13$estimate), " (", sprintf("%.4f", contrast_13$lower.CL), " to ",
    sprintf("%.4f", contrast_13$upper.CL), ", p = ", format_p(contrast_13$p.value), "). Areas 2 and 3 did not differ: ",
    "the area 2 minus area 3 difference was ", sprintf("%.4f", contrast_23$estimate), " (",
    sprintf("%.4f", contrast_23$lower.CL), " to ", sprintf("%.4f", contrast_23$upper.CL), ", p = ",
    format_p(contrast_23$p.value), ")."
  ),
  paste0(
    "The bin-width sensitivity model retained an overall brain-area effect (F(",
    sprintf("%.0f", sensitivity_area_effect$NumDF), ", ", sprintf("%.0f", sensitivity_area_effect$DenDF), ") = ",
    sprintf("%.2f", sensitivity_area_effect[["F value"]]), ", p = ",
    format_p(sensitivity_area_effect[["Pr(>F)"]]), "), but showed neither an overall bin-width effect (p = ",
    format_p(sensitivity_width_effect[["Pr(>F)"]]), ") nor a brain-area-by-bin-width interaction (p = ",
    format_p(sensitivity_interaction[["Pr(>F)"]]), "). The descriptive ranking was therefore stable across 10, 20, ",
    "and 50 ms bins."
  ),
  paste0(
    "Therefore, brain area 1 had the highest descriptive population spiking complexity from stimulus presentation to ",
    "trial end. However, it was not significantly higher than both other areas in the prespecified primary 20 ms ",
    "analysis. Under the conservative multiple-choice decision rule used for the preceding questions, the Q15 answer ",
    "is: ", q15_answer, "."
  ),
  paste0(
    "Code: D:\\git_repos\\con2phys_submission\\code\\questions\\q15_complexity.py, ",
    "D:\\git_repos\\con2phys_submission\\code\\questions\\q15_complexity_stats.R"
  ),
  paste0(
    "Model output: ", file.path(output_dir, "q15_complexity_stats_results.txt")
  )
)

answer_lines <- unlist(lapply(seq_along(answer_paragraphs), function(index) {
  paragraph_lines <- strwrap(answer_paragraphs[index], width = 115)
  if (index < length(answer_paragraphs)) {
    c(paragraph_lines, "")
  } else {
    paragraph_lines
  }
}))
writeLines(answer_lines, answer_path)

############################################################
### Statistical plots
############################################################

mean_ci_plot <- ggplot(area_means_ci, aes(x = brain_area, y = emmean)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbar(
    aes(ymin = lower.CL, ymax = upper.CL),
    width = 0.08,
    linewidth = 0.7
  ) +
  geom_point(size = 3) +
  scale_x_discrete(labels = area_labels) +
  labs(
    title = "Model-estimated spiking complexity and 95% CI",
    subtitle = "Stimulus to trial end; primary analysis with 20 ms bins",
    x = "Brain area",
    y = "Rate-normalized LZ compression complexity"
  ) +
  theme_classic(base_size = 14) +
  theme(
    panel.grid.major = element_line(color = "grey75", linewidth = 0.4),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

ggsave(mean_plot_path, plot = mean_ci_plot, width = 7.2, height = 4.8, dpi = 150)

contrast_span <- diff(range(c(pairwise_tukey$lower.CL, pairwise_tukey$upper.CL)))
pairwise_tukey$label_x <- pairwise_tukey$upper.CL + max(contrast_span * 0.04, 0.001)
pairwise_plot <- ggplot(pairwise_tukey, aes(x = estimate, y = contrast_label)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbar(
    aes(xmin = lower.CL, xmax = upper.CL),
    orientation = "y",
    width = 0.12,
    linewidth = 0.7
  ) +
  geom_point(size = 3) +
  geom_text(aes(x = label_x, label = significance), hjust = 0, size = 5) +
  scale_x_continuous(expand = expansion(mult = c(0.08, 0.18))) +
  labs(
    title = "Pairwise brain-area differences and 95% CI",
    subtitle = "20 ms bins; Tukey-adjusted confidence intervals and p-values",
    x = "Difference in normalized LZ complexity",
    y = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    panel.grid.major.x = element_line(color = "grey75", linewidth = 0.4),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

ggsave(pairwise_plot_path, plot = pairwise_plot, width = 9.0, height = 4.8, dpi = 150)

sensitivity_plot <- ggplot(
  sensitivity_means_ci,
  aes(x = bin_width_factor, y = emmean, color = brain_area, group = brain_area)
) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey60") +
  geom_errorbar(
    aes(ymin = lower.CL, ymax = upper.CL),
    width = 0.08,
    linewidth = 0.6,
    position = position_dodge(width = 0.12)
  ) +
  geom_point(size = 2.8, position = position_dodge(width = 0.12)) +
  geom_line(linewidth = 0.8) +
  scale_color_discrete(labels = area_labels) +
  labs(
    title = "Spiking-complexity sensitivity to bin width",
    x = "Spike-raster bin width",
    y = "Rate-normalized LZ compression complexity",
    color = "Brain area"
  ) +
  theme_classic(base_size = 14) +
  theme(
    panel.grid.major = element_line(color = "grey80", linewidth = 0.35),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5),
    legend.position = "top"
  )

ggsave(
  sensitivity_plot_path,
  plot = sensitivity_plot,
  width = 7.8,
  height = 5.0,
  dpi = 150
)

############################################################
### Human-readable analysis record
############################################################

sink(results_path)
cat("Q15 stimulus-to-trial-end population spiking complexity statistics\n")
cat("==================================================================\n\n")
cat("Input:", data_path, "\n")
cat("Primary response: mean rate-normalized Lempel-Ziv compression complexity\n")
cat("Definition: observed DEFLATE compressed bytes divided by mean rate-matched surrogate compressed bytes\n")
cat("Primary bin width: 20 ms\n")
cat("Independent unit: mouse\n")
cat("Mice:", nlevels(complexity$mouse_id), "\n")
cat("Brain areas:", paste(area_levels, collapse = ", "), "\n")
cat("Neuron matching: minimum area population separately within each mouse\n")
cat("Neuron subsamples per mouse-area:", expected_subsamples, "\n")
cat("Rate-matched surrogates per neuron subsample: 1 (20 independently seeded surrogates per mouse-area-bin mean)\n\n")

cat("Primary repeated-measures mixed model\n")
cat("-------------------------------------\n")
cat("Model: mean_normalized_lz_complexity ~ brain_area + (1 | mouse_id)\n")
cat("Degrees of freedom: Satterthwaite\n\n")
print(summary(primary_model))

cat("\nOverall effect of brain area\n")
print(overall_effect)

cat("\nModel-estimated area means and 95% confidence intervals\n")
print(area_means_ci)

cat("\nTukey-adjusted pairwise brain-area comparisons\n")
print(pairwise_tukey)

cat("\nBin-width sensitivity model\n")
cat("---------------------------\n")
cat("Model: mean_normalized_lz_complexity ~ brain_area * bin_width_factor + (1 | mouse_id)\n\n")
print(summary(sensitivity_model))
cat("\nSensitivity-model effects\n")
print(sensitivity_effects)
cat("\nArea means by bin width\n")
print(sensitivity_means_ci)
cat("\nTukey-adjusted area comparisons within bin width\n")
print(sensitivity_pairwise)

cat("\nMultiple-choice decision\n")
cat("------------------------\n")
cat("Answer:", q15_answer, "\n")
cat(decision_reason, "\n")

cat("\nSaved statistical plots\n")
cat("Primary area means:", mean_plot_path, "\n")
cat("Primary pairwise comparisons:", pairwise_plot_path, "\n")
cat("Bin-width sensitivity:", sensitivity_plot_path, "\n")

cat("\nSession information\n")
print(sessionInfo())
sink()

cat("Saved Q15 complexity statistics to", output_dir, "\n")
cat("Q15 answer:", q15_answer, "\n")
