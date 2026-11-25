suppressPackageStartupMessages({
  library(lmerTest)
  library(emmeans)
  library(ggplot2)
})

output_dir <- Sys.getenv(
  "Q5_OUTPUT_DIR", unset = "D:/git_repos/con2phys_submission/results/q5"
)
data_path <- file.path(output_dir, "sttc_pairs_100ms.csv")
results_path <- file.path(output_dir, "sttc_mixed_model_results.txt")
means_path <- file.path(output_dir, "sttc_mixed_model_area_pair_means.csv")
contrasts_path <- file.path(output_dir, "sttc_mixed_model_pairwise_contrasts.csv")
answer_path <- file.path(output_dir, "answer.txt")
area_pair_counts <- c("1-2" = 18560L, "1-3" = 11598L, "2-3" = 9167L)
area_pair_labels <- names(area_pair_counts)
alpha <- 0.05

sttc_pairs <- read.csv(data_path)
required_columns <- c(
  "mouse_id", "area_pair", "brain_area_a", "brain_area_b",
  "cluster_id_a", "cluster_id_b", "sttc_dt_sec", "sttc"
)
missing_columns <- setdiff(required_columns, names(sttc_pairs))
if (length(missing_columns) > 0L) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "))
}
if (anyNA(sttc_pairs[required_columns]) ||
    any(!is.finite(sttc_pairs$sttc) | abs(sttc_pairs$sttc) > 1)) {
  stop("Missing data or STTC outside [-1, 1]")
}
if (any(sttc_pairs$sttc_dt_sec != 0.1)) {
  stop("Expected a 100 ms STTC half-window")
}
if (any(sttc_pairs$area_pair != paste(sttc_pairs$brain_area_a, sttc_pairs$brain_area_b, sep = "-")) ||
    any(!sttc_pairs$area_pair %in% area_pair_labels) ||
    any(sttc_pairs$cluster_id_a == sttc_pairs$cluster_id_b)) {
  stop("Expected distinct units with endpoints ordered by brain area")
}
if (anyDuplicated(sttc_pairs[c("mouse_id", "area_pair", "cluster_id_a", "cluster_id_b")])) {
  stop("Duplicate cross-area unit pairs found")
}

sttc_pairs$mouse_id <- factor(sttc_pairs$mouse_id)
sttc_pairs$area_pair <- factor(sttc_pairs$area_pair, levels = area_pair_labels)
observed_counts <- table(sttc_pairs$mouse_id, sttc_pairs$area_pair)
if (nrow(observed_counts) != 18L || any(observed_counts == 0L) ||
    any(colSums(observed_counts) != area_pair_counts)) {
  stop("Expected the full Q5 dataset with all three area pairs in each of 18 mice")
}

# Use the original STTC scale and the same REML/Satterthwaite analysis as Q4.
# Raise the emmeans limit to retain Satterthwaite degrees of freedom.
emm_options(lmerTest.limit = nrow(sttc_pairs) + 1L)
model <- lmer(sttc ~ area_pair + (1 | mouse_id), data = sttc_pairs, REML = TRUE)
overall <- anova(model, ddf = "Satterthwaite")
area_means <- emmeans(model, ~ area_pair, lmer.df = "satterthwaite")
area_means_ci <- as.data.frame(confint(area_means, level = 0.95))
pairwise_tests <- pairs(area_means, adjust = "tukey")
contrasts <- as.data.frame(summary(
  pairwise_tests, infer = c(TRUE, TRUE), level = 0.95
))

leading_pair <- as.character(area_means_ci$area_pair[which.max(area_means_ci$emmean)])
# A coefficient of -1 reverses a contrast to express the leader minus its alternative.
weights <- coef(pairwise_tests)
leader_weights <- as.numeric(weights[weights$area_pair == leading_pair, -1])
compares_leader <- leader_weights != 0
leader_differences <- (leader_weights * contrasts$estimate)[compares_leader]
leader_p_values <- contrasts$p.value[compares_leader]
has_winner <- all(leader_differences > 0 & leader_p_values < alpha)
answer <- if (has_winner) {
  paste("Brain area", sub("-", " <-> Brain area ", leading_pair, fixed = TRUE))
} else {
  "Not enough data / no differences"
}

format_p <- function(p, threshold) {
  bound <- format(threshold, scientific = threshold < 0.0001)
  ifelse(p < threshold, paste("<", bound), paste("=", signif(p, 3)))
}
format_count <- function(n) format(n, big.mark = ",", trim = TRUE)
n_units <- length(unique(c(
  paste(sttc_pairs$mouse_id, sttc_pairs$cluster_id_a),
  paste(sttc_pairs$mouse_id, sttc_pairs$cluster_id_b)
)))
count_text <- sprintf("%s for areas %s", format_count(colSums(observed_counts)),
                      sub("-", " and ", area_pair_labels, fixed = TRUE))
mean_text <- sprintf(
  "%.4f for area pair %s (95%% CI %.4f to %.4f)",
  area_means_ci$emmean, area_means_ci$area_pair,
  area_means_ci$lower.CL, area_means_ci$upper.CL
)
contrast_text <- sprintf(
  "%.4f for %s (95%% CI %.4f to %.4f, p %s)",
  contrasts$estimate, contrasts$contrast, contrasts$lower.CL, contrasts$upper.CL,
  format_p(contrasts$p.value, 0.0001)
)
conclusion <- if (has_winner) {
  sprintf(
    "Therefore, areas %s had the strongest undirected functional connectivity at the 100-ms timescale.",
    sub("-", " and ", leading_pair, fixed = TRUE)
  )
} else {
  "The comparisons gave insufficient evidence to identify a single strongest area pair."
}
answer_paragraphs <- c(
  paste(
    "Pairwise spike train interactions were estimated from single-unit activity over the entire recording.",
    "All unordered pairs of simultaneously recorded units from different brain areas within the same mouse",
    "were included. This yielded", format_count(nrow(sttc_pairs)), "pairs from", format_count(n_units),
    "units and", nlevels(sttc_pairs$mouse_id), "mice, with",
    paste0(paste(count_text[1:2], collapse = ", "), ", and ", count_text[3], ".")
  ),
  paste(
    "Interactions were quantified using the Spike Time Tiling Coefficient (STTC) with dt = 100 ms.",
    "We used the same vectorized isttc implementation as in Q4 (https://github.com/iinnpp/isttc).",
    "Spike timestamps were kept in floating-point seconds, with dt specified as 0.1 s.",
    "For each mouse, the observation window began at the earliest observed spike and extended for the",
    "full LFP-derived recording duration, retaining valid negative timestamps.",
    "STTC is described in Cutts, C. S., & Eglen, S. J. (2014)",
    "(https://doi.org/10.1523/JNEUROSCI.2767-14.2014)."
  ),
  paste(
    "STTC values were analyzed on their original scale. The linear mixed-effects model included",
    "area pair as a fixed effect and a random intercept for mouse. The Satterthwaite test for the",
    sprintf("overall area-pair effect gave F(%g, %.1f) = %.2f, p %s.",
            overall[1, "NumDF"], overall[1, "DenDF"], overall[1, "F value"],
            format_p(overall[1, "Pr(>F)"], 2.2e-16)),
    "The model-estimated mean STTC values were",
    paste0(paste(mean_text[1:2], collapse = ", "), ", and ", mean_text[3], ".")
  ),
  paste(
    "Tukey-adjusted pairwise comparisons gave differences in mean STTC of",
    paste0(paste(contrast_text[1:2], collapse = ", "), ", and ", contrast_text[3], "."),
    conclusion
  )
)
answer_lines <- c(
  unlist(lapply(answer_paragraphs, function(paragraph) c(strwrap(paragraph, width = 105), ""))),
  "Code",
  "D:/git_repos/con2phys_submission/code/questions/q5.ipynb",
  "D:/git_repos/con2phys_submission/code/questions/q5.r",
  "",
  "Model output",
  normalizePath(results_path, winslash = "/", mustWork = FALSE)
)

writeLines(capture.output({
  cat("Input:", data_path, "\n")
  cat("Method: plain STTC, dt = 100 ms, full recordings\n")
  cat("Pairing: all unordered cross-area unit pairs within each mouse\n")
  cat("Model: sttc ~ area_pair + (1 | mouse_id), original STTC scale\n")
  cat("Selected unit pairs:", nrow(sttc_pairs), "\n")
  cat("Selected mice:", nlevels(sttc_pairs$mouse_id), "\n")
  cat("\nPairs per area pair\n")
  print(colSums(observed_counts))
  cat("\nMixed model summary\n")
  print(summary(model))
  cat("\nOverall effect of area pair on STTC\n")
  print(overall)
  cat("\nModel-estimated STTC means and 95% confidence intervals\n")
  print(area_means_ci)
  cat("\nTukey-adjusted pairwise STTC differences\n")
  print(contrasts)
  cat("\nQ5 answer:", answer, "\n")
}), results_path)

write.csv(area_means_ci, means_path, row.names = FALSE)
write.csv(contrasts, contrasts_path, row.names = FALSE)
writeLines(answer_lines, answer_path)

mean_plot <- ggplot(area_means_ci, aes(x = area_pair, y = emmean)) +
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), width = 0.08, linewidth = 0.7) +
  geom_point(size = 3) +
  labs(
    title = "Model-estimated mean and 95% CI",
    x = "Brain area pair",
    y = "Mean STTC (dt = 100 ms)"
  ) +
  theme_classic(base_size = 14) +
  theme(
    panel.grid.major.y = element_line(color = "grey80", linewidth = 0.4),
    plot.title = element_text(hjust = 0.5)
  )

plot_contrasts <- contrasts
plot_contrasts$significance <- ifelse(
  plot_contrasts$p.value < 0.001, "***",
  ifelse(plot_contrasts$p.value < 0.01, "**",
         ifelse(plot_contrasts$p.value < 0.05, "*", "ns"))
)
contrast_span <- diff(range(c(plot_contrasts$lower.CL, plot_contrasts$upper.CL)))
plot_contrasts$star_x <- plot_contrasts$upper.CL + max(contrast_span * 0.04, 0.002)

contrast_plot <- ggplot(plot_contrasts, aes(x = estimate, y = contrast)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
  geom_errorbar(aes(xmin = lower.CL, xmax = upper.CL), orientation = "y", width = 0.12) +
  geom_point(size = 3) +
  geom_text(aes(x = star_x, label = significance), hjust = 0, size = 5) +
  scale_x_continuous(expand = expansion(mult = c(0.08, 0.18))) +
  labs(
    title = "Pairwise contrasts and 95% CI",
    subtitle = "Tukey-adjusted confidence intervals and p-values",
    x = "Difference in mean STTC",
    y = NULL
  ) +
  theme_classic(base_size = 14) +
  theme(
    panel.grid.major.x = element_line(color = "grey80", linewidth = 0.4),
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

ggsave(file.path(output_dir, "sttc_mixed_model_mean_and_95_ci.png"),
       mean_plot, width = 5.7, height = 4.7, dpi = 150)
ggsave(file.path(output_dir, "sttc_mixed_model_pairwise_contrasts.png"),
       contrast_plot, width = 7.2, height = 4.7, dpi = 150)
cat("Q5 answer:", answer, "\n")
cat("Saved mixed-model results to", output_dir, "\n")
