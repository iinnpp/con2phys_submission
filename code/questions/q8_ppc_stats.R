# Usage: Rscript code/questions/q8_ppc_stats.R [results directory]
# Requires nlme and emmeans.
args <- commandArgs(trailingOnly = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
repo <- normalizePath(file.path(dirname(script), "..", ".."), winslash = "/", mustWork = TRUE)
output <- if (length(args)) args[1] else Sys.getenv("Q8_OUTPUT_DIR", file.path(repo, "results", "q8"))

units <- read.csv(file.path(output, "q8_ppc_units.csv"))
areas <- 1:3
methods <- c(median_trace = "ppc_median_trace", median_channel = "ppc_median_channel")
method_titles <- c(median_trace = "Median LFP trace", median_channel = "Median channel-wise PPC")
area_colors <- c("#0072B2", "#D55E00", "#00856A")
required <- c("mouse_id", "brain_area", "cluster_id", unname(methods))
if (!all(required %in% names(units)) || anyNA(units[c("mouse_id", "brain_area", "cluster_id")]) ||
    anyDuplicated(units[c("mouse_id", "cluster_id")]) || !all(units$brain_area %in% areas)) {
  stop("The unit table must contain one row per mouse and cluster and valid area labels.")
}

analyze_method <- function(method) {
  values <- units[[methods[[method]]]]
  keep <- is.finite(values)
  method_units <- units[keep, c("mouse_id", "brain_area")]
  method_units$ppc <- values[keep]
  method_units$n_units <- 1L
  if (!nrow(method_units)) stop("No valid unit estimates for ", method)
  mice <- merge(aggregate(ppc ~ mouse_id + brain_area, method_units, mean),
                aggregate(n_units ~ mouse_id + brain_area, method_units, sum),
                by = c("mouse_id", "brain_area"))
  area_counts <- table(mice$mouse_id)
  complete_ids <- as.numeric(names(area_counts)[area_counts == length(areas)])
  mice$included_in_stats <- mice$mouse_id %in% complete_ids
  mice$method <- method
  n_mice <- length(complete_ids)
  if (n_mice < 3) stop("Expected at least three mice with complete area data for ", method)

  model_data <- mice[mice$included_in_stats, ]
  model_data$mouse_id <- factor(model_data$mouse_id)
  model_data$brain_area <- factor(model_data$brain_area, levels = areas)
  model <- nlme::lme(ppc ~ 0 + brain_area, data = model_data, random = ~ 1 | mouse_id,
                     weights = nlme::varIdent(form = ~ 1 | brain_area), method = "REML")
  # Use the independent-mouse count for the small-sample t and F references.
  marginal <- emmeans::emmeans(model, ~ brain_area, mode = "containment", df = n_mice - 1L)
  mean_ci <- as.data.frame(confint(marginal, level = 0.95, adjust = "none"))
  unit_counts <- tapply(model_data$n_units, model_data$brain_area, sum)
  means <- with(mean_ci, data.frame(
    method = method, brain_area = as.integer(as.character(brain_area)), n_mice = n_mice,
    n_units = as.integer(unit_counts[as.character(brain_area)]), estimate = emmean,
    lower_95 = lower.CL, upper_95 = upper.CL, se = SE, df = df))

  pairwise <- emmeans::contrast(marginal, method = "pairwise", adjust = "tukey")
  pair_ci <- as.data.frame(summary(pairwise, infer = c(TRUE, TRUE), level = 0.95))
  coefficients <- coef(pairwise)
  pair_weights <- as.matrix(coefficients[-1])
  area_ids <- as.integer(as.character(coefficients$brain_area))
  contrasts <- with(pair_ci, data.frame(
    method = method, area_a = area_ids[max.col(t(pair_weights))],
    area_b = area_ids[max.col(-t(pair_weights))], n_mice = n_mice,
    estimate = estimate, lower_95 = lower.CL, upper_95 = upper.CL, se = SE, df = df,
    p_value = 2 * pt(-abs(t.ratio), df), p_adjusted = p.value, adjustment = "Tukey"))
  overall <- emmeans::test(emmeans::contrast(marginal, method = "trt.vs.ctrl"), joint = TRUE)

  top <- means$brain_area[which.max(means$estimate)]
  supportive <- with(contrasts, p_adjusted < 0.05 &
                       ((area_a == top & estimate > 0) | (area_b == top & estimate < 0)))
  winner <- if (sum(supportive) == 2) top else NA_integer_
  list(mice = mice, model = model, means = means, contrasts = contrasts,
       overall = overall, top = top, winner = winner)
}

results <- lapply(names(methods), analyze_method)
names(results) <- names(methods)
for (name in c("mice", "means", "contrasts")) {
  table <- do.call(rbind, lapply(results, `[[`, name))
  write.csv(table, file.path(output, paste0("q8_ppc_stats_", name, ".csv")), row.names = FALSE)
}

model_report <- capture.output({
  for (method in names(results)) {
    result <- results[[method]]
    cat("\n", method_titles[[method]], "\n", sep = "")
    print(summary(result$model))
    cat("\nJoint test of area contrasts and estimated marginal means (mouse-level reference df)\n")
    print(result$overall)
    print(result$means, row.names = FALSE)
    cat("\nTukey-adjusted pairwise contrasts\n")
    print(result$contrasts, row.names = FALSE)
  }
})
writeLines(model_report, file.path(output, "q8_ppc_stats_model.txt"))

plot_panels <- function(filename, margins, heading, subtitle, draw_panel) {
  png(file.path(output, filename), width = 9.6, height = 4.4, units = "in", res = 180)
  on.exit(dev.off())
  par(mfrow = c(1, 2), mar = margins, oma = c(2, 0, 2.3, 0), las = 1,
      family = "sans", mgp = c(2.4, 0.65, 0), bty = "l", col.axis = "#333333")
  for (method in names(methods)) draw_panel(results[[method]], method_titles[[method]])
  mtext(heading, side = 3, outer = TRUE, line = 0.5, font = 2, cex = 1.2)
  mtext(subtitle, side = 1, outer = TRUE, line = 0.45, col = "#444444", cex = 0.83)
}

all_means <- do.call(rbind, lapply(results, `[[`, "means"))
mean_limits <- range(0, all_means$lower_95, all_means$upper_95)
plot_panels("q8_ppc_stats_mean_and_95_ci.png", c(4.1, 4.4, 2.5, 1.0),
            "4-10 Hz phase locking by brain area", "Mixed-effects model  |  Pointwise 95% confidence intervals",
            function(result, title) {
  means <- result$means
  plot(means$brain_area, means$estimate, type = "n", ylim = mean_limits, xaxt = "n",
       xlab = "Brain area", ylab = "Model-estimated mean PPC", main = title)
  axis(1, at = areas, labels = areas)
  abline(h = 0, col = "#AAAAAA", lty = 2)
  with(means, arrows(brain_area, lower_95, brain_area, upper_95, code = 3, angle = 90,
                      length = 0.07, col = area_colors, lwd = 2))
  points(means$brain_area, means$estimate, col = area_colors, pch = 19, cex = 1.35)
})

all_contrasts <- do.call(rbind, lapply(results, `[[`, "contrasts"))
contrast_limits <- range(0, all_contrasts$lower_95, all_contrasts$upper_95)
p_label_x <- contrast_limits[2] + 0.06 * diff(contrast_limits)
plot_limits <- c(contrast_limits[1], contrast_limits[2] + 0.42 * diff(contrast_limits))
plot_panels("q8_ppc_stats_pairwise_tukey.png", c(4.1, 5.9, 2.5, 1.0),
            "Paired differences in 4-10 Hz phase locking",
            "Tukey-adjusted 95% CIs and p-values  |  * p < 0.05   ** p < 0.01   *** p < 0.001",
            function(result, title) {
  contrasts <- result$contrasts
  y <- rev(seq_len(nrow(contrasts)))
  plot(contrasts$estimate, y, type = "n", xlim = plot_limits, ylim = range(y) + c(-0.45, 0.45),
       yaxt = "n", xaxt = "n", xlab = "Difference in mean PPC", ylab = "", main = title)
  ticks <- pretty(contrast_limits)
  axis(1, at = ticks[ticks >= par("usr")[1] & ticks <= contrast_limits[2]])
  axis(2, at = y, labels = sprintf("Area %d - %d", contrasts$area_a, contrasts$area_b), tick = FALSE)
  abline(v = 0, col = "#999999", lty = 2)
  with(contrasts, arrows(lower_95, y, upper_95, y, code = 3, angle = 90,
                          length = 0.07, col = "#333333", lwd = 1.6))
  points(contrasts$estimate, y, pch = 19, col = "#333333", cex = 1.2)
  stars <- ifelse(contrasts$p_adjusted < 0.001, "***",
                  ifelse(contrasts$p_adjusted < 0.01, "**",
                         ifelse(contrasts$p_adjusted < 0.05, "*", "")))
  labels <- trimws(sprintf("p = %.2g  %s", contrasts$p_adjusted, stars))
  text(p_label_x, y, labels = labels, pos = 4, offset = 0,
       cex = 0.77, col = "#333333")
})

primary <- results$median_trace
sensitivity <- results$median_channel
paragraph <- function(...) c(strwrap(paste(...), width = 116), "")
join_three <- function(values) paste(paste(values[1:2], collapse = ", "), values[3], sep = ", and ")

describe_winner <- function(result) {
  if (is.na(result$winner)) {
    sprintf("Area %d had the highest estimate. The paired comparisons were inconclusive about the strongest area.",
            result$top)
  } else {
    sprintf("Area %d exceeded both other areas in Tukey-adjusted comparisons (p < 0.05).", result$winner)
  }
}

counts <- primary$means$n_units
primary_means <- with(primary$means, sprintf("%.6f for area %d (95%% CI %.6f to %.6f)",
                                           estimate, brain_area, lower_95, upper_95))
primary_differences <- with(primary$contrasts, sprintf(
  "the area %d minus area %d difference was %.6f (95%% CI %.6f to %.6f, p = %.4g)",
  area_a, area_b, estimate, lower_95, upper_95, p_adjusted))
sensitivity_means <- with(sensitivity$means, sprintf("%.6f for area %d", estimate, brain_area))
conclusion <- if (is.na(primary$winner)) describe_winner(primary) else sprintf(
  "Therefore, brain area %d had the strongest spike-LFP phase locking in the 4-10 Hz range.", primary$winner)
output_path <- normalizePath(output, winslash = "/", mustWork = TRUE)
report <- c(
  paragraph("Spike-LFP phase locking was estimated from single-unit activity over the entire recording.",
            "Each unit was paired with the median LFP voltage trace across channels from the same mouse and brain area.",
            sprintf("This yielded %s units from %d mice, with %d units in area 1, %d in area 2, and %d in area 3.",
                    format(sum(counts), big.mark = ","), primary$means$n_mice[1], counts[1], counts[2], counts[3])),
  paragraph("Phase locking was quantified with pairwise phase consistency (PPC), the dimensionless mean cosine of",
            "phase differences across distinct spike pairs. The LFP was band-pass filtered at 4-10 Hz using NeuroDSP",
            "with a zero-phase, third-order Butterworth filter. Instantaneous phase was estimated with the Hilbert",
            "transform and sampled at the 500 Hz LFP sample nearest each spike. Spikes outside the LFP window or",
            "within one second of an LFP segment boundary were excluded. Each unit needed at least 31 usable spikes."),
  paragraph("PPC values were analyzed on their original scale. I averaged PPC within each mouse and brain area and",
            "fitted a linear mixed-effects model by restricted maximum likelihood, with brain area as a fixed effect",
            "and mouse_id as a random intercept. Residual variance was estimated separately for each area. The",
            "reference degrees of freedom for inference were the number of mice minus one.",
            with(primary$overall, sprintf("The overall test of brain area gave F(%d, %d) = %.2f, p = %.4g.",
                                          df1, df2, F.ratio, p.value)),
            "Model-estimated mean PPC values were", paste0(join_three(primary_means), ".")),
  paragraph("Tukey-adjusted pairwise comparisons were reported as differences in mean PPC.",
            "In the primary analysis,", paste0(join_three(primary_differences), ".")),
  paragraph("As a sensitivity check, I calculated each unit's PPC separately against each same-area LFP",
            "channel and took the median across channels. The model-estimated mean PPC values were",
            paste0(join_three(sensitivity_means), "."), describe_winner(sensitivity), conclusion),
  "Code",
  file.path(repo, "code/questions", c("q8_ppc.py", "q8_ppc_plots.ipynb", "q8_ppc_stats.R")), "",
  "Model output",
  file.path(output_path, c("q8_ppc_stats_model.txt", "q8_ppc_stats_means.csv", "q8_ppc_stats_contrasts.csv"))
)
writeLines(report, file.path(output, "answer.txt"))
cat(paste(report, collapse = "\n"), "\n", sep = "")
