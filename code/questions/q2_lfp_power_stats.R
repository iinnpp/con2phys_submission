# Usage: Rscript code/questions/q2_lfp_power_stats.R [results directory]
args <- commandArgs(trailingOnly = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
repo <- normalizePath(file.path(dirname(script), "..", ".."), winslash = "/", mustWork = TRUE)
output <- if (length(args)) args[1] else file.path(repo, "results", "q2")

data <- read.csv(file.path(output, "q2_mouse_area_power.csv"), stringsAsFactors = FALSE)
areas <- as.character(1:3)
required <- c("mouse_id", "brain_area", "broadband_power", "power_unit", "n_valid_channels")
if (!all(required %in% names(data)) || anyNA(data[required]) ||
    anyDuplicated(data[c("mouse_id", "brain_area")]) ||
    !setequal(unique(data$brain_area), areas) || any(table(data$mouse_id) != length(areas)) ||
    any(!is.finite(data$broadband_power) | data$broadband_power <= 0) ||
    length(unique(data$power_unit)) != 1) {
  stop("Expected complete, unique mouse-area rows with positive power in a common unit.")
}
data$mouse_id <- factor(data$mouse_id)
data$brain_area <- factor(data$brain_area, levels = areas)
data$log10_power <- log10(data$broadband_power)
n_mice <- nlevels(data$mouse_id)
if (n_mice < 3) stop("At least three mice are required for the model.")

model <- nlme::lme(log10_power ~ brain_area, random = ~ 1 | mouse_id,
                   data = data, method = "REML")
overall <- anova(model)["brain_area", , drop = FALSE]
design <- model.matrix(~ brain_area, data.frame(brain_area = factor(areas, levels = areas)))
coefficients <- nlme::fixef(model)
covariance <- vcov(model)
log_means <- as.numeric(design %*% coefficients)
mean_se <- sqrt(diag(design %*% covariance %*% t(design)))

# Use mouse-level degrees of freedom for pointwise area means. Differences
# use the within-mouse area residual df from the balanced blocked model.
mean_df <- n_mice - 1L
mean_margin <- qt(0.975, mean_df) * mean_se
means <- data.frame(
  brain_area = areas, n_mice = n_mice, power_unit = data$power_unit[1],
  geometric_mean = 10^log_means, lower_95 = 10^(log_means - mean_margin),
  upper_95 = 10^(log_means + mean_margin), log10_mean = log_means, log10_se = mean_se,
  df = mean_df
)

pairs <- combn(seq_along(areas), 2)
contrast_design <- design[pairs[1, ], , drop = FALSE] - design[pairs[2, ], , drop = FALSE]
log_ratios <- as.numeric(contrast_design %*% coefficients)
contrast_se <- sqrt(diag(contrast_design %*% covariance %*% t(contrast_design)))
contrast_df <- overall$denDF[1]
t_value <- log_ratios / contrast_se
tukey_margin <- qtukey(0.95, length(areas), contrast_df) / sqrt(2) * contrast_se
tukey_p <- ptukey(sqrt(2) * abs(t_value), length(areas), contrast_df, lower.tail = FALSE)
# The studentized range CDF can round to one for extreme contrasts.
# The union bound over all pairwise t tests gives a conservative p-value bound.
p_bound <- ifelse(tukey_p == 0, ncol(pairs) * 2 * pt(-abs(t_value), contrast_df), NA_real_)
tukey_p[tukey_p == 0] <- NA_real_
contrasts <- data.frame(
  area_a = areas[pairs[1, ]], area_b = areas[pairs[2, ]],
  ratio_a_over_b = 10^log_ratios,
  lower_95 = 10^(log_ratios - tukey_margin), upper_95 = 10^(log_ratios + tukey_margin),
  log10_ratio = log_ratios, log10_se = contrast_se, df = contrast_df,
  p_adjusted = tukey_p, p_adjusted_upper_bound = p_bound
)
reported_p <- with(contrasts, ifelse(is.na(p_adjusted), p_adjusted_upper_bound, p_adjusted))

top <- means$brain_area[which.max(means$geometric_mean)]
supports_top <- with(contrasts, (area_a == top & lower_95 > 1) |
                               (area_b == top & upper_95 < 1))
answer <- if (sum(supports_top) == length(areas) - 1) {
  paste("Brain area", top)
} else {
  "Not enough data / no differences"
}

write.csv(means, file.path(output, "q2_stats_means.csv"), row.names = FALSE)
write.csv(contrasts, file.path(output, "q2_stats_contrasts.csv"), row.names = FALSE)
writeLines(capture.output({
  print(summary(model))
  cat("\nOverall area effect\n")
  print(overall)
  cat("\nGeometric means and pointwise 95% CIs\n")
  print(means, row.names = FALSE)
  cat("\nPower ratios with Tukey-adjusted 95% CIs and p-values\n")
  print(contrasts, row.names = FALSE)
  cat("\nQ2 answer:", answer, "\n")
}), file.path(output, "q2_stats_model.txt"))

area_colors <- c("#0072B2", "#D55E00", "#00856A")
power_unit <- means$power_unit[1]
axis_unit <- if (power_unit == "uV^2") quote(mu * V^2) else power_unit
format_power <- function(x, digits = 0L) formatC(x, format = "f", digits = digits, big.mark = ",")

plot_area_means <- function() {
  png(file.path(output, "q2_stats_means_95ci.png"),
      width = 6.4, height = 4.4, units = "in", res = 200)
  on.exit(dev.off())
  par(mar = c(4, 6.5, 4.6, 1.4), las = 1, bty = "l", mgp = c(2.7, 0.7, 0))
  x <- seq_len(nrow(means))
  unit_label <- bquote("Broadband LFP power (" * .(axis_unit) * ")")
  y_lim <- 10^(range(log10(c(means$lower_95, means$upper_95))) + c(-0.12, 0.22))
  plot(x, means$geometric_mean, type = "n", log = "y", xlim = c(0.5, 3.5),
       ylim = y_lim, xaxt = "n", yaxt = "n", xlab = "", ylab = "")
  axis(1, at = x, labels = paste("Area", means$brain_area), tick = FALSE)
  ticks <- axTicks(2)
  axis(2, at = ticks, labels = format_power(ticks, if (min(ticks) < 1) 3L else 0L))
  mtext(unit_label, side = 2, line = 5, las = 0)
  abline(h = ticks, col = "#E6E8EB")
  with(means, arrows(x, lower_95, x, upper_95, code = 3, angle = 90,
                     length = 0.07, col = area_colors, lwd = 2))
  points(x, means$geometric_mean, col = area_colors, pch = 19, cex = 1.35)
  text(x, means$upper_95, labels = format_power(means$geometric_mean, 1L),
       pos = 3, offset = 0.65, col = area_colors, cex = 0.9)
  title("Broadband LFP power by area", line = 2.4)
  mtext(sprintf("1-100 Hz geometric means and 95%% CIs across %d mice", n_mice),
        side = 3, line = 0.8, cex = 0.86, col = "#444444")
}

plot_power_ratios <- function() {
  png(file.path(output, "q2_stats_pairwise_tukey.png"),
      width = 7.4, height = 4.35, units = "in", res = 200)
  on.exit(dev.off())
  par(mar = c(5.7, 8, 4.6, 1.7), las = 1, bty = "l", mgp = c(2.7, 0.7, 0))
  y <- rev(seq_len(nrow(contrasts)))
  x_range <- range(log10(c(contrasts$lower_95, contrasts$upper_95, 1)))
  x_lim <- 10^(x_range + c(-0.08, 0.07) * max(diff(x_range), 1))
  plot(contrasts$ratio_a_over_b, y, type = "n", log = "x", xlim = x_lim,
       ylim = range(y) + c(-0.48, 0.48), yaxt = "n", xlab = "Ratio of geometric mean power",
       ylab = "")
  axis(2, at = y, labels = with(contrasts, sprintf("Area %s / Area %s", area_a, area_b)),
       tick = FALSE)
  abline(h = y, col = "#E6E8EB")
  abline(v = 1, col = "#777777", lty = 2)
  with(contrasts, arrows(lower_95, y, upper_95, y, code = 3, angle = 90,
                         length = 0.07, col = "#283C50", lwd = 2))
  points(contrasts$ratio_a_over_b, y, pch = 19, col = "#283C50", cex = 1.3)
  stars <- ifelse(reported_p < 0.001, "***",
                  ifelse(reported_p < 0.01, "**", ifelse(reported_p < 0.05, "*", "")))
  ratio_labels <- trimws(sprintf("%.3f %s", contrasts$ratio_a_over_b, stars))
  text(contrasts$ratio_a_over_b, y, labels = ratio_labels,
       pos = 3, offset = 0.6, col = "#283C50", cex = 0.88)
  title("Pairwise ratios of broadband LFP power", line = 2.4)
  mtext("Power ratios and Tukey-adjusted 95% CIs", side = 3, line = 0.8,
        cex = 0.86, col = "#444444")
  mtext("Tukey-adjusted p-values     * p < 0.05     ** p < 0.01     *** p < 0.001",
        side = 1, line = 4.25, cex = 0.78, col = "#444444")
}

plot_area_means()
plot_power_ratios()

paragraph <- function(...) c(strwrap(paste(...), width = 115), "")
channel_counts <- tapply(data$n_valid_channels, data$brain_area, sum)
estimate_unit <- if (power_unit == "uV^2") "microvolts squared" else power_unit
area_estimates <- with(means, sprintf("%s for area %s (95%% CI %s-%s)",
  format_power(geometric_mean, 1L), brain_area, format_power(lower_95, 1L), format_power(upper_95, 1L)))
overall_p <- with(overall, pf(`F-value`, numDF, denDF, lower.tail = FALSE))

higher_is_b <- contrasts$ratio_a_over_b < 1
reported_ratio <- with(contrasts, ifelse(higher_is_b, 1 / ratio_a_over_b, ratio_a_over_b))
reported_lower <- with(contrasts, ifelse(higher_is_b, 1 / upper_95, lower_95))
reported_upper <- with(contrasts, ifelse(higher_is_b, 1 / lower_95, upper_95))
p_label <- ifelse(reported_p < 0.0001, "p < 0.0001", sprintf("p = %.4g", reported_p))
area_ratios <- with(contrasts, sprintf("Area %s had %.2f times the power of area %s (95%% CI %.2f-%.2f, %s).",
  ifelse(higher_is_b, area_b, area_a), reported_ratio,
  ifelse(higher_is_b, area_a, area_b), reported_lower, reported_upper, p_label))
conclusion <- if (startsWith(answer, "Brain area")) {
  sprintf("Therefore, brain area %s had the highest broadband LFP power.", top)
} else {
  sprintf("Area %s had the highest estimate. The pairwise comparisons did not distinguish it from both other areas.", top)
}

writeLines(c(
  paragraph(
    "Broadband LFP power was calculated over the entire recording. For each channel, I estimated the power",
    "spectral density (PSD) at 500 Hz using Welch's method in NeuroDSP with 4 s Hann windows, 50% overlap,",
    "and the median across windows. I requested PSDs from 0.1 to 150 Hz, took the pointwise median across",
    "channels within each mouse and brain area, and integrated that spectrum from 1 to 100 Hz.",
    sprintf("The analysis included %s channels from %d mice (area 1: %d channels; area 2: %d; area 3: %d).",
            format_power(sum(channel_counts)), n_mice, channel_counts[1], channel_counts[2], channel_counts[3])
  ),
  paragraph(
    "I fitted a linear mixed-effects model to log10 broadband power with brain area as a fixed effect and",
    "mouse_id as a random intercept to account for the three areas recorded in each mouse.",
    sprintf("The model showed an overall effect of brain area (F(%d, %d) = %.2f, p = %.2e).",
            overall$numDF, overall$denDF, overall[["F-value"]], overall_p),
    sprintf("Back-transformed model estimates, interpreted as geometric means in %s, were", estimate_unit),
    paste0(paste(area_estimates[1:2], collapse = ", "), ", and ", area_estimates[3], ".")
  ),
  paragraph(
    "Tukey-adjusted pairwise comparisons were performed on the log scale and are reported as ratios",
    "of geometric means.", paste(area_ratios, collapse = " "), conclusion
  ),
  paste("Code:", paste(file.path(repo, "code", "questions",
    c("q2_lfp_power.py", "q2_lfp_power_stats.R", "q2_lfp_power_plots.ipynb")), collapse = ", ")),
  "",
  paste("Model output:", normalizePath(file.path(output, "q2_stats_model.txt"), winslash = "/"))
), file.path(output, "answer.txt"))
