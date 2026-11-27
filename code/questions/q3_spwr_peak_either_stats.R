# Usage: Rscript code/questions/q3_spwr_peak_either_stats.R [results directory]
args <- commandArgs(trailingOnly = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
repo <- normalizePath(file.path(dirname(script), "..", ".."), winslash = "/", mustWork = TRUE)
output <- if (length(args)) args[1] else file.path(repo, "results", "q3", "psd_spwr")

all_rates <- read.csv(file.path(output, "q3_spwr_mouse_area_rates.csv"))
data <- subset(all_rates, matching_rule == "either")
areas <- 1:3
required <- c("mouse_id", "brain_area", "peak_rate_per_min", "peak_ripple_channel_row", "n_channels")
if (!all(required %in% names(data)) || anyNA(data[required]) ||
    anyDuplicated(data[c("mouse_id", "brain_area")]) ||
    !setequal(data$brain_area, areas) ||
    any(!is.finite(data$peak_rate_per_min) | data$peak_rate_per_min < 0)) {
  stop("Expected unique mouse-area rows and nonnegative peak rates for the either-offset rule.")
}
data$brain_area <- factor(data$brain_area, levels = areas)
data$mouse_id <- factor(data$mouse_id)
mouse_counts <- as.integer(table(data$brain_area))
if (any(mouse_counts < 3L)) stop("At least three mice per area are required.")

# One highest-count channel per mouse and area, based on either-offset matches.
model <- nlme::lme(peak_rate_per_min ~ brain_area, random = ~ 1 | mouse_id,
                   data = data, method = "REML")
overall <- anova(model)["brain_area", , drop = FALSE]
overall_test <- data.frame(
  effect = "brain_area", numerator_df = overall$numDF,
  denominator_df = overall$denDF, F = overall[["F-value"]],
  p_value = pf(overall[["F-value"]], overall$numDF, overall$denDF, lower.tail = FALSE)
)

design <- model.matrix(~ brain_area, data.frame(brain_area = factor(areas, levels = areas)))
beta <- nlme::fixef(model)
covariance <- vcov(model)
estimated_mean <- as.numeric(design %*% beta)
mean_se <- sqrt(diag(design %*% covariance %*% t(design)))
# Use the contributing mouse count for the pointwise mean intervals.
mean_df <- mouse_counts - 1L
mean_margin <- qt(0.975, mean_df) * mean_se
means <- data.frame(
  brain_area = areas, n_mice = mouse_counts,
  estimated_mean_per_min = estimated_mean,
  lower_95_per_min = estimated_mean - mean_margin,
  upper_95_per_min = estimated_mean + mean_margin,
  arithmetic_observed_mean_per_min = as.numeric(tapply(data$peak_rate_per_min, data$brain_area, mean)),
  se_per_min = mean_se, df = mean_df
)

pairs <- combn(seq_along(areas), 2)
contrast_design <- design[pairs[2, ], , drop = FALSE] - design[pairs[1, ], , drop = FALSE]
difference <- as.numeric(contrast_design %*% beta)
contrast_se <- sqrt(diag(contrast_design %*% covariance %*% t(contrast_design)))
contrast_df <- overall$denDF[1]
t_value <- difference / contrast_se
raw_p <- 2 * pt(-abs(t_value), contrast_df)
# Studentized-range approximation for all pairwise mixed-model contrasts.
tukey_p <- ptukey(sqrt(2) * abs(t_value), length(areas), contrast_df, lower.tail = FALSE)
family_margin <- qtukey(0.95, length(areas), contrast_df) / sqrt(2) * contrast_se
contrasts <- data.frame(
  area_a = areas[pairs[2, ]], area_b = areas[pairs[1, ]],
  difference_per_min = difference, lower_95_per_min = difference - family_margin,
  upper_95_per_min = difference + family_margin, se_per_min = contrast_se,
  df = contrast_df, p_value = raw_p, p_adjusted = tukey_p, adjustment = "Tukey"
)

prefix <- file.path(output, "q3_spwr_peak_either_stats_")
write.csv(means, paste0(prefix, "means.csv"), row.names = FALSE)
write.csv(contrasts, paste0(prefix, "contrasts.csv"), row.names = FALSE)
write.csv(overall_test, paste0(prefix, "overall.csv"), row.names = FALSE)
writeLines(capture.output({
  cat("Either-offset sharp-wave-ripple overlap rates on the highest-count channel\n")
  cat("One selected channel per mouse-area. A ripple qualifies with either deeper offset.\n")
  cat("The selected channel is the one with the most matched candidates in that mouse-area.\n")
  cat("The model uses all eligible mouse-area rows; Mouse 15 has no Area 1 observation.\n")
  cat("The model uses rates in candidates per minute on the original scale.\n\n")
  print(summary(model))
  cat("\nOverall area effect\n")
  print(overall_test, row.names = FALSE)
  cat("\nModel-estimated means and pointwise 95% CIs (candidates/min)\n")
  print(means, row.names = FALSE)
  cat("\nArea differences with Tukey-adjusted p-values and simultaneous 95% CIs\n")
  cat("Tukey adjustment uses the studentized-range approximation for mixed-model contrasts.\n")
  print(contrasts, row.names = FALSE)
}), paste0(prefix, "model.txt"))

colors <- c("#0072B2", "#D55E00", "#00856A")
plot_means <- function() {
  png(paste0(prefix, "means_95ci.png"), width = 6.5, height = 4.45, units = "in", res = 200)
  on.exit(dev.off())
  par(mar = c(4.2, 5.9, 4.6, 1.5), las = 1, bty = "l", mgp = c(2.8, 0.7, 0))
  ylim <- range(c(0, means$lower_95_per_min, means$upper_95_per_min))
  ylim[2] <- ylim[2] + 0.12 * diff(ylim)
  plot(areas, means$estimated_mean_per_min, type = "n",
       xlim = c(0.5, 3.5), ylim = ylim, xaxt = "n", yaxt = "n", xlab = "",
       ylab = "Matched candidates/min")
  axis(1, at = areas, labels = sprintf("Area %d\nn = %d", areas, mouse_counts),
       tick = FALSE, line = 0.2)
  ticks <- axTicks(2)
  axis(2, at = ticks, labels = format(ticks, trim = TRUE, scientific = FALSE))
  abline(h = ticks, col = "#E1E6EB")
  with(means, arrows(brain_area, lower_95_per_min, brain_area, upper_95_per_min,
                     code = 3, angle = 90, length = 0.07, col = colors, lwd = 2))
  points(areas, means$estimated_mean_per_min, col = colors, pch = 19, cex = 1.35)
  text(areas, means$upper_95_per_min, sprintf("%.2f", means$estimated_mean_per_min),
       pos = 3, offset = 0.55, col = colors, cex = 0.9)
  title("Peak-channel overlap rates by area", line = 2.4)
  mtext("Either offset  |  Model means and 95% CIs",
        side = 3, line = 0.8, cex = 0.85, col = "#465463")
}

plot_contrasts <- function() {
  png(paste0(prefix, "pairwise_tukey.png"), width = 7.7, height = 4.35,
      units = "in", res = 200)
  on.exit(dev.off())
  par(mar = c(5.6, 8.0, 4.6, 1.5), las = 1, bty = "l", mgp = c(2.8, 0.7, 0))
  y <- rev(seq_len(nrow(contrasts)))
  bounds <- range(c(0, contrasts$lower_95_per_min, contrasts$upper_95_per_min))
  span <- max(diff(bounds), 0.3)
  xlim <- bounds + c(-0.10, 0.30) * span
  plot(contrasts$difference_per_min, y, type = "n", xlim = xlim,
       ylim = range(y) + c(-0.47, 0.47), yaxt = "n", ylab = "",
       xlab = "Difference in matched candidates/min")
  axis(2, at = y, labels = with(contrasts,
       sprintf("Area %d - Area %d", area_a, area_b)), tick = FALSE)
  abline(h = y, col = "#E1E6EB")
  abline(v = 0, col = "#777777", lty = 2)
  with(contrasts, arrows(lower_95_per_min, y, upper_95_per_min, y, code = 3, angle = 90,
                         length = 0.07, col = "#283C50", lwd = 2))
  points(contrasts$difference_per_min, y, pch = 19, col = "#283C50", cex = 1.25)
  stars <- ifelse(contrasts$p_adjusted < 0.001, "***",
                  ifelse(contrasts$p_adjusted < 0.01, "**",
                         ifelse(contrasts$p_adjusted < 0.05, "*", "")))
  text(contrasts$difference_per_min, y, trimws(sprintf("%.2f %s", contrasts$difference_per_min, stars)),
       pos = 3, offset = 0.65, col = "#283C50", cex = 0.9)
  p_text <- sprintf("p = %.2g", contrasts$p_adjusted)
  text(rep(bounds[2] + 0.08 * span, length(y)), y, p_text,
       adj = c(0, 0.5), cex = 0.75, col = "#465463")
  title("Pairwise peak-channel rate differences", line = 2.4)
  mtext("Either offset  |  Tukey-adjusted 95% CIs",
        side = 3, line = 0.8, cex = 0.85, col = "#465463")
  mtext("Adjusted p-values     * p < 0.05     ** p < 0.01     *** p < 0.001",
        side = 1, line = 4.3, cex = 0.77, col = "#465463")
}

plot_means()
plot_contrasts()
