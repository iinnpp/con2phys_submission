# Shared mouse-level tests and plots for the Q6 directed-connectivity estimators.
defaults <- list(
  setting_columns = c("bin_ms", "history_ms"), metric_column = "bits_per_spike",
  units = "bits per target spike", show_shift = TRUE,
  timing_text = function(setting) sprintf("Primary %s: %d ms bins and %d ms short history.",
                                          estimator_label, setting$bin_ms, setting$history_ms),
  timing_plot = function(setting, n) sprintf("%d ms bins, %d ms history, %d mice",
                                              setting$bin_ms, setting$history_ms, n),
  sensitivity_label = "bin width/history (ms)",
  closing = c("Scores describe directional predictability. Unmeasured shared inputs may contribute to the gains.",
              "Local source shifts are a diagnostic for disrupted alignment. They also disrupt source relationships",
              "with other predictors, so the primary inference uses the paired mouse estimates.")
)
spec <- modifyList(defaults, get0("stats_spec", ifnotfound = list()))
setting_columns <- spec$setting_columns
ordered_rows <- function(frame, ...) {
  frame[do.call(order, c(list(!frame$primary), unname(as.list(frame[setting_columns])), list(...))), ]
}
result_file <- function(suffix) file.path(output, paste0(result_prefix, "_", suffix))
scores <- read.csv(result_file("mouse_scores.csv"))
scores$score <- scores[[spec$metric_column]]
scores$primary <- tolower(as.character(scores$primary)) == "true"
scores$candidate <- tolower(as.character(scores$candidate)) == "true"
scores$direction <- paste(scores$source, scores$target, sep = "->")
directions <- c("1->2", "3->2", "3->1")
all_directions <- c(directions, "2->1", "2->3", "1->3")

keys <- c("mouse_id", setting_columns, "direction")
settings <- unique(scores[c(setting_columns, "primary")])
if (anyDuplicated(scores[keys]) || any(!is.finite(scores$score)) ||
    sum(settings$primary) != 1L || any(scores$candidate != (scores$direction %in% directions))) {
  stop("Invalid or duplicate mouse scores")
}
mice <- sort(unique(scores$mouse_id))
if (length(mice) < 3L) stop("Inference requires at least three mice")
if (nrow(scores) != length(mice) * nrow(settings) * length(all_directions) ||
    !setequal(scores$direction, all_directions) || length(unique(scores$n_subsamples)) != 1L) {
  stop("Expected a balanced table with six directions for every mouse and setting")
}

estimate <- function(x) {
  n <- length(x)
  center <- mean(x)
  se <- sd(x) / sqrt(n)
  half_width <- qt(0.975, df = n - 1) * se
  p <- if (se > 0) 2 * pt(-abs(center / se), df = n - 1) else if (center == 0) 1 else 0
  data.frame(n_mice = n, estimate = center, ci95_low = center - half_width,
             ci95_high = center + half_width, se = se, df = n - 1, p_value = p)
}

pieces <- split(scores, scores[c(setting_columns, "direction")], drop = TRUE)
means <- do.call(rbind, lapply(pieces, function(piece) {
  values <- estimate(piece$score)
  values[c("se", "df", "p_value")] <- NULL
  row <- data.frame(piece[1, c(setting_columns, "primary", "direction", "candidate")], values)
  if (spec$show_shift) {
    row$shifted_mean <- if (all(is.na(piece$shifted_bits_per_spike))) NA_real_ else mean(piece$shifted_bits_per_spike)
  }
  row
}))
means <- ordered_rows(means, match(means$direction, all_directions))
primary <- scores[scores$primary & scores$candidate, ]
wide <- reshape(primary[c("mouse_id", "direction", "score")], idvar = "mouse_id",
                timevar = "direction", direction = "wide")
wide <- wide[order(wide$mouse_id), paste0("score.", directions)]
names(wide) <- directions
zero_tests <- do.call(rbind, lapply(directions, function(direction) {
  data.frame(direction = direction, reference = "zero", estimate(wide[[direction]]))
}))
zero_tests$adjustment <- "Holm"
zero_tests$p_adjusted <- p.adjust(zero_tests$p_value, method = "holm")

pairwise_tests <- do.call(rbind, lapply(combn(directions, 2L, simplify = FALSE), function(pair) {
  data.frame(direction = pair[1], reference = pair[2], estimate(wide[[pair[1]]] - wide[[pair[2]]]))
}))
# Keep the paired standard error for each contrast. The Tukey studentized-range
# adjustment is approximate when the contrast standard errors differ.
k <- length(directions)
q <- with(pairwise_tests, ifelse(se > 0, sqrt(2) * abs(estimate) / se,
                                ifelse(estimate == 0, 0, Inf)))
pairwise_tests$adjustment <- "Tukey"
pairwise_tests$p_adjusted <- ptukey(q, nmeans = k, df = pairwise_tests$df, lower.tail = FALSE)
half_width <- with(pairwise_tests, qtukey(0.95, nmeans = k, df = df) * se / sqrt(2))
pairwise_tests$ci95_low <- pairwise_tests$estimate - half_width
pairwise_tests$ci95_high <- pairwise_tests$estimate + half_width
tests <- rbind(zero_tests, pairwise_tests)

candidate_means <- means[means$primary & means$candidate, ]
winner <- candidate_means$direction[which.max(candidate_means$estimate)]
positive <- with(zero_tests, direction == winner & estimate > 0 & p_adjusted < 0.05)
win_contrasts <- with(pairwise_tests, p_adjusted < 0.05 &
                        ((direction == winner & estimate > 0) | (reference == winner & estimate < 0)))
answer <- if (any(positive) && sum(win_contrasts) == 2L) {
  areas <- strsplit(winner, "->", fixed = TRUE)[[1]]
  sprintf("Brain area %s => Brain area %s", areas[1], areas[2])
} else {
  "Not enough data / no differences"
}

candidates <- means[means$candidate, ]
sensitivity <- do.call(rbind, lapply(split(candidates, candidates[setting_columns], drop = TRUE), function(piece) {
  ranked <- piece[order(-piece$estimate), ]
  data.frame(ranked[1, setting_columns, drop = FALSE],
             primary = ranked$primary[1], highest_mean_direction = ranked$direction[1],
             highest_mean = ranked$estimate[1], margin_over_runner_up = diff(rev(ranked$estimate[1:2])))
}))
sensitivity <- ordered_rows(sensitivity)
sensitivity_names <- do.call(paste, c(unname(as.list(sensitivity[setting_columns])), sep = "/"))

write.csv(means, result_file("stats_means.csv"), row.names = FALSE)
write.csv(tests, result_file("stats_primary_tests.csv"), row.names = FALSE)
write.csv(sensitivity, result_file("stats_sensitivity.csv"), row.names = FALSE)

format_score <- function(x) formatC(x, digits = 5, format = "f")
primary_setting <- settings[settings$primary, ]
mean_lines <- with(candidate_means, sprintf("  %s: %s [%s, %s]", direction, format_score(estimate),
                                           format_score(ci95_low), format_score(ci95_high)))
if (spec$show_shift) mean_lines <- paste0(mean_lines, "; shifted ", format_score(candidate_means$shifted_mean))
lines <- c(
  paste("Q6 answer:", answer),
  "",
  sprintf("Full-recording population spike counts from %d mice, with %d matched unit draws per mouse.",
          length(mice), scores$n_subsamples[1]),
  spec$timing_text(primary_setting),
  estimator_description,
  "The mouse is the independent unit. Direction mean confidence intervals are marginal 95% t intervals.",
  "The three paired direction comparisons use Tukey-adjusted p-values and simultaneous 95% intervals.",
  "Their standard errors come from within-mouse differences. Tukey adjustment is approximate when these standard errors differ.",
  "The three tests against zero form a separate family with Holm-adjusted p-values.",
  "A direction is selected if it is above zero and exceeds both alternatives at adjusted p < 0.05.",
  "",
  sprintf("Primary mean %s [95%% CI]%s:", spec$units,
          if (spec$show_shift) " and source-shift diagnostic mean" else ""),
  mean_lines,
  "",
  "Primary tests against zero (Holm-adjusted p):",
  with(zero_tests, sprintf("  %s: %s, p = %s", direction, format_score(estimate),
                           format.pval(p_adjusted, digits = 3))),
  "",
  sprintf("Primary paired contrasts in %s (Tukey-adjusted 95%% CI and p):", spec$units),
  with(pairwise_tests, sprintf("  %s vs %s: %s [%s, %s], p = %s", direction, reference,
                               format_score(estimate), format_score(ci95_low), format_score(ci95_high),
                               format.pval(p_adjusted, digits = 3))),
  "",
  sprintf("Descriptive sensitivity ranks by %s:", spec$sensitivity_label),
  sprintf("  %s: %s", sensitivity_names, sensitivity$highest_mean_direction),
  "",
  spec$closing
)
writeLines(lines, file.path(output, "answer.txt"))
writeLines(lines, file.path(output, "stats_output.txt"))

direction_label <- function(direction) sub("->", paste0(" ", intToUtf8(8594), " "), direction, fixed = TRUE)
plot_means <- candidate_means[match(directions, candidate_means$direction), ]
position <- seq_len(nrow(plot_means))
png(result_file("stats_mean_and_95_ci.png"), width = 5.7, height = 4.7,
    units = "in", res = 150, type = "cairo")
par(mar = c(4.5, 6.3, 4.4, 1.2), las = 1, cex.axis = 1.05)
plot(position, plot_means$estimate, type = "n", xaxt = "n", bty = "l",
     xlim = range(position) + c(-0.4, 0.4),
     ylim = range(0, plot_means$ci95_low, plot_means$ci95_high),
     xlab = "Brain area direction", ylab = "")
axis(1, at = position, labels = direction_label(plot_means$direction))
mtext(sprintf("Mean %s (%s)", if (spec$show_shift) "gain" else "asymmetry", spec$units),
      side = 2, line = 4.6, las = 0)
grid(nx = NA, ny = NULL, col = "grey85", lty = 1)
abline(h = 0, lty = 2, col = "grey50")
with(plot_means, {
  segments(position, ci95_low, position, ci95_high, lwd = 1.5)
  segments(position - 0.045, ci95_low, position + 0.045, ci95_low, lwd = 1.5)
  segments(position - 0.045, ci95_high, position + 0.045, ci95_high, lwd = 1.5)
  points(position, estimate, pch = 19, cex = 1.3)
})
title("Mean connectivity and 95% CI", line = 2.4)
mtext(spec$timing_plot(primary_setting, length(mice)), side = 3, line = 0.9, cex = 0.9)
invisible(dev.off())

contrast_labels <- with(pairwise_tests, paste(direction_label(direction), "vs", direction_label(reference)))
position <- rev(seq_len(nrow(pairwise_tests)))
span <- diff(range(0, pairwise_tests$ci95_low, pairwise_tests$ci95_high))
star_x <- pairwise_tests$ci95_high + 0.045 * span
stars <- as.character(symnum(pairwise_tests$p_adjusted, corr = FALSE, na = FALSE,
                              cutpoints = c(0, 0.001, 0.01, 0.05, 1),
                              symbols = c("***", "**", "*", "ns")))
png(result_file("stats_pairwise_contrasts.png"), width = 7.2, height = 4.7,
    units = "in", res = 150, type = "cairo")
par(mar = c(4.5, 8.6, 4.4, 1.3), las = 1, cex.axis = 1.05)
plot(pairwise_tests$estimate, position, type = "n", yaxt = "n", bty = "l",
     xlim = range(0, pairwise_tests$ci95_low, pairwise_tests$ci95_high) + c(-0.04, 0.14) * span,
     ylim = range(position) + c(-0.45, 0.45),
     xlab = sprintf("Difference in mean %s (%s)", if (spec$show_shift) "gain" else "asymmetry", spec$units),
     ylab = "")
axis(2, at = position, labels = contrast_labels, tick = FALSE)
grid(nx = NULL, ny = NA, col = "grey85", lty = 1)
abline(v = 0, lty = 2, col = "grey50")
with(pairwise_tests, {
  segments(ci95_low, position, ci95_high, position, lwd = 1.5)
  segments(ci95_low, position - 0.065, ci95_low, position + 0.065, lwd = 1.5)
  segments(ci95_high, position - 0.065, ci95_high, position + 0.065, lwd = 1.5)
  points(estimate, position, pch = 19, cex = 1.3)
})
text(star_x, position, labels = stars, adj = c(0, 0.5))
title("Paired contrasts and 95% CI", line = 2.4)
mtext("Tukey-adjusted confidence intervals and p-values", side = 3, line = 0.9, cex = 0.9)
invisible(dev.off())

cat(paste(lines, collapse = "\n"), "\n")
