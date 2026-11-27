# Q9 statistics for mouse-area FOOOF aperiodic exponents.
# Usage: Rscript code/questions/q9_fooof_stats.R [results directory]

args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(), value = TRUE)[1]
script <- sub("^--file=", "", script_arg)
repo <- normalizePath(file.path(dirname(script), "..", ".."), winslash = "/", mustWork = TRUE)
output <- if (length(args)) args[1] else file.path(repo, "results", "q9")

fits <- read.csv(file.path(output, "q9_fooof_fits.csv"), stringsAsFactors = FALSE)
areas <- as.character(1:3)
primary_range <- "25-45"
required <- c(
  "mouse_id", "brain_area", "fit_range", "aperiodic_exponent",
  "relative_excitation_proxy", "r_squared", "passes_r_squared"
)
if (!all(required %in% names(fits)) || anyNA(fits[required]) ||
    anyDuplicated(fits[c("mouse_id", "brain_area", "fit_range")]) ||
    !setequal(unique(as.character(fits$brain_area)), areas) ||
    any(!is.finite(fits$aperiodic_exponent)) || any(!is.finite(fits$r_squared)) ||
    any(abs(fits$relative_excitation_proxy + fits$aperiodic_exponent) > 1e-10)) {
  stop("Expected unique, finite FOOOF results for all three brain areas.")
}
if (!requireNamespace("nlme", quietly = TRUE)) {
  stop("This analysis requires the R package nlme.")
}

fits$mouse_id <- factor(fits$mouse_id)
fits$brain_area <- factor(fits$brain_area, levels = areas)
fits$fit_range <- factor(fits$fit_range,
  levels = c("15-35", "20-40", "25-45", "30-50"))
fits$passes_r_squared <- as.logical(fits$passes_r_squared)

analyze_range <- function(range_name) {
  rows <- fits[as.character(fits$fit_range) == range_name, ]
  complete <- aggregate(
    cbind(n_areas = rep(1L, nrow(rows)), n_pass = as.integer(rows$passes_r_squared)) ~ mouse_id,
    data = rows,
    FUN = sum
  )
  complete_ids <- as.character(complete$mouse_id[
    complete$n_areas == length(areas) & complete$n_pass == length(areas)
  ])
  data <- droplevels(rows[as.character(rows$mouse_id) %in% complete_ids, ])
  n_mice <- nlevels(data$mouse_id)
  if (n_mice < 3L || nrow(data) != n_mice * length(areas)) {
    stop(sprintf("Range %s has fewer than three mice with valid fits in every area.", range_name))
  }

  # The random intercept accounts for the three within-mouse area measurements.
  # Area-specific residual variances avoid assuming equal spread across regions.
  model <- nlme::lme(
    aperiodic_exponent ~ brain_area,
    random = ~ 1 | mouse_id,
    weights = nlme::varIdent(form = ~ 1 | brain_area),
    data = data,
    method = "REML"
  )
  overall <- anova(model)["brain_area", , drop = FALSE]
  design <- model.matrix(
    ~ brain_area,
    data.frame(brain_area = factor(areas, levels = areas))
  )
  coefficients <- nlme::fixef(model)
  covariance <- vcov(model)
  exponent_means <- as.numeric(design %*% coefficients)
  mean_se <- sqrt(diag(design %*% covariance %*% t(design)))
  mean_df <- n_mice - 1L
  mean_margin <- qt(0.975, mean_df) * mean_se
  means <- data.frame(
    fit_range = range_name,
    brain_area = areas,
    n_mice = n_mice,
    exponent = exponent_means,
    exponent_se = mean_se,
    exponent_lower_95 = exponent_means - mean_margin,
    exponent_upper_95 = exponent_means + mean_margin,
    relative_excitation_proxy = -exponent_means,
    excitation_lower_95 = -(exponent_means + mean_margin),
    excitation_upper_95 = -(exponent_means - mean_margin),
    df = mean_df
  )

  pair_indices <- combn(seq_along(areas), 2)
  contrast_design <- (design[pair_indices[1, ], , drop = FALSE]
                      - design[pair_indices[2, ], , drop = FALSE])
  differences <- as.numeric(contrast_design %*% coefficients)
  contrast_se <- sqrt(diag(contrast_design %*% covariance %*% t(contrast_design)))
  contrast_df <- overall$denDF[1]
  t_value <- differences / contrast_se
  tukey_margin <- qtukey(0.95, length(areas), contrast_df) / sqrt(2) * contrast_se
  tukey_p <- ptukey(
    sqrt(2) * abs(t_value), length(areas), contrast_df, lower.tail = FALSE
  )
  contrasts <- data.frame(
    fit_range = range_name,
    area_a = areas[pair_indices[1, ]],
    area_b = areas[pair_indices[2, ]],
    exponent_difference_a_minus_b = differences,
    exponent_lower_95 = differences - tukey_margin,
    exponent_upper_95 = differences + tukey_margin,
    relative_excitation_difference_a_minus_b = -differences,
    excitation_lower_95 = -(differences + tukey_margin),
    excitation_upper_95 = -(differences - tukey_margin),
    se = contrast_se,
    df = contrast_df,
    t_ratio = t_value,
    p_adjusted = tukey_p
  )

  winner <- means$brain_area[which.min(means$exponent)]
  winner_supported <- with(contrasts,
    (area_a == winner & exponent_upper_95 < 0) |
    (area_b == winner & exponent_lower_95 > 0)
  )
  answer <- if (sum(winner_supported) == length(areas) - 1L) {
    paste("Brain area", winner)
  } else {
    "Not enough data / no differences"
  }
  overall_table <- data.frame(
    fit_range = range_name,
    n_mice = n_mice,
    numerator_df = overall$numDF,
    denominator_df = overall$denDF,
    f_value = overall$`F-value`,
    p_value = overall$`p-value`,
    lowest_exponent_area = winner,
    q9_answer = answer
  )
  exclusions <- rows
  exclusions$complete_mouse_for_range <- as.character(rows$mouse_id) %in% complete_ids

  list(
    data = data,
    means = means,
    contrasts = contrasts,
    overall = overall_table,
    exclusions = exclusions,
    model = model,
    answer = answer,
    winner = winner
  )
}

ranges <- levels(fits$fit_range)
analyses <- lapply(ranges, analyze_range)
names(analyses) <- ranges

means <- do.call(rbind, lapply(analyses, `[[`, "means"))
contrasts <- do.call(rbind, lapply(analyses, `[[`, "contrasts"))
overall <- do.call(rbind, lapply(analyses, `[[`, "overall"))
inclusion <- do.call(rbind, lapply(analyses, `[[`, "exclusions"))
primary <- analyses[[primary_range]]

write.csv(means, file.path(output, "q9_stats_means.csv"), row.names = FALSE)
write.csv(contrasts, file.path(output, "q9_stats_contrasts.csv"), row.names = FALSE)
write.csv(overall, file.path(output, "q9_stats_overall.csv"), row.names = FALSE)
write.csv(inclusion, file.path(output, "q9_stats_fit_inclusion.csv"), row.names = FALSE)

writeLines(capture.output({
  cat("Q9 primary model: fixed, peak-free FOOOF exponent over 25-45 Hz\n")
  cat("Only mice with R-squared >= 0.95 in all three areas enter a range-specific model.\n\n")
  print(summary(primary$model))
  cat("\nPrimary overall area effect\n")
  print(primary$overall, row.names = FALSE)
  cat("\nEstimated marginal mean exponents\n")
  print(primary$means, row.names = FALSE)
  cat("\nTukey-adjusted exponent contrasts\n")
  print(primary$contrasts, row.names = FALSE)
  cat("\nSensitivity across fit ranges\n")
  print(overall, row.names = FALSE)
}), file.path(output, "q9_stats_model.txt"))

format_p <- function(x) ifelse(x < 0.0001, "p < 0.0001", sprintf("p = %.4g", x))
primary_means <- primary$means
estimate_text <- with(primary_means, sprintf(
  "area %s: %.3f (95%% CI %.3f to %.3f)",
  brain_area, exponent, exponent_lower_95, exponent_upper_95
))
primary_overall <- primary$overall[1, ]
sensitivity_text <- vapply(analyses, function(item) {
  sprintf("%s Hz: lowest exponent in area %s; answer = %s",
          item$overall$fit_range, item$winner, item$answer)
}, character(1))

answer_lines <- c(
  paste0("Q9 answer: ", primary$answer),
  "",
  paste0(
    "The primary analysis used the 25-45 Hz FOOOF aperiodic exponent. ",
    "A smaller exponent indicates a flatter spectrum and stronger relative excitation."
  ),
  sprintf(
    "The mixed-effects model included %d complete mice and gave F(%d, %d) = %.2f, %s.",
    primary_overall$n_mice, primary_overall$numerator_df,
    primary_overall$denominator_df, primary_overall$f_value,
    format_p(primary_overall$p_value)
  ),
  paste("Model-estimated exponents:", paste(estimate_text, collapse = "; "), "."),
  if (primary$answer == "Not enough data / no differences") {
    sprintf(
      "Area %s had the smallest estimated exponent, but Tukey-adjusted comparisons did not distinguish it from both other areas.",
      primary$winner
    )
  } else {
    sprintf(
      "Area %s had a significantly smaller exponent than both other areas and therefore the strongest relative excitation.",
      primary$winner
    )
  },
  "",
  "Sensitivity fits (Supplementary Fig. 2 ranges):",
  paste0("- ", sensitivity_text),
  "",
  "The aperiodic exponent is a population-level proxy for relative E-I balance, not a direct measurement of synaptic excitation or inhibition."
)
writeLines(answer_lines, file.path(output, "answer.txt"))
