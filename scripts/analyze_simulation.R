#!/usr/bin/env Rscript
# VIVA Simulation Analysis in R
# Run: Rscript scripts/analyze_simulation.R

cat("\n")
cat("╔════════════════════════════════════════════════════════════╗\n")
cat("║           VIVA STATISTICAL ANALYSIS                        ║\n")
cat("║           R Analysis Suite                                 ║\n")
cat("╚════════════════════════════════════════════════════════════╝\n\n")

# Find latest v2 simulation
files <- list.files("data/simulations", pattern = "sim_v2_.*_stats\\.csv", full.names = TRUE)
if (length(files) == 0) {
  stop("No v2 simulation data found!")
}
latest <- sort(files, decreasing = TRUE)[1]
run_id <- gsub("_stats\\.csv", "", basename(latest))
cat("Analyzing:", run_id, "\n\n")

# Load data
stats <- read.csv(paste0("data/simulations/", run_id, "_stats.csv"))
pads <- read.csv(paste0("data/simulations/", run_id, "_pads.csv"))
personalities <- read.csv(paste0("data/simulations/", run_id, "_personalities.csv"))
events <- read.csv(paste0("data/simulations/", run_id, "_events.csv"))

# Merge personality info
pads <- merge(pads, personalities, by = "viva_id")
pads$personality <- pads$personality.x
pads$personality.x <- NULL
pads$personality.y <- NULL

cat("═══════════════════════════════════════════════════════════════\n")
cat("                    DESCRIPTIVE STATISTICS                      \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# Overall summary
cat("Overall PAD Summary:\n")
cat("────────────────────\n")
summary_stats <- data.frame(
  Dimension = c("Pleasure", "Arousal", "Dominance"),
  Mean = c(mean(pads$pleasure), mean(pads$arousal), mean(pads$dominance)),
  SD = c(sd(pads$pleasure), sd(pads$arousal), sd(pads$dominance)),
  Min = c(min(pads$pleasure), min(pads$arousal), min(pads$dominance)),
  Max = c(max(pads$pleasure), max(pads$arousal), max(pads$dominance)),
  Skewness = c(
    (mean(pads$pleasure) - median(pads$pleasure)) / sd(pads$pleasure),
    (mean(pads$arousal) - median(pads$arousal)) / sd(pads$arousal),
    (mean(pads$dominance) - median(pads$dominance)) / sd(pads$dominance)
  )
)
print(summary_stats, row.names = FALSE)

cat("\n\n═══════════════════════════════════════════════════════════════\n")
cat("                    PERSONALITY ANALYSIS                        \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# Stats by personality
cat("Mean PAD by Personality:\n")
cat("────────────────────────\n")
personality_means <- aggregate(cbind(pleasure, arousal, dominance) ~ personality,
                                data = pads, FUN = mean)
print(personality_means, row.names = FALSE)

cat("\n\nStandard Deviation by Personality:\n")
cat("──────────────────────────────────\n")
personality_sd <- aggregate(cbind(pleasure, arousal, dominance) ~ personality,
                             data = pads, FUN = sd)
print(personality_sd, row.names = FALSE)

cat("\n\n═══════════════════════════════════════════════════════════════\n")
cat("                    ANOVA: Personality Effects                   \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# ANOVA for each PAD dimension
cat("Pleasure ~ Personality:\n")
aov_p <- aov(pleasure ~ personality, data = pads)
print(summary(aov_p))

cat("\nArousal ~ Personality:\n")
aov_a <- aov(arousal ~ personality, data = pads)
print(summary(aov_a))

cat("\nDominance ~ Personality:\n")
aov_d <- aov(dominance ~ personality, data = pads)
print(summary(aov_d))

cat("\n═══════════════════════════════════════════════════════════════\n")
cat("                    CORRELATION ANALYSIS                        \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# Correlations between PAD dimensions
cat("PAD Correlations:\n")
cat("─────────────────\n")
cor_matrix <- cor(pads[, c("pleasure", "arousal", "dominance")])
print(round(cor_matrix, 4))

cat("\n\nCorrelation Significance Tests:\n")
cat("───────────────────────────────\n")
cat("P-A:", cor.test(pads$pleasure, pads$arousal)$p.value, "\n")
cat("P-D:", cor.test(pads$pleasure, pads$dominance)$p.value, "\n")
cat("A-D:", cor.test(pads$arousal, pads$dominance)$p.value, "\n")

cat("\n═══════════════════════════════════════════════════════════════\n")
cat("                    TEMPORAL DYNAMICS                           \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# Analyze change over time
cat("Temporal Regression (PAD means over ticks):\n")
cat("───────────────────────────────────────────\n")

lm_p <- lm(mean_p ~ tick, data = stats)
lm_a <- lm(mean_a ~ tick, data = stats)
lm_d <- lm(mean_d ~ tick, data = stats)

cat("\nPleasure trend: slope =", coef(lm_p)[2], ", R² =", summary(lm_p)$r.squared, "\n")
cat("Arousal trend:  slope =", coef(lm_a)[2], ", R² =", summary(lm_a)$r.squared, "\n")
cat("Dominance trend: slope =", coef(lm_d)[2], ", R² =", summary(lm_d)$r.squared, "\n")

cat("\n═══════════════════════════════════════════════════════════════\n")
cat("                    EVENT IMPACT ANALYSIS                       \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

cat("Global Events:\n")
cat("──────────────\n")
print(events)

# Analyze variance before/after events
if (nrow(events) > 0) {
  cat("\n\nVariance Change After Events:\n")
  cat("─────────────────────────────\n")
  for (i in 1:nrow(events)) {
    event_tick <- events$tick[i]
    before <- pads[pads$tick == event_tick - 25, ]
    after <- pads[pads$tick == event_tick + 25, ]

    if (nrow(before) > 0 && nrow(after) > 0) {
      var_change_p <- var(after$pleasure) - var(before$pleasure)
      var_change_a <- var(after$arousal) - var(before$arousal)
      cat(sprintf("Event '%s' (t=%d): ΔVar(P)=%.4f, ΔVar(A)=%.4f\n",
                  events$label[i], event_tick, var_change_p, var_change_a))
    }
  }
}

cat("\n═══════════════════════════════════════════════════════════════\n")
cat("                    EMOTIONAL QUADRANTS                         \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# Classify into emotional quadrants
pads$quadrant <- ifelse(pads$pleasure > 0 & pads$arousal > 0, "Excited/Happy",
                 ifelse(pads$pleasure > 0 & pads$arousal <= 0, "Calm/Happy",
                 ifelse(pads$pleasure <= 0 & pads$arousal > 0, "Stressed",
                        "Depressed")))

cat("Overall Quadrant Distribution:\n")
cat("──────────────────────────────\n")
quad_dist <- table(pads$quadrant)
quad_pct <- prop.table(quad_dist) * 100
print(data.frame(Quadrant = names(quad_dist),
                 Count = as.vector(quad_dist),
                 Percent = round(as.vector(quad_pct), 2)))

cat("\n\nQuadrant Distribution by Personality:\n")
cat("─────────────────────────────────────\n")
quad_by_pers <- table(pads$personality, pads$quadrant)
print(prop.table(quad_by_pers, margin = 1) * 100)

cat("\n═══════════════════════════════════════════════════════════════\n")
cat("                    CHI-SQUARE TEST                             \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

cat("Is Personality associated with Emotional Quadrant?\n")
chi_test <- chisq.test(quad_by_pers)
print(chi_test)

cat("\n═══════════════════════════════════════════════════════════════\n")
cat("                    STABILITY ANALYSIS                          \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# Calculate emotional stability per VIVA
stability <- aggregate(cbind(pleasure, arousal, dominance) ~ viva_id + personality,
                        data = pads, FUN = sd)
names(stability)[3:5] <- c("p_volatility", "a_volatility", "d_volatility")
stability$total_volatility <- stability$p_volatility + stability$a_volatility + stability$d_volatility

cat("Emotional Volatility by Personality:\n")
cat("────────────────────────────────────\n")
vol_by_pers <- aggregate(total_volatility ~ personality, data = stability,
                          FUN = function(x) c(mean = mean(x), sd = sd(x)))
print(vol_by_pers)

cat("\n\nMost Stable VIVAs (lowest volatility):\n")
stable_vivas <- stability[order(stability$total_volatility), ][1:5, ]
print(stable_vivas[, c("viva_id", "personality", "total_volatility")])

cat("\n\nMost Volatile VIVAs:\n")
volatile_vivas <- stability[order(-stability$total_volatility), ][1:5, ]
print(volatile_vivas[, c("viva_id", "personality", "total_volatility")])

cat("\n═══════════════════════════════════════════════════════════════\n")
cat("                    ADVANCED METRICS (Qwen3 Suggestions)        \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# 1. Emotional Inertia (Autocorrelation at lag 1)
cat("Emotional Inertia (Autocorrelation lag-1):\n")
cat("──────────────────────────────────────────\n")

inertia_by_viva <- function(viva_data) {
  if (nrow(viva_data) < 3) return(NA)
  viva_data <- viva_data[order(viva_data$tick), ]
  acf_p <- acf(viva_data$pleasure, lag.max = 1, plot = FALSE)$acf[2]
  acf_a <- acf(viva_data$arousal, lag.max = 1, plot = FALSE)$acf[2]
  acf_d <- acf(viva_data$dominance, lag.max = 1, plot = FALSE)$acf[2]
  return(c(p = acf_p, a = acf_a, d = acf_d))
}

inertia_results <- do.call(rbind, lapply(split(pads, pads$viva_id), function(v) {
  inertia <- inertia_by_viva(v)
  data.frame(viva_id = v$viva_id[1], personality = v$personality[1],
             inertia_p = inertia["p"], inertia_a = inertia["a"], inertia_d = inertia["d"])
}))

inertia_by_pers <- aggregate(cbind(inertia_p, inertia_a, inertia_d) ~ personality,
                              data = inertia_results, FUN = mean, na.rm = TRUE)
print(inertia_by_pers, row.names = FALSE)

cat("\nInterpretation: Higher inertia = emotional states persist longer\n")

# 2. RMSSD (Root Mean Square of Successive Differences)
cat("\n\nRMSSD (Emotional Variability):\n")
cat("─────────────────────────────\n")

rmssd <- function(x) {
  diffs <- diff(x)
  sqrt(mean(diffs^2, na.rm = TRUE))
}

rmssd_by_viva <- do.call(rbind, lapply(split(pads, pads$viva_id), function(v) {
  v <- v[order(v$tick), ]
  data.frame(viva_id = v$viva_id[1], personality = v$personality[1],
             rmssd_p = rmssd(v$pleasure), rmssd_a = rmssd(v$arousal), rmssd_d = rmssd(v$dominance))
}))

rmssd_by_pers <- aggregate(cbind(rmssd_p, rmssd_a, rmssd_d) ~ personality,
                            data = rmssd_by_viva, FUN = mean)
print(rmssd_by_pers, row.names = FALSE)

cat("\nInterpretation: Higher RMSSD = more tick-to-tick variability\n")

# 3. Event Impact Score (ΔPAD in first 100 ticks after event)
cat("\n\nEvent Impact Scores:\n")
cat("───────────────────\n")

impact_scores <- data.frame()
for (i in 1:nrow(events)) {
  event_tick <- events$tick[i]
  pre_window <- pads[pads$tick >= (event_tick - 50) & pads$tick < event_tick, ]
  post_window <- pads[pads$tick >= event_tick & pads$tick < (event_tick + 100), ]

  if (nrow(pre_window) > 0 && nrow(post_window) > 0) {
    pre_mean <- c(mean(pre_window$pleasure), mean(pre_window$arousal), mean(pre_window$dominance))
    post_mean <- c(mean(post_window$pleasure), mean(post_window$arousal), mean(post_window$dominance))
    delta <- post_mean - pre_mean

    impact_scores <- rbind(impact_scores, data.frame(
      event = events$label[i],
      tick = event_tick,
      delta_p = round(delta[1], 4),
      delta_a = round(delta[2], 4),
      delta_d = round(delta[3], 4),
      magnitude = round(sqrt(sum(delta^2)), 4)
    ))
  }
}
print(impact_scores, row.names = FALSE)

# 4. State Transition Matrix (GLOBAL)
cat("\n\nEmotional State Transition Matrix (GLOBAL - all personalities):\n")
cat("───────────────────────────────────────────────────────────────\n")

pads_sorted <- pads[order(pads$viva_id, pads$tick), ]
pads_sorted$next_quadrant <- c(pads_sorted$quadrant[-1], NA)
pads_sorted$same_viva <- c(pads_sorted$viva_id[-1] == pads_sorted$viva_id[-nrow(pads_sorted)], FALSE)
transitions <- pads_sorted[pads_sorted$same_viva, c("quadrant", "next_quadrant", "personality")]

trans_matrix <- table(transitions$quadrant, transitions$next_quadrant)
trans_prob <- prop.table(trans_matrix, margin = 1) * 100
cat("Transition probabilities (%):\n")
print(round(trans_prob, 2))

# 4b. Transition Matrix BY PERSONALITY (diagonal = stickiness)
cat("\n\nStickiness by Personality (% staying in same state):\n")
cat("────────────────────────────────────────────────────\n")

stickiness_by_pers <- sapply(split(transitions, transitions$personality), function(t) {
  same_state <- sum(t$quadrant == t$next_quadrant, na.rm = TRUE)
  total <- nrow(t)
  round(same_state / total * 100, 2)
})

stickiness_df <- data.frame(personality = names(stickiness_by_pers),
                            stickiness_pct = as.vector(stickiness_by_pers))
stickiness_df <- stickiness_df[order(-stickiness_df$stickiness_pct), ]
print(stickiness_df, row.names = FALSE)

cat("\nInterpretation: High stickiness + low entropy = trapped in ONE state\n")
cat("                High stickiness + high entropy = stable within MULTIPLE states\n")

# 5. Effect Sizes
cat("\n\n═══════════════════════════════════════════════════════════════\n")
cat("                    EFFECT SIZES                                \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# Eta-squared for ANOVA
cat("Eta-squared (η²) for ANOVA:\n")
cat("──────────────────────────\n")

eta_sq <- function(aov_result) {
  ss <- summary(aov_result)[[1]][["Sum Sq"]]
  ss[1] / sum(ss)
}

eta_p <- eta_sq(aov_p)
eta_a <- eta_sq(aov_a)
eta_d <- eta_sq(aov_d)

cat(sprintf("Pleasure:  η² = %.4f (%s effect)\n", eta_p,
            ifelse(eta_p > 0.14, "large", ifelse(eta_p > 0.06, "medium", "small"))))
cat(sprintf("Arousal:   η² = %.4f (%s effect)\n", eta_a,
            ifelse(eta_a > 0.14, "large", ifelse(eta_a > 0.06, "medium", "small"))))
cat(sprintf("Dominance: η² = %.4f (%s effect)\n", eta_d,
            ifelse(eta_d > 0.14, "large", ifelse(eta_d > 0.06, "medium", "small"))))

# Cramer's V for Chi-Square
cat("\n\nCramer's V for Chi-Square:\n")
cat("─────────────────────────\n")

cramers_v <- function(chi_result) {
  n <- sum(chi_result$observed)
  k <- min(nrow(chi_result$observed), ncol(chi_result$observed))
  sqrt(chi_result$statistic / (n * (k - 1)))
}

v <- cramers_v(chi_test)
cat(sprintf("Cramer's V = %.4f (%s association)\n", v,
            ifelse(v > 0.5, "strong", ifelse(v > 0.3, "moderate", ifelse(v > 0.1, "weak", "negligible")))))

# 6. Entropy of Emotional States
cat("\n\nState Entropy by Personality:\n")
cat("────────────────────────────\n")

entropy <- function(probs) {
  probs <- probs[probs > 0]
  -sum(probs * log2(probs))
}

entropy_by_pers <- sapply(split(pads, pads$personality), function(p) {
  quad_freq <- table(p$quadrant) / nrow(p)
  entropy(as.vector(quad_freq))
})

entropy_df <- data.frame(personality = names(entropy_by_pers), entropy = round(entropy_by_pers, 4))
print(entropy_df, row.names = FALSE)

cat("\nMax entropy (4 states) = 2.00. Higher = more diverse emotional states.\n")
cat("\nNOTE: Entropy measures state DISTRIBUTION, not transition speed!\n")
cat("- neurotic: low entropy (0.03) = always in Stressed (99.6%)\n")
cat("- calm: high entropy (1.75) = spread across states (51% Excited, 23% Calm, 13% Depressed, 13% Stressed)\n")
cat("- This does NOT contradict high stickiness! calm visits many states but stays long in each.\n")

cat("\n═══════════════════════════════════════════════════════════════\n")
cat("                    ADVANCED ANALYSIS (Qwen3 Next Steps)        \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# ═══════════════════════════════════════════════════════════════
# STEP 1: Second-Order Markov Chain (Contextual Transitions via Events)
# ═══════════════════════════════════════════════════════════════
cat("1. CONTEXTUAL TRANSITIONS (Second-Order Markov)\n")
cat("───────────────────────────────────────────────\n")
cat("How do events affect state transitions?\n\n")

# Find transitions near events
event_window <- 50  # ticks before/after event
contextual_trans <- data.frame()

for (i in 1:nrow(events)) {
  event_tick <- events$tick[i]
  event_label <- events$label[i]

  # Get transitions within window after event
  post_event <- transitions[transitions$personality %in% unique(pads$personality), ]
  # We need tick info - reconstruct from pads_sorted
  pads_with_trans <- pads_sorted[pads_sorted$same_viva &
                                  pads_sorted$tick >= event_tick &
                                  pads_sorted$tick < event_tick + event_window, ]

  if (nrow(pads_with_trans) > 0) {
    trans_count <- sum(pads_with_trans$quadrant != pads_with_trans$next_quadrant, na.rm = TRUE)
    total <- nrow(pads_with_trans)
    trans_rate <- trans_count / total * 100

    contextual_trans <- rbind(contextual_trans, data.frame(
      event = event_label,
      tick = event_tick,
      transitions = trans_count,
      total_ticks = total,
      transition_rate = round(trans_rate, 2)
    ))
  }
}

cat("Transition Rate by Event (% state changes in 50 ticks after):\n")
print(contextual_trans, row.names = FALSE)

# Compare to baseline
baseline_trans_rate <- sum(transitions$quadrant != transitions$next_quadrant, na.rm = TRUE) /
                        nrow(transitions) * 100
cat(sprintf("\nBaseline transition rate: %.2f%%\n", baseline_trans_rate))
cat("Events with rate > baseline indicate 'stickiness breaking' events.\n")

# ═══════════════════════════════════════════════════════════════
# STEP 2: Emotional Fluency Metric
# ═══════════════════════════════════════════════════════════════
cat("\n\n2. EMOTIONAL FLUENCY METRIC\n")
cat("───────────────────────────\n")
cat("Fluency = (1 - stickiness) × entropy × recovery_speed\n\n")

# Calculate recovery speed from inertia (inverse relationship)
# Lower inertia = faster recovery
recovery_speed_by_pers <- 1 - inertia_by_pers$inertia_p

# Merge stickiness and entropy
fluency_df <- merge(stickiness_df, entropy_df, by = "personality")
fluency_df$recovery_speed <- recovery_speed_by_pers[match(fluency_df$personality, inertia_by_pers$personality)]

# Calculate fluency
fluency_df$fluency <- (1 - fluency_df$stickiness_pct/100) * fluency_df$entropy * fluency_df$recovery_speed
fluency_df$fluency <- round(fluency_df$fluency, 6)

# Sort by fluency
fluency_df <- fluency_df[order(-fluency_df$fluency), ]

cat("Emotional Fluency by Personality:\n")
print(fluency_df[, c("personality", "stickiness_pct", "entropy", "recovery_speed", "fluency")], row.names = FALSE)

cat("\nInterpretation:\n")
cat("- Higher fluency = better emotional adaptability\n")
cat("- Balanced/Calm should have highest (fluid but stable)\n")
cat("- Neurotic lowest (stuck in one state)\n")

# ═══════════════════════════════════════════════════════════════
# STEP 3: Opportunity Windows (Post-Crisis Intervention Efficacy)
# ═══════════════════════════════════════════════════════════════
cat("\n\n3. OPPORTUNITY WINDOWS (Post-Event Intervention Timing)\n")
cat("────────────────────────────────────────────────────────\n")
cat("When is the best time to intervene after a crisis?\n\n")

# Analyze PAD recovery trajectory after crisis events
crisis_events <- events[events$label %in% c("crisis", "disappointment", "uncertainty"), ]

window_analysis <- data.frame()
windows <- c(10, 25, 50, 100, 200)

for (i in 1:nrow(crisis_events)) {
  event_tick <- crisis_events$tick[i]
  event_label <- crisis_events$label[i]

  # Get pre-event baseline
  pre_data <- pads[pads$tick >= (event_tick - 50) & pads$tick < event_tick, ]
  if (nrow(pre_data) == 0) next

  baseline_p <- mean(pre_data$pleasure)

  # Measure recovery at different windows
  for (w in windows) {
    post_data <- pads[pads$tick >= event_tick & pads$tick < (event_tick + w), ]
    if (nrow(post_data) == 0) next

    current_p <- mean(post_data$pleasure)
    delta_from_baseline <- current_p - baseline_p
    recovery_pct <- ifelse(baseline_p != 0,
                           (1 - abs(delta_from_baseline) / abs(baseline_p)) * 100,
                           NA)

    window_analysis <- rbind(window_analysis, data.frame(
      event = event_label,
      window = w,
      delta_p = round(delta_from_baseline, 4),
      recovery_pct = round(recovery_pct, 1)
    ))
  }
}

# Aggregate by window
window_summary <- aggregate(cbind(delta_p, recovery_pct) ~ window,
                            data = window_analysis, FUN = mean, na.rm = TRUE)
window_summary$delta_p <- round(window_summary$delta_p, 4)
window_summary$recovery_pct <- round(window_summary$recovery_pct, 1)

cat("Recovery Trajectory After Crisis Events:\n")
print(window_summary, row.names = FALSE)

cat("\nOptimal Intervention Window: ")
optimal_window <- window_summary$window[which.max(abs(window_summary$delta_p))]
cat(sprintf("%d ticks (maximum delta = highest potential impact)\n", optimal_window))

# By personality
cat("\n\nRecovery by Personality (at 100 ticks post-crisis):\n")
recovery_by_pers <- data.frame()

for (pers in unique(pads$personality)) {
  pers_data <- pads[pads$personality == pers, ]

  for (i in 1:nrow(crisis_events)) {
    event_tick <- crisis_events$tick[i]

    pre <- pers_data[pers_data$tick >= (event_tick - 50) & pers_data$tick < event_tick, ]
    post <- pers_data[pers_data$tick >= event_tick & pers_data$tick < (event_tick + 100), ]

    if (nrow(pre) > 0 && nrow(post) > 0) {
      delta <- mean(post$pleasure) - mean(pre$pleasure)
      recovery_by_pers <- rbind(recovery_by_pers, data.frame(
        personality = pers,
        event = crisis_events$label[i],
        delta_p = delta
      ))
    }
  }
}

recovery_summary <- aggregate(delta_p ~ personality, data = recovery_by_pers,
                              FUN = function(x) round(mean(x), 4))
recovery_summary <- recovery_summary[order(recovery_summary$delta_p, decreasing = TRUE), ]
print(recovery_summary, row.names = FALSE)

# ═══════════════════════════════════════════════════════════════
# STEP 4: Transition Heatmap by Personality
# ═══════════════════════════════════════════════════════════════
cat("\n\n4. TRANSITION HEATMAP BY PERSONALITY\n")
cat("────────────────────────────────────\n")

# Build transition matrix for each personality
states <- c("Calm/Happy", "Depressed", "Excited/Happy", "Stressed")

cat("\nTransition Matrices by Personality (% probabilities):\n")

for (pers in unique(pads$personality)) {
  pers_trans <- transitions[transitions$personality == pers, ]

  if (nrow(pers_trans) > 10) {
    trans_table <- table(pers_trans$quadrant, pers_trans$next_quadrant)
    trans_prob <- prop.table(trans_table, margin = 1) * 100

    cat(sprintf("\n=== %s ===\n", toupper(pers)))
    print(round(trans_prob, 1))
  }
}

# Visual ASCII heatmap
cat("\n\nASCII Trajectory Visualization (state sequence pattern):\n")
cat("─────────────────────────────────────────────────────────\n")

state_chars <- c("Calm/Happy" = "C", "Depressed" = "D", "Excited/Happy" = "E", "Stressed" = "S")

for (pers in unique(pads$personality)) {
  pers_data <- pads[pads$personality == pers & pads$viva_id == min(pads$viva_id[pads$personality == pers]), ]
  pers_data <- pers_data[order(pers_data$tick), ]

  # Sample every 250 ticks for visualization
  sampled <- pers_data[seq(1, nrow(pers_data), by = 25), ]
  trajectory <- paste(state_chars[sampled$quadrant], collapse = "")

  # Show first 60 chars
  trajectory <- substr(trajectory, 1, 60)
  cat(sprintf("%10s: %s\n", pers, trajectory))
}

cat("\nLegend: C=Calm/Happy, D=Depressed, E=Excited/Happy, S=Stressed\n")

cat("\n═══════════════════════════════════════════════════════════════\n")
cat("                    ANALYSIS COMPLETE                           \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# Generate plots
cat("Generating plots...\n")

png(paste0("data/simulations/", run_id, "_analysis.png"), width = 1600, height = 1200, res = 150)
par(mfrow = c(2, 3))

# 1. PAD over time
plot(stats$tick, stats$mean_p, type = "l", col = "red", lwd = 2,
     xlab = "Tick", ylab = "PAD Value", main = "PAD Dimensions Over Time",
     ylim = c(-1, 1))
lines(stats$tick, stats$mean_a, col = "green", lwd = 2)
lines(stats$tick, stats$mean_d, col = "blue", lwd = 2)
legend("topright", c("Pleasure", "Arousal", "Dominance"),
       col = c("red", "green", "blue"), lwd = 2, cex = 0.8)
abline(h = 0, lty = 2, col = "gray")

# 2. Variance over time
plot(stats$tick, stats$std_p, type = "l", col = "red", lwd = 2,
     xlab = "Tick", ylab = "Standard Deviation", main = "Emotional Diversity Over Time")
lines(stats$tick, stats$std_a, col = "green", lwd = 2)
lines(stats$tick, stats$std_d, col = "blue", lwd = 2)
legend("topright", c("P std", "A std", "D std"),
       col = c("red", "green", "blue"), lwd = 2, cex = 0.8)

# 3. PA scatter by personality (final state)
final_tick <- max(pads$tick)
final_pads <- pads[pads$tick == final_tick, ]
cols <- c("optimist" = "gold", "neurotic" = "purple", "calm" = "skyblue",
          "energetic" = "orange", "balanced" = "gray")
plot(final_pads$pleasure, final_pads$arousal,
     col = cols[final_pads$personality], pch = 19,
     xlab = "Pleasure", ylab = "Arousal", main = "Final State by Personality",
     xlim = c(-1, 1), ylim = c(-1, 1))
abline(h = 0, v = 0, lty = 2, col = "gray")
legend("topright", names(cols), col = cols, pch = 19, cex = 0.7)

# 4. Boxplot by personality
boxplot(pleasure ~ personality, data = pads, col = cols,
        main = "Pleasure Distribution by Personality",
        xlab = "Personality", ylab = "Pleasure")

# 5. Quadrant pie chart
pie(quad_dist, main = "Emotional Quadrant Distribution",
    col = c("lightgreen", "lightblue", "salmon", "gray"))

# 6. Volatility by personality
boxplot(total_volatility ~ personality, data = stability, col = cols,
        main = "Emotional Volatility by Personality",
        xlab = "Personality", ylab = "Total Volatility")

dev.off()

cat("Saved:", paste0("data/simulations/", run_id, "_analysis.png"), "\n")
cat("\nDone!\n")
