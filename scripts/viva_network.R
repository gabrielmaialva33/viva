#!/usr/bin/env Rscript
# VIVA Network Science Visualization
# Analyzes holographic memory space dynamics
#
# Usage: Rscript scripts/viva_network.R

library(tidyverse)
library(viridis)

# Configuration
VIVA_API <- "http://localhost:8888"
OUTPUT_DIR <- "output"

# Ensure output directory exists
dir.create(OUTPUT_DIR, showWarnings = FALSE)

cat("=== VIVA Network Science Visualization ===\n\n")

# Fetch data from VIVA telemetry server
cat("Fetching data from", VIVA_API, "...\n")
data <- tryCatch({
  read_csv(paste0(VIVA_API, "/api/export/csv"), show_col_types = FALSE)
}, error = function(e) {
  cat("Error: Could not connect to VIVA server.\n")
  cat("Make sure the server is running: gleam run -m viva/telemetry/demo\n")
  quit(status = 1)
})

cat("Loaded", nrow(data), "memory bodies\n\n")

# === 1. HRR-4D Scatter Plot (x, y projection) ===
cat("Generating HRR-4D scatter plot...\n")
p1 <- ggplot(data, aes(x = x, y = y, color = energy, size = energy)) +
  geom_point(alpha = 0.8) +
  geom_text(aes(label = label), vjust = -1.5, size = 3, color = "gray30") +
  scale_color_viridis(option = "plasma", name = "Energy") +
  scale_size_continuous(range = c(3, 12), guide = "none") +
  labs(
    title = "VIVA Holographic Memory Space",
    subtitle = "HRR-4D projected to 2D (x, y coordinates)",
    x = "X (HRR dimension 1)",
    y = "Y (HRR dimension 2)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "gray50"),
    panel.grid.minor = element_blank()
  ) +
  coord_fixed()

ggsave(file.path(OUTPUT_DIR, "viva_scatter_xy.png"), p1, width = 10, height = 8, dpi = 150)
cat("  -> output/viva_scatter_xy.png\n")

# === 2. Alternative projection (z, w) ===
cat("Generating z-w projection...\n")
p2 <- ggplot(data, aes(x = z, y = w, color = energy, size = energy)) +
  geom_point(alpha = 0.8) +
  geom_text(aes(label = label), vjust = -1.5, size = 3, color = "gray30") +
  scale_color_viridis(option = "magma", name = "Energy") +
  scale_size_continuous(range = c(3, 12), guide = "none") +
  labs(
    title = "VIVA Memory Space (Z-W Projection)",
    subtitle = "Higher dimensions of HRR-4D space",
    x = "Z (HRR dimension 3)",
    y = "W (HRR dimension 4)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "gray50")
  ) +
  coord_fixed()

ggsave(file.path(OUTPUT_DIR, "viva_scatter_zw.png"), p2, width = 10, height = 8, dpi = 150)
cat("  -> output/viva_scatter_zw.png\n")

# === 3. Energy Distribution ===
cat("Generating energy distribution...\n")
p3 <- ggplot(data, aes(x = energy, fill = sleeping)) +
  geom_histogram(bins = 20, alpha = 0.8, color = "white") +
  geom_density(aes(y = after_stat(count)), alpha = 0.3, fill = NA, color = "purple", linewidth = 1) +
  scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "coral"),
                    labels = c("TRUE" = "Dormant", "FALSE" = "Active"),
                    name = "State") +
  labs(
    title = "Energy Distribution of Memory Bodies",
    subtitle = paste("N =", nrow(data), "bodies"),
    x = "Energy Level",
    y = "Count"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "gray50")
  )

ggsave(file.path(OUTPUT_DIR, "viva_energy_dist.png"), p3, width = 10, height = 6, dpi = 150)
cat("  -> output/viva_energy_dist.png\n")

# === 4. Island Clustering ===
cat("Generating island clustering...\n")
p4 <- ggplot(data, aes(x = x, y = y, color = factor(island_id))) +
  geom_point(size = 5, alpha = 0.8) +
  geom_text(aes(label = label), vjust = -1.5, size = 3, color = "gray30") +
  scale_color_brewer(palette = "Set2", name = "Island") +
  labs(
    title = "Memory Clustering by Island",
    subtitle = "Spatial grouping of related memories",
    x = "X", y = "Y"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "gray50")
  ) +
  coord_fixed()

ggsave(file.path(OUTPUT_DIR, "viva_islands.png"), p4, width = 10, height = 8, dpi = 150)
cat("  -> output/viva_islands.png\n")

# === 5. Summary Statistics ===
cat("\n=== Summary Statistics ===\n")
cat("Total bodies:", nrow(data), "\n")
cat("Active:", sum(!data$sleeping), "\n")
cat("Dormant:", sum(data$sleeping), "\n")
cat("Islands:", n_distinct(data$island_id), "\n")
cat("Mean energy:", round(mean(data$energy), 4), "\n")
cat("Energy variance:", round(var(data$energy), 6), "\n")

# Spatial spread
cat("\nSpatial extent:\n")
cat("  X range: [", round(min(data$x), 2), ",", round(max(data$x), 2), "]\n")
cat("  Y range: [", round(min(data$y), 2), ",", round(max(data$y), 2), "]\n")
cat("  Z range: [", round(min(data$z), 2), ",", round(max(data$z), 2), "]\n")
cat("  W range: [", round(min(data$w), 2), ",", round(max(data$w), 2), "]\n")

cat("\n=== Done! ===\n")
