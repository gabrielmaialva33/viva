# VIVA-QD Paper Documentation

This directory contains the academic paper structure for GECCO 2025 submission.

## Paper Structure

| File | Section | Word Count (target) |
|------|---------|---------------------|
| [abstract.md](abstract.md) | Abstract | 250 words |
| [introduction.md](introduction.md) | 1. Introduction | 800 words |
| [related_work.md](related_work.md) | 2. Related Work | 1000 words |
| [methodology.md](methodology.md) | 3. Methodology | 1500 words |
| [experiments.md](experiments.md) | 4. Experiments | 1200 words |
| [conclusion.md](conclusion.md) | 5. Discussion & Conclusion | 800 words |

**Total target: ~5,500 words** (GECCO full paper limit: 8 pages)

## Compilation

To compile into a single LaTeX document:

```bash
# Using pandoc
pandoc abstract.md introduction.md related_work.md methodology.md \
       experiments.md conclusion.md \
       -o viva_qd_gecco2025.tex \
       --template=acmart.tex \
       --bibliography=references.bib

# Or directly to PDF
pandoc *.md -o viva_qd_gecco2025.pdf \
       --template=acmart.tex \
       --citeproc
```

## Key Metrics to Cite

| Metric | Value | Source |
|--------|-------|--------|
| Peak throughput | 320,000 evals/sec | `experiments.md` Section 4.2.1 |
| Speedup | 50,000x | `experiments.md` Section 4.2.2 |
| MAP-Elites coverage | 57.9% | `experiments.md` Section 4.3.1 |
| QD-Score | 908.3 | `experiments.md` Section 4.3.2 |
| Parallel simulations | 4,800 | `methodology.md` Section 3.4 |

## Figures (to be added)

- [ ] System architecture diagram
- [ ] HoloMAP-Elites algorithm flowchart
- [ ] Throughput vs batch size graph
- [ ] Training progression (fitness over generations)
- [ ] MAP-Elites heatmap visualization
- [ ] Ablation study comparison chart

## Target Venue

**GECCO 2025** - Genetic and Evolutionary Computation Conference
- Submission deadline: TBD (typically February)
- Conference: July 2025, Melbourne, Australia
- Track: Neuroevolution / Quality Diversity

## Authors

Gabriel Maia
Independent Researcher
Capao Bonito, SP, Brazil

## Contact

For collaboration or questions about the paper:
- GitHub Issues: https://github.com/gabrielmaialva33/viva/issues
