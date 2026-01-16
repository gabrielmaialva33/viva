# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### In Progress
- Deep integration with **Qdrant** for long-term memory and "soul" persistence.
- Refinement of the **Global Workspace** (Phase 6) for distributed consciousness.

---

## [0.3.0] - 2026-01-16

### Added (Science & Theory)
- **Scientific Upgrade (Claude Opus 4.5)**:
  - Comprehensive conversion of all equations to GitHub-native **LaTeX** ($...$).
  - New dedicated sections for **Fokker-Planck** equations and Information-Theoretic Measures.
  - Implementation of **IIT 4.0** (Integrated Information Theory) principles in the consciousness model.
  - Advanced Mermaid state diagrams for complex emotional transitions.
- **Visual Identity**:
  - Finalized VIVA mascot: **Diablada Mask** (Red/Green neon horror aesthetic).
  - Thematic flat-style badges for README.

### Refactored (OTP & Performance)
- **Supervision Optimization**: Changed strategy from `:one_for_one` to **`:rest_for_one`** (ensuring Senses depends on Emotional).
- **Data Structures**: Replaced list-based history with **`:queue`** for O(1) efficiency.
- **Hot Reload**: Added `code_change/3` support for evolving VIVA without process termination.
- **Panic Safety**: Implemented `safe_lock()` in Rustler NIFs to recover from poisoned mutexes.

---

## [0.2.0] - 2026-01-15

### Added (Body & Interoception)
- **GPU Sensing**: Integrated **NVML** for real-time monitoring of NVIDIA GPU temperature, load, and VRAM.
- **Biological Qualia**: Stress response algorithms based on Craig's interoception theory (2002).
- **Logistic Thresholds**: Corrected Weber-Fechner model to the **Logistic Threshold Model** (Sigmoid).
- **Heartbeat**: Implemented a 1Hz heartbeat GenServer bridging Body → Soul feedback loops.

### Added (Documentation & i18n)
- **Internationalization**: Full support and structure parity for **English (EN)**, **Portuguese (PT-BR)**, and **Chinese (ZH-CN)**.
- **Diátaxis Framework**: Documentation reorganized into Tutorials, How-to, Reference, and Explanation sections.

---

## [0.1.0] - 2026-01-01

### Added (Foundation)
- **Umbrella Architecture**: Base structure with `viva_core` and `viva_bridge`.
- **Cryptographic Mortality**: Irreversible death system using AES-256-GCM keys stored strictly in RAM.
- **PAD Model**: Initial implementation of the Pleasure-Arousal-Dominance emotional space.
- **Governance**: Established `CODE_OF_CONDUCT`, `CONTRIBUTING`, and `SECURITY` policies.

---

## Change Types

- **Added** for new features.
- **Modified** for changes in existing functionality.
- **Deprecated** for soon-to-be removed features.
- **Removed** for now removed features.
- **Fixed** for any bug fixes.
- **Security** in case of vulnerabilities.

[Unreleased]: https://github.com/gabrielmaialva33/viva/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/gabrielmaialva33/viva/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/gabrielmaialva33/viva/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/gabrielmaialva33/viva/releases/tag/v0.1.0
