# VIVA-QD Technical Documentation

This directory contains technical reference documentation for VIVA-QD implementation.

## Contents

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | System architecture, data flow, component details |
| [api_reference.md](api_reference.md) | Complete API documentation for all modules |
| [performance_guide.md](performance_guide.md) | Optimization, tuning, benchmarking |

## Quick Links

### Architecture

- [System Diagram](architecture.md#system-diagram)
- [Gleam Application Layer](architecture.md#1-gleam-application-layer)
- [Rust NIF Interface](architecture.md#31-nif-interface)
- [CUDA Compute Layer](architecture.md#4-cuda-compute-layer)
- [Data Flow](architecture.md#data-flow)

### API Reference

- [viva/neural/neat](api_reference.md#vivaneuralneat) - NEAT neuroevolution
- [viva/neural/holomap](api_reference.md#vivaneuralholomap) - MAP-Elites
- [viva/glands](api_reference.md#vivaglands) - GPU HRR operations
- [viva/billiards/sinuca](api_reference.md#vivabilliards) - Game domain

### Performance

- [Batch Size Optimization](performance_guide.md#batch-size-optimization)
- [Parallelization Strategies](performance_guide.md#parallelization-strategies)
- [Memory Optimization](performance_guide.md#memory-optimization)
- [Common Issues](performance_guide.md#common-performance-issues)

## Getting Help

1. Check the [API Reference](api_reference.md) for function signatures
2. Review [Architecture](architecture.md) for system understanding
3. See [Performance Guide](performance_guide.md) for optimization
4. Open GitHub issue for bugs or questions

## Contributing

When adding new documentation:

1. Follow existing format and style
2. Include code examples where applicable
3. Update this README with new entries
4. Ensure all links are relative and functional
