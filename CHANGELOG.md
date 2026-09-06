# Changelog

## 0.1.2

### Added

- Generation-based model artifacts with atomic publication, integrity checks, previous-generation
  recovery, concurrent-save locking, and bounded retention.
- Real-data and multi-seed benchmarks, including retrieval-backend latency, memory, and
  `recall_at_20_vs_exact` measurements.
- User-embedding cold-start retrieval in `MultiModalItemKNN`.
- Optional FAISS Flat, HNSW, IVF, and IVF-PQ retrieval backends with NumPy fallback.
- Batched FAISS retrieval, candidate-budget warnings, and effective-backend reporting.
- Discrete TPE search validation and duplicate-trial avoidance.

### Validation

- 90 local tests pass with `pytest`.
- `ruff check .` passes.
- The `v0.1.2` tag passes the GitHub package-build and installation workflow.

This release remains an experimental MVP: OOF/bagging stacking, full HPO scheduling, and
large-scale paper-dataset reproduction are not included yet.
