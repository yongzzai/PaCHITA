# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PatchBPAD (Patch-based Business Process Anomaly Detection) — a research project implementing a novel architecture that decomposes event logs into separate attribute channels, encodes each independently using patch embeddings + Transformer encoders, and reconstructs next-patch predictions via CrossAttention + GRU decoders. Anomalies are detected through reconstruction error. The full architecture is specified in `docs/PLAN.md`.

The core model (`model/model.py`) is not yet implemented. `model/compo.py` contains an RMSNorm component. The `baseline/` directory contains 10+ existing anomaly detection methods used for comparison.

## Commands

```bash
# Generate synthetic event logs from process models
uv run python -m generator.gen_anomalous_eventlog_syn

# Run baseline experiments (all baselines x all datasets, each in a subprocess)
uv run python main_unsup.py

# Run proposed method
uv run main.py
```

## Architecture

### Data Pipeline

1. **Process models** (PLG2 XML files) live in `generator/process_models/`
2. **Generator** (`generator/`) creates synthetic event logs with injected anomalies (Skip, Rework, Early, Late, Insert, Attribute) → outputs gzipped JSON to `eventlogs/`
3. **Dataset** (`utils/dataset.py`) loads event logs, integer-encodes categorical attributes, caches as `eventlogs/cache/*.pkl.gz`. Features are stored as a list of NumPy arrays (one per attribute), shape `(num_cases, max_case_len)`. Start/end symbols `▶`/`■` are prepended/appended to traces.
4. **Evaluation** (`utils/eval.py`) computes best precision/recall/F1 at optimal threshold + AUPR via sklearn

### Baseline Interface

All anomaly detectors extend `baseline/basic.py:AnomalyDetector` with two methods:
- `fit(dataset)` — train on a `Dataset` object
- `detect(dataset)` — returns a tuple of `(trace_scores, event_scores, attr_scores)` where each is a NumPy array or `None`
- **PatchBPAD class must follow this class format**

Results are written per-detector to `results/result_{name}.csv`.

### Key Domain Concepts

- **Channels**: Activity is the primary channel; additional event attributes (Attr1, Attr2, ...) are secondary channels, all categorical
- **Patches**: Sliding windows of size `w` over per-channel token sequences; each patch is embedded via `RMSNorm → Concat → Linear`
- **Anomaly types** (`utils/enums.py:Class`): Insert, Skip, Rework, Early, Late, Shift, Replace, Attribute — encoded as integers 2-9, with 0=Normal and 1=generic Anomaly
- **Anomaly scoring**: For each predicted position, score = sum of probabilities exceeding P(true label); per-token score = max across overlapping patches

### Process Mining Core (`processmining/`)

`EventLog` loads from JSON(.gz), XES(.gz), CSV, or SQL. `Case` = single trace with events and attributes. `HeuristicsMiner` mines adjacency-matrix process models.

### Dataset Naming

Generated logs follow `{model}-{anomaly_p:.2f}-{number}` pattern (e.g., `small-0.10-1.json.gz`). Real-life logs are any name not matching the synthetic pattern keywords (gigantic, huge, large, medium, p2p, paper, small, wide).