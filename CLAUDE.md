# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PatchBPAD (Patch-based Business Process Anomaly Detection) — a research project implementing a novel architecture that decomposes event logs into separate attribute channels, encodes each independently using patch embeddings + Transformer encoders, and reconstructs next-patch predictions via CrossAttention + GRU decoders. Anomalies are detected through reconstruction error. The full architecture is specified in `docs/PLAN.md`.

The core model is implemented across `model/model.py` (PatchBPAD wrapper + PatchBPADNet nn.Module) and `model/compo.py` (PatchEmbedding, ChannelEncoder, ChannelDecoderAct, ChannelDecoderAttr, RMSNorm). The `baseline/` directory contains 10+ existing anomaly detection methods used for comparison.

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
- **Patches**: Sliding windows of size `w` over per-channel token sequences; each patch is embedded via `Concat[TokenEmb, PosEmb] → RMSNorm → Flatten → Linear`
- **Anomaly types** (`utils/enums.py:Class`): Insert, Skip, Rework, Early, Late, Shift, Replace, Attribute — encoded as integers 2-9, with 0=Normal and 1=generic Anomaly
- **Anomaly scoring**: For each predicted position, score = sum of probabilities exceeding P(true label); per-token score = max across overlapping patches

### Process Mining Core (`processmining/`)

`EventLog` loads from JSON(.gz), XES(.gz), CSV, or SQL. `Case` = single trace with events and attributes. `HeuristicsMiner` mines adjacency-matrix process models.

### PatchBPAD Model (`model/`)

**`model/compo.py`** — Neural network components:
- `RMSNorm` — Root mean square normalization
- `PatchEmbedding` — `(B, P, w)` → `(B, P, d_model)`: Concat[TokenEmb, PosEmb] → RMSNorm(d_emb*2) → Linear(w*d_emb*2 → d_model, bias=False)
- `ChannelEncoder` — `(B, P, d_model)` → `(B, P, d_model)`: TransformerEncoder with configurable `enc_dropout`, supports `src_key_padding_mask`
- `ChannelDecoderAct` — Activity decoder. Input: patch_emb, enc_output. Output: logits `(B, P, w, V+1)`, hc_act `(B, P, d_model)` (detached). Flow: CrossAttn → Add&RMSNorm → GRU(bias=False, h0 from enc mean-pool, input=hc_act) → Head(cat[dropout(patch_emb), gru_out], bias=False)
- `ChannelDecoderAttr` — Attribute decoder. Input: patch_emb, enc_output, hc_act. Output: logits `(B, P, w, V+1)`. Flow: CrossAttn → Add&RMSNorm → GRU(bias=False, h0 from enc mean-pool, input=cat[hc_attr, hc_act]) → Head(cat[dropout(patch_emb), hc_act, gru_out], bias=False)

**`model/model.py`** — Model integration:
- `PatchBPADNet(nn.Module)` — Wires K channel-specific PatchEmbeddings, Encoders, and Decoders. Forward pass: embed → encode all channels → decode activity first → decode attributes with detached `hc_act`. Each channel's PatchEmbedding is shared between its encoder and decoder.
- `PatchBPAD` — Wrapper with `fit(dataset)` / `detect(dataset)`. Calls `dataset._gen_patches(window_size)` for preprocessing, trains with AdamW (`weight_decay=1e-3`) + CosineAnnealingLR (`eta_min=lr/10`) + gradient clipping (`max_norm=5.0`). Loss: masked cross-entropy (uniform weights). Scoring: softmax probability aggregation with max-overlap token scoring.

**Default hyperparameters:** `window_size=3, d_emb=16, d_model=64, nhead=4, num_enc_layers=2, num_dec_layers=2, d_ff=128, d_gru=64, enc_dropout=0.3, dec_dropout=0.3, n_epochs=16, batch_size=64, lr=0.0004`

**Dataset patch preprocessing** (`utils/dataset.py:_gen_patches`): Called by the model, stores `patches`, `decoder_patches`, `patch_mask`, `patch_padding_mask` on the Dataset instance. Decoder input at step `t` = target patch at step `t-1` (zero-padded for step 0). `patch_padding_mask` marks patches that are entirely padding (used for encoder/decoder attention masking).

### Dataset Naming

Generated logs follow `{model}-{anomaly_p:.2f}-{number}` pattern (e.g., `small-0.10-1.json.gz`). Real-life logs are any name not matching the synthetic pattern keywords (gigantic, huge, large, medium, p2p, paper, small, wide).