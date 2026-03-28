# Financial Watermark Detector

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Task](https://img.shields.io/badge/Task-Financial%20Text%20LLM%20Watermarking-0A7EA4)](#overview)
[![License](https://img.shields.io/badge/License-MIT-F2C94C)](#license)

A research-oriented project for **LLM watermark generation and detection in financial text**.  
It uses a financial text corpus to synthesize paired original/watermarked samples, trains a detector, and exports reproducible artifacts for demos, reports, and GitHub presentation.

## Overview

This repository focuses on a practical question:

**Can we inject lightweight watermarks into financial text while preserving semantics, and then detect those watermarks reliably?**

The current version provides:

- A corpus loader for financial text datasets
- A lightweight lexical watermark engine
- A paired dataset builder for watermark detection experiments
- A logistic-regression detector with interpretable features
- Exported reports, samples, and feature-importance figures

## Repository Structure

```text
financial-watermark-detector/
├── configs/
│   └── default_config.yaml
├── data/
│   ├── data.csv
│   └── watermark_detection_dataset.csv
├── docs/
│   └── architecture.md
├── examples/
│   └── basic_usage.py
├── models/
│   └── watermark_detector.joblib
├── notebooks/
│   └── 01_data_exploration.ipynb
├── reports/
│   ├── corpus_summary.json
│   ├── watermark_feature_importance.png
│   ├── watermark_metrics.json
│   └── watermark_samples.json
├── results/
│   ├── figures/
│   └── reports/
├── src/
│   └── financial_watermark_detector/
│       ├── __init__.py
│       ├── data.py
│       ├── detector.py
│       ├── pipeline.py
│       └── watermarking.py
├── tests/
│   └── test_watermarking.py
├── .gitignore
├── main.py
└── requirements.txt
```

## Method

The pipeline is intentionally simple and easy to extend:

1. Load financial text from `data/data.csv` or `data/data.cvs`
2. Apply a lexical watermark using a secret-key-controlled greenlist rule
3. Build paired samples: `original` vs `watermarked`
4. Extract watermark-aware statistical features
5. Train a detector and export reports

### Current Watermark Strategy

The current watermarking mechanism is a lightweight lexical prototype:

- It targets finance-friendly replaceable words such as `profit -> earnings` and `market -> trading`
- It uses a deterministic key-driven rule to decide whether a candidate position is "green"
- It creates detectable distributional bias without needing an online LLM backend

This makes the repository good for:

- Coursework demos
- Early-stage research framing
- GitHub portfolio presentation
- Later replacement with stronger token-level watermark methods

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare data

Place your financial text corpus at one of these locations:

- `data/data.csv`
- `data/data.cvs`

The default text column is `Sentence`.

### 3. Run the full pipeline

```bash
python3 main.py
```

### 4. Example usage

```bash
python3 examples/basic_usage.py
```

## Generated Artifacts

After running the pipeline, the repository exports:

- `models/watermark_detector.joblib`
- `data/watermark_detection_dataset.csv`
- `reports/corpus_summary.json`
- `reports/watermark_metrics.json`
- `reports/watermark_feature_importance.png`
- `reports/watermark_samples.json`

## Current Results

On the current corpus-derived paired dataset, the detector generated:

- Accuracy: `0.9564`
- Macro F1: `0.9564`
- ROC-AUC: `0.9952`

These numbers reflect the current synthetic lexical watermark setting, so they should be interpreted as a **baseline experiment**, not a final benchmark.

## Why This Repository Is Structured This Way

The repository is organized to look and feel like a clean GitHub research project:

- `src/financial_watermark_detector/` keeps the core logic package-based
- `configs/` makes the pipeline easier to explain and extend
- `docs/` holds architecture-level documentation
- `examples/` gives immediate runnable entry points
- `tests/` adds a minimal verification layer
- `reports/` and `results/` make outputs presentation-friendly

## Next Research Extensions

- Replace lexical watermarking with token-level LLM watermarking
- Add paraphrase, truncation, and summarization attack evaluation
- Compare logistic baselines with BiLSTM or Transformer detectors
- Measure semantic fidelity vs detectability trade-offs
- Extend from financial headlines to reports, filings, and QA outputs

## License

MIT
