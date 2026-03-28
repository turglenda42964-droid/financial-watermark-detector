# Architecture

## Overview

The project follows a simple research pipeline:

1. Load a financial text corpus from `data/`.
2. Apply a lightweight lexical watermark to create paired samples.
3. Extract watermark-aware statistical features.
4. Train a detector to distinguish original vs watermarked text.
5. Export reports, figures, and reproducible artifacts.

## Modules

- `src/financial_watermark_detector/data.py`
  Handles corpus loading and summary generation.
- `src/financial_watermark_detector/watermarking.py`
  Implements lexical watermark embedding and paired dataset construction.
- `src/financial_watermark_detector/detector.py`
  Defines feature extraction, logistic detector training, and feature importance plots.
- `src/financial_watermark_detector/pipeline.py`
  Orchestrates the end-to-end experiment and writes artifacts.

## Output Artifacts

- `models/watermark_detector.joblib`
- `data/watermark_detection_dataset.csv`
- `reports/corpus_summary.json`
- `reports/watermark_metrics.json`
- `reports/watermark_feature_importance.png`
- `reports/watermark_samples.json`
