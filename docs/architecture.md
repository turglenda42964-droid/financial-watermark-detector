# Architecture

## Overview

The project follows a simple research pipeline:

1. Load a financial text corpus from `data/`.
2. Apply a lightweight lexical watermark to create paired samples.（有监督学习）
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
就做了一个比较完整的文本水印检测流程。首先对金融文本进行词汇级水印嵌入，构建带标签的数据集，然后通过特征工程提取统计特征，并使用逻辑回归模型进行分类检测。整个系统采用流水线结构，支持实验复现和结果分析。但目前模型比较简单（自己太菜了），水印策略也较基础，在复杂的情况下会出现优化空间。
