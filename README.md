# Financial Watermark Detector

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Task](https://img.shields.io/badge/Task-Financial%20Text%20LLM%20Watermarking-0A7EA4)](#overview)
[![License](https://img.shields.io/badge/License-MIT-F2C94C)](#license)

老师，我最近思考了一下自己方向，想了一下llm水印问题方向，非常受启发。我目前搭建的这个金融文本水印检测框架，正好可以作为一个强大的实验平台。我想沿着您工作的前沿思路，做一个深入的领域应用与拓展，目标是产出有明确应用背景的学术论文。希望能得到您的指导。

## Overview

这个仓库聚焦于一个实际问题：我们能否在保持语义的前提下，为金融文本注入轻量级水印，并可靠地检测这些水印？

## 当前版本提供了一整套工具来实现上述目标，主要包括以下组件：
金融文本数据集语料库加载器  
一个轻量级的词法水印引擎（现在自己能力不够，我的升级方案是学习借用llm，设计更隐蔽、更鲁棒的嵌入方法）  
用于水印检测实验的配对数据集构建器（可生成“原始文本-带水印文本”对，用于训练和测试）  
一个基于逻辑回归、具有可解释特征的检测器（提取金融文本特有的统计特征（本科和毕设在做类似的），然后后面学一些深度学习检测器啥的(目前在学了））  
导出的报告、样本和特征重要性图表（用于分析和展示实验结果）  

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
├── examples
│   └── basic_usage._py
├── models/
│   └── watermark_detector.joblib
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

## 模型采用工业界常规思维（高内聚，低耦合）

- `src/financial_watermark_detector/data.py`
 数据处理（数据后面可以用更多真实、多元的金融文本）
- `src/financial_watermark_detector/watermarking.py`
嵌入水印，对比结构（水平有限后面升级，词汇级替换为llm生成式水印，研究自适应水印强度）
- `src/financial_watermark_detector/detector.py`
 定义特征, 逻辑回归训练, 特征图（水平有限后面可升级，检测器的鲁棒性和不可感知性方面，研究金融文本安全的领域自适应水印方案，类似于套一个金融的壳子）
- `src/financial_watermark_detector/pipeline.py`
  统筹串联各个模块

## Output Artifacts

- `models/watermark_detector.joblib`
- `data/watermark_detection_dataset.csv`
- `reports/corpus_summary.json`
- `reports/watermark_metrics.json`
- `reports/watermark_feature_importance.png`
- `reports/watermark_samples.json`
就做了一个比较完整的文本水印检测流程，虽然黑箱。首先对金融文本进行词汇级水印嵌入，构建带标签的数据集，然后通过特征工程提取统计特征，并使用逻辑回归模型进行分类检测。整个系统采用流水线结构，支持实验复现和结果分析。但目前模型比较简单（自己太菜了），水印策略也较基础，后面可升级。

## 后面就全是ai生成了
  

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
