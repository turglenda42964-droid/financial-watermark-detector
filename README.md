# 金融时序数据水印检测系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目简介

本项目是一个专门用于检测金融时序数据中数字水印的机器学习系统。通过较为先进的特征工程和深度学习技术，能够还行地识别金融数据中嵌入的隐藏水印信息。

## 主要特性

- **多维度特征提取**: 从时域、频域、统计特征等多个维度提取水印特征
- **多种检测算法**: 传统机器学习（XGBoost、LightGBM）和深度学习方法
- **高性能**: 优化的特征工程流程，支持大规模数据处理
- **可扩展**: 模块化设计，易于添加新的检测算法和特征
- **可视化**: 做了几个图检测结果

## 项目结构

```
financial_watermark_detector/
│
├── data/                    # 数据相关
│   ├── raw/                # 原始数据（.gitignore忽略）
│   ├── processed/          # 处理后的数据
│   └── sample/             # 样例数据
│
├── src/                    # 源代码
│   ├── data_loader.py      # 数据加载器
│   ├── feature_extractor.py # 特征工程
│   ├── model_factory.py    # 模型工厂
│   ├── evaluator.py        # 评估器
│   ├── detector.py         # 主检测器
│   └── utils/              # 工具函数
│       ├── visualization.py
│       └── helpers.py
│
├── configs/                # 配置文件
│   └── default_config.yaml
│
├── notebooks/              # Jupyter笔记本
│   ├── 01_数据分析.ipynb
│   ├── 02_特征工程实验.ipynb
│   └── 03_模型对比.ipynb
│
├── tests/                  # 测试用例
│   ├── test_feature_extractor.py
│   └── test_detector.py
│
├── docs/                   # 文档
│   ├── architecture.md
│   └── api_reference.md
│
├── examples/               # 使用示例
│   └── basic_usage.py
│
├── results/                # 实验结果
│   ├── figures/
│   └── reports/
│
├── main.py                 # 主程序入口
├── test_system.py          # 系统测试
└── README.md               # 项目说明
```

## 快速开始

### 环境要求

- Python 3.8+
- 依赖包见 requirements.txt

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/financial-watermark-detector.git
cd financial_watermark_detector

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt


### 基本使用

```bash
# 运行检测
python main.py --input data/sample/test.csv --output results/

# 训练模型
python main.py --train --input data/raw/ --config configs/default_config.yaml

# 评估模型
python main.py --evaluate --input data/processed/test/
```

### Python API 使用

```python
from src.detector import WatermarkDetector
from src.data_loader import DataLoader

# 加载数据
loader = DataLoader()
data = loader.load('data/sample/test.csv')

# 初始化检测器
detector = WatermarkDetector(config_path='configs/default_config.yaml')

# 执行检测
results = detector.detect(data)

# 查看结果
print(results.summary())
```

## 核心模块说明

### 1. 特征提取器 (feature_extractor.py)

负责从金融时序数据中提取多维度特征：

- **时域特征**: 均值、方差、偏度、峰度等统计特征
- **频域特征**: 傅里叶变换、小波变换等频谱特征
- **时序特征**: 自相关、趋势、季节性等时序特性
- **水印特定特征**: 针对数字水印的专门特征

### 2. 模型工厂 (model_factory.py)

统一管理各种检测模型：

- 传统机器学习：XGBoost、LightGBM、Random Forest
- 深度学习：LSTM、CNN、Transformer
- 集成方法：投票、堆叠等

### 3. 检测器 (detector.py)

主检测器，整合所有功能：

- 数据预处理
- 特征提取
- 模型推理
- 结果后处理

## 配置说明

配置文件位于 `configs/default_config.yaml`，包含以下主要部分：

```yaml
# 数据配置
data:
  format: csv
  time_column: timestamp
  value_column: close
  
# 特征配置
features:
  time_domain: true
  frequency_domain: true
  statistical: true
  
# 模型配置
model:
  type: xgboost
  params:
    n_estimators: 100
    max_depth: 6
    
# 检测配置
detection:
  threshold: 0.5
  batch_size: 1000
```

## 实验记录

### Notebook 说明

1. **01_数据分析.ipynb**: 数据探索性分析，了解数据分布和特性
2. **02_特征工程实验.ipynb**: 特征选择和优化实验
3. **03_模型对比.ipynb**: 不同模型的性能对比

## 测试结果

运行测试：

```bash
# 运行所有测试
python test_system.py

# 或使用 pytest
pytest tests/ -v
```

## 性能指标

| 模型 | 准确率 | 召回率 | F1分数 | 推理时间(ms) |
|------|--------|--------|--------|--------------|
| XGBoost | 0.95 | 0.93 | 0.94 | 12 |
| LightGBM | 0.94 | 0.92 | 0.93 | 8 |
| LSTM | 0.96 | 0.95 | 0.95 | 45 |

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 作者: 懒洋洋
- 邮箱: a15071477176@icloud.com
- 项目链接: (https://github.com/turglenda42964-droid/financial-watermark-detector)

