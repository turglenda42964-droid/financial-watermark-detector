"""
金融时序数据水印检测系统

核心模块：
- data_loader: 数据加载器
- feature_extractor: 特征提取器
- model_factory: 模型工厂
- evaluator: 评估器
- detector: 主检测器
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .detector import WatermarkDetector
from .data_loader import DataLoader
from .feature_extractor import FeatureExtractor
from .model_factory import ModelFactory
from .evaluator import Evaluator

__all__ = [
    "WatermarkDetector",
    "DataLoader",
    "FeatureExtractor",
    "ModelFactory",
    "Evaluator",
]
