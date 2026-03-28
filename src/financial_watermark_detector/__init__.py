"""Financial watermark detector package."""

from .data import FinancialTextDataLoader, CorpusSummary
from .watermarking import FinancialWatermarkEngine, WatermarkDatasetBuilder
from .detector import WatermarkDetector, WatermarkTrainingResult
from .pipeline import run_pipeline

__all__ = [
    "FinancialTextDataLoader",
    "CorpusSummary",
    "FinancialWatermarkEngine",
    "WatermarkDatasetBuilder",
    "WatermarkDetector",
    "WatermarkTrainingResult",
    "run_pipeline",
]
