#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测器测试
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.detector import WatermarkDetector, DetectionResult


class TestWatermarkDetector(unittest.TestCase):
    """测试水印检测器"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试配置
        self.config = {
            'data': {
                'time_column': 'timestamp',
                'value_column': 'close',
                'label_column': 'watermark_label',
            },
            'features': {
                'window_size': 10,
                'step_size': 5,
                'feature_groups': ['statistical', 'time_domain'],
            },
            'model': {
                'type': 'random_forest',
                'params': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'random_state': 42,
                }
            },
            'detection': {
                'threshold': 0.5,
                'batch_size': 100,
            },
            'output': {
                'save_features': False,
                'save_model': False,
            }
        }
        
        # 创建测试数据
        np.random.seed(42)
        n_samples = 100
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
        
        self.test_df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 101,
            'low': np.random.randn(n_samples).cumsum() + 99,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_samples),
            'watermark_label': np.random.randint(0, 2, n_samples),
        })
        self.test_df.set_index('timestamp', inplace=True)
        
        # 保存测试数据
        self.data_path = Path(self.temp_dir) / 'test_data.csv'
        self.test_df.to_csv(self.data_path)
        
        # 初始化检测器
        self.detector = WatermarkDetector(config=self.config)
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector.config)
        self.assertIsNotNone(self.detector.data_loader)
        self.assertIsNotNone(self.detector.feature_extractor)
        self.assertIsNotNone(self.detector.model_factory)
        self.assertIsNotNone(self.detector.evaluator)
        self.assertFalse(self.detector.is_trained)
    
    def test_default_config(self):
        """测试默认配置"""
        detector = WatermarkDetector()
        
        self.assertIn('data', detector.config)
        self.assertIn('features', detector.config)
        self.assertIn('model', detector.config)
        self.assertIn('detection', detector.config)
    
    def test_train(self):
        """测试训练"""
        # 训练模型
        self.detector.train(self.data_path, validation_split=0.3)
        
        # 检查是否已训练
        self.assertTrue(self.detector.is_trained)
    
    def test_detect(self):
        """测试检测"""
        # 先训练
        self.detector.train(self.data_path, validation_split=0.3)
        
        # 检测
        result = self.detector.detect(self.data_path)
        
        # 检查结果
        self.assertIsInstance(result, DetectionResult)
        self.assertIsInstance(result.predictions, np.ndarray)
        self.assertIsInstance(result.probabilities, np.ndarray)
        self.assertEqual(len(result.predictions), len(result.probabilities))
    
    def test_detect_without_training(self):
        """测试未训练时检测"""
        # 未训练时应该发出警告但仍返回结果
        result = self.detector.detect(self.data_path)
        
        self.assertIsInstance(result, DetectionResult)
    
    def test_evaluate(self):
        """测试评估"""
        # 先训练
        self.detector.train(self.data_path, validation_split=0.3)
        
        # 评估
        metrics = self.detector.evaluate(self.data_path)
        
        # 检查指标
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
    
    def test_save_and_load_model(self):
        """测试模型保存和加载"""
        # 训练模型
        self.detector.train(self.data_path, validation_split=0.3)
        
        # 保存模型
        model_path = Path(self.temp_dir) / 'test_model.pkl'
        self.detector.model_factory.save(model_path)
        
        # 创建新检测器并加载模型
        new_detector = WatermarkDetector(config=self.config)
        new_detector.load_model(model_path)
        
        # 检查是否已训练
        self.assertTrue(new_detector.is_trained)
    
    def test_train_with_dataframe(self):
        """测试使用DataFrame训练"""
        # 直接使用DataFrame训练
        self.detector.train(self.test_df, validation_split=0.3)
        
        self.assertTrue(self.detector.is_trained)


class TestDetectionResult(unittest.TestCase):
    """测试结果类"""
    
    def setUp(self):
        """测试前准备"""
        self.predictions = np.array([0, 1, 0, 1, 1])
        self.probabilities = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.4, 0.6],
        ])
        self.feature_importance = {
            'feature1': 0.5,
            'feature2': 0.3,
            'feature3': 0.2,
        }
        
        self.result = DetectionResult(
            predictions=self.predictions,
            probabilities=self.probabilities,
            feature_importance=self.feature_importance,
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertTrue(np.array_equal(self.result.predictions, self.predictions))
        self.assertTrue(np.array_equal(self.result.probabilities, self.probabilities))
        self.assertEqual(self.result.feature_importance, self.feature_importance)
    
    def test_summary(self):
        """测试摘要生成"""
        summary = self.result.summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn('总样本数', summary)
        self.assertIn('检测到水印', summary)
    
    def test_to_dataframe(self):
        """测试转换为DataFrame"""
        df = self.result.to_dataframe()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('prediction', df.columns)
        self.assertIn('probability_no_watermark', df.columns)
        self.assertIn('probability_watermark', df.columns)
        self.assertEqual(len(df), len(self.predictions))
    
    def test_save(self):
        """测试保存结果"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            output_dir = Path(temp_dir) / 'results'
            self.result.save(output_dir)
            
            # 检查文件是否创建
            self.assertTrue((output_dir / 'predictions.csv').exists())
            self.assertTrue((output_dir / 'summary.txt').exists())
        finally:
            shutil.rmtree(temp_dir)


class TestConfigLoading(unittest.TestCase):
    """测试配置加载"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试配置文件
        self.config_path = Path(self.temp_dir) / 'test_config.yaml'
        config = {
            'data': {
                'time_column': 'timestamp',
                'value_column': 'close',
            },
            'model': {
                'type': 'xgboost',
                'params': {'n_estimators': 50}
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_from_config_file(self):
        """测试从配置文件加载"""
        detector = WatermarkDetector(config_path=str(self.config_path))
        
        self.assertEqual(detector.config['data']['time_column'], 'timestamp')
        self.assertEqual(detector.config['model']['type'], 'xgboost')


if __name__ == '__main__':
    unittest.main()
