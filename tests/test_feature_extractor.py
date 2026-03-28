#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取器测试
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.feature_extractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    """测试特征提取器"""
    
    def setUp(self):
        """测试前准备"""
        self.extractor = FeatureExtractor(
            window_size=10,
            step_size=2,
            feature_groups=['statistical', 'time_domain']
        )
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='1min')
        self.test_df = pd.DataFrame({
            'close': np.random.randn(50).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.extractor.window_size, 10)
        self.assertEqual(self.extractor.step_size, 2)
        self.assertEqual(len(self.extractor.feature_groups), 2)
    
    def test_extract(self):
        """测试特征提取"""
        features_df = self.extractor.extract(self.test_df, target_column='close')
        
        # 检查返回类型
        self.assertIsInstance(features_df, pd.DataFrame)
        
        # 检查是否包含窗口信息
        self.assertIn('window_start', features_df.columns)
        self.assertIn('window_end', features_df.columns)
        
        # 检查样本数量
        expected_samples = (len(self.test_df) - self.extractor.window_size) // self.extractor.step_size + 1
        self.assertEqual(len(features_df), expected_samples)
    
    def test_statistical_features(self):
        """测试统计特征"""
        extractor = FeatureExtractor(
            window_size=10,
            step_size=5,
            feature_groups=['statistical']
        )
        
        features_df = extractor.extract(self.test_df, target_column='close')
        
        # 检查统计特征
        self.assertIn('stat_mean', features_df.columns)
        self.assertIn('stat_std', features_df.columns)
        self.assertIn('stat_min', features_df.columns)
        self.assertIn('stat_max', features_df.columns)
    
    def test_time_domain_features(self):
        """测试时域特征"""
        extractor = FeatureExtractor(
            window_size=10,
            step_size=5,
            feature_groups=['time_domain']
        )
        
        features_df = extractor.extract(self.test_df, target_column='close')
        
        # 检查时域特征
        self.assertIn('td_return_mean', features_df.columns)
        self.assertIn('td_return_std', features_df.columns)
        self.assertIn('td_trend_slope', features_df.columns)
    
    def test_frequency_domain_features(self):
        """测试频域特征"""
        extractor = FeatureExtractor(
            window_size=20,
            step_size=5,
            feature_groups=['frequency_domain']
        )
        
        features_df = extractor.extract(self.test_df, target_column='close')
        
        # 检查频域特征
        self.assertIn('fd_fft_mean', features_df.columns)
        self.assertIn('fd_psd_mean', features_df.columns)
    
    def test_complexity_features(self):
        """测试复杂度特征"""
        extractor = FeatureExtractor(
            window_size=20,
            step_size=5,
            feature_groups=['complexity']
        )
        
        features_df = extractor.extract(self.test_df, target_column='close')
        
        # 检查复杂度特征
        self.assertIn('complexity_approx_entropy', features_df.columns)
        self.assertIn('complexity_higuchi_fd', features_df.columns)
    
    def test_watermark_specific_features(self):
        """测试水印特定特征"""
        extractor = FeatureExtractor(
            window_size=10,
            step_size=5,
            feature_groups=['watermark_specific']
        )
        
        features_df = extractor.extract(self.test_df, target_column='close')
        
        # 检查水印特定特征
        self.assertIn('wm_lsb_uniformity', features_df.columns)
        self.assertIn('wm_diff_mean', features_df.columns)
    
    def test_fit_transform(self):
        """测试拟合和转换"""
        X_scaled = self.extractor.fit_transform(self.test_df, target_column='close')
        
        # 检查返回类型
        self.assertIsInstance(X_scaled, np.ndarray)
        
        # 检查标准化（均值为0，标准差为1）
        np.testing.assert_almost_equal(X_scaled.mean(axis=0), 0, decimal=5)
        np.testing.assert_almost_equal(X_scaled.std(axis=0), 1, decimal=5)
    
    def test_transform(self):
        """测试转换（使用已拟合的scaler）"""
        # 先拟合
        self.extractor.fit_transform(self.test_df, target_column='close')
        
        # 再转换
        X_scaled = self.extractor.transform(self.test_df, target_column='close')
        
        self.assertIsInstance(X_scaled, np.ndarray)
    
    def test_empty_data(self):
        """测试空数据"""
        empty_df = pd.DataFrame({'close': []})
        
        with self.assertRaises(Exception):
            self.extractor.extract(empty_df, target_column='close')
    
    def test_invalid_column(self):
        """测试无效列名"""
        with self.assertRaises(KeyError):
            self.extractor.extract(self.test_df, target_column='invalid_column')


class TestFeatureExtractorHelpers(unittest.TestCase):
    """测试特征提取器辅助函数"""
    
    def test_calculate_max_drawdown(self):
        """测试最大回撤计算"""
        extractor = FeatureExtractor()
        
        # 上升趋势，无回撤
        values_up = np.array([1, 2, 3, 4, 5])
        self.assertEqual(extractor._calculate_max_drawdown(values_up), 0)
        
        # 下降趋势
        values_down = np.array([5, 4, 3, 2, 1])
        drawdown = extractor._calculate_max_drawdown(values_down)
        self.assertGreater(drawdown, 0)
    
    def test_calculate_entropy(self):
        """测试熵计算"""
        extractor = FeatureExtractor()
        
        # 均匀分布，熵最大
        uniform = np.ones(10)
        entropy_uniform = extractor._calculate_entropy(uniform)
        
        # 单点分布，熵为0
        single = np.zeros(10)
        single[0] = 1
        entropy_single = extractor._calculate_entropy(single)
        
        self.assertGreater(entropy_uniform, entropy_single)
    
    def test_detect_periodicity(self):
        """测试周期性检测"""
        extractor = FeatureExtractor()
        
        # 正弦波（周期性）
        t = np.linspace(0, 4*np.pi, 100)
        sine_wave = np.sin(t)
        periodicity_sine = extractor._detect_periodicity(sine_wave)
        
        # 随机噪声（非周期性）
        np.random.seed(42)
        noise = np.random.randn(100)
        periodicity_noise = extractor._detect_periodicity(noise)
        
        # 正弦波的周期性应该更强
        self.assertGreater(periodicity_sine, periodicity_noise)


if __name__ == '__main__':
    unittest.main()
